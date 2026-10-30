"""Phase 13 arm VF ("VLFM-style", report only), simulator side: VLFM's frontier choice (a BLIP-2 image-text value map,
Yokoyama et al. 2024, ITMPolicyV2) on MapMaD's own online map, with our camera (pin025), LIMO footprint, YOLOE target
layer, and our planner + motion in place of VLFM's PointNav policy. VF differs from arm F (search_exam.FrontierAgent)
ONLY in which frontier it explores while no target mark is active.

Every control step, after search_v2.SearchMap.add_frame (sidecar_server.exam_outputs, `VFAgent.step`):
1. ITM: the colour frame + the prompt "Seems like there is a <word> ahead." (VLFM base_objectnav_policy.py:377;
   <word> = the recogniser's word: the episode's target.prompt, else photo_labels.PROMPTS[category], e.g. sofa ->
   "couch", tv_monitor -> "tv", plant -> "potted plant") -> the BLIP-2 ITC cosine from mapmad_recog.itm_server (one
   server per GPU, shared by that GPU's pairs; `ITMSource` keeps the connection across episodes).
2. Value map (`ValueMap`, port of VLFM mapping/value_map.py) on the storey grid of SearchMap.fmap (same shape, origin,
   10 cm cells): the frame's field-of-view cone (`frame_cone`) with VLFM's confidence (cos^2 of the angle off the
   optical axis, remapped to [min_confidence 0.25, 1], value_map.py:337-355), cut where the depth frame ends (VLFM
   value_map.py:221-286: per image column the farthest point the depth frame shows -> a contour, everything beyond it
   is zeroed), drawn at VLFM's own 5 cm (20 px/m) in the camera's frame and read at each storey cell centre (nearest
   pixel). Our version of the per-column reach: the farthest HORIZONTAL forward distance among the column's back-projected
   pixels (depth clipped to the map builder's max depth, 4 m, as VLFM clips to its max depth), and that point's own
   lateral offset (VLFM: depth x tan(column angle), a level camera; ours is pitched -2 deg). Fusion (value_map.py:
   357-429, VLFM's default config: use_max_confidence False, base_objectnav_policy.py:381): a new confidence below
   decision_threshold (0.35) and below the old one is ignored; else value and confidence become the confidence-weighted
   averages (value = (v c + cos c') / (c + c'), confidence = (c^2 + c'^2) / (c + c')). sync_explored_areas is VLFM's
   default False (itm_policy.py:43): no obstacle-map masking.
3. Decision (same dict as FrontierAgent.step, so closed_loop/frontier_policy.py steers it unchanged; extras under "vf"):
   - a layer mark exists (F's rule: the strongest target-layer cell >= tau seen from >= 2 views, not near a visited
     mark) -> exactly F: FrontierAgent.step (same mark, planner path, arrive / face / continue rule);
   - else VLFM's choice (itm_policy.py:76-152): frontiers = map_v2.frontier_clusters(obstacle | bumped, explored,
     min_cluster 4) on the planner window, ONE point per cluster: the reachable cluster cell nearest the cluster's
     centroid (VLFM takes one waypoint per frontier, the frontier_exploration package's midpoint; that package is not in
     the source copy, so the medoid-like cell stands for it and is reachable by construction). Value of a point = the
     median of the positive value-map cells within value_radius_m (0.5 m) of it, -1 if none (value_map.py:146-187,
     img_utils.py:213-266). Sort by value (descending); keep the last frontier (the same point, else the nearest point
     within 0.5 m of it) if its value + 0.01 > the last chosen value (hysteresis, itm_policy.py:101-124); else the best
     point not suppressed by VLFM's acyclic enforcer (the same robot position + frontier + top-two values chosen
     before, policy/utils/acyclic_enforcer.py); all suppressed -> VLFM's fallback as written (itm_policy.py:137-143:
     the point FARTHEST from the robot by index into the unsorted list, then looked up in the sorted list).
     Path = the explorer's planner on OwnMap (frontier-free cells, clearance-shaped path, lookahead 0.5 m);
     no candidate -> mode no_frontier (the policy turns left on the spot, as F on a teacher failure).
   Two of F's implementation rules (explorer.py docstring, p8_rules.md §8) are kept so the robot cannot stand forever
   at a frontier its camera never clears (our blind zone; VLFM's PointNav would stop there too): points within
   commit_reach_m (0.5 m) of the robot are not candidates, and a pursued point the robot came within 0.5 m of that is
   still a frontier is dropped with radius reached_radius_m (0.2 m) for the rest of the episode.
Deterministic: no random numbers (ties: stable sort, cluster order); the ITM server is deterministic per GPU.
"""

import math
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mapmad_sim.camera import focal_px
from mapmad_sim.explorer import (NO_FRONTIER, SEARCH, OwnMap, _frontier_field, _frontier_free, _Planner, centre_of,
                                 disk, frontier_clusters, path_point)
from mapmad_sim.frames import to_2d
from mapmad_sim.search_exam import FrontierAgent, FrontierSpec, _xy

PROMPT = "Seems like there is a {target} ahead."  # VLFM base_objectnav_policy.py:377 ("target_object" -> {target})


# --- settings -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueSpec:
    """configs/p13_vf.yaml `vf` (VLFM's numbers, source lines there)."""

    prompt: str = PROMPT
    min_confidence: float = 0.25
    decision_threshold: float = 0.35
    use_max_confidence: bool = False
    value_radius_m: float = 0.5
    stick_margin: float = 0.01
    stick_radius_m: float = 0.5
    local_px_per_m: int = 20

    @classmethod
    def from_config(cls, d: Dict[str, Any]) -> "ValueSpec":
        return cls(**{k: type(getattr(cls, k))(v) for k, v in d.items()})


@dataclass(frozen=True)
class VFSpec:
    """The `vf` wire block: F's teacher block (marks, planner) + the value-map numbers + the ITM server address."""

    frontier: FrontierSpec
    value: ValueSpec
    itm: Tuple[str, int]

    @classmethod
    def from_wire(cls, d: Dict[str, Any]) -> "VFSpec":
        return cls(FrontierSpec.from_wire(d["teacher"]), ValueSpec.from_config(d["value_map"]),
                   (str(d["itm"]["host"]), int(d["itm"]["port"])))


@dataclass(frozen=True)
class CameraView:
    """What the value map needs of the frame: planar depth (m), the colour/depth camera's 4x4 camera-to-world matrix
    (Habitat), its horizontal FOV and the reach (the map builder's max depth)."""

    depth: np.ndarray
    cam_tf: np.ndarray
    hfov_deg: float
    max_depth_m: float


# --- the FOV cone (VLFM value_map.py:221-355) ---------------------------------------------------------------------

def remap(value: Any, from_low: float, from_high: float, to_low: float, to_high: float) -> Any:
    """VLFM value_map.py:432-445."""
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def confidence_cone(fov: float, max_depth: float, px_per_m: int, min_confidence: float) -> np.ndarray:
    """VLFM _get_blank_cone_mask + _get_confidence_mask (value_map.py:321-355): a (2s+1)^2 picture, s = int(max_depth x
    px_per_m), camera at the centre pixel, looking along +row; cells in the cone carry
    remap(cos^2(remap(angle off the axis, 0, fov/2, 0, pi/2)), 0, 1, min_confidence, 1), the rest 0."""
    size = int(max_depth * px_per_m)
    cone = np.zeros((size * 2 + 1, size * 2 + 1))
    cone = cv2.ellipse(cone, (size, size), (size, size), 0, -np.rad2deg(fov) / 2 + 90, np.rad2deg(fov) / 2 + 90, 1, -1)
    rows, cols = np.mgrid[0:cone.shape[0], 0:cone.shape[1]]
    angle = np.arctan2(np.abs(cols - cone.shape[1] // 2), np.abs(rows - cone.shape[0] // 2))
    conf = remap(np.cos(remap(angle, 0, fov / 2, 0, np.pi / 2)) ** 2, 0, 1, min_confidence, 1)
    return conf.astype(np.float32) * cone


def column_reach(view: CameraView) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per image column (right to left = increasing lateral offset): the farthest horizontal forward distance the depth
    frame shows and that point's lateral offset (left +), in the camera's 2D frame; plus the camera's 2D position and
    heading (yaw of its optical axis). Depth clipped to max_depth_m (VLFM normalises to [min, max]); depth 0 (no
    return) gives the camera position."""
    depth = np.minimum(np.asarray(view.depth, dtype=np.float64), view.max_depth_m)
    h, w = depth.shape
    f = focal_px(w, view.hfov_deg)
    rows, cols = np.mgrid[0:h, 0:w]
    pts = np.stack([(cols + 0.5 - w / 2.0) / f * depth, -(rows + 0.5 - h / 2.0) / f * depth, -depth], axis=-1)
    rot = np.asarray(view.cam_tf, dtype=np.float64)[:3, :3]
    off = to_2d(pts @ rot.T)  # (h, w, 2) horizontal offsets from the camera, 2D frame (directions: no translation)
    axis = to_2d(rot @ np.array([0.0, 0.0, -1.0]))
    yaw = math.atan2(axis[1], axis[0])
    fwd = off[..., 0] * math.cos(yaw) + off[..., 1] * math.sin(yaw)
    lat = -off[..., 0] * math.sin(yaw) + off[..., 1] * math.cos(yaw)
    k = np.argmax(fwd, axis=0)
    c = np.arange(w)
    reach_f, reach_l = np.maximum(fwd[k, c], 0.0), lat[k, c]
    cam_xy = to_2d(np.asarray(view.cam_tf, dtype=np.float64)[:3, 3])
    return reach_f[::-1], reach_l[::-1], cam_xy, yaw


def visible_cone(cone: np.ndarray, reach_f: np.ndarray, reach_l: np.ndarray, px_per_m: int) -> np.ndarray:
    """VLFM _process_local_data (value_map.py:221-286): the polygon from one far corner through the per-column reach
    points to the other far corner is filled with 0, so only the part of the cone the depth frame shows keeps its
    confidence. Pixel = int(metres x px_per_m + shape / 2), as VLFM."""
    x = (reach_f * px_per_m + cone.shape[0] / 2).astype(int)
    y = (reach_l * px_per_m + cone.shape[1] / 2).astype(int)
    last_row, last_col = cone.shape[0] - 1, cone.shape[1] - 1
    contour = np.concatenate((np.array([[0, last_col]]), np.stack((y, x), axis=1), np.array([[last_row, last_col]])),
                             axis=0).astype(np.int32)
    return cv2.drawContours(cone.copy(), [contour], -1, 0, -1)


def frame_cone(view: CameraView, shape: Tuple[int, int], origin: Sequence[float], res: float,
               spec: ValueSpec, cone_cache: Dict[Tuple[float, float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(flat storey cells, their new confidence > 0) of this frame: the visible cone read at each cell centre (nearest
    5 cm pixel of the camera-frame picture; VLFM rotates the picture into its map instead)."""
    fov = math.radians(view.hfov_deg)
    key = (fov, view.max_depth_m)
    if key not in cone_cache:
        cone_cache[key] = confidence_cone(fov, view.max_depth_m, spec.local_px_per_m, spec.min_confidence)
    reach_f, reach_l, cam_xy, yaw = column_reach(view)
    local = visible_cone(cone_cache[key], reach_f, reach_l, spec.local_px_per_m)
    r = int(math.ceil(view.max_depth_m / res)) + 1
    i0, j0 = int(math.floor((cam_xy[0] - origin[0]) / res)), int(math.floor((cam_xy[1] - origin[1]) / res))
    ii, jj = np.mgrid[max(0, i0 - r):min(shape[0], i0 + r + 1), max(0, j0 - r):min(shape[1], j0 + r + 1)]
    dx, dy = origin[0] + (ii + 0.5) * res - cam_xy[0], origin[1] + (jj + 0.5) * res - cam_xy[1]
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    lat = -dx * math.sin(yaw) + dy * math.cos(yaw)
    pr = np.floor(fwd * spec.local_px_per_m + local.shape[0] / 2).astype(np.int64)
    pc = np.floor(lat * spec.local_px_per_m + local.shape[1] / 2).astype(np.int64)
    ok = (pr >= 0) & (pr < local.shape[0]) & (pc >= 0) & (pc < local.shape[1])
    conf = np.zeros(ii.shape, dtype=np.float64)
    conf[ok] = local[pr[ok], pc[ok]]
    keep = conf > 0
    return (ii[keep] * shape[1] + jj[keep]).astype(np.int64), conf[keep]


# --- the value map ------------------------------------------------------------------------------------------------

class ValueMap:
    """VLFM's value + confidence maps (one value channel) on the storey grid (module docstring, item 2)."""

    def __init__(self, shape: Tuple[int, int], spec: ValueSpec) -> None:
        self.shape, self.spec = (int(shape[0]), int(shape[1])), spec
        self.value = np.zeros(self.shape[0] * self.shape[1], dtype=np.float64)
        self.conf = np.zeros(self.shape[0] * self.shape[1], dtype=np.float64)

    def fuse(self, cells: np.ndarray, new: np.ndarray, value: float) -> None:
        """VLFM _fuse_new_data, fusion "default" (value_map.py:395-429). Cells outside the new cone keep their value
        and confidence (VLFM: weight 0 for the new data there)."""
        old = self.conf[cells]
        new = np.asarray(new, dtype=np.float64).copy()
        new[(new < self.spec.decision_threshold) & (new < old)] = 0.0
        if self.spec.use_max_confidence:
            higher = new > old
            self.value[cells[higher]] = value
            self.conf[cells[higher]] = new[higher]
            return
        denom = old + new
        with np.errstate(invalid="ignore", divide="ignore"):
            w1, w2 = old / denom, new / denom
        self.value[cells] = np.nan_to_num(self.value[cells] * w1 + value * w2)
        self.conf[cells] = np.nan_to_num(old * w1 + new * w2)

    def point_value(self, xy: Sequence[float], origin: Sequence[float], res: float) -> float:
        """VLFM sort_waypoints' get_value (value_map.py:161-174 + img_utils.pixel_value_within_radius, reduction
        "median"): the median of the positive values within value_radius_m of the point; -1 if there are none."""
        rc = int(self.spec.value_radius_m / res)
        i0, j0 = int(math.floor((xy[0] - origin[0]) / res)), int(math.floor((xy[1] - origin[1]) / res))
        vals = []
        for di, dj in disk(rc):
            i, j = i0 + di, j0 + dj
            if 0 <= i < self.shape[0] and 0 <= j < self.shape[1]:
                vals.append(self.value[i * self.shape[1] + j])
        v = np.asarray(vals, dtype=np.float64)
        v = v[v > 0]
        return float(np.median(v)) if v.size else -1.0


# --- VLFM's frontier choice (itm_policy.py:76-152) ----------------------------------------------------------------

def closest_point_within_threshold(points: np.ndarray, target: Sequence[float], threshold: float) -> int:
    """VLFM geometry_utils.py:183-202."""
    d = np.sqrt((points[:, 0] - target[0]) ** 2 + (points[:, 1] - target[1]) ** 2)
    return int(np.argmin(d)) if np.any(d <= threshold) else -1


@dataclass
class FrontierChooser:
    """BaseITMPolicy._get_best_frontier's memory: last frontier + its value, and the acyclic enforcer's history."""

    spec: ValueSpec
    last_frontier: Optional[np.ndarray] = None
    last_value: float = float("-inf")
    history: set = field(default_factory=set)

    @staticmethod
    def _key(robot_xy: np.ndarray, frontier: np.ndarray, top_two: Tuple[float, ...]) -> str:
        return f"{robot_xy}_{frontier}_{top_two}"  # acyclic_enforcer.StateAction.__hash__'s string

    def choose(self, points: np.ndarray, values: List[float], robot_xy: np.ndarray) -> Tuple[np.ndarray, float]:
        """The frontier to pursue and its value (module docstring, item 3)."""
        order = np.argsort([-v for v in values], kind="stable")
        pts = points[order]
        vals = [values[i] for i in order]
        top_two = tuple(vals[:2])
        best = None
        if self.last_frontier is not None:
            curr = next((k for k, p in enumerate(pts) if np.array_equal(p, self.last_frontier)), None)
            if curr is None:
                c = closest_point_within_threshold(pts, self.last_frontier, self.spec.stick_radius_m)
                curr = c if c != -1 else None
            if curr is not None and vals[curr] + self.spec.stick_margin > self.last_value:
                best = curr
        if best is None:
            for k, p in enumerate(pts):
                if self._key(robot_xy, p, top_two) in self.history:
                    continue  # "Suppressed cyclic frontier."
                best = k
                break
        if best is None:  # "All frontiers are cyclic. Just choosing the closest one." -- as written: max, unsorted index
            best = max(range(len(points)), key=lambda i: np.linalg.norm(points[i] - robot_xy))
        self.history.add(self._key(robot_xy, pts[best], top_two))
        self.last_value, self.last_frontier = vals[best], pts[best]
        return pts[best], vals[best]


# --- the ITM connection ---------------------------------------------------------------------------------------------

class ITMSource:
    """The ITM server connection of one simulator server (kept across episodes; reconnected once on a broken
    connection; still down -> RuntimeError "itm sidecar down", an infrastructure failure for the runner)."""

    def __init__(self) -> None:
        self.sock: Optional[socket.socket] = None
        self.addr: Optional[Tuple[str, int]] = None
        self.last_ms = 0.0

    def ensure(self, addr: Tuple[str, int], wait_s: float = 1800.0) -> None:
        if addr != self.addr:
            self.close()
            self._connect(addr, wait_s)

    def _connect(self, addr: Tuple[str, int], wait_s: float = 1800.0) -> None:
        from mapmad_bridge import protocol

        deadline = time.time() + wait_s
        while True:
            try:
                sock = socket.create_connection(addr, timeout=600)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(5)  # the server is still loading its model
        protocol.send(sock, {"cmd": "hello", "key": protocol.auth_key()})
        reply = protocol.recv(sock)
        if "error" in reply:
            sock.close()
            raise RuntimeError(f"itm sidecar: {reply['error']}")
        self.sock, self.addr = sock, addr

    def _ask(self, rgb: np.ndarray, text: str) -> float:
        from mapmad_bridge import protocol

        protocol.send(self.sock, {"cmd": "itm", "rgb": np.ascontiguousarray(rgb, np.uint8), "text": text})
        reply = protocol.recv(self.sock)
        if "error" in reply:
            raise RuntimeError(f"itm sidecar: {reply['error']}")
        return float(reply["cosine"])

    def cosine(self, rgb: np.ndarray, text: str) -> float:
        t0 = time.perf_counter()
        try:
            c = self._ask(rgb, text)
        except (OSError, EOFError, RuntimeError) as exc:
            addr = self.addr
            self.close()
            try:
                self._connect(addr, wait_s=300.0)
                c = self._ask(rgb, text)
            except (OSError, EOFError, RuntimeError) as exc2:
                self.close()
                raise RuntimeError(f"itm sidecar down: {type(exc).__name__}: {exc}; then "
                                   f"{type(exc2).__name__}: {exc2}") from exc2
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return c

    def close(self) -> None:
        if self.sock is not None:
            from mapmad_bridge import protocol

            try:
                protocol.send(self.sock, {"cmd": "bye"})
                protocol.recv(self.sock)
            except (OSError, EOFError):
                pass
            self.sock.close()
        self.sock, self.addr = None, None


# --- arm VF -------------------------------------------------------------------------------------------------------

@dataclass
class VFAgent:
    """Arm VF on the live map of one episode (module docstring)."""

    spec: VFSpec
    itm: Any  # ITMSource-like: cosine(rgb, text) -> float
    prompt: str  # the full ITM text of the episode
    shape: Tuple[int, int]
    f: FrontierAgent = field(init=False)
    values: ValueMap = field(init=False)
    chooser: FrontierChooser = field(init=False)
    dropped: List[Tuple[float, float, float]] = field(default_factory=list)  # (x, y, radius): reached, still frontier
    cones: Dict[Tuple[float, float], np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.f = FrontierAgent(self.spec.frontier)
        self.values = ValueMap(self.shape, self.spec.value)
        self.chooser = FrontierChooser(self.spec.value)

    @staticmethod
    def text(spec: ValueSpec, word: str) -> str:
        return spec.prompt.format(target=word)

    def observe(self, search: Any, rgb: np.ndarray, view: CameraView) -> float:
        """Item 1-2: the frame's cosine fused into the value map; returns the cosine."""
        cos = self.itm.cosine(rgb, self.prompt)
        cells, conf = frame_cone(view, self.shape, search.fmap.origin, search.spec.map_spec.resolution,
                                 self.spec.value, self.cones)
        self.values.fuse(cells, conf, cos)
        return cos

    def step(self, search: Any, pose: Tuple[float, float, float], rgb: np.ndarray,
             view: CameraView) -> Dict[str, Any]:
        """The decision at this step (FrontierAgent.step's dict) plus "vf" = {cosine, frontier, value, n_frontiers}."""
        cos = self.observe(search, rgb, view)
        own = self.f.own_map(search)
        if self.f.layer_mark(search, own) is not None:  # a target mark: exactly arm F
            out = self.f.step(search, pose)
            out["vf"] = {"cosine": cos, "frontier": None, "value": None, "n_frontiers": None, "mark_active": True}
            return out
        out, info = self.explore(search, own, pose)
        out["vf"] = {"cosine": cos, **info, "mark_active": False}
        return out

    def candidates(self, pl: Any, labels: np.ndarray, dist: np.ndarray, pose: Tuple[float, float, float],
                   reach_m: float) -> List[Tuple[int, int]]:
        """One point per frontier cluster (local window cells): the reachable cluster cell (not within reach_m of the
        robot, not dropped) nearest the cluster's centroid (ties: the first in row-major order)."""
        out = []
        for k in range(1, int(labels.max()) + 1):
            ci, cj = np.nonzero(labels == k)
            centres = np.asarray(centre_of(ci + pl.win.i0, cj + pl.win.j0, pl.m), dtype=np.float64).reshape(-1, 2)
            ok = np.isfinite(dist[ci, cj]) & (np.hypot(centres[:, 0] - pose[0], centres[:, 1] - pose[1]) > reach_m)
            for x, y, r in self.dropped:
                ok &= np.hypot(centres[:, 0] - x, centres[:, 1] - y) > r
            if not ok.any():
                continue
            mean = centres.mean(axis=0)
            idx = np.flatnonzero(ok)
            m = int(idx[np.argmin(np.hypot(centres[idx, 0] - mean[0], centres[idx, 1] - mean[1]))])
            out.append((int(ci[m]), int(cj[m])))
        return out

    def _drop_reached(self, pl: Any, labels: np.ndarray, pose: Tuple[float, float, float], spec: Any) -> None:
        """F's "reached" rule for the pursued point: within commit_reach_m and still a frontier -> dropped."""
        last = self.chooser.last_frontier
        if last is None or math.hypot(last[0] - pose[0], last[1] - pose[1]) > spec.commit_reach_m:
            return
        c = pl.local((int(math.floor((last[0] - pl.m.origin[0]) / pl.m.res)),
                      int(math.floor((last[1] - pl.m.origin[1]) / pl.m.res))))
        if pl.inside(c) and labels[c] > 0 and (float(last[0]), float(last[1]), spec.reached_radius_m) not in self.dropped:
            self.dropped.append((float(last[0]), float(last[1]), spec.reached_radius_m))

    def explore(self, search: Any, own: OwnMap, pose: Tuple[float, float, float]
                ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """No mark: path to VLFM's best-value frontier (module docstring, item 3)."""
        ts = self.spec.frontier.teacher
        last = self.chooser.last_frontier
        pl = _Planner(own, pose, [tuple(last)] if last is not None else [], ts)
        labels = frontier_clusters(pl.obstacle, pl.explored, search.spec.min_cluster)
        n_clusters = int(labels.max())
        self._drop_reached(pl, labels, pose, ts)
        dist, _ = _frontier_field(pl)
        cells = self.candidates(pl, labels, dist, pose, ts.commit_reach_m)
        base = {"mode": SEARCH, "pose": [float(v) for v in pose], "goal": None, "next_point": None, "face_yaw": None,
                "path_length_m": float("nan"), "new_choice": False, "mark": None}
        if not cells:
            return dict(base, mode=NO_FRONTIER), {"frontier": None, "value": None, "n_frontiers": 0,
                                                  "n_clusters": n_clusters}
        res, origin = own.res, own.origin
        points = np.array([pl.xy(c) for c in cells], dtype=np.float64)
        values = [self.values.point_value(p, origin, res) for p in points]
        prev = None if last is None else last.copy()
        goal, value = self.chooser.choose(points, values, np.array(pose[:2], dtype=np.float64))
        cell = cells[int(np.flatnonzero((points == goal).all(axis=1))[0])]
        path = pl.path_xy(pl.shaped_path(_frontier_free(pl), cell), pose)
        out = dict(base, goal=_xy(goal), next_point=_xy(path_point(path, ts.lookahead_m)),
                   path_length_m=float(dist[cell]), new_choice=prev is None or not np.array_equal(prev, goal))
        return out, {"frontier": _xy(goal), "value": float(value), "n_frontiers": len(cells), "n_clusters": n_clusters}


def vf_word(episode: Dict[str, Any]) -> str:
    """The episode's target word, as the recogniser is asked (sidecar_server._target)."""
    from mapmad_sim.photo_labels import PROMPTS

    t = episode["target"]
    return t.get("prompt") or PROMPTS[t["category"]]


__all__ = ["PROMPT", "ValueSpec", "VFSpec", "CameraView", "confidence_cone", "visible_cone", "frame_cone", "ValueMap",
           "FrontierChooser", "ITMSource", "VFAgent", "vf_word", "remap"]
