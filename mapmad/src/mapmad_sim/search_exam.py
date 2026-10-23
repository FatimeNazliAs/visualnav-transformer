"""Phase 12 additions to the closed-loop search map (search_v2.SearchMap), on the simulator side (block items 2-4).

1. Target source (`TargetSource`): what draws the noisy "target seen here" layer the network reads and the rule stop
   judges -- `truth_noisy` (Phase 11: simulator truth with the Phase 8 noise), `recogniser` (a Phase 9 model in the
   recogniser sidecar, every control step, one category per episode, its detections through Phase 8's fusion:
   recogniser.frame_scores -> map_v2.TargetFusion), or `none` (no detections at all). The clean layer and the truth
   "seen" / "in view" (search / approach split) always come from simulator truth, except with `none`.
2. Rule-stop value (`rule_value`): the largest noisy-layer value among cells <= ahead_m (1.0 m) in front of the robot
   (|bearing| <= half_angle_deg, 30 deg; cell centre vs the turning centre's pose) that were seen from >= min_views (2)
   views (TargetFusion.count); logged every step; the rule stop = the first step with value >= tau (tau is applied
   afterwards, so one run gives both scores). `strongest_cell` = the same at any distance (arm F's "seen").
3. Arm F (`FrontierAgent`): explorer.teacher_step on the same online map (obstacle / explored as known now + bumped
   cells), every control step, its state carried. "Seen" = `layer`: the strongest cell >= tau seen from >= 2 views at
   any distance, not within mark_radius_m of a mark F already went to -> goal = the nearest free cell (explored, not
   within the inflation of an obstacle or bump) within mark_radius_m (1 m) of it, given to the teacher as one seen
   instance (centre = the cell, view point = the goal); or `truth`: Phase 8's simulator-truth seen instances (the F
   check). Arrived at a layer mark and facing it (teacher ARRIVED, within face_tolerance_deg): the mark is remembered
   as visited and the search goes on (a false mark does not end F's run; the rule stop is scored from the log).
   The NoMaD side steers to the decision's next point as the v1 planner arm (closed_loop/frontier_policy.py).
"""

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.explorer import (APPROACH, ARRIVED, OwnMap, SeenInstance, TeacherSpec, TeacherState, cell_of, dilate,
                                 teacher_step)

TARGET_SOURCES = ("truth_noisy", "recogniser", "none")


@dataclass(frozen=True)
class RuleSpec:
    """configs/p12_exam.yaml `run.rule_stop`."""

    ahead_m: float = 1.0
    half_angle_deg: float = 30.0
    min_views: int = 2

    @classmethod
    def from_wire(cls, d: Optional[Dict[str, Any]]) -> Optional["RuleSpec"]:
        return None if not d else cls(float(d["ahead_m"]), float(d["half_angle_deg"]), int(d["min_views"]))


def ahead_cells(shape: Tuple[int, int], origin: Sequence[float], res: float, pose: Sequence[float],
                spec: RuleSpec) -> np.ndarray:
    """Flat indices of the storey cells whose centre lies <= ahead_m from the pose and within +-half_angle_deg of
    its heading."""
    r = int(math.ceil(spec.ahead_m / res)) + 1
    i0, j0 = int(math.floor((pose[0] - origin[0]) / res)), int(math.floor((pose[1] - origin[1]) / res))
    ii, jj = np.mgrid[max(0, i0 - r):min(shape[0], i0 + r + 1), max(0, j0 - r):min(shape[1], j0 + r + 1)]
    cx, cy = origin[0] + (ii + 0.5) * res, origin[1] + (jj + 0.5) * res
    dx, dy = cx - pose[0], cy - pose[1]
    dist = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx) - pose[2]
    ang = np.abs(np.arctan2(np.sin(ang), np.cos(ang)))
    ok = (dist <= spec.ahead_m) & (ang <= math.radians(spec.half_angle_deg))
    return (ii[ok] * shape[1] + jj[ok]).astype(np.int64)


def rule_value(value: np.ndarray, count: np.ndarray, shape: Tuple[int, int], origin: Sequence[float], res: float,
               pose: Sequence[float], spec: RuleSpec) -> Tuple[float, Optional[Tuple[float, float]]]:
    """(largest value among the ahead cells seen from >= min_views views, that cell's centre) -- (0.0, None) when no
    ahead cell qualifies (module docstring, item 2)."""
    cells = ahead_cells(shape, origin, res, pose, spec)
    cells = cells[count[cells] >= spec.min_views]
    if cells.size == 0:
        return 0.0, None
    k = int(cells[int(np.argmax(value[cells]))])  # ties: the first in row-major order
    return float(value[k]), cell_centre(k, shape, origin, res)


def strongest_cell(value: np.ndarray, count: np.ndarray, min_views: int,
                   exclude: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """(largest value among all cells seen from >= min_views views and not excluded, its flat index; -1 if none)."""
    ok = count >= min_views
    if exclude is not None:
        ok &= ~exclude
    if not ok.any():
        return 0.0, -1
    masked = np.where(ok, value, -1.0)
    k = int(np.argmax(masked))
    return float(masked[k]), k


def cell_centre(k: int, shape: Tuple[int, int], origin: Sequence[float], res: float) -> Tuple[float, float]:
    i, j = divmod(int(k), shape[1])
    return float(origin[0] + (i + 0.5) * res), float(origin[1] + (j + 0.5) * res)


# --- the target source --------------------------------------------------------------------------------------------

class TargetSource:
    """Holds the recogniser connection of one simulator server (reconnected once on a broken connection; still down ->
    RuntimeError "recogniser sidecar down", which the runner records as an infrastructure failure)."""

    def __init__(self) -> None:
        self.rec: Optional[Any] = None
        self.key: Optional[Tuple] = None
        self.last_ms = 0.0
        self.last_n = 0

    def ensure(self, spec: Dict[str, Any]) -> None:
        """Connect to the recogniser named in the reset settings ({host, port, model, threshold, p99, vlm})."""
        key = (spec["host"], int(spec["port"]), spec["model"], float(spec["threshold"]), float(spec["p99"]),
               bool(spec["vlm"]))
        if key != self.key:
            self.close()
            self._connect(key)

    def _connect(self, key: Tuple, wait_s: float = 300.0) -> None:
        from mapmad_bridge import protocol
        from mapmad_sim.recogniser import connect

        host, port, model, threshold, p99, vlm = key
        self.rec = connect(model, host, port, threshold, p99, vlm, protocol.auth_key(), wait_s=wait_s)
        self.key = key

    def detect(self, rgb: np.ndarray, depth: np.ndarray, prompt: str) -> List[Any]:
        """The recogniser's detections for this word (PROMPTS[category] or the OVON name, resolved by the server)."""
        t0 = time.perf_counter()
        try:
            dets = self.rec.detect_prompt(rgb, prompt)
        except (OSError, EOFError, RuntimeError) as exc:  # broken connection or a sidecar error: reconnect once
            key = self.key
            self.close()
            try:
                self._connect(key)
                dets = self.rec.detect_prompt(rgb, prompt)
            except (OSError, EOFError, RuntimeError) as exc2:
                self.close()
                raise RuntimeError(f"recogniser sidecar down: {type(exc).__name__}: {exc}; then "
                                   f"{type(exc2).__name__}: {exc2}") from exc2
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.last_n = len(dets)
        return dets

    def close(self) -> None:
        if self.rec is not None:
            self.rec.close()
        self.rec, self.key = None, None


# --- arm F --------------------------------------------------------------------------------------------------------

@dataclass
class FrontierSpec:
    """The `teacher` wire block: TeacherSpec + tau + seen / mode + the mark radius."""

    teacher: TeacherSpec
    tau: float = 1.0
    seen: str = "layer"  # layer | truth
    mode: str = "search"  # search | explore
    mark_radius_m: float = 1.0
    min_views: int = 2

    @classmethod
    def from_wire(cls, d: Dict[str, Any]) -> "FrontierSpec":
        return cls(TeacherSpec.from_config(d["spec"]), float(d.get("tau", 1.0)), d.get("seen", "layer"),
                   d.get("mode", "search"), float(d.get("mark_radius_m", 1.0)), int(d.get("min_views", 2)))


@dataclass
class FrontierAgent:
    """Arm F's teacher on the live map of one episode (module docstring, item 3)."""

    spec: FrontierSpec
    state: TeacherState = field(default_factory=TeacherState)
    visited_marks: List[Tuple[float, float]] = field(default_factory=list)
    mark: Optional[Tuple[int, Tuple[float, float], Tuple[float, float]]] = None  # (cell, centre, goal) of this step

    def own_map(self, search: Any) -> OwnMap:
        lay = search.fmap.first_seen
        shape = search.fmap.shape
        return OwnMap(lay["obstacle"].reshape(shape) >= 0, lay["explored"].reshape(shape) >= 0,
                      tuple(search.fmap.origin), search.spec.map_spec.resolution, search.bumped, None)

    def layer_mark(self, search: Any, own: OwnMap) -> Optional[SeenInstance]:
        """The strongest qualifying cell (>= tau, >= min_views views, not near a visited mark) as a seen instance whose
        view point is the nearest free cell within mark_radius_m; None if there is none."""
        fus = search.fusions["noisy"]
        shape, origin, res = search.fmap.shape, search.fmap.origin, own.res
        exclude = None
        if self.visited_marks:
            exclude = np.zeros(shape[0] * shape[1], dtype=bool)
            for c in self.visited_marks:
                exclude |= disk_mask(shape, origin, res, c, self.spec.mark_radius_m).ravel()
        while True:
            v, k = strongest_cell(fus.value, fus.count, self.spec.min_views, exclude)
            if k < 0 or v < self.spec.tau:
                self.mark = None
                return None
            centre = cell_centre(k, shape, origin, res)
            goal = nearest_free(own, centre, self.spec.mark_radius_m, self.spec.teacher.inflation_m)
            if goal is not None:
                self.mark = (k, centre, goal)
                return SeenInstance(int(k), centre, (goal,))
            exclude = np.zeros(shape[0] * shape[1], dtype=bool) if exclude is None else exclude
            exclude[k] = True  # no free cell near it: try the next strongest

    def step(self, search: Any, pose: Tuple[float, float, float]) -> Dict[str, Any]:
        """The teacher's decision at this step (as a JSON-ready dict for the policy)."""
        own = self.own_map(search)
        searching = self.spec.mode == "search"
        if not searching:
            seen: List[SeenInstance] = []
        elif self.spec.seen == "truth":
            seen = [search.seen[k] for k in sorted(search.seen)]
        else:
            m = self.layer_mark(search, own)
            seen = [m] if m is not None else []
        dec, self.state = teacher_step(own, pose, self.state, seen, self.spec.teacher, searching)
        out = {"mode": dec.mode, "pose": [float(v) for v in pose], "goal": _xy(dec.goal), "next_point": _xy(dec.next_point),
               "face_yaw": dec.face_yaw, "path_length_m": dec.path_length_m, "new_choice": bool(dec.new_choice),
               "mark": list(self.mark[1]) if (self.mark is not None and seen) else None}
        if (dec.mode == ARRIVED and self.spec.seen == "layer" and self.mark is not None and dec.face_yaw is not None
                and abs(_wrap(dec.face_yaw - pose[2])) <= math.radians(self.spec.teacher.face_tolerance_deg)):
            self.visited_marks.append(self.mark[1])  # arrived and facing a layer mark: go on searching
            self.state = replace(self.state, approach=None, known_instances=())
            out["mark_done"] = True
        return out


def nearest_free(own: OwnMap, centre: Tuple[float, float], radius_m: float,
                 inflation_m: float) -> Optional[Tuple[float, float]]:
    """Centre of the nearest free cell (explored, not within the inflation of a known obstacle or bump; the v1
    planner's blocked rule) within radius_m of `centre`; None if there is none."""
    res = own.res
    shape = own.explored.shape
    r = int(math.ceil(radius_m / res)) + 1
    i0, j0 = cell_of(centre, own)
    sl = (slice(max(0, i0 - r - 3), min(shape[0], i0 + r + 4)), slice(max(0, j0 - r - 3), min(shape[1], j0 + r + 4)))
    obstacle = own.obstacle[sl] | (own.bumped[sl] if own.bumped is not None else False)
    blocked = dilate(obstacle, inflation_m / res + 0.5)
    free = own.explored[sl] & ~blocked
    ii, jj = np.nonzero(free)
    if ii.size == 0:
        return None
    gi, gj = ii + sl[0].start, jj + sl[1].start
    cx, cy = own.origin[0] + (gi + 0.5) * res, own.origin[1] + (gj + 0.5) * res
    d = np.hypot(cx - centre[0], cy - centre[1])
    ok = d <= radius_m
    if not ok.any():
        return None
    k = int(np.flatnonzero(ok)[np.argmin(d[ok])])
    return float(cx[k]), float(cy[k])


def disk_mask(shape: Tuple[int, int], origin: Sequence[float], res: float, centre: Sequence[float],
              radius_m: float) -> np.ndarray:
    """(H, W) bool: cells whose centre lies within radius_m of `centre`."""
    out = np.zeros(shape, dtype=bool)
    r = int(math.ceil(radius_m / res)) + 1
    i0, j0 = int(math.floor((centre[0] - origin[0]) / res)), int(math.floor((centre[1] - origin[1]) / res))
    ii, jj = np.mgrid[max(0, i0 - r):min(shape[0], i0 + r + 1), max(0, j0 - r):min(shape[1], j0 + r + 1)]
    d = np.hypot(origin[0] + (ii + 0.5) * res - centre[0], origin[1] + (jj + 0.5) * res - centre[1])
    out[ii[d <= radius_m], jj[d <= radius_m]] = True
    return out


def _xy(p: Optional[Sequence[float]]) -> Optional[List[float]]:
    return None if p is None else [float(p[0]), float(p[1])]


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


__all__ = ["TARGET_SOURCES", "RuleSpec", "rule_value", "strongest_cell", "TargetSource", "FrontierSpec",
           "FrontierAgent", "nearest_free", "APPROACH"]
