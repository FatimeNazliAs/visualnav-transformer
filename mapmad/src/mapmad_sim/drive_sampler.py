"""Where a practice drive starts and ends (plan D11; Phase 2 tasks 2.2).

- Real floor: start and end lie on a navmesh island of at least min_island_area_m2 (no bed or table tops) and the
  floor probe finds the floor below; in labelled homes the surface under the robot must also be floor / carpet /
  rug / mat (Phase 1 rule, episodes.EpisodeBuilder.on_real_floor).
- Spot target: a real-floor point 3-15 m (geodesic, LIMO's floor map) from the start, same floor.
- Object target (labelled homes): one ObjectNav v2 goal instance (category drawn uniformly among the home's
  categories, then the instance). The drive ends at the instance's ObjectNav view point nearest (geodesic) to the
  start, snapped to LIMO's floor map (a view point more than 0.2 m off is unusable -> the next nearest), then turns
  to face the object box's centre. No usable view point: the nearest navigable real-floor point to the box
  (end_fallback = True).
- Start heading: random. Wall-recovery start: a point close to the floor map's edge and a heading whose central
  20 x 20 depth pixels have a median of 0.2-0.5 m (facing a wall or furniture up close).
- Start visibility is recorded: object = share of target pixels at the start heading and whether any of 8
  headings shows it; spot = whether the target point is inside the picture and not hidden behind something.
Positions are Habitat (x, y, z) navmesh points; yaws are Habitat/NoMaD yaws (the same, frames.py).
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.camera import centre_depth, project_points
from mapmad_sim.episodes import EpisodeBuilder, box_distance_xz, rounded
from mapmad_sim.frames import to_2d, yaw_of
from mapmad_sim.objectnav import Goal
from mapmad_sim.objects import object_box
from mapmad_sim.robot import LimoSim, horizontal_distance


@dataclass
class Target:
    """End of a drive. object_* fields only for object drives."""

    kind: str  # "spot" | "object"
    end_position: List[float]  # Habitat navmesh point where the drive stops
    final_yaw: Optional[float] = None  # object drives: face the box centre
    category: Optional[str] = None
    object_id: Optional[int] = None
    object_name: Optional[str] = None
    box_center: Optional[List[float]] = None
    box_size: Optional[List[float]] = None
    view_point: Optional[List[float]] = None  # the ObjectNav view point used (unsnapped)
    view_point_snap_m: Optional[float] = None
    end_fallback: bool = False


@dataclass
class Start:
    position: List[float]
    yaw: float
    wall_recovery: bool = False
    centre_depth_m: Optional[float] = None
    visibility: Dict[str, Any] = field(default_factory=dict)


class DriveSampler:
    """Starts and targets in one home (LimoSim with depth=True; semantic=True in labelled homes)."""

    def __init__(self, robot: LimoSim, cfg: Dict[str, Any], rng: np.random.Generator,
                 goals: Optional[Sequence[Goal]] = None) -> None:
        self.robot, self.cfg, self.rng = robot, cfg, rng
        self.pf = robot.pathfinder
        self.labelled = goals is not None
        self.labels = EpisodeBuilder(robot, {"floor_names": cfg["sampling"]["floor_names"]}, rng) if self.labelled else None
        self.goals = list(goals or [])
        self._island_ok: Dict[int, bool] = {}
        self.floor_checks: Counter = Counter()  # outcome of every real-floor check (logged per home)

    # --- floor checks ------------------------------------------------------------------------------------------
    def random_point(self) -> np.ndarray:
        while True:  # habitat returns NaN now and then; seeded from our rng (reproducible)
            self.pf.seed(int(self.rng.integers(2**31 - 1)))
            p = np.array(self.pf.get_random_navigable_point(), dtype=np.float64)
            if np.isfinite(p).all():
                return p

    def big_island(self, p: np.ndarray) -> bool:
        island = int(self.pf.get_island(p.astype(np.float32)))
        if island not in self._island_ok:
            self._island_ok[island] = island >= 0 and self.pf.island_area(island) >= self.cfg["sampling"]["min_island_area_m2"]
        return self._island_ok[island]

    def on_real_floor(self, p: np.ndarray) -> bool:
        """Big island, floor found by the probe, and (labelled homes) a floor-like surface under the robot."""
        if not self.big_island(p):
            self.floor_checks["small_island"] += 1
            return False
        if self.labels is not None:
            ok = self.labels.on_real_floor(p)
            self.floor_checks["ok" if ok else "not_floor_label_or_no_floor"] += 1
            return ok
        try:
            self.robot.place(p, 0.0)
        except ValueError:
            self.floor_checks["no_floor_probe"] += 1
            return False
        self.floor_checks["ok"] += 1
        return True

    def same_floor(self, a: Sequence[float], b: Sequence[float]) -> bool:
        return abs(a[1] - b[1]) <= self.cfg["sampling"]["same_floor_max_dy_m"]

    def in_range(self, geodesic: float) -> bool:
        lo, hi = self.cfg["sampling"]["geodesic_m"]
        return math.isfinite(geodesic) and lo <= geodesic <= hi

    # --- starts ------------------------------------------------------------------------------------------------
    def centre_depth(self) -> float:
        """Median of the central n x n depth pixels at the current pose (0 readings = holes are skipped)."""
        return centre_depth(self.robot.observe()["depth"], self.cfg["variations"]["wall_recovery"]["centre_px"])

    def start(self) -> Start:
        """A random real-floor start with a random heading."""
        while True:
            p = self.random_point()
            if self.on_real_floor(p):
                return Start(rounded(p), round(float(self.rng.uniform(-math.pi, math.pi)), 4))

    def wall_start(self) -> Optional[Start]:
        """A real-floor start facing a wall or furniture 0.2-0.5 m away (central depth median), or None."""
        rule = self.cfg["variations"]["wall_recovery"]
        lo, hi = rule["depth_m"]
        for _ in range(rule["tries"]):
            p = self.random_point()
            if self.pf.distance_to_closest_obstacle(p.astype(np.float32), 1.0) > rule["max_navmesh_clearance_m"]:
                continue
            if not self.on_real_floor(p):
                continue
            offset = float(self.rng.uniform(0.0, 2.0 * math.pi / rule["headings"]))
            found = []
            for k in range(rule["headings"]):
                yaw = offset + 2.0 * math.pi * k / rule["headings"]
                self.robot.place(p, yaw)
                d = self.centre_depth()
                if lo <= d <= hi:
                    found.append((yaw, d))
            if found:
                yaw, d = found[int(self.rng.integers(len(found)))]
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                return Start(rounded(p), round(yaw, 4), wall_recovery=True, centre_depth_m=round(d, 3))
        return None

    # --- targets -----------------------------------------------------------------------------------------------
    def spot_target(self, start: Start) -> Optional[Tuple[Target, float]]:
        """(target, geodesic) of a spot drive from this start, or None."""
        s = np.array(start.position)
        for _ in range(self.cfg["sampling"]["target_tries"]):
            p = self.random_point()
            if not self.same_floor(s, p):
                continue
            geo = self.robot.geodesic(s, p)
            if self.in_range(geo) and self.on_real_floor(p):
                return Target("spot", rounded(p)), geo
        return None

    def pick_goal(self) -> Goal:
        cats = sorted({g.category for g in self.goals})
        cat = cats[int(self.rng.integers(len(cats)))]
        pool = [g for g in self.goals if g.category == cat]
        return pool[int(self.rng.integers(len(pool)))]

    def usable_view_points(self, goal: Goal) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """(snapped point, view point, snap distance) of every view point within view_point_max_snap_m of
        LIMO's floor map, on the real floor."""
        out = []
        for vp in goal.view_points:
            vp = np.asarray(vp, dtype=np.float64)
            p = np.array(self.pf.snap_point(vp.astype(np.float32)), dtype=np.float64)
            if not np.isfinite(p).all():
                continue
            d = horizontal_distance(p, vp)
            if d <= self.cfg["sampling"]["view_point_max_snap_m"] and abs(p[1] - vp[1]) <= 0.5 and self.on_real_floor(p):
                out.append((p, vp, d))
        return out

    def fallback_points(self, center: np.ndarray, size: np.ndarray) -> List[np.ndarray]:
        """Real-floor navigable points within fallback_target_point_max_m of the box, nearest first."""
        near = []
        bottom = center[1] - size[1] / 2.0
        for _ in range(300):
            self.pf.seed(int(self.rng.integers(2**31 - 1)))
            p = np.array(self.pf.get_random_navigable_point_near(center.astype(np.float32), 2.0, 100), dtype=np.float64)
            if np.isfinite(p).all() and -2.0 <= p[1] - bottom <= 0.5:
                d = box_distance_xz(p, center, size)
                if d <= self.cfg["sampling"]["fallback_target_point_max_m"]:
                    near.append((d, tuple(p)))
        return [np.array(p) for _, p in sorted(set(near)) if self.on_real_floor(np.array(p))]

    def object_target(self, goal: Goal, start: Start, cache: Dict[int, Any]) -> Optional[Tuple[Target, float]]:
        """(target, geodesic) of an object drive to `goal` from this start, or None (start too near / far / other
        floor). `cache` keeps each goal's view points between tries."""
        if goal.object_id not in cache:
            obj = self.labels.objects.get(goal.object_id)
            if obj is None:
                cache[goal.object_id] = None
            else:
                center, size = object_box(obj)
                vps = self.usable_view_points(goal)
                cache[goal.object_id] = (center, size, vps, [] if vps else self.fallback_points(center, size))
        entry = cache[goal.object_id]
        if entry is None:
            return None
        center, size, vps, fallback = entry
        s = np.array(start.position)
        ends = [(self.robot.geodesic(s, p), p, vp, d) for p, vp, d in vps if self.same_floor(s, p)]
        if not ends:
            ends = [(self.robot.geodesic(s, p), p, None, None) for p in fallback if self.same_floor(s, p)]
        ends = [e for e in ends if math.isfinite(e[0])]
        if not ends:
            return None
        geo, p, vp, snap = min(ends, key=lambda e: e[0])
        if not self.in_range(geo):
            return None
        final_yaw = yaw_of(to_2d(p), to_2d(center))
        return Target("object", rounded(p), round(final_yaw, 4), goal.category, goal.object_id,
                      self.labels.names.get(goal.object_id, ""), rounded(center), rounded(size),
                      None if vp is None else rounded(vp), None if snap is None else round(snap, 3),
                      end_fallback=vp is None), geo

    # --- visibility --------------------------------------------------------------------------------------------
    def object_visibility(self, start: Start, object_id: int) -> Dict[str, Any]:
        p = np.array(start.position)
        share = self.labels.target_share(p, start.yaw, object_id)
        n = self.cfg["sampling"]["visibility_headings"]
        any_heading = share > 0 or any(self.labels.target_share(p, start.yaw + 2 * math.pi * k / n, object_id) > 0
                                       for k in range(1, n))
        return {"target_share": round(share, 5), "visible_any_heading": bool(any_heading)}

    def spot_visibility(self, start: Start, target: Target) -> Dict[str, Any]:
        """Is the target floor point (5 cm above the real floor) inside the start picture and unoccluded?"""
        r = self.robot
        r.place(np.array(start.position), start.yaw)
        depth = r.observe()["depth"]
        cam = r.camera_transform()
        q = np.array(target.end_position, dtype=np.float64)
        q[1] = r.floor_y + 0.05
        h, w = depth.shape
        uv = project_points(q[None], cam, w, h, r.spec.hfov_deg)[0]
        if not np.isfinite(uv).all():  # behind the camera
            return {"in_view": False}
        col, row = int(math.floor(uv[0])), int(math.floor(uv[1]))
        if not (0 <= row < h and 0 <= col < w):
            return {"in_view": False}
        distance = float(-((q - cam[:3, 3]) @ cam[:3, :3])[2])  # along the viewing axis, like planar depth
        seen = float(depth[row, col])
        return {"in_view": bool(seen == 0.0 or seen >= distance - 0.15), "in_picture": True}
