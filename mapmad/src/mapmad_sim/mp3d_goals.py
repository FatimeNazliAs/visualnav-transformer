"""Phase 13 (13.2): target objects and view points of an MP3D home WITHOUT official ObjectNav episodes (the 18 test
homes), built the way ObjectNav builds them, so exam B's test homes get goals and success zones like its val homes.

For every semantic object whose mpcat40 category is one of our 6 (chair, sofa, bed, toilet, tv_monitor, plant):
- candidates = points on a `grid_m` (0.1 m, the official spacing) grid whose horizontal distance to the object's
  axis-aligned box (objects.object_box, x-z rectangle) is <= radius_m;
- each candidate is snapped onto LIMO's floor map (asked at the box bottom, so the object's own storey); kept if the
  snapped point lies <= snap_m (0.05 m) horizontally from the grid point and at most `below_m` (2.0 m) below the box
  bottom and not above the box centre;
- visibility: the robot (pin025 camera) on the point, facing the box centre, sees >= min_pixels (1) pixel of the object.
The kept points are the object's view points (objectnav.Goal, position = box centre); the set builder then applies
its own usable rule (snapped <= 0.2 m, real floor) exactly as to official view points. Objects of the category with
no view point stay in the goal list (they count as instances for "nothing visible at the start", like ObjectNav's
category ids) but can never be a target.
radius_m is calibrated on MP3D TRAIN homes against their official view points (make_p13_sets.py vp-calibrate;
p13_rules.md), never on exam homes.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from mapmad_sim.categories import CATEGORIES
from mapmad_sim.objectnav import Goal
from mapmad_sim.objects import object_box
from mapmad_sim.robot import horizontal_distance, yaw_towards
from mapmad_sim.scene_labels import category_name


@dataclass(frozen=True)
class VpRule:
    """The view-point numbers (configs/p13_claim.yaml `mp3d_view_points`)."""

    radius_m: float = 1.0
    grid_m: float = 0.1
    snap_m: float = 0.05
    below_m: float = 2.0
    min_pixels: int = 1

    @classmethod
    def from_config(cls, c: Dict[str, Any]) -> "VpRule":
        return cls(float(c["radius_m"]), float(c["grid_m"]), float(c["snap_m"]), float(c["below_m"]),
                   int(c["min_pixels"]))


def box_distance_xz(p: Sequence[float], centre: np.ndarray, size: np.ndarray) -> float:
    """Horizontal distance from point p to the x-z rectangle of an axis-aligned box (0 inside)."""
    dx = max(abs(p[0] - centre[0]) - size[0] / 2.0, 0.0)
    dz = max(abs(p[2] - centre[2]) - size[2] / 2.0, 0.0)
    return math.hypot(dx, dz)


def grid_candidates(centre: np.ndarray, size: np.ndarray, rule: VpRule) -> np.ndarray:
    """(N, 3) grid points (y = box bottom) within radius_m of the box's x-z rectangle, on the world grid_m lattice."""
    lo = np.floor((centre[[0, 2]] - size[[0, 2]] / 2.0 - rule.radius_m) / rule.grid_m).astype(int)
    hi = np.ceil((centre[[0, 2]] + size[[0, 2]] / 2.0 + rule.radius_m) / rule.grid_m).astype(int)
    xs, zs = np.meshgrid(np.arange(lo[0], hi[0] + 1) * rule.grid_m, np.arange(lo[1], hi[1] + 1) * rule.grid_m,
                         indexing="ij")
    pts = np.stack([xs.ravel(), np.full(xs.size, centre[1] - size[1] / 2.0), zs.ravel()], axis=1)
    keep = [box_distance_xz(p, centre, size) <= rule.radius_m for p in pts]
    return pts[np.asarray(keep, dtype=bool)]


def view_points(robot: Any, object_id: int, centre: np.ndarray, size: np.ndarray, rule: VpRule) -> List[List[float]]:
    """The generated view points of one object (module docstring), in grid order."""
    bottom = centre[1] - size[1] / 2.0
    out: List[List[float]] = []
    seen = set()
    for g in grid_candidates(centre, size, rule):
        s = np.array(robot.pathfinder.snap_point(g.astype(np.float32)), dtype=np.float64)
        if not np.isfinite(s).all() or horizontal_distance(s, g) > rule.snap_m:
            continue
        if not bottom - rule.below_m <= s[1] <= centre[1]:
            continue
        key = tuple(np.round(s, 3))
        if key in seen:
            continue
        seen.add(key)
        try:
            robot.place(s, yaw_towards(s, centre))
        except ValueError:
            continue
        if int((robot.observe()["semantic"] == object_id).sum()) >= rule.min_pixels:
            out.append([round(float(v), 5) for v in s])
    return out


def generated_goals(robot: Any, rule: VpRule, categories: Sequence[str] = CATEGORIES) -> List[Goal]:
    """Goal objects of our categories with generated view points (module docstring), ordered by object id."""
    goals = []
    for o in sorted((o for o in robot.sim.semantic_scene.objects if o is not None), key=lambda o: o.semantic_id):
        cat = category_name(o, True)
        if cat not in categories:
            continue
        centre, size = object_box(o)
        goals.append(Goal(int(o.semantic_id), cat, [round(float(v), 5) for v in centre],
                          view_points(robot, int(o.semantic_id), centre, size, rule)))
    return goals
