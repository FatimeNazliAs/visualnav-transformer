"""A practice drive's route and its random variations (Phase 2 task 2.3; used by datagen.py).

- Variations: what a drive is asked to include, drawn before it runs: a wall-recovery start, 1-2 detours, a kick.
- plan_route: the clearance-aware shortest path on the start's island (floor_grid.py); for detour drives through
  1-2 detour points offset 0.5-1.5 m sideways, accepted only if (a) each point is reachable sideways in a straight
  line with room around it, (b) the route is not much longer than the bulges explain (no loop behind a wall),
  (c) the smoothed route turns at most max_turn_deg within any turn_window_m (no hairpin) and (d) still leaves the
  shortest path by >= realised_m after smoothing; then smoothed (expert.smooth_path).
All points are in the 2D frame of traj_data.pkl (frames.py).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mapmad_sim.expert import TrackedPath, resample, smooth_path
from mapmad_sim.floor_grid import FloorGrid


@dataclass
class Variations:
    """What a drive is asked to include (drawn before it runs; kept when a discarded drive is redrawn)."""

    wall_recovery: bool = False
    detours: int = 0
    kick_deg: float = 0.0  # signed; 0 = no kick
    kick_at: float = 0.0  # fraction of the path length


def draw_variations(cfg: Dict[str, Any], rng: np.random.Generator) -> Variations:
    v = cfg["variations"]
    out = Variations()
    out.wall_recovery = bool(rng.random() < v["wall_recovery"]["share"])
    if rng.random() < v["detour"]["share"]:
        lo, hi = v["detour"]["count"]
        out.detours = int(rng.integers(lo, hi + 1))
    if rng.random() < v["kick"]["share"]:
        out.kick_deg = float(rng.uniform(*v["kick"]["angle_deg"]) * rng.choice([-1.0, 1.0]))
        out.kick_at = float(rng.uniform(*v["kick"]["at_progress"]))
    return out


@dataclass
class Route:
    path: np.ndarray  # smooth path to follow (N, 2)
    shortest: np.ndarray  # clearance-aware shortest path, resampled (M, 2)
    detour_points: List[List[float]]
    detour_offsets_m: List[float]
    max_deviation_m: float  # farthest the smooth path gets from the shortest one


def path_length(points: np.ndarray) -> float:
    return float(np.hypot(*np.diff(points, axis=0).T).sum()) if len(points) > 1 else 0.0


def detour_point(grid: FloorGrid, shortest: np.ndarray, frac: float, cfg: Dict[str, Any],
                 rng: np.random.Generator) -> Optional[Tuple[np.ndarray, float]]:
    """A point offset sideways from the shortest path at fraction frac of its length, with room around it."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(shortest, axis=0).T))])
    i = int(np.clip(np.searchsorted(s, frac * s[-1]), 1, len(shortest) - 2))
    tangent = shortest[min(i + 3, len(shortest) - 1)] - shortest[max(i - 3, 0)]
    tangent /= max(np.linalg.norm(tangent), 1e-9)
    normal = np.array([-tangent[1], tangent[0]])
    lo, hi = cfg["offset_m"]
    first = float(rng.uniform(lo, hi))
    offsets = [first] + [o for o in np.arange(first - 0.1, lo - 1e-9, -0.1)]
    side = float(rng.choice([-1.0, 1.0]))
    for sgn in (side, -side):
        for off in offsets:
            p = shortest[i] + sgn * off * normal
            # reachable sideways in a straight line: a bulge of the path, not a trip into a side room
            if grid.clearance_at(p) >= cfg["min_clearance_m"] and grid.segment_clear(shortest[i], p, cfg["min_clearance_m"]):
                return p, float(off)
    return None


def deviation(path: np.ndarray, shortest: np.ndarray) -> float:
    """Farthest distance (m) of a path's points from the shortest path's points."""
    return float(np.hypot(*(path[:, None, :] - shortest[None, :, :]).transpose(2, 0, 1)).min(axis=1).max())


def max_turn(path: np.ndarray, window_m: float) -> float:
    """Largest heading change (rad) of a path within any window_m of its length."""
    tracked = TrackedPath(path)
    return max((tracked.turn_ahead(float(s), window_m) for s in tracked.s[::2]), default=0.0)


def plan_route(grid: FloorGrid, a: np.ndarray, b: np.ndarray, detours: int, cfg: Dict[str, Any],
               rng: np.random.Generator) -> Optional[Route]:
    """Route from a to b (2D points), through `detours` detour points when the path is long enough."""
    ex, dc = cfg["expert"], cfg["variations"]["detour"]
    raw = grid.plan(a, b)
    if raw is None:
        return None
    shortest = resample(raw, ex["smooth_step_m"])
    length = path_length(shortest)
    vias: List[Tuple[np.ndarray, float]] = []
    if detours and length >= dc["min_path_m"]:
        fracs = [rng.uniform(0.3, 0.7)] if detours == 1 else [rng.uniform(0.2, 0.4), rng.uniform(0.6, 0.8)]
        for _ in range(dc["tries"]):
            vias = [v for v in (detour_point(grid, shortest, f, dc, rng) for f in fracs) if v is not None]
            pieces = [grid.plan(p, q) for p, q in zip([a] + [v[0] for v in vias], [v[0] for v in vias] + [b])]
            if vias and all(pc is not None for pc in pieces):
                full = np.concatenate([pieces[0]] + [pc[1:] for pc in pieces[1:]])
                smooth = smooth_path(full, grid.clearance_at, ex["smooth_step_m"], ex["smooth_window_m"],
                                     ex["smooth_passes"], ex["smooth_min_clearance_m"])
                if path_length(full) <= length + 2.5 * sum(v[1] for v in vias) + 1.0 and \
                        max_turn(smooth, dc["turn_window_m"]) <= math.radians(dc["max_turn_deg"]) and \
                        deviation(smooth, shortest) >= dc["realised_m"]:  # no hairpins; smoothing keeps the bulge
                    raw = full
                    break
            vias = []
            fracs = [float(np.clip(f + rng.uniform(-0.1, 0.1), 0.15, 0.85)) for f in fracs]
    path = smooth_path(raw, grid.clearance_at, ex["smooth_step_m"], ex["smooth_window_m"], ex["smooth_passes"],
                       ex["smooth_min_clearance_m"])
    return Route(path, shortest, [[round(float(x), 3) for x in v[0]] for v in vias],
                 [round(v[1], 3) for v in vias], deviation(path, shortest))
