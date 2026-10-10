"""The "perfect driver" of the practice drives (plan D12): a smooth curve along the planned path, followed by
pure pursuit within LIMO's speed limits. No simulator needed (tested in tests/test_expert.py).

- `smooth_path`: the planner's 5 cm cell path is resampled every 5 cm and averaged over a sliding window
  (corners become curves); a smoothed point is kept only where it still has room (clearance), else the
  original point stays.
- `PurePursuit`: aims at the path point `lookahead_m` (0.5 m) ahead of the robot's progress (0.3 m where the
  path turns sharply, so tight corners are not cut). The arc through that point has curvature
  k = 2 sin(a) / d (a = angle to it, d = distance); speed v = max_v, lowered so that |w| = |k v| <= max_w
  (slower in tight turns), lowered again where the path itself bends harder within the next lookahead_m
  (v <= 0.8 max_w / curvature), and near the end. Turning on the spot is not part of it: the drive loop
  (datagen.py) turns on the spot only at the start, in wall recoveries and in the final face-the-object turn;
  if the aim point ever falls behind the robot (> 90 deg off, e.g. after a squeeze), pure pursuit returns
  a spot turn flagged `forced`, which is logged and counted against gate G2 row 2. Never reverses.
All poses are (x, y, yaw) in the 2D frame of traj_data.pkl (frames.py).
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np

from mapmad_sim.frames import to_robot

Pose = Tuple[float, float, float]


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def resample(points: np.ndarray, step_m: float) -> np.ndarray:
    """Polyline -> points every step_m along it (first and last point kept)."""
    seg = np.hypot(*np.diff(points, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0.0:
        return points[:1].copy()
    n = max(2, int(math.ceil(s[-1] / step_m)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])], axis=1)


def smooth_path(points: np.ndarray, clearance: Callable[[np.ndarray], float], step_m: float = 0.05,
                window_m: float = 0.5, passes: int = 3, min_clearance_m: float = 0.05) -> np.ndarray:
    """Planner path -> smooth curve (see the module docstring); endpoints stay fixed."""
    path = resample(points, step_m)
    if len(path) < 3:
        return path
    k = max(1, int(round(window_m / step_m / 2)))
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    for _ in range(passes):
        padded = np.concatenate([np.repeat(path[:1], k, 0), path, np.repeat(path[-1:], k, 0)])
        avg = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)], axis=1)
        ok = np.array([clearance(p) >= min(min_clearance_m, clearance(q)) for p, q in zip(avg, path)])
        ok[0] = ok[-1] = False
        path = np.where(ok[:, None], avg, path)
    return resample(path, step_m)


@dataclass(frozen=True)
class PursuitSpec:
    """Expert-driver numbers (`expert` block of p2_datagen.yaml) + the robot's limits."""

    max_v: float
    max_w: float
    dt: float
    lookahead_m: float = 0.5
    tight_lookahead_m: float = 0.3
    tight_turn_deg: float = 45.0  # path turns more than this within the next lookahead_m -> tight look-ahead
    min_v: float = 0.04  # slowest forward speed while following (well above the 0.02 m/s spot-turn line)
    approach_gain: float = 0.8  # v <= gain * distance to the end (1/s)
    goal_tolerance_m: float = 0.05
    start_turn_deg: float = 10.0  # spot turn at the start while the aim point is more than this off
    final_turn_tolerance_deg: float = 10.0
    curve_speed_factor: float = 0.8  # v <= factor * max_w / (path curvature over the next lookahead_m)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], max_v: float, max_w: float, dt: float) -> "PursuitSpec":
        keys = ("lookahead_m", "tight_lookahead_m", "tight_turn_deg", "min_v", "approach_gain", "goal_tolerance_m",
                "start_turn_deg", "final_turn_tolerance_deg", "curve_speed_factor")
        return cls(max_v=max_v, max_w=max_w, dt=dt, **{k: float(cfg[k]) for k in keys})


@dataclass(frozen=True)
class Command:
    v: float
    w: float
    forced_spot_turn: bool = False
    done: bool = False


class TrackedPath:
    """A smooth path with its arc length, and the robot's progress along it (never goes back)."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        seg = np.hypot(*np.diff(self.points, axis=0).T) if len(self.points) > 1 else np.zeros(0)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.index = 0

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def update(self, xy: np.ndarray, window_m: float = 1.0) -> int:
        """Move the progress index to the nearest path point within window_m ahead of it."""
        hi = int(np.searchsorted(self.s, self.s[self.index] + window_m, side="right"))
        d = np.hypot(*(self.points[self.index:hi] - xy).T)
        self.index += int(d.argmin())
        return self.index

    def point_at(self, s: float) -> np.ndarray:
        s = min(max(s, 0.0), self.length)
        return np.array([np.interp(s, self.s, self.points[:, 0]), np.interp(s, self.s, self.points[:, 1])])

    def turn_ahead(self, from_s: float, over_m: float) -> float:
        """Total absolute heading change (rad) of the path between from_s and from_s + over_m."""
        lo = int(np.searchsorted(self.s, from_s))
        hi = int(np.searchsorted(self.s, from_s + over_m, side="right"))
        pts = self.points[lo:hi + 1]
        if len(pts) < 3:
            return 0.0
        h = np.arctan2(*np.diff(pts, axis=0).T[::-1])
        return float(np.abs(np.angle(np.exp(1j * np.diff(h)))).sum())


class PurePursuit:
    """Follows a TrackedPath; `command(pose)` gives the next (v, w)."""

    def __init__(self, spec: PursuitSpec, path: TrackedPath) -> None:
        self.spec, self.path = spec, path

    def aim(self, pose: Pose) -> Tuple[np.ndarray, float]:
        """(aim point, look-ahead used) for the current pose; updates the progress."""
        sp, path = self.spec, self.path
        path.update(np.array(pose[:2]))
        s = float(path.s[path.index])
        tight = path.turn_ahead(s, sp.lookahead_m) > math.radians(sp.tight_turn_deg)
        look = sp.tight_lookahead_m if tight else sp.lookahead_m
        return path.point_at(s + look), look

    def angle_to_aim(self, pose: Pose) -> float:
        """Angle (rad, + = left) from the robot's heading to the aim point."""
        target, _ = self.aim(pose)
        f, l = to_robot(target[None], pose)[0]
        return math.atan2(l, f)

    def distance_to_end(self, pose: Pose) -> float:
        return float(math.hypot(*(self.path.points[-1] - np.array(pose[:2]))))

    def command(self, pose: Pose) -> Command:
        sp = self.spec
        end = self.distance_to_end(pose)
        if end <= sp.goal_tolerance_m:
            return Command(0.0, 0.0, done=True)
        target, _ = self.aim(pose)
        f, l = to_robot(target[None], pose)[0]
        d, alpha = math.hypot(f, l), math.atan2(l, f)
        if abs(alpha) > math.pi / 2:  # aim point behind the robot: no forward arc reaches it
            return Command(0.0, math.copysign(sp.max_w, alpha), forced_spot_turn=True)
        k = 2.0 * math.sin(alpha) / max(d, 1e-6)
        v = sp.max_v
        if abs(k) > 1e-9:
            v = min(v, sp.max_w / abs(k))
        # the path may bend harder than the arc to the aim point: slow down for it too, or the robot runs wide
        bend = self.path.turn_ahead(float(self.path.s[self.path.index]), sp.lookahead_m) / sp.lookahead_m
        if bend > 1e-6:
            v = min(v, sp.curve_speed_factor * sp.max_w / bend)
        remaining = max(end, self.path.length - float(self.path.s[self.path.index]))
        v = max(sp.min_v, min(v, sp.approach_gain * remaining))
        return Command(v, float(np.clip(k * v, -sp.max_w, sp.max_w)))


def spot_turn(error_rad: float, max_w: float, dt: float) -> float:
    """Turn rate that removes a heading error on the spot, at most max_w, landing on it in the last step."""
    return float(np.clip(error_rad / dt, -max_w, max_w))


def is_spot_turn(v: float, w: float, max_v: float = 0.02, min_w: float = 0.1) -> bool:
    """Gate G2 row 2's definition of a turn-on-the-spot frame: |v| < 0.02 m/s and |w| > 0.1 rad/s."""
    return abs(v) < max_v and abs(w) > min_w


def heading_error(pose: Pose, yaw: float) -> float:
    return wrap(yaw - pose[2])

