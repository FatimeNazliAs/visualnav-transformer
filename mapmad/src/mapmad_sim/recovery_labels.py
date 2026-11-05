"""W1 recovery labels (Phase 14b-3 item 2; DAgger relabelling with a recovery answer, review 15: DAgger, DART, MAER-Nav).

Plain words: in Phase 11 the teacher had no answer when the robot was pressed against a wall (it never reverses), so
38% of the robot's own states were thrown away. Now, when the robot is stuck, the teacher answers "back up 25 cm,
then turn toward where I want to go" -- and the network learns that answer like any other label.

- **Stuck at frame t** (the Phase 14a W4 rule, on the drive's own record): >= 16 collision steps among the 20 moves
  before t, or < 0.10 m net movement over the last 40 steps (t >= 40).
- **Direction:** the teacher's path (teacher_step at t, frozen blacklist) -> the first path point >= 0.5 m from the
  backed-up position (or the path's end); no teacher path (teacher fail) -> the longest free ray on the robot's own map
  around the backed-up position (W4's rule: +-180 deg in 10 deg steps, from 0.3 m, 0.05 m march, 3.15 m reach,
  obstacles and bumped cells block).
- **Label motion** (the Phase 8 motion numbers, 4 Hz, 40 steps): mode "reverse_turn" = 10 steps backward at 0.1 m/s
  (0.25 m straight back), then turn in place at <= 0.4 rad/s to the direction, then -- because the network's 8
  waypoints are positions only (no heading) -- straight ahead at 0.2 m/s for the steps left, so the turn shows in the
  waypoints. Mode "turn" (the pre-set fallback "W1-turn": no backward waypoints) = the same without the backing up.
  The poses after every 5th step -> 8 waypoints in the robot frame at t (points behind the robot allowed).
- No collision check on the label motion (the teacher's labels never had one; the frozen map is the robot's own).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.explorer import Decision, OwnMap, cell_of
from mapmad_sim.expert import wrap

Pose = Tuple[float, float, float]
MODES = ("reverse_turn", "turn")


@dataclass(frozen=True)
class RecoverySpec:
    """configs/p14b3_v3.yaml `rollouts.recovery` (W4's stuck rule and manoeuvre numbers)."""

    mode: str = "reverse_turn"
    window_collisions: int = 20
    min_collisions: int = 16
    window_still: int = 40
    min_move_m: float = 0.10
    reverse_m: float = 0.25
    reverse_v: float = 0.1
    forward_v: float = 0.2
    max_w: float = 0.4
    dt: float = 0.25
    aim_m: float = 0.5
    ray_step_deg: float = 10.0
    ray_start_m: float = 0.3
    ray_march_m: float = 0.05
    ray_reach_m: float = 3.15

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"recovery mode {self.mode!r} not in {MODES}")


def stuck_frames(collided: Sequence[bool], xy: np.ndarray, spec: RecoverySpec) -> np.ndarray:
    """Bool per frame t: stuck by W4's rule on the record (module docstring). collided[i] = the move after frame i."""
    c = np.asarray(collided, dtype=bool)
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    out = np.zeros(n, dtype=bool)
    cs = np.concatenate([[0], np.cumsum(c.astype(np.int64))])
    for t in range(n):
        if t >= spec.window_collisions and cs[t] - cs[t - spec.window_collisions] >= spec.min_collisions:
            out[t] = True
        elif t >= spec.window_still and math.hypot(*(xy[t] - xy[t - spec.window_still])) < spec.min_move_m:
            out[t] = True
    return out


def blocked(m: OwnMap, p: Tuple[float, float]) -> bool:
    """Obstacle or bumped cell (outside the grid = blocked)."""
    i, j = cell_of(p, m)
    if not (0 <= i < m.obstacle.shape[0] and 0 <= j < m.obstacle.shape[1]):
        return True
    return bool(m.obstacle[i, j]) or (m.bumped is not None and bool(m.bumped[i, j]))


def free_ray(m: OwnMap, pose: Pose, spec: RecoverySpec) -> float:
    """World heading of the longest free ray around pose (W4's open_direction on the world grid; ties -> the smaller
    turn, then left)."""
    x, y, yaw = pose
    best, best_angle = None, 0.0
    for k in range(int(round(360.0 / spec.ray_step_deg))):
        deg = -180.0 + k * spec.ray_step_deg
        a = yaw + math.radians(deg)
        length, d = spec.ray_start_m, spec.ray_start_m
        while d <= spec.ray_reach_m:
            if blocked(m, (x + d * math.cos(a), y + d * math.sin(a))):
                break
            length = d
            d += spec.ray_march_m
        key = (round(length, 6), -abs(deg), 1.0 if deg > 0 else 0.0)
        if best is None or key > best:
            best, best_angle = key, a
    return best_angle


def path_heading(dec: Decision, at: Tuple[float, float], spec: RecoverySpec) -> Optional[float]:
    """World heading from `at` to the first path point >= aim_m away (or the path's end); None without a path."""
    path = np.asarray(dec.path, dtype=np.float64)
    if path.ndim != 2 or len(path) < 2:
        return None
    d = np.hypot(path[:, 0] - at[0], path[:, 1] - at[1])
    far = np.flatnonzero(d >= spec.aim_m)
    q = path[far[0]] if len(far) else path[-1]
    if math.hypot(q[0] - at[0], q[1] - at[1]) < 1e-6:
        return None
    return math.atan2(q[1] - at[1], q[0] - at[0])


def recovery_poses(pose: Pose, dec: Optional[Decision], own: OwnMap, spec: RecoverySpec, steps: int) -> np.ndarray:
    """(steps + 1, 3) poses of the recovery motion from pose (module docstring)."""
    from mapmad_sim.dagger_labels import arc

    poses: List[Pose] = [tuple(pose)]
    if spec.mode == "reverse_turn":
        n_rev = int(math.ceil(spec.reverse_m / (spec.reverse_v * spec.dt) - 1e-9))
        for _ in range(min(n_rev, steps)):
            poses.append(arc(poses[-1], -spec.reverse_v, 0.0, spec.dt))
    p = poses[-1]
    heading = path_heading(dec, p[:2], spec) if dec is not None else None
    if heading is None:
        heading = free_ray(own, p, spec)
    delta = wrap(heading - p[2])
    n_turn = int(math.ceil(abs(delta) / (spec.max_w * spec.dt) - 1e-9))
    w = delta / (n_turn * spec.dt) if n_turn else 0.0
    for _ in range(n_turn):
        if len(poses) > steps:
            break
        poses.append(arc(poses[-1], 0.0, w, spec.dt))
    while len(poses) <= steps:
        poses.append(arc(poses[-1], spec.forward_v, 0.0, spec.dt))
    return np.asarray(poses[:steps + 1], dtype=np.float64)
