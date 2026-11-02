"""Phase 14a driving switches outside the network (kickoff 14.4; arm table fields `pick` and `unstuck`; configs
p14a_finish.yaml `switches`). Switched off, a run is exactly Phase 13's.

A1 "stick with your choice" (`StickPicker`, arm `pick: stick`; BID 2408.17355 backward coherence with rho = 0.5,
IMLE Policy 2502.12371 periodic reset C = 10): of the 8 diffusion samples, take the one closest to the plan chosen at
the previous step. The previous plan (8 waypoints, metres, robot frame: x forward, y left) is first moved into the
current robot frame by the command the robot was given (unicycle, (v, w) held for one control period), then
distance(sample) = sum_k rho^k * |sample_k - previous_k| (k = 0..7, Euclidean per waypoint); the smallest wins (ties:
the lower index). The first step of an episode, every step with step % reset_every == 0 (10 = 2.5 s) and the first
step after a get-unstuck manoeuvre take sample 0 (Phase 13's choice). All 8 samples are still drawn every step, so the
random numbers are the same as without the switch.

W4 get-unstuck rule (`Unstuck`, arm `unstuck: w4`; What Matters 2510.01830, Fare 2510.24680): after every executed
step the rule looks at the last 20 steps (stuck if >= 16 had a collision) and the last 40 steps (stuck if the robot's
position moved < 0.10 m net). Stuck, and fewer than 3 manoeuvres started in the last 100 steps -> manoeuvre:
1. reverse 0.25 m at 0.1 m/s (10 control steps of v = -0.1, w = 0; the only place a run may reverse);
2. turn in place towards the most open direction of the near map window (the obstacle layer of the 64 x 64, 10 cm
   window the server sends every arm): rays every 10 deg in [-180, 180) from 0.3 m outward in 0.05 m steps until an
   obstacle cell or the window edge; the longest ray wins (ties: the smaller |angle|, then left); turned at most
   0.4 rad/s in ceil(|angle| / (0.4 rad/s x 0.25 s)) equal steps;
3. hand back to the policy (both histories cleared; A1 restarts from sample 0).
The network is not asked during a manoeuvre. Every manoeuvre is counted and logged (step line `policy.unstuck`).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vint_train.mapmad.local_map import cell_centres


# --- A1 -----------------------------------------------------------------------------------------------------------
def shift_plan(plan: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
    """A plan (n, 2) in the robot frame before a move (v, w held for dt) -> the same points in the frame after it."""
    th = w * dt
    if abs(w) < 1e-9:
        dx, dy = v * dt, 0.0
    else:
        dx, dy = (v / w) * math.sin(th), (v / w) * (1.0 - math.cos(th))
    p = np.asarray(plan, np.float64) - np.array([dx, dy])
    c, s = math.cos(th), math.sin(th)
    return np.stack([c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]], axis=1)


def coherence_pick(samples: np.ndarray, previous: np.ndarray, rho: float) -> int:
    """Index of the sample (S, n, 2) with the smallest rho-weighted waypoint distance to `previous` (n, 2)."""
    w = rho ** np.arange(samples.shape[1])
    d = (np.linalg.norm(samples - previous[None], axis=2) * w[None]).sum(axis=1)
    return int(np.argmin(d))


@dataclass
class StickPicker:
    """A1 (module docstring). `pick` once per network step; `executed` with the command sent; `reset` per episode."""

    rho: float = 0.5
    reset_every: int = 10
    dt: float = 0.25
    previous: Optional[np.ndarray] = None
    picks: int = 0

    def reset(self) -> None:
        self.previous = None

    def pick(self, samples_m: np.ndarray, step: int) -> int:
        """samples_m (S, n, 2) metres, robot frame; step = the runner's step index."""
        if self.previous is None or step % self.reset_every == 0:
            k = 0
        else:
            k = coherence_pick(samples_m, self.previous, self.rho)
        self.previous = np.asarray(samples_m[k], np.float64)
        self.picks += 1
        return k

    def executed(self, v: float, w: float) -> None:
        """The command the robot got for the plan just picked: move the plan into the next step's frame."""
        if self.previous is not None:
            self.previous = shift_plan(self.previous, v, w, self.dt)


# --- W4 -----------------------------------------------------------------------------------------------------------
def open_direction(obstacle: np.ndarray, resolution: float = 0.1, step_deg: float = 10.0, start_m: float = 0.3,
                   march_m: float = 0.05) -> Tuple[float, float]:
    """(angle in rad, free length in m) of the longest free ray of a robot-centred window (row 0 = farthest ahead,
    col 0 = farthest left: local_map.cell_centres); module docstring step 2."""
    size = obstacle.shape[0]
    half = size / 2.0 - 0.5
    reach = half * resolution
    best: Tuple[float, float, float, int] = (-1.0, 0.0, 0.0, 0)  # (length, -|angle|, side, ...) maximised
    best_angle, best_len = 0.0, 0.0
    for k in range(int(round(360.0 / step_deg))):
        deg = -180.0 + k * step_deg
        a = math.radians(deg)
        length = start_m
        d = start_m
        while d <= reach:
            r = int(round(half - d * math.cos(a) / resolution))
            c = int(round(half - d * math.sin(a) / resolution))
            if not (0 <= r < size and 0 <= c < size) or obstacle[r, c] > 0:
                break
            length = d
            d += march_m
        key = (round(length, 6), -abs(deg), 1.0 if deg > 0 else 0.0, 0)
        if key > best:
            best, best_angle, best_len = key, a, length
    return best_angle, best_len


@dataclass
class UnstuckSpec:
    """configs/p14a_finish.yaml `switches.w4`."""

    window_collisions: int = 20
    min_collisions: int = 16
    window_still: int = 40
    min_move_m: float = 0.10
    reverse_m: float = 0.25
    reverse_v: float = 0.1
    turn_step_deg: float = 10.0
    max_per_100: int = 3

    @classmethod
    def from_config(cls, d: Dict[str, Any]) -> "UnstuckSpec":
        return cls(int(d["window_collisions"]), int(d["min_collisions"]), int(d["window_still"]),
                   float(d["min_move_m"]), float(d["reverse_m"]), float(d["reverse_v"]), float(d["turn_step_deg"]),
                   int(d["max_per_100"]))


@dataclass
class Unstuck:
    """W4 (module docstring). Per step: `command(step, extras)` -> (v, w, reverse, info) during a manoeuvre, else
    None (the policy drives); then `observe(step, move, position)` with what the step did."""

    spec: UnstuckSpec
    max_w: float = 0.4
    dt: float = 0.25
    collided: Deque[bool] = field(default_factory=deque)
    positions: Deque[Tuple[float, float]] = field(default_factory=deque)
    plan: List[Tuple[float, float, bool, str]] = field(default_factory=list)  # (v, w, reverse, phase) still to send
    starts: List[int] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    just_finished: bool = False

    def reset(self) -> None:
        self.collided, self.positions = deque(maxlen=self.spec.window_collisions), deque(maxlen=self.spec.window_still + 1)
        self.plan, self.starts, self.events, self.just_finished = [], [], [], False

    def __post_init__(self) -> None:
        self.reset()

    def stuck(self) -> Optional[str]:
        """Why the robot counts as stuck now ("collisions" | "still"), or None."""
        s = self.spec
        if len(self.collided) >= s.window_collisions and sum(self.collided) >= s.min_collisions:
            return "collisions"
        if len(self.positions) > s.window_still:
            (x0, y0), (x1, y1) = self.positions[0], self.positions[-1]
            if math.hypot(x1 - x0, y1 - y0) < s.min_move_m:
                return "still"
        return None

    def observe(self, step: int, collided: bool, position_xy: Sequence[float]) -> None:
        """After an executed step (the policy's or a manoeuvre's)."""
        if self.plan:
            return  # inside a manoeuvre: histories restart when it ends
        if self.just_finished:
            self.collided.clear()
            self.positions.clear()
            self.just_finished = False
        self.collided.append(bool(collided))
        self.positions.append((float(position_xy[0]), float(position_xy[1])))

    def command(self, step: int, extras: Dict[str, Any]) -> Optional[Tuple[float, float, bool, Dict[str, Any]]]:
        """The manoeuvre's command for this step, or None (the policy drives)."""
        if not self.plan:
            why = self.stuck()
            recent = [s for s in self.starts if s > step - 100]
            if why is None or len(recent) >= self.spec.max_per_100:
                return None
            n_rev = int(math.ceil(self.spec.reverse_m / (self.spec.reverse_v * self.dt) - 1e-9))
            self.plan = [(-self.spec.reverse_v, 0.0, True, "reverse")] * n_rev + [(0.0, 0.0, False, "turn?")]
            self.starts.append(step)
            self.events.append({"step": step, "why": why})
        v, w, rev, phase = self.plan.pop(0)
        if phase == "turn?":  # reversing done: choose the turn on the map of this step
            maps = extras.get("maps")
            angle, free = open_direction(np.asarray(maps[0][0]) if maps is not None else np.zeros((64, 64)),
                                         step_deg=self.spec.turn_step_deg)
            n = int(math.ceil(abs(angle) / (self.max_w * self.dt) - 1e-9))
            self.events[-1].update(turn_deg=round(math.degrees(angle), 1), free_m=round(free, 2), turn_steps=n)
            if n == 0:
                self.just_finished = True
                return None
            self.plan = [(0.0, angle / (n * self.dt), False, "turn")] * n
            v, w, rev, phase = self.plan.pop(0)
        if not self.plan:
            self.just_finished = True
        return v, w, rev, {"unstuck": phase, "unstuck_n": len(self.starts)}
