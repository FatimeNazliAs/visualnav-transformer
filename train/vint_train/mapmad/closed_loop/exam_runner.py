"""One Phase 5 episode: simulator (via the bridge) + policy, until primary success (oracle stop) or the step limit.

Same pattern as Phase 1's runner.py (which stays unchanged for Phase 1), with the exam's metrics (block item 10):
- primary success = walking distance to the nearest usable ObjectNav view point of the target (`vp_geodesic_m`
  from the server) <= success_m (0.2 m); the episode stops at the first step where it holds;
- SPL to that view point (shortest = its walking distance at the start, path = driven length);
- secondary, "reached at any step": target point (Phase 1 point, `geodesic_m`) <= 1.0 m, any side = nearest view
  point <= 1.0 m; success@500 is read from success_step later;
- collisions (move > 1 cm short), stuck streaks (>= stuck_steps consecutive collision steps), final distance.
The policy sees pictures, the goal photo (photo arms), the map tensor (map arm) or its grid (planner) -- never the
geodesic distances or the target. The JSONL log holds no times or host names: the same seed gives a byte-identical
log (gate G5 row 6). It is written to <log>.partial and renamed when the episode is complete.
"""

import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mapmad_bridge import steplog
from mapmad_sim.exam_arms import ExamArm  # noqa: F401  (arms live in mapmad_sim; re-exported here)
from mapmad_sim.run_layout import partial_path
from vint_train.mapmad.closed_loop import metrics
from vint_train.mapmad.closed_loop.policy import Observation, Policy
from vint_train.mapmad.closed_loop.runner import clean


@dataclass
class ExamSettings:
    """configs/p5_exam.yaml `run`."""

    success_key: str
    success_m: float
    max_steps: int
    secondary_m: Dict[str, float]
    stuck_steps: int
    topdown_m_per_px: float

    @classmethod
    def from_config(cls, run: Dict[str, Any]) -> "ExamSettings":
        return cls(run["success_key"], float(run["success_m"]), int(run["max_steps"]), dict(run["secondary_m"]),
                   int(run["stuck_steps"]), float(run["topdown_m_per_px"]))


@dataclass
class Trace:
    """Exact per-step inputs kept for the map-parity check (--trace-maps)."""

    maps: List[np.ndarray] = field(default_factory=list)
    positions: List[List[float]] = field(default_factory=list)
    yaws: List[float] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, maps=np.stack(self.maps), positions=np.asarray(self.positions, np.float64),
                            yaws=np.asarray(self.yaws, np.float64))


class FloorMapMismatch(RuntimeError):
    """The episode was built on another floor map."""


def reached(value: Optional[float], limit: float) -> bool:
    return value is not None and value <= limit


def run_episode(sim, policy: Policy, episode: Dict[str, Any], arm: ExamArm, run: ExamSettings, seed_index: int,
                seed: int, log_path: Path, fingerprint: str, map_settings: Dict[str, Any],
                header_extra: Optional[Dict[str, Any]] = None, trace: Optional[Trace] = None
                ) -> Tuple[Dict[str, Any], List[float]]:
    """Drive one episode; writes log_path; returns (result line, policy milliseconds per step)."""
    import time

    outputs = dict(arm.outputs, map_settings=map_settings)
    first = sim.reset(episode, arm.hfov_deg, goal_photo=arm.photo, topdown_m_per_px=run.topdown_m_per_px,
                      outputs=outputs)
    if first.navmesh_sha256 != episode["navmesh_sha256"]:
        raise FloorMapMismatch(f"{episode['episode_id']}: live floor map {first.navmesh_sha256[:12]} != the set's "
                               f"{episode['navmesh_sha256'][:12]} (robot_limo.yaml navmesh changed?)")
    goal = first.goal_rgb if arm.photo else None
    policy.reset({"seed": seed, "episode_id": episode["episode_id"], "goal": arm.reset_goal,
                  "goal_rect": first.extras.get("goal_rect")})
    frames = deque([first.rgb], maxlen=policy.context_frames)
    extras = first.extras
    state = first.state
    start_vp, start_tp = state[run.success_key], state["geodesic_m"]
    sec = run.secondary_m
    step, path_m, collisions, streak, longest = 0, 0.0, 0, 0, 0
    success = reached(start_vp, run.success_m)
    success_step = 0 if success else None
    tp_step = 0 if reached(start_tp, sec["target_point"]) else None
    as_step = 0 if reached(start_vp, sec["any_side"]) else None
    if trace is not None and "map" in extras:
        trace.maps.append(extras["map"].copy())
        trace.positions.append(list(state["position"]))
        trace.yaws.append(state["yaw"])
    policy_ms: List[float] = []
    partial = partial_path(log_path)
    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "w") as log:
        def write(line: Dict[str, Any]) -> None:
            log.write(json.dumps(clean(line), sort_keys=True) + "\n")

        write({"kind": "episode", "episode_id": episode["episode_id"], "arm": arm.name, "seed_index": seed_index,
               "seed": seed, "home": episode["home"], "category": episode["target"]["category"],
               "distance_bin": episode["distance_bin"], "hfov_deg": arm.hfov_deg, "photo": arm.photo, "map": arm.map,
               "policy": policy.name, "episodes_fingerprint": fingerprint, "start": state,
               "start_vp_geodesic_m": start_vp, "start_target_point_geodesic_m": start_tp,
               "navmesh_sha256": first.navmesh_sha256, "max_steps": run.max_steps, "success_key": run.success_key,
               "success_m": run.success_m, **(header_extra or {})})
        while not success and step < run.max_steps:
            t0 = time.perf_counter()
            cmd = policy.act(Observation(frames=list(frames), goal_image=goal, step=step, extras=extras))
            policy_ms.append((time.perf_counter() - t0) * 1000.0)
            frame = sim.step(cmd.v, cmd.w)
            step += 1
            frames.append(frame.rgb)
            extras, state, move = frame.extras, frame.state, frame.move
            path_m += move["travelled_m"]
            collisions += int(move["collided"])
            streak = streak + 1 if move["collided"] else 0
            longest = max(longest, streak)
            vp, tp = state[run.success_key], state["geodesic_m"]
            success = reached(vp, run.success_m)
            success_step = step if success else None
            tp_step = step if tp_step is None and reached(tp, sec["target_point"]) else tp_step
            as_step = step if as_step is None and reached(vp, sec["any_side"]) else as_step
            if trace is not None and "map" in extras:
                trace.maps.append(extras["map"].copy())
                trace.positions.append(list(state["position"]))
                trace.yaws.append(state["yaw"])
            write({"kind": "step", "step": step, "v": move["v"], "w": move["w"], "policy": cmd.info,
                   "position": state["position"], "yaw": state["yaw"], "vp_geodesic_m": vp, "geodesic_m": tp,
                   "collided": move["collided"], "travelled_m": move["travelled_m"], "short_m": move["short_m"],
                   "refused_substeps": move["refused_substeps"], "floor_below_feet_m": state["floor_below_feet_m"]})
        shortest = start_vp if start_vp is not None else math.inf
        result = {"kind": "result", "episode_id": episode["episode_id"], "arm": arm.name, "seed_index": seed_index,
                  "seed": seed, "home": episode["home"], "category": episode["target"]["category"],
                  "distance_bin": episode["distance_bin"], "success": bool(success), "success_step": success_step,
                  "steps": step, "spl": metrics.spl(success, shortest, path_m), "collisions": collisions,
                  "collision_share": collisions / step if step else 0.0, "longest_collision_streak": longest,
                  "path_length_m": path_m, "final_vp_geodesic_m": state[run.success_key],
                  "final_target_point_geodesic_m": state["geodesic_m"], "start_vp_geodesic_m": start_vp,
                  "start_target_point_geodesic_m": start_tp, "target_point_step": tp_step, "any_side_step": as_step}
        write(result)
    os.replace(partial, log_path)
    log = steplog.read(log_path)
    result["stuck_streaks"] = len(steplog.stuck_streaks(log, run.stuck_steps))
    return clean(result), policy_ms


def noise_seed(salt: str, episode_id: str, seed_index: int) -> int:
    """Block item 7: the diffusion-noise seed of one (episode, seed index), the same for every arm."""
    import hashlib

    return int(hashlib.sha256(f"{salt}/{episode_id}/{seed_index}".encode()).hexdigest()[:8], 16)
