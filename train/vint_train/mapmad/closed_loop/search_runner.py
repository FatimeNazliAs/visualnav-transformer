"""One Phase 11 search episode: simulator (via the bridge, online map v2) + a policy, until the robot first enters the
success zone (oracle stop) or the step limit (v2 plan §8.8 metrics, Phase 11 block item 1).

- success = walking distance to the nearest usable ObjectNav view point of ANY instance of the category
  (`vp_geodesic_m`) <= success_m (0.2 m), first entry; SPL to that view point set (shortest = its start value);
- search / approach split: `first_in_view_step` = the first frame where an instance of the category passes the
  simulator-truth "seen" rule (>= 200 px at <= 4 m; server state); search steps = up to it, approach steps = after it;
- collisions (move > 1 cm short), longest collision streak, stuck streaks (>= stuck_steps consecutive collision steps).
The policy sees pictures and the map windows only -- never the distances or the target. The JSONL log holds no times or
host names: the same seed on the same GPU gives a byte-identical log (block item 9 determinism). It is written to
<log>.partial and renamed when the episode is complete. With `save_dir` the server also writes the whole episode in
the Phase 8 drive format (rollouts, block item 2).

Phase 12 (run settings `stop_at_success: false`, `rule_stop`, `house_seen`; absent = Phase 11 exactly): every run
drives max_steps (1,000) and does not stop at the success zone; success / success_step / SPL stay the oracle stop
(first entry; SPL with the path driven up to it); every step line adds `rule_value` + `rule_cell` (the rule-stop value
of that frame, search_exam.rule_value; the rule stop is judged from the log with tau, closed_loop/p12_score.py),
`n_dets` (detections drawn into the target layer) and `house_seen` (house_seen.HouseSeen: house cells seen so far);
the header adds `house_cells`. The recogniser's round-trip time goes to the timing file only (never into the log).
"""

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mapmad_bridge import steplog
from mapmad_sim.run_layout import partial_path
from vint_train.mapmad.closed_loop import metrics
from vint_train.mapmad.closed_loop.policy import Observation, Policy
from vint_train.mapmad.closed_loop.runner import clean


@dataclass
class SearchSettings:
    """configs/p11_dagger.yaml / p12_exam.yaml `run`."""

    success_m: float
    max_steps: int
    stuck_steps: int
    topdown_m_per_px: float
    stop_at_success: bool = True  # Phase 12: False (every run drives max_steps)
    house_seen: Optional[Dict[str, Any]] = None  # Phase 12: {max_m, cell_m}
    live_rule_tau: Optional[float] = None  # Phase 12 check only: stop at the first step with rule_value >= tau

    @classmethod
    def from_config(cls, run: Dict[str, Any]) -> "SearchSettings":
        tau = run.get("live_rule_tau")
        return cls(float(run["success_m"]), int(run["max_steps"]), int(run["stuck_steps"]),
                   float(run["topdown_m_per_px"]), bool(run.get("stop_at_success", True)), run.get("house_seen"),
                   None if tau is None else float(tau))


class FloorMapMismatch(RuntimeError):
    """The episode was built on another floor map."""


def reached(value: Optional[float], limit: float) -> bool:
    return value is not None and value <= limit


def run_search_episode(sim, policy: Policy, episode: Dict[str, Any], arm: str, run: SearchSettings, seed_index: int,
                       seed: int, log_path: Path, fingerprint: str, search_v2: Dict[str, Any], camera: Dict[str, Any],
                       header_extra: Optional[Dict[str, Any]] = None, save_dir: Optional[Path] = None,
                       outputs_extra: Optional[Dict[str, Any]] = None, rec_ms: Optional[List[float]] = None
                       ) -> Tuple[Dict[str, Any], List[float]]:
    """Drive one episode; writes log_path (and the drive to save_dir); returns (result line, policy ms per step).
    search_v2 = search_v2.search_settings(...) + {"noise_seed", "record"} (+ Phase 12 "target", "rule", "teacher");
    outputs_extra = more server outputs (V1: "map" + "map_settings"); rec_ms collects the recogniser's round trips."""
    first = sim.reset(episode, camera["hfov_deg"], goal_photo=False, topdown_m_per_px=run.topdown_m_per_px,
                      outputs={"search_v2": search_v2, **(outputs_extra or {})}, camera=camera)
    if first.navmesh_sha256 != episode["navmesh_sha256"]:
        raise FloorMapMismatch(f"{episode['episode_id']}: live floor map {first.navmesh_sha256[:12]} != the set's "
                               f"{episode['navmesh_sha256'][:12]}")
    policy.reset({"seed": seed, "episode_id": episode["episode_id"], "goal": "masked"})
    frames = deque([first.rgb], maxlen=policy.context_frames)
    extras, state = first.extras, first.state
    start_vp = state["vp_geodesic_m"]
    step, path_m, collisions, streak, longest = 0, 0.0, 0, 0, 0
    success = reached(start_vp, run.success_m)
    success_step = 0 if success else None
    path_at_success = 0.0
    exam = "rule" in extras  # Phase 12 per-step fields (module docstring)
    house = None
    if exam and run.house_seen:
        from vint_train.mapmad.closed_loop.house_seen import HouseSeen

        td = first.topdown
        house = HouseSeen(np.asarray(td["grid"]), td["origin"], td["m_per_px"], state["position"],
                          camera["hfov_deg"], float(run.house_seen["max_m"]), float(run.house_seen["cell_m"]))
        house.update(state["position"], state["yaw"])
    policy_ms: List[float] = []
    partial = partial_path(log_path)
    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "w") as log:
        def write(line: Dict[str, Any]) -> None:
            log.write(json.dumps(clean(line), sort_keys=True) + "\n")

        write({"kind": "episode", "episode_id": episode["episode_id"], "arm": arm, "seed_index": seed_index,
               "seed": seed, "home": episode["home"], "category": episode["target"]["category"],
               "distance_bin": episode["distance_bin"], "policy": policy.name, "episodes_fingerprint": fingerprint,
               "start": state, "start_vp_geodesic_m": start_vp, "navmesh_sha256": first.navmesh_sha256,
               "max_steps": run.max_steps, "success_m": run.success_m, "camera": camera["name"],
               "noise_seed": search_v2["noise_seed"], **(header_extra or {}),
               **({"house_cells": house.house_cells} if house is not None else {}),
               **({"start_rule": extras["rule"], "start_n_dets": state.get("n_dets")} if exam else {})})
        if exam and rec_ms is not None:
            rec_ms.append(float(extras["timing"]["rec_ms"]))
        rule_stop_step = None
        while not (success and run.stop_at_success) and rule_stop_step is None and step < run.max_steps:
            t0 = time.perf_counter()
            cmd = policy.act(Observation(frames=list(frames), goal_image=None, step=step, extras=extras))
            policy_ms.append((time.perf_counter() - t0) * 1000.0)
            frame = sim.step(cmd.v, cmd.w)
            step += 1
            frames.append(frame.rgb)
            extras, state, move = frame.extras, frame.state, frame.move
            path_m += move["travelled_m"]
            collisions += int(move["collided"])
            streak = streak + 1 if move["collided"] else 0
            longest = max(longest, streak)
            if not success and reached(state["vp_geodesic_m"], run.success_m):
                success, success_step, path_at_success = True, step, path_m  # oracle stop: the first entry
            line = {"kind": "step", "step": step, "v": move["v"], "w": move["w"], "policy": cmd.info,
                    "position": state["position"], "yaw": state["yaw"], "vp_geodesic_m": state["vp_geodesic_m"],
                    "collided": move["collided"], "travelled_m": move["travelled_m"], "short_m": move["short_m"],
                    "in_view": state["in_view"], "seen_instances": state["seen_instances"]}
            if exam:
                line.update(rule_value=extras["rule"]["value"], rule_cell=extras["rule"]["cell"],
                            n_dets=state.get("n_dets"))
                if run.live_rule_tau is not None and clean(extras["rule"]["value"]) >= run.live_rule_tau - 1e-12:
                    rule_stop_step = step  # the robot stops here (live rule stop, check runs only)
                if house is not None:
                    line["house_seen"] = house.update(state["position"], state["yaw"])
                if rec_ms is not None:
                    rec_ms.append(float(extras["timing"]["rec_ms"]))
            write(line)
        if success and run.stop_at_success:
            path_at_success = path_m
        first_seen = state["first_in_view_step"]
        shortest = start_vp if start_vp is not None else math.inf
        result = {"kind": "result", "episode_id": episode["episode_id"], "arm": arm, "seed_index": seed_index,
                  "seed": seed, "home": episode["home"], "category": episode["target"]["category"],
                  "distance_bin": episode["distance_bin"], "success": bool(success), "success_step": success_step,
                  "steps": step, "spl": metrics.spl(success, shortest, path_at_success), "collisions": collisions,
                  "longest_collision_streak": longest, "path_length_m": path_m,
                  "final_vp_geodesic_m": state["vp_geodesic_m"], "start_vp_geodesic_m": start_vp,
                  "first_in_view_step": first_seen,
                  "search_steps": first_seen if first_seen is not None else step,
                  "approach_steps": (step - first_seen) if first_seen is not None else None,
                  "seen_usable": len(state["seen_instances"])}
        if run.live_rule_tau is not None:
            result.update(live_rule_tau=run.live_rule_tau, rule_stop_step=rule_stop_step,
                          rule_stop_success=rule_stop_step is not None and reached(state["vp_geodesic_m"], run.success_m))
        write(result)
    os.replace(partial, log_path)
    log = steplog.read(log_path)
    result["stuck_streaks"] = len(steplog.stuck_streaks(log, run.stuck_steps))
    if save_dir is not None:
        meta = {"drive_id": episode["episode_id"], "outcome": "success" if success else "timeout", "kept": True,
                "seed": seed, "seed_index": seed_index, "arm": arm, "policy_sha256": (header_extra or {}).get("weights_sha256"),
                "start": {"position": episode["start_position"], "yaw": episode["start_yaw"]},
                "start_geodesic_m": episode["start_geodesic_m"], "camera": camera["name"],
                "steps": step, "success_step": success_step, "first_in_view_step": first_seen,
                "collision_steps": collisions, "longest_collision_streak": longest}
        result["drive_bytes"] = sim.save(str(save_dir), meta)["bytes"]
    return clean(result), policy_ms
