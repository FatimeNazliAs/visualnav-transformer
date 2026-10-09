"""Run one episode: simulator (via the bridge) + policy, until success (oracle stop) or the step limit.

Every step goes into a JSONL log (first line = episode header, then one line per step, last line = result).
The log holds no times or host names, so two runs with the same seed must give byte-identical logs (gate G1
row 1). Videos are drawn after the episode from the kept frames.
"""

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from mapmad_sim.episodes import floor_map_ok
from vint_train.mapmad.closed_loop import metrics, video
from vint_train.mapmad.closed_loop.policy import Observation, Policy

DIGITS = 6


def clean(value: Any) -> Any:
    """JSON-ready copy with floats rounded to DIGITS (fixed text for identical numbers) and inf -> None."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(value) else round(float(value), DIGITS)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@dataclass
class Arm:
    """One condition of a comparison (configs/p1_baseline.yaml `arms`)."""

    name: str
    goal: str  # "photo" or "masked"
    episodes: str  # episode type the arm runs on
    hfov_deg: float
    policy: str = "nomad"


@dataclass
class RunSettings:
    success_m: float
    max_steps: int
    topdown_m_per_px: float
    video_fps: float


@dataclass
class EpisodeRecord:
    """What the video needs; filled while the episode runs."""

    frames: List[np.ndarray] = field(default_factory=list)
    positions: List[List[float]] = field(default_factory=list)
    yaws: List[float] = field(default_factory=list)
    geodesics: List[Optional[float]] = field(default_factory=list)
    goal_rgb: Optional[np.ndarray] = None
    topdown: Optional[Dict[str, Any]] = None
    policy_ms: List[float] = field(default_factory=list)  # wall-clock of each act() (kept out of the log)


class FloorMapMismatch(RuntimeError):
    """The episode was built on another floor map and was not re-checked on this one."""


def run_episode(sim, policy: Policy, episode: Dict[str, Any], arm: Arm, run: RunSettings, seed: int,
                log_path: Path, fingerprint: str, checked_floor_maps: Optional[Dict[str, str]] = None
                ) -> (Dict[str, Any], EpisodeRecord):
    """Drive one episode; writes log_path and returns (result summary, record for the video).
    The policy sees only pictures and the goal photo; geodesic distance and target stay here.
    Refuses (FloorMapMismatch, before any log is written) if the live floor map differs from the one the episode was
    built on, unless checked_floor_maps (episodes.checked_floor_maps) says the episode passed on the live one."""
    first = sim.reset(episode, arm.hfov_deg, goal_photo=arm.goal == "photo", topdown_m_per_px=run.topdown_m_per_px)
    if not floor_map_ok(episode, first.navmesh_sha256, checked_floor_maps or {}):
        raise FloorMapMismatch(
            f"{episode['episode_id']}: live floor map {str(first.navmesh_sha256)[:12]} != stored "
            f"{str(episode.get('navmesh_sha256'))[:12]} (robot_limo.yaml navmesh changed?); run "
            "mapmad/scripts/check_p1_episodes.py on this episode file, or rebuild the episodes")
    goal = first.goal_rgb if arm.goal == "photo" else None
    policy.reset({"seed": seed, "episode_id": episode["episode_id"], "goal": arm.goal})
    frames = deque([first.rgb], maxlen=policy.context_frames)
    rec = EpisodeRecord(frames=[first.rgb], positions=[first.state["position"]], yaws=[first.state["yaw"]],
                        geodesics=[first.state["geodesic_m"]], goal_rgb=goal, topdown=first.topdown)
    start_geo = first.state["geodesic_m"]
    geo = start_geo if start_geo is not None else math.inf
    success = geo <= run.success_m
    path_m, collisions, step = 0.0, 0, 0
    success_step = 0 if success else None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        def write(line: Dict[str, Any]) -> None:
            log.write(json.dumps(clean(line), sort_keys=True) + "\n")

        write({"kind": "episode", "episode_id": episode["episode_id"], "arm": arm.name, "seed": seed,
               "home": episode["home"], "type": episode["type"], "hfov_deg": arm.hfov_deg, "goal": arm.goal,
               "policy": policy.name, "episodes_fingerprint": fingerprint, "start_geodesic_m": start_geo,
               "start": first.state, "navmesh_sha256": first.navmesh_sha256, "max_steps": run.max_steps,
               "success_m": run.success_m})
        while not success and step < run.max_steps:
            t0 = time.perf_counter()
            cmd = policy.act(Observation(frames=list(frames), goal_image=goal, step=step))
            rec.policy_ms.append((time.perf_counter() - t0) * 1000.0)
            frame = sim.step(cmd.v, cmd.w)
            step += 1
            frames.append(frame.rgb)
            move = frame.move
            path_m += move["travelled_m"]
            collisions += int(move["collided"])
            geo = frame.state["geodesic_m"] if frame.state["geodesic_m"] is not None else math.inf
            success = geo <= run.success_m
            if success:
                success_step = step
            write({"kind": "step", "step": step, "v": move["v"], "w": move["w"], "policy": cmd.info,
                   "position": frame.state["position"], "yaw": frame.state["yaw"], "geodesic_m": geo,
                   "collided": move["collided"], "travelled_m": move["travelled_m"], "short_m": move["short_m"],
                   "refused_substeps": move["refused_substeps"], "floor_below_feet_m": frame.state["floor_below_feet_m"]})
            rec.frames.append(frame.rgb)
            rec.positions.append(frame.state["position"])
            rec.yaws.append(frame.state["yaw"])
            rec.geodesics.append(frame.state["geodesic_m"])
        shortest = start_geo if start_geo is not None else math.inf  # SPL: geodesic start -> target point
        result = {"kind": "result", "episode_id": episode["episode_id"], "arm": arm.name, "seed": seed,
                  "home": episode["home"], "category": episode["target"]["category"], "type": episode["type"],
                  "success": bool(success), "success_step": success_step, "steps": step,
                  "spl": metrics.spl(success, shortest, path_m), "collisions": collisions,
                  "collision_share": collisions / step if step else 0.0, "path_length_m": path_m, "final_geodesic_m": geo,
                  "start_geodesic_m": start_geo}
        write(result)
    return clean(result), rec


def write_video(path: Path, episode: Dict[str, Any], arm: Arm, result: Dict[str, Any], rec: EpisodeRecord,
                run: RunSettings) -> None:
    """Draw the kept frames into an mp4 (2x real time at video_fps 8); the last frame is held 1 s."""
    td = rec.topdown
    tmap = video.TopdownMap(td["grid"], td["origin"], td["m_per_px"],
                            [episode["start_position"], episode["target"]["point"]] + rec.positions)
    writer = None
    n = len(rec.frames)
    for i, rgb in enumerate(rec.frames):
        geo = rec.geodesics[i]
        last = i == n - 1
        if last:
            status, colour = ("SUCCESS", video.GREEN) if result["success"] else ("FAILED", video.RED)
        else:
            status, colour = f"step {i}", video.WHITE
        caption = (f"{arm.name} | {episode['episode_id']} | {episode['target']['category']} | step {i}/{run.max_steps}"
                   f" | geo {geo:.2f} m" if geo is not None else f"{arm.name} | {episode['episode_id']} | geo -")
        frame = video.compose(rgb, video.draw_map(tmap, episode, rec.positions[:i + 1], rec.yaws[i], run.success_m),
                              caption, status, colour, rec.goal_rgb)
        if writer is None:
            writer = video.VideoWriter(path, (frame.shape[1], frame.shape[0]), run.video_fps)
        for _ in range(int(run.video_fps) if last else 1):
            writer.write(frame)
    writer.close()
