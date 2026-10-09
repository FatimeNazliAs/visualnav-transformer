"""Phase 1 diagnostics from the existing step logs (no policy runs): the robot is only put back on logged poses.

    python mapmad/scripts/p1_replay.py frames --arm photo_iv --episode p1-iv-00081-000 --step 118
    python mapmad/scripts/p1_replay.py anyside                    # every log -> <out>/anyside/<arm>/<episode>.json
    python mapmad/scripts/p1_replay.py black                      # every log -> <out>/black/<arm>/<episode>.json

frames: the 4 camera pictures NoMaD was given for the command logged at `step` (the pictures after steps
step-4 .. step-1; step 0 = start pose), re-rendered at the logged poses with the arm's camera, saved as lossless
PNG (320x240) in <out>/diagnostics/<arm>__<episode>__<step>/frame_<k>.png (+ goal.png for photo arms). The camera
height comes from the floor probe at that pose, exactly as during the run.
anyside (SECONDARY metric): geodesic distance from each logged position to the nearest ObjectNav view point of the
target instance (view points snapped onto the LIMO floor map, kept if within 0.25 m); any-side success if
<= success_m (1.0 m) at any step, view-point success if <= CLOSE_VIEW_POINT_M (0.2 m) at any step. Both are lower
bounds: an episode stops at its primary success (geodesic to the single target point <= 1.0 m, unchanged).
black: for every stuck streak (>= STUCK_STEPS consecutive collision steps), the share of near-black pixels
(max channel < 16: HM3D mesh holes) in the 4 pictures NoMaD was given for the streak's first command.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from mapmad_bridge import steplog
from mapmad_sim import config, objectnav
from mapmad_sim.episodes import load_episodes
from mapmad_sim.robot import LimoSim, RobotSpec, horizontal_distance
from mapmad_sim.run_layout import P1_CONFIG, RunLayout

SNAP_MAX_M = 0.25
CLOSE_VIEW_POINT_M = 0.2
STUCK_STEPS = 100  # = vint_train.mapmad.closed_loop.analysis.STUCK_STEPS
BLACK_LEVEL = 16  # = episodes.EpisodeBuilder.black_share


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def frames(args: argparse.Namespace, layout: RunLayout) -> None:
    log = steplog.read(layout.log(args.arm, args.episode))
    episode = {e["episode_id"]: e for e in load_episodes(layout.episodes_file)[0]}[args.episode]
    arm = layout.cfg["arms"][args.arm]
    target = layout.diagnostics(args.arm, args.episode, args.step)
    target.mkdir(parents=True, exist_ok=True)
    pose = steplog.poses(log)
    with LimoSim(episode["split"], episode["home"], RobotSpec.from_config(config.robot(), hfov_deg=arm["hfov_deg"]),
                 gpu=args.gpu) as robot:
        for k, i in enumerate(range(args.step - 4, args.step)):
            robot.place(*pose[i])
            save_rgb(target / f"frame_{k}.png", robot.observe()["rgb"])
        if arm["goal"] == "photo":
            robot.place(episode["goal_photo_position"], episode["goal_photo_yaw"])
            save_rgb(target / "goal.png", robot.observe()["rgb"])
    print(f"wrote {target}")


def view_point_ends(robot: LimoSim, goals: List[objectnav.Goal], object_id: int) -> List[np.ndarray]:
    """The target's ObjectNav view points snapped onto the LIMO floor map (kept if within SNAP_MAX_M)."""
    ends = []
    for vp in (v for g in goals if g.object_id == object_id for v in g.view_points):
        s = np.array(robot.pathfinder.snap_point(np.asarray(vp, np.float32)))
        if np.isfinite(s).all() and horizontal_distance(s, vp) <= SNAP_MAX_M:
            ends.append(s.astype(np.float32))
    return ends


def nearest_view_point_m(robot: LimoSim, position: List[float], ends: List[np.ndarray]) -> float:
    import habitat_sim

    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.asarray(position, np.float32)
    path.requested_ends = ends
    return float(path.geodesic_distance) if robot.pathfinder.find_path(path) else math.inf


def anyside(args: argparse.Namespace, layout: RunLayout) -> None:
    cfg = layout.cfg
    episodes = {e["episode_id"]: e for e in load_episodes(layout.episodes_file)[0]}
    by_home = defaultdict(list)
    for path in layout.logs():
        by_home[episodes[path.stem]["home"]].append(path)
    success_m = cfg["run"]["success_m"]
    for home, paths in sorted(by_home.items()):
        goals = objectnav.home_goals(cfg["episodes"]["objectnav_split"], home)
        with LimoSim(cfg["episodes"]["split"], home, RobotSpec.from_config(config.robot()), gpu=args.gpu) as robot:
            for path in paths:
                e = episodes[path.stem]
                ends = view_point_ends(robot, goals, e["target"]["object_id"])
                log = steplog.read(path)
                cache: Dict[tuple, float] = {}
                first = {success_m: None, CLOSE_VIEW_POINT_M: None}
                nearest = math.inf
                for step, (position, _) in enumerate(steplog.poses(log)):
                    key = tuple(np.round(position, 3))
                    if key not in cache:
                        cache[key] = nearest_view_point_m(robot, position, ends) if ends else math.inf
                    nearest = min(nearest, cache[key])
                    for limit in first:
                        if first[limit] is None and cache[key] <= limit:
                            first[limit] = step
                arm = steplog.header(log)["arm"]
                row = {"episode_id": e["episode_id"], "arm": arm, "view_points": len(ends),
                       "anyside_success": first[success_m] is not None, "anyside_step": first[success_m],
                       "view_point_02_success": first[CLOSE_VIEW_POINT_M] is not None,
                       "view_point_02_step": first[CLOSE_VIEW_POINT_M],
                       "nearest_view_point_m": nearest if math.isfinite(nearest) else None,
                       "primary_success": steplog.result(log)["success"]}
                dest = layout.anyside(arm, e["episode_id"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(row, sort_keys=True) + "\n")
        print(f"{home}: {len(paths)} logs", flush=True)


def black(args: argparse.Namespace, layout: RunLayout) -> None:
    """Near-black share of NoMaD's input at the first command of every stuck streak."""
    episodes = {e["episode_id"]: e for e in load_episodes(layout.episodes_file)[0]}
    groups = defaultdict(list)  # (home, hfov) -> logs: one simulator per home and camera
    for path in layout.logs():
        arm = path.parent.name
        groups[(episodes[path.stem]["home"], layout.cfg["arms"][arm]["hfov_deg"])].append(path)
    for (home, hfov), paths in sorted(groups.items()):
        with LimoSim(layout.cfg["episodes"]["split"], home, RobotSpec.from_config(config.robot(), hfov_deg=hfov),
                     gpu=args.gpu) as robot:
            for path in paths:
                log = steplog.read(path)
                pose = steplog.poses(log)
                streaks = []
                for start, length in steplog.stuck_streaks(log, STUCK_STEPS):
                    pictures = []
                    for i in range(max(start - 4, 0), start):  # the pictures after steps start-4 .. start-1
                        robot.place(*pose[i])
                        pictures.append(robot.observe()["rgb"])
                    share = float(np.mean([(p.max(axis=2) < BLACK_LEVEL).mean() for p in pictures]))
                    streaks.append({"start_step": start, "length": length, "input_black_share": round(share, 4)})
                arm = steplog.header(log)["arm"]
                dest = layout.root / "black" / arm / f"{path.stem}.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps({"episode_id": path.stem, "arm": arm, "streaks": streaks}, sort_keys=True) + "\n")
        print(f"{home} hfov {hfov:g}: {len(paths)} logs", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["frames", "anyside", "black"])
    p.add_argument("--config", type=Path, default=P1_CONFIG)
    p.add_argument("--arm")
    p.add_argument("--episode")
    p.add_argument("--step", type=int)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    {"frames": frames, "anyside": anyside, "black": black}[args.what](args, layout)


if __name__ == "__main__":
    main()
