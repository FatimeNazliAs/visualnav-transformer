"""Phase 7 video: the same episode driven by the official NoMaD with every camera, side by side (p7_rules.md §10).
Inside naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/p7_videos.py --config mapmad/configs/p7_sweep.yaml --subdir sweep/runs --pick 7710
    python mapmad/scripts/p7_videos.py ... --episode <episode id> --seed-index 0

Episode: given, or a seeded random in-view episode (--pick seed). Every B_<camera> arm's logged poses are replayed
with that arm's camera: one tile per camera (4 per row): the camera picture as NoMaD gets it before the 96 x 96
resize (goal photo of that camera inset), the floor plan (target box red, view points red dots, start green, path
blue, robot yellow), and a caption (camera, SUCCESS @step / FAILED, collisions). Finished arms hold their last frame;
8 fps = 2x real time. Writes <out_dir>/sweep/videos/<episode>.mp4 (+ a small copy for upload).
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapmad_bridge import steplog  # noqa: E402
from mapmad_sim import config  # noqa: E402
from mapmad_sim.camera_setting import cameras_of  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.robot import LimoSim, RobotSpec  # noqa: E402
from mapmad_sim.run_layout import RunLayout, run_id  # noqa: E402
from p5_videos import FloorPlan  # noqa: E402
from vint_train.mapmad.closed_loop import video  # noqa: E402

CAM = (320, 240)
PLAN = 240
BAR = 40  # tile height 280 (multiple of 8 for H.264)


def replay(cam_name: str, cam, episode: Dict[str, Any], log: List[Dict[str, Any]], sigma: float) -> Dict[str, Any]:
    """Frames (camera picture + plan per pose), goal photo, result of one arm's run."""
    poses = steplog.poses(log)
    with LimoSim(episode["split"], episode["home"], cam.apply(RobotSpec.from_config(config.robot()))) as robot:
        robot.place(episode["goal_photo_position"], episode["goal_photo_yaw"])
        goal = robot.observe()["rgb"]
        plan = FloorPlan(robot, episode, [[p for p, _ in poses]], sigma)
        frames = []
        for i, (pos, yaw) in enumerate(poses):
            robot.place(np.array(pos), yaw)
            frames.append({"rgb": robot.observe()["rgb"], "plan": plan.draw([p for p, _ in poses[:i + 1]], yaw, PLAN)})
    steps = steplog.steps(log)
    return {"frames": frames, "goal": goal, "res": steplog.result(log), "coll": np.cumsum([0] + [s["collided"] for s in steps])}


def tile(name: str, run: Dict[str, Any], i: int) -> np.ndarray:
    k = min(i, len(run["frames"]) - 1)
    f = run["frames"][k]
    cam = cv2.resize(f["rgb"], CAM, interpolation=cv2.INTER_LINEAR)
    cam[4:58, 4:76] = cv2.resize(run["goal"], (72, 54), interpolation=cv2.INTER_AREA)
    cv2.rectangle(cam, (4, 4), (76, 58), video.WHITE, 1)
    body = np.hstack([cam, cv2.resize(f["plan"], (CAM[1], CAM[1]))])
    bar = np.zeros((BAR, body.shape[1], 3), np.uint8)
    video.text(bar, f"{name}  step {k}  coll {int(run['coll'][k])}", (6, 22), 0.55)
    if k == len(run["frames"]) - 1:
        ok = run["res"]["success"]
        video.text(bar, f"SUCCESS @{run['res']['success_step']}" if ok else "FAILED", (body.shape[1] - 150, 22), 0.55,
                   video.GREEN if ok else video.RED)
    return np.vstack([bar, body])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="sweep/runs")
    p.add_argument("--episode", default=None)
    p.add_argument("--pick", type=int, default=7710, help="seed of the random in-view episode (without --episode)")
    p.add_argument("--seed-index", type=int, default=0)
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    episodes, _ = load_episodes(layout.episodes_file)
    if args.episode is None:
        iv = [e for e in episodes if e["type"] == "in_view"]
        args.episode = iv[int(np.random.default_rng(args.pick).integers(len(iv)))]["episode_id"]
    episode = next(e for e in episodes if e["episode_id"] == args.episode)
    rid = run_id(args.episode, args.seed_index)
    runs = {}
    for name, cam in cameras_of(layout.cfg).items():
        log = layout.log(f"B_{name}", rid)
        if log.exists():
            runs[name] = replay(name, cam, episode, steplog.read(log), sigma=0.3)
    n = max(len(r["frames"]) for r in runs.values())
    out = layout.base / "sweep" / "videos" / f"{args.episode}__s{args.seed_index}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    names = list(runs)
    for i in range(n + 16):  # hold the end 2 s
        tiles = [tile(nm, runs[nm], i) for nm in names]
        rows = [np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]
        width = max(r.shape[1] for r in rows)
        frame = np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows])
        if writer is None:
            writer = video.VideoWriter(out, (frame.shape[1], frame.shape[0]), 8)
        writer.write(frame)
    writer.close()
    print(out)


if __name__ == "__main__":
    main()
