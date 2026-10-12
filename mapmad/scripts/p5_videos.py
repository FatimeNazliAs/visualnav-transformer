"""Phase 5 videos, rendered from the step logs after the runs (block item 14). Inside naz_mapmad_habitat, from
/app/visualnav-transformer:

    python mapmad/scripts/p5_videos.py --config mapmad/configs/p5_exam.yaml --subdir runs \
        --episode <episode id> --seed-index 0 --arms A B C D E F --label "random pick (seeded)" --out videos/<name>.mp4
    python mapmad/scripts/p5_videos.py ... --arms C --out videos/<name>_C.mp4          # one arm, larger tiles

Every arm's logged poses are replayed in Habitat: the camera picture is rendered again at each pose and the robot's
own online map is rebuilt from the depth picture there (mapmad_sim.online_map; byte-identical to what the policy got,
see p5_map_parity.py). One tile per arm (3 per row):
- top left: the LIMO camera (goal photo inset for photo arms); top right: the 64 x 64 map as MapMaD gets it
  (unknown dark grey, explored light grey, obstacle red, heat orange; robot yellow, pointing up);
- bottom left: the floor plan (walkable area of the start's storey, grey) with the target box (red), its usable
  view points (small red dots; success = within 0.2 m walking distance of one), the start (green), the driven path
  (blue), the robot (yellow arrow) and the heat (orange glow around the box: the heat rule without the window);
- bottom right: step, command, walking distance to the nearest view point, collisions;
- caption: arm, policy, SUCCESS (green, with the step) / FAILED (red) stamped when the arm's run ends.
Finished arms hold their last frame. 8 fps = 2x real time; the last frame is held 2 s.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))

from mapmad_bridge import steplog  # noqa: E402
from mapmad_sim import config  # noqa: E402
from mapmad_sim.depth_map import MapSpec  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.online_map import OnlineMap  # noqa: E402
from mapmad_sim.robot import LimoSim, RobotSpec  # noqa: E402
from mapmad_sim.exam_arms import arms_of  # noqa: E402
from mapmad_sim.run_layout import RunLayout, read_config, run_id  # noqa: E402
from vint_train.mapmad import map_viz  # noqa: E402
from vint_train.mapmad.closed_loop import video  # noqa: E402
from vint_train.mapmad.config import load_config  # noqa: E402

ARM_NAMES = {"A": "NoMaD explore", "B": "NoMaD + goal photo", "C": "MapMaD map goal", "D": "MapMaD map masked",
             "E": "map planner (A*)", "F": "MapMaD + photo, map masked", "C_wp1": "MapMaD map goal (index 1)",
             "C_wp2": "MapMaD map goal (index 2)"}
CAM = (320, 240)
LOCAL_PX = 240
PLAN_PX = 280
BAR = 40
HEAT_RGB = np.array([255, 140, 0], np.float32)


def heat_overlay(img: np.ndarray, heat: np.ndarray, strength: float = 0.75) -> np.ndarray:
    """Blend heat (H, W in 0-1) into img in orange."""
    a = (np.clip(heat, 0, 1) * strength)[..., None]
    return (img.astype(np.float32) * (1 - a) + HEAT_RGB * a).astype(np.uint8)


class FloorPlan:
    """The episode's storey from above, with the heat glow around the target box (pixels of video.TopdownMap)."""

    def __init__(self, robot: LimoSim, episode: Dict[str, Any], paths: List[List[List[float]]], sigma: float) -> None:
        grid, origin = robot.topdown(0.05, float(episode["start_position"][1]))
        keep = [episode["start_position"], episode["target"]["box_center"]] + [p for path in paths for p in path]
        self.t = video.TopdownMap(grid, origin, 0.05, keep)
        x, z = self.t.centres()  # same pixel convention as the box, path and robot (TopdownMap.px)
        c, s = episode["target"]["box_center"], episode["target"]["box_size"]
        d = np.hypot(np.maximum(np.abs(x - c[0]) - s[0] / 2, 0), np.maximum(np.abs(z - c[2]) - s[2] / 2, 0))
        self.base = heat_overlay(self.t.base, np.exp(-d ** 2 / (2 * sigma ** 2)), 0.6)
        self.episode = episode

    def draw(self, path: List[List[float]], yaw: float) -> np.ndarray:
        img = self.base.copy()
        e, t = self.episode, self.t
        for vp in e["target"]["view_points"]:
            cv2.circle(img, t.px(vp), 2, video.RED, -1)
        c, s = e["target"]["box_center"], e["target"]["box_size"]
        cv2.rectangle(img, t.px([c[0] - s[0] / 2, 0, c[2] - s[2] / 2]), t.px([c[0] + s[0] / 2, 0, c[2] + s[2] / 2]),
                      video.RED, 2)
        cv2.circle(img, t.px(e["start_position"]), 7, video.GREEN, -1, cv2.LINE_AA)
        if len(path) > 1:
            cv2.polylines(img, [np.array([t.px(p) for p in path], np.int32)], False, video.BLUE, 2, cv2.LINE_AA)
        here = np.array(t.px(path[-1]))
        tip = here + (np.array([-math.sin(yaw), -math.cos(yaw)]) * 16).astype(int)
        cv2.arrowedLine(img, tuple(int(v) for v in here), tuple(int(v) for v in tip), video.YELLOW, 3, cv2.LINE_AA,
                        tipLength=0.5)
        return cv2.resize(img, (PLAN_PX, PLAN_PX), interpolation=cv2.INTER_AREA)


def local_picture(tensor: np.ndarray, given: bool) -> np.ndarray:
    """The 64 x 64 map with heat; labelled with whether this arm's policy received it."""
    img = heat_overlay(map_viz.render(tensor[:2], LOCAL_PX), cv2.resize(tensor[2], (LOCAL_PX, LOCAL_PX),
                                                                         interpolation=cv2.INTER_NEAREST))
    map_viz.draw_robot(img, tensor.shape[-1], hfov_deg=66.5)
    video.text(img, "map + heat: MapMaD's input" if given else "robot's map (not given to this arm)",
               (4, LOCAL_PX - 8), 0.4)
    return img


def arm_frames(robot: LimoSim, episode: Dict[str, Any], log: List[Dict[str, Any]], plan: FloorPlan, map_spec: MapSpec,
               crop: Dict[str, Any], goal_rgb, map_given: bool) -> List[Dict[str, Any]]:
    """Per logged pose: camera picture, local map picture, floor plan, numbers."""
    head, steps, res = steplog.header(log), steplog.steps(log), steplog.result(log)
    poses = steplog.poses(log)
    om = None
    out = []
    for i, (pos, yaw) in enumerate(poses):
        robot.place(np.array(pos), yaw)
        if om is None:
            om = OnlineMap(robot, episode, map_spec, crop)
        obs = robot.observe()
        om.add_frame(robot, obs["depth"])
        s = steps[i - 1] if i else None
        out.append({"rgb": obs["rgb"], "local": local_picture(om.tensor(robot), map_given),
                    "plan": plan.draw([p for p, _ in poses[:i + 1]], yaw), "step": i,
                    "vp": s["vp_geodesic_m"] if s else head["start_vp_geodesic_m"],
                    "v": s["v"] if s else 0.0, "w": s["w"] if s else 0.0,
                    "collisions": sum(x["collided"] for x in steps[:i])})
    return out, res, goal_rgb


def tile(arm: str, policy: str, f: Dict[str, Any], res: Dict[str, Any], last: bool, max_steps: int, goal_rgb) -> np.ndarray:
    cam = f["rgb"].copy()
    if goal_rgb is not None:
        inset = cv2.resize(goal_rgb, (96, 72), interpolation=cv2.INTER_AREA)
        cam[4:76, 4:100] = inset
        cv2.rectangle(cam, (4, 4), (100, 76), video.WHITE, 1)
        video.text(cam, "goal photo", (6, 90), 0.4)
    info = np.full((PLAN_PX, CAM[0] + LOCAL_PX - PLAN_PX, 3), 25, np.uint8)
    vp = "-" if f["vp"] is None else f"{f['vp']:.2f} m"
    for k, s in enumerate([f"step {f['step']} / {max_steps}", f"v {f['v']:.2f} m/s  w {f['w']:+.2f} rad/s",
                           f"to nearest view point: {vp}", f"collisions so far: {f['collisions']}"]):
        video.text(info, s, (10, 30 + 30 * k), 0.5)
    body = np.vstack([np.hstack([cam, f["local"]]), np.hstack([f["plan"], info])])
    bar = np.zeros((BAR, body.shape[1], 3), np.uint8)
    video.text(bar, f"{arm}: {ARM_NAMES.get(arm, policy)}", (8, 24), 0.6)
    if last:
        ok = res["success"]
        video.text(bar, f"SUCCESS @{res['success_step']}" if ok else "FAILED", (body.shape[1] - 170, 25), 0.65,
                   video.GREEN if ok else video.RED)
    return np.vstack([bar, body])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="runs")
    p.add_argument("--episode", required=True)
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--label", default="", help="caption line, e.g. 'random pick (seeded)' or 'hand-picked: stuck'")
    p.add_argument("--out", required=True, help="below <outputs>/<out_dir>")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cols", type=int, default=3)
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    episode = {e["episode_id"]: e for e in load_episodes(layout.episodes_file)[0]}[args.episode]
    map_spec = MapSpec.from_config(read_config(config.CONFIG_DIR / layout.cfg["map_settings"]["builder"])["map"])
    crop = load_config(str(layout.resolve(layout.cfg["map_settings"]["crop"])))["map"]
    max_steps = int(layout.cfg["run"]["max_steps"])
    logs = {}
    arms = {a.name: a for a in arms_of(layout.cfg, args.arms)}
    for name, arm in arms.items():
        logs[name] = steplog.read(layout.log(name, run_id(args.episode, arm.run_seed_index(args.seed_index))))
    tiles: Dict[str, Any] = {}
    with LimoSim(episode["split"], episode["home"], RobotSpec.from_config(config.robot()), depth=True,
                 gpu=args.gpu) as robot:
        plan = FloorPlan(robot, episode, [[p for p, _ in steplog.poses(l)] for l in logs.values()], crop["heat_sigma"])
        robot.place(np.array(episode["goal_photo_position"]), episode["goal_photo_yaw"])
        photo = robot.observe()["rgb"]
        for name, log in logs.items():
            g = photo if arms[name].photo else None
            tiles[name] = arm_frames(robot, episode, log, plan, map_spec, crop, g, arms[name].map_given)
    n = max(len(v[0]) for v in tiles.values())
    out = layout.base / args.out
    writer = None
    head = (f"{episode['episode_id']} | home {episode['home'][:5]} | find: {episode['target']['category']} | start "
            f"{episode['start_geodesic_m']:.1f} m from the nearest view point | seed index {args.seed_index}"
            + (f" | {args.label}" if args.label else ""))
    for i in range(n + 16):
        row_tiles = []
        for arm in args.arms:
            frames, res, g = tiles[arm]
            j = min(i, len(frames) - 1)
            row_tiles.append(tile(arm, logs[arm][0]["policy"], frames[j], res, i >= len(frames) - 1, max_steps, g))
        while len(row_tiles) % args.cols and len(row_tiles) > args.cols:
            row_tiles.append(np.zeros_like(row_tiles[0]))
        cols = min(args.cols, len(row_tiles))
        grid = np.vstack([np.hstack(row_tiles[r:r + cols]) for r in range(0, len(row_tiles), cols)])
        top = np.zeros((BAR, grid.shape[1], 3), np.uint8)
        video.text(top, head, (8, 24), 0.55)
        frame = np.vstack([top, grid])
        if writer is None:
            writer = video.VideoWriter(out, (frame.shape[1], frame.shape[0]), 8)
        writer.write(frame)
    writer.close()
    print(f"wrote {out} ({n} steps, {len(args.arms)} arms)")


if __name__ == "__main__":
    main()
