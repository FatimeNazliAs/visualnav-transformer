"""Phase 5 focused video: ONE driver of one episode, rendered from its step log (no new run). Inside
naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/p5_focused_video.py --config mapmad/configs/p5_exam.yaml --subdir runs \
        --episode p5e-00810-bed-0 --arm C --label "video 11: C success, B fails" \
        --out videos/focused/p5e-00810-bed-0_C.mp4

The logged poses are replayed in Habitat as in p5_videos.py (camera picture rendered again; the online map rebuilt
from the depth picture, byte-identical to the policy's input, see p5_map_parity.py). Frame i = the pose after step
i. Layout, 1280 px wide:
- top bar: episode, target, driver, label; SUCCESS / FAILED at the end;
- camera (320 x 240 render shown at 2x; goal photo inset for photo arms) | MapMaD's 64 x 64 map input at 7.5x
  (unknown dark grey, explored light grey, obstacle red, heat orange; robot yellow in the centre, forward = up) with
  the policy's plan made at this pose: the 8 waypoints of the used sample (cyan) and the chosen one (magenta ring);
  the planner logs only its steer point (magenta ring) | legend;
- whole-house overview (p5_videos.FloorPlan: target box, view points, heat glow, start, path so far, robot) | text
  panel: step, walking distance to the nearest view point, collisions so far, and STUCK (red) on every frame inside
  a >= stuck_steps collision streak.
8 fps = 2x real time (4 Hz control); H.264 CRF 28, +faststart; the last frame is held 2 s.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import p5_videos as P  # same folder; puts the repository's train/ on sys.path
from mapmad_bridge import steplog
from mapmad_sim import config
from mapmad_sim.depth_map import MapSpec
from mapmad_sim.episodes import load_episodes
from mapmad_sim.exam_arms import arms_of
from mapmad_sim.online_map import OnlineMap
from mapmad_sim.robot import LimoSim, RobotSpec
from mapmad_sim.run_layout import RunLayout, read_config, run_id
from vint_train.mapmad import map_viz
from vint_train.mapmad.closed_loop import video
from vint_train.mapmad.config import load_config

WIDTH = 1280
TOP = 48
CAM_W, CAM_H = 640, 480
MAP_PX = 480
LEGEND_W = WIDTH - CAM_W - MAP_PX
LOW_H = 400
FPS, HOLD_S, CRF = 8, 2, 28
WAYPOINTS, CHOSEN = (0, 230, 255), (255, 0, 200)


def plan_points(policy_info: Optional[Dict[str, Any]]) -> Tuple[List[Sequence[float]], Optional[Sequence[float]]]:
    """(the 8 planned waypoints, the chosen point) in robot-frame metres (forward, left) from one step's policy
    info: NoMaD / MapMaD log `path_m` + `waypoint_m`, the planner logs its steer point `aim_m`; ([], None) if none."""
    info = policy_info or {}
    if "path_m" in info:
        return info["path_m"], info.get("waypoint_m")
    return [], info.get("aim_m")


def in_streak(step: int, streaks: Sequence[Tuple[int, int]]) -> bool:
    """Frame `step` (the pose after that step) lies inside a stuck streak (first step number, length)."""
    return any(first <= step < first + n for first, n in streaks)


def map_picture(tensor: np.ndarray, given: bool, points: List[Sequence[float]], chosen: Optional[Sequence[float]]) -> np.ndarray:
    """The 64 x 64 map with heat, the robot and the policy's plan, MAP_PX square."""
    img = P.heat_overlay(map_viz.render(tensor[:2], MAP_PX),
                         cv2.resize(tensor[2], (MAP_PX, MAP_PX), interpolation=cv2.INTER_NEAREST))
    size = tensor.shape[-1]
    map_viz.draw_robot(img, size)  # under the plan: NoMaD's short waypoints start inside the triangle
    px = [tuple(int(round(v)) for v in map_viz.local_px(f, l, size, 0.1, MAP_PX)) for f, l in points]
    if px:
        cv2.polylines(img, [np.array(px, np.int32)], False, WAYPOINTS, 2, cv2.LINE_AA)
        for q in px:
            cv2.circle(img, q, 4, WAYPOINTS, -1, cv2.LINE_AA)
    if chosen is not None:
        q = tuple(int(round(v)) for v in map_viz.local_px(chosen[0], chosen[1], size, 0.1, MAP_PX))
        cv2.circle(img, q, 8, CHOSEN, 3, cv2.LINE_AA)
    video.text(img, "MapMaD's map input (6.4 m x 6.4 m)" if given else "robot's map (NOT given to this driver)",
               (8, 22), 0.55)
    return img


def legend(planner: bool) -> np.ndarray:
    img = np.full((CAM_H, LEGEND_W, 3), 25, np.uint8)
    rows = [(map_viz.UNKNOWN, "unknown"), (map_viz.EXPLORED, "explored"), (map_viz.OBSTACLE, "obstacle"),
            (tuple(int(v) for v in P.HEAT_RGB), "heat (target)"), (map_viz.ROBOT, "robot")]
    rows += [(CHOSEN, "steer point")] if planner else [(WAYPOINTS, "8 waypoints"), (CHOSEN, "chosen")]
    video.text(img, "map", (10, 28), 0.6)
    for k, (c, s) in enumerate(rows):
        y = 60 + 34 * k
        cv2.rectangle(img, (10, y - 14), (28, y + 4), c, -1)
        video.text(img, s, (36, y), 0.45)
    return img


def letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """img scaled to fit w x h (aspect kept), centred on dark grey."""
    s = min(w / img.shape[1], h / img.shape[0])
    small = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 25, np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def text_panel(step: int, max_steps: int, vp: Optional[float], collisions: int, stuck: bool) -> np.ndarray:
    img = np.full((LOW_H, WIDTH - CAM_W, 3), 25, np.uint8)
    lines = [f"step {step} / {max_steps}",
             "walking distance to target: " + ("-" if vp is None else f"{vp:.2f} m"),
             "  (nearest view point; success <= 0.2 m)", f"collisions so far: {collisions}"]
    for k, s in enumerate(lines):
        video.text(img, s, (24, 60 + 52 * k), 0.8 if k != 2 else 0.55)
    if stuck:
        cv2.rectangle(img, (24, 290), (250, 360), video.RED, -1)
        video.text(img, "STUCK", (44, 344), 1.6)
    return img


def frame(top: np.ndarray, cam: np.ndarray, local: np.ndarray, leg: np.ndarray, plan: np.ndarray,
          panel: np.ndarray) -> np.ndarray:
    return np.vstack([top, np.hstack([cam, local, leg]), np.hstack([letterbox(plan, CAM_W, LOW_H), panel])])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="runs")
    p.add_argument("--episode", required=True)
    p.add_argument("--arm", required=True, help="the config arm whose log is rendered (e.g. C, or C_wp1 on dev)")
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True, help="below <outputs>/<out_dir>")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    episode = {e["episode_id"]: e for e in load_episodes(layout.episodes_file)[0]}[args.episode]
    map_spec = MapSpec.from_config(read_config(config.CONFIG_DIR / layout.cfg["map_settings"]["builder"])["map"])
    crop = load_config(str(layout.resolve(layout.cfg["map_settings"]["crop"])))["map"]
    max_steps, stuck_steps = int(layout.cfg["run"]["max_steps"]), int(layout.cfg["run"]["stuck_steps"])
    arm = arms_of(layout.cfg, [args.arm])[0]
    log = steplog.read(layout.log(arm.name, run_id(args.episode, arm.run_seed_index(args.seed_index))))
    head, steps, res, poses = steplog.header(log), steplog.steps(log), steplog.result(log), steplog.poses(log)
    streaks = steplog.stuck_streaks(log, stuck_steps)
    out = layout.base / args.out
    title = (f"{episode['episode_id']} | find: {episode['target']['category']} | start "
             f"{episode['start_geodesic_m']:.1f} m | {arm.name}: {P.ARM_NAMES.get(arm.name, arm.policy)}"
             + (f" | {args.label}" if args.label else ""))
    width = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)[0][0]
    title_scale = min(0.55, (WIDTH - 210) / width)  # leave room for SUCCESS / FAILED
    leg = legend(arm.policy == "planner")
    writer = None
    with LimoSim(episode["split"], episode["home"], RobotSpec.from_config(config.robot()), depth=True,
                 gpu=args.gpu) as robot:
        plan = P.FloorPlan(robot, episode, [[q for q, _ in poses]], crop["heat_sigma"])
        goal = None
        if arm.photo:
            robot.place(np.array(episode["goal_photo_position"]), episode["goal_photo_yaw"])
            goal = cv2.resize(robot.observe()["rgb"], video.GOAL_INSET, interpolation=cv2.INTER_AREA)
        om = None
        last = len(poses) - 1
        for i, (pos, yaw) in enumerate(poses):
            robot.place(np.array(pos), yaw)
            om = om or OnlineMap(robot, episode, map_spec, crop)
            obs = robot.observe()
            om.add_frame(robot, obs["depth"])
            cam = cv2.resize(obs["rgb"], (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
            if goal is not None:
                cam[8:8 + goal.shape[0], 8:8 + goal.shape[1]] = goal
                cv2.rectangle(cam, (8, 8), (8 + goal.shape[1], 8 + goal.shape[0]), video.WHITE, 2)
                video.text(cam, "goal photo", (12, goal.shape[0] + 28), 0.55)
            points, chosen = plan_points(steps[i]["policy"] if i < len(steps) else None)
            local = map_picture(om.tensor(robot), arm.map_given, points, chosen)
            vp = steps[i - 1]["vp_geodesic_m"] if i else head["start_vp_geodesic_m"]
            panel = text_panel(i, max_steps, vp, sum(s["collided"] for s in steps[:i]), in_streak(i, streaks))
            top = np.zeros((TOP, WIDTH, 3), np.uint8)
            video.text(top, title, (10, 30), title_scale)
            if i == last:
                ok = res["success"]
                video.text(top, f"SUCCESS @{res['success_step']}" if ok else "FAILED", (WIDTH - 190, 31), 0.75,
                           video.GREEN if ok else video.RED)
            img = frame(top, cam, local, leg, plan.draw([q for q, _ in poses[:i + 1]], yaw, px=None), panel)
            if writer is None:
                writer = video.VideoWriter(out, (img.shape[1], img.shape[0]), FPS, crf=CRF, faststart=True)
            for _ in range(FPS * HOLD_S if i == last else 1):
                writer.write(img)
    writer.close()
    print(f"wrote {out} ({last} steps, {len(streaks)} stuck streaks, plan logged: "
          f"{any('path_m' in (s['policy'] or {}) for s in steps)})")


if __name__ == "__main__":
    main()
