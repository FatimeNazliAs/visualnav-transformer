"""Phase 12 videos, rendered from the step logs after the runs (block item 11). Inside naz_mapmad_habitat, from
/app/visualnav-transformer (a recogniser sidecar on --recog-port redraws the YOLOE arms' target layer):

    python mapmad/scripts/p12_videos.py pick --config mapmad/configs/p12_exam.yaml           # -> videos/video_list.json
    python mapmad/scripts/p12_videos.py render --config ... --gpu 0 --recog-port 5790 [--only <name>]
    python mapmad/scripts/p12_videos.py review --config ... --gpu 0 --recog-port 5790 [--only <name>]  # review copies
    python mapmad/scripts/p12_videos.py per-arm --config ... --gpu 0 --recog-port 5790 [--only <name>] # one per arm

Pick (seeded, from the exam runs' scores, seed index 0): 8 random exam episodes (seed `videos.seed`) + 4 by rule, each
the first of a seeded order among the episodes that qualify: (a) headline maps succeeds / maps hidden fails (oracle);
(b) headline stuck (a stuck streak); (c) headline wrong-object rule stop (tau); (d) F succeeds / headline fails
(oracle). A rule with no qualifying episode is logged and skipped.
Render: every video arm's logged run (seed index 0) is replayed in Habitat through the exam's own server code
(sidecar_server.SimService, the arm's search_v2 settings from search_wire.arm_wire, as the runner, and its noise seed; the robot is put on every logged pose, the
logged move goes through the bump rule): the camera picture and the robot's own near / far map windows are exactly
what the arm's policy got. One tile per arm (4 per row): camera | near (6.4 m) / far (25.6 m) window (unknown dark,
explored grey, obstacle red, frontier blue, target layer orange; robot yellow, forward = up) | floor plan (walkable
area, target instances red, their view points red dots, start green, path blue, robot yellow). Caption: arm, step,
"IN ZONE @t" once the oracle stop happened, "RULE STOP @t" (green = success, red = failure, "wrong object") at tau.
8 fps = 2x real time; a finished arm holds its last frame. Full size -> videos/<name>.mp4; small copy (<= 1280 px wide,
H.264 CRF 31, faststart, no audio) -> videos/small/<name>.mp4. Each video's inputs (log sha256s) -> video_list.json.
Review copies (Naz 10-26; the overview's 8 tiles are too small): videos/review/<name>.mp4, 1920 x 1080, 2 x 2 grid of
M21, M21h, N, F (every rule pick's pair is among them); per arm a caption band (>= 18 px letters: arm, step, and the
events at the step they happen -- first sight, oracle stop, rule stop with in zone / wrong object / near miss /
outside; a new event is shown large with a coloured frame for 2 s), the camera, the near and far windows at 256 px
(nearest neighbour) with the true target boxes, the path and the robot drawn in, a floor plan and a colour legend;
16 fps = 4x real time; H.264 CRF 23. videos/video_list.tsv lists every video.
Per-arm videos (Naz 10-26, 10-27): videos/per_arm/<pick>__<arm file name>.mp4 (MapMaD_v21_maps_shown,
MapMaD_v21_maps_hidden, NoMaD_explore, Map_frontier_planner) for 3 picks, 1920 x 1080, real time (one 0.25 s step = 6
frames at 24 fps): 3 caption lines (episode, find: '<category>', distance, the arm's full name; "this robot sees: ...";
step + events in plain words), the camera (x3) with the arm's name on it, the near and far windows (472 px, turn with
the robot), the whole floor north up (fixed: path, robot, true target, the online map's target marks), the legend
("target mark = YOLOE '<category>'"); pop-ups FIRST SAW THE TARGET / REACHED THE TARGET (success zone) / WOULD STOP
HERE BY ITSELF -- correct / wrong object / close but not there / far away. Text in DejaVu Sans (matplotlib's copy).
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))

from mapmad_bridge import steplog  # noqa: E402
from mapmad_sim import config  # noqa: E402
from mapmad_sim.arm_table import ArmTable  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.robot import Move  # noqa: E402
from mapmad_sim.run_layout import RunLayout, write_text_atomic  # noqa: E402
from mapmad_sim.run_plan import RunPlan  # noqa: E402
from mapmad_sim.search_wire import RECOG_HOST, arm_wire  # noqa: E402
from mapmad_sim.sidecar_server import SimService, limo_factory  # noqa: E402
from vint_train.mapmad.closed_loop import video  # noqa: E402

CAM = (256, 192)
WIN = 96
PLAN = 192
CAP = 22
TILE = (CAM[0] + WIN + PLAN, CAM[1] + CAP)
COLS = 4
FPS = 8
UNKNOWN, EXPLORED, OBST, FRONT, TGT = (45, 45, 45), (170, 170, 170), (200, 60, 60), (70, 120, 255), (255, 150, 0)


def window_image(win: np.ndarray, size: int, marker: bool = True) -> np.ndarray:
    """[4, 64, 64] (obstacle, explored, frontier, target) -> RGB (size x size), forward = up, robot yellow."""
    obstacle, explored, frontier, target = (win[k] for k in range(4))
    img = np.zeros(obstacle.shape + (3,), np.float32) + UNKNOWN
    img[explored > 0] = EXPLORED
    img[frontier > 0] = FRONT
    img[obstacle > 0] = OBST
    a = np.clip(target, 0, 1)[..., None]
    img = img * (1 - a) + np.array(TGT, np.float32) * a
    img = cv2.resize(img.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    if marker:
        c = size // 2
        cv2.drawMarker(img, (c, c), (255, 220, 0), cv2.MARKER_TRIANGLE_UP, 8, 2)
    return img


def plan_image(tmap: video.TopdownMap, episode: Dict[str, Any], path: List, yaw: float) -> np.ndarray:
    img = tmap.base.copy()
    for inst in episode["target"]["instances"]:
        c, s = inst["centre"], inst["size"]
        cv2.rectangle(img, tmap.px([c[0] - s[0] / 2, 0, c[2] - s[2] / 2]), tmap.px([c[0] + s[0] / 2, 0, c[2] + s[2] / 2]),
                      video.RED, 2)
        for vp in inst["view_points"][::12]:  # a few of them: success = within 0.2 m walking distance of one
            cv2.circle(img, tmap.px(vp), 1, video.RED, -1)
    cv2.circle(img, tmap.px(episode["start_position"]), 6, video.GREEN, -1, cv2.LINE_AA)
    if len(path) > 1:
        cv2.polylines(img, [np.array([tmap.px(p) for p in path], np.int32)], False, video.BLUE, 2, cv2.LINE_AA)
    here = np.array(tmap.px(path[-1]))
    tip = here + (np.array([-math.sin(yaw), -math.cos(yaw)]) * 14).astype(int)
    cv2.arrowedLine(img, tuple(int(v) for v in here), tuple(int(v) for v in tip), video.YELLOW, 3, cv2.LINE_AA, tipLength=0.5)
    return cv2.resize(img, (PLAN, PLAN), interpolation=cv2.INTER_AREA)


def tile(rgb: np.ndarray, maps: np.ndarray, plan: np.ndarray, caption: str, status: str, colour) -> np.ndarray:
    out = np.zeros((TILE[1], TILE[0], 3), np.uint8)
    out[CAP:CAP + CAM[1], :CAM[0]] = cv2.resize(rgb, CAM, interpolation=cv2.INTER_AREA)
    out[CAP:CAP + WIN, CAM[0]:CAM[0] + WIN] = window_image(maps[0], WIN)
    out[CAP + WIN:CAP + 2 * WIN, CAM[0]:CAM[0] + WIN] = window_image(maps[1], WIN)
    out[CAP:CAP + PLAN, CAM[0] + WIN:] = plan
    video.text(out, caption, (4, 15), 0.45)
    video.text(out, status, (CAM[0] + 4, 15), 0.45, colour)
    video.text(out, "near", (CAM[0] + 2, CAP + 12), 0.35)
    video.text(out, "far", (CAM[0] + 2, CAP + WIN + 12), 0.35)
    return out


def replay_states(service: SimService, log_path: Path, episode: Dict[str, Any], wire: Dict[str, Any],
                  camera: Dict[str, Any]):
    """The arm's logged run replayed pose by pose (module docstring). Yields (head, topdown map, t, server reply at step
    t, path so far (Habitat positions), yaw at t) for t = 0 .. steps."""
    log = steplog.read(log_path)
    head, steps = log[0], steplog.steps(log)
    sv = dict(wire, noise_seed=head["noise_seed"], record=False)
    reply = service.reset(episode, camera["hfov_deg"], False, 0.05, {"search_v2": sv}, camera)
    td = reply["topdown"]
    tmap = video.TopdownMap(np.asarray(td["grid"]), td["origin"], td["m_per_px"],
                            [episode["start_position"]] + [vp for i in episode["target"]["instances"] for vp in i["view_points"]])
    path = [head["start"]["position"]]
    for t in range(len(steps) + 1):
        yield head, tmap, t, reply, path, (head["start"]["yaw"] if t == 0 else steps[t - 1]["yaw"])
        if t == len(steps):
            break
        s = steps[t]
        service.robot.move_to(s["position"], float(s["yaw"]))  # a logged pose (holes: last floor height kept)
        move = Move(v=float(s["v"]), w=float(s["w"]), travelled_m=float(s["travelled_m"]), short_m=float(s["short_m"]),
                    refused_substeps=0, collided=bool(s["collided"]))
        service.search.after_move(move, service.pose2d(), service.robot.spec.dt)
        reply = service.observation()
        path.append(s["position"])


def replay_arm(service: SimService, layout: RunLayout, arm: str, log_path: Path, episode: Dict[str, Any],
               wire: Dict[str, Any], camera: Dict[str, Any], score: Dict[str, Any], tau: str) -> List[np.ndarray]:
    """Every frame of one arm's overview tile, replayed from its log (module docstring)."""
    o_step = score.get("oracle_step")
    r = (score.get("rule") or {}).get(tau) or {}
    label = ArmTable.from_layout(layout).arm(arm).label or arm
    frames = []
    for _, tmap, t, reply, path, yaw in replay_states(service, log_path, episode, wire, camera):
        status, colour = "", video.WHITE
        if o_step is not None and t >= o_step:
            status, colour = f"IN ZONE @{o_step}", video.GREEN
        if r.get("step") is not None and t >= r["step"]:
            status += (" | " if status else "") + f"RULE STOP @{r['step']}" + \
                ("" if r["success"] else (" wrong object" if r["wrong_object"] else " outside"))
            colour = video.GREEN if r["success"] else video.RED
        frames.append(tile(reply["rgb"], reply["maps"], plan_image(tmap, episode, path, yaw),
                           f"{arm}: {label[:26]} | t={t}", status, colour))
    return frames


def scores_of(layout: RunLayout, subdir: str, arm: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    f = layout.base / subdir / "scores" / f"{arm}.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    return {(r["episode_id"], r["seed_index"]): r for r in rows}


def cmd_pick(args, layout: RunLayout) -> None:
    table = ArmTable.from_layout(layout)
    ch = table.choices()
    tau = f"{float(ch['tau']):.2f}"
    hl, hlh = RunPlan.from_layout(layout).headline_pair()
    eps, fp = load_episodes(layout.base / layout.cfg["sets"]["exam"]["folder"] / "episodes.json")
    ids = [e["episode_id"] for e in eps]
    v = layout.cfg["videos"]
    rng = np.random.default_rng(int(v["seed"]))
    random_ids = [ids[int(i)] for i in rng.permutation(len(ids))[:int(v["random"])]]
    s = {a: scores_of(layout, "runs/exam", a) for a in (hl, hlh, "F")}
    order = [ids[int(i)] for i in rng.permutation(len(ids))]

    def first(pred) -> Optional[str]:
        return next((e for e in order if e not in random_ids and pred(e)), None)

    def ok(arm, e, key="oracle_success"):
        r = s[arm].get((e, 0))
        return r is not None and bool(r[key])

    rules = {"maps_ok_hidden_fails": first(lambda e: ok(hl, e) and (e, 0) in s[hlh] and not ok(hlh, e)),
             "headline_stuck": first(lambda e: (e, 0) in s[hl] and s[hl][(e, 0)]["stuck_streaks"] > 0),
             "headline_wrong_object": first(lambda e: (e, 0) in s[hl] and s[hl][(e, 0)]["rule"][tau]["wrong_object"]),
             "F_ok_headline_fails": first(lambda e: ok("F", e) and (e, 0) in s[hl] and not ok(hl, e))}
    picks = [{"name": f"random_{k + 1:02d}", "episode": e, "why": "seeded random pick"} for k, e in enumerate(random_ids)]
    picks += [{"name": f"rule_{n}", "episode": e, "why": n} for n, e in rules.items() if e is not None]
    out = {"seed": int(v["seed"]), "tau": tau, "headline": ch["headline"], "exam_fingerprint": fp, "seed_index": 0,
           "arms": v["arms"], "picks": picks, "rules_without_episode": [n for n, e in rules.items() if e is None]}
    write_text_atomic(layout.base / "videos" / "video_list.json", json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def cmd_render(args, layout: RunLayout) -> None:
    lst = json.loads((layout.base / "videos" / "video_list.json").read_text())
    eps = {e["episode_id"]: e for e in load_episodes(layout.base / layout.cfg["sets"]["exam"]["folder"] / "episodes.json")[0]}
    tau = float(lst["tau"])
    table = ArmTable.from_layout(layout)
    from vint_train.mapmad.config import load_config

    service = SimService(limo_factory(args.gpu))
    arms = [a for a in lst["arms"] if (layout.base / "runs/exam/logs" / a).exists()]
    try:
        for pick in lst["picks"]:
            if args.only and pick["name"] not in args.only:
                continue
            e = eps[pick["episode"]]
            tiles, used = {}, {}
            for arm in arms:
                log = layout.base / "runs/exam/logs" / arm / f"{e['episode_id']}__s0.jsonl"
                if not log.exists():
                    continue
                score = scores_of(layout, "runs/exam", arm).get((e["episode_id"], 0), {})
                w = arm_wire(table, arm, load_config, (RECOG_HOST, args.recog_port), tau)  # the run's own settings
                tiles[arm] = replay_arm(service, layout, arm, log, e, w.search_v2, w.camera, score, lst["tau"])
                used[arm] = hashlib.sha256(log.read_bytes()).hexdigest()
            n = max(len(f) for f in tiles.values())
            rows = math.ceil(len(tiles) / COLS)
            banner_h = 28
            size = (COLS * TILE[0], rows * TILE[1] + banner_h)
            out = layout.base / "videos" / f"{pick['name']}.mp4"
            w = video.VideoWriter(out, size, FPS)
            for t in range(n + 2 * FPS):
                frame = np.zeros((size[1], size[0], 3), np.uint8)
                video.text(frame, f"{pick['name']} ({pick['why']}) | {e['episode_id']} | find '{e['target']['category']}' "
                           f"| {e['start_geodesic_m']:.1f} m | seed 0 | tau {lst['tau']} | 2x speed", (6, 20), 0.55)
                for k, (arm, fr) in enumerate(tiles.items()):
                    r, c = divmod(k, COLS)
                    frame[banner_h + r * TILE[1]:banner_h + (r + 1) * TILE[1], c * TILE[0]:(c + 1) * TILE[0]] = fr[min(t, len(fr) - 1)]
                w.write(frame)
            w.close()
            small = layout.base / "videos" / "small" / out.name
            small.parent.mkdir(parents=True, exist_ok=True)
            import subprocess

            import imageio_ffmpeg

            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(out), "-vf",
                            "scale='min(1280,iw)':-2", "-c:v", "libx264", "-crf", "31", "-movflags", "+faststart", "-an",
                            str(small)], check=True)
            pick.update(file=str(out.relative_to(layout.base)), small=str(small.relative_to(layout.base)), logs=used,
                        frames=n)
            print(json.dumps({k: pick[k] for k in ("name", "episode", "file", "frames")}), flush=True)
    finally:
        service.shutdown()
    write_text_atomic(layout.base / "videos" / "video_list.json", json.dumps(lst, indent=1) + "\n")


# --- review copies (Naz 10-26: the 8-arm overview is too small to review) -------------------------------------------
REVIEW_ARMS = ("M21", "M21h", "N", "F")  # 2 x 2 grid, row by row; every rule pick's pair is among them
REVIEW_SIZE = (1920, 1080)
CELL = (960, 540)
REVIEW_FPS = 16  # 4 Hz control -> 4x real time
REVIEW_CRF = 23
CAPTION_PX = 18  # minimum caption letter height (capital H)
HEAD = 70  # caption band of a cell
CAM_R = (448, 336)
MAP_R = 256  # near / far window side: 64 cells x 4 px (nearest neighbour)
PLAN_R = 214
HILITE = 2 * REVIEW_FPS  # an event is highlighted for 2 s of video
TRUE_TGT, PATH_C, ROBOT_C = (255, 60, 255), (255, 255, 255), (255, 220, 0)
LEGEND = [("unknown", UNKNOWN), ("explored", EXPLORED), ("obstacle", OBST), ("frontier", FRONT),
          ("target mark (layer)", TGT), ("true target", TRUE_TGT), ("path", PATH_C), ("robot", ROBOT_C)]


def caption_scale(px: int = CAPTION_PX, thickness: int = 2) -> float:
    """The smallest Hershey-simplex scale whose capital letters are >= px pixels high."""
    s = 0.5
    while cv2.getTextSize("H", cv2.FONT_HERSHEY_SIMPLEX, s, thickness)[0][1] < px:
        s += 0.05
    return round(s, 2)


CAP_SCALE = caption_scale()


def big_text(img: np.ndarray, s: str, org: Tuple[int, int], scale: float = CAP_SCALE, colour=video.WHITE) -> int:
    """Outlined caption text; returns the x just after it."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, video.BLACK, 5, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)
    return org[0] + cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]


def to_window_px(points: np.ndarray, pose2d: Tuple[float, float, float], cell_m: float, side: int) -> np.ndarray:
    """Habitat points (N, 3) -> pixel (col, row) in a robot-centred window picture (forward = up, left = left; the
    map_v2 / local_map cell convention: row 0 farthest ahead, col 0 farthest left)."""
    from mapmad_sim.frames import to_2d

    xy = to_2d(np.asarray(points, np.float64).reshape(-1, 3))
    x, y, yaw = pose2d
    c, s_ = math.cos(yaw), math.sin(yaw)
    dx, dy = xy[:, 0] - x, xy[:, 1] - y
    fwd, left = dx * c + dy * s_, -dx * s_ + dy * c
    k = side / 64.0
    return np.stack([(32.0 - left / cell_m) * k, (32.0 - fwd / cell_m) * k], axis=1)


def map_panel(win: np.ndarray, pose2d, cell_m: float, episode: Dict[str, Any], path: List, label: str,
              side: int = MAP_R) -> np.ndarray:
    """One window (4 layers) at `side` px with the true target boxes, the path and the robot drawn in."""
    img = window_image(win, side, marker=False)
    for inst in episode["target"]["instances"]:
        c, sz = inst["centre"], inst["size"]
        corners = [[c[0] + a * sz[0] / 2, c[1], c[2] + b * sz[2] / 2] for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        poly = to_window_px(np.array(corners), pose2d, cell_m, side)
        cv2.polylines(img, [np.round(poly).astype(np.int32)], True, TRUE_TGT, 2, cv2.LINE_AA)
    if len(path) > 1:
        pts = to_window_px(np.array(path), pose2d, cell_m, side)
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, PATH_C, 1, cv2.LINE_AA)
    c, k = side // 2, side / MAP_R
    tri = np.round(np.array([[c, c - 9 * k], [c - 6 * k, c + 6 * k], [c + 6 * k, c + 6 * k]])).astype(np.int32)  # forward = up
    cv2.fillPoly(img, [tri], ROBOT_C)
    video.text(img, label, (4, int(16 * k)), 0.5 * k)
    return img


def legend_panel(size: Tuple[int, int], info: str) -> np.ndarray:
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for k, (name, colour) in enumerate(LEGEND):
        x, y = 8 + (k % 2) * 220, 6 + (k // 2) * 22
        if name == "true target":
            cv2.rectangle(img, (x, y), (x + 18, y + 14), colour, 2)
        elif name == "path":
            cv2.line(img, (x, y + 7), (x + 18, y + 7), colour, 2)
        elif name == "robot":
            cv2.fillPoly(img, [np.array([[x + 9, y], [x + 2, y + 14], [x + 16, y + 14]], np.int32)], colour)
        else:
            cv2.rectangle(img, (x, y), (x + 18, y + 14), colour, -1)
        video.text(img, name, (x + 26, y + 13), 0.55)
    video.text(img, "IN ZONE = oracle stop (<= 0.2 m to a view point)", (8, size[1] - 26), 0.5, (200, 200, 200))
    video.text(img, info, (8, size[1] - 6), 0.5, (200, 200, 200))
    return img


def arm_events(score: Dict[str, Any], tau: str) -> List[Tuple[int, str, Tuple[int, int, int]]]:
    """(step, text, colour) of the run's events: first sight (simulator truth), oracle stop (zone entered), rule stop."""
    ev = []
    if score.get("first_sight") is not None:
        ev.append((int(score["first_sight"]), f"first sight @{score['first_sight']}", (90, 220, 255)))
    if score.get("oracle_step") is not None:
        ev.append((int(score["oracle_step"]), f"IN ZONE @{score['oracle_step']}", video.GREEN))
    r = (score.get("rule") or {}).get(tau) or {}
    if r.get("step") is not None:
        how = "in zone" if r["success"] else ("WRONG OBJECT" if r["wrong_object"] else
                                              ("near miss" if r["near_miss"] else "outside zone"))
        ev.append((int(r["step"]), f"RULE STOP @{r['step']}: {how}", video.GREEN if r["success"] else video.RED))
    return sorted(ev)


def review_frames(service: SimService, layout: RunLayout, arm: str, log_path: Path, episode: Dict[str, Any],
                  w: Any, score: Dict[str, Any], tau: str, info: str) -> List[np.ndarray]:
    """Every frame of one arm's review cell (CELL px): caption band (arm, step, events at their step), camera, near
    and far windows (MAP_R px each), floor plan, legend."""
    label = ArmTable.from_layout(layout).arm(arm).label or arm
    events = arm_events(score, tau)
    near_m, far_m = float(w.search_v2["near_m"]), float(w.search_v2["far_m"])
    legend = legend_panel((CAM_R[0], CELL[1] - HEAD - CAM_R[1]), info)
    frames = []
    for head, tmap, t, reply, path, yaw in replay_states(service, log_path, episode, w.search_v2, w.camera):
        from mapmad_sim.frames import to_2d

        xy = to_2d(np.asarray(path[-1], np.float64))
        pose2d = (float(xy[0]), float(xy[1]), float(yaw))
        cell = np.zeros((CELL[1], CELL[0], 3), np.uint8)
        x = big_text(cell, f"{arm}: {label[:40]}", (10, 28))
        big_text(cell, f"step {t}", (max(x + 20, CELL[0] - 150), 28), colour=(200, 200, 200))
        x, fresh = 10, None
        for step, txt, colour in events:
            if t >= step:
                x = big_text(cell, txt, (x, 60), colour=colour) + 18
                if t - step < HILITE:
                    fresh = (txt, colour)
        cell[HEAD:HEAD + CAM_R[1], :CAM_R[0]] = cv2.resize(reply["rgb"], CAM_R, interpolation=cv2.INTER_LINEAR)
        cell[HEAD + CAM_R[1]:, :CAM_R[0]] = legend
        mx = CAM_R[0]
        cell[HEAD:HEAD + MAP_R, mx:mx + MAP_R] = map_panel(reply["maps"][0], pose2d, near_m, episode, path,
                                                            f"near {64 * near_m:.1f} m")
        cell[HEAD:HEAD + MAP_R, mx + MAP_R:mx + 2 * MAP_R] = map_panel(reply["maps"][1], pose2d, far_m, episode, path,
                                                                        f"far {64 * far_m:.1f} m")
        plan = cv2.resize(review_plan(tmap, episode, path, yaw), (PLAN_R, PLAN_R), interpolation=cv2.INTER_AREA)
        py = HEAD + MAP_R + (CELL[1] - HEAD - MAP_R - PLAN_R) // 2
        cell[py:py + PLAN_R, mx + (2 * MAP_R - PLAN_R) // 2:mx + (2 * MAP_R + PLAN_R) // 2] = plan
        if fresh is not None:  # the event of this step, large over the camera, and a frame around the cell
            cv2.rectangle(cell, (0, 0), (CELL[0] - 1, CELL[1] - 1), fresh[1], 6)
            for k, line in enumerate(fresh[0].split(": ")):  # two lines at most, on the camera picture only
                big_text(cell, line, (12, HEAD + 40 + 38 * k), CAP_SCALE * 1.25, fresh[1])
        frames.append(cell)
    return frames


def review_plan(tmap: video.TopdownMap, episode: Dict[str, Any], path: List, yaw: float) -> np.ndarray:
    """Floor plan in the review colours: walkable grey, true target boxes, start green, path, robot."""
    img = tmap.base.copy()
    for inst in episode["target"]["instances"]:
        c, s_ = inst["centre"], inst["size"]
        cv2.rectangle(img, tmap.px([c[0] - s_[0] / 2, 0, c[2] - s_[2] / 2]),
                      tmap.px([c[0] + s_[0] / 2, 0, c[2] + s_[2] / 2]), TRUE_TGT, 3)
    cv2.circle(img, tmap.px(episode["start_position"]), 7, video.GREEN, -1, cv2.LINE_AA)
    if len(path) > 1:
        cv2.polylines(img, [np.array([tmap.px(p) for p in path], np.int32)], False, PATH_C, 2, cv2.LINE_AA)
    here = np.array(tmap.px(path[-1]))
    tip = here + (np.array([-math.sin(yaw), -math.cos(yaw)]) * 18).astype(int)
    cv2.arrowedLine(img, tuple(int(v) for v in here), tuple(int(v) for v in tip), ROBOT_C, 4, cv2.LINE_AA, tipLength=0.5)
    return img


def cmd_review(args, layout: RunLayout) -> None:
    """Review copies: videos/review/<name>.mp4, 1920 x 1080, 2 x 2 grid of REVIEW_ARMS, 4x speed, H.264 CRF 23; the
    inputs (log sha256s) -> video_list.json picks[].review; one line per video -> videos/video_list.tsv."""
    lst = json.loads((layout.base / "videos" / "video_list.json").read_text())
    eps = {e["episode_id"]: e for e in load_episodes(layout.base / layout.cfg["sets"]["exam"]["folder"] / "episodes.json")[0]}
    table = ArmTable.from_layout(layout)
    from vint_train.mapmad.config import load_config

    pair_of = {"rule_maps_ok_hidden_fails": ("M21", "M21h"), "rule_headline_stuck": ("M21",),
               "rule_headline_wrong_object": ("M21",), "rule_F_ok_headline_fails": ("F", "M21")}
    assert all(set(v) <= set(REVIEW_ARMS) for v in pair_of.values()) and lst["headline"] == "v21"
    service = SimService(limo_factory(args.gpu))
    try:
        for pick in lst["picks"]:
            if args.only and pick["name"] not in args.only:
                continue
            e = eps[pick["episode"]]
            info = f"{e['episode_id']} | {e['start_geodesic_m']:.1f} m | seed 0 | tau {lst['tau']} | 4x"
            cells, used = [], {}
            for arm in REVIEW_ARMS:
                log = layout.base / "runs/exam/logs" / arm / f"{e['episode_id']}__s0.jsonl"
                score = scores_of(layout, "runs/exam", arm)[(e["episode_id"], 0)]
                w = arm_wire(table, arm, load_config, (RECOG_HOST, args.recog_port), float(lst["tau"]))
                cells.append(review_frames(service, layout, arm, log, e, w, score, lst["tau"], info))
                used[arm] = hashlib.sha256(log.read_bytes()).hexdigest()
            n = max(len(c) for c in cells)
            out = layout.base / "videos" / "review" / f"{pick['name']}.mp4"
            wr = video.VideoWriter(out, REVIEW_SIZE, REVIEW_FPS, crf=REVIEW_CRF, faststart=True)
            for t in range(n + 2 * REVIEW_FPS):
                frame = np.zeros((REVIEW_SIZE[1], REVIEW_SIZE[0], 3), np.uint8)
                for k, fr in enumerate(cells):
                    r, c = divmod(k, 2)
                    frame[r * CELL[1]:(r + 1) * CELL[1], c * CELL[0]:(c + 1) * CELL[0]] = fr[min(t, len(fr) - 1)]
                cv2.line(frame, (CELL[0], 0), (CELL[0], REVIEW_SIZE[1]), (90, 90, 90), 2)
                cv2.line(frame, (0, CELL[1]), (REVIEW_SIZE[0], CELL[1]), (90, 90, 90), 2)
                wr.write(frame)
            wr.close()
            pick["review"] = {"file": str(out.relative_to(layout.base)), "arms": list(REVIEW_ARMS), "logs": used,
                              "frames": n, "fps": REVIEW_FPS, "size": list(REVIEW_SIZE), "crf": REVIEW_CRF,
                              "caption_scale": CAP_SCALE}
            print(json.dumps({"name": pick["name"], "file": pick["review"]["file"], "frames": n}), flush=True)
    finally:
        service.shutdown()
    write_text_atomic(layout.base / "videos" / "video_list.json", json.dumps(lst, indent=1) + "\n")
    write_tsv(layout, lst, eps)


# --- per-arm videos (Naz 10-26 / 10-27: each arm on its own, real time) -----------------------------------------------
PER_ARM_PICKS = ("rule_maps_ok_hidden_fails", "rule_F_ok_headline_fails", "random_01")
PA_FPS, PA_REPEAT = 24, 6  # one 0.25 s control step = 6 frames at 24 fps: real time (1x)
PA_HEAD = 136  # 3 caption lines
PA_CAM = (960, 720)  # the 320 x 240 camera, x3
PA_MAP = 472  # near / far window side (>= 384 px; 64 cells, nearest neighbour)
PA_FLOOR = 472  # whole floor, north up (does not turn)
ARMS_PLAIN = {  # file name, full name, what the robot itself gets
    "M21": ("MapMaD_v21_maps_shown", "MapMaD v2.1 — maps shown", "this robot sees: camera + both maps (near and far)"),
    "M21h": ("MapMaD_v21_maps_hidden", "MapMaD v2.1 — maps hidden",
             "this robot sees: camera only — the maps are drawn for you, not seen by the robot"),
    "N": ("NoMaD_explore", "Original NoMaD — explore",
          "this robot sees: camera only, no target — original NoMaD explore (maps drawn for you)"),
    "F": ("Map_frontier_planner", "Map + frontier planner", "this robot sees: a planner on the map, no network"),
}
RULE_PLAIN = {"success": "correct", "wrong_object": "wrong object", "near_miss": "close but not there",
              "other": "far away"}


def font(px: int):
    """DejaVu Sans Bold at px pixels (matplotlib's copy: no system font path)."""
    from matplotlib import font_manager
    from PIL import ImageFont

    return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")), px)


class Writer:
    """Text drawn with a TrueType font (dashes, quotes) on an RGB frame: queue items, then `draw` once per frame."""

    def __init__(self) -> None:
        self.fonts: Dict[int, Any] = {}
        self.items: List[Tuple[Tuple[int, int], str, int, Tuple[int, int, int]]] = []

    def add(self, xy: Tuple[int, int], s: str, px: int, colour=video.WHITE) -> int:
        """Queues s at xy (top left); returns the x just after it."""
        f = self.fonts.setdefault(px, font(px))
        self.items.append((xy, s, px, colour))
        return xy[0] + int(f.getlength(s))

    def draw(self, img: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        for xy, s, px, colour in self.items:
            d.text(xy, s, font=self.fonts[px], fill=tuple(colour), stroke_width=max(2, px // 12), stroke_fill=(0, 0, 0))
        self.items = []
        return np.asarray(pil).copy()


def plain_events(score: Dict[str, Any], tau: str) -> List[Tuple[int, str, str, Tuple[int, int, int]]]:
    """(step, caption text, pop-up text, colour) of the run's events in plain words."""
    ev = []
    if score.get("first_sight") is not None:
        s = int(score["first_sight"])
        ev.append((s, f"first saw the target @{s}", "FIRST SAW THE TARGET", (90, 220, 255)))
    if score.get("oracle_step") is not None:
        s = int(score["oracle_step"])
        ev.append((s, f"reached the target @{s}", "REACHED THE TARGET (success zone)", video.GREEN))
    r = (score.get("rule") or {}).get(tau) or {}
    if r.get("step") is not None:
        how = RULE_PLAIN["success" if r["success"] else ("wrong_object" if r["wrong_object"] else
                                                         ("near_miss" if r["near_miss"] else "other"))]
        ev.append((int(r["step"]), f"would stop by itself @{r['step']}: {how}", f"WOULD STOP HERE BY ITSELF — {how}",
                   video.GREEN if r["success"] else video.RED))
    return sorted(ev)


def per_arm_legend(size: Tuple[int, int], category: str) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int], str]]]:
    """Colour legend: the swatches (picture) and where its words go (drawn with the frame's text)."""
    img = np.zeros((size[1], size[0], 3), np.uint8)
    names = [n if n != "target mark (layer)" else f"target mark = YOLOE ‘{category}’" for n, _ in LEGEND]
    places = []
    for k, ((key, colour), name) in enumerate(zip(LEGEND, names)):
        x, y = (16, 14 + 3 * 40) if key == "target mark (layer)" else (16 + (k % 3) * 315, 14 + (k // 3) * 40)
        if key == "true target":
            cv2.rectangle(img, (x, y), (x + 26, y + 22), colour, 3)
        elif key == "path":
            cv2.line(img, (x, y + 11), (x + 26, y + 11), colour, 3)
        elif key == "robot":
            cv2.fillPoly(img, [np.array([[x + 13, y], [x + 2, y + 22], [x + 24, y + 22]], np.int32)], colour)
        else:
            cv2.rectangle(img, (x, y), (x + 26, y + 22), colour, -1)
        places.append(((x + 36, y - 1), name))
    return img, places


def floor_lookup(tmap: video.TopdownMap, service: SimService) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every floor-plan pixel: the online-map cell under its centre (i, j) and whether it lies inside the map."""
    x, z = tmap.centres()
    X, Y = -z, -x  # Habitat (x, z) -> traj frame (X, Y) = (-z, -x) (frames.to_2d)
    origin = np.asarray(service.search.fmap.origin, np.float64)
    res = float(service.search.spec.map_spec.resolution)
    shape = service.search.fmap.shape
    i = np.floor((X - origin[0]) / res).astype(np.int64)
    j = np.floor((Y - origin[1]) / res).astype(np.int64)
    inside = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    return np.where(inside, i, 0), np.where(inside, j, 0), inside


def floor_panel(tmap: video.TopdownMap, episode: Dict[str, Any], path: List, yaw: float, marks: np.ndarray,
                lookup) -> np.ndarray:
    """Whole floor, north up (fixed): walkable floor, the target marks of the online map (orange, stronger = more
    sure), true target boxes, start, path, robot."""
    img = tmap.base.copy().astype(np.float32)
    i, j, inside = lookup
    a = np.where(inside, np.clip(marks[i, j], 0, 1), 0)[..., None]
    img = (img * (1 - a) + np.array(TGT, np.float32) * a).astype(np.uint8)
    for inst in episode["target"]["instances"]:
        c, s_ = inst["centre"], inst["size"]
        cv2.rectangle(img, tmap.px([c[0] - s_[0] / 2, 0, c[2] - s_[2] / 2]),
                      tmap.px([c[0] + s_[0] / 2, 0, c[2] + s_[2] / 2]), TRUE_TGT, 2)
    cv2.circle(img, tmap.px(episode["start_position"]), 7, video.GREEN, -1, cv2.LINE_AA)
    if len(path) > 1:
        cv2.polylines(img, [np.array([tmap.px(p) for p in path], np.int32)], False, PATH_C, 2, cv2.LINE_AA)
    here = np.array(tmap.px(path[-1]))
    tip = here + (np.array([-math.sin(yaw), -math.cos(yaw)]) * 18).astype(int)
    cv2.arrowedLine(img, tuple(int(v) for v in here), tuple(int(v) for v in tip), ROBOT_C, 4, cv2.LINE_AA, tipLength=0.5)
    return img


def render_per_arm(service: SimService, arm: str, log_path: Path, episode: Dict[str, Any], w: Any,
                   score: Dict[str, Any], tau: str, out: Path) -> int:
    """One arm's own video (module docstring), streamed to `out`; returns the number of logged frames (steps + 1)."""
    from mapmad_sim.frames import to_2d

    _, name, sees = ARMS_PLAIN[arm]
    cat = episode["target"]["category"]
    events = plain_events(score, tau)
    near_m, far_m = float(w.search_v2["near_m"]), float(w.search_v2["far_m"])
    W, H = REVIEW_SIZE
    cy = PA_HEAD
    text = Writer()
    legend, places = per_arm_legend((PA_CAM[0], H - cy - PA_CAM[1]), cat)
    title = f"{episode['episode_id']}  |  find: ‘{cat}’  |  {episode['start_geodesic_m']:.1f} m  |  {name}"
    vw = video.VideoWriter(out, REVIEW_SIZE, PA_FPS, crf=REVIEW_CRF, faststart=True)
    n, f, lookup = 0, None, None
    fx = PA_CAM[0] + (W - PA_CAM[0] - PA_FLOOR) // 2
    for head, tmap, t, reply, path, yaw in replay_states(service, log_path, episode, w.search_v2, w.camera):
        if lookup is None:
            lookup = floor_lookup(tmap, service)
        xy = to_2d(np.asarray(path[-1], np.float64))
        pose2d = (float(xy[0]), float(xy[1]), float(yaw))
        f = np.zeros((H, W, 3), np.uint8)
        f[cy:cy + PA_CAM[1], :PA_CAM[0]] = cv2.resize(reply["rgb"], PA_CAM, interpolation=cv2.INTER_LINEAR)
        f[cy + PA_CAM[1]:, :PA_CAM[0]] = legend
        f[cy:cy + PA_MAP, PA_CAM[0]:PA_CAM[0] + PA_MAP] = map_panel(
            reply["maps"][0], pose2d, near_m, episode, path, "", PA_MAP)
        f[cy:cy + PA_MAP, PA_CAM[0] + PA_MAP:PA_CAM[0] + 2 * PA_MAP] = map_panel(
            reply["maps"][1], pose2d, far_m, episode, path, "", PA_MAP)
        text.add((PA_CAM[0] + 8, cy + 6), f"near {64 * near_m:.1f} m — turns with the robot", 20)
        text.add((PA_CAM[0] + PA_MAP + 8, cy + 6), f"far {64 * far_m:.1f} m — turns with the robot", 20)
        floor = floor_panel(tmap, episode, path, yaw, service.search.fusions["noisy"].grid(), lookup)
        f[cy + PA_MAP:cy + PA_MAP + PA_FLOOR, fx:fx + PA_FLOOR] = cv2.resize(floor, (PA_FLOOR, PA_FLOOR),
                                                                             interpolation=cv2.INTER_AREA)
        text.add((fx + 8, cy + PA_MAP + 6), "whole floor, north up (fixed)", 20)
        text.add((12, 8), title, 30)
        text.add((12, 50), sees, 30, (255, 230, 120))
        x, fresh = text.add((12, 94), f"step {t}", 26, (200, 200, 200)) + 28, None
        for step, cap, pop, colour in events:
            if t >= step:
                x = text.add((x, 94), cap, 26, colour) + 28
                if t - step < HILITE:
                    fresh = (pop, colour)
        text.add((16, cy + PA_CAM[1] - 64), name, 44)  # the arm, big on the camera picture
        if fresh is not None:  # the event of this step, large over the camera, and a frame around the camera
            cv2.rectangle(f, (0, cy), (PA_CAM[0] - 1, cy + PA_CAM[1] - 1), fresh[1], 8)
            for k, line in enumerate(fresh[0].split(" — ")):
                text.add((22, cy + 16 + 52 * k), line, 40, fresh[1])
        for xy_, s in places:
            text.add((xy_[0], cy + PA_CAM[1] + xy_[1]), s, 22)
        text.add((16, H - 30), "turns with the robot: near / far windows, forward = up.  "
                 "REACHED THE TARGET = within 0.2 m walk of a view point", 17, (200, 200, 200))
        f = text.draw(f)
        for _ in range(PA_REPEAT):
            vw.write(f)
        n += 1
    for _ in range(2 * PA_FPS):  # hold the last frame 2 s
        vw.write(f)
    vw.close()
    return n


def cmd_per_arm(args, layout: RunLayout) -> None:
    """Per-arm videos: videos/per_arm/<pick>__<arm file name>.mp4 for PER_ARM_PICKS x REVIEW_ARMS; inputs ->
    video_list.json picks[].per_arm (files + log sha256s); videos/video_list.tsv rewritten."""
    lst = json.loads((layout.base / "videos" / "video_list.json").read_text())
    eps = {e["episode_id"]: e for e in load_episodes(layout.base / layout.cfg["sets"]["exam"]["folder"] / "episodes.json")[0]}
    table = ArmTable.from_layout(layout)
    from vint_train.mapmad.config import load_config

    service = SimService(limo_factory(args.gpu))
    try:
        for pick in lst["picks"]:
            if pick["name"] not in PER_ARM_PICKS or (args.only and pick["name"] not in args.only):
                continue
            e = eps[pick["episode"]]
            files, used, frames = {}, {}, {}
            for arm in REVIEW_ARMS:
                log = layout.base / "runs/exam/logs" / arm / f"{e['episode_id']}__s0.jsonl"
                score = scores_of(layout, "runs/exam", arm)[(e["episode_id"], 0)]
                w = arm_wire(table, arm, load_config, (RECOG_HOST, args.recog_port), float(lst["tau"]))
                out = layout.base / "videos" / "per_arm" / f"{pick['name']}__{ARMS_PLAIN[arm][0]}.mp4"
                frames[arm] = render_per_arm(service, arm, log, e, w, score, lst["tau"], out)
                files[arm] = str(out.relative_to(layout.base))
                used[arm] = hashlib.sha256(log.read_bytes()).hexdigest()
                print(json.dumps({"name": pick["name"], "arm": arm, "file": files[arm], "frames": frames[arm]}), flush=True)
            pick["per_arm"] = {"files": files, "logs": used, "frames": frames, "fps": PA_FPS, "repeat": PA_REPEAT,
                               "speed": "1x (one 0.25 s step = 6 frames at 24 fps)", "size": list(REVIEW_SIZE),
                               "crf": REVIEW_CRF}
    finally:
        service.shutdown()
    write_text_atomic(layout.base / "videos" / "video_list.json", json.dumps(lst, indent=1) + "\n")
    write_tsv(layout, lst, eps)


def write_tsv(layout: RunLayout, lst: Dict[str, Any], eps: Dict[str, Dict[str, Any]]) -> None:
    """videos/video_list.tsv: one line per pick (overview, review copy, per-arm videos)."""
    rows = ["\t".join(["name", "episode", "category", "start_m", "why", "review", "review_arms", "per_arm", "overview",
                       "overview_small", "frames"])]
    for pick in lst["picks"]:
        e = eps[pick["episode"]]
        rv = pick.get("review") or {}
        pa = (pick.get("per_arm") or {}).get("files") or {}
        rows.append("\t".join([pick["name"], e["episode_id"], e["target"]["category"], f"{e['start_geodesic_m']:.1f}",
                               pick["why"], rv.get("file", "-"), ",".join(rv.get("arms", [])),
                               ",".join(pa.values()) or "-", pick.get("file", "-"), pick.get("small", "-"),
                               str(rv.get("frames", pick.get("frames", "-")))]))
    write_text_atomic(layout.base / "videos" / "video_list.tsv", "\n".join(rows) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["pick", "render", "review", "per-arm"])
    p.add_argument("--config", type=Path, default=config.CONFIG_DIR / "p12_exam.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--recog-port", type=int)
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    {"pick": cmd_pick, "render": cmd_render, "review": cmd_review, "per-arm": cmd_per_arm}[args.cmd](args, layout)


if __name__ == "__main__":
    main()
