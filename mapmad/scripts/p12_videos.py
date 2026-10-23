"""Phase 12 videos, rendered from the step logs after the runs (block item 11). Inside naz_mapmad_habitat, from
/app/visualnav-transformer (a recogniser sidecar on --recog-port redraws the YOLOE arms' target layer):

    python mapmad/scripts/p12_videos.py pick --config mapmad/configs/p12_exam.yaml           # -> videos/video_list.json
    python mapmad/scripts/p12_videos.py render --config ... --gpu 0 --recog-port 5790 [--only <name>]

Pick (seeded, from the exam runs' scores, seed index 0): 8 random exam episodes (seed `videos.seed`) + 4 by rule, each
the first of a seeded order among the episodes that qualify: (a) headline maps succeeds / maps hidden fails (oracle);
(b) headline stuck (a stuck streak); (c) headline wrong-object rule stop (tau); (d) F succeeds / headline fails
(oracle). A rule with no qualifying episode is logged and skipped.
Render: every video arm's logged run (seed index 0) is replayed in Habitat through the exam's own server code
(sidecar_server.SimService, the arm's search_v2 settings and noise seed; the robot is put on every logged pose, the
logged move goes through the bump rule): the camera picture and the robot's own near / far map windows are exactly
what the arm's policy got. One tile per arm (4 per row): camera | near (6.4 m) / far (25.6 m) window (unknown dark,
explored grey, obstacle red, frontier blue, target layer orange; robot yellow, forward = up) | floor plan (walkable
area, target instances red, their view points red dots, start green, path blue, robot yellow). Caption: arm, step,
"IN ZONE @t" once the oracle stop happened, "RULE STOP @t" (green = success, red = failure, "wrong object") at tau.
8 fps = 2x real time; a finished arm holds its last frame. Full size -> videos/<name>.mp4; small copy (<= 1280 px wide,
H.264 CRF 31, faststart, no audio) -> videos/small/<name>.mp4. Each video's inputs (log sha256s) -> video_list.json.
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
from mapmad_sim.search_wire import camera_wire, datagen_config, search_settings  # noqa: E402
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


def window_image(win: np.ndarray, size: int) -> np.ndarray:
    """[4, 64, 64] (obstacle, explored, frontier, target) -> RGB (size x size), forward = up, robot yellow."""
    obstacle, explored, frontier, target = (win[k] for k in range(4))
    img = np.zeros(obstacle.shape + (3,), np.float32) + UNKNOWN
    img[explored > 0] = EXPLORED
    img[frontier > 0] = FRONT
    img[obstacle > 0] = OBST
    a = np.clip(target, 0, 1)[..., None]
    img = img * (1 - a) + np.array(TGT, np.float32) * a
    img = cv2.resize(img.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
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


def replay_arm(service: SimService, layout: RunLayout, arm: str, log_path: Path, episode: Dict[str, Any],
               wire: Dict[str, Any], camera: Dict[str, Any], score: Dict[str, Any], tau: str) -> List[np.ndarray]:
    """Every frame of one arm's tile, replayed from its log (module docstring)."""
    log = steplog.read(log_path)
    head, steps = log[0], steplog.steps(log)
    sv = dict(wire, noise_seed=head["noise_seed"], record=False)
    reply = service.reset(episode, camera["hfov_deg"], False, 0.05, {"search_v2": sv}, camera)
    td = reply["topdown"]
    tmap = video.TopdownMap(np.asarray(td["grid"]), td["origin"], td["m_per_px"],
                            [episode["start_position"]] + [vp for i in episode["target"]["instances"] for vp in i["view_points"]])
    o_step = score.get("oracle_step")
    r = (score.get("rule") or {}).get(tau) or {}
    label = ArmTable.from_layout(layout).arm(arm).label or arm
    path = [head["start"]["position"]]
    frames = []
    for t in range(len(steps) + 1):
        status, colour = "", video.WHITE
        if o_step is not None and t >= o_step:
            status, colour = f"IN ZONE @{o_step}", video.GREEN
        if r.get("step") is not None and t >= r["step"]:
            status += (" | " if status else "") + f"RULE STOP @{r['step']}" + \
                ("" if r["success"] else (" wrong object" if r["wrong_object"] else " outside"))
            colour = video.GREEN if r["success"] else video.RED
        yaw = head["start"]["yaw"] if t == 0 else steps[t - 1]["yaw"]
        frames.append(tile(reply["rgb"], reply["maps"], plan_image(tmap, episode, path, yaw),
                           f"{arm}: {label[:26]} | t={t}", status, colour))
        if t == len(steps):
            break
        s = steps[t]
        service.robot.place(np.asarray(s["position"], np.float64), float(s["yaw"]))
        move = Move(v=float(s["v"]), w=float(s["w"]), travelled_m=float(s["travelled_m"]), short_m=float(s["short_m"]),
                    refused_substeps=0, collided=bool(s["collided"]))
        service.search.after_move(move, service.pose2d(), service.robot.spec.dt)
        reply = service.observation()
        path.append(s["position"])
    return frames


def wire_of(layout: RunLayout, arm: str, recog_port: Optional[int], tau: float) -> Dict[str, Any]:
    """The arm's server settings as run_search.Wire builds them (Habitat side; V1's map is not drawn)."""
    from vint_train.mapmad.config import load_config

    table = ArmTable.from_layout(layout)
    a = table.arm(arm)
    dg = datagen_config(layout.cfg)
    sv = search_settings(dg, load_config(str(table.resolve("repo:train/config/mapmad_v2.yaml")))["map"])
    sv["rule"] = layout.cfg["run"]["rule_stop"]
    target: Dict[str, Any] = {"source": a.target}
    if a.target == "recogniser":
        rc = table.recogniser(a.recogniser)
        target["recogniser"] = {"host": "naz_mapmad_recog", "port": int(recog_port), "model": rc["model"],
                                "threshold": rc["threshold"], "p99": rc["p99"], "vlm": rc["vlm"]}
    sv["target"] = target
    if a.policy == "frontier":
        sv["teacher"] = {"spec": dg["teacher"], "seen": a.teacher_seen, "mode": a.teacher_mode, "tau": tau,
                         "mark_radius_m": layout.cfg["policies"]["frontier"]["mark_radius_m"]}
    return sv


def scores_of(layout: RunLayout, subdir: str, arm: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    f = layout.base / subdir / "scores" / f"{arm}.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    return {(r["episode_id"], r["seed_index"]): r for r in rows}


def cmd_pick(args, layout: RunLayout) -> None:
    table = ArmTable.from_layout(layout)
    ch = table.choices()
    tau = f"{float(ch['tau']):.2f}"
    hl, hlh = {"v2": ("M2", "M2h"), "v21": ("M21", "M21h")}[ch["headline"]]
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
    camera = camera_wire(layout.cfg["camera"])
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
                tiles[arm] = replay_arm(service, layout, arm, log, e, wire_of(layout, arm, args.recog_port, tau),
                                        camera, score, lst["tau"])
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["pick", "render"])
    p.add_argument("--config", type=Path, default=config.CONFIG_DIR / "p12_exam.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--recog-port", type=int)
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    {"pick": cmd_pick, "render": cmd_render}[args.cmd](args, layout)


if __name__ == "__main__":
    main()
