"""Phase 2 checks on a generated dataset: statistics (gate G2 row 2), videos and alignment overlays (G2 row 1).

    python mapmad/scripts/p2_report.py --dataset pilot                       # inside naz_mapmad_habitat
    python mapmad/scripts/p2_report.py --dataset full --videos 0 --overlays 0   # statistics only

Writes <outputs>/p2_datagen/<dataset>/report/:
- stats.json, stats.md: drives (spot / object), frames, MB, seconds and CPU seconds per drive, mean metres per
  frame, collisions, discards, detour / wall-recovery / kick shares, spot-turn frames (inside and outside the
  allowed phases), end_fallback share, and the full-size projection for configs/p2_datagen.yaml's drive counts;
- videos/<drive>.mp4: camera (left) + the 64 x 64 local map as known at each frame (right; vint_train.mapmad.local_map);
- overlays/<drive>_<t>.png: the robot is put back on the logged pose, the depth picture's obstacle-band pixels are
  tinted red, and the local map's obstacle cells (cut by local_map.py from the stored map and traj_data.pkl pose,
  the exact path Phase 3's loader uses) are projected into the picture as green dots at 10 cm above the floor.
  Aligned = green dots sit on the red wall bases. Right: the local map with the camera's field of view.
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))  # numpy/OpenCV-only NoMaD-side modules (local_map, map_viz, video)

from mapmad_sim import config, frames  # noqa: E402
from mapmad_sim.datagen import PHASES  # noqa: E402
from mapmad_sim.depth_map import camera_points, world_points  # noqa: E402
from mapmad_sim.camera import project_points  # noqa: E402
from mapmad_sim.home_splits import train_homes  # noqa: E402
from mapmad_sim.robot import LimoSim, RobotSpec  # noqa: E402
from mapmad_sim.run_layout import read_config  # noqa: E402
from vint_train.mapmad import local_map as lm, map_viz  # noqa: E402
from vint_train.mapmad.closed_loop.video import VideoWriter, text  # noqa: E402

CONFIG = config.CONFIG_DIR / "p2_datagen.yaml"
RED, GREEN = (230, 40, 40), (40, 230, 60)


# --- statistics -------------------------------------------------------------------------------------------------

def load_metas(data: Path) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(data.glob("*_*/mapmad_meta.json"))]


def load_homes(data: Path) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((data / "_homes").glob("*.json"))]


def share(xs) -> float:
    xs = list(xs)
    return round(float(np.mean(xs)), 4) if xs else float("nan")


def floor_checks(homes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Real-floor check outcomes summed over labelled and unlabelled homes, with the rejected share."""
    out = {}
    for kind, sel in (("labelled", True), ("unlabelled", False)):
        total: Dict[str, int] = {}
        for h in homes:
            if h["labelled"] == sel:
                for k, v in h.get("floor_checks", {}).items():
                    total[k] = total.get(k, 0) + v
        n = sum(total.values())
        out[kind] = {**total, "rejected_share": round(1 - total.get("ok", 0) / n, 4) if n else None}
    return out


def statistics(metas: List[Dict[str, Any]], homes: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    frames_total = sum(m["frames"] for m in metas)
    obj = [m for m in metas if m["kind"] == "object"]
    attempts = [a for h in homes for s in h["summaries"] for a in s.get("discarded_attempts", [])]
    kept = len(metas)
    discard_reasons: Dict[str, int] = {}
    for a in attempts:
        discard_reasons[a["discard"]] = discard_reasons.get(a["discard"], 0) + 1
    mb = np.array([m["bytes"] for m in metas]) / 1e6
    jpg_mb = np.array([m["jpg_bytes"] for m in metas]) / 1e6 if all("jpg_bytes" in m for m in metas) else mb
    stats = {
        "homes": len(homes), "drives": kept, "spot": kept - len(obj), "object": len(obj),
        "missing_drives": sum(len(h["missing"]) for h in homes),
        "frames": frames_total, "frames_per_drive": round(frames_total / max(kept, 1), 1),
        "mb_per_drive": round(float(mb.mean()), 3), "jpg_mb_per_drive": round(float(jpg_mb.mean()), 3),
        "gb_total": round(float(mb.sum()) / 1e3, 3),
        "s_per_drive": round(float(np.mean([m["seconds"] for m in metas])), 3),
        "cpu_s_per_drive": round(float(np.mean([m["cpu_seconds"] for m in metas])), 3),
        "home_s_total": round(sum(h["seconds"] for h in homes), 1),
        "mean_m_per_frame": round(sum(m["path_m"] for m in metas) / max(frames_total - kept, 1), 4),
        "drives_with_collision": share(m["collision_steps"] > 0 for m in metas),
        "collision_step_share": round(sum(m["collision_steps"] for m in metas) / max(frames_total, 1), 5),
        "discarded_attempts": len(attempts), "discard_rate": round(len(attempts) / max(len(attempts) + kept, 1), 4),
        "discard_reasons": discard_reasons,
        "homes_flagged_many_discards": [h["home"] for h in homes if h["flag_many_discards"]],
        "detour_realised_share": share(m["detour_realised"] for m in metas),
        "detour_asked_share": share(m["variations"]["detours"] > 0 for m in metas),
        "wall_recovery_share": share(m["wall_recovery_realised"] for m in metas),
        "wall_recovery_asked_share": share(m["variations"]["wall_recovery"] for m in metas),
        "wall_start_centre_depth_m": [round(float(q), 3) for q in np.percentile(
            [m["start"]["centre_depth_m"] for m in metas if m["wall_recovery_realised"]] or [float("nan")], [0, 50, 100])],
        "wall_start_unavailable": sum(bool(m["notes"].get("wall_start_unavailable")) for m in metas),
        "floor_checks": floor_checks(homes),
        "kick_done_share": share(m["kick_done"] for m in metas),
        "kick_asked_share": share(m["variations"]["kick_deg"] != 0 for m in metas),
        "spot_turn_frame_share_outside": round(sum(m["spot_turn_frames_outside"] for m in metas) / max(frames_total, 1), 5),
        "spot_turn_frame_share_all": round(sum(m["spot_turn_frames"] for m in metas) / max(frames_total, 1), 5),
        "forced_spot_turn_frames": sum(m["forced_spot_turn_frames"] for m in metas),
        "end_fallback_share_of_object": share(m["target"]["end_fallback"] for m in obj),
        "end_error_m_max": max(m["end_error_m"] for m in metas),
        "final_heading_error_deg_max": max((m["final_heading_error_deg"] for m in obj), default=None),
        "geodesic_m": [round(float(q), 2) for q in np.percentile([m["geodesic_m"] for m in metas], [0, 50, 100])],
        "object_start_visible_share": share(m["start"]["visibility"].get("target_share", 0) > 0 for m in obj),
        "object_categories": {c: sum(m["target"]["category"] == c for m in obj) for c in sorted({m["target"]["category"] for m in obj})},
    }
    stats["gate_row2"] = {
        "spot_turn_outside_le_2pct": stats["spot_turn_frame_share_outside"] <= 0.02,
        "detours_ge_25pct": stats["detour_realised_share"] >= 0.25,
        "wall_recoveries_ge_10pct": stats["wall_recovery_share"] >= 0.10,
        "wall_start_depth_in_0.2_0.5": bool(0.2 <= stats["wall_start_centre_depth_m"][0] and stats["wall_start_centre_depth_m"][2] <= 0.5),
        "collision_drives_lt_2pct": stats["drives_with_collision"] < 0.02,
    }
    return stats


def projection(stats: Dict[str, Any], cfg: Dict[str, Any], n_labelled: int, n_unlabelled: int,
               drives_per_s: float) -> Dict[str, Any]:
    """Full-run size and time from the pilot's per-drive numbers (object and spot drives weighted alike)."""
    per = cfg["drives_per_home"]
    drives = n_labelled * sum(per["labelled"].values()) + n_unlabelled * sum(per["unlabelled"].values())
    return {"drives": drives, "frames": round(drives * stats["frames_per_drive"]),
            "drives_gb": round(drives * stats["mb_per_drive"] / 1e3, 1),
            "lmdb_cache_gb_est": round(drives * stats["jpg_mb_per_drive"] / 1e3, 1),
            "hours_at_measured_throughput": round(drives / drives_per_s / 3600, 2), "drives_per_s": drives_per_s}


def markdown(stats: Dict[str, Any], proj: Dict[str, Any]) -> str:
    lines = ["# Phase 2 dataset statistics", ""]
    lines += [f"- **{k}**: {v}" for k, v in stats.items()]
    lines += ["", "## Full-run projection", ""] + [f"- **{k}**: {v}" for k, v in proj.items()]
    return "\n".join(lines) + "\n"


# --- pictures ---------------------------------------------------------------------------------------------------

def drive_files(d: Path):
    traj = pickle.loads((d / "traj_data.pkl").read_bytes())
    with np.load(d / "mapmad_frames.npz") as f:
        fr = {k: f[k] for k in f.files}
    return traj, fr, lm.load_drive_map(d), json.loads((d / "mapmad_meta.json").read_text())


def local_panel(dmap, traj, meta, t: int, rcfg, hfov: float, px: int = 480) -> np.ndarray:
    pose = (float(traj["position"][t, 0]), float(traj["position"][t, 1]), float(traj["yaw"][t]))
    layers = lm.local_map(dmap, t, pose, rcfg["local_map_size"], rcfg["local_map_resolution_m"])
    img = map_viz.render(layers, px)
    map_viz.draw_robot(img, rcfg["local_map_size"], hfov, resolution=rcfg["local_map_resolution_m"])
    end = frames.to_2d(meta["target"]["end_position"])
    map_viz.draw_target(img, frames.to_robot(end[None], pose)[0], rcfg["local_map_size"], rcfg["local_map_resolution_m"])
    return img


def make_video(d: Path, out: Path, rcfg, hfov: float) -> None:
    traj, fr, dmap, meta = drive_files(d)
    n = len(traj["yaw"])
    writer = None
    for t in range(n):
        cam = cv2.cvtColor(cv2.imread(str(d / f"{t}.jpg")), cv2.COLOR_BGR2RGB)
        cam = cv2.resize(cam, (640, 480), interpolation=cv2.INTER_LINEAR)
        body = np.hstack([cam, local_panel(dmap, traj, meta, t, rcfg, hfov)])
        bar = np.zeros((40, body.shape[1], 3), np.uint8)
        phase = PHASES[int(fr["phase"][t])]
        text(bar, f"{meta['drive_id']} {meta['kind']} t={t}/{n - 1} {phase} v={fr['v'][t]:.2f} w={fr['w'][t]:+.2f}"
                  f"{' COLLISION' if fr['collided'][t] else ''}", (8, 26))
        frame = np.vstack([bar, body])
        if writer is None:
            writer = VideoWriter(out, (frame.shape[1], frame.shape[0]), rcfg["video_fps"])
        writer.write(frame)
    writer.close()


def make_overlay(robot: LimoSim, d: Path, t: int, out: Path, cfg, rcfg) -> Dict[str, Any]:
    traj, fr, dmap, meta = drive_files(d)
    hfov, mspec = robot.spec.hfov_deg, cfg["map"]
    robot.place(fr["position_habitat"][t], float(traj["yaw"][t]))
    obs = robot.observe()
    saved = cv2.cvtColor(cv2.imread(str(d / f"{t}.jpg")), cv2.COLOR_BGR2RGB)
    rerender_diff = float(np.abs(saved.astype(int) - obs["rgb"].astype(int)).mean())
    cam = robot.camera_transform()
    h, w = obs["depth"].shape
    # obstacle-band pixels of this frame's depth (heights from the real floor)
    pts = world_points(obs["depth"], hfov, mspec["max_depth_m"], cam[:3, 3], cam[:3, :3])
    _, keep = camera_points(obs["depth"], hfov, mspec["max_depth_m"])
    height = pts[:, 1] - robot.floor_y
    lo, hi = mspec["obstacle_band_m"]
    band = np.zeros(h * w, bool)
    band[keep[(height >= lo) & (height <= hi)]] = True
    img = obs["rgb"].copy()
    img[band.reshape(h, w)] = (0.55 * img[band.reshape(h, w)] + 0.45 * np.array(RED)).astype(np.uint8)
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
    # local-map obstacle cells (NoMaD-side crop from traj_data.pkl's pose) -> picture
    size, res = rcfg["local_map_size"], rcfg["local_map_resolution_m"]
    pose = (float(traj["position"][t, 0]), float(traj["position"][t, 1]), float(traj["yaw"][t]))
    layers = lm.local_map(dmap, t, pose, size, res)
    fwd, left = lm.cell_centres(size, res)
    sel = layers[0] > 0.5
    wx, wy = lm.local_to_world(fwd[sel], left[sel], pose)
    world = np.stack([-wy, np.full(wx.shape, robot.floor_y + 0.10), -wx], axis=1)  # frames.to_3d, vectorised
    uv = project_points(world, cam, w, h, hfov) * 2.0  # picture is shown at 2x
    inside = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
    for x, y in uv[inside]:
        cv2.circle(img, (int(x), int(y)), 3, GREEN, -1, cv2.LINE_AA)
    panel = local_panel(dmap, traj, meta, t, rcfg, hfov)
    body = np.hstack([img, panel])
    bar = np.zeros((40, body.shape[1], 3), np.uint8)
    text(bar, f"{meta['drive_id']} t={t}: red = depth obstacle band (this frame), green = map obstacle cells "
              f"known at t ({int(inside.sum())} in view)", (8, 26), 0.55)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack([bar, body]), cv2.COLOR_RGB2BGR))
    return {"drive": meta["drive_id"], "t": t, "cells_in_view": int(inside.sum()), "rerender_mean_abs_diff": round(rerender_diff, 2),
            "file": str(out)}


def pick_video_drives(metas: List[Dict[str, Any]], n: int, rng: random.Random) -> List[str]:
    """Drives that show the variations: wall recovery + detour, object + kick, spot + detour; then random."""
    rules = [lambda m: m["wall_recovery_realised"] and m["detour_realised"],
             lambda m: m["kind"] == "object" and m["kick_done"],
             lambda m: m["kind"] == "spot" and m["detour_realised"]]
    picked: List[str] = []
    for rule in rules[:n]:
        pool = [m["drive_id"] for m in metas if rule(m) and m["drive_id"] not in picked]
        if pool:
            picked.append(rng.choice(pool))
    rest = [m["drive_id"] for m in metas if m["drive_id"] not in picked]
    return picked + rng.sample(rest, max(0, n - len(picked)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--dataset", choices=["pilot", "full"], default="pilot")
    p.add_argument("--data-name", help="read <mapmad_data>/<name> instead of the dataset folder (debug)")
    p.add_argument("--videos", type=int)
    p.add_argument("--overlays", type=int)
    p.add_argument("--drives-per-s", type=float, help="measured total throughput, for the projection")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    cfg = read_config(args.config)
    rcfg = cfg["report"]
    paths = config.paths()
    data = paths["mapmad_data"] / (args.data_name or cfg["datasets"][args.dataset])
    out = paths["outputs"] / cfg["out_dir"] / (args.data_name or args.dataset) / "report"
    out.mkdir(parents=True, exist_ok=True)
    metas, homes = load_metas(data), load_homes(data)
    stats = statistics(metas, homes, cfg)
    labelled = set(config.labelled_homes(cfg["split"]))
    all_homes = train_homes(cfg["split"])
    proj = projection(stats, cfg, sum(h in labelled for h in all_homes), sum(h not in labelled for h in all_homes),
                      args.drives_per_s or 1.0 / stats["s_per_drive"])
    (out / "stats.json").write_text(json.dumps({"stats": stats, "projection": proj}, indent=1) + "\n")
    (out / "stats.md").write_text(markdown(stats, proj))
    print(json.dumps({"stats": stats, "projection": proj}, indent=1))

    rng = random.Random(args.seed)
    n_videos = rcfg["videos"] if args.videos is None else args.videos
    for drive in pick_video_drives(metas, n_videos, rng) if n_videos else []:
        make_video(data / drive, out / "videos" / f"{drive}.mp4", rcfg, RobotSpec.from_config(config.robot()).hfov_deg)
        print("video", out / "videos" / f"{drive}.mp4", flush=True)
    n_over = rcfg["overlays"] if args.overlays is None else args.overlays
    if n_over:
        picks = [(m["home"], m["drive_id"], rng.randrange(m["frames"])) for m in rng.sample(metas, n_over)]
        results = []
        spec = RobotSpec.from_config(config.robot())
        for home in sorted({h for h, _, _ in picks}):
            with LimoSim(cfg["split"], home, spec, depth=True, gpu=args.gpu) as robot:
                for _, drive, t in [x for x in picks if x[0] == home]:
                    results.append(make_overlay(robot, data / drive, t, out / "overlays" / f"{drive}_{t}.png", cfg, rcfg))
        (out / "overlays.json").write_text(json.dumps(results, indent=1) + "\n")
        print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
