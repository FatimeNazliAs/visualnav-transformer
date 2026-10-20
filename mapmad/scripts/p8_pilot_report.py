"""Phase 8 S-pilot: numbers, full-run estimate, the pilot sheet and 2 videos (block item 6).

    python mapmad/scripts/p8_pilot_report.py --gpu 1                       # inside naz_mapmad_habitat
    python mapmad/scripts/p8_pilot_report.py --gpu 1 --dataset-dir /mapmad_data/<other> --out <dir>   # a fix attempt

Reads the pilot drives (<mapmad_data>/pilot_habitat_mapmad_v2: kept drives + _failed/ + _homes/*.json) and writes to
<outputs>/p8_search_data/pilot/:
- pilot_numbers.json / .md: teacher success and failures by cause, steps, s/drive, GB/drive, predicted full-run GB and
  hours with the cut rule of block item 5, spot turns by type, ms per teacher step, variation shares, phase shares;
- pilot_sheet.png: the numbers + 4 example frames (search, approach, arrived, exploration), each: camera | near window
  (obstacle, explored, frontier, target noisy, target clean, all) | far window (same) | teacher path on the house plan
  (LIMO floor map) with the phase label;
- videos/<drive>.mp4 for 1 search + 1 exploration drive (the same panels for every frame, 8 fps = 2x real time).
The house plan is the LIMO floor map (navmesh) of the drive's storey: loading the train home needs a GPU (ours).
"""

import argparse
import glob
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from mapmad_sim import config
from mapmad_sim.datagen_v2 import MOTIONS, PHASES_V2, SPOT_TURN_TYPES, robot_spec
from mapmad_sim.frames import to_2d
from mapmad_sim.robot import LimoSim
from mapmad_sim.run_layout import read_config, write_text_atomic
from vint_train.mapmad.closed_loop.video import VideoWriter
from vint_train.mapmad.map_v2 import TARGET_FILE, far_window, layers_at, near_window

CONFIG = config.CONFIG_DIR / "p8_datagen.yaml"
TILE = 112  # px per 64-cell window tile
FONT = cv2.FONT_HERSHEY_SIMPLEX
SEARCH_FAILURES = ("timeout", "no_frontier", "blacklist_exhausted", "no_path", "collision", "not_at_viewpoint")


# --- numbers ------------------------------------------------------------------------------------------------------

def load_summaries(dataset: Path) -> List[Dict[str, Any]]:
    return [d for f in sorted(dataset.glob("_homes/*.json")) for d in json.loads(f.read_text())["summaries"]]


def drive_folder(dataset: Path, did: str) -> Optional[Path]:
    for p in (dataset / did, dataset / "_failed" / did):
        if p.exists():
            return p
    return None


def pct(a: float, b: float) -> float:
    return round(100.0 * a / b, 1) if b else float("nan")


def numbers(dataset: Path, cfg: Dict[str, Any], workers: int) -> Dict[str, Any]:
    """Everything block item 6 asks for, from the pilot drives."""
    s = load_summaries(dataset)
    search = [d for d in s if d["kind"] == "search" and d["outcome"] != "no_sample"]
    explore = [d for d in s if d["kind"] == "explore"]
    kept = [d for d in s if d.get("kept")]
    frames = sum(d["frames"] for d in kept)
    spot = Counter()
    for d in kept:
        spot.update(d["spot_turn_frames_by_type"])
    allowed = {"start", "wall_recovery", "face", "retarget"}
    outside = sum(v for k, v in spot.items() if k not in allowed)
    ms = []
    for d in s:
        f = drive_folder(dataset, d["drive_id"])
        if f is not None:
            with np.load(f / "mapmad_frames.npz") as z:
                ms.append(z["teacher_ms"])
    ms = np.concatenate(ms) if ms else np.zeros(1)
    phases = Counter()
    for d in kept:
        phases.update(d["frames_by_phase"])
    by_kind = {k: [d for d in s if d["kind"] == k and "frames" in d] for k in ("search", "explore")}
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    kb_per_frame = sum(d.get("bytes", 0) for d in kept) / max(frames, 1) / 1024.0
    est = estimate(cfg, by_kind, kb_per_frame, workers, s)
    out = {
        "drives": len(s), "search_attempted": len(search), "no_sample": sum(d["outcome"] == "no_sample" for d in s),
        "teacher_success": {"n": sum(d["outcome"] == "success" for d in search), "of": len(search),
                            "pct": pct(sum(d["outcome"] == "success" for d in search), len(search))},
        "search_failures": {c: sum(d["outcome"] == c for d in search) for c in SEARCH_FAILURES},
        "explore": dict(Counter(d["outcome"] for d in explore)),
        "steps": {k: {"median": float(np.median([d["frames"] for d in v])) if v else 0,
                      "p95": float(np.percentile([d["frames"] for d in v], 95)) if v else 0,
                      "mean": round(mean([d["frames"] for d in v]), 1)} for k, v in by_kind.items()},
        "seconds_per_drive": {k: round(mean([d["seconds"] for d in v]), 2) for k, v in by_kind.items()},
        "gb_per_drive": {k: round(mean([d["bytes"] for d in v if d.get("kept")]) / 1e9, 4) for k, v in by_kind.items()},
        "kb_per_frame": round(kb_per_frame, 2),
        "kept_frames": frames,
        "spot_turn_frames_by_type": dict(spot), "spot_turn_share_pct": {k: pct(v, frames) for k, v in spot.items()},
        "outside_allowed_pct": pct(outside, frames),
        "teacher_ms": {"median": round(float(np.median(ms)), 2), "p95": round(float(np.percentile(ms, 95)), 2),
                       "max": round(float(ms.max()), 1)},
        "collision_drives_pct": pct(sum(d["collision_steps"] > 0 for d in kept), len(kept)),
        "detour_realised_pct": pct(sum(d["detour_realised"] for d in kept), len(kept)),
        "wall_recovery_pct": pct(sum(d["wall_recovery_realised"] for d in kept), len(kept)),
        "kick_pct": pct(sum(d["kick_done"] for d in kept), len(kept)),
        "safety_step_frames_pct": pct(sum(d["safety_steps"] for d in kept), frames),
        "phase_share_pct": {k: pct(v, frames) for k, v in phases.items()},
        "categories": dict(Counter(d["category"] for d in search)),
        "estimate": est,
    }
    return out


def estimate(cfg: Dict[str, Any], by_kind: Dict[str, List[Dict]], kb_per_frame: float, workers: int,
             summaries: List[Dict]) -> Dict[str, Any]:
    """Full-run GB and hours from the pilot means; block item 5's cut if a limit is exceeded."""
    n_lab, n_all = 145, 800
    per = cfg["drives_v2"]
    mean_f = {k: np.mean([d["frames"] for d in v if d.get("kept")]) if v else 0 for k, v in by_kind.items()}
    keep_rate = {k: np.mean([bool(d.get("kept")) for d in v]) if v else 0 for k, v in by_kind.items()}
    mean_s = {k: np.mean([d["seconds"] for d in v]) if v else 0 for k, v in by_kind.items()}
    load_s = 6.0  # home load (navmesh rebuild) per home, measured below if possible
    homes = {}
    for d in summaries:
        homes.setdefault(d["home"], 0.0)
        homes[d["home"]] += d.get("seconds", 0.0)

    def predict(search: int, explore: int) -> Tuple[float, float]:
        n_s, n_e = n_lab * search, n_all * explore
        gb = (n_s * keep_rate["search"] * mean_f["search"] + n_e * keep_rate["explore"] * mean_f["explore"]) * kb_per_frame / 1024 ** 2
        hours = (n_s * mean_s["search"] + n_e * mean_s["explore"] + n_all * load_s) / workers / 3600.0
        return round(float(gb), 1), round(float(hours), 2)

    s0, e0 = per["labelled"]["search"], per["labelled"]["explore"]
    gb, h = predict(s0, e0)
    plan = {"counts": {"search": s0, "explore": e0}, "gb": gb, "hours": h, "cut": "none"}
    limit_gb, limit_h = cfg["budget"]["gb"], cfg["budget"]["hours"]
    if gb > limit_gb or h > limit_h:
        e1 = cfg["cut_order"]["explore_min"]
        gb1, h1 = predict(s0, e1)
        plan = {"counts": {"search": s0, "explore": e1}, "gb": gb1, "hours": h1, "cut": "explore -> 4 / home"}
        if gb1 > limit_gb or h1 > limit_h:
            f = min(limit_gb / gb1, limit_h / h1)
            s2, e2 = int(math.floor(s0 * f)), int(math.floor(e1 * f))
            gb2, h2 = predict(s2, e2)
            plan = {"counts": {"search": s2, "explore": e2}, "gb": gb2, "hours": h2,
                    "cut": f"explore -> 4 / home, then both x {f:.3f}"}
    return {"uncut": {"search_per_home": s0, "explore_per_home": e0, "gb": gb, "hours": h}, "plan": plan,
            "workers": workers, "assumed_home_load_s": load_s,
            "mean_frames_kept": {k: round(float(v), 1) for k, v in mean_f.items()},
            "keep_rate": {k: round(float(v), 3) for k, v in keep_rate.items()},
            "mean_seconds_per_drive": {k: round(float(v), 2) for k, v in mean_s.items()}}


def numbers_md(n: Dict[str, Any]) -> str:
    e, ts = n["estimate"], n["teacher_success"]
    lines = ["# Phase 8 pilot — numbers", "",
             f"- Teacher success (search): **{ts['n']}/{ts['of']} = {ts['pct']}%** (gate ≥ 90%)",
             f"- Search failures by cause: {n['search_failures']}; start sampling failed (no_sample): {n['no_sample']}",
             f"- Exploration drives: {n['explore']}",
             f"- Steps per drive: search median {n['steps']['search']['median']} / p95 {n['steps']['search']['p95']}; "
             f"exploration median {n['steps']['explore']['median']} / p95 {n['steps']['explore']['p95']}",
             f"- s/drive (12 workers in parallel): {n['seconds_per_drive']}; GB/drive (kept): {n['gb_per_drive']}; {n['kb_per_frame']} KB/frame",
             f"- Teacher ms/step: median {n['teacher_ms']['median']}, p95 {n['teacher_ms']['p95']} (target ≤ 10, not a gate)",
             f"- Spot turns (frames of kept drives): {n['spot_turn_share_pct']} %; outside the 4 allowed types: **{n['outside_allowed_pct']}%** (≤ 2%)",
             f"- Collision drives: **{n['collision_drives_pct']}%** (< 2%); detours realised {n['detour_realised_pct']}% (≥ 25%); "
             f"wall recoveries {n['wall_recovery_pct']}% (≥ 10%); kicks {n['kick_pct']}%",
             f"- Safety-step frames: {n['safety_step_frames_pct']}%; phase shares {n['phase_share_pct']} %",
             f"- Categories (search): {n['categories']}", "",
             "## Full-run estimate (block item 5)",
             f"- Uncut (40 search / labelled home + 8 exploration / home): {e['uncut']['gb']} GB, {e['uncut']['hours']} h on {e['workers']} workers",
             f"- Plan: **{e['plan']['counts']} per home → {e['plan']['gb']} GB, {e['plan']['hours']} h**; cut: {e['plan']['cut']}",
             f"- From: mean kept frames {e['mean_frames_kept']}, keep rate {e['keep_rate']}, s/drive {e['mean_seconds_per_drive']}, home load {e['assumed_home_load_s']} s"]
    return "\n".join(lines) + "\n"


# --- panels -------------------------------------------------------------------------------------------------------

def label(img: np.ndarray, text: str, y: int = 14, scale: float = 0.42, colour=(255, 255, 255)) -> np.ndarray:
    cv2.putText(img, text, (4, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (4, y), FONT, scale, colour, 1, cv2.LINE_AA)
    return img


def tile(layer: np.ndarray, name: str, colour: Tuple[int, int, int]) -> np.ndarray:
    """One 64 x 64 layer (0..1) as a TILE x TILE picture: black -> colour."""
    v = np.clip(layer, 0.0, 1.0)[..., None] * np.asarray(colour, np.float32)[None, None, :]
    img = cv2.resize(v.astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(img, (0, 0), (TILE - 1, TILE - 1), (90, 90, 90), 1)
    c = TILE // 2  # the robot: between the middle cells, facing up
    cv2.arrowedLine(img, (c, c + 6), (c, c - 6), (255, 255, 255), 1, tipLength=0.5)
    return label(img, name, 12, 0.36)


def composite(win: np.ndarray, target: np.ndarray) -> np.ndarray:
    """All layers in one picture: explored grey, obstacle red, frontier yellow, target green (BGR)."""
    img = np.zeros((64, 64, 3), np.float32)
    img += win[1][..., None] * 110
    img[win[0] > 0] = (40, 40, 220)
    img[win[2] > 0] = (0, 220, 255)
    g = np.clip(target, 0, 1)
    img[..., 1] = np.maximum(img[..., 1], g * 255)
    img = cv2.resize(img.astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    c = TILE // 2
    cv2.arrowedLine(img, (c, c + 6), (c, c - 6), (255, 255, 255), 1, tipLength=0.5)
    return label(img, "all (noisy)", 12, 0.36)


def window_block(noisy: np.ndarray, clean: np.ndarray, title: str) -> np.ndarray:
    """2 x 3 tiles: obstacle, explored, frontier / target noisy, target clean, all."""
    top = np.hstack([tile(noisy[0], "obstacle", (60, 60, 230)), tile(noisy[1], "explored", (200, 200, 200)),
                     tile(noisy[2], "frontier", (0, 220, 255))])
    bottom = np.hstack([tile(noisy[3], "target noisy", (60, 230, 60)), tile(clean[3], "target clean", (60, 230, 60)),
                        composite(noisy, noisy[3])])
    head = label(np.zeros((18, top.shape[1], 3), np.uint8), title, 13, 0.42, (255, 255, 160))
    return np.vstack([head, top, bottom])


class HousePlan:
    """The LIMO floor map of one storey as a picture, with world -> pixel."""

    def __init__(self, robot: LimoSim, floor_height: float, scale_px_per_m: float = 24.0) -> None:
        mpp = 0.05
        grid = np.asarray(robot.pathfinder.get_topdown_view(mpp, floor_height), dtype=bool)
        low = robot.pathfinder.get_bounds()[0]
        self.low_xz, self.mpp, self.k = np.array([low[0], low[2]]), mpp, scale_px_per_m * mpp
        img = np.where(grid[..., None], np.uint8(95), np.uint8(25)).repeat(3, axis=2)
        self.base = cv2.resize(img, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_NEAREST)

    def px(self, xy: np.ndarray) -> Tuple[int, int]:
        """2D (X, Y) -> picture (col, row): rows = Habitat z, cols = Habitat x."""
        x, z = -xy[1], -xy[0]
        return int((x - self.low_xz[0]) / self.mpp * self.k), int((z - self.low_xz[1]) / self.mpp * self.k)


def plan_panel(plan: HousePlan, xy: np.ndarray, yaw: np.ndarray, t: int, fr: Dict[str, np.ndarray],
               meta: Dict[str, Any], explored: np.ndarray, origin: np.ndarray, res: float) -> np.ndarray:
    img = plan.base.copy()
    ii, jj = np.nonzero(explored)  # own map: explored cells, faint
    for i, j in zip(ii[::2], jj[::2]):
        c = plan.px(np.array([origin[0] + (i + 0.5) * res, origin[1] + (j + 0.5) * res]))
        if 0 <= c[1] < img.shape[0] and 0 <= c[0] < img.shape[1]:
            img[c[1], c[0]] = (150, 150, 150)
    for inst in meta.get("instances", []):
        for vp in inst["view_points"]:
            cv2.circle(img, plan.px(to_2d(vp)), 1, (200, 0, 200), -1)
        cv2.drawMarker(img, plan.px(to_2d(inst["centre"])), (255, 120, 0), cv2.MARKER_CROSS, 9, 2)
    pts = np.array([plan.px(p) for p in xy[: t + 1]], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (0, 220, 0), 1, cv2.LINE_AA)
    g = fr["goal_xy"][t]
    if np.isfinite(g).all():
        cv2.line(img, plan.px(xy[t]), plan.px(g), (0, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(img, plan.px(g), 4, (0, 220, 255), 1)
    for b in fr["blacklist_events"]:
        if b[0] <= t:
            cv2.circle(img, plan.px(b[1:3]), max(2, int(b[3] * 24)), (0, 0, 255), 1)
    p = plan.px(xy[t])
    q = plan.px(xy[t] + 0.4 * np.array([math.cos(yaw[t]), math.sin(yaw[t])]))
    cv2.arrowedLine(img, p, q, (255, 255, 255), 2, tipLength=0.4)
    return img


def crop_plan(img: np.ndarray, centre: Tuple[int, int], size: Tuple[int, int]) -> np.ndarray:
    h, w = size
    pad = np.pad(img, ((h, h), (w, w), (0, 0)))
    r, c = centre[1] + h, centre[0] + w
    return pad[r - h // 2:r + h // 2, c - w // 2:c + w // 2]


class DriveView:
    """All panels of one saved drive, frame by frame."""

    def __init__(self, folder: Path, plan: HousePlan) -> None:
        self.folder = folder
        self.meta = json.loads((folder / "mapmad_meta.json").read_text())
        with np.load(folder / "mapmad_map.npz") as f:
            self.dm = {k: f[k] for k in f.files}
        with np.load(folder / TARGET_FILE) as f:
            self.tgt = {k: f[k] for k in f.files}
        with np.load(folder / "mapmad_frames.npz") as f:
            self.fr = {k: f[k] for k in f.files}
        with open(folder / "traj_data.pkl", "rb") as f:
            traj = pickle.load(f)
        self.xy, self.yaw = traj["position"], traj["yaw"]
        self.plan = plan
        self.n = len(self.xy)
        self.has_target = len(self.tgt.get("noisy_cell", [])) > 0

    def panel(self, t: int) -> np.ndarray:
        pose = (float(self.xy[t, 0]), float(self.xy[t, 1]), float(self.yaw[t]))
        origin, res = self.dm["origin"], float(self.dm["resolution"])
        tgt = self.tgt if self.has_target else None
        noisy = layers_at(self.dm, t, tgt, "noisy")
        clean = layers_at(self.dm, t, tgt, "clean")
        near_n, near_c = near_window(noisy, origin, res, pose), near_window(clean, origin, res, pose)
        far_n, far_c = far_window(noisy, origin, res, pose), far_window(clean, origin, res, pose)
        cam = cv2.imread(str(self.folder / f"{t}.jpg"))
        phase = str(self.fr["phase_names"][self.fr["phase"][t]])
        motion = str(self.fr["motion_names"][self.fr["motion"][t]])
        mode = str(self.fr["teacher_mode_names"][self.fr["teacher_mode"][t]])
        label(cam, f"t={t}  {phase.upper()}  ({motion}, teacher: {mode})", 16, 0.45, (160, 255, 255))
        near = window_block(near_n, near_c, "near 6.4 m (10 cm)")
        far = window_block(far_n, far_c, "far 25.6 m (40 cm)")
        h = near.shape[0]
        plan = plan_panel(self.plan, self.xy, self.yaw, t, self.fr, self.meta, noisy["explored"], origin, res)
        plan = crop_plan(plan, self.plan.px(self.xy[t]), (h, 330))
        label(plan, "teacher path on the house plan", 14, 0.42, (255, 255, 160))
        cam = np.vstack([cam, np.zeros((h - cam.shape[0], cam.shape[1], 3), np.uint8)])
        return np.hstack([cam, near, far, plan])

    def title(self) -> str:
        m = self.meta
        what = f"search '{m['category']}' {m['start_geodesic_m']} m" if m["kind"] == "search" else "exploration"
        return f"{m['drive_id']}  {what}  -> {m['outcome']} in {m['frames']} steps"


def load_plan(cfg: Dict[str, Any], home: str, meta: Dict[str, Any], gpu: int, cache: Dict[str, HousePlan]) -> HousePlan:
    if home not in cache:
        with LimoSim(cfg["split"], home, robot_spec(cfg), semantic=False, depth=False, gpu=gpu) as robot:
            cache[home] = HousePlan(robot, float(meta["start"]["position"][1]))
    return cache[home]


def write_video(view: DriveView, path: Path, fps: int) -> None:
    """Every frame's panels as an H.264 mp4 (closed_loop.video.VideoWriter: imageio-ffmpeg's bundled ffmpeg)."""
    first = view.panel(0)
    h, w = first.shape[0] + 24, first.shape[1] + first.shape[1] % 8
    h += (8 - h % 8) % 8
    writer = VideoWriter(path, (w, h), fps, crf=30, faststart=True)
    try:
        for t in range(view.n):
            p = view.panel(t)
            frame = np.zeros((h, w, 3), np.uint8)
            label(frame, view.title(), 16, 0.5, (255, 255, 255))
            frame[24:24 + p.shape[0], :p.shape[1]] = p
            writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()


def pick(summaries: List[Dict[str, Any]], kind: str, rng: np.random.Generator) -> Dict[str, Any]:
    """A kept drive of this kind with a typical length (between the 25th and 75th percentile), drawn with the seed."""
    pool = [d for d in summaries if d["kind"] == kind and d.get("kept")]
    lo, hi = np.percentile([d["frames"] for d in pool], [25, 75])
    mid = [d for d in pool if lo <= d["frames"] <= hi and (kind != "search" or d["frames_by_phase"].get("search", 0) > 20)]
    mid = mid or pool
    return mid[int(rng.integers(len(mid)))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()
    cfg = read_config(CONFIG)
    paths = config.paths()
    dataset = args.dataset_dir or paths["mapmad_data"] / cfg["datasets"]["pilot"]
    out = args.out or paths["outputs"] / cfg["out_dir"] / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    n = numbers(dataset, cfg, args.workers)
    write_text_atomic(out / "pilot_numbers.json", json.dumps(n, indent=1) + "\n")
    write_text_atomic(out / "pilot_numbers.md", numbers_md(n))
    print(numbers_md(n))

    summaries = load_summaries(dataset)
    rng = np.random.default_rng(args.seed)
    chosen = {"search": pick(summaries, "search", rng), "explore": pick(summaries, "explore", rng)}
    cache: Dict[str, HousePlan] = {}
    views = {}
    for kind, d in chosen.items():
        folder = drive_folder(dataset, d["drive_id"])
        meta = json.loads((folder / "mapmad_meta.json").read_text())
        views[kind] = DriveView(folder, load_plan(cfg, d["home"], meta, args.gpu, cache))
        write_video(views[kind], out / "videos" / f"{d['drive_id']}.mp4", args.fps)
    # the sheet: numbers + 4 example frames
    sv, ev = views["search"], views["explore"]
    ph = sv.fr["phase"]
    t_search = int(np.nonzero(ph == PHASES_V2.index("search"))[0][len(np.nonzero(ph == 0)[0]) // 2])
    appr = np.nonzero(ph == PHASES_V2.index("approach"))[0]
    t_seen = int(appr[0]) if appr.size else sv.n - 1
    rows = [(sv, t_search, "search: exploring frontiers"), (sv, t_seen, "approach: target just seen"),
            (sv, sv.n - 1, "arrived: facing the object"), (ev, ev.n // 2, "exploration drive")]
    panels = []
    for v, t, what in rows:
        pnl = v.panel(t)
        head = label(np.zeros((22, pnl.shape[1], 3), np.uint8), f"{what} -- {v.title()}", 15, 0.48, (255, 255, 255))
        panels.append(np.vstack([head, pnl]))
    width = max(p_.shape[1] for p_ in panels)
    text = np.zeros((22 * 16, width, 3), np.uint8)
    for k, line in enumerate(numbers_md(n).splitlines()):
        label(text, line.replace("**", "")[:190], 18 + 21 * k, 0.45)
    sheet = np.vstack([text] + [np.pad(p_, ((0, 4), (0, width - p_.shape[1]), (0, 0))) for p_ in panels])
    cv2.imwrite(str(out / "pilot_sheet.png"), sheet)
    write_text_atomic(out / "pilot_media.json", json.dumps({"videos": {k: d["drive_id"] for k, d in chosen.items()},
                                                            "sheet_frames": [(v.meta["drive_id"], t, w) for v, t, w in rows],
                                                            "seed": args.seed}, indent=1) + "\n")


if __name__ == "__main__":
    main()
