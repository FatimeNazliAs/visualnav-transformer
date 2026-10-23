"""Phase 11 DAgger labels (block item 3; mapmad_sim/dagger_labels.py). Inside naz_mapmad_habitat (CPU only), from
/app/visualnav-transformer:

    python mapmad/scripts/p11_labels.py check --workers 16    # label check on 50 Phase 8 test-split search drives
    python mapmad/scripts/p11_labels.py label --workers 24    # labels.npz into every rollout drive
    python mapmad/scripts/p11_labels.py splits                # DAgger split folders (train / test by home)
    python mapmad/scripts/p11_labels.py sheet                 # 10 random labelled states, one picture each

`check`: seeded choice of `label_check_drives` search drives of the habitat_mapmad_v2 TEST split (after
exclude_drives); every frame with 40 future frames where the teacher does not fail: look-ahead label (the drive's own
teacher spec) vs the real future waypoints; error = mean of the 8 waypoint distances (m); result
<out>/labels/label_check.json (median, p95, per phase, relabelled goal = logged goal share). Exit code 3 if the median
is above `label_check_max_median_m` (automatic stop, block item 9).
`label`: every drive of <mapmad_data>/<rollouts.data_folder> (--ready <summaries>: only finished runs; a partial pass
during the rollouts) (blacklist counter frozen) -> <drive>/labels.npz
(resumable: written atomically, skipped when present); <out>/labels/label_stats.json (labelled / teacher-fail /
collision shares, overall and per category). Exit code 3 if teacher-fail states are > max_teacher_fail_share of the
visited states (automatic stop).
`splits`: <mapmad_data>/<rollouts.splits>/{train,test}/traj_names.txt (test = the rollout set's dev-home episodes) +
sha256 of each list. `sheet`: <out>/sheets/label_sheet.png + label_sheet.json (seeded states: storey map around the
robot, the student's next 40 steps, the teacher's label and path, the camera frame).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Tuple  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mapmad_sim import config  # noqa: E402
from mapmad_sim.dagger_labels import (LABELS_FILE, REASONS, LabelSpec, frozen_blacklist, label_check_drive,  # noqa: E402
                                      label_drive, label_poses, load_drive, teacher_states, waypoints_of)
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.explorer import TeacherSpec  # noqa: E402
from mapmad_sim.expert import PursuitSpec  # noqa: E402
from mapmad_sim.frames import to_robot  # noqa: E402
from mapmad_sim.robot import RobotSpec  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, write_text_atomic  # noqa: E402
from mapmad_sim.search_wire import datagen_config  # noqa: E402
from vint_train.mapmad.local_map import known_at  # noqa: E402

CONFIG = config.CONFIG_DIR / "p11_dagger.yaml"
_CTX: Dict[str, Any] = {}


def specs(layout: RunLayout) -> Tuple[TeacherSpec, PursuitSpec, LabelSpec]:
    """Teacher (Phase 8 option A), motion (Phase 8 expert + LIMO limits) and label numbers."""
    dg = datagen_config(layout.cfg)
    r = RobotSpec.from_config(config.robot())
    ro, mo = layout.cfg["rollouts"], dg["motion"]
    return (TeacherSpec.from_config(dg["teacher"]), PursuitSpec.from_config(dg["expert"], r.max_v, r.max_w, r.dt),
            LabelSpec(int(ro["label_steps"]), int(ro["label_every"]), int(ro["collision_skip_steps"]),
                      float(mo["retarget_turn_deg"]), float(dg["teacher"]["face_tolerance_deg"]),
                      bool(ro["label_replan"]), ro["skip_order"] == "collision_first"))


def _init(ctx: Dict[str, Any]) -> None:
    _CTX.update(ctx)


def _check_one(folder: str) -> Dict[str, Any]:
    return label_check_drive(Path(folder), _CTX["teacher"], _CTX["pursuit"], _CTX["spec"])


def _label_one(folder: str) -> Dict[str, Any]:
    out = Path(folder) / LABELS_FILE
    t0 = time.time()
    stale = False
    if out.exists():
        with np.load(out) as z:
            stale = "teacher_fail" not in z.files  # written by an older labeller: redone
    attempts = 0
    if not out.exists() or stale:
        while True:  # labels are deterministic: a retry gives the same arrays (an intermittent numpy error under
            attempts += 1  # the 24-worker load hit 1 of 1,448 drives on 2026-10-23; the same drive labels fine alone)
            try:
                arrays = label_drive(Path(folder), _CTX["teacher"], _CTX["pursuit"], _CTX["spec"], freeze_blacklist=True)
                break
            except ValueError as exc:
                if attempts >= 3:
                    raise
                print(f"[labels] {Path(folder).name}: attempt {attempts} failed ({exc}); retrying", flush=True)
        tmp = out.with_name(".labels.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
    with np.load(out) as z:
        reason, tf, coll = z["reason"], z["teacher_fail"], z["collision_streak"]
    meta = json.loads((Path(folder) / "mapmad_meta.json").read_text())
    return {"drive": Path(folder).name, "category": meta["category"], "frames": int(len(reason)),
            "counts": {**{REASONS[i]: int((reason == i).sum()) for i in range(len(REASONS))},
                       "teacher_fail_any": int(tf.sum()), "collision_any": int(coll.sum()),
                       "overlap": int((tf & coll).sum())},
            "seconds": round(time.time() - t0, 2), "attempts": attempts}


def cmd_check(args, layout: RunLayout) -> None:
    teacher, pursuit, spec = specs(layout)
    ro = layout.cfg["rollouts"]
    split = Path("/mapmad_data/splits/habitat_mapmad_v2/test/traj_names.txt")
    excluded = set(Path("/mapmad_data/splits/habitat_mapmad_v2/collision_drives.txt").read_text().split())
    names = sorted(x for x in split.read_text().split() if "-search-" in x and x not in excluded)
    pick = sorted(names[i] for i in np.random.default_rng(int(ro["label_check_seed"])).choice(
        len(names), int(ro["label_check_drives"]), replace=False))
    data = config.paths()["mapmad_data"] / "habitat_mapmad_v2"
    t0 = time.time()
    with Pool(args.workers, _init, ({"teacher": teacher, "pursuit": pursuit, "spec": spec},)) as pool:
        rows = pool.map(_check_one, [str(data / d) for d in pick])
    err = np.concatenate([np.asarray(r["errors"]) for r in rows])
    wp = np.concatenate([np.asarray(r["per_waypoint"]).reshape(-1, 8) for r in rows])
    phase = np.concatenate([np.asarray(r["phases"]) for r in rows])
    same = np.concatenate([np.asarray(r["same_goal"]) for r in rows])
    limit, limit1 = float(ro["label_check_max_median_m"]), float(ro["label_check_max_wp1_median_m"])
    res = {"drives": pick, "states": int(len(err)), "median_m": float(np.median(err)),
           "p95_m": float(np.percentile(err, 95)), "mean_m": float(err.mean()),
           "per_waypoint_median_m": np.median(wp, axis=0).round(4).tolist(),
           "per_waypoint_p95_m": np.percentile(wp, 95, axis=0).round(4).tolist(),
           "wp1_median_m": float(np.median(wp[:, 0])), "limit_wp1_median_m": limit1,
           "per_phase": {p: {"states": int((phase == p).sum()), "median_m": float(np.median(err[phase == p])),
                             "p95_m": float(np.percentile(err[phase == p], 95))} for p in sorted(set(phase.tolist()))},
           "relabelled_goal_equal_share": float(same.mean()), "limit_m": limit, "replan": spec.replan,
           "pass": bool(np.median(err) <= limit and np.median(wp[:, 0]) <= limit1),
           "seconds": round(time.time() - t0, 1)}
    path = layout.base / "labels" / "label_check.json"
    write_text_atomic(path, json.dumps(res, indent=1) + "\n")
    save_run_info(path.parent / "_runs", {**vars(args), "rollouts": ro}, int(ro["label_check_seed"]),
                  name="run_info_check.json")
    print(json.dumps({k: v for k, v in res.items() if k != "drives"}, indent=1))
    if not res["pass"]:
        raise SystemExit(3)


def rollout_drives(layout: RunLayout) -> List[Path]:
    data = config.paths()["mapmad_data"] / layout.cfg["rollouts"]["data_folder"]
    return sorted(p for p in data.iterdir() if p.is_dir() and p.name.startswith("p11r-"))


def cmd_label(args, layout: RunLayout) -> None:
    teacher, pursuit, spec = specs(layout)
    ro = layout.cfg["rollouts"]
    drives = rollout_drives(layout)
    if args.ready:  # during the rollouts: only drives whose run is finished (summary written after the drive)
        drives = [d for d in drives if (args.ready / f"{d.name}__s{ro['seed_index']}.json").exists()]
    t0 = time.time()
    rows = []
    with Pool(args.workers, _init, ({"teacher": teacher, "pursuit": pursuit, "spec": spec},)) as pool:
        for k, row in enumerate(pool.imap_unordered(_label_one, [str(d) for d in drives], chunksize=2)):
            rows.append(row)
            if (k + 1) % 100 == 0:
                print(f"[labels] {k + 1}/{len(drives)} drives, {time.time() - t0:.0f} s", flush=True)
    total = Counter()
    per_cat: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        total.update(r["counts"])
        per_cat[r["category"]].update(r["counts"])
    frames = sum(total[k] for k in REASONS)
    fails = sum(total[k] for k in ("no_frontier", "blacklist_exhausted", "no_path"))

    def shares(c: Counter) -> Dict[str, Any]:
        n = sum(c[k] for k in REASONS)
        return {"frames": n, **{k: round(c[k] / n, 4) if n else 0.0 for k in REASONS},
                "teacher_fail": round(sum(c[k] for k in ("no_frontier", "blacklist_exhausted", "no_path")) / n, 4) if n else 0.0,
                **{k: round(c[k] / n, 4) if n else 0.0 for k in ("teacher_fail_any", "collision_any", "overlap")}}

    res = {"drives": len(rows), "frames": frames, "overall": shares(total),
           "per_category": {c: shares(v) for c, v in sorted(per_cat.items())},
           "teacher_fail_share": fails / frames if frames else 0.0,
           "limit_teacher_fail_share": float(ro["max_teacher_fail_share"]), "replan": spec.replan,
           "skip_order": ro["skip_order"],
           "both_orders": {"collision_first": {"teacher_fail": round(fails / frames, 4) if frames else 0.0,
                                               "collision": round(total["collision_any"] / frames, 4) if frames else 0.0},
                           "teacher_fail_first": {"teacher_fail": round(total["teacher_fail_any"] / frames, 4) if frames else 0.0,
                                                  "collision": round((total["collision_any"] - total["overlap"]) / frames, 4) if frames else 0.0},
                           "overlap": round(total["overlap"] / frames, 4) if frames else 0.0},
           "frozen_blacklist_window": frozen_blacklist(teacher).blacklist_window_steps,
           "wall_seconds": round(time.time() - t0, 1), "cpu_seconds": round(sum(r["seconds"] for r in rows), 1)}
    res["pass"] = res["teacher_fail_share"] <= res["limit_teacher_fail_share"]
    path = layout.base / "labels" / "label_stats.json"
    write_text_atomic(path, json.dumps(res, indent=1) + "\n")
    save_run_info(path.parent / "_runs", {**vars(args), "rollouts": ro, "spec": spec.__dict__}, 0,
                  name="run_info_label.json")
    print(json.dumps({k: v for k, v in res.items() if k != "per_category"}, indent=1))
    if not res["pass"]:
        raise SystemExit(3)


def cmd_splits(args, layout: RunLayout) -> None:
    ro = layout.cfg["rollouts"]
    eps, fp = load_episodes(layout.base / layout.cfg["sets"]["rollout"]["folder"] / "episodes.json")
    role = {e["episode_id"]: e["split_role"] for e in eps}
    have = {p.name for p in rollout_drives(layout) if (p / LABELS_FILE).exists()}
    root = config.paths()["mapmad_data"] / ro["splits"]
    out = {"rollout_set_fingerprint": fp, "drives_with_labels": len(have), "missing": sorted(set(role) - have)[:20]}
    for split in ("train", "test"):
        names = sorted(d for d in have if role.get(d) == split)
        f = root / split / "traj_names.txt"
        f.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(f, "".join(n + "\n" for n in names))
        out[split] = {"drives": len(names), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    write_text_atomic(root / "splits.json", json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def sheet_tile(folder: Path, t: int, teacher: TeacherSpec, pursuit: PursuitSpec, spec: LabelSpec) -> Tuple[np.ndarray, Dict]:
    """One state: 8 m x 8 m of the robot's own storey map at t (unknown dark, explored light, obstacle red, frontier
    yellow), robot (blue arrow), the student's real next 40 steps (magenta), the teacher's label waypoints (green
    dots) and planned path (cyan); right: the camera frame t."""
    drive = load_drive(folder)
    for k, pose, dec, seen, own, state in teacher_states(drive, frozen_blacklist(teacher)):
        if k == t:
            break
    lab = np.load(folder / LABELS_FILE)
    dm, traj = drive["map"], drive["traj"]
    res, origin = float(dm["resolution"]), np.asarray(dm["origin"], float)
    px, half = 6, 40  # 6 px per 10 cm cell, 8 m window
    ci, cj = int((pose[0] - origin[0]) / res), int((pose[1] - origin[1]) / res)
    obst, expl = known_at(dm["obstacle_first_seen"], t), known_at(dm["explored_first_seen"], t)
    from vint_train.mapmad.map_v2 import frontier_clusters
    front = frontier_clusters(obst | own.bumped, expl) > 0
    img = np.full((2 * half, 2 * half, 3), 60, np.uint8)
    for a in range(2 * half):
        for b in range(2 * half):
            i, j = ci - half + a, cj - half + b
            if 0 <= i < obst.shape[0] and 0 <= j < obst.shape[1]:
                img[a, b] = (40, 40, 220) if obst[i, j] else (0, 220, 255) if front[i, j] else (200, 200, 200) if expl[i, j] else (60, 60, 60)
    img = cv2.resize(img, (2 * half * px, 2 * half * px), interpolation=cv2.INTER_NEAREST)

    def to_px(xy):  # world (X, Y) -> image (col = Y axis, row = X axis) like the storey grid
        return int(((xy[1] - origin[1]) / res - (cj - half)) * px), int(((xy[0] - origin[0]) / res - (ci - half)) * px)

    path = dec.path
    for a, b in zip(path[:-1], path[1:]):
        cv2.line(img, to_px(a), to_px(b), (255, 255, 0), 2)
    fut = traj["position"][t:t + 41]
    for a, b in zip(fut[:-1], fut[1:]):
        cv2.line(img, to_px(a), to_px(b), (255, 0, 255), 2)
    c, s_ = math.cos(pose[2]), math.sin(pose[2])
    for f, l in lab["waypoints_m"][t]:
        if np.isfinite(f):
            cv2.circle(img, to_px((pose[0] + f * c - l * s_, pose[1] + f * s_ + l * c)), 5, (0, 200, 0), -1)
    tip = (pose[0] + 0.4 * c, pose[1] + 0.4 * s_)
    cv2.arrowedLine(img, to_px(pose), to_px(tip), (255, 80, 0), 3, tipLength=0.4)
    cam = cv2.imread(str(folder / f"{t}.jpg"))
    cam = cv2.resize(cam, (img.shape[0] * cam.shape[1] // cam.shape[0], img.shape[0]))
    tile = np.hstack([img, cam])
    reason = REASONS[int(lab["reason"][t])]
    text = f"{folder.name} t={t} teacher={dec.mode} label={reason}"
    cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    real = to_robot(traj["position"][t:t + 41:5][1:], pose) if t + 40 < len(traj["position"]) else None
    return tile, {"drive": folder.name, "t": t, "teacher_mode": dec.mode, "reason": reason,
                  "label_m": np.round(lab["waypoints_m"][t], 3).tolist(),
                  "student_next_m": None if real is None else np.round(real, 3).tolist()}


def cmd_sheet(args, layout: RunLayout) -> None:
    teacher, pursuit, spec = specs(layout)
    ro = layout.cfg["rollouts"]
    drives = [d for d in rollout_drives(layout) if (d / LABELS_FILE).exists()]
    rng = np.random.default_rng(int(ro["sheet_seed"]))
    tiles, rows = [], []
    while len(tiles) < int(ro["sheet_states"]):
        d = drives[int(rng.integers(len(drives)))]
        with np.load(d / LABELS_FILE) as z:
            valid = np.flatnonzero(z["valid"])
        if not len(valid):
            continue
        tile, row = sheet_tile(d, int(valid[int(rng.integers(len(valid)))]), teacher, pursuit, spec)
        tiles.append(tile)
        rows.append(row)
    out = layout.base / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(str(out / "label_sheet.png"), grid)
    write_text_atomic(out / "label_sheet.json", json.dumps({"seed": int(ro["sheet_seed"]), "states": rows,
                                                             "legend": sheet_tile.__doc__}, indent=1) + "\n")
    print(json.dumps(rows, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["check", "label", "splits", "sheet"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--ready", type=Path, help="label: only drives with a run summary in this folder (partial pass)")
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    {"check": cmd_check, "label": cmd_label, "splits": cmd_splits, "sheet": cmd_sheet}[args.cmd](args, layout)


if __name__ == "__main__":
    main()
