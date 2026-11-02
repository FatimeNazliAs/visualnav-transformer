"""Phase 14a step 14.3: wrong place vs bad driving (mapmad_sim/diagnosis.py), report only. Inside naz_mapmad_habitat
(CPU only), from /app/visualnav-transformer:

    python mapmad/scripts/p14_diagnosis.py practice --config mapmad/configs/p14a_finish.yaml --subdir runs/baseline \
        --arms M21b M21bh F --workers 16
    python mapmad/scripts/p14_diagnosis.py dagger --config mapmad/configs/p14a_finish.yaml --workers 16
    python mapmad/scripts/p14_diagnosis.py report --config mapmad/configs/p14a_finish.yaml

practice: per run, the own map at every search-phase step from the stop evidence (first-seen maps <= t, cells bumped
before t; stop_evidence.py), the teacher asked at every step with its state carried (frozen blacklist, nothing seen:
dagger_labels' rule), 20-step windows classified -> diagnosis/practice/<arm>/<run>.json.
dagger: the already labelled rollouts of Phase 11 (map-DAgger, `habitat_mapmad_v2_dagger`) and Phase 13 (hidden-DAgger,
`habitat_mapmad_v2_dagger_hidden`), HM3D train homes: the teacher's goal and path length per frame from labels.npz,
its direction from the label's look-ahead waypoints; search phase = frames before the first seen event ->
diagnosis/dagger/<set>/<drive>.json.
report: shares per arm / set -> diagnosis/diagnosis.md + .json.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

from mapmad_bridge import steplog  # noqa: E402
from mapmad_sim import config  # noqa: E402
from mapmad_sim.dagger_labels import frozen_blacklist  # noqa: E402
from mapmad_sim.diagnosis import (CLASSES, TeacherView, Window, classify, cluster_of_point, path_direction,  # noqa: E402
                                  robot_cluster, summarise, windows_of)
from mapmad_sim.explorer import OwnMap, TeacherSpec, TeacherState, teacher_step  # noqa: E402
from mapmad_sim.run_layout import RunLayout, read_config, write_text_atomic  # noqa: E402
from mapmad_sim.stop_evidence import load as load_evidence  # noqa: E402
from vint_train.mapmad.local_map import known_at  # noqa: E402
from vint_train.mapmad.map_v2 import bumped_cells, frontier_clusters  # noqa: E402

DAGGER_SETS = {"map_dagger_p11": "habitat_mapmad_v2_dagger", "hidden_dagger_p13": "habitat_mapmad_v2_dagger_hidden"}


def own_map(first_obst: np.ndarray, first_expl: np.ndarray, shape, origin, res, bumps, t: int) -> OwnMap:
    ob = known_at(first_obst, t).reshape(shape)
    ex = known_at(first_expl, t).reshape(shape)
    return OwnMap(ob, ex, tuple(origin), res, bumped_cells(shape, origin, res, bumps, t))


def clusters_at(own: OwnMap, min_cluster: int) -> np.ndarray:
    return frontier_clusters(own.obstacle | own.bumped, own.explored, min_cluster)


def window_of(t0: int, t1: int, pos: np.ndarray, coll: np.ndarray, views: Dict[int, TeacherView],
              labels: np.ndarray, origin, res) -> Window:
    tv0 = views[t0]
    p0, p1 = (float(pos[t0, 0]), float(pos[t0, 1])), (float(pos[t1, 0]), float(pos[t1, 1]))
    tc = cluster_of_point(labels, origin, res, tv0.goal) if tv0.goal is not None else None
    return Window(t0, t1, (p1[0] - p0[0], p1[1] - p0[1]), int(coll[t0:t1].sum()), tv0, views[t1], p0, p1,
                  robot_cluster(labels, origin, res, p0, p1), tc)


# --- practice runs ---------------------------------------------------------------------------------------------------
def practice_run(job: Tuple[str, str, str, Dict[str, Any], int]) -> Tuple[str, Optional[str]]:
    log_file, ev_file, out_file, teacher_cfg, min_cluster = job
    try:
        log = steplog.read(Path(log_file))
        ev = load_evidence(Path(ev_file))
        result = log[-1]
        steps = [x for x in log if x["kind"] == "step"]
        coll = np.array([0] + [int(s["collided"]) for s in steps])  # coll[t] = the move into frame t
        coll = np.concatenate([coll[1:], [0]])  # coll[t] = the move after frame t (window [t0, t1) counts t0..t1-1)
        end = int(result["first_in_view_step"]) if result.get("first_in_view_step") is not None else len(steps)
        shape, origin, res = tuple(int(v) for v in ev["shape"]), tuple(float(v) for v in ev["origin"]), float(ev["res"])
        spec = frozen_blacklist(TeacherSpec.from_config(teacher_cfg))
        state = TeacherState()
        views: Dict[int, TeacherView] = {}
        wins = windows_of(end)
        need = {t for w in wins for t in w}
        lab_at: Dict[int, np.ndarray] = {}
        for t in range(0, (max(need) if need else -1) + 1):
            own = own_map(ev["first_obstacle"], ev["first_explored"], shape, origin, res, ev["bumps"], t)
            pose = tuple(float(v) for v in ev["pose"][t])
            dec, state = teacher_step(own, pose, state, [], spec, searching=True)
            if t in need:
                fail = dec.goal is None
                views[t] = TeacherView(None if fail else (float(dec.goal[0]), float(dec.goal[1])),
                                       float(dec.path_length_m), None if fail else path_direction(dec.path, pose))
                lab_at[t] = clusters_at(own, min_cluster)
        out = []
        for t0, t1 in wins:
            w = window_of(t0, t1, ev["pose"], coll, views, lab_at[t0], origin, res)
            out.append({"t0": t0, "t1": t1, "class": classify(w), "collisions": w.collisions,
                        "move_m": round(math.hypot(*w.move), 3)})
        write_text_atomic(Path(out_file), json.dumps({"run": Path(log_file).stem, "search_end": end,
                                                      "windows": out}) + "\n")
        return out_file, None
    except Exception as exc:
        return out_file, f"{type(exc).__name__}: {exc}"


def cmd_practice(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    jobs = []
    for arm in args.arms:
        for f in sorted((layout.root / "logs" / arm).glob("*.jsonl")):
            rid = f.name[:-len(".jsonl")]
            out = layout.base / "diagnosis" / "practice" / arm / f"{rid}.json"
            if out.exists():
                continue
            jobs.append((str(f), str(layout.root / "evidence" / arm / f"{rid}.npz"), str(out), dg["teacher"],
                         int(dg["teacher"]["min_cluster"])))
    run_jobs(practice_run, jobs, args.workers)


# --- DAgger rollouts ---------------------------------------------------------------------------------------------
def dagger_drive(job: Tuple[str, str, int]) -> Tuple[str, Optional[str]]:
    folder, out_file, min_cluster = job
    try:
        import pickle

        d = Path(folder)
        with np.load(d / "labels.npz") as z:
            lab = {k: z[k] for k in z.files}
        with np.load(d / "mapmad_map.npz") as z:
            dm = {k: z[k] for k in z.files}
        with np.load(d / "mapmad_frames.npz") as z:
            fr = {k: z[k] for k in z.files}
        with open(d / "traj_data.pkl", "rb") as f:
            traj = pickle.load(f)
        pos = np.asarray(traj["position"], np.float64)[:, :2]
        yaw = np.asarray(traj["yaw"], np.float64)
        n = len(pos)
        seen = np.asarray(fr["seen_events"], np.int64).reshape(-1, 2)
        end = int(seen[:, 0].min()) if len(seen) else n - 1
        coll = np.asarray(fr["collided"], bool).astype(int)
        origin, res = tuple(float(v) for v in dm["origin"]), float(dm["resolution"])
        shape = dm["obstacle_first_seen"].shape
        views: Dict[int, TeacherView] = {}
        wins = windows_of(end)
        for t in {t for w in wins for t in w}:
            g = lab["goal_xy"][t]
            wp = lab["waypoints_m"][t]
            direction = None
            if np.isfinite(wp).all():
                fx, fy = float(wp[-1, 0]), float(wp[-1, 1])
                c, s = math.cos(yaw[t]), math.sin(yaw[t])
                vx, vy = fx * c - fy * s, fx * s + fy * c
                nrm = math.hypot(vx, vy)
                direction = (vx / nrm, vy / nrm) if nrm > 1e-6 else None
            views[t] = TeacherView(None if not np.isfinite(g).all() else (float(g[0]), float(g[1])),
                                   float(lab["path_length_m"][t]), direction)
        out = []
        for t0, t1 in wins:
            own = OwnMap(known_at(dm["obstacle_first_seen"], t0), known_at(dm["explored_first_seen"], t0), origin, res,
                         bumped_cells(shape, origin, res, fr["bump_events"], t0))
            w = window_of(t0, t1, pos, coll, views, clusters_at(own, min_cluster), origin, res)
            out.append({"t0": t0, "t1": t1, "class": classify(w), "collisions": w.collisions})
        write_text_atomic(Path(out_file), json.dumps({"drive": d.name, "search_end": end, "windows": out}) + "\n")
        return out_file, None
    except Exception as exc:
        return out_file, f"{type(exc).__name__}: {exc}"


def cmd_dagger(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    root = config.paths()["mapmad_data"]
    jobs = []
    for name, folder in DAGGER_SETS.items():
        drives = sorted(p for p in (root / folder).iterdir() if (p / "labels.npz").exists())
        if args.sample:
            idx = np.random.default_rng(14301).choice(len(drives), min(args.sample, len(drives)), replace=False)
            drives = [drives[i] for i in sorted(idx)]
        for d in drives:
            out = layout.base / "diagnosis" / "dagger" / name / f"{d.name}.json"
            if not out.exists():
                jobs.append((str(d), str(out), int(dg["teacher"]["min_cluster"])))
    run_jobs(dagger_drive, jobs, args.workers)


def run_jobs(fn, jobs, workers: int) -> None:
    print(f"{len(jobs)} jobs on {workers} workers", flush=True)
    errors = []
    with Pool(workers) as pool:
        for k, (out, err) in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            if err:
                errors.append((out, err))
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} failed: {errors[:3]}")


# --- report ------------------------------------------------------------------------------------------------------------
def cmd_report(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    base = layout.base / "diagnosis"
    groups: Dict[str, List[Path]] = {}
    for sub in ("practice", "dagger"):
        for g in sorted((base / sub).glob("*")) if (base / sub).exists() else []:
            groups[f"{sub}/{g.name}"] = sorted(g.glob("*.json"))
    res = {}
    lines = ["# 14.3 diagnosis: wrong place vs bad driving (report only)", "",
             "Search phase only (before the first true sight); 20-step windows; teacher = explorer.teacher_step with "
             "a frozen blacklist on the robot's own map (practice runs) or the DAgger labels (rollouts). Classes: "
             "agree, wrong_place, bad_driving, other (+ other_still: moved < 0.1 m; other_teacher: teacher without a "
             "goal). Arm F is the check (≈ all agree).", "",
             "| group | runs | windows | search steps | " + " | ".join(CLASSES) + " |",
             "|---|---|---|---|" + "---|" * len(CLASSES)]
    for name, files in groups.items():
        pairs = []
        for f in files:
            d = json.loads(f.read_text())
            pairs += [(w["class"], w["t1"] - w["t0"]) for w in d["windows"]]
        s = summarise(pairs)
        res[name] = dict(s, runs=len(files))
        lines.append(f"| {name} | {len(files)} | {s['windows']} | {s['steps']} | " +
                     " | ".join(f"{100 * (s['window_share'][c] or 0):.1f}% ({100 * (s['step_share'][c] or 0):.1f}%)"
                                for c in CLASSES) + " |")
    lines += ["", "Cells: share of windows (share of search steps)."]
    write_text_atomic(base / "diagnosis.md", "\n".join(lines) + "\n")
    write_text_atomic(base / "diagnosis.json", json.dumps(res, indent=1) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["practice", "dagger", "report"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="runs/baseline")
    p.add_argument("--arms", nargs="+", default=["M21b", "M21bh", "F"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--sample", type=int, default=0, help="dagger: a seeded sample of this many drives per set (0 = all)")
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    dg = read_config(config.CONFIG_DIR / layout.cfg["datagen"])
    {"practice": cmd_practice, "dagger": cmd_dagger, "report": cmd_report}[args.cmd](args, layout, dg)


if __name__ == "__main__":
    main()
