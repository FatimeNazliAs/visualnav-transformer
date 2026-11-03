"""Phase 14b-1 step 14b1.1: replay the Phase 14a practice runs pose by pose and write, for every logged detection and
true target instance, the storey cells of every projection variant (mapmad_sim/mark_projection.py). The 14a stop
evidence holds cells, not masks or depth (kickoff §3 fallback "masks / depth"), so the replay renders depth at the
logged poses (habitat-sim, the arm's camera) and re-runs YOLOE on the rendered pictures. Nothing here changes a run.
Inside naz_mapmad_habitat, from /app/visualnav-transformer (one YOLOE sidecar on --recog-port per shard, started by
the caller; MAPMAD_SIDECAR_KEY set):

    python mapmad/scripts/p14b_replay.py --config mapmad/configs/p14b1_marks.yaml --subdir runs/baseline \
        --arms M21b M21bh N F VF --gpu G --recog-port P --shard k/n [--episodes E ...]

Replay = p12_videos.replay_states at the evidence's full-precision poses (replay_exact; sidecar_server.SimService in-process, search_wire.arm_wire settings, the logged
noise seed, robot.move_to on every logged pose), with the arm's own policy blocks removed (F's teacher, VF's value map,
the rule value, logging): they never change the map or the recogniser's input. Per frame t:
- the picture's YOLOE detections (masks, raw scores, boxes) are matched to the 14a evidence rows of frame t: same count
  and every |raw - logged raw| <= 1e-4 -> row by row (the replay check, kickoff §3); otherwise each logged row takes
  the unused replay detection with box IoU >= 0.9 and |raw diff| <= 1e-4, else the fallback "F1-box" (the logged box
  filled with pixels within 0.3 m of its median depth; its base cells stay the logged ones);
- true instances = video_events.true_boxes of the semantic picture (= the evidence's true rows), mask = their pixels;
- every row's cells per variant (base must equal the 14a evidence cells: counted).
Output <out_dir>/replay/<subdir name>/<arm>/<run>.npz (MarkTable arrays of det_* and true_* rows + per-row checks) and
<run>.json (counts, check numbers, seconds); resumable (a run with both files is skipped).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapmad_sim import stop_evidence  # noqa: E402
from mapmad_sim import video_events as ve  # noqa: E402
from mapmad_sim.arm_table import ArmTable  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.mark_projection import (FrameGeometry, MarkTable, ProjSpec, box_fill_mask,  # noqa: E402
                                        project_mask)
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic  # noqa: E402
from mapmad_sim.search_wire import RECOG_HOST, arm_wire  # noqa: E402
from mapmad_sim.sidecar_server import SimService, limo_factory  # noqa: E402

RAW_TOL, MATCH_IOU = 1e-4, 0.9
STRIP = ("teacher", "vf", "rule", "log_dets", "log_evidence")


class ObsKeeper:
    """A robot factory around limo_factory whose robots keep their last observation (`last`)."""

    def __init__(self, gpu: int) -> None:
        self.base = limo_factory(gpu)
        self.last: Dict[str, np.ndarray] = {}

    def __call__(self, *a: Any, **k: Any) -> Any:
        robot = self.base(*a, **k)
        observe = robot.observe

        def keep() -> Dict[str, np.ndarray]:
            self.last = observe()
            return self.last

        robot.observe = keep
        return robot


def no_windows() -> None:
    """The replay never reads the network's map windows: skip building them (frontier clustering per frame)."""
    from mapmad_sim.search_v2 import SearchMap

    SearchMap.windows = lambda self, pose: None


def replay_exact(service: SimService, log_path: Path, episode: Dict[str, Any], sv: Dict[str, Any],
                 camera: Dict[str, Any], pose: np.ndarray):
    """p12_videos.replay_states with the evidence's full-precision 2D poses (the step log rounds positions and yaw to
    6 digits, enough to move YOLOE scores by up to ~1e-3): frame t+1 is rendered at to_3d(pose[t+1][:2], logged
    height), yaw pose[t+1][2]; the move record and everything else as replay_states. Yields t = 0 .. steps."""
    from mapmad_bridge import steplog
    from mapmad_sim.frames import to_3d
    from mapmad_sim.robot import Move

    log = steplog.read(log_path)
    head, steps = log[0], steplog.steps(log)
    service.reset(episode, camera["hfov_deg"], False, 0.05,
                  {"search_v2": dict(sv, noise_seed=head["noise_seed"], record=False)}, camera)
    yield 0
    for t, s in enumerate(steps):
        x, y, yaw = (float(v) for v in pose[t + 1])
        service.robot.move_to(to_3d((x, y), exact_height(service.robot, (x, y), float(s["position"][1]), t + 1)), yaw)
        move = Move(v=float(s["v"]), w=float(s["w"]), travelled_m=float(s["travelled_m"]), short_m=float(s["short_m"]),
                    refused_substeps=0, collided=bool(s["collided"]))
        service.search.after_move(move, service.pose2d(), service.robot.spec.dt)
        service.observation()
        yield t + 1


HEIGHT_MISSES: List[int] = []  # frames of the current run whose height is the log's rounded one


def exact_height(robot: Any, xy: Tuple[float, float], logged: float, t: int = -1) -> float:
    """The navmesh height at the exact (X, Y) (pathfinder.snap_point, float32 as the drive's try_step) when it agrees
    with the logged 6-digit height (|diff| <= 1e-6); else the logged height (frame noted in HEIGHT_MISSES: on slopes /
    steps snap_point and try_step can differ, and such a frame may not match the original byte for byte)."""
    from mapmad_sim.frames import to_3d

    snap = np.asarray(robot.pathfinder.snap_point(to_3d(xy, logged).astype(np.float32)), np.float64)
    if np.all(np.isfinite(snap)) and abs(snap[1] - logged) <= 1e-6:
        return float(snap[1])
    HEIGHT_MISSES.append(t)
    return logged


def match_rows(logged_raw: np.ndarray, logged_box: np.ndarray, dets: Sequence[Any]) -> Tuple[List[Optional[int]], float]:
    """Logged evidence rows of one frame -> replay detection index (None = F1-box fallback), and the frame's max |raw
    diff| when matched row by row (NaN otherwise). Module docstring."""
    n = len(logged_raw)
    raws = np.array([float(d.raw_score) for d in dets])
    if len(dets) == n and (n == 0 or np.max(np.abs(raws - logged_raw)) <= RAW_TOL):
        return list(range(n)), float(np.max(np.abs(raws - logged_raw))) if n else 0.0
    used, out = set(), []
    for k in range(n):
        best, best_iou = None, MATCH_IOU
        for j, d in enumerate(dets):
            if j in used or d.box is None or abs(raws[j] - logged_raw[k]) > RAW_TOL:
                continue
            i = ve.iou(list(d.box), list(logged_box[k]))
            if i >= best_iou:
                best, best_iou = j, i
        if best is not None:
            used.add(best)
        out.append(best)
    return out, float("nan")


def replay_run(service: SimService, keeper: ObsKeeper, layout: RunLayout, table: ArmTable, arm: str,
               episode: Dict[str, Any], rid: str, recog_port: int, spec: ProjSpec) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """One run's projection arrays + its check record (module docstring)."""
    from vint_train.mapmad.config import load_config

    t0 = time.time()
    tau = float(table.choices()["tau"])
    w = arm_wire(table, arm, load_config, (RECOG_HOST, recog_port), tau, ("unused", 0))
    sv = {k: v for k, v in w.search_v2.items() if k not in STRIP}
    ev = stop_evidence.load(layout.root / "evidence" / arm / f"{rid}.npz")
    det_rows = {int(t): [] for t in np.unique(ev["det_t"])}
    for k, t in enumerate(ev["det_t"].tolist()):
        det_rows[t].append(k)
    true_rows: Dict[int, List[int]] = {}
    for k, t in enumerate(ev["true_t"].tolist()):
        true_rows.setdefault(int(t), []).append(k)
    ids = [int(i) for i in episode["target"]["category_ids"]]
    dets_tab, true_tab = MarkTable("det", spec), MarkTable("true", spec)
    chk = {"frames": 0, "frames_rowwise": 0, "max_raw_diff": 0.0, "fallback_rows": 0, "matched_rows": 0,
           "det_base_same": 0, "true_rows_same_count": 0, "true_frames_count_mismatch": 0, "true_base_same": 0,
           "pose_max_diff": 0.0}
    HEIGHT_MISSES.clear()
    for t in replay_exact(service, layout.log(arm, rid), episode, sv, w.camera, ev["pose"]):
        obs, s, robot = keeper.last, service.search, service.robot
        pose = service.pose2d()
        chk["pose_max_diff"] = max(chk["pose_max_diff"], float(np.max(np.abs(np.subtract(pose[:2], ev["pose"][t][:2])))))
        chk["frames"] += 1
        geo = FrameGeometry.build(obs["depth"], s.last_cells.pixel_cell, robot.spec.hfov_deg, robot.camera_transform(),
                                  s.spec.noise.max_depth_m)
        rows = det_rows.get(t, [])
        dets = list(s.last_dets)
        match, diff = match_rows(ev["det_raw"][rows], ev["det_box"][rows], dets)
        if not np.isnan(diff):
            chk["frames_rowwise"] += 1
            chk["max_raw_diff"] = max(chk["max_raw_diff"], diff)
        for k, j in zip(rows, match):
            logged = ev["det_cells"][ev["det_cells_ptr"][k]:ev["det_cells_ptr"][k + 1]]
            box = ev["det_box"][k]
            if j is None or dets[j].mask is None:
                marks = project_mask(box_fill_mask(box, obs["depth"]), box, geo, spec)
                marks.cells["base"] = np.asarray(logged, np.int32)
                chk["fallback_rows"] += 1
                dets_tab.add(marks, raw_replay=float("nan"), fallback=1.0, base_same=1.0)
                chk["det_base_same"] += 1
                continue
            d = dets[j]
            marks = project_mask(d.mask, d.box, geo, spec)
            same = np.array_equal(marks.cells["base"], np.sort(logged))
            chk["matched_rows"] += 1
            chk["det_base_same"] += int(same)
            dets_tab.add(marks, raw_replay=float(d.raw_score), fallback=0.0, base_same=float(same))
        trues = ve.true_boxes(obs["semantic"], ids)
        logged_t = true_rows.get(t, [])
        if len(trues) != len(logged_t):
            chk["true_frames_count_mismatch"] += 1
        sem = obs["semantic"]
        for q, k in enumerate(logged_t):
            iid = int(ev["true_id"][k])
            logged = ev["true_cells"][ev["true_cells_ptr"][k]:ev["true_cells_ptr"][k + 1]]
            marks = project_mask(sem == iid, ev["true_box"][k], geo, spec)
            same = np.array_equal(marks.cells["base"], np.sort(logged))
            chk["true_base_same"] += int(same)
            chk["true_rows_same_count"] += 1
            true_tab.add(marks, base_same=float(same))
    if chk["frames"] != len(ev["pose"]):
        raise RuntimeError(f"{arm} {rid}: replayed {chk['frames']} frames, evidence has {len(ev['pose'])}")
    if len(dets_tab) != len(ev["det_t"]) or len(true_tab) != len(ev["true_t"]):
        raise RuntimeError(f"{arm} {rid}: rows {len(dets_tab)}/{len(true_tab)} vs evidence {len(ev['det_t'])}/{len(ev['true_t'])}")
    arrays = {**dets_tab.arrays(), **true_tab.arrays(), "version": np.array(1), "variants": np.array(spec.names())}
    chk.update(det_rows=len(dets_tab), true_rows=len(true_tab), seconds=round(time.time() - t0, 1),
               height_log_frames=len(HEIGHT_MISSES))
    return arrays, chk


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def original_gpus(root: Path) -> Dict[Tuple[str, str], int]:
    """(arm, run id) -> the GPU that drove it: the shard of its finished line in progress.<arms>.shard<s>.log
    (run_p13_pairs.sh: shard = 10 x GPU + pair); a run named in two GPUs' logs is left out (replayed anywhere)."""
    import re

    seen: Dict[Tuple[str, str], set] = {}
    for f in root.glob("progress.*.log"):
        gpu = int(re.search(r"shard(\d+)\.log$", f.name).group(1)) // 10
        for line in f.read_text().splitlines():
            m = re.match(r"\S+ \d+/\d+ (\S+) (\S+__s\d+): (.*)", line)
            if m and "steps" in m.group(3):
                seen.setdefault((m.group(1), m.group(2)), set()).add(gpu)
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def jobs_of(layout: RunLayout, arms: Sequence[str], episodes: Optional[Sequence[str]],
            orig_gpu: Optional[int] = None) -> List[Tuple[str, str]]:
    """(arm, run id) of every logged run of the arms (optionally only these episodes; only runs first driven on
    orig_gpu -- same-GPU replays are byte-identical, kickoff §3), sorted."""
    gpus = original_gpus(layout.root) if orig_gpu is not None else {}
    out = []
    for arm in arms:
        for f in layout.logs(arm):
            rid = f.name[:-len(".jsonl")]
            if episodes and parse_run_id(rid)[0] not in episodes:
                continue
            if orig_gpu is not None and gpus.get((arm, rid), 0) != orig_gpu:
                continue
            out.append((arm, rid))
    return out


def practice_episodes(layout: RunLayout) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name in layout.cfg["sets"]["practice"]["parts"]:
        eps, _ = load_episodes(layout.base / layout.cfg["sets"][name]["folder"] / "episodes.json")
        out.update({e["episode_id"]: e for e in eps})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--recog-port", type=int, required=True)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--orig-gpu", type=int, help="only runs first driven on this GPU (unknown -> GPU 0)")
    p.add_argument("--episodes", nargs="*")
    p.add_argument("--limit", type=int, default=0, help="at most this many runs of this shard (DRY)")
    p.add_argument("--out", type=Path, help="output folder (default <out_dir>/replay/<subdir name>; DRY: p14b1_dry)")
    args = p.parse_args()
    runs = RunLayout.load_runs(args.config, subdir=args.subdir)
    out_base = args.out or RunLayout.load(args.config).base / "replay" / Path(args.subdir).name
    table = ArmTable.from_layout(runs)
    eps = practice_episodes(runs)
    spec = ProjSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in (runs.cfg.get("projection") or {}).items()})
    k, n = (int(x) for x in args.shard.split("/"))
    jobs = [j for i, j in enumerate(jobs_of(runs, args.arms, args.episodes, args.orig_gpu)) if i % n == k]
    if args.limit:
        jobs = jobs[:args.limit]
    save_run_info(out_base, {"args": {a: str(v) for a, v in vars(args).items()}, "spec": spec.__dict__}, 0,
                  name=f"run_info.shard{k}.json")
    first_gpu = original_gpus(runs.root)
    no_windows()
    keeper = ObsKeeper(args.gpu)
    service = SimService(keeper)
    done = 0
    try:
        for arm, rid in jobs:
            npz, rec = out_base / arm / f"{rid}.npz", out_base / arm / f"{rid}.json"
            if npz.exists() and rec.exists():
                continue
            arrays, chk = replay_run(service, keeper, runs, table, arm, eps[parse_run_id(rid)[0]], rid,
                                     args.recog_port, spec)
            save_npz(npz, arrays)
            write_text_atomic(rec, json.dumps({"arm": arm, "run": rid, "gpu": args.gpu,
                                                    "orig_gpu": first_gpu.get((arm, rid)), **chk}) + "\n")
            done += 1
            print(json.dumps({"arm": arm, "run": rid, "seconds": chk["seconds"], "fallback": chk["fallback_rows"],
                              "max_raw_diff": chk["max_raw_diff"]}), flush=True)
    finally:
        service.shutdown()
    print(json.dumps({"shard": args.shard, "replayed": done, "jobs": len(jobs)}), flush=True)


if __name__ == "__main__":
    main()
