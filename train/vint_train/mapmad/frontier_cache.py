"""Per-drive cache of the teacher's frontier (Phase 10 block item 3): `map_v2.teacher_frontier_at` for every frame of
a Phase 8 drive, stored as its changes, so the loader rebuilds frame t in milliseconds instead of re-clustering the
storey map.

One file per drive, `<cache_dir>/<drive>.npz`:
- `cell`, `frame`, `value`: every change of the frontier mask (flat cell index, frame, 1 = became frontier,
  0 = stopped being one), frames non-decreasing -- the format of the target layer's sparse updates, so
  `map_v2.target_at` rebuilds frame t (the last change of each cell with frame <= t);
- `n_clusters`: the number of frontier clusters per frame; `teacher_clusters`: the count the live teacher logged
  (mapmad_frames.npz `frontier_clusters`); `checked`: the frames where the teacher explored (modes search, explore,
  no_frontier, blacklist_exhausted), on which the two must be equal -- an independent check of the shared function
  against the generator, frame by frame; `shape`.
A frame is re-clustered only when its map or bump cells changed (a cell first seen at t or a bump logged at t - 1).

Commands (inside naz_mapmad, from /app/visualnav-transformer/train; CPU only):

    python -m vint_train.mapmad.frontier_cache build --config config/mapmad_v2.yaml --workers 20 \
        --out-json /outputs/mapmad/p10_v2_train/checks/frontier_cache_build.json
    python -m vint_train.mapmad.frontier_cache check --config config/mapmad_v2.yaml --frames 20 \
        --out-json /outputs/mapmad/p10_v2_train/checks/frontier_cache_check.json

`build` covers the train and test drives of `habitat_mapmad_v2` (excluded drives skipped) and is resumable (a drive
file is written atomically and skipped when present). Exit code 3 if any checked frame differs from the teacher.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vint_train.mapmad.local_map import load_drive_map
from vint_train.mapmad.map_v2 import target_at, teacher_frontier_clusters

FRAMES_FILE = "mapmad_frames.npz"
EXPLORING_MODES = ("search", "explore", "no_frontier", "blacklist_exhausted")  # teacher_step ran rule 1 (frontier)


def load_frames(drive_dir: str) -> Dict[str, np.ndarray]:
    with np.load(os.path.join(drive_dir, FRAMES_FILE)) as z:
        return {k: z[k] for k in ("bump_events", "frontier_clusters", "teacher_mode", "teacher_mode_names")}


def changed_frames(drive_map: Dict[str, np.ndarray], bump_events: np.ndarray, n_frames: int) -> np.ndarray:
    """Bool per frame: does the teacher's frontier input change at t (a cell first seen at t, a bump logged at t-1)?
    Frame 0 always counts."""
    out = np.zeros(n_frames, dtype=bool)
    out[0] = True
    for key in ("obstacle_first_seen", "explored_first_seen"):
        fs = drive_map[key].ravel()
        fs = fs[(fs >= 0) & (fs < n_frames)]
        out[np.unique(fs)] = True
    bumps = np.asarray(bump_events, dtype=np.float64).reshape(-1, 3)[:, 0].astype(np.int64) + 1
    out[bumps[bumps < n_frames]] = True
    return out


def build_drive(drive_dir: str, min_cluster: int = 4) -> Dict[str, np.ndarray]:
    """The cache arrays of one drive (module docstring)."""
    drive_map = load_drive_map(drive_dir)
    frames = load_frames(drive_dir)
    n = len(frames["frontier_clusters"])
    shape = drive_map["obstacle_first_seen"].shape
    redo = changed_frames(drive_map, frames["bump_events"], n)
    prev = np.zeros(shape[0] * shape[1], dtype=bool)
    cells: List[np.ndarray] = []
    frame_ids: List[np.ndarray] = []
    values: List[np.ndarray] = []
    counts = np.zeros(n, dtype=np.int32)
    count = 0
    for t in range(n):
        if redo[t]:
            labels = teacher_frontier_clusters(drive_map, frames["bump_events"], t, min_cluster)
            count = int(labels.max())
            cur = labels.ravel() > 0
            diff = np.flatnonzero(cur != prev)
            if diff.size:
                cells.append(diff.astype(np.int32))
                frame_ids.append(np.full(diff.size, t, dtype=np.int32))
                values.append(cur[diff].astype(np.uint8))
            prev = cur
        counts[t] = count
    names = [str(x) for x in frames["teacher_mode_names"]]
    checked = np.isin(frames["teacher_mode"], [names.index(m) for m in EXPLORING_MODES])
    cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
    return {"cell": cat(cells, np.int32), "frame": cat(frame_ids, np.int32), "value": cat(values, np.uint8),
            "n_clusters": counts, "teacher_clusters": frames["frontier_clusters"].astype(np.int32),
            "checked": checked, "shape": np.asarray(shape, dtype=np.int32)}


def mismatched_frames(cache: Dict[str, np.ndarray]) -> np.ndarray:
    """Frames where the teacher explored and its logged cluster count differs from the shared function's."""
    return np.flatnonzero(cache["checked"] & (cache["n_clusters"] != cache["teacher_clusters"]))


def cache_path(cache_dir: str, drive: str) -> str:
    return os.path.join(cache_dir, f"{drive}.npz")


def load_cache(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    out["value_f"] = out["value"].astype(np.float32)  # target_at's value type
    return out


def frontier_from_cache(cache: Dict[str, np.ndarray], t: int) -> np.ndarray:
    """The teacher's frontier at frame t (bool, storey shape) rebuilt from a drive's cache."""
    value = cache["value_f"] if "value_f" in cache else cache["value"].astype(np.float32)
    return target_at(cache["cell"], cache["frame"], value, t, cache["shape"]) > 0


def write_atomic_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _build_one(args) -> Dict[str, Any]:
    data_folder, cache_dir, drive, min_cluster = args
    path = cache_path(cache_dir, drive)
    t0 = time.time()
    if os.path.exists(path):
        cache = load_cache(path)
        skipped = True
    else:
        cache = build_drive(os.path.join(data_folder, drive), min_cluster)
        write_atomic_npz(path, cache)
        skipped = False
    bad = mismatched_frames(cache)
    return {"drive": drive, "frames": int(len(cache["n_clusters"])), "changes": int(len(cache["cell"])),
            "checked": int(cache["checked"].sum()), "mismatched": int(len(bad)), "first_bad": bad[:5].tolist(),
            "bytes": os.path.getsize(path), "seconds": time.time() - t0, "skipped": skipped}


def split_drives(dataset_cfg: Dict[str, Any], splits: Sequence[str] = ("train", "test")) -> List[str]:
    """Drive names of the given splits, minus `exclude_drives`, sorted and unique."""
    from vint_train.mapmad.dataset import read_drive_list

    excluded = read_drive_list(dataset_cfg.get("exclude_drives"))
    names = set()
    for split in splits:
        with open(os.path.join(dataset_cfg[split], "traj_names.txt")) as f:
            names |= {x.strip() for x in f if x.strip()}
    return sorted(names - excluded)


def build_all(data_folder: str, cache_dir: str, drives: Sequence[str], workers: int, min_cluster: int = 4,
              progress_every: int = 500) -> Dict[str, Any]:
    """Cache every drive (resumable); returns totals + the drives with mismatched frames."""
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    jobs = [(data_folder, cache_dir, d, min_cluster) for d in drives]
    with Pool(workers) as pool:
        for k, row in enumerate(pool.imap_unordered(_build_one, jobs, chunksize=4)):
            rows.append(row)
            if progress_every and (k + 1) % progress_every == 0:
                print(f"[frontier_cache] {k + 1}/{len(jobs)} drives, {time.time() - t0:.0f} s", flush=True)
    bad = sorted((r for r in rows if r["mismatched"]), key=lambda r: r["drive"])
    return {"drives": len(rows), "frames": sum(r["frames"] for r in rows), "checked_frames": sum(r["checked"] for r in rows),
            "mismatched_frames": sum(r["mismatched"] for r in rows), "mismatched_drives": bad,
            "changes": sum(r["changes"] for r in rows), "bytes": sum(r["bytes"] for r in rows),
            "skipped": sum(r["skipped"] for r in rows), "wall_seconds": time.time() - t0,
            "cpu_seconds": sum(r["seconds"] for r in rows), "workers": workers, "cache_dir": cache_dir}


def check_frames(data_folder: str, cache_dir: str, drives: Sequence[str], n_frames: int, seed: int,
                 min_cluster: int = 4, need_drives: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """n_frames random (drive, frame) pairs, one per drive where possible (need_drives first): the cache rebuild must
    equal teacher_frontier_at exactly."""
    from vint_train.mapmad.map_v2 import teacher_frontier_at

    rng = np.random.default_rng(seed)
    first = list(need_drives or [])
    rest = [d for d in drives if d not in set(first)]
    order = first + [rest[i] for i in rng.permutation(len(rest))]
    out = []
    for drive in order[:n_frames]:
        d = os.path.join(data_folder, drive)
        cache = load_cache(cache_path(cache_dir, drive))
        t = int(rng.integers(len(cache["n_clusters"])))
        live = teacher_frontier_at(load_drive_map(d), load_frames(d)["bump_events"], t, min_cluster)
        got = frontier_from_cache(cache, t)
        out.append({"drive": drive, "t": t, "cells": int(live.sum()), "equal": bool(np.array_equal(live, got))})
    return {"frames": out, "all_equal": all(r["equal"] for r in out), "n_drives": len({r["drive"] for r in out}),
            "seed": seed}


def _config_dataset(config_path: str) -> Dict[str, Any]:
    from vint_train.mapmad.config import load_config

    cfg = load_config(config_path)
    return cfg, cfg["datasets"]["habitat_mapmad_v2"]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--out-json", required=True)
    p.add_argument("--limit", type=int, default=None, help="first N drives only (pre-checks)")
    p = sub.add_parser("check")
    p.add_argument("--config", required=True)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--out-json", required=True)
    args = parser.parse_args(argv)
    cfg, d = _config_dataset(args.config)
    cache_dir = cfg["map"]["frontier_cache"]
    drives = split_drives(d)
    if args.cmd == "build":
        res = build_all(d["data_folder"], cache_dir, drives[:args.limit] if args.limit else drives, args.workers)
    else:
        test = split_drives(d, ("test",))
        held_out = [test[i] for i in np.random.default_rng(args.seed).permutation(len(test))[:5]]
        res = check_frames(d["data_folder"], cache_dir, drives, args.frames, args.seed, need_drives=held_out)
        res["held_out_drives"] = held_out
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("frames", "mismatched_drives")}, indent=1))
    if args.cmd == "build" and res["mismatched_frames"]:
        print(f"MISMATCH: {res['mismatched_frames']} checked frames differ from the teacher's cluster count", flush=True)
        raise SystemExit(3)
    if args.cmd == "check" and not res["all_equal"]:
        print("MISMATCH: cache != teacher_frontier_at", flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
