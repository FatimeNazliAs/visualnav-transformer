"""Phase 10 frontier parity (block item 3, chain step 1): on the frames the NoMaD side checked
(`frontier_cache check` -> frontier_cache_check.json: 20 random frames, >= 10 drives, 5 held-out), three computations
of the teacher's frontier at frame t must be equal cell for cell:

1. Habitat side, the live teacher code: explorer.teacher_frontier on the OwnMap the generator built at frame t (saved
   first-seen map as known at t; bumped cells replayed from mapmad_frames.npz in log order through
   depth_map.FirstSeenMap.cells, only events logged before frame t; planning window around the pose and the goal
   committed at t - 1; TeacherSpec of p8_datagen_v3a.yaml);
2. NoMaD side, the shared function map_v2.teacher_frontier_at;
3. the per-drive cache (frontier_cache.frontier_from_cache), as the loader reads it.
Also: the cluster count of (1) equals the one the generator logged at t when the teacher explored.

    python mapmad/scripts/p10_frontier_parity.py --frames-json /outputs/mapmad/p10_v2_train/checks/frontier_cache_check.json \
        --cache-dir /mapmad_data/p10_frontier_cache/habitat_mapmad_v2 --out /outputs/mapmad/p10_v2_train/checks/frontier_parity.json

Inside naz_mapmad_habitat, CPU only. Exit code 3 on any difference.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from mapmad_sim import config
from mapmad_sim.depth_map import FirstSeenMap, MapSpec
from mapmad_sim.explorer import OwnMap, TeacherSpec, teacher_frontier
from mapmad_sim.run_layout import read_config, write_text_atomic
from vint_train.mapmad.frontier_cache import EXPLORING_MODES, frontier_from_cache, load_cache, load_frames
from vint_train.mapmad.local_map import known_at, load_drive_map
from vint_train.mapmad.map_v2 import teacher_frontier_clusters

CONFIG = config.CONFIG_DIR / "p8_datagen_v3a.yaml"


def live_frontier(drive_dir: Path, t: int, spec: TeacherSpec) -> np.ndarray:
    """(1) of the module docstring."""
    m = load_drive_map(drive_dir)
    frames = load_frames(str(drive_dir))
    with np.load(drive_dir / "mapmad_frames.npz") as z:
        goals = z["goal_xy"]
    with open(drive_dir / "traj_data.pkl", "rb") as f:
        traj = pickle.load(f)
    shape = m["obstacle_first_seen"].shape
    fmap = FirstSeenMap(MapSpec(resolution=float(m["resolution"])), m["origin"], shape)
    bumped = np.zeros(shape, dtype=bool)
    for k, x, y in frames["bump_events"]:
        if k < t:
            flat, inside = fmap.cells(np.array([[x, y]]))
            if inside[0]:
                bumped.flat[flat[0]] = True
    own = OwnMap(known_at(m["obstacle_first_seen"], t), known_at(m["explored_first_seen"], t), tuple(m["origin"]),
                 float(m["resolution"]), bumped, None)
    pose = (float(traj["position"][t][0]), float(traj["position"][t][1]), float(traj["yaw"][t]))
    committed = tuple(goals[t - 1]) if t > 0 and np.isfinite(goals[t - 1]).all() else None
    return teacher_frontier(own, pose, committed, spec)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--frames-json", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG)
    args = p.parse_args()
    spec = TeacherSpec.from_config(read_config(args.config)["teacher"])
    data = config.paths()["mapmad_data"] / "habitat_mapmad_v2"
    picked = json.loads(args.frames_json.read_text())["frames"]
    rows = []
    for r in picked:
        d, t = data / r["drive"], int(r["t"])
        drive_map, frames = load_drive_map(d), load_frames(str(d))
        live = live_frontier(d, t, spec)
        labels = teacher_frontier_clusters(drive_map, frames["bump_events"], t, spec.min_cluster)
        cached = frontier_from_cache(load_cache(str(args.cache_dir / f"{r['drive']}.npz")), t)
        names = [str(x) for x in frames["teacher_mode_names"]]
        exploring = names[int(frames["teacher_mode"][t])] in EXPLORING_MODES
        row = {"drive": r["drive"], "t": t, "cells": int(live.sum()), "live_eq_shared": bool(np.array_equal(live, labels > 0)),
               "live_eq_cache": bool(np.array_equal(live, cached)), "teacher_exploring": exploring}
        if exploring:  # the generator's logged cluster count at t
            row["count_eq_teacher_log"] = int(labels.max()) == int(frames["frontier_clusters"][t])
        rows.append(row)
    ok =all(r["live_eq_shared"] and r["live_eq_cache"] and r.get("count_eq_teacher_log", True) for r in rows)
    res = {"frames": rows, "n_frames": len(rows), "n_drives": len({r["drive"] for r in rows}), "all_equal": ok,
           "spec": {"min_cluster": spec.min_cluster, "frontier_ignore_trajectory_m": spec.frontier_ignore_trajectory_m,
                    "bbox_margin_m": spec.bbox_margin_m}, "config": str(args.config)}
    write_text_atomic(args.out, json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}))
    if not ok:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
