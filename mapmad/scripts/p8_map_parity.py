"""Phase 8.1 / G8 row 1: the online map (built step by step, as in a closed loop) = the generator map rebuilt from the
saved arrays, on 2 drives in dev homes (val-drive homes; never an exam home).

    python mapmad/scripts/p8_map_parity.py --gpu 1      # inside naz_mapmad_habitat
    python mapmad/scripts/p8_map_parity.py --gpu 1 --config mapmad/configs/p8_datagen_v2.yaml --out-name map_parity_v2.json

For each drive, at every step t the live state (first-seen map, clean + noisy target fusion, pose) is cut into the near
and far windows (map_v2, layers obstacle / explored / frontier / target) and hashed. After the drive is written, the
same windows are rebuilt from mapmad_map.npz + mapmad_target.npz + traj_data.pkl alone (map_v2.layers_at) and must be
byte-identical at every step. With the rules v2 trajectory ring (teacher.frontier_ignore_trajectory_m > 0) both
frontiers are checked: the plain one and the one the teacher used (ring from traj_data.pkl's positions). Drives go to <mapmad_data>/p8_checks_parity/ (check data, not a dataset); the result to
<outputs>/p8_search_data/checks/map_parity.json.
"""

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from mapmad_sim import config, home_splits
from mapmad_sim.datagen_v2 import HomeGeneratorV2, drive_id
from mapmad_sim.run_layout import read_config, write_text_atomic
from vint_train.mapmad.map_v2 import LAYERS_V2, TARGET_FILE, far_window, frontier, layers_at, near_window

CONFIG = config.CONFIG_DIR / "p8_datagen.yaml"
DRIVES = [("00746", "search", 0), ("00440", "explore", 0)]  # dev homes (Phase 1 homes, val-drive split)


def window_hash(layers: Dict[str, np.ndarray], origin, res: float, pose) -> str:
    near = near_window(layers, origin, res, pose)
    far = far_window(layers, origin, res, pose)
    return hashlib.sha256(near.tobytes() + far.tobytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--out-name", default="map_parity.json")
    args = p.parse_args()
    cfg = read_config(args.config)
    ring = float(cfg["teacher"].get("frontier_ignore_trajectory_m", 0.0))
    variants = [("plain", 0.0)] + ([("ring", ring)] if ring > 0 else [])
    paths = config.paths()
    out = paths["mapmad_data"] / "p8_checks_parity"
    if out.exists():  # our own check output, rebuilt every time
        shutil.rmtree(out)
    homes = home_splits.train_homes(cfg["split"])
    report: List[Dict] = []
    for prefix5, kind, n in DRIVES:
        home = home_splits.by_prefix(homes, [prefix5])[0]
        did = drive_id("p8x", home, kind, n)
        live: Dict[str, List[str]] = {f"{key}_{v}": [] for key in ("noisy", "clean") for v, _ in variants}
        poses: List = []

        def on_step(t, fmap, fusions, pose):
            first = {"obstacle_first_seen": fmap.first_seen["obstacle"].reshape(fmap.shape),
                     "explored_first_seen": fmap.first_seen["explored"].reshape(fmap.shape),
                     "origin": np.asarray(fmap.origin), "resolution": np.asarray(fmap.spec.resolution)}
            poses.append(pose)
            xy = np.asarray(poses)[:, :2]
            for v, m in variants:
                base = layers_at(first, t, positions=xy, visited_m=m)
                for key in ("noisy", "clean"):
                    lay = dict(base, target=fusions[key].grid().copy())
                    live[f"{key}_{v}"].append(window_hash(lay, fmap.origin, fmap.spec.resolution, pose))

        gen = HomeGeneratorV2(cfg, home, labelled=kind == "search", gpu=args.gpu)
        try:
            meta = gen.drive(did, kind, out, failed_dir=out / "_failed", on_step=on_step)
        finally:
            gen.close()
        folder = out / did if meta["kept"] else out / "_failed" / did
        with np.load(folder / "mapmad_map.npz") as f:
            dm = {k: f[k] for k in f.files}
        with np.load(folder / TARGET_FILE) as f:
            tgt = {k: f[k] for k in f.files}
        with open(folder / "traj_data.pkl", "rb") as f:
            traj = pickle.load(f)
        mismatches = {k: 0 for k in live}
        steps = len(poses)
        for t in range(steps):
            pose = (float(traj["position"][t, 0]), float(traj["position"][t, 1]), float(traj["yaw"][t]))
            for v, m in variants:
                for key in ("noisy", "clean"):
                    lay = layers_at(dm, t, tgt, target_key=key, positions=traj["position"], visited_m=m)
                    if window_hash(lay, dm["origin"], float(dm["resolution"]), pose) != live[f"{key}_{v}"][t]:
                        mismatches[f"{key}_{v}"] += 1
        nonzero_target = int((tgt["noisy_value"] > 0).sum())
        row = {"drive": did, "kind": kind, "outcome": meta["outcome"], "steps": steps,
               "frontier_ignore_trajectory_m": meta.get("frontier_ignore_trajectory_m"),
               "mismatched_steps": mismatches, "pose_matches_traj": bool(np.allclose(
                   np.asarray(poses)[:, :2], traj["position"][: len(poses)])),
               "noisy_target_updates": int(len(tgt["noisy_value"])), "noisy_target_nonzero_updates": nonzero_target,
               "pass": not any(mismatches.values())}
        report.append(row)
        print(json.dumps(row))
    path = paths["outputs"] / cfg["out_dir"] / "checks" / args.out_name
    write_text_atomic(path, json.dumps({"layers": list(LAYERS_V2), "drives": report,
                                        "pass": all(r["pass"] for r in report)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
