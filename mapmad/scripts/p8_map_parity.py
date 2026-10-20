"""Phase 8.1 / G8 row 1: the online map (built step by step, as in a closed loop) = the generator map rebuilt from the
saved arrays, on 2 drives in dev homes (val-drive homes; never an exam home).

    python mapmad/scripts/p8_map_parity.py --gpu 1      # inside naz_mapmad_habitat

For each drive, at every step t the live state (first-seen map, clean + noisy target fusion, pose) is cut into the near
and far windows (map_v2, layers obstacle / explored / frontier / target) and hashed. After the drive is written, the
same windows are rebuilt from mapmad_map.npz + mapmad_target.npz + traj_data.pkl alone (map_v2.layers_at) and must be
byte-identical at every step. Drives go to <mapmad_data>/p8_checks_parity/ (check data, not a dataset); the result to
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
    args = p.parse_args()
    cfg = read_config(CONFIG)
    paths = config.paths()
    out = paths["mapmad_data"] / "p8_checks_parity"
    if out.exists():  # our own check output, rebuilt every time
        shutil.rmtree(out)
    homes = home_splits.train_homes(cfg["split"])
    report: List[Dict] = []
    for prefix5, kind, n in DRIVES:
        home = home_splits.by_prefix(homes, [prefix5])[0]
        did = drive_id("p8x", home, kind, n)
        live: Dict[str, List[str]] = {"noisy": [], "clean": []}
        poses: List = []

        def on_step(t, fmap, fusions, pose):
            first = {"obstacle_first_seen": fmap.first_seen["obstacle"].reshape(fmap.shape),
                     "explored_first_seen": fmap.first_seen["explored"].reshape(fmap.shape)}
            base = layers_at(first, t)
            for key in live:
                lay = dict(base, target=fusions[key].grid().copy())
                live[key].append(window_hash(lay, fmap.origin, fmap.spec.resolution, pose))
            poses.append(pose)

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
        mismatches = {"noisy": 0, "clean": 0}
        for t in range(len(live["noisy"])):
            pose = (float(traj["position"][t, 0]), float(traj["position"][t, 1]), float(traj["yaw"][t]))
            for key in live:
                lay = layers_at(dm, t, tgt, target_key=key)
                if window_hash(lay, dm["origin"], float(dm["resolution"]), pose) != live[key][t]:
                    mismatches[key] += 1
        nonzero_target = int((tgt["noisy_value"] > 0).sum())
        row = {"drive": did, "kind": kind, "outcome": meta["outcome"], "steps": len(live["noisy"]),
               "mismatched_steps": mismatches, "pose_matches_traj": bool(np.allclose(
                   np.asarray(poses)[:, :2], traj["position"][: len(poses)])),
               "noisy_target_updates": int(len(tgt["noisy_value"])), "noisy_target_nonzero_updates": nonzero_target,
               "pass": mismatches == {"noisy": 0, "clean": 0}}
        report.append(row)
        print(json.dumps(row))
    path = paths["outputs"] / cfg["out_dir"] / "checks" / "map_parity.json"
    write_text_atomic(path, json.dumps({"layers": list(LAYERS_V2), "drives": report,
                                        "pass": all(r["pass"] for r in report)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
