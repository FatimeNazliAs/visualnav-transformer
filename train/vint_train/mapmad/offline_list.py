"""The frozen Habitat sample list of the offline harness (Phase 3 confirmation item 14): 5,000 val-drive test
samples spread evenly over the drives, frames t with t - 15 >= 0 and t + 40 <= last frame (valid for waypoint
spacing 1, 2 and 5), stored as JSON with the sha256 of its canonical form. `offline_eval make-list` writes it.
"""

import hashlib
import json
import os
from typing import Any, Dict, List, Sequence

import numpy as np

N_SAMPLES = 5000
LIST_SEED = 20261010
BACK_FRAMES, AHEAD_FRAMES = 15, 40  # 3 context frames x spacing 5 back; 8 waypoints x spacing 5 ahead


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_list(data_folder: str, drives: Sequence[str], n: int = N_SAMPLES, seed: int = LIST_SEED) -> List[Dict[str, Any]]:
    """n samples spread evenly over `drives` (n // D or n // D + 1 each, the +1 drives drawn with the seed)."""
    import pickle

    rng = np.random.default_rng(seed)
    drives = sorted(drives)
    per = np.full(len(drives), n // len(drives))
    per[rng.choice(len(drives), n - per.sum(), replace=False)] += 1
    out = []
    for drive, k in zip(drives, per):
        with open(os.path.join(data_folder, drive, "traj_data.pkl"), "rb") as f:
            length = len(pickle.load(f)["position"])
        with open(os.path.join(data_folder, drive, "mapmad_meta.json")) as f:
            kind = json.load(f)["target"]["kind"]
        valid = np.arange(BACK_FRAMES, length - 1 - AHEAD_FRAMES + 1)
        if len(valid) < k:
            raise ValueError(f"{drive}: only {len(valid)} valid frames for {k} samples")
        for t in np.sort(rng.choice(valid, k, replace=False)):
            out.append({"drive": drive, "t": int(t), "type": kind})
    for i, s in enumerate(out):
        s["id"] = i
    return out


def load_list(path: str) -> Dict[str, Any]:
    with open(path) as f:
        body = json.load(f)
    if canonical_sha256(body["samples"]) != body["sha256"]:
        raise SystemExit(f"{path}: sha256 mismatch, the list was changed")
    return body
