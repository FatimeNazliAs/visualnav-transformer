"""The frozen Habitat sample list of the offline harness (Phase 3 confirmation item 14): 5,000 val-drive test
samples spread evenly over the drives, frames t with t - 15 >= 0 and t + 40 <= last frame (valid for waypoint
spacing 1, 2 and 5), stored as JSON with the sha256 of its canonical form. `offline_eval make-list` writes it.
"""

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

N_SAMPLES = 5000
LIST_SEED = 20261010
BACK_FRAMES, AHEAD_FRAMES = 15, 40  # 3 context frames x spacing 5 back; 8 waypoints x spacing 5 ahead


def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def v1_type(meta: Dict[str, Any], traj: Dict[str, np.ndarray], t: int) -> Dict[str, Any]:
    """v1 list fields: the drive's target kind (spot / object)."""
    return {"type": meta["target"]["kind"]}


def make_list(data_folder: str, drives: Sequence[str], n: int = N_SAMPLES, seed: int = LIST_SEED,
              fields: Callable[[Dict[str, Any], Dict[str, np.ndarray], int], Dict[str, Any]] = v1_type
              ) -> List[Dict[str, Any]]:
    """n samples spread evenly over `drives` (n // D or n // D + 1 each, the +1 drives drawn with the seed);
    `fields(meta, traj, t)` gives each sample's extra fields (v1: "type"; v2: list_v2.v2_fields)."""
    import pickle

    rng = np.random.default_rng(seed)
    drives = sorted(drives)
    per = np.full(len(drives), n // len(drives))
    per[rng.choice(len(drives), n - per.sum(), replace=False)] += 1
    out = []
    for drive, k in zip(drives, per):
        with open(os.path.join(data_folder, drive, "traj_data.pkl"), "rb") as f:
            traj = pickle.load(f)
        length = len(traj["position"])
        with open(os.path.join(data_folder, drive, "mapmad_meta.json")) as f:
            meta = json.load(f)
        valid = np.arange(BACK_FRAMES, length - 1 - AHEAD_FRAMES + 1)
        if len(valid) < k:
            raise ValueError(f"{drive}: only {len(valid)} valid frames for {k} samples")
        for t in np.sort(rng.choice(valid, k, replace=False)):
            out.append({"drive": drive, "t": int(t), **fields(meta, traj, int(t))})
    for i, s in enumerate(out):
        s["id"] = i
    return out


def load_list(path: str) -> Dict[str, Any]:
    with open(path) as f:
        body = json.load(f)
    if canonical_sha256(body["samples"]) != body["sha256"]:
        raise SystemExit(f"{path}: sha256 mismatch, the list was changed")
    return body
