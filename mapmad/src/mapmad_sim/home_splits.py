"""Which HM3D train homes give training drives, held-out val-drive drives, and pilot drives (Phase 2).

- val-drive homes: the 10 Phase 1 homes + n random UNLABELLED train homes (fixed seed); whole homes held out.
- pilot homes: n train homes outside val-drive, at least n_labelled of them labelled (fixed seed).
- train homes: every other train home. The 36 ObjectNav val homes (HM3D val split) are never listed.
The lists are frozen in configs/p2_home_splits.json with the sha256 of each list (scripts/make_p2_splits.py).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from mapmad_sim import config


def train_homes(split: str = "train") -> List[str]:
    """All homes of an HM3D split, sorted (train: 800)."""
    root = config.paths()["hm3d_scenes"] / split
    return sorted(p.name for p in root.glob("*-*") if p.is_dir())


def list_sha256(homes: Sequence[str]) -> str:
    """sha256 of the home names, one per line, in the given order."""
    return hashlib.sha256("\n".join(homes).encode()).hexdigest()


def by_prefix(homes: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """Full home names ('00081-5biL7VEkByM') for 5-digit prefixes ('00081'); raises if one is missing."""
    found = {h[:5]: h for h in homes}
    missing = [p for p in prefixes if p not in found]
    if missing:
        raise ValueError(f"homes not found: {missing}")
    return [found[p] for p in prefixes]


def draw_splits(homes: Sequence[str], labelled: Sequence[str], cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """val_drive / pilot / train lists from the `homes` block of p2_datagen.yaml (deterministic)."""
    labelled = set(labelled)
    val_cfg, pilot_cfg = cfg["val_drive"], cfg["pilot"]
    phase1 = by_prefix(homes, val_cfg["phase1_homes"])
    unlabelled = [h for h in homes if h not in labelled]
    rng = np.random.default_rng(val_cfg["seed"])
    extra = [unlabelled[i] for i in sorted(rng.choice(len(unlabelled), val_cfg["n_random_unlabelled"], replace=False))]
    val_drive = sorted(phase1 + extra)
    rest = [h for h in homes if h not in set(val_drive)]
    rng = np.random.default_rng(pilot_cfg["seed"])
    rest_lab = [h for h in rest if h in labelled]
    rest_unlab = [h for h in rest if h not in labelled]
    n_lab = pilot_cfg["n_labelled"]
    pilot = sorted([rest_lab[i] for i in rng.choice(len(rest_lab), n_lab, replace=False)]
                   + [rest_unlab[i] for i in rng.choice(len(rest_unlab), pilot_cfg["n_homes"] - n_lab, replace=False)])
    return {"val_drive": val_drive, "pilot": pilot, "train": rest}


def load(path: Path) -> Dict[str, Any]:
    """The frozen splits file; raises if a stored sha256 no longer matches its list."""
    doc = json.loads(Path(path).read_text())
    for name, homes in doc["homes"].items():
        if list_sha256(homes) != doc["sha256"][name]:
            raise ValueError(f"{path}: {name} list changed (sha256 mismatch)")
    return doc
