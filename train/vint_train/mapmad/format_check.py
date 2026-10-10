"""Gate G2 row 3, part 1: NoMaD's UNCHANGED loader (ViNT_Dataset) reads a MapMaD dataset; the map files are ignored.

    cd train && python -m vint_train.mapmad.format_check --config config/mapmad_format_check.yaml   # naz_mapmad
    ... --split-dir /mapmad_data/splits/habitat_mapmad     # the same, on the full train/test splits

Builds every dataset of the config exactly as train.py does (config/defaults.yaml + the config, same arguments),
reads `--samples` random items of each split, checks their shapes and that the normalised actions of
habitat_mapmad look like NoMaD's (forward > 0 on average, about 1 unit per frame at 0.2 m/s), and prints the
LMDB cache size next to the split files. Part 2 (one training step) is train.py itself with the same config.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from vint_train.data.vint_dataset import ViNT_Dataset

TRAIN = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> Dict[str, Any]:
    """defaults.yaml updated with the given config (train.py's rule: top-level keys replaced)."""
    config = yaml.safe_load((TRAIN / "config" / "defaults.yaml").read_text())
    config.update(yaml.safe_load(Path(path).read_text()))
    config.setdefault("index_context_size", config["context_size"])  # train.py main()'s defaults
    config.setdefault("context_stride", 1)
    return config


def build(config: Dict[str, Any], name: str, split: str) -> ViNT_Dataset:
    """One ViNT_Dataset, with train.py's defaults and arguments."""
    d = dict(config["datasets"][name])
    d.setdefault("negative_mining", True)
    d.setdefault("goals_per_obs", 1)
    d.setdefault("end_slack", 0)
    d.setdefault("waypoint_spacing", 1)
    return ViNT_Dataset(data_folder=d["data_folder"], data_split_folder=d[split], dataset_name=name,
                        image_size=config["image_size"], waypoint_spacing=d["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"], max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"], negative_mining=d["negative_mining"],
                        len_traj_pred=config["len_traj_pred"], learn_angle=config["learn_angle"],
                        context_size=config["context_size"], context_type=config["context_type"],
                        context_stride=config["context_stride"], index_context_size=config["index_context_size"],
                        end_slack=d["end_slack"], goals_per_obs=d["goals_per_obs"], normalize=config["normalize"],
                        goal_type=config["goal_type"])


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9


def check(config: Dict[str, Any], samples: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    for name, d in config["datasets"].items():
        for split in ("train", "test"):
            if split not in d:
                continue
            ds = build(config, name, split)
            idx = rng.choice(len(ds), min(samples, len(ds)), replace=False)
            items = [ds[int(i)] for i in idx]
            obs, goal, actions = items[0][0], items[0][1], items[0][2]  # ViNT_Dataset item: obs, goal, actions, ...
            fwd = np.array([float(it[2][:, 0].mean()) for it in items])
            out[f"{name}/{split}"] = {
                "samples": len(ds), "trajectories": len(ds.traj_names), "obs_shape": list(obs.shape),
                "goal_shape": list(goal.shape), "action_shape": list(actions.shape),
                "mean_normalised_forward": round(float(fwd.mean()), 3), "share_forward_positive": round(float((fwd > 0).mean()), 3),
                "lmdb_gb": round(dir_size_gb(Path(d[split]) / f"dataset_{name}.lmdb"), 3),
            }
            ds.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="also write the result here (json)")
    p.add_argument("--split-dir", type=Path, help="read <dir>/train and <dir>/test instead of the config's splits "
                   "(every dataset of the config)")
    args = p.parse_args()
    os.chdir(TRAIN)
    config = load_config(args.config)
    if args.split_dir:
        for d in config["datasets"].values():
            d["train"], d["test"] = f"{args.split_dir}/train/", f"{args.split_dir}/test/"
    result = check(config, args.samples, args.seed)
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
