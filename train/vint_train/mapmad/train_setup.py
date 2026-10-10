"""Data loaders for a MapMaD training run (train.py, `map_input: true`).

Train: every dataset in `config["datasets"]` as a MapMaDDataset (train split, heat perturbation on), joined by
`KeyedConcatDataset` and drawn by `EpochSampler` (`epoch_samples` per epoch, `dataset_weights`).
Test (in-loop eval): per dataset a fixed seeded subset of `eval_samples` test samples (no perturbation; each
sample's mode is fixed by its index), loaded in order.
"""

from typing import Any, Dict, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset

from vint_train.mapmad.dataset import build_mapmad_dataset
from vint_train.mapmad.sampler import EpochSampler, KeyedConcatDataset, dataset_weights


def eval_subset(dataset, n: int, seed: int) -> Subset:
    """n test samples drawn without replacement with a fixed seed (all if the split is smaller), in index order."""
    rng = np.random.default_rng([seed, 104729])
    idx = np.arange(len(dataset)) if n >= len(dataset) else np.sort(rng.choice(len(dataset), n, replace=False))
    return Subset(dataset, idx.tolist())


def build_mapmad_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, EpochSampler, Dict[str, DataLoader]]:
    """(train loader, its epoch sampler, {"<name>_test": loader})."""
    if config.get("gradient_accumulation_steps", 1) != 1:
        raise ValueError("the MapMaD loop does not implement gradient_accumulation_steps")
    names, train_sets, tests = [], [], {}
    for name, d in config["datasets"].items():
        if "train" in d:
            ds = build_mapmad_dataset(config, name, "train", train=True)
            names.append(name)
            train_sets.append(ds)
            print(f"[mapmad] train {name}: {len(ds)} samples, {len(ds.traj_names)} drives "
                  f"({ds.n_excluded} listed as excluded), waypoint_spacing {ds.waypoint_spacing}", flush=True)
        if "test" in d:
            ds = build_mapmad_dataset(config, name, "test", train=False)
            sub = eval_subset(ds, int(config.get("eval_samples", 2048)), config["seed"])
            tests[f"{name}_test"] = DataLoader(sub, batch_size=config.get("eval_batch_size", config["batch_size"]),
                                               shuffle=False, num_workers=min(4, config["num_workers"]),
                                               drop_last=False)
            print(f"[mapmad] test {name}: {len(ds)} samples, eval subset {len(sub)}", flush=True)
    sampler = EpochSampler([len(d) for d in train_sets], dataset_weights(names, config["dataset_weights"]),
                           config["epoch_samples"], config["seed"])
    loader = DataLoader(KeyedConcatDataset(train_sets), batch_size=config["batch_size"], sampler=sampler,
                        num_workers=config["num_workers"], drop_last=False,
                        persistent_workers=config["num_workers"] > 0)
    return loader, sampler, tests
