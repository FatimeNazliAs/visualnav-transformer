"""MapMaD's epoch (Phase 3 confirmation item 9): 1 epoch = `epoch_samples` draws WITH replacement, each draw first
picks a dataset by weight (70% Habitat / 30% GoStanford), then a uniform sample of it; reseeded per epoch with
(seed, epoch).

The sampler yields one integer code per draw, code = key * total + global_index, with
key = epoch * epoch_samples + position in the epoch. `KeyedConcatDataset` decodes it and hands the key to the
dataset, which seeds that sample's own Generator with it (mode, photo goal, heat perturbation). The sampler runs
in the main process, so `set_epoch` works with persistent workers.
"""

from typing import Dict, Iterator, List, Sequence

import numpy as np
from torch.utils.data import ConcatDataset, Sampler


class KeyedConcatDataset(ConcatDataset):
    """ConcatDataset of MapMaDDatasets indexed by the sampler's codes (plain ints < total also work, key = index)."""

    def __getitem__(self, code: int):
        total = self.cumulative_sizes[-1]
        key, idx = divmod(int(code), total)
        if key == 0:
            key = idx  # plain index (no epoch code)
        ds_i = int(np.searchsorted(self.cumulative_sizes, idx, side="right"))
        local = idx - (self.cumulative_sizes[ds_i - 1] if ds_i > 0 else 0)
        return self.datasets[ds_i].get(local, key)


class EpochSampler(Sampler):
    """`epoch_samples` weighted draws with replacement per epoch, reseeded with (seed, epoch)."""

    def __init__(self, sizes: Sequence[int], weights: Sequence[float], epoch_samples: int, seed: int) -> None:
        if len(sizes) != len(weights) or min(sizes) <= 0:
            raise ValueError(f"sizes {sizes} / weights {weights} do not match or a dataset is empty")
        w = np.asarray(weights, dtype=np.float64)
        self.p = w / w.sum()
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]])
        self.total = int(self.sizes.sum())
        self.epoch_samples, self.seed, self.epoch = int(epoch_samples), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def draws(self, epoch: int) -> np.ndarray:
        """Global sample indices of one epoch (deterministic in seed and epoch)."""
        rng = np.random.default_rng([self.seed, epoch, 7919])
        which = rng.choice(len(self.sizes), size=self.epoch_samples, p=self.p)
        local = (rng.random(self.epoch_samples) * self.sizes[which]).astype(np.int64)
        return self.offsets[which] + local

    def __iter__(self) -> Iterator[int]:
        base = (self.epoch * self.epoch_samples + 1) * self.total  # +1 keeps key > 0 (key 0 = plain index)
        for pos, idx in enumerate(self.draws(self.epoch)):
            yield int(base + pos * self.total + idx)

    def __len__(self) -> int:
        return self.epoch_samples


def dataset_weights(names: List[str], weights: Dict[str, float]) -> List[float]:
    """Weights in dataset order; every training dataset must have one."""
    missing = [n for n in names if n not in weights]
    if missing:
        raise ValueError(f"dataset_weights has no entry for {missing}")
    return [float(weights[n]) for n in names]
