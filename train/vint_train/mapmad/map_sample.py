"""The 3-layer map input of one sample (Phase 3 confirmation items 1-3): float32 [3, 64, 64] =
obstacle 0/1 · explored 0/1 (unknown = 0) · heat 0-1. No normalisation, no augmentation (a flip would break
left/right vs the actions). Only frame t gets a map; cells outside the stored floor grid are 0.

`DriveStore` keeps every drive's target (read once from mapmad_meta.json) and an LRU cache of floor maps
(one cache per DataLoader worker, since each worker holds its own copy of the dataset).
`MapSampleBuilder.build` cuts the map with `local_map.local_map` and draws the heat with `heat.draw_heat`.
"""

import json
import os
from collections import OrderedDict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from vint_train.mapmad import heat as heat_lib
from vint_train.mapmad.local_map import load_drive_map, local_map

PERTURBATIONS = ("shift", "blur", "false_blob")


class DriveStore:
    """Targets of all drives (preloaded) + floor maps (per-process LRU of `cache_size` drives)."""

    def __init__(self, data_folder: str, drives: Iterable[str], cache_size: int = 256) -> None:
        self.data_folder = data_folder
        self.cache_size = cache_size
        self.targets: Dict[str, heat_lib.Target] = {}
        for d in drives:
            with open(os.path.join(data_folder, d, "mapmad_meta.json")) as f:
                self.targets[d] = heat_lib.target_from_meta(json.load(f))
        self._maps: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_maps"] = OrderedDict()  # workers start with an empty cache
        return state

    def floor_map(self, drive: str) -> Dict[str, np.ndarray]:
        """The drive's mapmad_map.npz contents (cached)."""
        if drive in self._maps:
            self._maps.move_to_end(drive)
            return self._maps[drive]
        m = load_drive_map(os.path.join(self.data_folder, drive))
        self._maps[drive] = m
        if len(self._maps) > self.cache_size:
            self._maps.popitem(last=False)
        return m


class MapSampleBuilder:
    """Builds map tensors from the config's `map` block."""

    def __init__(self, store: DriveStore, size: int = 64, resolution: float = 0.1, heat_sigma: float = 0.3,
                 border_inset: float = 0.3, perturb_prob: float = 0.0, perturb_shift: Tuple[float, float] = (0.3, 1.0),
                 perturb_blur_factor: float = 2.0, perturb_false_min_dist: float = 1.0, **_unused) -> None:
        self.store = store
        self.size, self.resolution = int(size), float(resolution)
        self.sigma, self.inset = float(heat_sigma), float(border_inset)
        self.perturb_prob = float(perturb_prob)
        self.shift = tuple(perturb_shift)
        self.blur_factor = float(perturb_blur_factor)
        self.false_min_dist = float(perturb_false_min_dist)

    def heat(self, drive: str, pose: Tuple[float, float, float], perturbation: Optional[str] = None,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """(size, size) heat for the drive's target; `perturbation` in PERTURBATIONS needs `rng`."""
        target = self.store.targets[drive]
        sigma = self.sigma
        if perturbation == "shift":
            target = heat_lib.perturb_target(target, rng, self.shift)
        elif perturbation == "blur":
            sigma = self.sigma * self.blur_factor  # a peak-1 Gaussian with twice the width = blurred + renormalised
        h = heat_lib.draw_heat(target, pose, self.size, self.resolution, sigma, self.inset)
        if perturbation == "false_blob":
            centre = heat_lib.blob_centre(target, pose, self.size, self.resolution, self.inset)
            h = heat_lib.add_false_blob(h, centre, rng, self.size, self.resolution, self.sigma, self.false_min_dist)
        return h

    def draw_perturbation(self, rng: np.random.Generator) -> Optional[str]:
        """With probability perturb_prob one of PERTURBATIONS (equal chance), else None."""
        if self.perturb_prob <= 0 or rng.random() >= self.perturb_prob:
            return None
        return PERTURBATIONS[int(rng.integers(len(PERTURBATIONS)))]

    def build(self, drive: str, t: int, pose: Tuple[float, float, float], heat_on: bool,
              perturbation: Optional[str] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """float32 [3, size, size]: obstacle, explored (cells first seen <= t) and heat (zeros if heat_on is False)."""
        out = np.zeros((3, self.size, self.size), dtype=np.float32)
        out[:2] = local_map(self.store.floor_map(drive), t, pose, self.size, self.resolution)
        if heat_on:
            out[2] = self.heat(drive, pose, perturbation, rng)
        return out
