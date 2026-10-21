"""The v2 map input of one sample (Phase 10 block item 3): float32 [2, 4, 64, 64] = (near, far) x (obstacle 0/1,
explored 0/1 or far mean, frontier 0/1, target seen here 0..1), robot-centred, forward = up, cut with Phase 8's
map_v2.near_window / far_window at frame t (cells first seen <= t, nothing later). No normalisation, no augmentation.

- obstacle / explored: the drive's mapmad_map.npz as known at t (as v1);
- frontier: the teacher's frontier at t (map_v2.teacher_frontier_at), read from the per-drive cache
  (frontier_cache.py) -- the network never sees the bumped cells themselves;
- target: rebuilt at t from mapmad_target.npz's sparse updates, `noisy` for training, `clean` for a check arm.

`DriveStoreV2` keeps every drive's kind and category (read once from mapmad_meta.json) and per-process LRU caches of
the three per-drive files (each DataLoader worker holds its own). `MapV2Builder.build` makes the tensor; its options
are the offline arms' (block item 5): clean layer, zeroed layers, target shifted 1 m.
"""

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from vint_train.mapmad.frontier_cache import cache_path, frontier_from_cache, load_cache
from vint_train.mapmad.local_map import known_at, load_drive_map
from vint_train.mapmad.map_v2 import LAYERS_V2, TARGET_FILE, far_window, near_window, target_at

N_WINDOWS = 2  # near, far
LAYER_INDEX = {name: k for k, name in enumerate(LAYERS_V2)}
Pose = Tuple[float, float, float]


class _LRU:
    """A tiny per-process LRU of per-drive file contents."""

    def __init__(self, size: int, load: Callable[[str], Dict[str, np.ndarray]]) -> None:
        self.size, self.load, self.items = int(size), load, OrderedDict()

    def get(self, key: str) -> Dict[str, np.ndarray]:
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        value = self.load(key)
        self.items[key] = value
        if len(self.items) > self.size:
            self.items.popitem(last=False)
        return value


def load_target(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k.startswith(("clean_", "noisy_")) or k == "shape"}


class DriveStoreV2:
    """Kinds/categories of all drives (preloaded) + per-process LRU caches of map, target and frontier cache."""

    def __init__(self, data_folder: str, drives: Iterable[str], frontier_dir: str, cache_size: int = 64) -> None:
        self.data_folder, self.frontier_dir, self.cache_size = data_folder, frontier_dir, int(cache_size)
        self.kind: Dict[str, str] = {}
        self.category: Dict[str, str] = {}
        for d in drives:
            with open(os.path.join(data_folder, d, "mapmad_meta.json")) as f:
                meta = json.load(f)
            self.kind[d] = meta["kind"]
            self.category[d] = meta["category"] if meta["kind"] == "search" else "explore"
        self._init_caches()

    def _init_caches(self) -> None:
        self.maps = _LRU(self.cache_size, lambda d: load_drive_map(os.path.join(self.data_folder, d)))
        self.targets = _LRU(self.cache_size, lambda d: load_target(os.path.join(self.data_folder, d, TARGET_FILE)))
        self.frontiers = _LRU(self.cache_size, lambda d: load_cache(cache_path(self.frontier_dir, d)))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in ("maps", "targets", "frontiers"):
            state.pop(k)  # workers start with empty caches
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__ = state
        self._init_caches()


@dataclass(frozen=True)
class MapOptions:
    """What the map input shows (training: the defaults; offline arms: block item 5)."""

    target_key: str = "noisy"  # "noisy" (training input) or "clean" (simulator truth, check arm)
    show: Tuple[bool, bool] = (True, True)  # (near, far); a hidden window is all zero and masked
    zero_layers: Tuple[str, ...] = ()  # e.g. ("target",) or ("frontier",)
    target_shift_m: Optional[Tuple[float, float]] = None  # world (dx, dy): the target layer drawn this far off


def layers_at_frame(store: DriveStoreV2, drive: str, t: int, target_on: bool, target_key: str) -> Dict[str, np.ndarray]:
    """Whole-storey layers at frame t: obstacle, explored, frontier (bool), target (float32; zeros if not target_on)."""
    dm = store.maps.get(drive)
    obstacle = known_at(dm["obstacle_first_seen"], t)
    layers = {"obstacle": obstacle, "explored": known_at(dm["explored_first_seen"], t),
              "frontier": frontier_from_cache(store.frontiers.get(drive), t)}
    if target_on:
        tg = store.targets.get(drive)
        layers["target"] = target_at(tg[f"{target_key}_cell"], tg[f"{target_key}_frame"], tg[f"{target_key}_value"], t,
                                     obstacle.shape)
    else:
        layers["target"] = np.zeros(obstacle.shape, dtype=np.float32)
    return layers


def windows(layers: Dict[str, np.ndarray], origin: Sequence[float], res: float, pose: Pose,
            near_m: float, far_m: float, size: int) -> np.ndarray:
    """[2, 4, size, size]: the near and far windows of the four layers."""
    return np.stack([near_window(layers, origin, res, pose, size, near_m), far_window(layers, origin, res, pose, size, far_m)])


class MapV2Builder:
    """Builds the [2, 4, 64, 64] map tensor from the config's `map` block (size, resolution, far_resolution)."""

    def __init__(self, store: DriveStoreV2, size: int = 64, resolution: float = 0.1, far_resolution: float = 0.4,
                 **_unused) -> None:
        self.store, self.size = store, int(size)
        self.near_m, self.far_m = float(resolution), float(far_resolution)

    def build(self, drive: str, t: int, pose: Pose, target_on: bool, options: MapOptions = MapOptions()) -> np.ndarray:
        """float32 [2, 4, size, size] (module docstring); hidden windows (options.show) are all zero."""
        layers = layers_at_frame(self.store, drive, t, target_on, options.target_key)
        dm = self.store.maps.get(drive)
        origin, res = dm["origin"], float(dm["resolution"])
        out = windows(layers, origin, res, pose, self.near_m, self.far_m, self.size)
        if options.target_shift_m is not None and target_on:
            dx, dy = options.target_shift_m  # sampling at pose - shift shows the layer shifted by +shift
            shifted = windows({name: layers["target"] for name in LAYERS_V2}, origin, res,
                              (pose[0] - dx, pose[1] - dy, pose[2]), self.near_m, self.far_m, self.size)
            out[:, LAYER_INDEX["target"]] = shifted[:, LAYER_INDEX["target"]]
        for name in options.zero_layers:
            out[:, LAYER_INDEX[name]] = 0.0
        for w, shown in enumerate(options.show):
            if not shown:
                out[w] = 0.0
        return out
