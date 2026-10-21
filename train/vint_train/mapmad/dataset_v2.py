"""MapMaD v2's dataset (Phase 10 block items 2-3): MapMaDDataset with the two-window map and the v2 modes.

Every sample is NoMaD's 7-tuple plus (map [2, 4, 64, 64], goal_mask, map_mask [2] (near, far; 1 = hidden), mode id).

Habitat (`habitat_mapmad_v2`), modes drawn per sample with the draw's own Generator (dataset.py), shares from
`habitat_modes` (modes.HABITAT_MODES_V2):
- search_map 45%: photo hidden, both maps, noisy target layer -- a sample of a SEARCH drive;
- explore_map 20%: photo hidden, both maps, target layer empty -- a sample of an EXPLORATION drive;
- explore 20%: photo and maps hidden (NoMaD's explore mode) -- any drive;
- photo 15%: photo goal 0-20 waypoints ahead (v1 rule, negatives at k = 0), maps hidden -- any drive.
The sampler hands over a uniform sample i of the whole split. For a mode limited to one drive kind, i is kept if its
drive is of that kind, else replaced by a uniform draw from that kind's samples -- so each mode's samples are uniform
over its pool and the mode shares are exactly the drawn ones.
GoStanford: maps hidden (all zero, both masks 1), photo shown/hidden 50/50 with NoMaD's own goal sampling.
The distance loss counts only shown photos (train_loop.py).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from vint_train.mapmad import modes as M
from vint_train.mapmad.dataset import MapMaDDataset, pose_at
from vint_train.mapmad.map_sample_v2 import N_WINDOWS, DriveStoreV2, MapOptions, MapV2Builder

MAP_DATASETS_V2 = ("habitat_mapmad_v2",)
MODE_KIND = {"search_map": "search", "explore_map": "explore"}  # modes limited to one drive kind


class MapMaDV2Dataset(MapMaDDataset):
    """v2 map input and modes (module docstring)."""

    map_datasets = MAP_DATASETS_V2

    def _init_modes(self, habitat_modes: Dict[str, float]) -> None:
        self.mode_p = M.habitat_mode_table(habitat_modes, M.HABITAT_MODES_V2)

    def _init_maps(self, map_cfg: Dict[str, Any], train: bool) -> None:
        if int(map_cfg.get("layers", 4)) != 4:
            raise ValueError("the v2 map has 4 layers")
        store = DriveStoreV2(self.data_folder, self.traj_names, map_cfg["frontier_cache"],
                             int(map_cfg.get("map_cache_size", 64)))
        self.builder = MapV2Builder(store, **map_cfg)
        kinds = np.array([store.kind[d] == "search" for d, _, _ in self.index_to_data])
        self.pools = {"search": np.flatnonzero(kinds), "explore": np.flatnonzero(~kinds)}
        self.in_pool = {"search": kinds, "explore": ~kinds}

    def empty_map(self) -> torch.Tensor:
        return torch.zeros((N_WINDOWS, 4, self.map_size, self.map_size), dtype=torch.float32)

    def map_mask_tensor(self, mode: M.Mode) -> torch.Tensor:
        return torch.full((N_WINDOWS,), mode.map_mask, dtype=torch.long)

    # --- drives --------------------------------------------------------------------------------------------
    def kind(self, drive: str) -> str:
        return self.builder.store.kind[drive]

    def category(self, drive: str) -> str:
        """The search drive's category, or "explore"."""
        return self.builder.store.category[drive]

    def pool_index(self, i: int, mode: M.Mode, rng: np.random.Generator) -> int:
        """Sample i if its drive suits the mode, else a uniform draw from the mode's pool (module docstring)."""
        kind = MODE_KIND.get(mode.name)
        if kind is None or self.in_pool[kind][i]:
            return int(i)
        pool = self.pools[kind]
        if len(pool) == 0:
            raise ValueError(f"mode {mode.name} needs {kind} drives; this split has none")
        return int(pool[int(rng.integers(len(pool)))])

    # --- samples -------------------------------------------------------------------------------------------
    def get(self, i: int, key: int) -> Tuple[torch.Tensor, ...]:
        rng = self.rng(key)
        if not self.has_maps:
            return self._gostanford(i, M.draw_gostanford_mode(self.goal_mask_prob, rng))
        mode = M.draw_habitat_mode(self.mode_p, rng, M.HABITAT_MODES_V2)
        drive, t, _ = self.index_to_data[self.pool_index(i, mode, rng)]
        return self.habitat_sample(drive, t, mode, rng)

    def habitat_sample(self, drive: str, t: int, mode: M.Mode, rng: np.random.Generator,
                       goal_offset: Optional[int] = None,
                       options: MapOptions = MapOptions()) -> Tuple[torch.Tensor, ...]:
        """One Habitat sample at frame t in `mode`; `options` = the offline arms' map variants (map_sample_v2)."""
        base = self.photo_part(drive, t, mode, rng, goal_offset)
        map_img = self.empty_map()
        map_mask = self.map_mask_tensor(mode)
        if mode.map_mask == 0:
            built = self.builder.build(drive, t, pose_at(self._get_trajectory(drive), t), mode.heat_on, options)
            map_img = torch.from_numpy(built)
            map_mask = torch.tensor([0 if shown else 1 for shown in options.show], dtype=torch.long)
        return base + (map_img, torch.tensor(mode.goal_mask), map_mask, torch.tensor(mode.id))
