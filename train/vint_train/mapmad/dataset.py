"""MapMaD's dataset (Phase 3 confirmation items 1, 7-9): NoMaD's ViNT_Dataset + map tensor + mode per sample.

Every sample is NoMaD's 7-tuple (obs, goal, actions, distance, goal_pos, dataset_index, action_mask) plus
(map [3, 64, 64], goal_mask, map_mask, mode_id). Masks: 1 = hidden. Modes: `modes.py`.

Randomness: each draw gets its own numpy Generator keyed on (seed, key); `key` comes from the epoch sampler
(`sampler.py`: epoch and draw position), so the mode, the Habitat photo goal and the heat perturbation are
reproducible and do not depend on the worker. GoStanford keeps NoMaD's own goal sampling (global numpy RNG,
negatives, action mask) unchanged; only its mode (photo shown 50/50) comes from the keyed Generator.

Habitat (`has_maps`):
- photo shown (photo, photo_map): NoMaD's goal rule at waypoint spacing s -- offset k ~ U{0..K} waypoints with
  K = min(max_dist_cat, (len - 1 - t) // s), goal frame t + k s, k = 0 -> a negative (a frame of another drive;
  never in photo_map, where k ~ U{1..K}); NoMaD's action mask (min_action < k < max_action, not negative).
  (NoMaD's own `_sample_goal` multiplies a frame offset by s again, which overshoots for s > 1.)
- photo hidden (map_goal, explore_map, explore): the goal picture is the current frame (masked, never seen by
  the transformer), distance label 0 (the distance loss only counts shown photos), action mask 1.
- map: frame t only, pose (x, y, yaw) at t; heat = the drive's final target; perturbation only when `train`.
- the 187 collision drives are dropped from the index (`exclude_drives`).
- the sample index is cached per waypoint spacing (`..._ws{s}_mapmad.pkl`; the LMDB image cache is shared).
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.mapmad import modes as M
from vint_train.mapmad.heat import rotate_180
from vint_train.mapmad.map_sample import DriveStore, MapSampleBuilder

MAP_DATASETS = ("habitat_mapmad",)


def read_drive_list(path: Optional[str]) -> set:
    """Drive names from a text file, one per line (empty set for None)."""
    if not path:
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


class MapMaDDataset(ViNT_Dataset):
    """ViNT_Dataset with a map input and MapMaD's modes; `train` switches heat perturbation on."""

    def __init__(self, *, map_cfg: Dict[str, Any], habitat_modes: Dict[str, float], goal_mask_prob: float,
                 seed: int, train: bool, exclude_drives: Optional[str] = None, **vint_kwargs) -> None:
        self.has_maps = vint_kwargs["dataset_name"] in MAP_DATASETS
        self.map_size = int(map_cfg.get("size", 64))
        super().__init__(**vint_kwargs)
        excluded = read_drive_list(exclude_drives)
        if excluded:
            self.traj_names = [t for t in self.traj_names if t not in excluded]
            self.index_to_data = [s for s in self.index_to_data if s[0] not in excluded]
            self.goals_index = [g for g in self.goals_index if g[0] not in excluded]
        self.n_excluded = len(excluded)
        self.seed, self.train, self.goal_mask_prob = int(seed), bool(train), float(goal_mask_prob)
        self.mode_p = M.habitat_mode_table(habitat_modes)
        self.builder: Optional[MapSampleBuilder] = None
        if self.has_maps:
            store = DriveStore(self.data_folder, self.traj_names, int(map_cfg.get("map_cache_size", 256)))
            builder_cfg = dict(map_cfg, perturb_prob=map_cfg.get("perturb_prob", 0.0) if train else 0.0)
            self.builder = MapSampleBuilder(store, **builder_cfg)

    # --- index ---------------------------------------------------------------------------------------------
    def _index_path(self) -> str:
        path = super()._index_path()
        return path[:-len(".pkl")] + f"_ws{self.waypoint_spacing}_mapmad.pkl" if self.has_maps else path

    def rng(self, key: int) -> np.random.Generator:
        """The Generator of one draw."""
        return np.random.default_rng([self.seed, int(key)])

    # --- samples -------------------------------------------------------------------------------------------
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        """Sample i with key i (used for test sets; training goes through `get(i, key)`)."""
        return self.get(i, i)

    def get(self, i: int, key: int) -> Tuple[torch.Tensor, ...]:
        rng = self.rng(key)
        if not self.has_maps:
            return self._gostanford(i, M.draw_gostanford_mode(self.goal_mask_prob, rng))
        drive, t, _ = self.index_to_data[i]
        mode = M.draw_habitat_mode(self.mode_p, rng)
        return self.habitat_sample(drive, t, mode, rng)

    def _gostanford(self, i: int, mode: M.Mode) -> Tuple[torch.Tensor, ...]:
        base = super().__getitem__(i)
        empty = torch.zeros((3, self.map_size, self.map_size), dtype=torch.float32)
        return base + (empty, torch.tensor(mode.goal_mask), torch.tensor(mode.map_mask), torch.tensor(mode.id))

    def max_goal_offset(self, drive: str, t: int) -> int:
        """K = the farthest photo goal in waypoints: min(max_dist_cat, (len - 1 - t) // s)."""
        n = len(self._get_trajectory(drive)["position"])
        return int(min(self.max_dist_cat, (n - 1 - t) // self.waypoint_spacing))

    def habitat_sample(self, drive: str, t: int, mode: M.Mode, rng: np.random.Generator,
                       goal_offset: Optional[int] = None, wrong_heat: bool = False,
                       perturbation: Optional[str] = "draw") -> Tuple[torch.Tensor, ...]:
        """One Habitat sample at frame t in `mode`.

        goal_offset: photo goal k waypoints ahead (None = NoMaD's random rule). wrong_heat: heat rotated 180 deg
        about the robot (offline control arm). perturbation: "draw" = drawn by the builder (training only),
        None = none, or one of map_sample.PERTURBATIONS.
        """
        s = self.waypoint_spacing
        traj = self._get_trajectory(drive)
        context = [self._load_image(drive, ct) for ct in self._context_times(t)]
        obs_image = torch.cat(context)
        negative, goal_drive, goal_time, distance = False, drive, t, 0
        if mode.goal_mask == 0:
            k = goal_offset
            if k is None:
                low = 1 if mode.name == "photo_map" else 0
                k = int(rng.integers(low, self.max_goal_offset(drive, t) + 1))
            if k == 0:
                negative = True
                goal_drive, goal_time = self.goals_index[int(rng.integers(len(self.goals_index)))]
                distance = self.max_dist_cat
            else:
                goal_time, distance = t + k * s, k
            goal_image = self._load_image(goal_drive, goal_time)
            action_mask = (self.min_action_distance < distance < self.max_action_distance) and not negative
        else:
            goal_image = context[-1]
            action_mask = True
        actions, goal_pos = self._compute_actions(traj, t, goal_time if goal_drive == drive else t)
        actions_torch = torch.as_tensor(actions, dtype=torch.float32)

        map_img = np.zeros((3, self.map_size, self.map_size), dtype=np.float32)
        if mode.map_mask == 0:
            pose = (float(traj["position"][t][0]), float(traj["position"][t][1]), float(traj["yaw"][t]))
            pert = self.builder.draw_perturbation(rng) if (perturbation == "draw" and mode.heat_on) else (
                None if perturbation == "draw" else perturbation)
            map_img = self.builder.build(drive, t, pose, mode.heat_on, pert, rng)
            if wrong_heat:
                map_img[2] = rotate_180(map_img[2])
        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            torch.as_tensor(goal_image, dtype=torch.float32),
            actions_torch,
            torch.as_tensor(distance, dtype=torch.int64),
            torch.as_tensor(goal_pos, dtype=torch.float32),
            torch.as_tensor(self.dataset_index, dtype=torch.int64),
            torch.as_tensor(action_mask, dtype=torch.float32),
            torch.from_numpy(map_img),
            torch.tensor(mode.goal_mask),
            torch.tensor(mode.map_mask),
            torch.tensor(mode.id),
        )


def build_mapmad_dataset(config: Dict[str, Any], dataset_name: str, split: str, train: bool) -> MapMaDDataset:
    """A MapMaDDataset from a train config (the same arguments train.py gives ViNT_Dataset, plus the map block)."""
    d = config["datasets"][dataset_name]
    return MapMaDDataset(
        map_cfg=config["map"], habitat_modes=config["habitat_modes"], goal_mask_prob=config["goal_mask_prob"],
        seed=config["seed"], train=train, exclude_drives=d.get("exclude_drives"),
        data_folder=d["data_folder"], data_split_folder=d[split], dataset_name=dataset_name,
        image_size=config["image_size"], waypoint_spacing=d.get("waypoint_spacing", 1),
        min_dist_cat=config["distance"]["min_dist_cat"], max_dist_cat=config["distance"]["max_dist_cat"],
        min_action_distance=config["action"]["min_dist_cat"], max_action_distance=config["action"]["max_dist_cat"],
        negative_mining=d.get("negative_mining", True), len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"], context_size=config["context_size"],
        context_type=config.get("context_type", "temporal"), context_stride=config.get("context_stride", 1),
        index_context_size=config.get("index_context_size"), end_slack=d.get("end_slack", 0),
        goals_per_obs=d.get("goals_per_obs", 1), normalize=config["normalize"],
        goal_type=config.get("goal_type", "image"),
    )
