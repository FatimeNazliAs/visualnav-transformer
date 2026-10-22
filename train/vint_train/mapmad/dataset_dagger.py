"""DAgger samples for v2.1 (Phase 11 block items 3-4, loader entry `habitat_mapmad_v2_dagger`): frames of the student's
own rollouts with the explorer teacher's look-ahead labels (mapmad/src/mapmad_sim/dagger_labels.py, `labels.npz`
in each drive folder).

- The rollout drives are in the Phase 8 drive format, so a `DaggerDataset` is a MapMaDV2Dataset over the DAgger folder
  and split (dataset name habitat_mapmad_v2: same metric waypoint spacing 0.05, same dataset index) whose sample index
  is the LABELLED frames (labels.npz `valid`: not a teacher-fail state, not inside a collision streak >= 20) with the
  full context (t >= context_size x waypoint_spacing = 15; no future frames are needed). Its map input is built
  exactly as in training (own frontier cache); its action target is the stored label (8 waypoints, metres) / (0.05 x
  waypoint_spacing 5) instead of the drive's future poses. Photo hidden, both maps shown: it serves the search_map
  mode only.
- Union (block item 4): the training habitat_mapmad_v2 dataset gets the DaggerDataset attached
  (`MapMaDV2Dataset.attach_dagger`); a search_map draw is then uniform over (Phase 8 search samples + DAgger samples),
  every labelled state as likely as a Phase 8 search sample; with `map.seen_oversample` = p > 0 the "seen before, now
  out of view" pool is the union of both sources' such frames. Modes stay 45/20/20/15, GoStanford 50/50. Without a
  `map.dagger` block (or for the test split) nothing changes: the v2 loader samples exactly as before.

Config (train config `map.dagger`): data_folder, train (split folder with traj_names.txt), frontier_cache, index_dir.

    python -m vint_train.mapmad.dataset_dagger prepare --config config/mapmad_v21.yaml --out <json>
builds the DAgger sample index, its LMDB image cache and reports the union shares (chain step "data index");
    python -m vint_train.mapmad.dataset_dagger frontier --config config/mapmad_v21.yaml --out <json>
builds the DAgger drives' frontier cache (frontier_cache.build_all; exit 3 if a frame differs from the live count);
    python -m vint_train.mapmad.dataset_dagger loss --config ... --checkpoint v2=<ckpt> v21=<ckpt> --out <json>
is the DAgger-state loss (report only).
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vint_train.mapmad import modes as M
from vint_train.mapmad.dataset_v2 import MapMaDV2Dataset
from vint_train.mapmad.map_sample_v2 import DriveStoreV2, MapOptions, MapV2Builder

LABELS_FILE = "labels.npz"


def load_labels(data_folder: str, drive: str) -> Tuple[np.ndarray, np.ndarray]:
    """(waypoints_m (N, 8, 2) float32, valid (N,) bool) of one rollout drive."""
    with np.load(os.path.join(data_folder, drive, LABELS_FILE)) as z:
        return z["waypoints_m"].astype(np.float32), z["valid"].astype(bool)


class DaggerDataset(MapMaDV2Dataset):
    """Labelled frames of the DAgger rollouts (module docstring); used through MapMaDV2Dataset.attach_dagger."""

    def _index_path(self) -> str:
        path = super()._index_path()
        return path[:-len(".pkl")] + "_dagger.pkl"

    def _build_index(self, use_tqdm: bool = False):
        """(drive, t, 0) for every labelled frame with full context; goals_index = every frame (the LMDB cache)."""
        samples, goals = [], []
        begin = self.index_context_size * self.waypoint_spacing
        for drive in self.traj_names:
            n = len(self._get_trajectory(drive)["position"])
            goals += [(drive, t) for t in range(n)]
            _, valid = load_labels(self.data_folder, drive)
            samples += [(drive, int(t), 0) for t in np.flatnonzero(valid) if t >= begin]
        return samples, goals

    def _init_maps(self, map_cfg: Dict[str, Any], train: bool) -> None:
        d = map_cfg["dagger"]
        store = DriveStoreV2(self.data_folder, self.traj_names, d["frontier_cache"], int(map_cfg.get("map_cache_size", 64)))
        self.builder = MapV2Builder(store, **map_cfg)
        self.labels: Dict[str, np.ndarray] = {}
        for drive in self.traj_names:
            self.labels[drive] = load_labels(self.data_folder, drive)[0]
        self.seen_p = float(map_cfg.get("seen_oversample", 0.0)) if train else 0.0
        self.pools = {"search": np.arange(len(self.index_to_data))}
        if self.seen_p > 0:
            from vint_train.mapmad.target_groups import detection_frames, seen_out_of_view

            det: Dict[str, np.ndarray] = {}
            seen = []
            for i, (drive, t, _) in enumerate(self.index_to_data):
                if drive not in det:
                    det[drive] = detection_frames(self.data_folder, drive)
                if seen_out_of_view(det[drive], int(t)):
                    seen.append(i)
            self.pools["seen"] = np.asarray(seen, dtype=np.int64)

    def label_actions(self, drive: str, t: int) -> torch.Tensor:
        """The stored label in the network's normalised format (vint_dataset._compute_actions)."""
        scale = self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
        return torch.as_tensor(self.labels[drive][t] / scale, dtype=torch.float32)

    def habitat_sample(self, drive: str, t: int, mode: M.Mode, rng: np.random.Generator,
                       goal_offset: Optional[int] = None, options: MapOptions = MapOptions()) -> Tuple[torch.Tensor, ...]:
        """A search_map sample at labelled frame t: the v2 sample with the label as its action target."""
        if mode.name != "search_map":
            raise ValueError(f"DAgger samples serve the search_map mode only, not {mode.name}")
        base = super().habitat_sample(drive, t, mode, rng, goal_offset, options)
        return base[:2] + (self.label_actions(drive, t),) + base[3:]

    def get(self, i: int, key: int) -> Tuple[torch.Tensor, ...]:
        """Sample i in search_map (offline checks: the DAgger-state loss)."""
        drive, t, _ = self.index_to_data[i]
        return self.habitat_sample(drive, t, M.BY_NAME["search_map"], self.rng(key))


def build_dagger_dataset(config: Dict[str, Any], split: str = "train", train: bool = True) -> DaggerDataset:
    """The DaggerDataset of a train config's `map.dagger` block (the v2 Habitat entry's other settings)."""
    d = config["datasets"]["habitat_mapmad_v2"]
    g = config["map"]["dagger"]
    return DaggerDataset(
        map_cfg=config["map"], habitat_modes=config["habitat_modes"], goal_mask_prob=config["goal_mask_prob"],
        seed=config["seed"], train=train, exclude_drives=None, index_dir=g["index_dir"],
        data_folder=g["data_folder"], data_split_folder=g[split], dataset_name="habitat_mapmad_v2",
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


def union_report(main: MapMaDV2Dataset) -> Dict[str, Any]:
    """Shares of the search_map pool (block item 4: the resulting DAgger share is reported)."""
    n8, nd = len(main.pools["search"]), len(main.dagger.pools["search"])
    out = {"phase8_search_samples": int(n8), "dagger_samples": int(nd), "dagger_share": nd / (n8 + nd),
           "dagger_drives": len(main.dagger.traj_names), "seen_oversample": main.seen_p}
    if main.seen_p > 0:
        s8, sd = len(main.pools["seen"]), len(main.dagger.pools["seen"])
        out.update(phase8_seen=int(s8), dagger_seen=int(sd), dagger_seen_share=sd / (s8 + sd))
    return out


def dagger_loss(cfg: Dict[str, Any], checkpoints: Dict[str, str], n: int, seed: int, device) -> Dict[str, Any]:
    """Report only (block item 6): the offline action loss (offline_scoring.score_arm, arm "maps": the training input,
    fixed per-item noise) of each checkpoint on n seeded labelled states of the DAgger TEST split (dev-home rollouts,
    never trained on); paired difference of the last two checkpoints with a drive-level bootstrap CI."""
    from vint_train.mapmad.offline_scoring import score_arm
    from vint_train.mapmad.weights import load_checkpoint

    ds = build_dagger_dataset(cfg, "test", train=False)
    idx = np.sort(np.random.default_rng(seed).choice(len(ds), min(n, len(ds)), replace=False))
    samples = [{"drive": ds.index_to_data[i][0], "t": int(ds.index_to_data[i][1]), "id": int(i)} for i in idx]
    losses = {}
    for label, path in checkpoints.items():
        model = load_checkpoint(cfg, path, device, map_input=True)
        losses[label] = score_arm(model, ds, samples, "maps", cfg, device, 100, 4)["gc_action_loss"]
    out: Dict[str, Any] = {"n_samples": len(samples), "n_drives": len({s["drive"] for s in samples}), "seed": seed,
                           "means": {k: float(v.mean()) for k, v in losses.items()}}
    if len(losses) >= 2:
        a, b = list(losses)[-2:]
        d = losses[b] - losses[a]
        drives = np.array([s["drive"] for s in samples])
        names = sorted(set(drives.tolist()))
        rng = np.random.default_rng(seed + 1)
        boot = []
        for _ in range(10000):
            pick = rng.choice(len(names), len(names))
            sel = np.concatenate([np.flatnonzero(drives == names[k]) for k in pick])
            boot.append(d[sel].mean())
        out["difference"] = {"arms": f"{b} - {a}", "mean": float(d.mean()),
                             "ci95_drive_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    return out


def main(argv: Optional[List[str]] = None) -> None:
    from vint_train.mapmad.config import load_config
    from vint_train.mapmad.dataset import build_mapmad_dataset

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--config", required=True)
    q.add_argument("--out", required=True)
    q = sub.add_parser("frontier")
    q.add_argument("--config", required=True)
    q.add_argument("--workers", type=int, default=16)
    q.add_argument("--out", required=True)
    q = sub.add_parser("loss")
    q.add_argument("--config", required=True)
    q.add_argument("--checkpoint", nargs="+", required=True, help="label=path (the last two are compared)")
    q.add_argument("--n", type=int, default=2000)
    q.add_argument("--seed", type=int, default=11601)
    q.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "frontier":
        from vint_train.mapmad.frontier_cache import build_all

        g = cfg["map"]["dagger"]
        drives = sorted({x.strip() for split in ("train", "test")
                         for x in open(os.path.join(g[split], "traj_names.txt")) if x.strip()})
        rep = build_all(g["data_folder"], g["frontier_cache"], drives, args.workers)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "mismatched_drives"}, indent=1))
        if rep["mismatched_frames"]:
            raise SystemExit(3)
        return
    if args.cmd == "loss":
        ckpts = dict(c.split("=", 1) for c in args.checkpoint)
        rep = dagger_loss(cfg, ckpts, args.n, args.seed, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))
        return
    main_ds = build_mapmad_dataset(cfg, "habitat_mapmad_v2", "train", train=True)
    rep = union_report(main_ds)
    test = build_dagger_dataset(cfg, "test", train=False)
    rep.update(dagger_test_samples=len(test), dagger_test_drives=len(test.traj_names))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
