"""Phase 2 splits: freeze the home lists, and later write NoMaD's traj_names.txt files for a generated dataset.

    python mapmad/scripts/make_p2_splits.py homes                 # -> mapmad/configs/p2_home_splits.json (frozen)
    python mapmad/scripts/make_p2_splits.py traj-names --dataset full    # after generation (also: --dataset pilot)
    python mapmad/scripts/make_p2_splits.py traj-names --dataset full --tiny 4   # format check: 4 + 2 drives
    python mapmad/scripts/make_p2_splits.py collisions --dataset full   # drives with any collision step

`homes`: val-drive homes (10 Phase 1 homes + 30 random unlabelled train homes), pilot homes (20, >= 10 labelled,
none val-drive) and the train homes, each list with its sha256 (mapmad_sim/home_splits.py). Refuses to replace an
existing file unless --force.
`traj-names`: every finished drive of the dataset goes to <mapmad_data>/splits/<dataset>/train/traj_names.txt, or
to .../test/ if its home is a val-drive home (pilot: all to train, plus test = 2 homes for loader checks).
NoMaD's loader writes its LMDB image cache next to these files, i.e. on the data disk, never in the repo.
`collisions`: <mapmad_data>/splits/<dataset>/collision_drives.txt = every drive with >= 1 collision step (one name
per line, sorted), so Phase 3 can leave them out.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mapmad_sim import config, home_splits
from mapmad_sim.run_info import git_state
from mapmad_sim.run_layout import read_config

CONFIG = config.CONFIG_DIR / "p2_datagen.yaml"


def cmd_homes(cfg: dict, force: bool) -> None:
    out = config.CONFIG_DIR / cfg["homes"]["splits_file"]
    if out.exists() and not force:
        raise SystemExit(f"{out} exists (frozen); pass --force to replace it")
    homes = home_splits.train_homes(cfg["split"])
    labelled = config.labelled_homes(cfg["split"])
    lists = home_splits.draw_splits(homes, labelled, cfg["homes"])
    doc = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_state(),
           "rule": cfg["homes"], "n_split_homes": len(homes), "n_labelled": len(labelled),
           "labelled": {k: sum(h in set(labelled) for h in v) for k, v in lists.items()},
           "sha256": {k: home_splits.list_sha256(v) for k, v in lists.items()}, "homes": lists}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: (len(v), doc["labelled"][k], doc["sha256"][k][:16]) for k, v in lists.items()}))
    print(f"wrote {out}")


def tiny(parts: dict, data: Path, n: int) -> dict:
    """First n train and n // 2 test drives with >= 150 frames (NoMaD needs long enough trajectories)."""
    def long_enough(d: str) -> bool:
        return json.loads((data / d / "mapmad_meta.json").read_text())["frames"] >= 150
    return {"train": [d for d in parts["train"] if long_enough(d)][:n],
            "test": [d for d in parts["test"] if long_enough(d)][:max(1, n // 2)]}


def cmd_traj_names(cfg: dict, dataset: str, n_tiny: int = 0) -> None:
    root = config.paths()["mapmad_data"]
    name = cfg["datasets"][dataset]
    data = root / name
    splits = home_splits.load(config.CONFIG_DIR / cfg["homes"]["splits_file"])["homes"]
    drives = sorted(p.name for p in data.iterdir() if p.is_dir() and (p / "traj_data.pkl").exists())
    if dataset == "full":
        held = set(splits["val_drive"])
    else:  # pilot: 2 of its homes as a tiny test split for the format check
        held = set(sorted({d.rsplit("_", 1)[0] for d in drives})[:2])
    parts = {"train": [d for d in drives if d.rsplit("_", 1)[0] not in held],
             "test": [d for d in drives if d.rsplit("_", 1)[0] in held]}
    if n_tiny:
        parts, name = tiny(parts, data, n_tiny), f"{name}_tiny"
    for part, names in parts.items():
        out = root / "splits" / name / part
        out.mkdir(parents=True, exist_ok=True)
        text = "\n".join(names) + "\n"
        (out / "traj_names.txt").write_text(text)
        print(f"{part}: {len(names)} drives -> {out / 'traj_names.txt'} sha256 {hashlib.sha256(text.encode()).hexdigest()}")


def cmd_collisions(cfg: dict, dataset: str) -> None:
    root = config.paths()["mapmad_data"]
    name = cfg["datasets"][dataset]
    hits = []
    for meta in sorted((root / name).glob("*_*/mapmad_meta.json")):
        m = json.loads(meta.read_text())
        if m["collision_steps"] > 0:
            hits.append((m["drive_id"], m["collision_steps"]))
    out = root / "splits" / name / "collision_drives.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{d}\n" for d, _ in hits)
    out.write_text(text)
    steps = sorted(n for _, n in hits)
    print(f"{len(hits)} drives with collisions (steps per drive: min {steps[0]}, median {steps[len(steps) // 2]}, "
          f"max {steps[-1]}) -> {out} sha256 {hashlib.sha256(text.encode()).hexdigest()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["homes", "traj-names", "collisions"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--dataset", choices=["full", "pilot"], default="full")
    p.add_argument("--force", action="store_true")
    p.add_argument("--tiny", type=int, default=0, help="write a tiny split (<name>_tiny) of this many train drives")
    args = p.parse_args()
    cfg = read_config(args.config)
    if args.command == "homes":
        cmd_homes(cfg, args.force)
    elif args.command == "collisions":
        cmd_collisions(cfg, args.dataset)
    else:
        cmd_traj_names(cfg, args.dataset, args.tiny)


if __name__ == "__main__":
    main()
