"""Generate Phase 8 drives (configs/p8_datagen.yaml): one worker taking homes from a shared queue.

    python mapmad/scripts/generate_drives_v2.py --dataset pilot --gpu 1 --token <launch>   # inside naz_mapmad_habitat
    mapmad/scripts/run_p8_gen.sh pilot                 # from the host: the driver (screen, exit trap, workers)

Homes: pilot = 20 labelled train homes outside the val-drive split (seed 808; frozen in <out>/pilot/pilot_homes.json);
full = all 800 HM3D train homes (the 40 val-drive homes' drives go to the test split later). The queue = the homes
sorted by drive count (labelled homes first), then by name; a worker takes the next home nobody has claimed
(<dataset>/_claims/<token>/<home>, run_layout.try_claim) and is not finished (<dataset>/_homes/<home>.json). Every drive
is seeded by its id (datagen_v2.drive_seed), so results do not depend on the number of workers, GPUs or the order.
A relaunch uses a new token: homes claimed but unfinished by a stopped launch are redone from their seeds.
GPU withdrawal: when the driver finds the worker's GPU no longer free (another user's job), it writes
<stop-dir>/STOP_gpu<k>_<token> (--stop-dir = the driver's run folder); before its next drive the worker gives its home back (datagen_v2.withdraw_home:
incomplete drives -> _partial/<home>__<token>, claim removed) and exits; workers on free GPUs pick the home up again
(they repeat the queue pass until a pass claims nothing).
Writes drives to <mapmad_data>/<dataset folder>/, progress to <outputs>/p8_search_data/<dataset>/progress_<worker>.jsonl,
config + seed + git commit to <dataset folder>/_runs/.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one CPU thread per worker (Phase 2 lesson, generate_drives.py)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, Iterator, List  # noqa: E402

import numpy as np  # noqa: E402

from mapmad_sim import config, home_splits, objectnav  # noqa: E402
from mapmad_sim.datagen_v2 import HomeWithdrawn, generate_home, withdraw_home  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import read_config, try_claim, write_text_atomic  # noqa: E402

CONFIG = config.CONFIG_DIR / "p8_datagen.yaml"


def labelled_homes(cfg: Dict[str, Any], homes: List[str]) -> List[str]:
    """Homes with semantic labels AND ObjectNav goals (the 145 labelled train homes)."""
    lab = set(config.labelled_homes(cfg["split"]))
    return [h for h in homes if h in lab and objectnav.has_goals(cfg["objectnav_split"], h)]


def pilot_homes(cfg: Dict[str, Any], all_homes: List[str], labelled: List[str]) -> List[str]:
    """Block item 6: 20 labelled train homes, none val-drive (seed 808), sorted."""
    splits = home_splits.load(config.CONFIG_DIR / cfg["homes"]["splits_file"])["homes"]
    pool = [h for h in labelled if h not in set(splits["val_drive"])]
    rule = cfg["homes"]["pilot_v2"]
    rng = np.random.default_rng(rule["seed"])
    return sorted(pool[i] for i in rng.choice(len(pool), rule["n_homes"], replace=False))


def queue(homes: List[str], labelled: set, per_home: Dict[str, Dict[str, int]]) -> List[str]:
    """Homes ordered by drive count (largest first), then name."""
    n = {h: sum(per_home["labelled" if h in labelled else "unlabelled"].values()) for h in homes}
    return sorted(homes, key=lambda h: (-n[h], h))


def gpu_stop_file(stop_dir: Path, gpu: int, token: str) -> Path:
    """The driver's withdrawal flag for one GPU of one launch (run_p8_gen.sh writes it)."""
    return stop_dir / f"STOP_gpu{gpu}_{token}"


def queue_passes(order: List[str], dataset_dir: Path, claims: Path,
                 stop: Callable[[], bool] = lambda: False) -> Iterator[str]:
    """Homes this worker claims, in queue order; passes repeat until one claims nothing, so a home given back by a
    withdrawn worker (claim removed) is picked up again. `stop` (the GPU's withdrawal flag) is checked before every
    claim: a withdrawn worker never claims another home."""
    while True:
        claimed = False
        for home in order:
            if stop():
                return
            if (dataset_dir / "_homes" / f"{home}.json").exists():
                continue
            if not try_claim(claims / home):
                continue
            claimed = True
            yield home
        if not claimed:
            return


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--dataset", choices=["pilot", "full"], required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--token", required=True, help="launch id: claims of this launch live under _claims/<token>/")
    p.add_argument("--worker", default="w0", help="name for the progress file")
    p.add_argument("--homes", nargs="*", help="debug: only these homes (5-digit prefixes)")
    p.add_argument("--out-name", help="debug / checks: write to <mapmad_data>/<out-name> instead of the dataset folder")
    p.add_argument("--only", nargs="*", help="debug / byte-identity: only these drive ids")
    p.add_argument("--count-unfinished", action="store_true", help="print the number of unfinished homes and exit")
    p.add_argument("--stop-dir", type=Path, help="driver's run folder: STOP_gpu<k>_<token> there withdraws this worker")
    p.add_argument("--counts", help='override drives per labelled home, e.g. \'{"search": 40, "explore": 4}\' (pro-rata cut)')
    args = p.parse_args()

    cfg = read_config(args.config)
    paths = config.paths()
    all_homes = home_splits.train_homes(cfg["split"])
    labelled = labelled_homes(cfg, all_homes)
    per_home = {k: dict(v) for k, v in cfg["pilot_drives_v2" if args.dataset == "pilot" else "drives_v2"].items()}
    if args.counts:
        over = json.loads(args.counts)
        per_home["labelled"].update(over)
        per_home["unlabelled"]["explore"] = over.get("explore", per_home["unlabelled"]["explore"])
    out_root = paths["outputs"] / cfg["out_dir"]
    homes = pilot_homes(cfg, all_homes, labelled) if args.dataset == "pilot" else all_homes
    if args.dataset == "pilot":
        frozen = out_root / "pilot" / "pilot_homes.json"
        if not frozen.exists():
            write_text_atomic(frozen, json.dumps({"rule": cfg["homes"]["pilot_v2"], "homes": homes}, indent=1) + "\n")
        elif json.loads(frozen.read_text())["homes"] != homes:
            raise SystemExit(f"{frozen}: pilot homes changed")
    if args.homes:  # debug / checks: any train homes (e.g. dev homes for the map parity check)
        homes = home_splits.by_prefix(all_homes, args.homes)
    order = queue(homes, set(labelled), per_home)

    name = args.out_name or cfg["datasets"][args.dataset]
    dataset_dir = paths["mapmad_data"] / name
    if args.count_unfinished:  # the driver's end check: homes of this queue without _homes/<home>.json
        print(sum(not (dataset_dir / "_homes" / f"{h}.json").exists() for h in order))
        return
    progress_dir = out_root / "checks" / args.out_name if args.out_name else out_root / args.dataset
    progress_dir.mkdir(parents=True, exist_ok=True)
    save_run_info(dataset_dir / "_runs", {**vars(args), "config_file": str(args.config), "config": cfg,
                                          "robot": config.robot(), "per_home": per_home, "queue": order}, cfg["seed"],
                  name=f"run_info_{args.token}_{args.worker}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    log = progress_dir / f"progress_{args.worker}.jsonl"
    prefix = cfg["id_prefix"][args.dataset]
    keep_failed = bool(cfg["record"]["keep_failed"][args.dataset])
    t0 = time.time()
    n = 0
    claims = dataset_dir / "_claims" / args.token
    stop_file = gpu_stop_file(args.stop_dir, args.gpu, args.token) if args.stop_dir else Path("/nonexistent/STOP")
    for home in queue_passes(order, dataset_dir, claims, stop_file.exists):
        lab = home in set(labelled)
        try:
            s = generate_home(cfg, dataset_dir, home, lab, args.gpu, prefix,
                              per_home["labelled" if lab else "unlabelled"], keep_failed, args.only,
                              should_stop=stop_file.exists)
        except HomeWithdrawn:
            dest = withdraw_home(dataset_dir, home, claims / home, args.token)
            line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "worker": args.worker, "gpu": args.gpu, "home": home,
                    "withdrawn": True, "partial": str(dest) if dest else None}
            with open(log, "a") as f:
                f.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)
            return
        n += 1
        outcomes: Dict[str, int] = {}
        for d in s["summaries"]:
            outcomes[d["outcome"]] = outcomes.get(d["outcome"], 0) + 1
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "worker": args.worker, "gpu": args.gpu, "home": home,
                "n": n, "kept": s["drives"], "outcomes": outcomes, "home_s": s["seconds"],
                "elapsed_s": round(time.time() - t0, 1)}
        with open(log, "a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
