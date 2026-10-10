"""Generate Phase 2 practice drives (configs/p2_datagen.yaml) for one worker's share of the homes.

    python mapmad/scripts/generate_drives.py --dataset pilot --gpu 0 --worker 0 --workers 4   # inside naz_mapmad_habitat
    mapmad/scripts/run_p2_gen.sh pilot 2        # from the host: 2 workers per GPU, each in its own screen

Homes: pilot = the 20 frozen pilot homes; full = all 800 HM3D train homes (training drives + the 40 val-drive
homes' held-out drives; configs/p2_home_splits.json says which is which). Worker k of n takes homes k, k+n, ...
of the sorted list. Home seed = seed + the home's index among the 800 sorted train homes, so the drives do not
depend on how homes are split between workers or GPUs. Resumable: finished homes are skipped (datagen.py).
Writes the drives to <mapmad_data>/<dataset folder>/, progress to <outputs>/p2_datagen/<dataset>/progress_w<k>.jsonl,
and config + seed + git commit to <dataset folder>/_runs/.
"""

import os

# One CPU thread per worker for numpy / OpenMP: with the libraries' default (one thread per core) a worker burnt
# ~35x the CPU time for a slower drive (Phase 2 pilot: 150 CPU-s vs 4.3 CPU-s for the same 4 drives). Set before
# numpy is imported; an explicit value in the environment wins.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from mapmad_sim import config, home_splits  # noqa: E402
from mapmad_sim.datagen import generate_home  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import read_config  # noqa: E402

CONFIG = config.CONFIG_DIR / "p2_datagen.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--dataset", choices=["pilot", "full"], required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--homes", nargs="*", help="only these homes (5-digit prefixes), e.g. for a quick test")
    p.add_argument("--max-homes", type=int, help="stop after this many homes of this worker")
    p.add_argument("--out-name", help="debug: write to <mapmad_data>/<out-name> instead of the dataset folder")
    p.add_argument("--max-drives", type=int, help="debug: at most this many drives per home")
    args = p.parse_args()

    cfg = read_config(args.config)
    paths = config.paths()
    all_homes = home_splits.train_homes(cfg["split"])
    index = {h: i for i, h in enumerate(all_homes)}
    labelled = set(config.labelled_homes(cfg["split"]))
    splits = home_splits.load(config.CONFIG_DIR / cfg["homes"]["splits_file"])["homes"]
    homes = splits["pilot"] if args.dataset == "pilot" else all_homes
    if args.homes:
        homes = home_splits.by_prefix(all_homes, args.homes)
    mine = homes[args.worker::args.workers][:args.max_homes]
    per_home = cfg["pilot_drives_per_home" if args.dataset == "pilot" else "drives_per_home"]

    dataset_dir = paths["mapmad_data"] / (args.out_name or cfg["datasets"][args.dataset])
    progress_dir = paths["outputs"] / cfg["out_dir"] / (args.out_name or args.dataset)
    progress_dir.mkdir(parents=True, exist_ok=True)
    save_run_info(dataset_dir / "_runs", {**vars(args), "config_file": str(args.config), "config": cfg,
                                          "robot": config.robot(), "homes": mine}, cfg["seed"],
                  name=f"run_info_w{args.worker}of{args.workers}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    log = progress_dir / f"progress_w{args.worker}of{args.workers}.jsonl"
    t0 = time.time()
    for n, home in enumerate(mine, 1):
        s = generate_home(cfg, dataset_dir, home, index[home], home in labelled, args.gpu, per_home, args.max_drives)
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "home": home, "n": n, "of": len(mine),
                "drives": s["drives"], "missing": len(s["missing"]), "discards": s["discards"],
                "flag": s["flag_many_discards"], "home_s": s["seconds"], "elapsed_s": round(time.time() - t0, 1)}
        with open(log, "a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
