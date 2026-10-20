"""Phase 8.3 determinism check: regenerate 5 drives of the full dataset from their ids and compare them byte by byte.

    python mapmad/scripts/p8_regen_check.py --gpu 1 --config mapmad/configs/p8_datagen_v3a.yaml   # naz_mapmad_habitat

Picks 3 search + 2 exploration drives of the finished full dataset (numpy seed 8, different homes), regenerates
exactly those ids (datagen_v2.generate_home with `only`) into <mapmad_data>/p8_regen_check/ (check data, rebuilt every
time), and compares every file of each drive folder: byte-identical, except mapmad_meta.json, where only the timing
fields (seconds, cpu_seconds) may differ. Result: <outputs>/p8_search_data/checks/regen_check.json.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mapmad_sim import config
from mapmad_sim.datagen_v2 import generate_home
from mapmad_sim.run_layout import read_config, write_text_atomic

TIMING = ("seconds", "cpu_seconds")


def pick(data: Path, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """3 search + 2 exploration drives, each from a different home."""
    metas = [json.loads(p.read_text()) for p in sorted(data.glob("*/mapmad_meta.json"))]
    out: List[Dict[str, Any]] = []
    for kind, n in (("search", 3), ("explore", 2)):
        pool = [m for m in metas if m["kind"] == kind and m["home"] not in {o["home"] for o in out}]
        for i in rng.choice(len(pool), n, replace=False):
            if all(pool[i]["home"] != o["home"] for o in out):
                out.append(pool[i])
    return out


def same_drive(a: Path, b: Path) -> Dict[str, Any]:
    """Compare two drive folders file by file."""
    names = sorted({p.name for p in a.iterdir()} | {p.name for p in b.iterdir()})
    differ = []
    for name in names:
        fa, fb = a / name, b / name
        if not (fa.exists() and fb.exists()):
            differ.append(name)
        elif name == "mapmad_meta.json":
            ja, jb = (json.loads(f.read_text()) for f in (fa, fb))
            if {k: v for k, v in ja.items() if k not in TIMING} != {k: v for k, v in jb.items() if k not in TIMING}:
                differ.append(name)
        elif fa.read_bytes() != fb.read_bytes():
            differ.append(name)
    return {"files": len(names), "differ": differ, "pass": not differ}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8)
    args = p.parse_args()
    cfg = read_config(args.config)
    paths = config.paths()
    data = paths["mapmad_data"] / cfg["datasets"]["full"]
    out = paths["mapmad_data"] / "p8_regen_check"
    if out.exists():  # our own check output
        shutil.rmtree(out)
    chosen = pick(data, np.random.default_rng(args.seed))
    rows = []
    for m in chosen:
        home_json = json.loads((data / "_homes" / f"{m['home']}.json").read_text())
        labelled = bool(home_json["labelled"])
        counts = cfg["drives_v2"]["labelled" if labelled else "unlabelled"]
        generate_home(cfg, out, m["home"], labelled, args.gpu, cfg["id_prefix"]["full"], counts, False, [m["drive_id"]])
        row = {"drive": m["drive_id"], "kind": m["kind"], "outcome": m["outcome"],
               **same_drive(data / m["drive_id"], out / m["drive_id"])}
        rows.append(row)
        print(json.dumps(row), flush=True)
    path = paths["outputs"] / cfg["out_dir"] / "checks" / "regen_check.json"
    write_text_atomic(path, json.dumps({"config": str(args.config), "drives": rows,
                                        "pass": all(r["pass"] for r in rows)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
