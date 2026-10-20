"""Phase 8.3 full-run statistics for the gate summary (G8 rows 3 and 5).

    python mapmad/scripts/p8_full_stats.py --config mapmad/configs/p8_datagen_v3a.yaml     # naz_mapmad_habitat (CPU)

Reads <mapmad_data>/habitat_mapmad_v2 (_homes/*.json + kept drives) and writes <outputs>/p8_search_data/full/:
- full_numbers.json / .md: the pilot report's numbers (p8_pilot_report.numbers: success, failures by cause, steps,
  spot turns, collisions, detours, wall recoveries, per category, start-filter drop rates), plus
- success and failure causes for the 20 pilot homes vs all other labelled homes (the same drive ids p8d-... in both
  groups: the full run regenerates the pilot homes with the full prefix);
- kept search drives per category (flag: < 500), and collision drives (>= 1 collision step) per kind and category;
- size: drive bytes (sum of `bytes`) vs bytes on disk.
"""

import argparse
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from mapmad_sim import config
from mapmad_sim.run_layout import read_config, write_text_atomic

REPORT = Path(__file__).resolve().parent / "p8_pilot_report.py"
FLAG_BELOW = 500  # kept search drives per category (Naz, 2026-10-20)


def load_report():
    spec = importlib.util.spec_from_file_location("p8_pilot_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def group(search: List[Dict[str, Any]], pct) -> Dict[str, Any]:
    ok = sum(d["outcome"] == "success" for d in search)
    return {"n": ok, "of": len(search), "pct": pct(ok, len(search)),
            "failures": dict(Counter(d["outcome"] for d in search if d["outcome"] != "success"))}


def disk_bytes(root: Path) -> int:
    """Bytes the dataset occupies on disk (allocated blocks, as du)."""
    total = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            total += os.lstat(os.path.join(dirpath, f)).st_blocks * 512
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    cfg = read_config(args.config)
    paths = config.paths()
    data = paths["mapmad_data"] / cfg["datasets"]["full"]
    out = paths["outputs"] / cfg["out_dir"] / "full"
    rep = load_report()
    n = rep.numbers(data, cfg, args.workers)
    s = rep.load_summaries(data)
    search = [d for d in s if d["kind"] == "search" and d["outcome"] != "no_sample"]
    pilot = set(json.loads((paths["outputs"] / cfg["out_dir"] / "pilot" / "pilot_homes.json").read_text())["homes"])
    n["pilot_homes_vs_others"] = {"pilot_20": group([d for d in search if d["home"] in pilot], rep.pct),
                                  "other_labelled": group([d for d in search if d["home"] not in pilot], rep.pct)}
    kept = [d for d in s if d.get("kept")]
    per_cat = Counter(d["category"] for d in kept if d["kind"] == "search")
    n["kept_search_per_category"] = {c: {"kept": per_cat[c], "flag_below_500": per_cat[c] < FLAG_BELOW}
                                     for c in sorted(per_cat)}
    coll = [d for d in kept if d["collision_steps"] > 0]
    n["collision_drives"] = {
        "total": len(coll), "of_kept": len(kept), "pct": rep.pct(len(coll), len(kept)),
        "by_kind": {k: {"n": sum(d["kind"] == k for d in coll), "of": sum(d["kind"] == k for d in kept),
                        "pct": rep.pct(sum(d["kind"] == k for d in coll), sum(d["kind"] == k for d in kept))}
                    for k in ("search", "explore")},
        "search_by_category": {c: rep.pct(sum(d.get("category") == c for d in coll), per_cat[c]) for c in sorted(per_cat)},
        "steps": dict(sorted(Counter(min(d["collision_steps"], 10) for d in coll).items()))}  # 10 = 10 or more
    n["size"] = {"drive_bytes_gb": round(sum(d.get("bytes", 0) for d in kept) / 1e9, 2),
                 "on_disk_gb": round(disk_bytes(data) / 1e9, 2)}
    write_text_atomic(out / "full_numbers.json", json.dumps(n, indent=1) + "\n")
    md = rep.numbers_md(n).replace("# Phase 8 pilot — numbers", "# Phase 8 full run — numbers")
    pv = n["pilot_homes_vs_others"]
    md += "\n".join([
        "", "## The 20 pilot homes vs the other labelled homes (search drives)",
        f"- pilot homes: {pv['pilot_20']['n']}/{pv['pilot_20']['of']} = {pv['pilot_20']['pct']}%, failures {pv['pilot_20']['failures']}",
        f"- other homes: {pv['other_labelled']['n']}/{pv['other_labelled']['of']} = {pv['other_labelled']['pct']}%, failures {pv['other_labelled']['failures']}",
        "", "## Kept search drives per category (flag < 500)",
        *[f"- {c}: {v['kept']}{'  **< 500**' if v['flag_below_500'] else ''}" for c, v in n["kept_search_per_category"].items()],
        "", "## Collision drives (>= 1 collision step; excluded via collision_drives.txt)",
        f"- {n['collision_drives']['total']}/{n['collision_drives']['of_kept']} = {n['collision_drives']['pct']}% of kept; "
        f"by kind {n['collision_drives']['by_kind']}",
        f"- search, % per category: {n['collision_drives']['search_by_category']}",
        f"- collision steps per drive (10 = 10+): {n['collision_drives']['steps']}",
        "", f"## Size: drive bytes {n['size']['drive_bytes_gb']} GB, on disk {n['size']['on_disk_gb']} GB", ""])
    write_text_atomic(out / "full_numbers.md", md)
    print(md)


if __name__ == "__main__":
    main()
