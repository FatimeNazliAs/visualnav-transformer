"""Two arms on the episodes both solve (report only; from the exam's runs.csv, no new runs). Inside naz_mapmad:

    python -m vint_train.mapmad.closed_loop.exam_overlap --config /app/visualnav-transformer/mapmad/configs/p5_exam.yaml \
        --a C --b E --seed-index 0

Per episode: arm a's run at --seed-index and arm b's run standing for that index (the planner: its index-0 run).
Reports how many of a's successes b also solved and vice versa, and on the episodes both solve: steps to success
(the run stops at success, so steps = the success step; 4 steps = 1 s) and SPL as medians per arm, the median of
the paired differences a - b, and the mean paired difference a - b with a home-clustered bootstrap 95% CI (10,000
resamples). Writes <out_dir>/<tables>/overlap_<a>_<b>.md and .json.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np

from mapmad_sim.exam_arms import arms_of
from mapmad_sim.run_layout import RunLayout
from vint_train.mapmad.closed_loop import exam_analysis as A
from vint_train.mapmad.closed_loop import exam_stats as S


def runs_at(rows, arm: str, seed_index: int) -> Dict[str, Dict[str, Any]]:
    """episode id -> the arm's runs.csv row at seed_index."""
    return {r["episode_id"]: r for r in rows if r["arm"] == arm and int(r["seed_index"]) == seed_index}


def overlap(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]], home_of: Dict[str, str],
            boot_seed: int = 0) -> Dict[str, Any]:
    """Counts of shared successes and the paired comparison on them (module docstring)."""
    ok_a = {i for i, r in a.items() if float(r["success"]) == 1.0}
    ok_b = {i for i, r in b.items() if float(r["success"]) == 1.0}
    both = sorted(ok_a & ok_b & set(a) & set(b))
    out: Dict[str, Any] = {"episodes": len(set(a) & set(b)), "a_successes": len(ok_a), "b_successes": len(ok_b),
                           "both": len(both), "a_only": len(ok_a - ok_b), "b_only": len(ok_b - ok_a)}
    for m in ("steps", "spl"):
        va = {i: float(a[i][m]) for i in both}
        vb = {i: float(b[i][m]) for i in both}
        d = np.array([va[i] - vb[i] for i in both])
        out[m] = {"median_a": float(np.median(list(va.values()))), "median_b": float(np.median(list(vb.values()))),
                  "median_paired_diff": float(np.median(d)), "paired": S.paired(va, vb, home_of, seed=boot_seed),
                  "a_better_share": float(np.mean(d < 0) if m == "steps" else np.mean(d > 0))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tables", default="tables")
    p.add_argument("--a", default="C")
    p.add_argument("--b", default="E")
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--boot-seed", type=int, default=0)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    tables = layout.base / args.tables
    with open(tables / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    arm_b = arms_of(layout.cfg, [args.b])[0]
    a = runs_at(rows, args.a, args.seed_index)
    b = runs_at(rows, args.b, arm_b.run_seed_index(args.seed_index))
    home_of = {r["episode_id"]: r["home"] for r in rows}
    res = overlap(a, b, home_of, args.boot_seed)
    x, y = args.a, args.b
    lines = [f"# {x} vs {y} on the episodes both solve (report only)", "",
             f"From `{tables / 'runs.csv'}`: {x} at seed index {args.seed_index}, {y} at seed index "
             f"{arm_b.run_seed_index(args.seed_index)}; {res['episodes']} episodes. Paired = {x} - {y} on the same "
             f"episodes; mean [95% CI, home-clustered bootstrap, {S.N_BOOT:,} resamples, seed {args.boot_seed}].", "",
             f"- {x} successes: {res['a_successes']}; of these {y} also solved {res['both']} "
             f"({res['both'] / max(res['a_successes'], 1):.0%}).",
             f"- {y} successes: {res['b_successes']}; of these {x} also solved {res['both']} "
             f"({res['both'] / max(res['b_successes'], 1):.0%}).",
             f"- Both solve: {res['both']} episodes ({res['steps']['paired']['homes']} homes); "
             f"only {x}: {res['a_only']}; only {y}: {res['b_only']}.", "",
             f"| on both-solved episodes | median {x} | median {y} | median of paired {x} - {y} | mean paired {x} - {y} "
             f"[home CI] | {x} better in |", "|---|---|---|---|---|---|"]
    for m, name, fmt in (("steps", "steps to success (4 = 1 s)", "{:.0f}"), ("spl", "SPL", "{:.3f}")):
        r = res[m]
        q = r["paired"]
        lines.append(f"| {name} | {fmt.format(r['median_a'])} | {fmt.format(r['median_b'])} | "
                     f"{r['median_paired_diff']:+.3f} | {q['diff']:+.3f} [{q['home_ci'][0]:+.3f}, {q['home_ci'][1]:+.3f}] | "
                     f"{r['a_better_share']:.0%} of episodes |")
    out = tables / f"overlap_{x}_{y}.md"
    out.write_text("\n".join(lines) + "\n")
    A.write_json(out.with_suffix(".json"), res)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
