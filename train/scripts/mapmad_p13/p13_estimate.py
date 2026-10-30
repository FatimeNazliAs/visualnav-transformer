#!/usr/bin/env python3
"""Phase 13 exam estimate (kickoff 13.7: "time 20 runs first; budget stop only at > 1.5x"), host side.

    python3 p13_estimate.py <p13 dir> <config (host path)>    # -> JSON on stdout (the chain writes driver/timing.json)

Seconds per run: HM3D network arms = the mean wall time of the 24 timing runs (runs/hm3d/timing, M21b + M21bh); MP3D
network arms = the dev runs of M21b + M21bh (runs/dev/timing); F and VF = their own dev runs (MP3D; used for both
sets). Estimate = sum over the exam calls of (runs not yet done) x seconds per run / pairs; pairs = 3 x the GPUs free
for the chain now (driver/gpu_free.json), at least 3.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(HERE, "..", "..", "..")), "mapmad", "src"))

from mapmad_sim.run_plan_p13 import P13Plan  # noqa: E402


def mean_wall(p13: str, subdir: str, arms) -> float:
    rows = [json.loads(ln) for a in arms for f in glob.glob(os.path.join(p13, subdir, "timing", f"{a}.shard*.jsonl"))
            for ln in open(f)]
    return sum(r["wall_s"] for r in rows) / len(rows) if rows else float("nan")


def main() -> None:
    p13, config = sys.argv[1], sys.argv[2]
    plan = P13Plan.load(config, os.path.dirname(p13))
    t = {"hm3d_net": mean_wall(p13, "runs/hm3d", ["M21b", "M21bh"]),
         "mp3d_net": mean_wall(p13, "runs/dev", ["M21b", "M21bh"]),
         "F": mean_wall(p13, "runs/dev", ["F"]), "VF": mean_wall(p13, "runs/dev", ["VF"])}
    for k, v in list(t.items()):
        if v != v:  # no runs: fall back to the HM3D network time
            t[k] = t["hm3d_net"]
    try:
        free = json.load(open(os.path.join(p13, "driver", "gpu_free.json")))["free"]["chain"]
    except (OSError, ValueError, KeyError):
        free = []
    pairs = max(3 * len(free), 3)
    total, per_call = 0.0, {}
    for c in plan.calls("exams"):
        n = plan.runs(c)
        done = sum(len(glob.glob(os.path.join(p13, c.subdir, "summaries", a, "*.json"))) for a in c.arms)
        s = sum(t["F" if a == "F" else "VF" if a == "VF" else ("hm3d_net" if c.set == "hm3d" else "mp3d_net")]
                for a in c.arms) / len(c.arms)
        per_call[c.label] = {"runs": n, "done": done, "s_per_run": round(s, 1), "hours": round(max(n - done, 0) * s / pairs / 3600, 2)}
        total += max(n - done, 0) * s / pairs
    print(json.dumps({"seconds_per_run": {k: round(v, 1) for k, v in t.items()}, "pairs": pairs, "calls": per_call,
                      "estimate_s": round(total), "budget_s": 40 * 3600}, indent=1))


if __name__ == "__main__":
    main()
