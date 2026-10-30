#!/usr/bin/env python3
"""Phase 13 run plan, host side: a command line over mapmad_sim/run_plan_p13.py for the chain's bash.

    python3 p13_plan.py <config> calls <stage>        # one line per call: label|subdir|set|recog|seeds|arms|runs
    python3 p13_plan.py <config> total <stage>        # runs of the stage
    python3 p13_plan.py <config> episodes <set>       # the set's episode ids
    python3 p13_plan.py <config> timing               # the timing call line + its 4 episodes (two lines)
    python3 p13_plan.py <config> video-calls          # 13.1 re-runs: gpu|arm|episode ... (each run on its Phase 12 GPU)

A call's seeds field is "" (each arm's own seeds) or the call's seed indices, space-separated. `video-calls`: the 12
Phase 12 video episodes x the 8 video arms, seed 0, grouped by the GPU that ran that run in Phase 12 (its progress
line's shard: 0-2 = GPU 0, 10-12 = GPU 1), so a re-run can reproduce the Phase 12 log exactly (YOLOE differs across the
two GPUs by float noise, Phase 9).
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mapmad", "src"))

from mapmad_sim.run_plan_p13 import P13Plan  # noqa: E402

LINE = re.compile(r"^\S+ \d+/\d+ (\S+) (\S+__s\d+): ")


def line_of(plan: P13Plan, c) -> str:
    seeds = " ".join(str(k) for k in c.seeds) if c.seeds is not None else ""
    return "|".join([c.label, c.subdir, c.set, c.recog, seeds, " ".join(c.arms), str(plan.runs(c))])


def phase12_gpu(exam_runs: str) -> dict:
    """(arm, run id) -> the GPU of its Phase 12 run (the last progress line wins)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(exam_runs, "progress.*.shard*.log"))):
        shard = int(f.rsplit("shard", 1)[1].split(".")[0])
        for ln in open(f):
            m = LINE.match(ln)
            if m:
                out[(m.group(1), m.group(2))] = shard // 10
    return out


def video_calls(plan: P13Plan) -> list:
    v = plan.cfg["videos"]
    vl = json.loads((plan.base / v["list"]).read_text())
    gpu = phase12_gpu(str(plan.table.outputs / "p12_search_exam" / "runs" / "exam"))
    groups = defaultdict(list)
    for p in vl["picks"]:
        for arm in v["arms"]:
            key = (arm, f"{p['episode']}__s0")
            if key not in gpu:
                raise SystemExit(f"no Phase 12 progress line for {key}")
            groups[(gpu[key], arm)].append(p["episode"])
    return [f"{g}|{arm}|{' '.join(sorted(set(eps)))}" for (g, arm), eps in sorted(groups.items())]


def main() -> None:
    config, cmd = sys.argv[1], sys.argv[2]
    outputs = os.environ.get("P13_OUTPUTS", "/mnt/shared_disk/nazli/nomad_outputs/mapmad")
    plan = P13Plan.load(config, outputs)
    if cmd == "calls":
        for c in plan.calls(sys.argv[3]):
            print(line_of(plan, c))
    elif cmd == "episodes":
        print("\n".join(plan.episode_ids(sys.argv[3])))
    elif cmd == "total":
        print(plan.total(sys.argv[3]))
    elif cmd == "timing":
        c, eps = plan.timing_call()
        print(line_of(plan, c))
        print(" ".join(eps))
    elif cmd == "video-calls":
        print("\n".join(video_calls(plan)))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
