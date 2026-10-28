#!/usr/bin/env python3
"""Phase 12 run plan, host side: a command line over mapmad_sim/run_plan.py (the plan itself: stages, calls, seeds,
trims; pure Python + PyYAML), for the chain's bash.

    python3 p12_plan.py <config> calls <stage>        # one line per call: label|subdir|set|recog|seeds|arms|runs
    python3 p12_plan.py <config> total <stage>        # runs of the stage (after trims)
    python3 p12_plan.py <config> pair                 # the headline pair: "<maps arm> <maps-hidden arm>"
    python3 p12_plan.py <config> episodes <set>       # the set's episode ids
    python3 p12_plan.py <config> trim <seconds_per_run> <pairs> <budget_s>   # applies trims (driver/trims.json)

A call's seeds field is "" (each arm's own seeds) or the call's seed indices, space-separated.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mapmad", "src"))

from mapmad_sim.run_plan import RunPlan  # noqa: E402


def main() -> None:
    config, cmd = sys.argv[1], sys.argv[2]
    outputs = os.environ.get("P12_OUTPUTS", "/mnt/shared_disk/nazli/nomad_outputs/mapmad")
    plan = RunPlan.load(config, outputs)
    if cmd == "calls":
        for c in plan.calls(sys.argv[3]):
            seeds = " ".join(str(k) for k in c.seeds) if c.seeds is not None else ""
            print("|".join([c.label, c.subdir, c.set, c.recog, seeds, " ".join(c.arms), str(plan.runs(c))]))
    elif cmd == "episodes":  # the episode ids of a set, in file order
        print("\n".join(plan.episode_ids(sys.argv[3])))
    elif cmd == "total":
        print(plan.total(sys.argv[3]))
    elif cmd == "pair":
        print(" ".join(plan.headline_pair()))
    elif cmd == "trim":
        res = plan.apply_trims(float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))
        os.makedirs(os.path.join(plan.base, "driver"), exist_ok=True)
        with open(os.path.join(plan.base, "driver", "trims.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
