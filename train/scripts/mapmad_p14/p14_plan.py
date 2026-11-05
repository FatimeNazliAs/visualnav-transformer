#!/usr/bin/env python3
"""Phase 14a (and 14b-3: configs p14b3_*) run plan, host side: a command line over mapmad_sim/run_plan_p14.py for the chain's bash.

    python3 p14_plan.py <config> calls <stage>     # one line per call: label|subdir|set|recog|seeds|arms|runs|episodes
    python3 p14_plan.py <config> total <stage>     # runs of the stage
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mapmad", "src"))

from mapmad_sim.run_plan_p14 import P14Plan  # noqa: E402
from mapmad_sim.run_plan_p14b3 import plan_for  # noqa: E402


def line_of(plan: P14Plan, c) -> str:
    seeds = " ".join(str(k) for k in c.seeds) if c.seeds is not None else ""
    return "|".join([c.label, c.subdir, c.set, c.recog, seeds, " ".join(c.arms), str(plan.runs(c)), " ".join(c.episodes)])


def main() -> None:
    config, cmd = sys.argv[1], sys.argv[2]
    if "P14_OUTPUTS" not in os.environ:
        raise SystemExit("set P14_OUTPUTS (the mapmad outputs folder; p14_lib.sh sets it from the naz_mapmad mount)")
    plan = plan_for(config).load(config, os.environ["P14_OUTPUTS"])  # Phase 14b-3 configs: P14b3Plan
    if cmd == "calls":
        for c in plan.calls(sys.argv[3]):
            print(line_of(plan, c))
    elif cmd == "total":
        print(plan.total(sys.argv[3]))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
