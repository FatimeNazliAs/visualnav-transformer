#!/usr/bin/env python3
"""Phase 12 run plan: which closed-loop calls the chain makes, in which order, and how many runs each has (host side:
standard library + PyYAML; the arm table itself, mapmad_sim/arm_table.py, is pure Python).

    python3 p12_plan.py <config> calls <stage>        # one line per call: label|subdir|set|recog|seeds|arms
    python3 p12_plan.py <config> total <stage>        # runs of the stage (after trims)
    python3 p12_plan.py <config> episodes <set>       # the set's episode ids
    python3 p12_plan.py <config> trim <seconds_per_run> <pairs> <budget_s>   # applies trims (driver/trims.json)

Stages: dev_hidden, dev_tau, exam, ovon, explore, unseeable, qwen (block item 9 order; exam = headline pair -> N -> V1
-> F -> other pair -> truth arm -> N-fish). A call's `seeds` is "" (each arm's own seeds from the arm table) or a
space-separated list (trim 2 / 4: that arm alone, seed 0). Trims (block item 9, in this order, each logged, applied
only while the estimate of the exam stages (exam + ovon + explore + unseeable; Qwen3-VL excluded) is over the budget):
(1) drop N-fish; (2) OVON N -> 1 seed; (3) drop exploration V1; (4) the non-headline maps-hidden arm -> 1 seed.
The estimate = runs x seconds per run / pairs (seconds per run and pairs measured by the chain's 20-run timing).
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mapmad", "src"))

import yaml  # noqa: E402

from mapmad_sim.arm_table import ArmTable  # noqa: E402

TRIMS = ("drop_nfish", "ovon_n_1seed", "drop_explore_v1", "other_hidden_1seed")
EXAM_STAGES = ("exam", "ovon", "explore", "unseeable")
Call = Tuple[str, str, str, str, str, List[str]]  # label, subdir, set, recog, seeds, arms


def read_config(path: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(open(path))
    base = cfg.pop("extends", None)
    if not base:
        return cfg

    def merge(a, b):
        out = dict(a)
        for k, v in b.items():
            if v is None:
                out.pop(k, None)
            elif isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    return merge(read_config(os.path.join(os.path.dirname(path), base)), cfg)


class Plan:
    def __init__(self, config: str, outputs: str) -> None:
        self.cfg = read_config(config)
        self.base = os.path.join(outputs, self.cfg["out_dir"])
        self.table = ArmTable.from_config(self.cfg, self.base, outputs, REPO)
        self.trims = self._trims()

    def _trims(self) -> List[str]:
        f = os.path.join(self.base, "driver", "trims.json")
        return json.load(open(f))["trims"] if os.path.exists(f) else []

    def pairs(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        ch = self.table.choices()
        hl = (ch or {}).get("headline", "v21")
        p = {"v2": ("M2", "M2h"), "v21": ("M21", "M21h")}
        return p[hl], p["v2" if hl == "v21" else "v21"]

    def n_episodes(self, set_name: str) -> int:
        f = os.path.join(self.base, self.cfg["sets"][set_name]["folder"], "episodes.json")
        return len(json.load(open(f))["episodes"]) if os.path.exists(f) else 0

    def calls(self, stage: str) -> List[Call]:
        (hm, hh), (om, oh) = self.pairs()
        t = self.trims
        if stage == "dev_hidden":
            return [("dev_hidden", "dev/hidden", "dev", "none", "", ["v2h_dev", "v21h_dev"])]
        if stage == "dev_tau":
            return [("dev_tau", "dev/tau", "dev", "yoloe", "", ["hl_yoloe_dev"])]
        if stage == "exam":
            out = [("exam_pair", "runs/exam", "exam", "yoloe", "", [hm, hh]),
                   ("exam_N", "runs/exam", "exam", "yoloe", "", ["N"]),
                   ("exam_V1", "runs/exam", "exam", "yoloe", "", ["V1"]),
                   ("exam_F", "runs/exam", "exam", "yoloe", "", ["F"])]
            if "other_hidden_1seed" in t:
                out += [("exam_other", "runs/exam", "exam", "yoloe", "", [om]),
                        ("exam_other_h", "runs/exam", "exam", "yoloe", "0", [oh])]
            else:
                out += [("exam_other", "runs/exam", "exam", "yoloe", "", [om, oh])]
            out += [("exam_MT", "runs/exam", "exam", "none", "", ["MT"])]
            if "drop_nfish" not in t:
                out += [("exam_Nfish", "runs/exam", "exam", "none", "", ["Nfish"])]
            return out
        if stage == "ovon":
            if "ovon_n_1seed" in t:
                return [("ovon_main", "runs/ovon", "ovon", "yoloe", "", [hm, hh, "F"]),
                        ("ovon_N", "runs/ovon", "ovon", "yoloe", "0", ["N"])]
            return [("ovon_main", "runs/ovon", "ovon", "yoloe", "", [hm, hh, "N", "F"])]
        if stage == "explore":
            arms = [f"X_{hm}", f"X_{hh}", "X_N"] + ([] if "drop_explore_v1" in t else ["X_V1"]) + ["X_F"]
            return [("explore", "runs/explore", "explore", "none", "", arms)]
        if stage == "unseeable":
            return [("unseeable", "runs/unseeable", "unseeable", "yoloe", "0", [hm, hh, "N"])]
        if stage == "qwen":
            return [("qwen", "runs/qwen", "qwen", "qwen3vl", "", ["MR1"])]
        raise SystemExit(f"unknown stage {stage}")

    def runs(self, call: Call) -> int:
        n = self.n_episodes(call[2])
        seeds = [int(s) for s in call[4].split()] if call[4] else list(range(int(self.cfg["run"]["seeds"])))
        return sum(n * len(self.table.seeds(a, seeds)) for a in call[5])

    def total(self, stage: str) -> int:
        return sum(self.runs(c) for c in self.calls(stage))


def main() -> None:
    config, cmd = sys.argv[1], sys.argv[2]
    outputs = os.environ.get("P12_OUTPUTS", "/mnt/shared_disk/nazli/nomad_outputs/mapmad")
    plan = Plan(config, outputs)
    if cmd == "calls":
        for c in plan.calls(sys.argv[3]):
            print("|".join([c[0], c[1], c[2], c[3], c[4], " ".join(c[5]), str(plan.runs(c))]))
    elif cmd == "episodes":  # the episode ids of a set, in file order
        f = os.path.join(plan.base, plan.cfg["sets"][sys.argv[3]]["folder"], "episodes.json")
        print("\n".join(e["episode_id"] for e in json.load(open(f))["episodes"]))
    elif cmd == "total":
        print(plan.total(sys.argv[3]))
    elif cmd == "trim":
        per_run, pairs, budget = float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
        log = []

        def estimate() -> float:
            return sum(plan.total(s) for s in EXAM_STAGES) * per_run / max(pairs, 1)

        est0 = estimate()
        for t in TRIMS:
            if estimate() <= budget:
                break
            before = estimate()
            plan.trims = plan.trims + [t] if t not in plan.trims else plan.trims
            log.append({"trim": t, "estimate_before_s": round(before), "estimate_after_s": round(estimate())})
        res = {"trims": plan.trims, "log": log, "estimate_untrimmed_s": round(est0), "estimate_s": round(estimate()),
               "budget_s": budget, "seconds_per_run": per_run, "pairs": pairs,
               "runs": {s: plan.total(s) for s in EXAM_STAGES}}
        os.makedirs(os.path.join(plan.base, "driver"), exist_ok=True)
        with open(os.path.join(plan.base, "driver", "trims.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res))
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
