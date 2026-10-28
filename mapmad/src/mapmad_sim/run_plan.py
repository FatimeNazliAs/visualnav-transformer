"""One Phase 12 run plan (clean-up candidate 1): which closed-loop calls a stage makes, with which arms and seeds, after
the trims -- answered in one place for the chain (train/scripts/mapmad_p12/p12_plan.py, host), the fairness check
(p12_score), the tables (p12_tables) and the video picks (p12_videos).

    plan = RunPlan.from_layout(layout)            # or RunPlan.load(config_file, outputs) on the host
    plan.calls("exam")                            # [Call(label, subdir, set, recog, seeds, arms), ...] in run order
    plan.headline_pair()                          # ("M21", "M21h") from P12_DEV_CHOICES.json
    plan.report_calls()                           # every call of the exam stages + Qwen3-VL (fairness, scores)

Stages: dev_hidden, dev_tau, exam, ovon, explore, unseeable, qwen (block item 9 order; exam = headline pair -> N -> V1
-> F -> other pair -> truth arm -> N-fish). A call's `seeds` is None (each arm's own seeds from the arm table) or the
seed indices of a trimmed / 1-seed call (that call's arms, those seeds). Trims (block item 9, in this order, each
logged, applied only while the estimate of the exam stages (exam + ovon + explore + unseeable; Qwen3-VL excluded) is
over the budget): (1) drop N-fish; (2) OVON N -> 1 seed; (3) drop exploration V1; (4) the non-headline maps-hidden arm
-> 1 seed. The estimate = runs x seconds per run / pairs. Applied trims live in <out_dir>/driver/trims.json.
Pure Python + PyYAML: the host and both containers use it.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from mapmad_sim.arm_table import ArmTable

TRIMS = ("drop_nfish", "ovon_n_1seed", "drop_explore_v1", "other_hidden_1seed")
EXAM_STAGES = ("exam", "ovon", "explore", "unseeable")
REPORT_STAGES = EXAM_STAGES + ("qwen",)
PAIRS = {"v2": ("M2", "M2h"), "v21": ("M21", "M21h")}  # checkpoint pair -> (maps, maps hidden)
TRIMS_FILE = "driver/trims.json"


class Call(NamedTuple):
    """One closed-loop call of the chain: these arms on this set, outputs below <out_dir>/<subdir>."""

    label: str
    subdir: str
    set: str
    recog: str
    seeds: Optional[Tuple[int, ...]]
    arms: Tuple[str, ...]


class RunPlan:
    """The calls of every stage of one Phase 12 config (module docstring)."""

    def __init__(self, cfg: Dict[str, Any], table: ArmTable, trims: Optional[List[str]] = None) -> None:
        self.cfg, self.table, self.base = cfg, table, table.base
        f = self.base / TRIMS_FILE
        self.trims = list(trims) if trims is not None else (json.loads(f.read_text())["trims"] if f.exists() else [])

    @classmethod
    def from_layout(cls, layout: Any) -> "RunPlan":
        return cls(layout.cfg, ArmTable.from_layout(layout))

    @classmethod
    def load(cls, config_file: Path, outputs: Path) -> "RunPlan":
        """The plan of a config file, outputs below `outputs` (host side: no paths.yaml lookup)."""
        from mapmad_sim.run_layout import REPO, read_config

        cfg = read_config(Path(config_file))
        return cls(cfg, ArmTable.from_config(cfg, Path(outputs) / cfg["out_dir"], Path(outputs), REPO))

    # --- pairs ------------------------------------------------------------------------------------------------------
    def headline(self) -> str:
        """The headline checkpoint pair (P12_DEV_CHOICES.json; v21 before the dev decision)."""
        return (self.table.choices() or {}).get("headline", "v21")

    def headline_pair(self) -> Tuple[str, str]:
        return PAIRS[self.headline()]

    def other_pair(self) -> Tuple[str, str]:
        return PAIRS["v2" if self.headline() == "v21" else "v21"]

    # --- calls ------------------------------------------------------------------------------------------------------
    def calls(self, stage: str) -> List[Call]:
        (hm, hh), (om, oh) = self.headline_pair(), self.other_pair()
        t = self.trims
        if stage == "dev_hidden":
            return [Call("dev_hidden", "dev/hidden", "dev", "none", None, ("v2h_dev", "v21h_dev"))]
        if stage == "dev_tau":
            return [Call("dev_tau", "dev/tau", "dev", "yoloe", None, ("hl_yoloe_dev",))]
        if stage == "exam":
            out = [Call("exam_pair", "runs/exam", "exam", "yoloe", None, (hm, hh)),
                   Call("exam_N", "runs/exam", "exam", "yoloe", None, ("N",)),
                   Call("exam_V1", "runs/exam", "exam", "yoloe", None, ("V1",)),
                   Call("exam_F", "runs/exam", "exam", "yoloe", None, ("F",))]
            if "other_hidden_1seed" in t:
                out += [Call("exam_other", "runs/exam", "exam", "yoloe", None, (om,)),
                        Call("exam_other_h", "runs/exam", "exam", "yoloe", (0,), (oh,))]
            else:
                out += [Call("exam_other", "runs/exam", "exam", "yoloe", None, (om, oh))]
            out += [Call("exam_MT", "runs/exam", "exam", "none", None, ("MT",))]
            if "drop_nfish" not in t:
                out += [Call("exam_Nfish", "runs/exam", "exam", "none", None, ("Nfish",))]
            return out
        if stage == "ovon":
            if "ovon_n_1seed" in t:
                return [Call("ovon_main", "runs/ovon", "ovon", "yoloe", None, (hm, hh, "F")),
                        Call("ovon_N", "runs/ovon", "ovon", "yoloe", (0,), ("N",))]
            return [Call("ovon_main", "runs/ovon", "ovon", "yoloe", None, (hm, hh, "N", "F"))]
        if stage == "explore":
            arms = (f"X_{hm}", f"X_{hh}", "X_N") + (() if "drop_explore_v1" in t else ("X_V1",)) + ("X_F",)
            return [Call("explore", "runs/explore", "explore", "none", None, arms)]
        if stage == "unseeable":
            return [Call("unseeable", "runs/unseeable", "unseeable", "yoloe", (0,), (hm, hh, "N"))]
        if stage == "qwen":
            return [Call("qwen", "runs/qwen", "qwen", "qwen3vl", None, ("MR1",))]
        raise ValueError(f"unknown stage {stage}")

    def report_calls(self) -> List[Call]:
        """Every call whose runs the tables report and the fairness check covers."""
        return [c for s in REPORT_STAGES for c in self.calls(s)]

    def stage_arms(self, stage: str, order: Sequence[str]) -> List[str]:
        """The arms a stage runs, in the given display order (arms not in `order` are left out)."""
        ran = {a for c in self.calls(stage) for a in c.arms}
        return [a for a in order if a in ran]

    def where(self, stage: str) -> Tuple[str, str]:
        """(subdir, set) of a stage (every call of a reported stage shares them)."""
        c = self.calls(stage)[0]
        return c.subdir, c.set

    # --- runs -------------------------------------------------------------------------------------------------------
    def n_episodes(self, set_name: str) -> int:
        f = self.base / self.cfg["sets"][set_name]["folder"] / "episodes.json"
        return len(json.loads(f.read_text())["episodes"]) if f.exists() else 0

    def episode_ids(self, set_name: str) -> List[str]:
        f = self.base / self.cfg["sets"][set_name]["folder"] / "episodes.json"
        return [e["episode_id"] for e in json.loads(f.read_text())["episodes"]]

    def arm_seeds(self, call: Call, arm: str) -> List[int]:
        """The seed indices `arm` runs in this call (the call's seeds, else run.seeds; limited by the arm table)."""
        given = list(call.seeds) if call.seeds is not None else list(range(int(self.cfg["run"]["seeds"])))
        return self.table.seeds(arm, given)

    def runs(self, call: Call) -> int:
        n = self.n_episodes(call.set)
        return sum(n * len(self.arm_seeds(call, a)) for a in call.arms)

    def total(self, stage: str) -> int:
        return sum(self.runs(c) for c in self.calls(stage))

    # --- trims ------------------------------------------------------------------------------------------------------
    def apply_trims(self, per_run_s: float, pairs: int, budget_s: float) -> Dict[str, Any]:
        """Adds trims in block item 9 order while the exam stages' estimate is over the budget; returns the record
        written to driver/trims.json (the caller writes it)."""
        def estimate() -> float:
            return sum(self.total(s) for s in EXAM_STAGES) * per_run_s / max(pairs, 1)

        est0, log = estimate(), []
        for t in TRIMS:
            if estimate() <= budget_s:
                break
            before = estimate()
            self.trims = self.trims + [t] if t not in self.trims else self.trims
            log.append({"trim": t, "estimate_before_s": round(before), "estimate_after_s": round(estimate())})
        return {"trims": self.trims, "log": log, "estimate_untrimmed_s": round(est0), "estimate_s": round(estimate()),
                "budget_s": budget_s, "seconds_per_run": per_run_s, "pairs": pairs,
                "runs": {s: self.total(s) for s in EXAM_STAGES}}
