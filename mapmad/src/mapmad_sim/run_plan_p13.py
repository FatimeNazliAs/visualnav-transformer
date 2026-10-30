"""The Phase 13 run plan (kickoff §4, p13_rules.md §8): which closed-loop calls a stage makes, with which arms and
seeds -- one place for the chain (train/scripts/mapmad_p13/p13_plan.py, host), the fairness check and the tables
(vint_train.mapmad.closed_loop.p13_tables).

    plan = P13Plan.load(config_file, outputs)       # configs/p13_claim.yaml
    plan.calls("exams")                             # [Call(label, subdir, set, recog, seeds, arms), ...] in run order
    plan.runs(call)                                 # episodes x seeds of every arm (deterministic arms: 1 seed)

Stages: `dev` (13.6: MP3D dev set, report only, seed 0), `exams` (13.7, in the kickoff's order: HM3D M21b / M21bh ->
exam B M21 / M21h -> exam B M21b / M21bh -> exam B N, F, VF -> HM3D VF). A call's recog is "yoloe" (one YOLOE sidecar
per pair) or "yoloe+itm" (VF: plus one BLIP-2 ITM server per GPU). `timing` = the first 4 HM3D episodes of the first
call (20-run timing: 4 episodes x 2 arms x 3 seeds = 24 runs, part of the exam); MP3D and VF seconds per run come from
the dev runs (13.6). The old HM3D arms (M21, M21h, M2,
M2h, N, F, MT, V1) are NOT re-run: their Phase 12 logs are reused (p13_rules.md §6). Pure Python + PyYAML.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from mapmad_sim.arm_table import ArmTable

STAGES = ("dev", "exams")
HM3D_OLD = ("M21", "M21h", "M2", "M2h", "N", "F", "MT", "V1")  # Phase 12 logs, reused
G13_ROWS = (("hm3d", "M21b", "M21bh"), ("examb", "M21", "M21h"), ("examb", "M21b", "M21bh"))  # rows 1-3


class Call(NamedTuple):
    """One closed-loop call of the chain: these arms on this set, outputs below <out_dir>/<subdir>."""

    label: str
    subdir: str
    set: str
    recog: str
    seeds: Optional[Tuple[int, ...]]
    arms: Tuple[str, ...]


class P13Plan:
    """The calls of every Phase 13 stage (module docstring)."""

    def __init__(self, cfg: Dict[str, Any], table: ArmTable) -> None:
        self.cfg, self.table, self.base = cfg, table, table.base

    @classmethod
    def load(cls, config_file: Path, outputs: Path) -> "P13Plan":
        from mapmad_sim.run_layout import REPO, read_config

        cfg = read_config(Path(config_file))
        return cls(cfg, ArmTable.from_config(cfg, Path(outputs) / cfg["out_dir"], Path(outputs), REPO))

    @classmethod
    def from_layout(cls, layout: Any) -> "P13Plan":
        return cls(layout.cfg, ArmTable.from_layout(layout))

    def calls(self, stage: str) -> List[Call]:
        if stage == "dev":
            return [Call("dev", "runs/dev", "dev", "yoloe", (0,), ("M21b", "M21bh", "F")),
                    Call("dev_VF", "runs/dev", "dev", "yoloe+itm", (0,), ("VF",))]
        if stage == "exams":
            return [Call("hm3d_pair", "runs/hm3d", "hm3d", "yoloe", None, ("M21b", "M21bh")),
                    Call("examb_v21", "runs/examb", "examb", "yoloe", None, ("M21", "M21h")),
                    Call("examb_v21b", "runs/examb", "examb", "yoloe", None, ("M21b", "M21bh")),
                    Call("examb_N", "runs/examb", "examb", "yoloe", None, ("N",)),
                    Call("examb_F", "runs/examb", "examb", "yoloe", None, ("F",)),
                    Call("examb_VF", "runs/examb", "examb", "yoloe+itm", None, ("VF",)),
                    Call("hm3d_VF", "runs/hm3d", "hm3d", "yoloe+itm", None, ("VF",))]
        raise ValueError(f"unknown stage {stage}")

    def report_calls(self) -> List[Call]:
        """Every call whose runs the tables report and the fairness check covers (exams; dev separately)."""
        return self.calls("exams")

    # --- runs ---------------------------------------------------------------------------------------------------
    def set_file(self, set_name: str) -> Path:
        return self.base / self.cfg["sets"][set_name]["folder"] / "episodes.json"

    def episode_ids(self, set_name: str) -> List[str]:
        f = self.set_file(set_name)
        return [e["episode_id"] for e in json.loads(f.read_text())["episodes"]] if f.exists() else []

    def arm_seeds(self, call: Call, arm: str) -> List[int]:
        """The seed indices `arm` runs in this call (the call's seeds, else run.seeds; limited by the arm table)."""
        given = list(call.seeds) if call.seeds is not None else list(range(int(self.cfg["run"]["seeds"])))
        return self.table.seeds(arm, given)

    def runs(self, call: Call) -> int:
        n = len(self.episode_ids(call.set))
        return sum(n * len(self.arm_seeds(call, a)) for a in call.arms)

    def total(self, stage: str) -> int:
        return sum(self.runs(c) for c in self.calls(stage))

    def timing_call(self) -> Tuple[Call, List[str]]:
        """The 20-run timing: the first 4 episodes of the first exam call (HM3D M21b / M21bh; 24 runs)."""
        c = self.calls("exams")[0]
        return c, self.episode_ids(c.set)[:4]
