"""The Phase 14a run plan (kickoff 14.0b / 14.4, p14a_rules.md §2): which closed-loop calls a stage makes -- one place
for the chain (train/scripts/mapmad_p14/p14_plan.py, host), the fairness check and the tables.

    plan = P14Plan.load(config_file, outputs)       # configs/p14a_finish.yaml
    plan.calls("baseline")                          # [Call(label, subdir, set, recog, seeds, arms), ...] in run order
    plan.runs(call)                                 # episodes x arms (seed index 0)

Stages: `baseline` (14.0b: M21b, M21bh, N, F, VF on the practice set = p13d (Phase 13 noise seeds: the M21b, M21bh,
F, VF runs repeat Phase 13's dev runs byte for byte) + p14d), `drive` (14.4: the network arms with A1, W4, A1 + W4),
`det` (row 7: one p13d episode, run twice on the same GPU into checks/det_a and checks/det_b). A call's recog is
"yoloe" (one YOLOE sidecar per pair) or "yoloe+itm" (VF: plus one BLIP-2 ITM server per GPU). DRY runs: the
environment variable P14_EPISODES (space-separated ids) keeps only those episodes (every call then names them).
Pure Python + PyYAML.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from mapmad_sim.arm_table import ArmTable

STAGES = ("baseline", "drive", "det", "r4", "r4det")
NET_ARMS = ("M21b", "M21bh", "N")
BASE_ARMS = ("M21b", "M21bh", "N", "F", "VF")
SWITCHES = (("a1", "_a1"), ("w4", "_w4"), ("a1w4", "_a1w4"))  # (subdir, arm suffix)
PARTS = ("p13d", "p14d")
DET_ARMS = ("M21b", "M21b_a1w4", "N_a1w4")
# Phase 14b-2 R4 (W4 on everywhere): (subdir, arm suffix, recogniser) -- "picked" = the R2 detector (the chain maps it
# to its sidecar key); detector base + A-mark off = the 14a W4 runs (reused, p14a_finish/runs/w4)
R4_CELLS = (("am", "_w4_am", "yoloe"), ("det", "_w4_d", "picked"), ("detam", "_w4_d_am", "picked"))
R4_DET_ARMS = ("M21b_w4_d_am", "N_w4_d_am")  # gate row 8: same episode twice, same GPU (one sidecar model per call)
R4_OFF_ARMS = ("M21b_w4", "N_w4")  # gate row 7: switch off with 14b-2 code = the 14a W4 runs byte for byte


class Call(NamedTuple):
    label: str
    subdir: str
    set: str
    recog: str
    seeds: Optional[Tuple[int, ...]]
    arms: Tuple[str, ...]
    episodes: Tuple[str, ...] = ()


class P14Plan:
    """The calls of every Phase 14a stage (module docstring)."""

    def __init__(self, cfg: Dict[str, Any], table: ArmTable) -> None:
        self.cfg, self.table, self.base = cfg, table, table.base
        self.only = set(os.environ.get("P14_EPISODES", "").split())  # DRY runs only

    @classmethod
    def load(cls, config_file: Path, outputs: Path) -> "P14Plan":
        from mapmad_sim.run_layout import REPO, read_config

        cfg = read_config(Path(config_file))
        return cls(cfg, ArmTable.from_config(cfg, Path(outputs) / cfg["out_dir"], Path(outputs), REPO))

    @classmethod
    def from_cfg(cls, layout: Any) -> "P14Plan":
        return cls(layout.cfg, ArmTable.from_layout(layout))

    def calls(self, stage: str) -> List[Call]:
        eps = (lambda s: tuple(self.episode_ids(s)) if self.only else ())  # noqa: E731
        if stage == "baseline":
            out = []
            for s in PARTS:
                out.append(Call(f"base_{s}", "runs/baseline", s, "yoloe", (0,), ("M21b", "M21bh", "N", "F"), eps(s)))
                out.append(Call(f"base_{s}_VF", "runs/baseline", s, "yoloe+itm", (0,), ("VF",), eps(s)))
            return out
        if stage == "drive":
            return [Call(f"{sub}_{s}", f"runs/{sub}", s, "yoloe", (0,), tuple(a + suf for a in NET_ARMS), eps(s))
                    for sub, suf in SWITCHES for s in PARTS]
        if stage == "r4":  # P14B2_BASE_ONLY=1: R2 fallback (no fine-tuned pick) -> the A-mark cell only
            cells = [c for c in R4_CELLS if not (os.environ.get("P14B2_BASE_ONLY") and c[2] == "picked")]
            return [Call(f"{sub}_{s}", f"runs/{sub}", s, recog, (0,), tuple(a + suf for a in NET_ARMS), eps(s))
                    for sub, suf, recog in cells for s in PARTS]
        if stage == "r4det":
            e = tuple(self.episode_ids("p13d")[:1])
            out = [Call(f"det_{x}", f"checks/det_{x}", "p13d", "picked", (0,), R4_DET_ARMS, e) for x in ("a", "b")]
            return out + [Call("off", "checks/off", "p13d", "yoloe", (0,), R4_OFF_ARMS, tuple(self.episode_ids("p13d")[:2]))]
        if stage == "det":
            e = self.episode_ids("p13d")[:1]
            return [Call(f"det_{x}", f"checks/det_{x}", "p13d", "yoloe", (0,), DET_ARMS, tuple(e)) for x in ("a", "b")]
        raise ValueError(f"unknown stage {stage}")

    def set_file(self, set_name: str) -> Path:
        return self.base / self.cfg["sets"][set_name]["folder"] / "episodes.json"

    def episode_ids(self, set_name: str) -> List[str]:
        f = self.set_file(set_name)
        ids = [e["episode_id"] for e in json.loads(f.read_text())["episodes"]] if f.exists() else []
        return [i for i in ids if i in self.only] if self.only else ids

    def runs(self, call: Call) -> int:
        n = len(call.episodes) if call.episodes else len(self.episode_ids(call.set))  # (episodes: DRY / det)
        return n * len(call.arms)

    def total(self, stage: str) -> int:
        return sum(self.runs(c) for c in self.calls(stage))
