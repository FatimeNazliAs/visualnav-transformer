"""The Phase 14b-3 run plan (kickoff §3 items 8-9, §4; p14b3_rules.md §5-§6): which closed-loop calls a stage makes --
one place for the chain (train/scripts/mapmad_p14/p14_plan.py --phase 14b3), the checks and the tables.

    plan = P14b3Plan.load(config_file, outputs)       # configs/p14b3_v3.yaml
    plan.calls("examb")                               # [Call(label, subdir, set, recog, seeds, arms, episodes), ...]
    plan.runs(call)                                   # episodes x arms x seed indices (a deterministic arm: 1 seed)

Stages:
- `practice` (V3): MapMaD v3 + P* maps shown / hidden, and maps shown without A-mark, on the MP3D practice set (p13d,
  p14d), seed 0. v2.1b + P* = the 14b-2 "both" cell (p14b2_recog/runs/detam), reused.
- `det` (gate row 8a): one p13d episode, M3_P, twice on the same GPU (checks/det_a, checks/det_b).
- `off` (gate row 8b + reuse check): per GPU g, (i) one Phase 13 exam-B M21b run (Phase 13's arm, every new switch off)
  that Phase 13 drove on GPU g -> checks/off13_g<g>, compared byte for byte with Phase 13's log; (ii) the 14b-2 "both"
  cell (M21b_w4_d_am, N_w4_d_am) on one p13d episode that 14b-2 drove on GPU g -> checks/off14b2_g<g>, compared with
  14b-2's logs (its practice runs are reused).
- `examb` (V4): exam B, seeds 0-2 (F: 0), in the pre-set priority order (rows 1-3 arms first, then N seed 0, F,
  N seeds 1-2, M21bh), one call per arm group and seed so a budget cut leaves whole seeds.
DRY runs: P14_EPISODES (space-separated ids) keeps only those episodes. Pure Python + PyYAML.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from mapmad_sim.run_layout import run_id
from mapmad_sim.run_plan_p14 import PARTS, Call, P14Plan

STAGES = ("practice", "det", "off", "examb")
PRAC_ARMS = ("M3_P", "M3h_P", "M3_Pnoam")
V21B_P = ("M21b_w4_d_am", "M21bh_w4_d_am")  # v2.1b + P* (14b-2 "both" cell; practice: reused)
REUSE_SUBDIR = "p14b2_recog/runs/detam"  # below <outputs>
P13_EXAMB = "p13_claim/runs/examb"  # below <outputs>
OFF13_ARM = "M21b"  # Phase 13's arm (base YOLOE, no switch)
OFF14B2_ARMS = ("M21b_w4_d_am", "N_w4_d_am")
# exam B: (label, arms, seeds) in run order (p14b3_rules.md §6: rows 1-3 arms first; then N seed 0, F, N 1-2, M21bh)
EXAM_ORDER: Tuple[Tuple[str, Tuple[str, ...], Tuple[int, ...]], ...] = (
    ("ex_v3_s0", ("M3_P", "M3h_P"), (0,)), ("ex_m21b_s0", ("M21b_w4_d_am",), (0,)),
    ("ex_v3_s1", ("M3_P", "M3h_P"), (1,)), ("ex_m21b_s1", ("M21b_w4_d_am",), (1,)),
    ("ex_v3_s2", ("M3_P", "M3h_P"), (2,)), ("ex_m21b_s2", ("M21b_w4_d_am",), (2,)),
    ("ex_n_s0", ("N_w4_d_am",), (0,)), ("ex_f", ("F_d",), (0,)),
    ("ex_n_s1", ("N_w4_d_am",), (1,)), ("ex_n_s2", ("N_w4_d_am",), (2,)),
    ("ex_m21bh_s0", ("M21bh_w4_d_am",), (0,)), ("ex_m21bh_s1", ("M21bh_w4_d_am",), (1,)),
    ("ex_m21bh_s2", ("M21bh_w4_d_am",), (2,)),
)
EXAM_ARMS = ("M3_P", "M3h_P", "M21b_w4_d_am", "M21bh_w4_d_am", "N_w4_d_am", "F_d")
LINE = re.compile(r"^\S+ \d+/\d+ (\S+) (\S+__s\d+): ")


def run_gpus(root: Path) -> Dict[Tuple[str, str], int]:
    """(arm, run id) -> GPU of its run (progress lines "<time> i/n <arm> <run>: ..."; shard // 10; last line wins)."""
    out = {}
    for f in sorted(Path(root).glob("progress.*.shard*.log")):
        g = int(f.name.rsplit("shard", 1)[1].split(".")[0]) // 10
        for ln in f.read_text().splitlines():
            m = LINE.match(ln)
            if m:
                out[(m.group(1), m.group(2))] = g
    return out


class P14b3Plan(P14Plan):
    """The calls of every Phase 14b-3 stage (module docstring)."""

    def calls(self, stage: str) -> List[Call]:
        eps = (lambda s: tuple(self.episode_ids(s)) if self.only else ())  # noqa: E731
        if stage == "practice":
            return [Call(f"prac_{s}", "runs/practice", s, "picked", (0,), PRAC_ARMS, eps(s)) for s in PARTS]
        if stage == "det":
            e = tuple(self.episode_ids("p13d")[:1])
            return [Call(f"det_{x}", f"checks/det_{x}", "p13d", "picked", (0,), ("M3_P",), e) for x in ("a", "b")]
        if stage == "off":
            out = []
            for g in (0, 1):
                e13 = self.episode_on_gpu(P13_EXAMB, "examb", OFF13_ARM, g)
                e2 = self.episode_on_gpu(REUSE_SUBDIR, "p13d", OFF14B2_ARMS[0], g)
                out.append(Call(f"off13_g{g}", f"checks/off13_g{g}", "examb", "yoloe", (0,), (OFF13_ARM,),
                                (e13,) if e13 else ()))
                out.append(Call(f"off14b2_g{g}", f"checks/off14b2_g{g}", "p13d", "picked", (0,), OFF14B2_ARMS,
                                (e2,) if e2 else ()))
            return out
        if stage == "examb":
            return [Call(label, "runs/examb", "examb", "picked", seeds, arms, eps("examb"))
                    for label, arms, seeds in EXAM_ORDER]
        raise ValueError(f"unknown stage {stage}")

    def episode_on_gpu(self, subdir: str, set_name: str, arm: str, gpu: int) -> str:
        """The first episode of the set (file order, P14_EPISODES ignored) whose seed-0 run of `arm` in
        <outputs>/<subdir> ran on GPU `gpu` (its progress lines); '' if none."""
        gpus = run_gpus(self.table.outputs / subdir)
        only, self.only = self.only, set()
        try:
            ids = self.episode_ids(set_name)
        finally:
            self.only = only
        return next((e for e in ids if gpus.get((arm, run_id(e, 0))) == gpu), "")

    def runs(self, call: Call) -> int:
        n = len(call.episodes) if call.episodes else len(self.episode_ids(call.set))
        seeds = list(call.seeds) if call.seeds is not None else list(range(int(self.cfg["run"]["seeds"])))
        return n * sum(len(self.table.seeds(a, seeds)) for a in call.arms)

    def planned_runs(self, call: Call) -> List[Tuple[str, str, int]]:
        """(arm, episode id, seed index) of every run of a call (the fairness check)."""
        ids = list(call.episodes) if call.episodes else self.episode_ids(call.set)
        seeds = list(call.seeds) if call.seeds is not None else list(range(int(self.cfg["run"]["seeds"])))
        return [(a, e, k) for a in call.arms for k in self.table.seeds(a, seeds) for e in ids]


def plan_for(config_file: Path) -> type:
    """P14b3Plan for a Phase 14b-3 config (out_dir p14b3_*), else P14Plan (the host CLI picks by config)."""
    return P14b3Plan if "p14b3" in Path(config_file).name else P14Plan


__all__ = ["P14b3Plan", "STAGES", "EXAM_ORDER", "EXAM_ARMS", "PRAC_ARMS", "V21B_P", "REUSE_SUBDIR", "P13_EXAMB",
           "run_gpus", "plan_for"]
