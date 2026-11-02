"""Phase 14a gate rows 6-7 (kickoff §7; p14a_rules.md §10). numpy only (naz_mapmad).

    python -m vint_train.mapmad.closed_loop.p14_checks same-p13 --config <p14a_finish.yaml>   # exit 3 on a mismatch
    python -m vint_train.mapmad.closed_loop.p14_checks det --config <p14a_finish.yaml>        # exit 3 on a mismatch
    python -m vint_train.mapmad.closed_loop.p14_checks fairness --config <p14a_finish.yaml> --stages baseline drive

same-p13 (row 6 "switch off = Phase 13", row 7 "Phase 13 dev episodes reproduced"): every p13d run of M21b, M21bh, F
and VF in runs/baseline vs Phase 13's dev run of the same arm and episode (p13_claim/runs/dev). The GPU of a run = its
progress line's shard // 10 (the last line wins). Runs on the SAME GPU as in Phase 13 must have byte-identical step
logs and dets files (>= 2 such runs required); runs on the other GPU are reported only (YOLOE's float noise differs
between the two cards, Phase 9 / Phase 13 lesson). det (row 7): the det call's runs (checks/det_a, checks/det_b; one episode, same GPU): logs and dets
byte-identical and the stop-evidence arrays equal. fairness (row 6): every planned arm x episode of the stages has a
summary (no crash record), a log whose header carries its set's fingerprint, the arm's checkpoint sha256 (network
arms) and the arm's switches, and an evidence file. Results are merged into <out_dir>/checks/checks.json + checks.md.
"""

import argparse
import filecmp
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mapmad_sim.arm_table import ArmTable
from mapmad_sim.episodes import load_episodes
from mapmad_sim.run_layout import RunLayout, run_id, write_text_atomic
from mapmad_sim.run_plan_p14 import P14Plan
from mapmad_sim.stop_evidence import load as load_evidence

P13_DEV = "p13_claim/runs/dev"
SAME_ARMS = ("M21b", "M21bh", "F", "VF")
STOP_EXIT = 3
LINE = re.compile(r"^\S+ \d+/\d+ (\S+) (\S+__s\d+): ")


def merge(layout: RunLayout, key: str, value: Dict[str, Any], summary_key: str, summary: str) -> None:
    f = layout.base / "checks" / "checks.json"
    doc = json.loads(f.read_text()) if f.exists() else {}
    doc[key] = value
    doc[summary_key] = summary
    write_text_atomic(f, json.dumps(doc, indent=1) + "\n")
    lines = ["# Phase 14a checks (gate rows 6-7)", ""]
    for k in ("fairness_summary", "determinism_summary", "same_p13_summary"):
        if k in doc:
            lines.append(f"- **{k.replace('_summary', '')}:** {doc[k]}")
    write_text_atomic(layout.base / "checks" / "checks.md", "\n".join(lines) + "\n")


def run_gpus(root: Path) -> Dict[Tuple[str, str], int]:
    """(arm, run id) -> GPU of its run (progress lines "<time> i/n <arm> <run>: ..."; shard // 10)."""
    out = {}
    for f in sorted(root.glob("progress.*.shard*.log")):
        g = int(f.name.rsplit("shard", 1)[1].split(".")[0]) // 10
        for ln in f.read_text().splitlines():
            m = LINE.match(ln)
            if m:
                out[(m.group(1), m.group(2))] = g
    return out


def cmd_same_p13(args) -> None:
    layout = RunLayout.load(args.config)
    plan = P14Plan.from_cfg(layout)
    old_root = layout.outputs / P13_DEV
    new_root = layout.base / "runs" / "baseline"
    g_new, g_old = run_gpus(new_root), run_gpus(old_root)
    same, other = {"identical": 0, "differ": []}, {"identical": 0, "differ": 0}
    for arm in SAME_ARMS:
        for e in plan.episode_ids("p13d"):
            rid = run_id(e, 0)
            ident = all((new_root / k / arm / f"{rid}.jsonl").exists() and (old_root / k / arm / f"{rid}.jsonl").exists()
                        and filecmp.cmp(new_root / k / arm / f"{rid}.jsonl", old_root / k / arm / f"{rid}.jsonl",
                                        shallow=False) for k in ("logs", "dets"))
            if g_new.get((arm, rid)) is not None and g_new.get((arm, rid)) == g_old.get((arm, rid)):
                if ident:
                    same["identical"] += 1
                else:
                    same["differ"].append(f"{arm}/{rid}")
            else:
                other["identical" if ident else "differ"] += 1
    n_same = same["identical"] + len(same["differ"])
    ok = not same["differ"] and same["identical"] >= 2
    summ = (f"same GPU as Phase 13: {same['identical']}/{n_same} p13d runs (M21b, M21bh, F, VF) byte-identical with "
            f"Phase 13's dev runs (log + dets); other GPU (report only): {other['identical']} identical, "
            f"{other['differ']} differ" + ("" if ok else f"; FAIL: differ {same['differ'][:5]} or < 2 same-GPU runs"))
    merge(layout, "same_p13", {"same_gpu": same, "other_gpu": other, "pass": ok}, "same_p13_summary", summ)
    print(summ, flush=True)
    if not ok:
        raise SystemExit(STOP_EXIT)


def cmd_det(args) -> None:
    layout = RunLayout.load(args.config)
    plan = P14Plan.from_cfg(layout)
    a, b = plan.calls("det")
    bad, n = [], 0
    for arm in a.arms:
        for e in a.episodes:
            rid = run_id(e, 0)
            for kind in ("logs", "dets"):
                fa, fb = (layout.base / c.subdir / kind / arm / f"{rid}.jsonl" for c in (a, b))
                n += 1
                if not (fa.exists() and fb.exists() and filecmp.cmp(fa, fb, shallow=False)):
                    bad.append(f"{arm}/{kind}")
            ea, eb = (load_evidence(layout.base / c.subdir / "evidence" / arm / f"{rid}.npz") for c in (a, b))
            n += 1
            if sorted(ea) != sorted(eb) or any(not np.array_equal(ea[k], eb[k]) for k in ea):
                bad.append(f"{arm}/evidence")
    ok = not bad
    summ = f"one p13d episode x {len(a.arms)} arms ({', '.join(a.arms)}) twice on the same GPU: " + \
        (f"{n}/{n} files identical (logs, dets, evidence)" if ok else f"DIFFER: {bad}")
    merge(layout, "determinism", {"files": n, "differ": bad, "pass": ok}, "determinism_summary", summ)
    print(summ, flush=True)
    if not ok:
        raise SystemExit(STOP_EXIT)


def cmd_fairness(args) -> None:
    layout = RunLayout.load(args.config)
    plan = P14Plan.from_cfg(layout)
    table = ArmTable.from_layout(layout)
    fps = {s: load_episodes(plan.set_file(s))[1] for s in ("p13d", "p14d")}
    problems: List[str] = []
    planned = 0
    for stage in args.stages:
        for c in plan.calls(stage):
            for arm in c.arms:
                a = table.arm(arm)
                sha = table.checkpoint(a.checkpoint).sha256 if a.checkpoint else None
                for e in plan.episode_ids(c.set):
                    planned += 1
                    rid = run_id(e, 0)
                    root = layout.base / c.subdir
                    summ = root / "summaries" / arm / f"{rid}.json"
                    log = root / "logs" / arm / f"{rid}.jsonl"
                    if not summ.exists():
                        problems.append(f"{arm} {rid}: " + ("crashed" if (root / "summaries" / arm / f"{rid}.crash.json").exists()
                                                            else "missing"))
                        continue
                    head = json.loads(log.open().readline())
                    if head["episodes_fingerprint"] != fps[c.set]:
                        problems.append(f"{arm} {rid}: fingerprint {head['episodes_fingerprint'][:12]}")
                    if sha is not None and head.get("weights_sha256") != sha:
                        problems.append(f"{arm} {rid}: weights sha256")
                    if head.get("switches", {}) != a.switches:
                        problems.append(f"{arm} {rid}: switches {head.get('switches')}")
                    if not (root / "evidence" / arm / f"{rid}.npz").exists():
                        problems.append(f"{arm} {rid}: no evidence")
    ok = not problems
    summ = (f"{planned - len(problems)}/{planned} planned runs ({', '.join(args.stages)}) complete on the frozen "
            f"fingerprints (p13d {fps['p13d'][:8]}…, p14d {fps['p14d'][:8]}…), checkpoint sha256 and switches matched, "
            f"evidence present" + ("" if ok else f"; problems: {problems[:5]} (+{max(len(problems) - 5, 0)})"))
    merge(layout, "fairness", {"planned": planned, "problems": problems, "pass": ok}, "fairness_summary", summ)
    print(summ, flush=True)
    if not ok:
        raise SystemExit(STOP_EXIT)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["same-p13", "det", "fairness"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stages", nargs="+", default=["baseline", "drive"])
    args = p.parse_args(argv)
    {"same-p13": cmd_same_p13, "det": cmd_det, "fairness": cmd_fairness}[args.cmd](args)


if __name__ == "__main__":
    main()
