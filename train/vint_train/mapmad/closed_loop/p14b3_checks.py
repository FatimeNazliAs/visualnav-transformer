"""Phase 14b-3 gate rows 5-8 (kickoff §7; p14b3_rules.md §8). numpy only (naz_mapmad). Each command merges its result
into <out_dir>/checks/checks.json (+ checks.md) and exits 3 on a failure (the chain's automatic stop).

    python -m vint_train.mapmad.closed_loop.p14b3_checks det --config <p14b3_v3.yaml>        # row 8a
    python -m vint_train.mapmad.closed_loop.p14b3_checks off --config ...                    # row 8b + 14b-2 reuse
    python -m vint_train.mapmad.closed_loop.p14b3_checks fairness --config ... --stages practice examb   # row 7
    python -m vint_train.mapmad.closed_loop.p14b3_checks contamination --config ...          # row 6 (merge)
    python -m vint_train.mapmad.closed_loop.p14b3_checks rules --config ... --tag <tag>      # row 6: exam rules sha256
    python -m vint_train.mapmad.closed_loop.p14b3_checks g10 --config ...                    # row 5 (merge)
    python -m vint_train.mapmad.closed_loop.p14b3_checks repro --config ... --gpu <g>        # addendum, per GPU

Addendum (M21bh completion, after the gate): `--key` writes fairness / contamination / rules under another key
(addendum_fairness, ...) so the gate's own entries stay as they were; contamination `--dir` reads the addendum's
contamination files. repro: the addrepro call of GPU g (one M21b_w4_d_am exam-B run of this phase that GPU g drove)
byte-identical (log + dets) with runs/examb; key addendum_repro_g<g>.

det: the det call's runs (checks/det_a, checks/det_b; one p13d episode, M3_P, same GPU): logs, dets and the stop
evidence identical. off: per GPU g, checks/off13_g<g> (Phase 13's arm M21b, every new switch off, one exam-B episode
Phase 13 drove on GPU g) byte-identical (log + dets) with Phase 13's run, and checks/off14b2_g<g> (the 14b-2 "both"
cell, one p13d episode 14b-2 drove on GPU g) byte-identical with 14b-2's runs (= the reuse of 14b-2's practice runs is
valid). fairness: every planned (arm, episode, seed) of the stages has a summary (no crash record), a log header with
its set's fingerprint, the arm's checkpoint sha256 and switches; calls the budget fallback cut are listed as "not run"
(not failures; p14b3_rules.md §6). contamination: checks/contamination*.json (make_p14b3_sets.py) merged. rules: the
sha256 of p14b3_rules.md now vs the value frozen before the first exam run (checks/exam_rules_frozen.sha256). g10:
offline/v3/g10_rows_1_2_cpu.json merged.
"""

import argparse
import filecmp
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mapmad_sim.arm_table import ArmTable
from mapmad_sim.episodes import load_episodes
from mapmad_sim.run_layout import RunLayout, run_id, write_text_atomic
from mapmad_sim.run_plan_p14b3 import EXAMB_SUBDIR, P13_EXAMB, REUSE_SUBDIR, P14b3Plan, run_gpus
from mapmad_sim.stop_evidence import load as load_evidence

STOP_EXIT = 3
SUMMARY_KEYS = ("g10_rows_1_2", "contamination", "exam_rules_sha256", "fairness", "determinism", "off13", "off14b2",
                "addendum_repro_g0", "addendum_repro_g1", "addendum_contamination", "addendum_rules_before",
                "addendum_rules_after", "addendum_fairness")


def merge(layout: RunLayout, key: str, value: Dict[str, Any]) -> None:
    f = layout.base / "checks" / "checks.json"
    doc = json.loads(f.read_text()) if f.exists() else {}
    doc[key] = value
    write_text_atomic(f, json.dumps(doc, indent=1) + "\n")
    lines = ["# Phase 14b-3 checks (gate rows 5-8)", ""]
    for k in SUMMARY_KEYS:
        if k in doc:
            lines.append(f"- **{k}:** {'PASS' if doc[k].get('pass') else 'FAIL'} — {doc[k].get('summary', '')}")
    write_text_atomic(layout.base / "checks" / "checks.md", "\n".join(lines) + "\n")


def finish(layout: RunLayout, key: str, value: Dict[str, Any]) -> None:
    merge(layout, key, value)
    print(f"{key}: {'PASS' if value['pass'] else 'FAIL'} — {value['summary']}", flush=True)
    if not value["pass"]:
        raise SystemExit(STOP_EXIT)


def same_files(a: Path, b: Path, rid: str, arm: str, kinds=("logs", "dets")) -> List[str]:
    """The kinds whose file differs (or is missing) between run roots a and b."""
    bad = []
    for kind in kinds:
        fa, fb = a / kind / arm / f"{rid}.jsonl", b / kind / arm / f"{rid}.jsonl"
        if not (fa.exists() and fb.exists() and filecmp.cmp(fa, fb, shallow=False)):
            bad.append(kind)
    return bad


def cmd_det(args) -> None:
    lay = RunLayout.load(args.config)
    a, b = P14b3Plan.from_cfg(lay).calls("det")
    bad, n = [], 0
    for arm in a.arms:
        for e in a.episodes:
            rid = run_id(e, 0)
            ra, rb = lay.base / a.subdir, lay.base / b.subdir
            n += 2
            bad += [f"{arm}/{k}" for k in same_files(ra, rb, rid, arm)]
            ea, eb = (load_evidence(r / "evidence" / arm / f"{rid}.npz") for r in (ra, rb))
            n += 1
            if sorted(ea) != sorted(eb) or any(not np.array_equal(ea[k], eb[k]) for k in ea):
                bad.append(f"{arm}/evidence")
    ok = not bad and n > 0
    finish(lay, "determinism", {"files": n, "differ": bad, "pass": ok,
                                "summary": f"one p13d episode x {', '.join(a.arms)} twice on the same GPU: "
                                           + (f"{n}/{n} files identical (logs, dets, evidence)" if ok else f"DIFFER {bad}")})


def cmd_off(args) -> None:
    lay = RunLayout.load(args.config)
    plan = P14b3Plan.from_cfg(lay)
    res: Dict[str, Dict[str, Any]] = {"off13": {"identical": 0, "differ": []}, "off14b2": {"identical": 0, "differ": []}}
    refs = {"off13": lay.outputs / P13_EXAMB, "off14b2": lay.outputs / REUSE_SUBDIR}
    for c in plan.calls("off"):
        kind = c.label.split("_g")[0]
        g = int(c.label.rsplit("_g", 1)[1])
        new_root = lay.base / c.subdir
        g_new = run_gpus(new_root)
        for arm in c.arms:
            for e in c.episodes:
                rid = run_id(e, 0)
                if g_new.get((arm, rid)) != g:
                    res[kind]["differ"].append(f"{arm}/{rid}: driven on GPU {g_new.get((arm, rid))}, not {g}")
                    continue
                bad = same_files(new_root, refs[kind], rid, arm)
                if bad:
                    res[kind]["differ"].append(f"{arm}/{rid}: {bad}")
                else:
                    res[kind]["identical"] += 1
    for kind, what in (("off13", "Phase 13 exam-B M21b runs (every new switch off)"),
                       ("off14b2", "14b-2 'both'-cell practice runs (reuse check)")):
        r = res[kind]
        r["pass"] = not r["differ"] and r["identical"] >= 2
        r["summary"] = (f"{r['identical']} {what} reproduced byte-identically on the GPU that drove them (log + dets)"
                        + ("" if r["pass"] else f"; FAIL: {r['differ'][:4]} (needs >= 2 identical, none differing)"))
        merge(lay, kind, r)
        print(f"{kind}: {'PASS' if r['pass'] else 'FAIL'} — {r['summary']}", flush=True)
    if not (res["off13"]["pass"] and res["off14b2"]["pass"]):
        raise SystemExit(STOP_EXIT)


def cmd_fairness(args) -> None:
    lay = RunLayout.load(args.config)
    plan = P14b3Plan.from_cfg(lay)
    table = ArmTable.from_layout(lay)
    fps: Dict[str, str] = {}
    problems: List[str] = []
    not_run: List[str] = []
    planned = present = 0
    cut = set(json.loads(args.cut.read_text())) if args.cut and args.cut.exists() else set()
    for stage in args.stages:
        for c in plan.calls(stage):
            if c.set not in fps:
                fps[c.set] = load_episodes(plan.set_file(c.set))[1]
            if c.label in cut:
                not_run.append(c.label)
                continue
            root = lay.base / c.subdir
            for arm, e, k in plan.planned_runs(c):
                a = table.arm(arm)
                sha = table.checkpoint(a.checkpoint).sha256 if a.checkpoint else None
                planned += 1
                rid = run_id(e, k)
                summ, log = root / "summaries" / arm / f"{rid}.json", root / "logs" / arm / f"{rid}.jsonl"
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
                present += 1
    ok = not problems
    summ = (f"{present}/{planned} planned runs ({', '.join(args.stages)}) complete on the frozen fingerprints ("
            + ", ".join(f"{s} {f[:8]}…" for s, f in fps.items()) + "), checkpoint sha256 and switches matched"
            + (f"; not run (budget fallback): {not_run}" if not_run else "")
            + ("" if ok else f"; problems: {problems[:5]} (+{max(len(problems) - 5, 0)})"))
    finish(lay, args.key or "fairness", {"planned": planned, "present": present, "problems": problems, "not_run": not_run,
                             "pass": ok, "summary": summ})


def cmd_contamination(args) -> None:
    lay = RunLayout.load(args.config)
    folder = lay.base / "checks" / (args.dir or "")
    docs = {f.name: json.loads(f.read_text()) for f in sorted(folder.glob("contamination*.json"))}
    # the addendum re-checks the training splits and round-2 drives (contamination.json); the rollout start set is unchanged
    need = {"contamination.json"} if args.dir else {"contamination_rollout_set.json", "contamination.json"}
    ok = need <= set(docs) and all(d.get("contamination_free") for d in docs.values())
    finish(lay, args.key or "contamination", {"files": sorted(docs), "pass": ok,
                                  "summary": ("rollout set, round-2 drives and every training split ∩ (photo test ∪ "
                                              "HM3D val ∪ MP3D) = ∅ (full id, scene id, home prefix)") if ok else
                                  f"FAIL or missing: { {k: d.get('contamination_free') for k, d in docs.items()} }"})


def cmd_rules(args) -> None:
    lay = RunLayout.load(args.config)
    frozen = (lay.base / "checks" / "exam_rules_frozen.sha256").read_text().split()[0]
    now = hashlib.sha256((lay.base / "p14b3_rules.md").read_bytes()).hexdigest()
    repo = hashlib.sha256(Path(args.repo_copy).read_bytes()).hexdigest() if args.repo_copy else now
    ok = frozen == now == repo
    finish(lay, args.key or "exam_rules_sha256", {"frozen": frozen, "now": now, "repo_copy": repo, "tag": args.tag, "pass": ok,
                                      "summary": f"p14b3_rules.md sha256 {now[:12]}… = the value frozen before the "
                                                 f"first exam run ({args.tag})" if ok else
                                      f"CHANGED: frozen {frozen[:12]} now {now[:12]} repo {repo[:12]}"})


def cmd_repro(args) -> None:
    """Addendum: the addrepro call of GPU args.gpu reproduced byte-identically (log + dets) on that GPU."""
    lay = RunLayout.load(args.config)
    plan = P14b3Plan.from_cfg(lay)
    c = next(x for x in plan.calls("addrepro") if x.label == f"addrepro_g{args.gpu}")
    new_root, ref_root = lay.base / c.subdir, lay.outputs / EXAMB_SUBDIR
    g_new, g_ref = run_gpus(new_root), run_gpus(ref_root)
    bad, ok = [], 0
    for arm in c.arms:
        for e in c.episodes:
            rid = run_id(e, 0)
            if g_new.get((arm, rid)) != args.gpu or g_ref.get((arm, rid)) != args.gpu:
                bad.append(f"{arm}/{rid}: driven on GPU {g_new.get((arm, rid))} (reference {g_ref.get((arm, rid))}), "
                           f"not {args.gpu}")
                continue
            diff = same_files(new_root, ref_root, rid, arm)
            if diff:
                bad.append(f"{arm}/{rid}: {diff}")
            else:
                ok += 1
    good = ok >= 1 and not bad
    finish(lay, f"addendum_repro_g{args.gpu}", {
        "gpu": args.gpu, "episodes": list(c.episodes), "identical": ok, "differ": bad, "pass": good,
        "summary": (f"{ok} exam-B M21b + P* run(s) of this phase reproduced byte-identically on GPU {args.gpu} "
                    f"(log + dets)") if good else f"GPU {args.gpu}: FAIL {bad or 'no reference run'}"})


def cmd_g10(args) -> None:
    lay = RunLayout.load(args.config)
    f = lay.base / "offline" / "v3" / "g10_rows_1_2_cpu.json"
    doc = json.loads(f.read_text()) if f.exists() else {}
    rows = {k: doc[k] for k in ("row1", "row2") if isinstance(doc.get(k), dict)}  # (v1 line: report only)
    ok = len(rows) == 2 and all(v["pass"] for v in rows.values())
    finish(lay, "g10_rows_1_2", {"file": str(f), "rows": {k: v.get("pass") for k, v in rows.items()}, "pass": ok,
                                 "summary": ("switch off byte-identical; maps hidden + past token hidden = official "
                                             "NoMaD (allclose 1e-6)") if ok else f"FAIL or missing: {rows or f}"})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["det", "off", "fairness", "contamination", "rules", "g10", "repro"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stages", nargs="+", default=["practice", "examb"])
    p.add_argument("--cut", type=Path, default=None, help="fairness: JSON list of call labels cut by the budget fallback")
    p.add_argument("--tag", default="")
    p.add_argument("--repo-copy", default=None)
    p.add_argument("--key", default=None, help="addendum: write the result under this key (default: the gate's key)")
    p.add_argument("--dir", default=None, help="contamination: subfolder of checks/ with the files (addendum)")
    p.add_argument("--gpu", type=int, default=0, help="repro: the GPU")
    args = p.parse_args(argv)
    {"det": cmd_det, "off": cmd_off, "fairness": cmd_fairness, "contamination": cmd_contamination,
     "rules": cmd_rules, "g10": cmd_g10, "repro": cmd_repro}[args.cmd](args)


if __name__ == "__main__":
    main()
