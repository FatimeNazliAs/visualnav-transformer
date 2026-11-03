"""Phase 14a stop-rule bench (kickoff 14.1, p14a_rules.md §3-§4): every stop rule scored on the same practice runs
from their stop evidence (mapmad_sim/stop_rules.py), the automatic pick, and the tables. numpy only (naz_mapmad).

    python -m vint_train.mapmad.closed_loop.p14_stopbench score --config <p14a_finish.yaml> --subdir runs/baseline \
        --arms M21b M21bh N F VF --workers 16                          # -> stopbench/runs/<arm>/<run>.json
    python -m vint_train.mapmad.closed_loop.p14_stopbench pick --config <p14a_finish.yaml> --subdir runs/baseline \
        --arms M21b M21bh N F VF                                       # -> stopbench/STOP_PICK.json + stopbench.md

Per run and rule variant (today tau 0.45, Rule 1 grid, Rule 2 grid): stop step (the first step t >= 1 whose frame
qualifies; none = no stop, a failure), success (the stop pose inside the 0.2 m zone, walking distance to the nearest
usable view point), SPL (path up to the stop), wrong object (stop cell > 1.0 m from every usable instance's box),
near miss (failed stop within 0.2 + 1.0 m walking). Oracle success from the same log (p12_score.score_log). Parity:
the rebuilt today-value equals the logged rule_value on every step (else the run is flagged; the kickoff's fallback
is then already in place: these logs ARE re-runs with full evidence).
Pick (§4b): per rule, the grid point with the highest mean rule-stop success over the 5 arms (arm mean = mean over its
episodes) on half A; ties -> the higher threshold (Rule 1: h, then s; Rule 2: v), then fewer looks (n / g).
Reported on halves A, B and all. Main stop = Rule 2's pick.

    python -m vint_train.mapmad.closed_loop.p14_stopbench score ... --marks truth --out stopbench_truth
    python -m vint_train.mapmad.closed_loop.p14_stopbench truth-report --config <p14a_finish.yaml> --subdir runs/baseline

Truth re-score (report only): the same logs with simulator-truth marks (stop_rules.truth_marks, MT-style) in place of
the recogniser; today / the Rule 1 pick / the Rule 2 pick (STOP_PICK.json, not re-picked) per arm. Read: oracle -
truth success = what the rule loses with perfect marks (rule errors); truth - recogniser success = what the
recogniser's marks cost (detector errors) -> stopbench_truth/truth.md + truth.json.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_bridge import steplog
from mapmad_sim import stop_evidence
from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic
from mapmad_sim.stop_rules import StopSpec, first_stops, parse_r1, parse_r2, r1_grid, r2_grid, truth_marks
from vint_train.mapmad.closed_loop.p12_score import TAU_GRID, box_distance_2d, score_log, spl

ARMS5 = ("M21b", "M21bh", "N", "F", "VF")
HALVES = ("A", "B", "all")


# --- one run ------------------------------------------------------------------------------------------------------
def practice(layout: RunLayout) -> Dict[str, Any]:
    return json.loads((layout.base / layout.cfg["sets"]["practice"]["file"]).read_text())


def practice_episodes(layout: RunLayout) -> Dict[str, Dict[str, Any]]:
    """episode id -> episode of the practice parts (sets.practice.parts)."""
    from mapmad_sim.episodes import load_episodes

    out = {}
    for name in layout.cfg["sets"]["practice"]["parts"]:
        eps, _ = load_episodes(layout.base / layout.cfg["sets"][name]["folder"] / "episodes.json")
        out.update({e["episode_id"]: e for e in eps})
    return out


def on_wrong_object(cell: Sequence[float], insts: Sequence[Dict[str, Any]], wrong_m: float = 1.0) -> bool:
    """The cell lies > wrong_m from every usable instance's box (the bench's wrong-object test)."""
    return bool(insts) and min(box_distance_2d(cell, i) for i in insts) > wrong_m


def judge(stop: Optional[Tuple[int, Tuple[float, float]]], vp: List[Optional[float]], path: np.ndarray,
          shortest: Optional[float], insts: Sequence[Dict[str, Any]], success_m: float = 0.2, near_m: float = 1.0,
          wrong_m: float = 1.0) -> Dict[str, Any]:
    """One rule's stop judged on the run (module docstring)."""
    if stop is None:
        return {"step": None, "success": False, "spl": 0.0, "wrong_object": False, "near_miss": False}
    t, cell = stop
    ok = vp[t] is not None and vp[t] <= success_m
    return {"step": t, "cell": [round(cell[0], 3), round(cell[1], 3)], "success": bool(ok),
            "spl": spl(ok, shortest, float(path[t])),
            "wrong_object": on_wrong_object(cell, insts, wrong_m),
            "near_miss": (not ok) and vp[t] is not None and vp[t] <= success_m + near_m}


def score_run(log_file: Path, ev_file: Path, episode: Dict[str, Any], spec: StopSpec = StopSpec(),
              marks: str = "recogniser") -> Dict[str, Any]:
    """All variants of one run (module docstring); marks = recogniser (the logged run) | truth (no parity check)."""
    log = steplog.read(log_file)
    head, steps = log[0], [x for x in log if x["kind"] == "step"]
    insts = episode["target"]["instances"]
    base = score_log([dict(head, instances=insts)] + log[1:], taus=TAU_GRID)
    vp = [head["start_vp_geodesic_m"]] + [s["vp_geodesic_m"] for s in steps]
    path = np.concatenate([[0.0], np.cumsum([s["travelled_m"] for s in steps])])
    ev = stop_evidence.load(ev_file)
    if marks == "truth":
        stops = first_stops(truth_marks(ev), spec)
    else:
        stops = first_stops(ev, spec, [s.get("rule_value") for s in steps])
    par = stops.pop("parity")
    rules = {k: judge(v, vp, path, head["start_vp_geodesic_m"], insts) for k, v in stops.items()}
    return {"marks": marks, "episode_id": head["episode_id"], "arm": head["arm"], "home": head["home"], "category": head["category"],
            "distance_bin": head["distance_bin"], "oracle_success": base["oracle_success"], "oracle_spl": base["oracle_spl"],
            "logged_today": base["rule"][f"{spec.tau:.2f}"]["step"], "parity": par, "rules": rules,
            "today_by_tau": {k: bool(v["success"]) for k, v in base["rule"].items()}}


def _score_one(job: Tuple[str, str, str, Dict[str, Any], str]) -> Tuple[str, Optional[str]]:
    log_file, ev_file, out_file, episode, marks = job
    try:
        r = score_run(Path(log_file), Path(ev_file), episode, marks=marks)
        write_text_atomic(Path(out_file), json.dumps(r, sort_keys=True) + "\n")
        return out_file, None
    except Exception as exc:  # reported per run; the caller stops on any
        return out_file, f"{type(exc).__name__}: {exc}"


def cmd_score(args) -> None:
    layout = RunLayout.load(args.config, subdir=args.subdir)
    eps = practice_episodes(layout)
    jobs = []
    for arm in args.arms:
        for f in sorted((layout.root / "logs" / arm).glob("*.jsonl")):
            rid = f.name[:-len(".jsonl")]
            out = layout.base / args.out / "runs" / arm / f"{rid}.json"
            if out.exists() and not args.force:
                continue
            ev = layout.root / "evidence" / arm / f"{rid}.npz"
            if not ev.exists():
                raise SystemExit(f"missing evidence for {arm} {rid}")
            jobs.append((str(f), str(ev), str(out), eps[parse_run_id(f.name)[0]], args.marks))
    print(f"scoring {len(jobs)} runs on {args.workers} workers", flush=True)
    errors = []
    with Pool(args.workers) as pool:
        for k, (out, err) in enumerate(pool.imap_unordered(_score_one, jobs, chunksize=2)):
            if err:
                errors.append((out, err))
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} runs failed: {errors[:3]}")


# --- aggregation ----------------------------------------------------------------------------------------------------
def read_bench(layout: RunLayout, out: str, arms: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    return {a: [json.loads(f.read_text()) for f in sorted((layout.base / out / "runs" / a).glob("*.json"))] for a in arms}


def in_half(rows: List[Dict[str, Any]], half: str, half_of: Dict[str, str]) -> List[Dict[str, Any]]:
    return rows if half == "all" else [r for r in rows if half_of[r["episode_id"]] == half]


def metrics(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Rule-stop success, SPL, capture, STOP precision, wrong-object and near-miss shares of one rule on runs."""
    n = len(rows)
    if n == 0:
        return {"runs": 0}
    rr = [r["rules"][key] for r in rows]
    succ = float(np.mean([x["success"] for x in rr]))
    orac = float(np.mean([r["oracle_success"] for r in rows]))
    stops = [x for x in rr if x["step"] is not None]
    share = (lambda f: float(np.mean([bool(x[f]) for x in stops])) if stops else None)
    return {"runs": n, "success": succ, "spl": float(np.mean([x["spl"] for x in rr])), "oracle": orac,
            "capture": succ / orac if orac > 0 else None, "stops": len(stops), "stop_share": len(stops) / n,
            "precision": share("success"), "wrong_object": share("wrong_object"), "near_miss": share("near_miss")}


def pick_rule(bench: Dict[str, List[Dict[str, Any]]], keys: Sequence[str], half_of: Dict[str, str], arms: Sequence[str],
              order) -> Tuple[str, Dict[str, float]]:
    """§4b: the key with the highest mean (over arms) rule-stop success on half A; `order(key)` breaks ties (larger
    = preferred)."""
    score = {}
    for k in keys:
        score[k] = float(np.mean([metrics(in_half(bench[a], "A", half_of), k)["success"] for a in arms]))
    best = max(keys, key=lambda k: (round(score[k], 12), order(k)))
    return best, score


def r1_order(key: str) -> Tuple[float, float, int]:
    s, n, h = parse_r1(key)
    return (h, s, -n)


def r2_order(key: str) -> Tuple[float, int]:
    v, g = parse_r2(key)
    return (v, -g)


def table(bench, keys_labels, arms, half_of) -> List[str]:
    lines = ["| stop | arm | half | runs | rule-stop success | SPL | oracle | capture | stops | STOP precision | "
             "wrong object | near miss |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for key, label in keys_labels:
        for a in arms:
            for h in HALVES:
                m = metrics(in_half(bench[a], h, half_of), key)
                lines.append(f"| {label} | {a} | {h} | {m['runs']} | {f(m.get('success'))} | {f(m.get('spl'))} | "
                             f"{f(m.get('oracle'))} | {f(m.get('capture'))} | {m.get('stops', 0)} | "
                             f"{f(m.get('precision'))} | {f(m.get('wrong_object'))} | {f(m.get('near_miss'))} |")
    return lines


def per_arm_tau(bench, arms, half_of) -> Dict[str, Dict[str, Any]]:
    """Report only (kickoff §3 item 5): each arm's own best tau of today's rule (0.30-0.95) on half A (ties -> the
    higher tau), with its success on halves A, B and all."""
    out = {}
    for a in arms:
        taus = sorted(bench[a][0]["today_by_tau"]) if bench[a] else []
        succ = {h: {t: float(np.mean([r["today_by_tau"][t] for r in in_half(bench[a], h, half_of)]))
                    for t in taus} for h in HALVES}
        best = max(taus, key=lambda t: (round(succ["A"][t], 12), float(t)))
        out[a] = {"tau": float(best), **{h: succ[h][best] for h in HALVES}}
    return out


def cmd_pick(args) -> None:
    layout = RunLayout.load(args.config, subdir=args.subdir)
    pr = practice(layout)
    half_of = pr["episode_half"]
    bench = read_bench(layout, args.out, args.arms)
    only = set(os.environ.get("P14_EPISODES", "").split())  # DRY runs only
    expected = len(only) if only else len(pr["episode_half"])
    missing = {a: expected - len(bench[a]) for a in args.arms}
    if any(missing.values()):
        raise SystemExit(f"stop bench incomplete: missing runs {missing}")
    bad = {a: sum(r["parity"]["same"] != r["parity"]["checked"] for r in bench[a]) for a in args.arms}
    r1k = [k for k, *_ in r1_grid()]
    r2k = [k for k, *_ in r2_grid()]
    p1, s1 = pick_rule(bench, r1k, half_of, args.arms, r1_order)
    p2, s2 = pick_rule(bench, r2k, half_of, args.arms, r2_order)
    per_arm = {}
    for a in args.arms:  # report only: each arm's own best Rule 2 grid point on half A
        sc = {k: metrics(in_half(bench[a], "A", half_of), k)["success"] for k in r2k}
        per_arm[a] = max(r2k, key=lambda k: (round(sc[k], 12), r2_order(k)))
    tau_arm = per_arm_tau(bench, args.arms, half_of)
    res = {"per_arm_tau_report_only": tau_arm, "rule1_pick": p1, "rule2_pick": p2, "main_stop": p2, "half_A_mean_success": {"rule1": s1, "rule2": s2},
           "per_arm_rule2_best_report_only": per_arm, "parity_mismatch_runs": bad, "arms": list(args.arms),
           "practice_fingerprint": pr["fingerprint"],
           "results": {k: {a: {h: metrics(in_half(bench[a], h, half_of), k) for h in HALVES} for a in args.arms}
                       for k in ("today", p1, p2)}}
    out = layout.base / args.out
    write_text_atomic(out / "STOP_PICK.json", json.dumps(res, indent=1) + "\n")
    lines = ["# 14.1 stop-rule bench (practice set; §4b pick on half A)", "",
             f"Practice set `{pr['fingerprint'][:12]}…`: {len(half_of)} episodes, halves A / B = "
             f"{pr['half_episodes']}. Arms {', '.join(args.arms)}; seed 0. Parity (rebuilt today-value = logged "
             f"rule_value on every step) mismatching runs: {bad}.", "",
             f"- **Rule 1 pick:** `{p1}` (half-A mean success {s1[p1]:.3f})",
             f"- **Rule 2 pick (main stop):** `{p2}` (half-A mean success {s2[p2]:.3f})",
             f"- Per-arm best Rule 2 point on half A (report only): {per_arm}",
             "- Per-arm best tau of today's rule on half A (report only): " +
             "; ".join(f"{a} tau {v['tau']:.2f} (A {v['A']:.3f}, B {v['B']:.3f}, all {v['all']:.3f})"
                       for a, v in tau_arm.items()), ""]
    lines += table(bench, [("today", "today tau 0.45"), (p1, f"Rule 1 {p1}"), (p2, f"Rule 2 {p2}")], args.arms, half_of)
    lines += ["", "## Rule 2 grid: half-A mean success over the arms", "", "| v | g=1 | g=2 | g=3 |", "|---|---|---|---|"]
    for v in sorted({parse_r2(k)[0] for k in r2k}):
        lines.append(f"| {v:.2f} | " + " | ".join(f"{s2[f'r2_v{v:.2f}_g{g}']:.3f}" for g in (1, 2, 3)) + " |")
    lines += ["", "## Rule 1 grid: half-A mean success over the arms (best 10)", "", "| key | success |", "|---|---|"]
    lines += [f"| {k} | {s1[k]:.3f} |" for k in sorted(r1k, key=lambda k: -s1[k])[:10]]
    lines += ["", "## Per category (all, picked rules)", "", "| stop | arm | category | runs | success | oracle |",
              "|---|---|---|---|---|---|"]
    for key in ("today", p2):
        for a in args.arms:
            cats = defaultdict(list)
            for r in bench[a]:
                cats[r["category"]].append(r)
            for c in sorted(cats):
                m = metrics(cats[c], key)
                lines.append(f"| {key} | {a} | {c} | {m['runs']} | {m['success']:.3f} | {m['oracle']:.3f} |")
    write_text_atomic(out / "stopbench.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:12]), flush=True)


def cmd_truth_report(args) -> None:
    """Truth re-score vs the recogniser on the same runs (module docstring), keys from STOP_PICK.json."""
    from vint_train.mapmad.closed_loop.exam_stats import paired

    layout = RunLayout.load(args.config, subdir=args.subdir)
    half_of = practice(layout)["episode_half"]
    home_of = {e: x["home"] for e, x in practice_episodes(layout).items()}
    pick = json.loads((layout.base / "stopbench" / "STOP_PICK.json").read_text())
    keys = [("today", "today tau 0.45"), (pick["rule1_pick"], f"Rule 1 {pick['rule1_pick']}"),
            (pick["rule2_pick"], f"Rule 2 {pick['rule2_pick']}")]
    rec = read_bench(layout, "stopbench", args.arms)
    tru = read_bench(layout, args.out, args.arms)
    if any(len(rec[a]) != len(tru[a]) for a in args.arms):
        raise SystemExit(f"truth bench incomplete: { {a: (len(rec[a]), len(tru[a])) for a in args.arms} }")
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    res: Dict[str, Any] = {"keys": [k for k, _ in keys], "arms": {}}
    lines = ["# 14.1 stop bench re-scored with simulator-truth marks (report only)", "",
             "Same runs, same rules, same picks (no re-pick); the recogniser's marks replaced by the true target "
             "pixels (stop_rules.truth_marks: every true instance in view = a look with score 1.0). Rule loss = oracle "
             "- truth success (the rule misses with perfect marks); detector loss = truth - recogniser success (what "
             "the recogniser's marks cost); paired over episodes, home-clustered bootstrap 95 % CI (10,000 draws, "
             "seed 14401). All 219 practice episodes; seed 0.", "",
             "| stop | arm | oracle | success truth | success recogniser | rule loss | detector loss [CI] | "
             "capture truth | capture recog. | precision truth | precision recog. | wrong obj. truth | "
             "wrong obj. recog. | near miss truth | near miss recog. |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, label in keys:
        for a in args.arms:
            mt, mr = metrics(tru[a], key), metrics(rec[a], key)
            d = paired({r["episode_id"]: float(r["rules"][key]["success"]) for r in tru[a]},
                       {r["episode_id"]: float(r["rules"][key]["success"]) for r in rec[a]}, home_of, 10_000, 14401)
            res["arms"].setdefault(a, {})[key] = {"truth": mt, "recogniser": mr, "detector_loss": d,
                                                  "rule_loss": mt["oracle"] - mt["success"]}
            lines.append(f"| {label} | {a} | {f(mt['oracle'])} | {f(mt['success'])} | {f(mr['success'])} | "
                         f"{mt['oracle'] - mt['success']:.3f} | {d['diff']:+.3f} [{d['home_ci'][0]:+.3f}, "
                         f"{d['home_ci'][1]:+.3f}] | {f(mt['capture'])} | {f(mr['capture'])} | {f(mt['precision'])} | "
                         f"{f(mr['precision'])} | {f(mt['wrong_object'])} | {f(mr['wrong_object'])} | "
                         f"{f(mt['near_miss'])} | {f(mr['near_miss'])} |")
    lines += ["", "## Mean over the arms", "", "| stop | oracle | success truth | success recogniser | rule loss | "
              "detector loss | precision truth | precision recog. |", "|---|---|---|---|---|---|---|---|"]
    for key, label in keys:
        rows = [res["arms"][a][key] for a in args.arms]
        mean = lambda g: float(np.mean([g(r) for r in rows]))  # noqa: E731
        lines.append(f"| {label} | {mean(lambda r: r['truth']['oracle']):.3f} | {mean(lambda r: r['truth']['success']):.3f} | "
                     f"{mean(lambda r: r['recogniser']['success']):.3f} | {mean(lambda r: r['rule_loss']):.3f} | "
                     f"{mean(lambda r: r['detector_loss']['diff']):+.3f} | "
                     f"{mean(lambda r: r['truth']['precision'] or 0):.3f} | {mean(lambda r: r['recogniser']['precision'] or 0):.3f} |")
    out = layout.base / args.out
    write_text_atomic(out / "truth.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(out / "truth.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["score", "pick", "truth-report"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", required=True)
    p.add_argument("--arms", nargs="+", default=list(ARMS5))
    p.add_argument("--out", default="stopbench")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--force", action="store_true")
    p.add_argument("--marks", choices=["recogniser", "truth"], default="recogniser")
    args = p.parse_args(argv)
    {"score": cmd_score, "pick": cmd_pick, "truth-report": cmd_truth_report}[args.cmd](args)


if __name__ == "__main__":
    main()
