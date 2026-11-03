"""Phase 14a tables (kickoff 14.2, 14.4, 14.6; p14a_rules.md §5, §8-§9). numpy only (naz_mapmad).

    python -m vint_train.mapmad.closed_loop.p14_tables qwen --config <p14a_finish.yaml>     # qwen/qwen.md + QWEN_KEEP.json
    python -m vint_train.mapmad.closed_loop.p14_tables drive --config <p14a_finish.yaml>    # tables/driving.md + .json
    python -m vint_train.mapmad.closed_loop.p14_tables gate --config <p14a_finish.yaml>     # tables/tables.md + G14a draft
    python -m vint_train.mapmad.closed_loop.p14_tables qwen-audit --config <p14a_finish.yaml>  # qwen/audit.md (report only)

qwen (§5): Rule 2's pick with the Qwen3-VL check (qwen/runs/<arm>/<run>.json, mapmad/scripts/p14_qwen_check.py) vs
without it, judged on the logs like the stop bench; kept if on half B the mean rule-stop success over the 5 arms rises
by >= +0.02 AND the pooled STOP precision (correct stops / stops over the 5 arms) does not fall; latency per call and
calls per run.
drive (§8): per network arm (M21b, M21bh, N) and switch (A1 = <arm>_a1, W4 = <arm>_w4, A1 + W4 = <arm>_a1w4) vs the
baseline run of the same episode (runs/baseline): oracle success, SPL, rule-stop success (today tau 0.45 and the main
stop), left/right flips per 100 turning steps, zigzag deg / step, collision steps per run, stuck runs (a streak of >= 100
consecutive collision steps), steps to the first true sight; paired differences with home-clustered bootstrap 95 % CIs
(10,000 draws, seed 14401); map effect M21b - M21bh per switch; the §4b reading (mean over the 3 arms: "helps" if oracle
success is not lower by more than 0.03 AND the target metric improves: A1 flips -25 % relative or more; W4 stuck runs
-25 % relative or more and collision steps per run lower).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_bridge import steplog
from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic
from vint_train.mapmad.closed_loop.exam_stats import cluster_bootstrap, paired
from vint_train.mapmad.closed_loop.p12_score import heading_change_deg, score_log
from vint_train.mapmad.closed_loop.p14_stopbench import (ARMS5, HALVES, in_half, judge, metrics, on_wrong_object,
                                                          practice, practice_episodes)

NET_ARMS = ("M21b", "M21bh", "N")
SWITCHES = {"A1": ("a1", "_a1"), "W4": ("w4", "_w4"), "A1+W4": ("a1w4", "_a1w4")}  # name -> (subdir, arm suffix)
BOOT_SEED = 14401
N_BOOT = 10_000


def f3(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.3f}"


# --- 14.2 ---------------------------------------------------------------------------------------------------------
def cmd_qwen(args) -> None:
    layout = RunLayout.load(args.config, subdir="runs/baseline")
    pr = practice(layout)
    half_of = pr["episode_half"]
    eps = practice_episodes(layout)
    pick = json.loads((layout.base / "stopbench" / "STOP_PICK.json").read_text())["rule2_pick"]
    rows: Dict[str, List[Dict[str, Any]]] = {}
    calls: List[float] = []
    per_run_calls: List[int] = []
    vetoes = 0
    for a in ARMS5:
        rows[a] = []
        for f in sorted((layout.base / "qwen" / "runs" / a).glob("*.json")):
            rid = f.name[:-len(".json")]
            q = json.loads(f.read_text())
            bench = json.loads((layout.base / "stopbench" / "runs" / a / f"{rid}.json").read_text())
            log = steplog.read(layout.root / "logs" / a / f"{rid}.jsonl")
            steps = steplog.steps(log)
            vp = [log[0]["start_vp_geodesic_m"]] + [s["vp_geodesic_m"] for s in steps]
            path = np.concatenate([[0.0], np.cumsum([s["travelled_m"] for s in steps])])
            stop = None if q["stop"] is None else (int(q["stop"][0]), tuple(q["stop"][1]))
            j = judge(stop, vp, path, log[0]["start_vp_geodesic_m"], eps[parse_run_id(rid)[0]]["target"]["instances"])
            rows[a].append(dict(bench, rules={"plain": bench["rules"][pick], "qwen": j}))
            per_run_calls.append(int(q["asks"]))
            vetoes += int(q["vetoes"])
            ans = layout.base / "qwen" / "answers" / a / f"{rid}.jsonl"
            if ans.exists():
                calls += [float(json.loads(x)["ms"]) for x in ans.read_text().splitlines() if json.loads(x)["ms"] > 0]
    expected = len(half_of)
    missing = {a: expected - len(rows[a]) for a in ARMS5}

    def pooled_precision(h: str, key: str) -> Optional[float]:
        st = [r["rules"][key] for a in ARMS5 for r in in_half(rows[a], h, half_of) if r["rules"][key]["step"] is not None]
        return float(np.mean([x["success"] for x in st])) if st else None

    res: Dict[str, Any] = {"rule2_pick": pick, "missing": missing}
    for h in HALVES:
        res[h] = {k: {"mean_success": float(np.mean([metrics(in_half(rows[a], h, half_of), k)["success"] for a in ARMS5])),
                      "pooled_precision": pooled_precision(h, k),
                      "per_arm": {a: metrics(in_half(rows[a], h, half_of), k) for a in ARMS5}} for k in ("plain", "qwen")}
    gain = res["B"]["qwen"]["mean_success"] - res["B"]["plain"]["mean_success"]
    pq, pp = res["B"]["qwen"]["pooled_precision"], res["B"]["plain"]["pooled_precision"]
    keep = gain >= 0.02 - 1e-12 and (pq is not None and (pp is None or pq >= pp - 1e-12))
    res.update(keep=bool(keep), half_B_gain=gain, latency_ms={"mean": float(np.mean(calls)) if calls else None,
                                                              "p95": float(np.percentile(calls, 95)) if calls else None,
                                                              "calls": len(calls)},
               calls_per_run=float(np.mean(per_run_calls)) if per_run_calls else None, vetoes=vetoes)
    lines = ["# 14.2 Qwen3-VL check on top of Rule 2's pick (practice set)", "",
             f"Rule 2 pick `{pick}`. Kept if on half B the mean rule-stop success over the 5 arms rises by >= +0.02 and "
             f"the pooled STOP precision does not fall (§4b). **Result: {'KEPT' if keep else 'DROPPED'}** "
             f"(half-B gain {gain:+.3f}; pooled precision {f3(pp)} -> {f3(pq)}).", "",
             f"Calls: {res['latency_ms']['calls']} answered (latency mean {f3(res['latency_ms']['mean'])} ms, p95 "
             f"{f3(res['latency_ms']['p95'])} ms); {f3(res['calls_per_run'])} asks per run; {vetoes} vetoes. Missing "
             f"runs: {missing}.", "",
             "| half | stop | mean success (5 arms) | pooled STOP precision | " + " | ".join(ARMS5) + " |",
             "|---|---|---|---|" + "---|" * len(ARMS5)]
    for h in HALVES:
        for k in ("plain", "qwen"):
            r = res[h][k]
            lines.append(f"| {h} | {'Rule 2' if k == 'plain' else 'Rule 2 + Qwen'} | {r['mean_success']:.3f} | "
                         f"{f3(r['pooled_precision'])} | " +
                         " | ".join(f"{r['per_arm'][a]['success']:.3f}" for a in ARMS5) + " |")
    out = layout.base / "qwen"
    write_text_atomic(out / "QWEN_KEEP.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(out / "qwen.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def box_overlap(asked: Sequence[float], true: Sequence[float]) -> float:
    """Intersection area / asked-box area of two (x0, y0, x1, y1) pixel boxes."""
    w = min(asked[2], true[2]) - max(asked[0], true[0])
    h = min(asked[3], true[3]) - max(asked[1], true[1])
    area = max(asked[2] - asked[0], 1e-9) * max(asked[3] - asked[1], 1e-9)
    return max(w, 0.0) * max(h, 0.0) / area


def cmd_qwen_audit(args) -> None:
    """Report only (Part 3): how each Rule 2 + Qwen stop came about (a "yes"; the 3-ask / 20-step fallback after
    "unsure"; no ask at all: the fired cell was not in view within 20 steps) and Qwen's yes / no against the truth,
    on the map (the asked cell within 1.0 m of a true instance's box: the stop bench's wrong-object test) and in the
    picture (a true instance's box covers >= 25 % of the asked box in that frame) -> qwen/audit.md + audit.json."""
    from mapmad_sim import stop_evidence
    layout = RunLayout.load(args.config, subdir="runs/baseline")
    eps = practice_episodes(layout)
    how = {a: {"yes": 0, "fallback_unsure": 0, "no_ask": 0, "no_stop": 0} for a in ARMS5}
    conf = {a: {k: 0 for k in ("yes_true", "yes_false", "no_true", "no_false", "unsure_true", "unsure_false")}
            for a in ARMS5}
    pic = {k: 0 for k in ("yes_true", "yes_false", "no_true", "no_false")}
    sources = {"recogniser": 0, "projection": 0}
    for a in ARMS5:
        for f in sorted((layout.base / "qwen" / "runs" / a).glob("*.json")):
            rid = f.name[:-len(".json")]
            q = json.loads(f.read_text())
            ans_file = layout.base / "qwen" / "answers" / a / f"{rid}.jsonl"
            calls = [json.loads(x) for x in ans_file.read_text().splitlines()] if ans_file.exists() else []
            if q["stop"] is None:
                how[a]["no_stop"] += 1
            else:
                at = [c for c in calls if c["step"] == q["stop"][0]]
                how[a]["yes" if any(c["answer"] == "yes" for c in at) else "fallback_unsure" if at else "no_ask"] += 1
            if not calls:
                continue
            insts = eps[parse_run_id(rid)[0]]["target"]["instances"]
            ev = stop_evidence.load(layout.root / "evidence" / a / f"{rid}.npz")
            for c in calls:
                truth = bool(insts) and not on_wrong_object(c["cell"], insts)
                conf[a][f"{c['answer']}_{'true' if truth else 'false'}"] += 1
                sources[c["source"]] = sources.get(c["source"], 0) + 1
                if c["box"] is not None and c["answer"] in ("yes", "no"):
                    tb = ev["true_box"][ev["true_t"] == c["step"]]
                    seen = any(box_overlap(c["box"], b) >= 0.25 for b in tb if np.isfinite(b).all())
                    pic[f"{c['answer']}_{'true' if seen else 'false'}"] += 1
    tot = lambda d: {k: sum(x[k] for x in d.values()) for k in next(iter(d.values()))}  # noqa: E731
    H, C = tot(how), tot(conf)

    def acc(c: Dict[str, int]) -> Dict[str, Optional[float]]:
        yn = c["yes_true"] + c["yes_false"] + c["no_true"] + c["no_false"]
        r = lambda a_, b_: a_ / b_ if b_ else None  # noqa: E731
        return {"accuracy_yes_no": r(c["yes_true"] + c["no_false"], yn), "yes_precision": r(c["yes_true"], c["yes_true"] + c["yes_false"]),
                "target_recall": r(c["yes_true"], c["yes_true"] + c["no_true"]),
                "wrong_object_rejected": r(c["no_false"], c["no_false"] + c["yes_false"]), "answers_yes_no": yn}

    res = {"how_stopped": how, "how_stopped_total": H, "answers_map_truth": conf, "answers_map_truth_total": C,
           "map": acc(C), "picture": acc(pic), "answers_picture_truth": pic, "box_source": sources}
    lines = ["# 14.2 Qwen3-VL check: how the robot stopped and how right Qwen was (report only)", "",
             "Rule 2 + Qwen on the practice runs (seed 0). Truth on the map = the asked cell within 1.0 m of a true "
             "instance's box (the stop bench's wrong-object test); in the picture = a true instance's box covers "
             ">= 25 % of the asked box in that frame.", "",
             "## How each run's stop came about", "", "| arm | stop after yes | stop after unsure (3 asks / 20 steps) | "
             "stop without asking (cell not in view within 20 steps) | no stop |", "|---|---|---|---|---|"]
    for a in list(ARMS5) + ["all"]:
        h = H if a == "all" else how[a]
        lines.append(f"| {a} | {h['yes']} | {h['fallback_unsure']} | {h['no_ask']} | {h['no_stop']} |")
    n_stops = H["yes"] + H["fallback_unsure"] + H["no_ask"]
    lines += ["", f"Stops without asking: {H['no_ask']} of {n_stops} stops "
              f"({100 * H['no_ask'] / max(n_stops, 1):.1f} %).", "",
              "## Qwen's answers against the truth", "", "| arm | yes, target there | yes, no target | no, target there | "
              "no, no target | unsure, target | unsure, no target |", "|---|---|---|---|---|---|---|"]
    for a in list(ARMS5) + ["all"]:
        c = C if a == "all" else conf[a]
        lines.append(f"| {a} | " + " | ".join(str(c[k]) for k in ("yes_true", "yes_false", "no_true", "no_false",
                                                                     "unsure_true", "unsure_false")) + " |")
    for name, m in (("on the map", res["map"]), ("in the picture", res["picture"])):
        lines.append(f"- **{name}:** yes/no accuracy {f3(m['accuracy_yes_no'])} over {m['answers_yes_no']} answers; "
                     f"a yes is right {f3(m['yes_precision'])}; true targets kept (yes) {f3(m['target_recall'])}; "
                     f"wrong objects rejected (no) {f3(m['wrong_object_rejected'])}")
    lines.append(f"- Box source of the asks: {sources}")
    out = layout.base / "qwen"
    write_text_atomic(out / "audit.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(out / "audit.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# --- 14.4 ---------------------------------------------------------------------------------------------------------
def run_metrics(log_file: Path, bench_file: Path, main_key: str, instances) -> Dict[str, Any]:
    log = steplog.read(log_file)
    s = score_log([dict(log[0], instances=instances)] + log[1:], taus=(0.45,))
    zig, flips = heading_change_deg(log)
    b = json.loads(bench_file.read_text())
    return {"episode_id": log[0]["episode_id"], "home": log[0]["home"], "oracle": float(s["oracle_success"]),
            "spl": float(s["oracle_spl"]), "rule_today": float(b["rules"]["today"]["success"]),
            "rule_main": float(b["rules"][main_key]["success"]), "flips": flips, "zigzag": zig,
            "collisions": float(s["collisions"]), "stuck": float(s["stuck_streaks"] > 0),
            "first_sight": float(s["first_sight"]) if s["first_sight"] is not None else float(log[-1]["steps"]),
            "unstuck_events": float(log[-1].get("unstuck_events", 0) or 0)}


METRICS = ("oracle", "spl", "rule_today", "rule_main", "flips", "zigzag", "collisions", "stuck", "first_sight",
           "unstuck_events")


def read_arm(layout: RunLayout, subdir: str, arm: str, main_key: str, eps) -> Dict[str, Dict[str, Any]]:
    out = {}
    for f in sorted((layout.base / subdir / "logs" / arm).glob("*.jsonl")):
        rid = f.name[:-len(".jsonl")]
        bench = layout.base / "stopbench" / "runs" / arm / f"{rid}.json"
        out[parse_run_id(rid)[0]] = run_metrics(f, bench, main_key, eps[parse_run_id(rid)[0]]["target"]["instances"])
    return out


def mean_ci(vals: Dict[str, float], home_of: Dict[str, str]) -> Tuple[float, float, float]:
    ids = sorted(vals)
    m, lo, hi, _ = cluster_bootstrap(np.array([vals[i] for i in ids]), [home_of[i] for i in ids], N_BOOT, BOOT_SEED)
    return m, lo, hi


def cmd_drive(args) -> None:
    layout = RunLayout.load(args.config)
    eps = practice_episodes(layout)
    home_of = {e: x["home"] for e, x in eps.items()}
    main_key = json.loads((layout.base / "stopbench" / "STOP_PICK.json").read_text())["main_stop"]
    data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {"base": {a: read_arm(layout, "runs/baseline", a, main_key, eps)
                                                                       for a in NET_ARMS}}
    for sw, (sub, suf) in SWITCHES.items():
        data[sw] = {a: read_arm(layout, f"runs/{sub}", a + suf, main_key, eps) for a in NET_ARMS}
    res: Dict[str, Any] = {"main_stop": main_key, "arms": {}, "map_effect": {}, "reading": {}}
    lines = ["# 14.4 driving tests (practice set, seed 0; paired by episode vs the baseline runs)", "",
             f"Main stop = `{main_key}` (14.1 pick). CIs: home-clustered bootstrap 95 % (10,000 draws, seed {BOOT_SEED}).",
             "", "| arm | switch | runs | oracle | SPL | rule today | rule main | flips / 100 turns | zigzag deg/step | "
             "collision steps | stuck runs | first sight step | manoeuvres |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in NET_ARMS:
        res["arms"][a] = {}
        for sw in ["base"] + list(SWITCHES):
            d = data[sw][a]
            if not d:
                continue
            row = {m: mean_ci({e: r[m] for e, r in d.items()}, home_of) for m in METRICS}
            res["arms"][a][sw] = {"runs": len(d), "mean": {m: row[m][0] for m in METRICS},
                                  "ci": {m: row[m][1:] for m in METRICS}}
            if sw != "base":
                res["arms"][a][sw]["paired"] = {m: paired({e: r[m] for e, r in d.items()},
                                                          {e: r[m] for e, r in data["base"][a].items()}, home_of,
                                                          N_BOOT, BOOT_SEED) for m in METRICS}
            lines.append(f"| {a} | {sw} | {len(d)} | " + " | ".join(
                f"{row[m][0]:.3f}" if m in ("oracle", "spl", "rule_today", "rule_main", "stuck") else f"{row[m][0]:.1f}"
                for m in METRICS) + " |")
    lines += ["", "## Paired differences vs baseline (switch - baseline) [home CI]", "",
              "| arm | switch | oracle | flips / 100 turns | collision steps | stuck runs | rule today | rule main |",
              "|---|---|---|---|---|---|---|---|"]
    for a in NET_ARMS:
        for sw in SWITCHES:
            p = res["arms"][a].get(sw, {}).get("paired")
            if not p:
                continue
            cell = lambda m, d=3: f"{p[m]['diff']:+.{d}f} [{p[m]['home_ci'][0]:+.{d}f}, {p[m]['home_ci'][1]:+.{d}f}]"  # noqa: E731
            lines.append(f"| {a} | {sw} | {cell('oracle')} | {cell('flips', 1)} | {cell('collisions', 1)} | "
                         f"{cell('stuck')} | {cell('rule_today')} | {cell('rule_main')} |")
    lines += ["", "## Map effect M21b - M21bh (paired) per switch: oracle, rule today, rule main", "",
              "| switch | oracle [home CI] | rule today [home CI] | rule main [home CI] | n |", "|---|---|---|---|---|"]
    for sw in ["base"] + list(SWITCHES):
        m, h = data[sw]["M21b"], data[sw]["M21bh"]
        if not m or not h:
            continue
        pm = {k: paired({e: r[k] for e, r in m.items()}, {e: r[k] for e, r in h.items()}, home_of, N_BOOT, BOOT_SEED)
              for k in ("oracle", "rule_today", "rule_main")}
        res["map_effect"][sw] = pm["oracle"]
        res.setdefault("map_effect_rules", {})[sw] = {k: pm[k] for k in ("rule_today", "rule_main")}
        lines.append(f"| {sw} | " + " | ".join(f"{pm[k]['diff']:+.3f} [{pm[k]['home_ci'][0]:+.3f}, {pm[k]['home_ci'][1]:+.3f}]"
                                                for k in ("oracle", "rule_today", "rule_main")) + f" | {pm['oracle']['n']} |")
    lines += ["", "## §4b reading (mean over M21b, M21bh, N; paired, all episodes)", "",
              "| switch | oracle diff | target metric | relative change | reading |", "|---|---|---|---|---|"]
    for sw in SWITCHES:
        arms = [a for a in NET_ARMS if data[sw][a]]
        if len(arms) < len(NET_ARMS):
            continue
        od = float(np.mean([res["arms"][a][sw]["paired"]["oracle"]["diff"] for a in arms]))

        def rel(m: str) -> float:
            base = float(np.mean([res["arms"][a]["base"]["mean"][m] for a in arms]))
            new = float(np.mean([res["arms"][a][sw]["mean"][m] for a in arms]))
            return (new - base) / base if base > 0 else 0.0

        if sw == "A1":
            r = rel("flips")
            helps = od >= -0.03 - 1e-12 and r <= -0.25 + 1e-12
            target = f"flips {r:+.1%}"
        else:
            r = rel("stuck")
            coll = rel("collisions")
            helps = od >= -0.03 - 1e-12 and r <= -0.25 + 1e-12 and coll < 0
            target = f"stuck runs {r:+.1%}, collision steps {coll:+.1%}"
        res["reading"][sw] = {"oracle_diff": od, "helps": bool(helps), "target": target}
        lines.append(f"| {sw} | {od:+.3f} | {target} | | **{'helps' if helps else 'does not help'}** |")
    out = layout.base / "tables"
    write_text_atomic(out / "driving.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(out / "driving.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# --- 14.6 -----------------------------------------------------------------------------------------------------------
def read_text(p: Path) -> str:
    return p.read_text() if p.exists() else f"(missing: {p.name})\n"


def cmd_gate(args) -> None:
    layout = RunLayout.load(args.config)
    b = layout.base
    parts = [("Stop-rule bench (14.1)", b / "stopbench" / "stopbench.md"),
             ("Stop bench with simulator-truth marks (14.1, report only)", b / "stopbench_truth" / "truth.md"),
             ("Truth marks on the wrong object: projection smear (report only)", b / "stopbench_truth" / "smear_note.md"),
             ("Qwen3-VL check (14.2)", b / "qwen" / "qwen.md"),
             ("Qwen3-VL audit: how stops came about, answers vs truth (report only)", b / "qwen" / "audit.md"),
             ("Diagnosis (14.3)", b / "diagnosis" / "diagnosis.md"), ("F check (14.3)", b / "diagnosis" / "f_check.md"),
             ("Driving tests (14.4)", b / "tables" / "driving.md"),
             ("Mark check (14.5)", b / "videos_marks" / "mark_counts.md"), ("Fairness / determinism", b / "checks" / "checks.md")]
    body = ["# Phase 14a tables (practice set; report only unless stated)", ""]
    for title, p in parts:
        body += [f"## {title}", "", read_text(p).replace("\n# ", "\n### "), ""]
    write_text_atomic(b / "tables" / "tables.md", "\n".join(body) + "\n")
    pick = json.loads(read_text(b / "stopbench" / "STOP_PICK.json")) if (b / "stopbench" / "STOP_PICK.json").exists() else {}
    q = json.loads((b / "qwen" / "QWEN_KEEP.json").read_text()) if (b / "qwen" / "QWEN_KEEP.json").exists() else {}
    dr = json.loads((b / "tables" / "driving.json").read_text()) if (b / "tables" / "driving.json").exists() else {}
    ck = json.loads((b / "checks" / "checks.json").read_text()) if (b / "checks" / "checks.json").exists() else {}
    au = json.loads((b / "qwen" / "audit.json").read_text()) if (b / "qwen" / "audit.json").exists() else {}
    hs = au.get("how_stopped_total") or {}
    n_stop = sum(hs.get(k, 0) for k in ("yes", "fallback_unsure", "no_ask"))
    tr = json.loads((b / "stopbench_truth" / "truth.json").read_text()) if (b / "stopbench_truth" / "truth.json").exists() else {}
    tmean = (lambda key, g: float(np.mean([g(tr["arms"][a][key]) for a in tr["arms"]]))) if tr else None
    vids = sorted((b / "videos_marks").glob("*.mp4"))
    rows = [
        ("1", "Stop bench reported", f"Rule 1 pick `{pick.get('rule1_pick')}`, Rule 2 pick `{pick.get('rule2_pick')}` "
                                     "(main stop); halves A / B / all in stopbench.md" +
                                     (f"; truth marks (Rule 2, mean over arms): success {tmean(pick['rule2_pick'], lambda r: r['truth']['success']):.3f} "
                                      f"vs YOLOE {tmean(pick['rule2_pick'], lambda r: r['recogniser']['success']):.3f} "
                                      f"(oracle {tmean(pick['rule2_pick'], lambda r: r['truth']['oracle']):.3f})" if tr else "")),
        ("2", "Qwen check", "not done" if not q else f"{'kept' if q.get('keep') else 'dropped'} (half-B gain "
                                                      f"{q.get('half_B_gain', 0):+.3f}; latency mean "
                                                      f"{f3((q.get('latency_ms') or {}).get('mean'))} ms; "
                                                      f"{f3(q.get('calls_per_run'))} calls per run)" +
                                                      (f"; {hs.get('no_ask', 0)}/{n_stop} stops without asking; "
                                                       f"yes/no accuracy vs map truth {f3(au['map']['accuracy_yes_no'])}"
                                                       if au else "")),
        ("3", "Diagnosis", "class shares per arm in diagnosis.md; F check in f_check.md" +
                           (" (no bug: F's low agree share = no-goal and mark-approach windows + the strict angle test)"
                            if (b / "diagnosis" / "f_check.md").exists() else "")),
        ("4", "Driving tests", "; ".join(f"{k}: {'helps' if v['helps'] else 'does not help'}"
                                         for k, v in (dr.get("reading") or {}).items()) or "missing"),
        ("5", "Mark check", f"renderer answer (code_check.json), {len(vids)} videos, counts in mark_counts.md"),
        ("6", "Fairness", ck.get("fairness_summary", "missing")),
        ("7", "Determinism", ck.get("determinism_summary", "missing") + "; Phase 13: " +
                             ck.get("same_p13_summary", "missing"))]
    lines = ["# G14a gate table (draft)", "", "| # | Check | Result |", "|---|---|---|"]
    lines += [f"| {a} | {b_} | {c} |" for a, b_, c in rows]
    write_text_atomic(b / "G14a_table_draft.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["qwen", "qwen-audit", "drive", "gate"])
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    {"qwen": cmd_qwen, "qwen-audit": cmd_qwen_audit, "drive": cmd_drive, "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
