"""Phase 14b-3 tables (NoMaD side, numpy only; rules p14b3_rules.md; config p14b3_v3.yaml):

    python -m vint_train.mapmad.closed_loop.p14b3_tables practice --config <p14b3_v3.yaml>   # V3 practice check
    python -m vint_train.mapmad.closed_loop.p14b3_tables examb --config ...                  # V4 exam B, rows 1-4
    python -m vint_train.mapmad.closed_loop.p14b3_tables gate --config ...                   # draft G14b3
    python -m vint_train.mapmad.closed_loop.p14b3_tables addendum --config ...               # M21bh completion

Every run is scored from its log with today's rule at tau 0.45 (P*'s stop; p12_score.score_log): correct stops =
rule-stop success; oracle success; SPL (oracle and rule); stops; wrong-object stops; collision steps; stuck runs
(>= 100 collision steps in a row); W4 events; A-mark detours; reverse steps (v3's controller); zigzag and left/right
flips per 100 turning steps (p12_score.heading_change_deg).
practice: v2.1b + P* = the 14b-2 "both" cell (p14b2_recog/runs/detam, read only) vs v3 + P* (runs/practice: M3_P,
M3h_P, M3_Pnoam), seed 0, halves A / B / all; paired by episode, home-clustered 95 % CIs (report only).
examb: per-episode means over the seed indices (F: its one run), paired differences over the episodes both arms ran;
rows 1-3 one-sided home-clustered bootstrap p (10,000 draws, seed 14301) + Holm over rows 1-3; row 3 compares with
Phase 13's exam-B M21b logs (p13_claim/runs/examb, read only; base YOLOE, tau 0.45) on the same episodes and seeds.
A call cut by the budget fallback is "not run": its finished runs are listed, never used in a comparison.
addendum (after the gate; report only, not a gate row, no Holm): M21b + P* − M21bh + P* on exam B, correct stops,
paired by episode over the seed indices complete for BOTH arms, the row-1 method (one-sided home-clustered bootstrap,
10,000 draws, seed 14301; raw p); per arm: correct stops (SR), oracle, SPL (rule), false-stop rate (share of runs
whose rule stop is not a correct stop), STOP precision; M21bh's reverse pattern (p14b3_reverse). Writes
tables/addendum_m21bh.{md,json}; tables/examb.* stay as the gate made them.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.episodes import load_episodes
from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic
from vint_train.mapmad.closed_loop import p14b_bench as pb
from vint_train.mapmad.closed_loop.exam_stats import paired
from vint_train.mapmad.closed_loop.p14_stopbench import practice, practice_episodes

TAU = 0.45
DRAWS, SEED = 10000, 14301
NAMES = {"M21b_w4_d_am": "MapMaD v2.1b, maps shown + P*", "M21bh_w4_d_am": "MapMaD v2.1b, maps hidden + P*",
         "M3_P": "MapMaD v3, maps shown + P*", "M3h_P": "MapMaD v3, maps hidden + P*",
         "M3_Pnoam": "MapMaD v3, maps shown + P* without drive-to-the-mark", "N_w4_d_am": "Original NoMaD explore + P*",
         "F_d": "Map + frontier planner (FT detector)", "M21b": "MapMaD v2.1b, Phase 13 pipeline (base YOLOE)"}
METRICS = ("oracle", "success", "spl", "rule_spl", "stopped", "wrong", "collisions", "stuck", "unstuck",
           "amark_trigger", "amark_confirmed", "amark_steps", "reverse_steps", "reverse_osc", "zigzag", "flips")
ROWS = (("1", "map effect with the finished pipeline", "M3_P", "M3h_P", "runs/examb"),
        ("2", "retraining effect", "M3_P", "M21b_w4_d_am", "runs/examb"),
        ("3", "pipeline effect (cross-run)", "M21b_w4_d_am", "M21b", "p13"))
P13_EXAMB = "p13_claim/runs/examb"


def fmt(x: Optional[float], p: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"


def sg(x: Optional[float], ci: Optional[Sequence[float]] = None) -> str:
    if x is None:
        return "-"
    return f"{x:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")


def holm(ps: Sequence[float]) -> List[float]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1), in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


# --- one run ----------------------------------------------------------------------------------------------------------
def run_row(log_file: Path, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One run's metrics (module docstring)."""
    from mapmad_bridge import steplog
    from vint_train.mapmad.closed_loop.p12_score import heading_change_deg, score_log

    log = steplog.read(log_file)
    s = score_log([dict(log[0], instances=instances)] + log[1:], taus=(TAU,))
    r = s["rule"][f"{TAU:.2f}"]
    res = log[-1]
    ev = res.get("amark_events") or {}
    steps = steplog.steps(log)
    rev = [bool((x.get("policy") or {}).get("reverse")) for x in steps]
    zig, flips = heading_change_deg(log)
    return {"episode_id": log[0]["episode_id"], "seed_index": int(log[0]["seed_index"]), "home": str(log[0]["home"]),
            "category": str(log[0]["category"]), "bin": int(log[0]["distance_bin"]),
            "oracle": float(s["oracle_success"]), "spl": float(s["oracle_spl"]), "success": float(r["success"]),
            "rule_spl": float(r["spl"]), "stopped": float(r["stopped"]), "wrong": float(bool(r.get("wrong_object"))),
            "collisions": float(s["collisions"]), "stuck": float(s["stuck_streaks"] > 0),
            "unstuck": float(res.get("unstuck_events", 0) or 0), "amark_trigger": float(ev.get("trigger", 0)),
            "amark_confirmed": float(ev.get("confirmed", 0)), "amark_steps": float(res.get("amark_steps", 0) or 0),
            "reverse_steps": float(sum(rev)), "reverse_osc": float(reverse_oscillations(rev, [x["v"] for x in steps])),
            "zigzag": zig, "flips": flips}


def reverse_oscillations(rev: Sequence[bool], v: Sequence[float]) -> int:
    """Reverse steps (v3's controller) followed by a forward step within 2 steps and a reverse step again within 4
    steps (forward / back oscillation; the fallback "W1-turn" fires above 20 % of reverse steps, p14b3_rules.md §4)."""
    n = 0
    for t, r in enumerate(rev):
        if r and any(v[k] > 0 for k in range(t + 1, min(t + 3, len(v)))) and any(rev[t + 2:t + 5]):
            n += 1
    return n


def _job(job) -> Tuple[str, Dict[str, Any]]:
    key, log_file, insts = job
    return key, run_row(Path(log_file), insts)


def load_rows(sources: Dict[str, Tuple[Path, str]], eps: Dict[str, Dict[str, Any]], workers: int,
              seeds: Optional[Sequence[int]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """name -> run rows, from sources name -> (run root, arm folder); only episodes of `eps` (and seeds)."""
    jobs = []
    for name, (root, arm) in sources.items():
        for f in sorted((root / "logs" / arm).glob("*.jsonl")):
            e, k = parse_run_id(f.name[:-len(".jsonl")])
            if e in eps and (seeds is None or k in seeds):
                jobs.append((name, str(f), eps[e]["target"]["instances"]))
    out: Dict[str, List[Dict[str, Any]]] = {n: [] for n in sources}
    for name, row in pb.imap_safe(_job, jobs, workers):
        out[name].append(row)
    return out


def ep_means(rows: Sequence[Dict[str, Any]], metric: str, keep=lambda r: True) -> Dict[str, float]:
    """Per-episode mean over its runs (seeds) of a metric."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if keep(r):
            acc[r["episode_id"]].append(r[metric])
    return {e: float(np.mean(v)) for e, v in acc.items()}


def summary(rows: Sequence[Dict[str, Any]], keep=lambda r: True) -> Dict[str, Any]:
    """Means over episodes (seed means first) of every metric, pooled STOP precision and wrong-object share."""
    sel = [r for r in rows if keep(r)]
    if not sel:
        return {"runs": 0}
    out: Dict[str, Any] = {"runs": len(sel), "episodes": len({r["episode_id"] for r in sel})}
    for m in METRICS:
        out[m] = float(np.mean(list(ep_means(sel, m).values())))
    stops = sum(r["stopped"] for r in sel)
    out["precision"] = (sum(r["success"] for r in sel) / stops) if stops else None
    rv = sum(r["reverse_steps"] for r in sel)
    out["reverse_osc_share"] = (sum(r["reverse_osc"] for r in sel) / rv) if rv else None
    out["wrong_share"] = (sum(r["wrong"] for r in sel) / stops) if stops else None
    return out


def compare(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]], home_of: Dict[str, str], metric: str,
            keep=lambda r: True) -> Dict[str, Any]:
    return paired(ep_means(a, metric, keep), ep_means(b, metric, keep), home_of, DRAWS, SEED)


def precision_change(a, b, home_of, keep=lambda r: True) -> Tuple[Optional[float], Optional[List[float]]]:
    def counts(rows):
        acc: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        for r in rows:
            if keep(r):
                acc[r["episode_id"]][0] += r["success"]
                acc[r["episode_id"]][1] += r["stopped"]
        return dict(acc)
    return pb.precision_diff_ci(counts(a), counts(b), home_of, DRAWS, SEED)


def arm_table(rows: Dict[str, List[Dict[str, Any]]], keep=lambda r: True, title: str = "") -> List[str]:
    lines = [title, "", "| arm | runs | oracle | correct stops | SPL (rule) | stops / run | STOP precision | wrong-object "
             "share of stops | collision steps | stuck runs | W4 events | A-mark detours | reverse steps | flips / 100 "
             "turning steps |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, rs in rows.items():
        s = summary(rs, keep)
        if not s["runs"]:
            lines.append(f"| {NAMES.get(name, name)} | 0 | not run | | | | | | | | | | | |")
            continue
        lines.append(f"| {NAMES.get(name, name)} (`{name}`) | {s['runs']} | {fmt(s['oracle'])} | {fmt(s['success'])} | "
                     f"{fmt(s['rule_spl'])} | {fmt(s['stopped'], 2)} | {fmt(s['precision'])} | {fmt(s['wrong_share'])} | "
                     f"{s['collisions']:.0f} | {fmt(s['stuck'])} | {s['unstuck']:.2f} | {s['amark_trigger']:.2f} | "
                     f"{s['reverse_steps']:.1f} | {s['flips']:.1f} |")
    return lines


# --- practice (V3) ----------------------------------------------------------------------------------------------------
PRAC_SOURCES = {"M21b_w4_d_am": ("p14b2_recog/runs/detam", "M21b_w4_d_am"),
                "M21bh_w4_d_am": ("p14b2_recog/runs/detam", "M21bh_w4_d_am"),
                "M3_P": (None, "M3_P"), "M3h_P": (None, "M3h_P"), "M3_Pnoam": (None, "M3_Pnoam")}
PRAC_PAIRS = (("v3 - v2.1b, maps shown", "M3_P", "M21b_w4_d_am"), ("v3 - v2.1b, maps hidden", "M3h_P", "M21bh_w4_d_am"),
              ("map effect v3 (shown - hidden)", "M3_P", "M3h_P"),
              ("map effect v2.1b (shown - hidden)", "M21b_w4_d_am", "M21bh_w4_d_am"),
              ("drive-to-the-mark in v3 (P* - P* without A-mark)", "M3_P", "M3_Pnoam"))


def cmd_practice(args) -> None:
    lay = RunLayout.load(args.config)
    eps = practice_episodes(lay)
    only = pb.episodes_filter()
    if only:
        eps = {e: x for e, x in eps.items() if e in only}
    half_of = practice(lay)["episode_half"]
    home_of = {e: x["home"] for e, x in eps.items()}
    src = {n: ((lay.outputs / s) if s else lay.base / "runs" / "practice", a) for n, (s, a) in PRAC_SOURCES.items()}
    rows = load_rows(src, eps, args.workers, seeds=(0,))
    res: Dict[str, Any] = {"arms": {}, "pairs": {}}
    lines = ["# V3 practice check (MP3D practice set 219 eps, seed 0; P* = FT detector + A-mark + W4, today's rule "
             "tau 0.45; report only)", ""]
    for half in ("all", "A", "B"):
        keep = (lambda r, h=half: h == "all" or half_of[r["episode_id"]] == h)
        res["arms"][half] = {n: summary(rs, keep) for n, rs in rows.items()}
        lines += arm_table(rows, keep, f"## Half {half}")
        lines += ["", f"### Paired differences, half {half} (home-clustered 95 % CI)", "",
                  "| comparison | correct stops [CI] | oracle [CI] | STOP precision change [CI] | collision steps | "
                  "stuck runs | flips / 100 | n |", "|---|---|---|---|---|---|---|---|"]
        res["pairs"][half] = {}
        for label, a, b in PRAC_PAIRS:
            if not (rows[a] and rows[b]):
                continue
            c = {m: compare(rows[a], rows[b], home_of, m, keep) for m in ("success", "oracle", "collisions", "stuck", "flips")}
            pdiff, pci = precision_change(rows[a], rows[b], home_of, keep)
            c["precision"] = {"diff": pdiff, "ci": pci}
            res["pairs"][half][label] = c
            lines.append(f"| {label} | {sg(c['success'].get('diff'), c['success'].get('home_ci'))} | "
                         f"{sg(c['oracle'].get('diff'), c['oracle'].get('home_ci'))} | {sg(pdiff, pci)} | "
                         f"{sg(c['collisions'].get('diff'))} | {sg(c['stuck'].get('diff'))} | "
                         f"{sg(c['flips'].get('diff'))} | {c['success'].get('n')} |")
        lines.append("")
    write_text_atomic(lay.base / "tables" / "practice.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(lay.base / "tables" / "practice.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:16]), flush=True)


# --- exam B (V4) ------------------------------------------------------------------------------------------------------
EXAM_SOURCES = ("M3_P", "M3h_P", "M21b_w4_d_am", "M21bh_w4_d_am", "N_w4_d_am", "F_d")


def complete_calls(lay: RunLayout, rows: Dict[str, List[Dict[str, Any]]], n_eps: int) -> Dict[str, Any]:
    """Per (arm, seed): finished runs; a (arm, seed) with fewer than all episodes = a call cut by the budget fallback
    ("not run": excluded from every comparison)."""
    from mapmad_sim.run_plan_p14b3 import EXAM_ORDER

    have = defaultdict(int)
    for arm, rs in rows.items():
        for r in rs:
            have[(arm, r["seed_index"])] += 1
    out = {}
    for label, arms, seeds in EXAM_ORDER:
        for a in arms:
            for k in seeds:
                out[f"{a}|{k}"] = {"call": label, "runs": have[(a, k)], "complete": have[(a, k)] >= n_eps}
    return out


def cmd_examb(args) -> None:
    lay = RunLayout.load(args.config)
    eps_list, fp = load_episodes(lay.base / lay.cfg["sets"]["examb"]["folder"] / "episodes.json")
    eps = {e["episode_id"]: e for e in eps_list}
    only = pb.episodes_filter()
    if only:
        eps = {e: x for e, x in eps.items() if e in only}
    home_of = {e: x["home"] for e, x in eps.items()}
    split_of = {e: ("val" if x.get("type") == "examb_val" else "test") for e, x in eps.items()}
    root = lay.base / "runs" / "examb"
    src = {a: (root, a) for a in EXAM_SOURCES}
    src["M21b"] = (lay.outputs / P13_EXAMB, "M21b")  # Phase 13 reference (read only)
    rows = load_rows(src, eps, args.workers)
    calls = complete_calls(lay, {a: rows[a] for a in EXAM_SOURCES}, len(eps))
    ok_seed = {(c.split("|")[0], int(c.split("|")[1])) for c, v in calls.items() if v["complete"]}
    used = {a: [r for r in rs if a == "M21b" or (a, r["seed_index"]) in ok_seed] for a, rs in rows.items()}
    seeds_used = {a: sorted({r["seed_index"] for r in rs}) for a, rs in used.items()}
    # row 3 pairs by episode AND seed index: Phase 13's M21b only on the seed indices M21b + P* finished
    used["M21b"] = [r for r in used["M21b"] if r["seed_index"] in set(seeds_used["M21b_w4_d_am"])]
    res: Dict[str, Any] = {"fingerprint": fp, "episodes": len(eps), "calls": calls, "seeds_used": seeds_used,
                           "rows": {}, "arms": {}}
    pvals = []
    for num, label, a, b, _ in ROWS:
        c = compare(used[a], used[b], home_of, "success")
        res["rows"][num] = {"label": label, "a": a, "b": b, **c}
        pvals.append(c.get("home_p_one_sided", 1.0) if c else 1.0)
    for (num, *_), p in zip(ROWS, holm(pvals)):
        res["rows"][num]["holm_p"] = p
        r = res["rows"][num]
        r["reading"] = ("passes: claim as stated" if r.get("diff") is not None and r["home_p_one_sided"] < 0.05
                        and p < 0.05 else "no evidence (pre-set reading; no extra runs)")
    for num, label, a, b, _ in ROWS:  # report-only companions of rows 1-3
        r = res["rows"][num]
        r["oracle"] = compare(used[a], used[b], home_of, "oracle")
        r["spl"] = compare(used[a], used[b], home_of, "rule_spl")
        r["precision"] = dict(zip(("diff", "ci"), precision_change(used[a], used[b], home_of)))
        r["by_split"] = {s: compare(used[a], used[b], home_of, "success", lambda x, s=s: split_of[x["episode_id"]] == s)
                         for s in ("val", "test")}
    res["arms"] = {a: summary(rs) for a, rs in used.items()}
    lines = ["# V4 exam B (MP3D, 291 episodes, 26 homes; second use; P* everywhere; correct stops = today's rule "
             "tau 0.45)", "", f"Fingerprint `{fp[:16]}…`; seeds used per arm: "
             + ", ".join(f"{a} {v}" for a, v in seeds_used.items()), "",
             "## Rows 1-3 (one-sided home-clustered bootstrap, 10,000 draws; Holm over rows 1-3)", "",
             "| row | comparison | correct stops a / b | difference [95 % CI] | raw p | Holm p | reading | n eps | homes |",
             "|---|---|---|---|---|---|---|---|---|"]
    for num, r in res["rows"].items():
        sa, sb = summary(used[r["a"]]), summary(used[r["b"]])
        lines.append(f"| {num} | {r['label']}: {NAMES[r['a']]} − {NAMES[r['b']]} | {fmt(sa.get('success'))} / "
                     f"{fmt(sb.get('success'))} | {sg(r.get('diff'), r.get('home_ci'))} | {fmt(r.get('home_p_one_sided'), 4)} | "
                     f"{fmt(r.get('holm_p'), 4)} | {r['reading']} | {r.get('n')} | {r.get('homes')} |")
    lines += ["", "## Row 4: companions of rows 1-3 (report only)", "",
              "| row | oracle [CI] | SPL (rule) [CI] | STOP precision change [CI] | correct stops val [CI] | test [CI] |",
              "|---|---|---|---|---|---|"]
    for num, r in res["rows"].items():
        lines.append(f"| {num} | {sg(r['oracle'].get('diff'), r['oracle'].get('home_ci'))} | "
                     f"{sg(r['spl'].get('diff'), r['spl'].get('home_ci'))} | {sg(r['precision']['diff'], r['precision']['ci'])} | "
                     f"{sg(r['by_split']['val'].get('diff'), r['by_split']['val'].get('home_ci'))} | "
                     f"{sg(r['by_split']['test'].get('diff'), r['by_split']['test'].get('home_ci'))} |")
    lines += [""] + arm_table(used, title="## All arms (per-episode seed means)")
    for key, title, vals in (("bin", "distance bin", (0, 1, 2)),
                             ("category", "category", sorted({x["target"]["category"] for x in eps.values()}))):
        lines += ["", f"## Correct stops / oracle per {title}", "",
                  "| arm | " + " | ".join(str(v) for v in vals) + " |", "|---|" + "---|" * len(vals)]
        res[f"by_{key}"] = {}
        for a, rs in used.items():
            cells = []
            res[f"by_{key}"][a] = {}
            for v in vals:
                s = summary(rs, lambda r, v=v: r[key] == v)
                res[f"by_{key}"][a][str(v)] = s
                cells.append(f"{fmt(s.get('success'))} / {fmt(s.get('oracle'))}" if s["runs"] else "-")
            lines.append(f"| {NAMES.get(a, a)} | " + " | ".join(cells) + " |")
    not_run = [k for k, v in calls.items() if not v["complete"]]
    lines += ["", "## Calls", "", f"Not run (budget fallback, finished runs listed only): {not_run or 'none'}", ""]
    write_text_atomic(lay.base / "tables" / "examb.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(lay.base / "tables" / "examb.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:12]), flush=True)


# --- addendum: M21bh completion ---------------------------------------------------------------------------------------
ADD_A, ADD_B = "M21b_w4_d_am", "M21bh_w4_d_am"


def with_false_stops(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows + `false_stop` = 1 if the run's rule stop is not a correct stop (a correct stop is always a stop)."""
    return [dict(r, false_stop=float(r["stopped"] and not r["success"])) for r in rows]


def addendum_arm(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-arm numbers of the addendum (per-episode seed means, as every exam-B arm)."""
    s = summary(rows)
    if not s["runs"]:
        return s
    s["false_stop_rate"] = float(np.mean(list(ep_means(rows, "false_stop").values())))
    return s


def cmd_addendum(args) -> None:
    from vint_train.mapmad.closed_loop.p14b3_reverse import arm_report

    lay = RunLayout.load(args.config)
    eps_list, fp = load_episodes(lay.base / lay.cfg["sets"]["examb"]["folder"] / "episodes.json")
    eps = {e["episode_id"]: e for e in eps_list}
    home_of = {e: x["home"] for e, x in eps.items()}
    root = lay.base / "runs" / "examb"
    rows = {a: with_false_stops(rs) for a, rs in load_rows({a: (root, a) for a in (ADD_A, ADD_B)}, eps, args.workers).items()}
    calls = complete_calls(lay, rows, len(eps))
    both = sorted(k for k in (0, 1, 2) if calls[f"{ADD_A}|{k}"]["complete"] and calls[f"{ADD_B}|{k}"]["complete"])
    used = {a: [r for r in rs if r["seed_index"] in both] for a, rs in rows.items()}
    c = compare(used[ADD_A], used[ADD_B], home_of, "success") if both else {}
    arms = {a: addendum_arm(rs) for a, rs in used.items()}
    rev_files = sorted(str(f) for f in (root / "logs" / ADD_B).glob("*.jsonl")
                       if parse_run_id(f.name[:-len(".jsonl")])[1] in both)
    rev = arm_report(rev_files) if rev_files else {}
    partial = {k: v for k, v in calls.items() if k.startswith(ADD_B) and not v["complete"]}
    res = {"fingerprint": fp, "episodes": len(eps), "seeds_both": both, "calls": {k: v for k, v in calls.items()
           if k.split("|")[0] in (ADD_A, ADD_B)}, "comparison": {"a": ADD_A, "b": ADD_B, "metric": "success", **c},
           "arms": arms, "reverse_m21bh": rev, "partial_m21bh_calls": partial}
    pa, pb_ = arms.get(ADD_A, {}), arms.get(ADD_B, {})
    lines = ["# Addendum: M21bh completion (exam B, report only — not a gate row, no Holm)", "",
             "An arm the frozen rules (p14b3_rules.md, v1.1) fixed in advance, cut by the budget fallback, completed "
             "afterward with no changes; third look at exam B, with no choice made or tuned from it.", "",
             f"Fingerprint `{fp[:16]}…`; seed indices complete for both arms: {both or 'none'}"
             + (f"; unfinished M21bh calls (not used): {sorted(partial)}" if partial else ""), "",
             "| comparison | correct stops a / b | difference [95 % CI] | raw p (one-sided) | n eps | homes |",
             "|---|---|---|---|---|---|",
             f"| {NAMES[ADD_A]} − {NAMES[ADD_B]} | {fmt(pa.get('success'))} / {fmt(pb_.get('success'))} | "
             f"{sg(c.get('diff'), c.get('home_ci'))} | {fmt(c.get('home_p_one_sided'), 4)} | {c.get('n')} | {c.get('homes')} |",
             "", "| arm | runs | correct stops (SR) | oracle | SPL (rule) | false-stop rate | STOP precision | "
             "reverse steps |", "|---|---|---|---|---|---|---|---|"]
    for a, s_ in arms.items():
        if not s_.get("runs"):
            lines.append(f"| {NAMES[a]} | 0 | not run | | | | | |")
            continue
        lines.append(f"| {NAMES[a]} (`{a}`) | {s_['runs']} | {fmt(s_['success'])} | {fmt(s_['oracle'])} | "
                     f"{fmt(s_['rule_spl'])} | {fmt(s_['false_stop_rate'])} | {fmt(s_['precision'])} | "
                     f"{s_['reverse_steps']:.1f} |")
    if rev:
        lines += ["", f"M21bh reverse pattern (p14b3_reverse, {rev['runs']} runs): share by phase "
                  f"{rev['reverse_share_by_phase']}, longest backward stretch {rev['stretch_max']} steps."]
    write_text_atomic(lay.base / "tables" / "addendum_m21bh.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(lay.base / "tables" / "addendum_m21bh.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# --- draft gate -------------------------------------------------------------------------------------------------------
def cmd_gate(args) -> None:
    lay = RunLayout.load(args.config)
    base = lay.base
    rd = lambda p: json.loads((base / p).read_text()) if (base / p).exists() else {}  # noqa: E731
    ex, chk = rd("tables/examb.json"), rd("checks/checks.json")
    g10 = rd("offline/v3t/recovery.json") or rd("offline/v3/recovery.json")  # (v3t = after the W1-turn fallback)
    prac = rd("tables/practice.json")
    lines = ["# G14b3 (draft, written by the chain)", "", "| # | Check | Pass rule | Verdict |", "|---|---|---|---|"]
    for num in ("1", "2", "3"):
        r = (ex.get("rows") or {}).get(num)
        v = (f"{sg(r.get('diff'), r.get('home_ci'))}, raw p {fmt(r.get('home_p_one_sided'), 4)}, Holm p "
             f"{fmt(r.get('holm_p'), 4)} → {r['reading']}") if r else "not available"
        lines.append(f"| {num} | {r['label'] if r else ''} | reported with p (raw + Holm) | {v} |")
    lines.append(f"| 4 | Oracle, SPL, precision, wrong-object, collisions, stuck, bins, categories; practice check; "
                 f"G10 rows 3–4; recovery share; past-move error | reported | tables/examb.md, tables/practice.md "
                 f"({'present' if prac else 'missing'}), offline/v3 ({'present' if g10 else 'missing'}) |")
    for num, key, rule in (("5", "g10_rows_1_2", "G10 rows 1–2 must pass"), ("6", "contamination", "no contamination"),
                           ("7", "fairness", "fairness"), ("8", "determinism", "determinism + Phase 13 reproduction")):
        c = chk.get(key) or {}
        extra = ""
        if num == "8":
            o13 = chk.get("off13") or {}
            extra = f"; Phase 13 reproduction: {'PASS' if o13.get('pass') else 'FAIL / missing'}"
            c = {"pass": bool(c.get("pass")) and bool(o13.get("pass")), "summary": c.get("summary", "")}
        if num == "6":
            r = chk.get("exam_rules_sha256") or {}
            c = {"pass": bool(c.get("pass")) and bool(r.get("pass")), "summary": c.get("summary", "") +
                 f"; exam rules sha256 unchanged: {r.get('pass')}"}
        lines.append(f"| {num} | {rule} | must pass | {'**PASS**' if c.get('pass') else '**FAIL / missing**'} "
                     f"({c.get('summary', '')}{extra}) |")
    write_text_atomic(base / "G14b3_table_draft.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["practice", "examb", "gate", "addendum"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args(argv)
    {"practice": cmd_practice, "examb": cmd_examb, "gate": cmd_gate, "addendum": cmd_addendum}[args.cmd](args)


if __name__ == "__main__":
    main()
