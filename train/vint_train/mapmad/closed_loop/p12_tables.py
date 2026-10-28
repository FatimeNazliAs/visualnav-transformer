"""Phase 12 tables from the per-run scores (p12_score.py `score` -> <subdir>/scores/<arm>.jsonl), numpy only.

    python -m vint_train.mapmad.closed_loop.p12_tables --config <p12_exam.yaml> [--out tables]

Statistics (block item 8, Phase 5 rules): per episode, a metric is first averaged over its seeds (a deterministic arm's
one run counts for all); arm means with a home-clustered bootstrap 95% CI; paired differences over the episodes both
arms ran (an episode with a crash record in either arm is dropped from that comparison), home-clustered bootstrap
(10,000, seed 12601): one-sided p only for the headline pair (G12 row 1), 95% CIs everywhere; episode-level CIs too.
Every stop: oracle and rule stop (tau from P12_DEV_CHOICES.json). Writes <out>/*.md + tables.json (byte-identical
when rerun on the same logs: the §0.8 re-check) and G12_table_draft.md in out_dir.
Tables: main, paired, bins, categories, homes, search / approach, mistakes (wrong-object stops, near-misses, STOP
precision), OVON per split, exploration (house seen at 250 / 500 / 1,000 steps, redundancy), unseeable set,
recogniser driving test (YOLOE vs the truth arm, all episodes; Qwen3-VL vs YOLOE on the subset, seed 0; Phase 9
photo-test F1 per category vs driving success per category).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.run_layout import RunLayout, write_text_atomic
from mapmad_sim.run_plan import PAIRS, RunPlan
from vint_train.mapmad.closed_loop.exam_stats import cluster_bootstrap, episode_bootstrap

BOOT_SEED = 12601
N_BOOT = 10_000
# Phase 9 photo test (half B, map-level F1 per category; v2 plan 8.4): recogniser -> category -> F1
PHOTO_F1 = {"qwen3vl": {"chair": 0.78, "sofa": 0.87, "bed": 0.83, "toilet": 0.83, "tv_monitor": 0.64, "plant": 0.33},
            "yoloe": {"chair": 0.48, "sofa": 0.79, "bed": 0.68, "toilet": 0.88, "tv_monitor": 0.63, "plant": 0.61}}
# display order of the arms in every table (which arms a table shows = the run plan's arms of that stage)
ARM_ORDER = ("M2", "M2h", "M21", "M21h", "MT", "N", "V1", "F", "Nfish", "MR1",
             "X_M2", "X_M2h", "X_M21", "X_M21h", "X_N", "X_V1", "X_F")

Row = Dict[str, Any]


def read_rows(base: Path, subdir: str, arm: str) -> List[Row]:
    f = base / subdir / "scores" / f"{arm}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def crashed(base: Path, subdir: str, arm: str) -> List[str]:
    from mapmad_sim.run_layout import parse_run_id

    return sorted({parse_run_id(p.name)[0] for p in (base / subdir / "summaries" / arm).glob("*.crash.json")})


def metric_fns(tau: str) -> Dict[str, Callable[[Row], Optional[float]]]:
    """Per-run metrics (None = not defined for this run)."""
    def rule(key: str) -> Callable[[Row], Optional[float]]:
        return lambda r: (float(r["rule"][tau][key]) if r.get("rule") else None)

    return {
        "oracle_success": lambda r: float(r["oracle_success"]),
        "oracle_spl": lambda r: float(r["oracle_spl"]),
        "oracle_success_500": lambda r: float(r["oracle_success_at_500"]),
        "rule_success": rule("success"),
        "rule_spl": rule("spl"),
        "rule_success_500": rule("success_at"),
        "seen": lambda r: float(r["seen"]),
        "collisions": lambda r: float(r["collisions"]),
        "stuck": lambda r: float(r["stuck_streaks"] > 0),
    }


def per_episode(rows: List[Row], fn: Callable[[Row], Optional[float]], keep: Optional[set] = None) -> Dict[str, float]:
    """episode -> mean over its runs of fn (runs where fn is None are left out; an episode with none is left out)."""
    by: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if keep is not None and r["episode_id"] not in keep:
            continue
        v = fn(r)
        if v is not None:
            by[r["episode_id"]].append(v)
    return {e: float(np.mean(v)) for e, v in sorted(by.items())}


def mean_ci(vals: Dict[str, float], home: Dict[str, str]) -> Tuple[float, float, float]:
    if not vals:
        return float("nan"), float("nan"), float("nan")
    ids = sorted(vals)
    m, lo, hi, _ = cluster_bootstrap(np.array([vals[i] for i in ids]), [home[i] for i in ids], N_BOOT, BOOT_SEED)
    return m, lo, hi


def paired_diff(a: Dict[str, float], b: Dict[str, float], home: Dict[str, str]) -> Dict[str, Any]:
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0}
    d = np.array([a[i] - b[i] for i in ids])
    h = cluster_bootstrap(d, [home[i] for i in ids], N_BOOT, BOOT_SEED)
    e = episode_bootstrap(d, N_BOOT, BOOT_SEED)
    return {"n": len(ids), "homes": len({home[i] for i in ids}), "diff": h[0], "home_ci": [h[1], h[2]],
            "home_p_one_sided": h[3], "episode_ci": [e[1], e[2]], "episode_p_one_sided": e[3]}


def fmt(m: float, lo: Optional[float] = None, hi: Optional[float] = None, d: int = 3) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "-"
    return f"{m:.{d}f}" + (f" [{lo:.{d}f}, {hi:.{d}f}]" if lo is not None else "")


def median_or_none(xs: Sequence[float]) -> Optional[float]:
    return float(np.median(xs)) if len(xs) else None


class Tables:
    """All Phase 12 tables of one out_dir (module docstring)."""

    def __init__(self, layout: RunLayout) -> None:
        self.layout, self.base, self.cfg = layout, layout.base, layout.cfg
        self.plan = RunPlan.from_layout(layout)
        ch = self.plan.table.choices() or {}
        self.choices = ch
        self.tau = f"{float(ch.get('tau', 1.0)):.2f}"
        self.headline = ch.get("headline")
        self.fns = metric_fns(self.tau)
        self.md: Dict[str, List[str]] = {}
        self.data: Dict[str, Any] = {"tau": self.tau, "headline": self.headline}

    def episodes(self, set_name: str) -> List[Dict[str, Any]]:
        from mapmad_sim.episodes import load_episodes

        f = self.base / self.cfg["sets"][set_name]["folder"] / "episodes.json"
        return load_episodes(f)[0] if f.exists() else []

    def arm_rows(self, subdir: str, arms: Sequence[str]) -> Dict[str, List[Row]]:
        return {a: read_rows(self.base, subdir, a) for a in arms if read_rows(self.base, subdir, a)}

    def headline_arms(self) -> Tuple[str, str]:
        return self.plan.headline_pair()

    def stage(self, name: str) -> Tuple[str, str, List[str]]:
        """(subdir, set, arms in display order) of a run-plan stage."""
        return self.plan.where(name) + (self.plan.stage_arms(name, ARM_ORDER),)

    # --- the main table and friends -----------------------------------------------------------------------------
    def arm_summary(self, rows: List[Row], home: Dict[str, str], keep: Optional[set] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"runs": len([r for r in rows if keep is None or r["episode_id"] in keep])}
        for k, fn in self.fns.items():
            vals = per_episode(rows, fn, keep)
            out[k] = mean_ci(vals, home)
            out["episodes"] = max(out.get("episodes", 0), len(vals))
        sel = [r for r in rows if keep is None or r["episode_id"] in keep]
        seen = [r for r in sel if r["seen"]]
        out["success_given_seen"] = float(np.mean([r["oracle_success"] for r in seen])) if seen else None
        out["rule_success_given_seen"] = (float(np.mean([r["rule"][self.tau]["success"] for r in seen]))
                                          if seen and sel[0].get("rule") else None)
        out["search_steps_median"] = median_or_none([r["first_sight"] if r["first_sight"] is not None else r["steps"]
                                                     for r in sel])
        out["found_by_1000"] = float(np.mean([r["seen"] for r in sel])) if sel else None
        out["oracle_approach_median"] = median_or_none([r["oracle_approach_steps"] for r in sel
                                                        if r["oracle_approach_steps"] is not None])
        if sel and sel[0].get("rule"):
            stops = [r["rule"][self.tau] for r in sel if r["rule"][self.tau]["stopped"]]
            out["rule_stops"] = len(stops)
            out["stop_precision"] = float(np.mean([s["success"] for s in stops])) if stops else None
            out["wrong_object_share"] = float(np.mean([s["wrong_object"] for s in stops])) if stops else None
            out["near_miss_share"] = float(np.mean([s["near_miss"] for s in stops])) if stops else None
            out["rule_approach_median"] = median_or_none([s["approach_steps"] for s in stops
                                                          if s["approach_steps"] is not None])
        return out

    def main_table(self, name: str, subdir: str, set_name: str, arms: Sequence[str], title: str) -> Dict[str, Any]:
        eps = self.episodes(set_name)
        home = {e["episode_id"]: e["home"] for e in eps}
        rows = self.arm_rows(subdir, arms)
        crash = set().union(*[set(crashed(self.base, subdir, a)) for a in rows]) if rows else set()
        keep = {e["episode_id"] for e in eps} - crash
        res = {a: self.arm_summary(r, home, keep) for a, r in rows.items()}
        lines = [f"## {title}", "", f"Set `{set_name}`: {len(eps)} episodes ({len(crash)} dropped for a crash record); "
                 f"tau = {self.tau}; per-episode means over seeds; 95% CI = home-clustered bootstrap (10,000, seed "
                 f"{BOOT_SEED}).", "",
                 "| arm | runs | oracle success | oracle SPL | rule success | rule SPL | success@500 (oracle / rule) | "
                 "seen share | success if seen | collisions | stuck runs |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, r in res.items():
            lines.append(f"| {a} | {r['runs']} | {fmt(*r['oracle_success'])} | {fmt(r['oracle_spl'][0])} | "
                         f"{fmt(*r['rule_success'])} | {fmt(r['rule_spl'][0])} | {fmt(r['oracle_success_500'][0])} / "
                         f"{fmt(r['rule_success_500'][0])} | {fmt(r['seen'][0])} | {fmt(r['success_given_seen'])} | "
                         f"{fmt(r['collisions'][0], d=1)} | {fmt(r['stuck'][0])} |")
        self.md[name] = lines
        return {"set": set_name, "subdir": subdir, "episodes": len(eps), "crash_episodes": sorted(crash), "arms": res}

    def phases_table(self, name: str, main: Dict[str, Any]) -> None:
        lines = [f"## Search and approach phases ({main['set']})", "",
                 "| arm | found by 1,000 | search steps (median) | approach steps oracle (median) | success if seen "
                 "(oracle / rule) | approach steps rule (median) |", "|---|---|---|---|---|---|"]
        for a, r in main["arms"].items():
            lines.append(f"| {a} | {fmt(r['found_by_1000'])} | {fmt(r['search_steps_median'], d=0)} | "
                         f"{fmt(r['oracle_approach_median'], d=0)} | {fmt(r['success_given_seen'])} / "
                         f"{fmt(r.get('rule_success_given_seen'))} | {fmt(r.get('rule_approach_median'), d=0)} |")
        lines += ["", f"## Rule-stop mistakes ({main['set']}, tau {self.tau})", "",
                  "| arm | rule stops | STOP precision | wrong-object stops (share of stops) | near-misses (share of stops) |",
                  "|---|---|---|---|---|"]
        for a, r in main["arms"].items():
            lines.append(f"| {a} | {r.get('rule_stops', '-')} | {fmt(r.get('stop_precision'))} | "
                         f"{fmt(r.get('wrong_object_share'))} | {fmt(r.get('near_miss_share'))} |")
        self.md[name] = lines

    def paired_table(self, name: str, subdir: str, set_name: str, pairs: Sequence[Tuple[str, str, bool]]) -> Dict[str, Any]:
        eps = self.episodes(set_name)
        home = {e["episode_id"]: e["home"] for e in eps}
        res: Dict[str, Any] = {}
        lines = [f"## Paired differences ({set_name}; a - b over the episodes both ran)", "",
                 "| a - b | episodes | oracle success | p (one-sided, homes) | rule success | oracle SPL | episode-level CI "
                 "(oracle success) |", "|---|---|---|---|---|---|---|"]
        for a, b, headline in pairs:
            ra, rb = read_rows(self.base, subdir, a), read_rows(self.base, subdir, b)
            if not ra or not rb:
                continue
            crash = set(crashed(self.base, subdir, a)) | set(crashed(self.base, subdir, b))
            keep = {e["episode_id"] for e in eps} - crash
            row = {k: paired_diff(per_episode(ra, self.fns[k], keep), per_episode(rb, self.fns[k], keep), home)
                   for k in ("oracle_success", "rule_success", "oracle_spl")}
            res[f"{a}-{b}"] = dict(row, headline=headline)
            o = row["oracle_success"]
            p = f"{o['home_p_one_sided']:.4f}" if headline else "(report only)"
            rs = row["rule_success"]
            lines.append(f"| {a} - {b}{' **(headline)**' if headline else ''} | {o['n']} | "
                         f"{o['diff']:+.3f} [{o['home_ci'][0]:+.3f}, {o['home_ci'][1]:+.3f}] | {p} | "
                         + (f"{rs['diff']:+.3f} [{rs['home_ci'][0]:+.3f}, {rs['home_ci'][1]:+.3f}]" if rs.get("n") else "-")
                         + f" | {row['oracle_spl']['diff']:+.3f} | [{o['episode_ci'][0]:+.3f}, {o['episode_ci'][1]:+.3f}] |")
        self.md[name] = lines
        return res

    def split_table(self, name: str, subdir: str, set_name: str, arms: Sequence[str], key: str, title: str,
                    label: Callable[[Any], str] = str) -> Dict[str, Any]:
        """Oracle / rule success per value of an episode field (distance_bin, category, home, type)."""
        eps = self.episodes(set_name)
        field = {e["episode_id"]: e.get(key, e["target"].get(key) if key == "category" else None) for e in eps}
        if key == "category":
            field = {e["episode_id"]: e["target"]["category"] for e in eps}
        groups = sorted({v for v in field.values()})
        rows = self.arm_rows(subdir, arms)
        crash = set().union(*[set(crashed(self.base, subdir, a)) for a in rows]) if rows else set()
        out: Dict[str, Any] = {}
        lines = [f"## {title}", "", "Oracle / rule success (episodes).", "",
                 "| arm | " + " | ".join(label(g) for g in groups) + " |", "|---|" + "---|" * len(groups)]
        for a, r in rows.items():
            cells, out[a] = [], {}
            for g in groups:
                keep = {e for e, v in field.items() if v == g} - crash
                o = per_episode(r, self.fns["oracle_success"], keep)
                ru = per_episode(r, self.fns["rule_success"], keep)
                out[a][str(g)] = {"episodes": len(o), "oracle": float(np.mean(list(o.values()))) if o else None,
                                  "rule": float(np.mean(list(ru.values()))) if ru else None}
                cells.append(f"{fmt(out[a][str(g)]['oracle'])} / {fmt(out[a][str(g)]['rule'])} ({len(o)})")
            lines.append(f"| {a} | " + " | ".join(cells) + " |")
        self.md[name] = lines
        return out

    def explore_table(self) -> Dict[str, Any]:
        subdir, set_name, arms = self.stage("explore")
        home = {e["episode_id"]: e["home"] for e in self.episodes(set_name)}
        rows = self.arm_rows(subdir, arms)
        out: Dict[str, Any] = {}
        lines = ["## Exploration side test (no target; 1 seed)", "", "House seen = share of the start storey's walkable "
                 "cells seen (in view, <= 3.0 m, line of sight on the true map); redundancy = 1 - distinct 0.5 m cells "
                 "/ cell entries.", "", "| arm | runs | house seen @250 | @500 | @1000 | redundancy | collisions |",
                 "|---|---|---|---|---|---|---|"]
        for a, r in rows.items():
            out[a] = {m: mean_ci(per_episode(r, lambda x, m=m: x["house_seen"][m]), home) for m in ("250", "500", "1000")}
            out[a]["redundancy"] = mean_ci(per_episode(r, lambda x: x["redundancy"]), home)
            out[a]["collisions"] = mean_ci(per_episode(r, self.fns["collisions"]), home)
            lines.append(f"| {a} | {len(r)} | {fmt(*out[a]['250'])} | {fmt(*out[a]['500'])} | {fmt(*out[a]['1000'])} | "
                         f"{fmt(*out[a]['redundancy'])} | {fmt(out[a]['collisions'][0], d=1)} |")
        pairs = [(f"X_{m}", f"X_{h}") for m, h in PAIRS.values()] + [(f"X_{self.headline_arms()[0]}", "X_N"),
                 (f"X_{self.headline_arms()[0]}", "X_F")]
        lines += ["", "| a - b | house seen @1000 | home-clustered CI |", "|---|---|---|"]
        for a, b in pairs:
            if a in rows and b in rows:
                d = paired_diff(per_episode(rows[a], lambda x: x["house_seen"]["1000"]),
                                per_episode(rows[b], lambda x: x["house_seen"]["1000"]), home)
                out[f"{a}-{b}"] = d
                lines.append(f"| {a} - {b} | {d['diff']:+.3f} | [{d['home_ci'][0]:+.3f}, {d['home_ci'][1]:+.3f}] |")
        self.md["explore"] = lines
        return out

    def recogniser_table(self, main: Dict[str, Any]) -> Dict[str, Any]:
        hl = self.headline_arms()[0]
        exam_dir, exam_set, _ = self.stage("exam")
        qwen_dir, qwen_set, (qwen_arm,) = self.stage("qwen")
        eps = self.episodes(exam_set)
        home = {e["episode_id"]: e["home"] for e in eps}
        out: Dict[str, Any] = {}
        lines = ["## Recogniser driving test", ""]
        arms = main["arms"]
        if hl in arms and "MT" in arms:
            d = paired_diff(per_episode(read_rows(self.base, exam_dir, hl), self.fns["oracle_success"]),
                            per_episode(read_rows(self.base, exam_dir, "MT"), self.fns["oracle_success"]), home)
            dr = paired_diff(per_episode(read_rows(self.base, exam_dir, hl), self.fns["rule_success"]),
                             per_episode(read_rows(self.base, exam_dir, "MT"), self.fns["rule_success"]), home)
            out["yoloe_minus_truth"] = {"oracle": d, "rule": dr}
            lines += [f"YOLOE ({hl}) - truth arm (MT), all exam episodes: oracle {d['diff']:+.3f} "
                      f"[{d['home_ci'][0]:+.3f}, {d['home_ci'][1]:+.3f}]; rule {dr['diff']:+.3f} "
                      f"[{dr['home_ci'][0]:+.3f}, {dr['home_ci'][1]:+.3f}].", ""]
        q = read_rows(self.base, qwen_dir, qwen_arm)
        if q:
            sub = {e["episode_id"] for e in self.episodes(qwen_set)}
            y0 = [r for r in read_rows(self.base, exam_dir, hl) if r["seed_index"] == 0 and r["episode_id"] in sub]
            dq = paired_diff(per_episode(q, self.fns["oracle_success"]), per_episode(y0, self.fns["oracle_success"]), home)
            dqr = paired_diff(per_episode(q, self.fns["rule_success"]), per_episode(y0, self.fns["rule_success"]), home)
            out["qwen_minus_yoloe_subset_seed0"] = {"oracle": dq, "rule": dqr}
            lines += [f"Qwen3-VL (MR1) - YOLOE ({hl}), the {len(sub)}-episode subset, seed 0: oracle {dq['diff']:+.3f} "
                      f"[{dq['home_ci'][0]:+.3f}, {dq['home_ci'][1]:+.3f}]; rule {dqr['diff']:+.3f} "
                      f"[{dqr['home_ci'][0]:+.3f}, {dqr['home_ci'][1]:+.3f}] (report only).", ""]
        cats = sorted(PHOTO_F1["yoloe"])
        cat_of = {e["episode_id"]: e["target"]["category"] for e in eps}
        pts = []
        lines += ["| recogniser | category | photo-test F1 | driving rule success | driving oracle success |",
                  "|---|---|---|---|---|"]
        for rec, rows in (("yoloe", read_rows(self.base, exam_dir, hl)), ("qwen3vl", q)):
            for c in cats:
                sel = [r for r in rows if cat_of.get(r["episode_id"]) == c]
                if not sel:
                    continue
                rs = float(np.mean([r["rule"][self.tau]["success"] for r in sel]))
                os_ = float(np.mean([r["oracle_success"] for r in sel]))
                pts.append((PHOTO_F1[rec][c], rs))
                lines.append(f"| {rec} | {c} | {PHOTO_F1[rec][c]:.2f} | {rs:.3f} ({len(sel)} runs) | {os_:.3f} |")
        if len(pts) >= 3:
            x, y = np.array(pts).T
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
            out["f1_vs_rule_success_pearson"] = {"r": r, "points": len(pts)}
            lines += ["", f"Descriptive only: Pearson r = {r:+.3f} between photo-test F1 and driving rule-stop success over "
                      f"the {len(pts)} recogniser x category points in the table above (YOLOE: all exam runs of the headline "
                      f"arm; Qwen3-VL: the 60-episode subset, 1 seed, 5-18 runs per category). Not a test: the points are "
                      f"not independent and the Qwen3-VL points rest on few runs."]
        self.md["recogniser"] = lines
        return out

    # --- extra tables before the gate (Naz, 10-25; logs only) -------------------------------------------------------
    def hidden_pairs_table(self) -> Dict[str, Any]:
        """(a) maps-hidden arms and NoMaD paired, oracle stop: all seeds (per-episode means) and seed 0 only (M2h has
        one seed)."""
        exam_dir, exam_set, _ = self.stage("exam")
        home = {e["episode_id"]: e["home"] for e in self.episodes(exam_set)}
        rows = {a: read_rows(self.base, exam_dir, a) for a in ("M21h", "M2h", "N")}
        out: Dict[str, Any] = {}
        lines = ["## (a) Maps hidden vs NoMaD, paired (main exam, oracle stop)", "",
                 "M2h ran 1 seed (seed 0; not the headline pair), M21h and N 3 seeds: the seed-0 rows compare the same "
                 "runs of every arm.", "", "| a - b | runs used | episodes | oracle success | home-clustered 95% CI | "
                 "episode-level CI |", "|---|---|---|---|---|---|"]
        for a, b in (("M21h", "M2h"), ("M21h", "N"), ("M2h", "N")):
            for label, keep in (("all seeds", None), ("seed 0", 0)):
                ra = [r for r in rows[a] if keep is None or r["seed_index"] == keep]
                rb = [r for r in rows[b] if keep is None or r["seed_index"] == keep]
                d = paired_diff(per_episode(ra, self.fns["oracle_success"]), per_episode(rb, self.fns["oracle_success"]),
                                home)
                out[f"{a}-{b} {label}"] = d
                lines.append(f"| {a} - {b} | {label} | {d['n']} | {d['diff']:+.3f} | [{d['home_ci'][0]:+.3f}, "
                             f"{d['home_ci'][1]:+.3f}] | [{d['episode_ci'][0]:+.3f}, {d['episode_ci'][1]:+.3f}] |")
        self.md["hidden_pairs"] = lines
        return out

    def rule_breakdown_tables(self, arms: Sequence[str]) -> Dict[str, Any]:
        """(b) rule-stop breakdown, (c) stuck share of the failures, (d) failed rule stops by walking distance."""
        out: Dict[str, Any] = {}
        b = ["## (b) Rule-stop breakdown (main exam, tau " + self.tau + ")", "",
             "Shares of the rule stops; 'before first sight' = the stop came before simulator truth first saw an "
             "instance; 'never seen' = truth never saw one in the run.", "",
             "| arm | runs | runs with a rule stop | STOP precision | wrong-object | near-miss | before first sight | "
             "never seen | stop - first sight (median steps, stops after sight) |", "|---|---|---|---|---|---|---|---|---|"]
        c = ["## (c) Failures with a stuck streak (main exam, oracle stop)", "",
             "| arm | failed runs | with a stuck streak (>= 100 collision steps) | share |", "|---|---|---|---|"]
        bins = [(0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, float("inf"))]
        d = ["## (d) Failed rule stops: walking distance from the stop pose to the nearest view point (main exam)", "",
             "| arm | failed rule stops | 0-0.2 m | 0.2-0.5 m | 0.5-1.0 m | > 1.0 m | unreachable |",
             "|---|---|---|---|---|---|---|"]
        f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        exam_dir = self.plan.where("exam")[0]
        for a in arms:
            rows = read_rows(self.base, exam_dir, a)
            if not rows:
                continue
            stops = [r["rule"][self.tau] for r in rows if r.get("rule") and r["rule"][self.tau]["stopped"]]
            n = len(stops)
            share = (lambda k: sum(bool(s[k]) for s in stops) / n if n else None)
            before = [s for s in stops if s["stop_minus_sight"] is not None and s["stop_minus_sight"] < 0]
            never = [s for s in stops if s["stop_minus_sight"] is None]
            after = [s["stop_minus_sight"] for s in stops if s["stop_minus_sight"] is not None and s["stop_minus_sight"] >= 0]
            out.setdefault("rule_breakdown", {})[a] = {
                "runs": len(rows), "rule_stops": n, "stop_precision": share("success"),
                "wrong_object": share("wrong_object"), "near_miss": share("near_miss"),
                "before_first_sight": len(before) / n if n else None, "never_seen": len(never) / n if n else None,
                "stop_minus_sight_median": median_or_none(after)}
            o = out["rule_breakdown"][a]
            b.append(f"| {a} | {len(rows)} | {n} ({n / len(rows):.3f}) | {f(o['stop_precision'])} | {f(o['wrong_object'])} | "
                     f"{f(o['near_miss'])} | {f(o['before_first_sight'])} | {f(o['never_seen'])} | "
                     f"{fmt(o['stop_minus_sight_median'], d=0)} |")
            failed = [r for r in rows if not r["oracle_success"]]
            stuck = sum(r["stuck_streaks"] > 0 for r in failed)
            out.setdefault("stuck_failures", {})[a] = {"failed": len(failed), "stuck": stuck,
                                                       "share": stuck / len(failed) if failed else None}
            c.append(f"| {a} | {len(failed)} | {stuck} | {f(stuck / len(failed) if failed else None)} |")
            fs = [s for s in stops if not s["success"]]
            counts = [sum(s["vp_at_stop"] is not None and lo <= s["vp_at_stop"] < hi for s in fs) for lo, hi in bins]
            unreach = sum(s["vp_at_stop"] is None for s in fs)
            out.setdefault("failed_stop_distance", {})[a] = {"failed_stops": len(fs), "bins_m": [list(x) for x in bins],
                                                             "counts": counts, "unreachable": unreach}
            pct = (lambda k: f"{k} ({k / len(fs):.2f})" if fs else "0")
            d.append(f"| {a} | {len(fs)} | " + " | ".join(pct(k) for k in counts) + f" | {unreach} |")
        self.md["rule_breakdown"], self.md["stuck_failures"], self.md["failed_stop_distance"] = b, c, d
        return out

    def spl_table(self, groups: Sequence[Tuple[str, str, Sequence[str]]]) -> Dict[str, Any]:
        """(e) SPL of every arm, both stops, with home-clustered CIs."""
        out: Dict[str, Any] = {}
        lines = ["## (e) SPL, both stops (per-episode means over seeds; home-clustered 95% CI)", "",
                 "| set | arm | oracle SPL | rule SPL |", "|---|---|---|---|"]
        for subdir, set_name, arms in groups:
            home = {e["episode_id"]: e["home"] for e in self.episodes(set_name)}
            for a in arms:
                rows = read_rows(self.base, subdir, a)
                if not rows:
                    continue
                o = mean_ci(per_episode(rows, self.fns["oracle_spl"]), home)
                r = mean_ci(per_episode(rows, self.fns["rule_spl"]), home)
                out[f"{set_name}/{a}"] = {"oracle_spl": o, "rule_spl": r}
                lines.append(f"| {set_name} | {a} | {fmt(*o)} | {fmt(*r)} |")
        self.md["spl"] = lines
        return out

    # --- all -------------------------------------------------------------------------------------------------------
    def build(self) -> Dict[str, Any]:
        hl_maps, hl_hidden = self.headline_arms()
        exam_dir, exam_set, exam_arms = self.stage("exam")
        main = self.main_table("main", exam_dir, exam_set, exam_arms, "Main exam (oracle and rule stop)")
        self.data["main"] = main
        self.phases_table("phases", main)
        om, oh = self.plan.other_pair()
        pairs = [(hl_maps, hl_hidden, True), (om, oh, False), ("M21", "M2", False), (hl_maps, "N", False),
                 (hl_maps, "V1", False), (hl_maps, "F", False), (hl_maps, "MT", False), (hl_hidden, "N", False)]
        self.data["paired"] = self.paired_table("paired", exam_dir, exam_set, pairs)
        self.data["bins"] = self.split_table("bins", exam_dir, exam_set, exam_arms, "distance_bin",
                                             "Per distance bin (3-6 / 6-10 / 10-15 m)",
                                             label=lambda b: ["3-6 m", "6-10 m", "10-15 m"][int(b)])
        self.data["categories"] = self.split_table("categories", exam_dir, exam_set, exam_arms, "category", "Per category")
        self.data["homes"] = self.split_table("homes", exam_dir, exam_set, [hl_maps, hl_hidden, "N", "F"], "home", "Per home")
        ovon_dir, ovon_set, ovon_arms = self.stage("ovon")
        self.data["ovon"] = self.split_table("ovon", ovon_dir, ovon_set, ovon_arms, "type", "OVON per split")
        self.data["ovon_main"] = self.main_table("ovon_main", ovon_dir, ovon_set, ovon_arms, "OVON (all splits)")
        self.data["ovon_paired"] = self.paired_table("ovon_paired", ovon_dir, ovon_set, [(hl_maps, hl_hidden, False),
                                                                                         (hl_maps, "N", False)])
        self.data["explore"] = self.explore_table()
        uns_dir, uns_set, uns_arms = self.stage("unseeable")
        self.data["unseeable"] = self.main_table("unseeable", uns_dir, uns_set, uns_arms,
                                                 "Unseeable set (dropped only by the start filter; oracle stop, 1 seed)")
        self.data["recogniser"] = self.recogniser_table(main)
        self.data["hidden_pairs"] = self.hidden_pairs_table()
        self.data.update(self.rule_breakdown_tables(exam_arms))
        self.data["spl"] = self.spl_table([self.stage(s) for s in ("exam", "ovon", "unseeable", "qwen")])
        return self.data

    def write(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        order = ["main", "paired", "phases", "bins", "categories", "homes", "ovon_main", "ovon", "ovon_paired",
                 "explore", "unseeable", "recogniser", "hidden_pairs", "rule_breakdown", "stuck_failures",
                 "failed_stop_distance", "spl"]
        text = [f"# Phase 12 tables (tau {self.tau}, headline {self.headline})", ""]
        for k in order:
            if k in self.md:
                text += self.md[k] + [""]
                write_text_atomic(out / f"{k}.md", "\n".join(self.md[k]) + "\n")
        write_text_atomic(out / "tables.md", "\n".join(text) + "\n")
        write_text_atomic(out / "tables.json", json.dumps(self.data, indent=1, sort_keys=True, default=float) + "\n")


def g12_draft(t: Tables, out_file: Path, fairness: Dict[str, Any], determinism: Dict[str, Any]) -> None:
    hl = t.data["paired"].get("-".join(t.headline_arms()), {}).get("oracle_success", {})
    row1 = "missing" if not hl.get("n") else (
        f"{'PASS' if hl['diff'] > 0 and hl['home_p_one_sided'] < 0.05 else 'FAIL'}: {t.headline_arms()[0]} - "
        f"{t.headline_arms()[1]} = {hl['diff']:+.3f} [{hl['home_ci'][0]:+.3f}, {hl['home_ci'][1]:+.3f}], one-sided p "
        f"{hl['home_p_one_sided']:.4f} ({hl['n']} episodes, {hl['homes']} homes)")
    lines = ["# G12 draft (written by the chain; Naz reviews row 5)", "",
             f"Headline checkpoint: {t.headline} (P12_DEV_CHOICES.json); tau {t.tau}.", "",
             "| # | Check | Pass | Result |", "|---|---|---|---|",
             f"| 1 | Headline | M2 - M2-h > 0, oracle stop, home-clustered one-sided p < 0.05 | {row1} |",
             "| 2 | Reported | M2 vs N, V1, F; M21 vs M2; rule stop; OVON; recogniser; exploration; phases | see tables/ |",
             f"| 3 | Fairness | frozen fingerprints, sha256 every run, no run missing | "
             f"{'PASS' if fairness.get('pass') else 'FAIL'}: {fairness.get('summary', '-')} |",
             f"| 4 | Determinism | one M2 episode twice, same seed, same GPU -> identical log | "
             f"{'PASS' if determinism.get('pass') else 'FAIL / not run'}: {determinism.get('summary', '-')} |",
             "| 5 | Videos | legible, captioned, reproducible from logs; reviewed by Naz | pending Naz |"]
    write_text_atomic(out_file, "\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", default="tables", help="below out_dir")
    args = p.parse_args(argv)
    layout = RunLayout.load(args.config)
    t = Tables(layout)
    t.build()
    t.write(layout.base / args.out)
    fair = json.loads((layout.base / "checks" / "fairness.json").read_text()) if (layout.base / "checks" / "fairness.json").exists() else {}
    det = json.loads((layout.base / "checks" / "determinism.json").read_text()) if (layout.base / "checks" / "determinism.json").exists() else {}
    g12_draft(t, layout.base / "G12_table_draft.md", fair, det)
    print("\n".join(t.md.get("main", []) + [""] + t.md.get("paired", [])))


if __name__ == "__main__":
    main()
