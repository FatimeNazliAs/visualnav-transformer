"""Phase 1 tables from the per-episode summaries (inside naz_mapmad):

    python -m vint_train.mapmad.closed_loop.tables            # writes <outputs>/p1_baseline/tables.md + CSVs
    python -m vint_train.mapmad.closed_loop.tables --config A.yaml --combine B.yaml   # pooled: <B's folder>/pooled.md

--combine pools the episode summaries of several runs (distinct episode ids) per arm, for the arms every run has,
and writes only the per-arm and paired tables (pooled.md, pooled.csv, pooled_paired.csv in the last run's folder).

Per arm: success rate (and success@500 read from the same runs), SPL, collisions per episode, share of steps with
a collision, path length, final geodesic distance, each as mean [95% bootstrap CI] (10,000 resamples, seed 0).
Paired differences (config `paired`, same episodes): success, success@500, SPL.
Diagnostics from the step logs (analysis.py): stuck streaks (>= 100 consecutive collision steps), failure types,
success per category and per home. SECONDARY metrics and the black-input count (if <out>/anyside and <out>/black
exist, from mapmad/scripts/p1_replay.py anyside / black): see secondary_table and open_issues.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from mapmad_sim.run_layout import P1_CONFIG, RunLayout
from vint_train.mapmad.closed_loop import analysis, metrics

BLACK_INPUT_SHARE = 0.30
LABELS = {"success": "SR", "spl": "SPL", "collisions": "collisions/ep", "collision_share": "collision share",
          "path_length_m": "path m", "final_geodesic_m": "final geo m"}


def load_rows(layout: RunLayout, arm: str, extra_steps: List[int]) -> List[Dict[str, Any]]:
    rows = [json.loads(p.read_text()) for p in sorted(layout.summary(arm, "*").parent.glob("*.json"))]
    for r in rows:
        for n in extra_steps:
            r[f"success@{n}"] = metrics.success_at(r, n)
    return rows


def fmt(ci, digits: int = 2) -> str:
    mean, lo, hi = ci
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def diagnostics_tables(layout: RunLayout) -> List[str]:
    """Markdown for stuck streaks, failure types, per-category / per-home success and the SECONDARY metric."""
    p1 = layout.cfg
    facts = {arm: [analysis.episode_facts(p) for p in layout.logs(arm)] for arm in p1["arms"]}
    lines = ["", f"## Stuck streaks (>= {analysis.STUCK_STEPS} consecutive collision steps) and failure types", "",
             "A collision step = the move came up > 1 cm short (pressing against or sliding along an obstacle).", "",
             "| arm | episodes stuck | median start step | " + " | ".join(analysis.FAILURE_TYPES) + " |",
             "|---|---|---|" + "---|" * len(analysis.FAILURE_TYPES)]
    for arm, rows in facts.items():
        if not rows:
            continue
        starts = [r["stuck_start"] for r in rows if r["stuck_start"] is not None]
        kinds = Counter(r["failure_type"] for r in rows)
        lines.append(f"| {arm} | {len(starts)}/{len(rows)} ({100 * len(starts) / len(rows):.0f}%) | "
                     f"{statistics.median(starts) if starts else '-'} | " + " | ".join(str(kinds[k]) for k in analysis.FAILURE_TYPES) + " |")
    lines.append(f"\nfail_near = failed, not stuck, got within {analysis.NEAR_M} m (geodesic); fail_far = the rest.")
    for key, title in (("category", "category"), ("home", "home")):
        groups = sorted({r[key] for rows in facts.values() for r in rows})
        lines += ["", f"## Success per {title} (successes / episodes)", "",
                  f"| arm | " + " | ".join(g[:5] if key == "home" else g for g in groups) + " |", "|---|" + "---|" * len(groups)]
        for arm, rows in facts.items():
            cells = []
            for g in groups:
                sel = [r for r in rows if r[key] == g]
                cells.append(f"{int(sum(r['success'] for r in sel))}/{len(sel)}" if sel else "-")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += secondary_table(layout)
    lines += open_issues(layout)
    return lines


def secondary_table(layout: RunLayout) -> List[str]:
    """SECONDARY success metrics next to the primary one, all arms (from p1_replay.py anyside)."""
    if not (layout.root / "anyside").exists():
        return []
    lines = ["", "## SECONDARY success metrics (not the primary metric)", "",
             "Primary (unchanged): geodesic <= 1.0 m to the target point. SECONDARY, at any logged step: any-side = "
             "geodesic <= 1.0 m to the nearest ObjectNav view point of the target instance; view point <= 0.2 m = "
             "geodesic <= 0.2 m to it (view points snapped onto the LIMO floor map). Both SECONDARY columns are lower "
             "bounds: an episode stopped at its primary success.", "",
             "| arm | n | primary SR | any-side SR (<= 1.0 m) | view point <= 0.2 m | primary fail, any-side success |",
             "|---|---|---|---|---|---|"]
    for arm in layout.cfg["arms"]:
        rows = [json.loads(p.read_text()) for p in sorted(layout.anyside(arm, "*").parent.glob("*.json"))]
        if rows:
            ci = {k: fmt(metrics.bootstrap_ci([float(r[k]) for r in rows]))
                  for k in ("primary_success", "anyside_success", "view_point_02_success")}
            lines.append(f"| {arm} | {len(rows)} | {ci['primary_success']} | {ci['anyside_success']} | "
                         f"{ci['view_point_02_success']} | "
                         f"{sum(r['anyside_success'] and not r['primary_success'] for r in rows)} |")
    return lines


def open_issues(layout: RunLayout) -> List[str]:
    """Open issues / Phase 2 notes that come with numbers."""
    lines = ["", "## Open issues / Phase 2 notes", "",
             "- Decide the Phase 5 success metric (primary vs the SECONDARY view-point metrics above)."]
    black_dir = layout.root / "black"
    if black_dir.exists():
        lines += [f"- Stuck streaks whose NoMaD input (the 4 pictures for the streak's first command) is >= "
                  f"{BLACK_INPUT_SHARE:.0%} near-black (HM3D mesh holes):", "",
                  "| arm | stuck streaks | input >= 30% black | share |", "|---|---|---|---|"]
        for arm in layout.cfg["arms"]:
            streaks = [st for p in sorted((black_dir / arm).glob("*.json")) for st in json.loads(p.read_text())["streaks"]]
            dark = sum(st["input_black_share"] >= BLACK_INPUT_SHARE for st in streaks)
            if streaks:
                lines.append(f"| {arm} | {len(streaks)} | {dark} | {dark / len(streaks):.0%} |")
    return lines


def summary_tables(p1: Dict[str, Any], rows: Dict[str, List[Dict[str, Any]]], extra: List[int], title: str):
    """(markdown lines, per-arm CSV rows, paired CSV rows) of the per-arm and paired tables."""
    cols = ["success"] + [f"success@{n}" for n in extra] + [m for m in metrics.METRICS if m != "success"]
    lines = [title, "",
             f"Timeout {p1['run']['max_steps']} steps, success = geodesic <= {p1['run']['success_m']} m (oracle stop). "
             "Mean [95% bootstrap CI, 10,000 resamples].", "",
             "| arm | goal | episodes | HFOV | n | " + " | ".join(LABELS.get(c, c) for c in cols) + " |",
             "|" + "---|" * (5 + len(cols))]
    table_csv = []
    for arm, spec in p1["arms"].items():
        r = rows[arm]
        if not r:
            continue
        cis = {c: metrics.bootstrap_ci([float(x[c]) for x in r]) for c in cols}
        lines.append(f"| {arm} | {spec['goal']} | {spec['episodes']} | {spec['hfov_deg']:g} | {len(r)} | "
                     + " | ".join(fmt(cis[c]) for c in cols) + " |")
        table_csv += [{"arm": arm, "metric": c, "n": len(r), "mean": cis[c][0], "low": cis[c][1], "high": cis[c][2]}
                      for c in cols]
    lines += ["", "## Paired differences (a - b, same episodes)", "",
              "| a | b | n | " + " | ".join(["SR"] + [f"SR@{n}" for n in extra] + ["SPL"]) + " |", "|---|---|---|---|" + "---|" * (len(extra) + 1)]
    paired_csv = []
    for a, b in p1.get("paired", []):
        keys = ["success"] + [f"success@{n}" for n in extra] + ["spl"]
        d = metrics.paired(rows.get(a, []), rows.get(b, []), keys)
        if d is None:
            continue
        n = len({x["episode_id"] for x in rows[a]} & {x["episode_id"] for x in rows[b]})
        lines.append(f"| {a} | {b} | {n} | " + " | ".join(fmt(d[k]) for k in keys) + " |")
        paired_csv += [{"a": a, "b": b, "metric": k, "n": n, "mean": d[k][0], "low": d[k][1], "high": d[k][2]} for k in keys]
    return lines, table_csv, paired_csv


def write(out: Path, lines: List[str], csvs: Dict[str, List[Dict[str, Any]]], md: str) -> None:
    (out / md).write_text("\n".join(lines) + "\n")
    for name, data in csvs.items():
        if data:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=P1_CONFIG)
    p.add_argument("--subdir", default="")
    p.add_argument("--combine", type=Path, nargs="+", default=[], help="run configs to pool with --config")
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    p1 = layout.cfg
    extra = list(p1["run"].get("report_success_at", []))
    if args.combine:
        layouts = [layout] + [RunLayout.load(c) for c in args.combine]
        per_run = [{arm: load_rows(l, arm, extra) for arm in p1["arms"]} for l in layouts]
        arms = [a for a in p1["arms"] if all(r[a] for r in per_run)]
        rows = {a: [x for r in per_run for x in r[a]] for a in arms}
        ids = [x["episode_id"] for a in arms[:1] for x in rows[a]]
        assert len(ids) == len(set(ids)), "pooled runs share episode ids"
        cfg = dict(p1, arms={a: p1["arms"][a] for a in arms})
        title = "# Pooled: " + " + ".join(l.cfg["out_dir"] for l in layouts)
        lines, table_csv, paired_csv = summary_tables(cfg, rows, extra, title)
        write(layouts[-1].root, lines, {"pooled.csv": table_csv, "pooled_paired.csv": paired_csv}, "pooled.md")
        return
    rows = {arm: load_rows(layout, arm, extra) for arm in p1["arms"]}
    lines, table_csv, paired_csv = summary_tables(p1, rows, extra, "# Phase 1 baseline: normal NoMaD in Habitat")
    lines += diagnostics_tables(layout)
    write(layout.root, lines, {"tables.csv": table_csv, "paired.csv": paired_csv}, "tables.md")


if __name__ == "__main__":
    main()
