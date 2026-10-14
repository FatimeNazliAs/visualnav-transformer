"""Phase 5 ablation table: ablation arms (seed index 0) against the exam's arm C (inside naz_mapmad):

    python -m vint_train.mapmad.closed_loop.exam_ablation \
        --config /app/visualnav-transformer/mapmad/configs/p5_ablation_heat.yaml

Rows: the exam's C over all its seed runs (the exam table's number), the exam's C at seed index 0 (same noise seeds
as the ablation runs), and each ablation arm. Metrics: success, SPL, stuck (share of runs with a >= 100-step
collision streak), stuck streaks per run; mean [95% CI, home-clustered bootstrap, 10,000 resamples]. Paired a - b on
the same episodes: ablation - C seed 0 (same noise seed; only the heat differs) and ablation - C (seed means).
Also paired a - C seed 0 per distance bin. Report only (no pass/fail). Writes <out_dir>/<out>/ablation.md and ablation.json.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from mapmad_sim.episodes import load_episodes
from mapmad_sim.run_layout import RunLayout
from vint_train.mapmad.closed_loop import exam_analysis as A
from vint_train.mapmad.closed_loop import exam_stats as S
from vint_train.mapmad.closed_loop.exam_tables import arm_ci, fmt, group_rates

METRICS = ("success", "spl", "stuck", "stuck_streaks")
LABELS = {"success": "SR", "spl": "SPL", "stuck": "stuck (share of runs)", "stuck_streaks": "stuck streaks/run"}


def with_stuck(rows: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """Rows relabelled with `label` as arm and a 0/1 `stuck` field (the run had a stuck streak)."""
    return [dict(r, arm=label, stuck=float(r["stuck_streaks"] > 0)) for r in rows]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="ablation_heat/runs")
    p.add_argument("--exam-subdir", default="runs")
    p.add_argument("--base", default="C", help="the exam arm compared against")
    p.add_argument("--arms", nargs="+", default=["C_h05", "C_h10"])
    p.add_argument("--out", default="ablation_heat/tables")
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--title", default="ablation 1: perturbed heat")
    args = p.parse_args()
    exam = RunLayout.load(args.config, subdir=args.exam_subdir)
    abl = RunLayout.load(args.config, subdir=args.subdir)
    cfg = exam.cfg
    episodes, fp = load_episodes(exam.episodes_file)
    by_id = {e["episode_id"]: e for e in episodes}
    home_of = {i: e["home"] for i, e in by_id.items()}
    bin_of = {i: e["distance_bin"] for i, e in by_id.items()}
    stuck_steps, at = int(cfg["run"]["stuck_steps"]), list(cfg["run"].get("report_success_at", []))
    base_rows, crashed_exam = A.load_runs(exam, [args.base], by_id, stuck_steps, at)
    abl_rows, crashed_abl = A.load_runs(abl, args.arms, by_id, stuck_steps, at)
    crashed = sorted(set(crashed_exam) | set(crashed_abl))
    all_seeds, seed0 = f"{args.base} (exam, all seeds)", f"{args.base} (exam, seed 0)"
    rows = with_stuck(base_rows, all_seeds) + with_stuck([r for r in base_rows if r["seed_index"] == 0], seed0)
    for arm in args.arms:
        rows += with_stuck([r for r in abl_rows if r["arm"] == arm], arm)
    rows = [r for r in rows if r["episode_id"] not in crashed]
    per = A.episode_means(rows, METRICS)
    names = [all_seeds, seed0] + args.arms

    lines = [f"# Phase 5 {args.title} (report only)", "",
             f"Exam set `{fp}` ({len(episodes)} episodes). Ablation runs from `{abl.root}`, seed index 0; the exam's "
             f"arm {args.base} from `{exam.root}`. Per episode = mean over its seed runs; mean [95% CI, home-clustered "
             f"bootstrap, {S.N_BOOT:,} resamples, seed {args.boot_seed}]. Crashed episodes dropped from all rows: "
             f"{crashed or 'none'}.", "",
             "| arm | episodes | runs | " + " | ".join(LABELS[m] for m in METRICS) + " |",
             "|---|---|---|" + "---|" * len(METRICS)]
    table: Dict[str, Any] = {}
    for name in names:
        if not per.get(name):
            continue
        cis = {m: arm_ci(per[name], m, home_of, args.boot_seed) for m in METRICS}
        table[name] = {"episodes": len(per[name]), "runs": sum(r["arm"] == name for r in rows), **cis}
        lines.append(f"| {name} | {len(per[name])} | {table[name]['runs']} | " +
                     " | ".join(fmt(cis[m]) for m in METRICS) + " |")

    paired: Dict[str, Any] = {}
    lines += ["", "## Paired differences (a - b, same episodes)", "",
              "| a - b | n | homes | " + " | ".join(f"{LABELS[m]}: diff [home CI]" for m in METRICS) + " |",
              "|---|---|---|" + "---|" * len(METRICS)]
    for arm in args.arms:
        for b in (seed0, all_seeds):
            if not (per.get(arm) and per.get(b)):
                continue
            res = {m: S.paired({i: v[m] for i, v in per[arm].items()}, {i: v[m] for i, v in per[b].items()}, home_of,
                               seed=args.boot_seed) for m in METRICS}
            paired[f"{arm} - {b}"] = res
            s = res["success"]
            lines.append(f"| {arm} - {b} | {s['n']} | {s['homes']} | " + " | ".join(
                f"{res[m]['diff']:+.3f} [{res[m]['home_ci'][0]:+.3f}, {res[m]['home_ci'][1]:+.3f}]" for m in METRICS) + " |")

    bins = cfg["set_rules"]["bins_m"]
    groups = sorted(set(bin_of.values()))
    lines += ["", "## Success per distance bin (rate, episodes)", "",
              "| arm | " + " | ".join(f"{bins[g]:g}-{bins[g + 1]:g} m" for g in groups) + " |", "|---|" + "---|" * len(groups)]
    for name in names:
        if per.get(name):
            r = group_rates(per[name], bin_of)
            lines.append(f"| {name} | " + " | ".join(f"{r[g][0]:.2f} ({int(r[g][1])})" if g in r else "-" for g in groups) + " |")
    by_bin: Dict[str, Any] = {}
    lines += ["", f"## Paired differences per distance bin (a - {seed0}, home CI)", "",
              "| a | bin | n | homes | " + " | ".join(LABELS[m] for m in ("success", "spl", "stuck")) + " |",
              "|---|---|---|---|---|---|---|"]
    for arm in args.arms:
        for g in groups:
            if not (per.get(arm) and per.get(seed0)):
                continue
            ids = {i for i in per[arm] if bin_of[i] == g}
            res = {m: S.paired({i: per[arm][i][m] for i in ids}, {i: v[m] for i, v in per[seed0].items() if i in ids},
                               home_of, seed=args.boot_seed) for m in ("success", "spl", "stuck")}
            name = f"{bins[g]:g}-{bins[g + 1]:g} m"
            by_bin[f"{arm} - {seed0} | {name}"] = res
            lines.append(f"| {arm} | {name} | {res['success']['n']} | {res['success']['homes']} | " + " | ".join(
                f"{res[m]['diff']:+.3f} [{res[m]['home_ci'][0]:+.3f}, {res[m]['home_ci'][1]:+.3f}]"
                for m in ("success", "spl", "stuck")) + " |")
    lines += ["", "## Failure types (runs)", "", "| arm | " + " | ".join(A.FAILURE_TYPES) + " |",
              "|---|" + "---|" * len(A.FAILURE_TYPES)]
    for name in names:
        c = A.failure_counts(rows, name)
        if sum(c.values()):
            lines.append(f"| {name} | " + " | ".join(str(c[k]) for k in A.FAILURE_TYPES) + " |")

    out = abl.base / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.md").write_text("\n".join(lines) + "\n")
    A.write_json(out / "ablation.json", {"fingerprint": fp, "crashed": crashed, "arms": table, "paired": paired,
                                         "paired_by_bin": by_bin})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
