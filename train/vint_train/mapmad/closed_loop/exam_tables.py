"""Phase 5 tables from the run logs (inside naz_mapmad; numbers regenerable from the logs alone):

    python -m vint_train.mapmad.closed_loop.exam_tables --config /app/visualnav-transformer/mapmad/configs/p5_exam.yaml \
        --subdir runs --out tables

Writes <out_dir>/<out>/: tables.md, runs.csv (one row per run), episodes.csv (seed means), paired.json, and with the
exam arms the G5 rows (gate.json). Statistics: exam_stats.py (home-clustered bootstrap primary, episode-level
reported, 10,000 resamples, seed `--boot-seed`). Failure types: exam_analysis.py.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from mapmad_sim.episodes import load_episodes
from mapmad_sim.run_layout import RunLayout
from vint_train.mapmad.closed_loop import exam_analysis as A
from vint_train.mapmad.closed_loop import exam_stats as S

LABELS = {"success": "SR", "success@500": "SR@500", "spl": "SPL", "target_point_reached": "target pt <= 1 m",
          "any_side_reached": "any side <= 1 m", "collisions": "collisions/run", "stuck_streaks": "stuck streaks/run",
          "final_vp_geodesic_m": "final dist m"}
TABLE_METRICS = ("success", "spl", "success@500", "target_point_reached", "any_side_reached", "collisions",
                 "stuck_streaks", "final_vp_geodesic_m")


def fmt(v: Sequence[float], digits: int = 2) -> str:
    return f"{v[0]:.{digits}f} [{v[1]:.{digits}f}, {v[2]:.{digits}f}]"


def arm_ci(per_ep: Dict[str, Dict[str, float]], metric: str, home_of: Dict[str, str], seed: int) -> List[float]:
    ids = [i for i in sorted(per_ep) if np.isfinite(per_ep[i][metric])]
    m = S.cluster_bootstrap(np.array([per_ep[i][metric] for i in ids]), [home_of[i] for i in ids], seed=seed)
    return [m[0], m[1], m[2]]


def group_rates(per_ep: Dict[str, Dict[str, float]], key_of: Dict[str, Any]) -> Dict[Any, List[float]]:
    """group -> [success rate, n episodes] (episode = seed mean)."""
    out: Dict[Any, List[float]] = {}
    for g in sorted(set(key_of.values()), key=str):
        ids = [i for i in per_ep if key_of[i] == g]
        if ids:
            out[g] = [float(np.mean([per_ep[i]["success"] for i in ids])), len(ids)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="runs")
    p.add_argument("--out", default="tables")
    p.add_argument("--arms", nargs="*", help="default: the config's arms that have logs")
    p.add_argument("--paired", nargs="*", default=None, help="pairs a-b, e.g. C-B C-D (default: config `paired`)")
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--title", default="Phase 5")
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    cfg = layout.cfg
    episodes, fp = load_episodes(layout.episodes_file)
    by_id = {e["episode_id"]: e for e in episodes}
    arms = args.arms or [a for a in cfg["arms"] if (layout.root / "logs" / a).exists()]
    success_at = list(cfg["run"].get("report_success_at", []))
    rows, crashed = A.load_runs(layout, arms, by_id, int(cfg["run"]["stuck_steps"]), success_at)
    metrics = list(A.RUN_METRICS)
    per = A.episode_means(rows, metrics)
    home_of = {e["episode_id"]: e["home"] for e in episodes}
    bin_of = {e["episode_id"]: e["distance_bin"] for e in episodes}
    cat_of = {e["episode_id"]: e["target"]["category"] for e in episodes}
    bins = cfg["set_rules"]["bins_m"]
    out = layout.base / args.out
    out.mkdir(parents=True, exist_ok=True)

    lines = [f"# {args.title}: closed-loop results", "",
             f"Episode set fingerprint `{fp}` ({len(episodes)} episodes); runs from `{layout.root}`. Primary success = "
             f"walking distance <= {cfg['run']['success_m']} m to the nearest usable view point of the target (oracle "
             f"stop, {cfg['run']['max_steps']} steps). Per episode = mean over its seed runs; mean [95% CI, home-"
             f"clustered bootstrap, {S.N_BOOT:,} resamples]. Crashed episodes dropped from all arms: {crashed or 'none'}.",
             "", "| arm | episodes | runs | " + " | ".join(LABELS[m] for m in TABLE_METRICS) + " |",
             "|---|---|---|" + "---|" * len(TABLE_METRICS)]
    for arm in arms:
        if not per.get(arm):
            continue
        n_runs = sum(r["arm"] == arm for r in rows)
        cells = [fmt(arm_ci(per[arm], m, home_of, args.boot_seed), 1 if m in ("collisions", "final_vp_geodesic_m") else 2)
                 for m in TABLE_METRICS]
        lines.append(f"| {arm} | {len(per[arm])} | {n_runs} | " + " | ".join(cells) + " |")

    pairs = [tuple(x.split("-")) for x in args.paired] if args.paired is not None else [tuple(x) for x in cfg["paired"]]
    pairs = [(a, b) for a, b in pairs if per.get(a) and per.get(b)]
    paired: Dict[str, Any] = {}
    lines += ["", "## Paired differences (a - b, same episodes and seeds)", "",
              "| a - b | n | success: diff [home CI] | home p (one-sided) | episode CI | SPL diff [home CI] |",
              "|---|---|---|---|---|---|"]
    for a, b in pairs:
        res = {m: S.paired({i: v[m] for i, v in per[a].items()}, {i: v[m] for i, v in per[b].items()}, home_of,
                           seed=args.boot_seed) for m in ("success", "spl", "collisions", "stuck_streaks")}
        paired[f"{a}-{b}"] = res
        s, q = res["success"], res["spl"]
        lines.append(f"| {a} - {b} | {s['n']} | {s['diff']:+.3f} [{s['home_ci'][0]:+.3f}, {s['home_ci'][1]:+.3f}] | "
                     f"{s['home_p_one_sided']:.4f} | [{s['episode_ci'][0]:+.3f}, {s['episode_ci'][1]:+.3f}] | "
                     f"{q['diff']:+.3f} [{q['home_ci'][0]:+.3f}, {q['home_ci'][1]:+.3f}] |")

    for title, key_of, names in (("distance bin", bin_of, {i: f"{bins[i]:g}-{bins[i + 1]:g} m" for i in range(len(bins) - 1)}),
                                 ("category", cat_of, None), ("home", home_of, None)):
        groups = sorted(set(key_of.values()), key=str)
        lines += ["", f"## Success per {title} (rate, episodes)", "",
                  "| arm | " + " | ".join((names or {}).get(g, str(g)[:5] if title == "home" else str(g)) for g in groups) + " |",
                  "|---|" + "---|" * len(groups)]
        for arm in arms:
            if per.get(arm):
                r = group_rates(per[arm], key_of)
                lines.append(f"| {arm} | " + " | ".join(f"{r[g][0]:.2f} ({int(r[g][1])})" if g in r else "-" for g in groups) + " |")

    lines += ["", "## Failure types (runs)", "", "| arm | " + " | ".join(A.FAILURE_TYPES) + " |",
              "|---|" + "---|" * len(A.FAILURE_TYPES)]
    for arm in arms:
        c = A.failure_counts(rows, arm)
        if sum(c.values()):
            lines.append(f"| {arm} | " + " | ".join(str(c[k]) for k in A.FAILURE_TYPES) + " |")
    lines += ["", "stuck = a streak of >= 100 collision steps; wrong_instance = within 1 m of another instance of the "
              "category; circling = > 50% of 0.25 m cell entries revisit a cell; timeout_moving = the rest (first match "
              "in this order)."]

    (out / "tables.md").write_text("\n".join(lines) + "\n")
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "episode_id", "home", "category", "distance_bin"] + metrics + ["runs"])
        for arm in arms:
            for i, v in sorted(per.get(arm, {}).items()):
                w.writerow([arm, i, home_of[i], cat_of[i], bin_of[i]] + [v[m] for m in metrics] + [v["runs"]])
    A.write_json(out / "paired.json", {"fingerprint": fp, "crashed": crashed, "paired": paired})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
