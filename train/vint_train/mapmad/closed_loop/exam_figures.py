"""Phase 5 figures from the tables folder (exam_tables.py output; inside naz_mapmad):

    python -m vint_train.mapmad.closed_loop.exam_figures --tables /outputs/mapmad/p5_exam/tables \
        --out /outputs/mapmad/p5_exam/figures

Writes PNG + PDF (light background, for the thesis):
- success_per_arm: primary success per arm, mean with the home-clustered 95% CI (C highlighted);
- success_per_bin: the same per distance bin (one panel per bin);
- paired_differences: a - b success differences with home-clustered CIs (zero line);
- failure_types: share of runs per outcome (success / stuck / wrong instance / circling / timeout while moving).
CIs are recomputed from episodes.csv with exam_stats (same seed as the tables), so the figures need no other input.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vint_train.mapmad.closed_loop import exam_analysis as A  # noqa: E402
from vint_train.mapmad.closed_loop import exam_stats as S  # noqa: E402

ARM_NAMES = {"A": "A  NoMaD explore", "B": "B  NoMaD + photo", "C": "C  MapMaD map goal", "D": "D  MapMaD masked",
             "E": "E  map planner", "F": "F  MapMaD + photo"}
BIN_NAMES = ["3-6 m", "6-10 m", "10-15 m"]
# reference palette (dataviz skill), light mode
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HIGHLIGHT, NEUTRAL = "#2a78d6", "#8a8984"
OUTCOME_COLORS = {"success": "#2a78d6", "stuck": "#eb6834", "wrong_instance": "#6250d6", "circling": "#eda100",
                  "timeout_moving": "#1baf7a"}
OUTCOME_NAMES = {"success": "success", "stuck": "stuck streak", "wrong_instance": "wrong instance",
                 "circling": "circling", "timeout_moving": "timeout while moving"}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def read_episodes(path: Path) -> Dict[str, List[Dict[str, str]]]:
    by_arm: Dict[str, List[Dict[str, str]]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_arm.setdefault(r["arm"], []).append(r)
    return by_arm


def ci(rows: List[Dict[str, str]], metric: str, seed: int) -> List[float]:
    if not rows:
        return [float("nan")] * 3
    v = np.array([float(r[metric]) for r in rows])
    m, lo, hi, _ = S.cluster_bootstrap(v, [r["home"] for r in rows], seed=seed)
    return [m, lo, hi]


def dot_ci(ax, labels: List[str], vals: List[List[float]], colors: List[str], xlabel: str) -> None:
    y = np.arange(len(labels))[::-1]
    for yi, (m, lo, hi), c in zip(y, vals, colors):
        if not np.isfinite(m):
            continue  # no episodes of this arm here
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2, solid_capstyle="round")
        ax.plot([m], [yi], "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(hi + 0.02, yi, f"{m:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    style(ax)


def save(fig, out: Path, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--rename", nargs="*", default=[], help="arm renames old=new, e.g. C_wp1=C (dev tables)")
    args = p.parse_args()
    rename = dict(x.split("=") for x in args.rename)
    args.out.mkdir(parents=True, exist_ok=True)
    eps = {rename.get(k, k): v for k, v in read_episodes(args.tables / "episodes.csv").items()}
    arms = [a for a in ARM_NAMES if a in eps]
    color = [HIGHLIGHT if a == "C" else NEUTRAL for a in arms]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    dot_ci(ax, [ARM_NAMES[a] for a in arms], [ci(eps[a], "success", args.boot_seed) for a in arms], color,
           "primary success (walking distance <= 0.2 m to a view point), 95% CI over homes")
    ax.set_xlim(0, 1.08)
    save(fig, args.out, "success_per_arm")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for b, ax in enumerate(axes):
        vals = [ci([r for r in eps[a] if int(r["distance_bin"]) == b], "success", args.boot_seed) for a in arms]
        n = sum(int(r["distance_bin"]) == b for r in eps[arms[0]])
        dot_ci(ax, [ARM_NAMES[a] for a in arms], vals, color, "success")
        ax.set_title(f"{BIN_NAMES[b]} ({n} episodes)", color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, 1.15)
    save(fig, args.out, "success_per_bin")

    paired = {"-".join(rename.get(x, x) for x in k.split("-")): v
              for k, v in json.loads((args.tables / "paired.json").read_text())["paired"].items()}
    names = list(paired)
    vals = [[paired[k]["success"]["diff"], *paired[k]["success"]["home_ci"]] for k in names]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(names) + 1.2))
    lo = min(v[1] for v in vals) - 0.1
    hi = max(v[2] for v in vals) + 0.15
    ax.axvline(0, color=INK2, linewidth=1)
    dot_ci(ax, [k.replace("-", " - ") + ("  (primary)" if k == "C-B" else "") for k in names], vals,
           [HIGHLIGHT if k == "C-B" else NEUTRAL for k in names], "success difference a - b, 95% CI over homes")
    ax.set_xlim(lo, hi)
    save(fig, args.out, "paired_differences")

    runs: Dict[str, Dict[str, int]] = {}
    with open(args.tables / "runs.csv") as f:
        for r in csv.DictReader(f):
            arm = rename.get(r["arm"], r["arm"])
            runs.setdefault(arm, {}).setdefault(r["failure_type"], 0)
            runs[arm][r["failure_type"]] += 1
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = np.arange(len(arms))[::-1]
    for yi, a in zip(y, arms):
        total, left = sum(runs[a].values()), 0.0
        for k in A.FAILURE_TYPES:
            share = runs[a].get(k, 0) / total
            if share > 0:
                ax.barh(yi, share, left=left, color=OUTCOME_COLORS[k], edgecolor=SURFACE, linewidth=2, height=0.6,
                        label=OUTCOME_NAMES[k] if yi == y[0] else None)
            left += share
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_NAMES[a] for a in arms], color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of runs", color=INK2)
    style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=OUTCOME_COLORS[k]) for k in A.FAILURE_TYPES]
    ax.legend(handles, [OUTCOME_NAMES[k] for k in A.FAILURE_TYPES], ncol=3, fontsize=8, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=INK2)
    save(fig, args.out, "failure_types")
    print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()
