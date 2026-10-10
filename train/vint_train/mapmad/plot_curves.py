"""Phase 4 training curves (confirmation item 6): PNGs from a run's metrics.jsonl step records and its
no_harm_trace.csv.

- `<out>/curves_<label>.png` per run: per-mode diffusion loss, distance loss, map attention share (map-goal
  samples of the batch), lr per parameter group (log scale), each against the global step;
- `<out>/curves_A_vs_B.png`: the same panels for two runs overlaid (map-goal and GoStanford-photo loss), plus the
  in-training checks per epoch: no-harm delta with its CI and the +0.08 selection line, R with its CI.

    python -m vint_train.mapmad.plot_curves --run A=<A folder> --run B=<B folder> --out <dir>

Training-log losses come from the noisy training objective (one random timestep per sample); never compare them
with offline-harness numbers.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vint_train.mapmad.training_checks import read_trace  # noqa: E402

# categorical slots in fixed order (dataviz reference palette, light mode)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODE_ORDER = ["photo", "map_goal", "photo_map", "explore_map", "explore", "gs_photo", "gs_explore"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NO_HARM_LIMIT = 0.08


def read_steps(folder: str) -> List[Dict[str, Any]]:
    """The `{"record": "step"}` lines of a run's metrics.jsonl, in order."""
    with open(os.path.join(folder, "metrics.jsonl")) as f:
        return [r for r in (json.loads(x) for x in f if x.strip()) if r.get("record") == "step"]


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def series(steps: List[Dict[str, Any]], get) -> tuple:
    """(global steps, values) for the records where get(record) is not None."""
    xs, ys = [], []
    for r in steps:
        v = get(r)
        if v is not None:
            xs.append(r["global_step"])
            ys.append(v)
    return np.array(xs), np.array(ys, dtype=float)


def plot_run(steps: List[Dict[str, Any]], label: str, path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    for i, m in enumerate(MODE_ORDER):
        x, y = series(steps, lambda r, m=m: r["train_mode_diffusion_loss"].get(m))
        if len(x):
            ax.plot(x, y, color=SLOTS[i], linewidth=1.5, label=m)
    style(ax, f"Run {label}: diffusion loss per mode (mean of each logging window)", "noise MSE")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax = axes[0, 1]
    x, y = series(steps, lambda r: r["dist_loss"])
    ax.plot(x, y, color=SLOTS[0], linewidth=1.5)
    style(ax, "Distance loss (photo shown)", "MSE")
    ax = axes[1, 0]
    x, y = series(steps, lambda r: r["map_attention_share"])
    ax.plot(x, y, color=SLOTS[0], linewidth=1.5)
    ax.axhline(0.20, color=MUTED, linewidth=1, linestyle="--")
    ax.text(x[0] if len(x) else 0, 0.205, "even share 0.20", fontsize=8, color=MUTED)
    style(ax, "Map attention share (map-goal samples of the batch)", "share")
    ax.set_xlabel("global step", fontsize=9, color=MUTED)
    ax = axes[1, 1]
    names = sorted({k for r in steps for k in r["lr"]})
    for i, n in enumerate(names):
        x, y = series(steps, lambda r, n=n: r["lr"].get(n))
        ax.plot(x, y, color=SLOTS[i], linewidth=1.5, label=n)
    ax.set_yscale("log")
    style(ax, "Learning rate per parameter group", "lr (log)")
    ax.set_xlabel("global step", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def plot_overlay(runs: Dict[str, List[Dict[str, Any]]], traces: Dict[str, Optional[List[Dict[str, Any]]]],
                 path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    panels = [("map_goal", "Diffusion loss, map-goal mode"), ("gs_photo", "Diffusion loss, GoStanford photo")]
    for j, (m, title) in enumerate(panels):
        ax = axes[0, j]
        for i, (label, steps) in enumerate(runs.items()):
            x, y = series(steps, lambda r, m=m: r["train_mode_diffusion_loss"].get(m))
            ax.plot(x, y, color=SLOTS[i], linewidth=1.5, label=f"run {label}")
        style(ax, title, "noise MSE")
        ax.set_xlabel("global step", fontsize=9, color=MUTED)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 2]
    for i, (label, steps) in enumerate(runs.items()):
        x, y = series(steps, lambda r: r["map_attention_share"])
        ax.plot(x, y, color=SLOTS[i], linewidth=1.5, label=f"run {label}")
    style(ax, "Map attention share (training batches)", "share")
    ax.set_xlabel("global step", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False)
    for col, (key, lo, hi, title) in enumerate([
            ("no_harm_delta", "no_harm_ci_low", "no_harm_ci_high", "No-harm delta (GoStanford subset, EMA)"),
            ("R", "R_ci_low", "R_ci_high", "Map effect R (offline subset, EMA)"),
            ("attention_share", None, None, "Attention share, map-goal (offline subset)")]):
        ax = axes[1, col]
        for i, (label, rows) in enumerate(traces.items()):
            if not rows:
                continue
            e = np.array([r["epoch"] for r in rows])
            v = np.array([r[key] for r in rows])
            if lo:
                ax.fill_between(e, [r[lo] for r in rows], [r[hi] for r in rows], color=SLOTS[i], alpha=0.15,
                                linewidth=0)
            ax.plot(e, v, color=SLOTS[i], linewidth=2, marker="o", markersize=5, label=f"run {label}")
        if key == "no_harm_delta":
            ax.axhline(NO_HARM_LIMIT, color=MUTED, linewidth=1, linestyle="--")
            ax.axvspan(11.5, 20.5, color=GRID, alpha=0.4, linewidth=0)
            ax.text(12, NO_HARM_LIMIT, " +0.08 limit, epochs 12-20", fontsize=8, color=MUTED, va="bottom")
        style(ax, title, key)
        ax.set_xlim(-0.5, 20.5)
        ax.set_xticks(range(0, 21, 2))
        ax.set_xlabel("epoch (1-based; 0 = before training)", fontsize=9, color=MUTED)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, help="LABEL=<run folder>, repeat per run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    folders = dict(r.split("=", 1) for r in args.run)
    os.makedirs(args.out, exist_ok=True)
    runs = {k: read_steps(v) for k, v in folders.items()}
    traces = {k: (read_trace(os.path.join(v, "no_harm_trace.csv"))
                  if os.path.exists(os.path.join(v, "no_harm_trace.csv")) else None) for k, v in folders.items()}
    for label, steps in runs.items():
        plot_run(steps, label, os.path.join(args.out, f"curves_{label}.png"))
    if len(runs) > 1:
        plot_overlay(runs, traces, os.path.join(args.out, "curves_" + "_vs_".join(runs) + ".png"))
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
