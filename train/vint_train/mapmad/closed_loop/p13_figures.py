"""Phase 13 thesis figures from the finished runs (no new runs), matplotlib only:

    python -m vint_train.mapmad.closed_loop.p13_figures --config <p13_claim.yaml>     # -> <out_dir>/figures/

1. success_both_sets.{png,pdf}: oracle success per arm on the HM3D exam and exam B, home-clustered 95% CI
   (tables.json "arms_both_sets", written by p13_tables);
2. map_gain_bins.{png,pdf}: map gain (maps shown - maps hidden, oracle, paired) per distance bin for v2.1 and v2.1b on
   both sets, home-clustered 95% CI (p13_tables.Sets.paired over the bin's episodes: the tables' statistics);
3. vlfm_port_check.{png,pdf}: the original VLFM's success at the 0.1 m rule vs the same stops re-scored at 0.2 m
   (Wilson 95% CI; tables.json "port_check") with the third-party HM3D v2 reference (63.6%).
figures_data.json holds every plotted number. Colours: categorical slots 1-2 of the reference palette (blue, orange),
recessive grid and axes, light surface.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mapmad_sim.run_layout import RunLayout, write_text_atomic  # noqa: E402
from vint_train.mapmad.closed_loop.p13_tables import Sets  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1-2
SETS = (("hm3d", "HM3D exam (300 episodes, 34 homes)"), ("examb", "Exam B, MP3D (291 episodes, 26 homes)"))
ARMS = ("N", "M21h", "M21bh", "M21", "M21b", "VF", "F")  # bottom to top in figure 1
ARM_LABEL = {"N": "NoMaD (explore)", "M21h": "MapMaD v2.1, maps hidden", "M21bh": "MapMaD v2.1b, maps hidden",
             "M21": "MapMaD v2.1, maps shown", "M21b": "MapMaD v2.1b, maps shown", "VF": "VLFM-style (VF)",
             "F": "Map + frontier planner (F)"}
BINS = ((0, "3–6 m"), (1, "6–10 m"), (2, "10–15 m"))
PAIRS = (("M21", "M21h", "MapMaD v2.1"), ("M21b", "M21bh", "MapMaD v2.1b (equal DAgger practice)"))
PORT_REF = 0.636


def style(ax) -> None:
    """Recessive axes: no top/right spines, light grid behind the marks, secondary ink for ticks."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelcolor=INK, length=3)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_success(t: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """Figure 1: horizontal grouped bars, one row per arm, HM3D and exam B side by side, CI whiskers."""
    data = {st: {a: t["arms_both_sets"][f"{st}:{a}"] for a in ARMS} for st, _ in SETS}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style(ax)
    y, h = np.arange(len(ARMS)), 0.36
    for k, (st, name) in enumerate(SETS):
        m = np.array([data[st][a]["oracle"] for a in ARMS])
        lo = np.array([data[st][a]["oracle_ci"][0] for a in ARMS])
        hi = np.array([data[st][a]["oracle_ci"][1] for a in ARMS])
        yy = y + (h / 2 + 0.02) * (1 if k == 0 else -1)
        ax.barh(yy, m, h, color=SERIES[k], label=name, edgecolor=SURFACE, linewidth=1)
        ax.errorbar(m, yy, xerr=[m - lo, hi - m], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABEL[a] for a in ARMS])
    ax.set_xlim(0, 0.8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Oracle success (mean of 3 seeds; 95% CI clustered by home)", color=INK2)
    ax.set_title("Success per arm on both exam sets", color=INK, loc="left", fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    save(fig, out, "success_both_sets")
    return data


def fig_bins(s: Sets, out: Path) -> Dict[str, Any]:
    """Figure 2: one panel per set; per bin the paired map gain of v2.1 and v2.1b with home CIs."""
    data: Dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    x, w = np.arange(len(BINS)), 0.36
    for ax, (st, name) in zip(axes, SETS):
        style(ax)
        rows = s.rows(st, "M21")
        data[st] = {}
        for k, (a, b, label) in enumerate(PAIRS):
            m, err = [], [[], []]
            for bi, bl in BINS:
                keep = {r["episode_id"] for r in rows if r["distance_bin"] == bi}
                p = s.paired(st, a, b, keep=keep)
                data[st][f"{a}-{b}:{bl}"] = {k2: p[k2] for k2 in ("diff", "home_ci", "n", "homes")}
                m.append(p["diff"])
                err[0].append(p["diff"] - p["home_ci"][0])
                err[1].append(p["home_ci"][1] - p["diff"])
            xx = x + (w / 2 + 0.02) * (-1 if k == 0 else 1)
            ax.bar(xx, m, w, color=SERIES[k], label=label, edgecolor=SURFACE, linewidth=1)
            ax.errorbar(xx, m, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([bl for _, bl in BINS])
        ax.set_xlabel("Start distance to the nearest view point", color=INK2)
        ax.set_title(name, color=INK, loc="left", fontsize=10)
    axes[0].set_ylabel("Map gain: maps shown − hidden\n(oracle success, 95% CI by home)", color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("The map's gain at every distance", color=INK, x=0.01, ha="left", fontsize=12)
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"map_gain_bins.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return data


def fig_port(t: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """Figure 3: two bars (0.1 m as run, 0.2 m re-scored) with Wilson CIs and the reference line."""
    port = t["port_check"]
    data = {k: port["radii"][k] for k in ("0.10", "0.20")}
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    style(ax)
    labels = ["0.1 m\n(VLFM / Habitat rule, as run)", "0.2 m\n(same stops re-scored;\nour exam's success zone)"]
    m = np.array([data[k]["success"] for k in data])
    lo = np.array([data[k]["ci"][0] for k in data])
    hi = np.array([data[k]["ci"][1] for k in data])
    ax.bar([0, 1], m, 0.5, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.errorbar([0, 1], m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for xi, v in enumerate(m):
        ax.text(xi, v / 2, f"{v:.0%}", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax.axhline(PORT_REF, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.45, PORT_REF + 0.015, "HM3D v2, reported by other papers (63.6%)", ha="left", va="bottom", color=INK2,
            fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.5, 1.4)
    ax.set_ylim(0, 0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_ylabel("Success (95% Wilson CI)", color=INK2)
    ax.set_title(f"Original VLFM, {port['n']} HM3D v2 val episodes", color=INK, loc="left", fontsize=11)
    save(fig, out, "vlfm_port_check")
    return data


def main(argv: List[str] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    layout = RunLayout.load(args.config)
    t = json.loads((layout.base / "tables" / "tables.json").read_text())
    out = layout.base / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = {"success_both_sets": fig_success(t, out), "map_gain_bins": fig_bins(Sets(layout), out),
            "vlfm_port_check": fig_port(t, out), "port_reference": PORT_REF}
    write_text_atomic(out / "figures_data.json", json.dumps(data, indent=1, sort_keys=True, default=float) + "\n")
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
