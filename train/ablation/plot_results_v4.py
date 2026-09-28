"""Phase 6: Figure 1 with V4 -- the Phase 5 cost decomposition plus the fused model.

Left: mean gc_action_loss per config on the headline (object-word) tier, V1 V2 V2p V3 (Phase 4)
and V4_photo V4_p V4_word, each with a 95% trajectory-bootstrap CI of the mean; dashed line at
V1. Inset: gc_dist_loss for V1, V2, V4_photo, V4_word, with the constant predictor (the train
mean distance, from the G0 probe) as a reference line.
Right: paired gaps with 95% trajectory-bootstrap CIs, parsed from the reports so the figure
matches them: the three Phase 5 gaps (gap_analysis.txt) and V4_photo - V1, V4_photo - V2,
V4_word - V4_p (gap_analysis_v4.txt). '*' / filled marker where the CI excludes 0.

Output (never overwritten): /outputs/nomad_clip_v6/eval/figure1_v4.png

Run inside the container, from /app/visualnav-transformer/train:

    python ablation/plot_results_v4.py
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_results import boot_ci, tier_masks
from analyze_v4 import PHASE4_DIR, load_v4
from plot_results import GAPS as PHASE5_GAPS, INK, MUTED, NUM, gc_line, section
from summarize_clip_eval import load

DEFAULT_V4_DIR = "/outputs/nomad_clip_v6/eval"
PROBE = "/outputs/nomad_clip_v6/probe_g0/results.json"
OLD, V4 = ["V1", "V2", "V2p", "V3"], ["V4_photo", "V4_p", "V4_word"]
V4_GAPS = [("V4_photo", "V1", "fusion vs vanilla (photo goal)"),
           ("V4_photo", "V2", "fusion vs CLIP goal only"),
           ("V4_word", "V4_p", "modality gap, fused model")]
BLUE, ORANGE = "#2a78d6", "#eb6834"  # reference categorical slots 1 and 2
SEED = 0


def v4_gap(text, a, b):
    """(gap, traj lo, traj hi) of gc_action_loss for a - b, headline section of gap_analysis_v4."""
    block = section(text, "== headline:")
    block = block[block.index(f"-- {a} - {b}"):]
    line = next(l for l in block.splitlines() if l.strip().startswith("gc_action_loss"))
    nums = [float(v) for v in re.findall(NUM, line)]
    return nums[0], nums[4], nums[5]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--v4-dir", default=DEFAULT_V4_DIR)
    args = parser.parse_args()
    out_path = os.path.join(args.v4_dir, "figure1_v4.png")
    if os.path.exists(out_path):
        raise SystemExit(f"FAIL: {out_path} exists; refusing to overwrite")

    data = load(os.path.join(PHASE4_DIR, "results.csv"))
    data.update(load_v4(os.path.join(args.v4_dir, "results_v4.csv"), data["V1"]))
    keep = tier_masks(data)[0]["headline"]
    traj = data["V1"]["traj_id"][keep]
    rng = np.random.default_rng(SEED)
    configs = OLD + V4
    means, cis = [], []
    for c in configs:
        v = data[c]["gc_action_loss"][keep]
        means.append(float(v.mean()))
        cis.append(boot_ci(v, traj, rng))
    dist = {c: float(data[c]["gc_dist_loss"][keep].mean()) for c in ("V1", "V2", "V4_photo", "V4_word")}
    with open(PROBE) as f:
        constant_mse = json.load(f)["results"]["P-int"]["photo / object words"]["constant_mse"]
    assert json.load(open(PROBE))["results"]["P-int"]["photo / object words"]["n"] == int(keep.sum())

    with open(os.path.join(PHASE4_DIR, "gap_analysis.txt")) as f:
        p5 = f.read()
    with open(os.path.join(args.v4_dir, "gap_analysis_v4.txt")) as f:
        p6 = f.read()
    gaps = []
    for a, b, meaning in PHASE5_GAPS:
        nums = [float(v) for v in re.findall(NUM, gc_line(section(p5, f"== b. paired gap {a} - {b} -- headline ==")))]
        gaps.append((f"{a} − {b}\n{meaning}", nums[0], nums[4], nums[5], BLUE))
    for a, b, meaning in V4_GAPS:
        gaps.append((f"{a} − {b}\n{meaning}", *v4_gap(p6, a, b), ORANGE))

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.25, 1.4]})
    colors = [BLUE] * len(OLD) + [ORANGE] * len(V4)
    x = np.arange(len(configs))
    left.bar(x, means, color=colors, width=0.62, edgecolor="white", linewidth=2)
    left.errorbar(x, means, yerr=[[m - lo for m, (lo, _) in zip(means, cis)],
                                  [hi - m for m, (_, hi) in zip(means, cis)]],
                  fmt="none", ecolor=INK, capsize=3, lw=1.2)
    for xi, m, (_, hi) in zip(x, means, cis):
        left.text(xi, hi + 0.08, f"{m:.2f}", ha="center", va="bottom", color=INK, fontsize=8.5)
    left.axhline(means[0], color=MUTED, ls="--", lw=1)
    left.set_xticks(x)
    left.set_xticklabels([c.replace("_", "\n") for c in configs])
    left.set_ylim(0, 9.5)
    left.set_yticks(range(0, 7))
    left.set_ylabel("gc_action_loss (lower is better)")
    left.set_title("Mean gc_action_loss, 95% trajectory CI", color=INK)

    inset = left.inset_axes([0.05, 0.72, 0.5, 0.25])
    names = list(dist)
    inset.bar(range(len(names)), [dist[n] for n in names],
              color=[BLUE, BLUE, ORANGE, ORANGE], width=0.6, edgecolor="white", linewidth=1.5)
    inset.axhline(constant_mse, color=INK, ls=":", lw=1)
    inset.text(-0.45, constant_mse + 2, f"constant {constant_mse:.1f}",
               ha="left", va="bottom", color=MUTED, fontsize=7)
    for i, n in enumerate(names):
        # V2 sits on the constant line; lift its label clear of the constant's.
        inset.text(i + (0.12 if n == "V2" else 0), dist[n] + 2, f"{dist[n]:.1f}", ha="center", va="bottom", color=INK, fontsize=7)
    inset.set_xticks(range(len(names)))
    inset.set_xticklabels(["V1", "V2", "V4ph", "V4w"], fontsize=7)
    inset.set_ylim(0, max(dist.values()) * 1.3)
    inset.tick_params(labelsize=7, colors=MUTED)
    inset.set_title("gc_dist_loss", fontsize=8, color=INK)
    inset.spines[["top", "right"]].set_visible(False)

    ys = list(range(len(gaps)))[::-1]
    for y, (label, g, lo, hi, color) in zip(ys, gaps):
        sig = lo > 0 or hi < 0
        right.errorbar(g, y, xerr=[[g - lo], [hi - g]], fmt="o", ms=8, color=color,
                       mfc=color if sig else "white", capsize=4, lw=2)
        right.text(g, y + 0.2, f"{g:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if sig else ''}",
                   va="bottom", ha="center", color=INK, fontsize=8.5)
    right.axvline(0, color=MUTED, lw=1)
    right.axhline(len(V4_GAPS) - 0.5, color=MUTED, lw=0.6, ls=":")
    right.set_yticks(ys)
    right.set_yticklabels([g[0] for g in gaps], fontsize=9)
    right.set_ylim(-0.6, len(gaps) - 0.3)
    right.set_xlabel("paired Δ gc_action_loss (positive = worse)")
    right.set_title("Paired gaps, 95% trajectory-bootstrap CI", color=INK)

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(handles, ["Phase 4/5 (V1 vanilla; V2, V2p, V3 CLIP goal only)", "Phase 6 V4 (CLIP current + goal fusion)"],
               loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"Figure 1 with V4 — headline tier ({int(keep.sum()):,} object-word goals)", color=INK)
    fig.text(0.5, 0.01, "* = trajectory CI excludes 0 (filled marker). Dashed line = V1. "
             "Inset dotted line = constant predictor (train mean distance).",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
