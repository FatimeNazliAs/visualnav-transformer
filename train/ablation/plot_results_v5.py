"""Phase 6b: Figure 1 with V5 -- the Phase 6 figure plus the text-mix-in model.

Left: mean gc_action_loss per config on the headline (object-word) tier, V1 V2 V2p V3
(Phase 4), V4_photo V4_p V4_word (Phase 6) and V5_photo V5_p V5_word, each with a 95%
trajectory-bootstrap CI of the mean; dashed line at V1. Inset: gc_dist_loss for V1, V2,
V4_photo, V4_word, V5_photo, V5_word, with the constant predictor as a reference line.
Right: paired gaps with 95% trajectory-bootstrap CIs, parsed from the reports so the figure
matches them: Phase 5 (gap_analysis.txt), Phase 6 (gap_analysis_v4.txt) and Phase 6b
(gap_analysis_v5.txt). '*' / filled marker where the CI excludes 0.

Output (never overwritten; figure1_v4.png is left alone): <v5 dir>/figure1_v5.png

Run inside the container, from /app/visualnav-transformer/train:

    python ablation/plot_results_v5.py
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
from analyze_v5 import PHASE4_DIR, load_all
from eval_v5 import DEFAULT_OUT_DIR, V4_DIR
from plot_results import GAPS as PHASE5_GAPS, INK, MUTED, NUM, gc_line, section
from plot_results_v4 import BLUE, ORANGE, PROBE, V4_GAPS, v4_gap

OLD, V4, V5 = ["V1", "V2", "V2p", "V3"], ["V4_photo", "V4_p", "V4_word"], ["V5_photo", "V5_p", "V5_word"]
V5_GAPS = [("V5_word", "V3", "text mix-in vs CLIP goal only (word)"),
           ("V5_word", "V5_p", "modality gap, text mix-in"),
           ("V5_photo", "V4_photo", "text mix-in cost on photo goals")]
GREEN = "#1f9e72"  # reference categorical slot 3
SEED = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--v5-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_path = os.path.join(args.v5_dir, "figure1_v5.png")
    if os.path.exists(out_path):
        raise SystemExit(f"FAIL: {out_path} exists; refusing to overwrite")

    data = load_all(args.v5_dir)
    keep = tier_masks(data)[0]["headline"]
    traj = data["V1"]["traj_id"][keep]
    rng = np.random.default_rng(SEED)
    configs = OLD + V4 + V5
    means, cis = [], []
    for c in configs:
        v = data[c]["gc_action_loss"][keep]
        means.append(float(v.mean()))
        cis.append(boot_ci(v, traj, rng))
    dist_names = ("V1", "V2", "V4_photo", "V4_word", "V5_photo", "V5_word")
    dist = {c: float(data[c]["gc_dist_loss"][keep].mean()) for c in dist_names}
    with open(PROBE) as f:
        probe = json.load(f)["results"]["P-int"]["photo / object words"]
    assert probe["n"] == int(keep.sum())
    constant_mse = probe["constant_mse"]

    with open(os.path.join(PHASE4_DIR, "gap_analysis.txt")) as f:
        p5 = f.read()
    with open(os.path.join(V4_DIR, "gap_analysis_v4.txt")) as f:
        p6 = f.read()
    with open(os.path.join(args.v5_dir, "gap_analysis_v5.txt")) as f:
        p6b = f.read()
    gaps = []
    for a, b, meaning in PHASE5_GAPS:
        nums = [float(v) for v in re.findall(NUM, gc_line(section(p5, f"== b. paired gap {a} - {b} -- headline ==")))]
        gaps.append((f"{a} − {b}\n{meaning}", nums[0], nums[4], nums[5], BLUE))
    for a, b, meaning in V4_GAPS:
        gaps.append((f"{a} − {b}\n{meaning}", *v4_gap(p6, a, b), ORANGE))
    for a, b, meaning in V5_GAPS:
        gaps.append((f"{a} − {b}\n{meaning}", *v4_gap(p6b, a, b), GREEN))

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.45, 1.3]})
    colors = [BLUE] * len(OLD) + [ORANGE] * len(V4) + [GREEN] * len(V5)
    x = np.arange(len(configs))
    left.bar(x, means, color=colors, width=0.62, edgecolor="white", linewidth=2)
    left.errorbar(x, means, yerr=[[m - lo for m, (lo, _) in zip(means, cis)],
                                  [hi - m for m, (_, hi) in zip(means, cis)]],
                  fmt="none", ecolor=INK, capsize=3, lw=1.2)
    for xi, m, (_, hi) in zip(x, means, cis):
        left.text(xi, hi + 0.08, f"{m:.2f}", ha="center", va="bottom", color=INK, fontsize=8)
    left.axhline(means[0], color=MUTED, ls="--", lw=1)
    left.set_xticks(x)
    left.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8.5)
    left.set_ylim(0, 9.5)
    left.set_yticks(range(0, 7))
    left.set_ylabel("gc_action_loss (lower is better)")
    left.set_title("Mean gc_action_loss, 95% trajectory CI", color=INK)

    inset = left.inset_axes([0.05, 0.72, 0.55, 0.25])
    inset.bar(range(len(dist_names)), [dist[n] for n in dist_names],
              color=[BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN], width=0.6, edgecolor="white", linewidth=1.5)
    inset.axhline(constant_mse, color=INK, ls=":", lw=1)
    inset.text(-0.45, constant_mse + 2, f"constant {constant_mse:.1f}",
               ha="left", va="bottom", color=MUTED, fontsize=7)
    for i, n in enumerate(dist_names):
        inset.text(i, dist[n] + 2, f"{dist[n]:.1f}", ha="center", va="bottom", color=INK, fontsize=7)
    inset.set_xticks(range(len(dist_names)))
    inset.set_xticklabels(["V1", "V2", "V4ph", "V4w", "V5ph", "V5w"], fontsize=7)
    inset.set_ylim(0, max(dist.values()) * 1.3)
    inset.tick_params(labelsize=7, colors=MUTED)
    inset.set_title("gc_dist_loss", fontsize=8, color=INK)
    inset.spines[["top", "right"]].set_visible(False)

    ys = list(range(len(gaps)))[::-1]
    for y, (label, g, lo, hi, color) in zip(ys, gaps):
        sig = lo > 0 or hi < 0
        right.errorbar(g, y, xerr=[[g - lo], [hi - g]], fmt="o", ms=7, color=color,
                       mfc=color if sig else "white", capsize=4, lw=2)
        right.text(g, y + 0.2, f"{g:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if sig else ''}",
                   va="bottom", ha="center", color=INK, fontsize=8)
    right.axvline(0, color=MUTED, lw=1)
    for boundary in (len(V5_GAPS) - 0.5, len(V5_GAPS) + len(V4_GAPS) - 0.5):
        right.axhline(boundary, color=MUTED, lw=0.6, ls=":")
    right.set_yticks(ys)
    right.set_yticklabels([g[0] for g in gaps], fontsize=8.5)
    right.set_ylim(-0.6, len(gaps) - 0.3)
    right.set_xlabel("paired Δ gc_action_loss (positive = worse)")
    right.set_title("Paired gaps, 95% trajectory-bootstrap CI", color=INK)

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GREEN)]
    fig.legend(handles, ["Phase 4/5 (V1 vanilla; V2, V2p, V3 CLIP goal only)",
                         "Phase 6 V4 (CLIP current + goal fusion)",
                         "Phase 6b V5 (V4 + text mix-in, p = 0.3)"],
               loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(f"Figure 1 with V5 — headline tier ({int(keep.sum()):,} object-word goals)", color=INK)
    fig.text(0.5, 0.01, "* = trajectory CI excludes 0 (filled marker). Dashed line = V1. "
             "Inset dotted line = constant predictor (train mean distance).",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
