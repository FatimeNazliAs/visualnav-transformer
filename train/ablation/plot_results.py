"""Phase 5: results figure from analyze_results.py's gap_analysis.txt.

Left: mean gc_action_loss per config on the headline tier, with a reference line at V1.
Right: the three paired gaps on the headline tier with 95% trajectory-bootstrap CIs; '*' where
the CI excludes 0. Values are parsed from gap_analysis.txt so the figure matches the report.

Output (next to gap_analysis.txt, never overwritten): phase5_results.png

Run inside the container, from /app/visualnav-transformer/train:

    python ablation/plot_results.py
"""
import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_REPORT = "/outputs/nomad_clip_eval/gap_analysis.txt"
CONFIGS = ["V1", "V2", "V2p", "V3"]
GAPS = [("V2", "V1", "CLIP goal encoding"),
        ("V2p", "V2", "word instead of exact frame"),
        ("V3", "V2p", "modality gap (text vs image)")]
NUM = r"([+-]?\d+\.\d+)"
INK, MUTED, BAR = "#0b0b0b", "#52514e", "#2a78d6"


def section(text, header):
    start = text.index(header)
    end = text.find("\n==", start + len(header))
    return text[start:end if end != -1 else None]


def gc_line(block):
    return next(line for line in block.splitlines() if line.startswith("gc_action_loss"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()
    out_path = os.path.join(os.path.dirname(args.report), "phase5_results.png")
    if os.path.exists(out_path):
        raise SystemExit(f"FAIL: {out_path} exists; refusing to overwrite")

    with open(args.report) as f:
        text = f.read()
    block = section(text, "== a. per-config mean ± std -- headline")
    n_cases = re.search(r"\(([\d,]+) cases", block).group(1)
    means = [float(v) for v in re.findall(NUM, gc_line(block))[0::2]]
    gaps = []
    for a, b, meaning in GAPS:
        nums = [float(v) for v in re.findall(NUM, gc_line(section(
            text, f"== b. paired gap {a} - {b} -- headline ==")))]
        gaps.append((f"{a} − {b}\n{meaning}", nums[0], nums[4], nums[5]))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2),
                                      gridspec_kw={"width_ratios": [1, 1.4]})
    left.bar(CONFIGS, means, color=BAR, width=0.6)
    for x, m in enumerate(means):
        left.text(x, m + 0.06, f"{m:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
    left.axhline(means[0], color=MUTED, ls="--", lw=1)
    left.set_ylim(0, max(means) * 1.12)
    left.set_ylabel("gc_action_loss (lower is better)")
    left.set_title("Mean gc_action_loss", color=INK)

    ys = list(range(len(gaps)))[::-1]
    for y, (label, g, lo, hi) in zip(ys, gaps):
        sig = lo > 0 or hi < 0
        right.errorbar(g, y, xerr=[[g - lo], [hi - g]], fmt="o", ms=8, color=INK,
                       mfc=INK if sig else "white", capsize=4, lw=2)
        right.text(hi, y + 0.18, f"  {g:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if sig else ''}",
                   va="bottom", ha="center", color=INK, fontsize=9)
    right.axvline(0, color=MUTED, lw=1)
    right.set_yticks(ys)
    right.set_yticklabels([g[0] for g in gaps])
    right.set_ylim(-0.6, len(gaps) - 0.3)
    right.set_xlabel("paired Δ gc_action_loss (positive = worse)")
    right.set_title("Paired gaps, 95% trajectory-bootstrap CI", color=INK)

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.suptitle(f"Phase 5 — headline tier ({n_cases} object-word goals)", color=INK)
    fig.text(0.5, 0.01, "* = CI excludes 0 (filled marker). Dashed line = V1.",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
