"""Assemble the capstone comparison from per-sample scores into one results record.

Reads what `eval_paired.py` wrote and answers the capstone's questions with both kinds of
uncertainty attached, never one alone:

- **paired SE** (within-run): could this gap be test-set sampling noise? Computed from the
  per-sample differences, so the variance the runs share cancels.
- **across-seed SD** (between-run): would a rerun of the same recipe land this far away?
  Taken from runs of one recipe that differ only in `seed`.

A contrast between runs carries *both*: its test-set noise, and the seed noise of every
single-seed run that went into it. Those are added in quadrature into one combined
uncertainty, and a contrast is judged against that -- not against the paired SE, which
would overstate the evidence, and not against one run's seed SD, which would understate it
for contrasts built from several runs. The additivity check is the case that matters: its
prediction is built from three other runs, so its seed noise is two ruler-widths, not one.

Run inside the container, from `/app/visualnav-transformer/train`:
    python ablation/assemble_capstone.py --scores /outputs/nomad_capstone/scores \
        --write-md ../docs/capstone_results.md
"""

import argparse
import os

import numpy as np

from eval_paired import LOWER_IS_BETTER, METRICS
from report_capstone import load_scores

PRIMARY = "gc_action_loss"

# The capstone's runs, by the name each was scored under.
RULER_SEEDS = ["ctx03_s0", "ctx03_s1", "ctx03_s2"]   # stock recipe, 30 epochs
STOCK100_SEEDS = ["stock100_s0", "stock100_s1"]      # stock recipe, 100 epochs, pinned data
BC30 = "bc30_s0"
BC100 = "bc100_s0"
STRIDE3 = "stride3_s0"
IMG160 = "img160x120_s0"
VANILLA100 = "vanilla_ema99"                         # stock recipe, 100 epochs, +26% data

ALL_RUNS = RULER_SEEDS + STOCK100_SEEDS + [BC30, BC100, STRIDE3, IMG160, VANILLA100]

UNCERTAINTY_HEADING = "## Two uncertainties, and how they are combined"

# A contrast is called only when it clears its combined uncertainty by this factor.
DECISION_SIGMAS = 2.0


def per_sample(scores, runs, metric):
    """Per-sample scores averaged over `runs` -- one run, or several seeds of one recipe."""
    return np.mean([scores[run][metric] for run in runs], axis=0)


def seed_sd(scores, runs, metric):
    """Across-seed SD of the run means (ddof=1). Two seeds give it one degree of freedom."""
    return float(np.std([scores[run][metric].mean() for run in runs], ddof=1))


class Contrast:
    """A signed combination of runs, e.g. bc100 - mean(stock100), with both uncertainties.

    `terms` maps a tuple of runs (averaged together as one recipe) to its coefficient. The
    seed variance of a term with k seeds and coefficient c is c^2 * sigma^2 / k, because
    averaging k seeds shrinks that recipe's seed noise by k.
    """

    def __init__(self, label, terms, why, directions=None):
        # `directions` names what a negative and a positive result MEAN, for contrasts
        # that are not "first run vs second run" -- an interaction whose positive sign
        # means an advantage narrowed must not be reported as "worse".
        self.label, self.terms, self.why, self.directions = label, terms, why, directions

    def evaluate(self, scores, metric, sigma):
        values = sum(
            coefficient * per_sample(scores, runs, metric)
            for runs, coefficient in self.terms
        )
        delta = float(values.mean())
        paired_se = float(values.std(ddof=1) / np.sqrt(len(values)))
        seed_noise = float(
            sigma * np.sqrt(sum(c * c / len(runs) for runs, c in self.terms))
        )
        combined = float(np.hypot(paired_se, seed_noise))
        return delta, paired_se, seed_noise, combined


def verdict(delta, combined, metric, directions=None):
    if abs(delta) < DECISION_SIGMAS * combined:
        return "not distinguishable"
    if directions:
        negative, positive = directions
        return negative if delta < 0 else positive
    improved = (delta < 0) if metric in LOWER_IS_BETTER else (delta > 0)
    return "better" if improved else "worse"


CONTRASTS = [
    Contrast(
        "HEADLINE — recipe effect at 100 epochs",
        [((BC100,), 1.0), (tuple(STOCK100_SEEDS), -1.0)],
        "best-combined@100 − clean-stock@100 (2-seed mean). Same pinned data, same 32×8 "
        "batch, same budget: only the recipe differs.",
    ),
    Contrast(
        "recipe effect at 100 epochs, vs stock seed 0 only",
        [((BC100,), 1.0), ((STOCK100_SEEDS[0],), -1.0)],
        "The same contrast against the seed-matched baseline alone, as a robustness check.",
    ),
    Contrast(
        "recipe effect at 30 epochs",
        [((BC30,), 1.0), (tuple(RULER_SEEDS), -1.0)],
        "best-combined@30 − ctx03 (3-seed mean).",
    ),
    Contrast(
        "does the recipe effect shrink with training?",
        [((BC100,), 1.0), (tuple(STOCK100_SEEDS), -1.0), ((BC30,), -1.0),
         (tuple(RULER_SEEDS), 1.0)],
        "(recipe effect @100) − (recipe effect @30). Positive = the advantage narrows "
        "given more epochs.",
        directions=("advantage grows", "advantage narrows"),
    ),
    Contrast(
        "additivity shortfall at 30 epochs",
        [((BC30,), 1.0), ((STRIDE3,), -1.0), ((IMG160,), -1.0), ((RULER_SEEDS[0],), 1.0)],
        "(bc30 − ctx03) − [(stride3 − ctx03) + (img160x120 − ctx03)] = bc30 − stride3 − "
        "img160x120 + ctx03, all seed 0. Zero = the two knobs add exactly; positive = they "
        "overlap. Four single-seed runs, so its seed noise is 2σ.",
        directions=("super-additive", "sub-additive"),
    ),
    Contrast(
        "budget effect, stock recipe",
        [(tuple(STOCK100_SEEDS), 1.0), (tuple(RULER_SEEDS), -1.0)],
        "clean-stock@100 − ctx03@30.",
    ),
    Contrast(
        "budget effect, improved recipe",
        [((BC100,), 1.0), ((BC30,), -1.0)],
        "best-combined@100 − best-combined@30.",
    ),
    Contrast(
        "compute check — best-combined@30 vs clean-stock@100",
        [((BC30,), 1.0), (tuple(STOCK100_SEEDS), -1.0)],
        "Epoch-MISmatched on purpose: best-combined@30 took ~5.7 GPU-hours, clean-stock@100 "
        "~10.5. Asks whether the improved recipe reaches the stock result on about half "
        "the compute.",
    ),
    Contrast(
        "SECONDARY — best-combined@100 vs vanilla@100",
        [((BC100,), 1.0), ((VANILLA100,), -1.0)],
        "Real-world reference only: vanilla trained on 26% more data with a true 256 batch. "
        "Confounded in vanilla's favour; see the confound row below.",
    ),
    Contrast(
        "confound measured — clean-stock@100 vs vanilla@100",
        [(tuple(STOCK100_SEEDS), 1.0), ((VANILLA100,), -1.0)],
        "Same recipe and budget; differs only in index_context_size (vanilla +26% data) and "
        "batch mechanics. This measures the confound directly at matched budget.",
        directions=("confound favours clean-stock", "confound favours vanilla"),
    ),
]

HEADLINE = CONTRASTS[0]


def fmt(value, metric):
    return f"{value:.5f}" if metric != "gc_dist_loss" else f"{value:.3f}"


def build_markdown(scores, manifests):
    sigma30 = seed_sd(scores, RULER_SEEDS, PRIMARY)
    sigma100 = seed_sd(scores, STOCK100_SEEDS, PRIMARY)
    n_samples = manifests[RULER_SEEDS[0]]["n_samples"]

    def cell(runs):
        means = [scores[run][PRIMARY].mean() for run in runs]
        mean = float(np.mean(means))
        if len(runs) == 1:
            return f"**{mean:.4f}**<br>n = 1 seed"
        return (
            f"**{mean:.4f}**<br>n = {len(runs)} seeds "
            f"({', '.join(f'{m:.4f}' for m in means)})<br>"
            f"across-seed SD {seed_sd(scores, runs, PRIMARY):.4f}"
        )

    lines = [
        "# Capstone results — best-combined recipe vs stock NoMaD",
        "",
        "## Setup",
        "",
        f"- Every number is scored by `eval_paired.py` on the same **{n_samples:,}** test "
        "samples (`index_context_size = 20`), each model at its own native input recipe. "
        "Scoring is deterministic: the same checkpoint scores bit-identically twice.",
        "- Primary metric **`gc_action_loss`** (↓). Final EMA weights throughout.",
        "- **Improved recipe** = `context_size 3` + `context_stride 3` + `image_size "
        "[160, 120]`. **Stock recipe** = `context_size 3`, stride 1, `[96, 96]`. Both use "
        "the pinned sample index and the 32×8 effective batch of 256 — except vanilla.",
        "",
        UNCERTAINTY_HEADING,
        "",
        "| error bar | what it answers | value on `gc_action_loss` |",
        "|---|---|---|",
        "| **paired SE** (within-run) | could this gap be test-set sampling noise? | "
        "per contrast, ≈ 0.012–0.024 |",
        f"| **across-seed SD** (between-run), ctx03 @30, n = 3 | would a reseed land this "
        f"far away? | **σ = {sigma30:.5f}** — the ruler used below |",
        f"| across-seed SD, clean-stock @100, n = 2 | same, at the headline's budget | "
        f"{sigma100:.5f} (1 degree of freedom — a spot check, not a ruler) |",
        "",
        "Each contrast's **seed noise** is σ scaled by how many single-seed runs it is "
        "built from (averaging k seeds of a recipe divides that recipe's seed variance by "
        "k). Its **combined** uncertainty adds seed noise and paired SE in quadrature. A "
        f"contrast is called only when it clears the combined uncertainty by "
        f"{DECISION_SIGMAS:g}×.",
        "",
        "Absolute cells are shown without a marginal SE on purpose: that SE (≈ 0.03) is "
        "dominated by sample-to-sample variance every run shares, so it says nothing "
        "about whether two cells differ. Compare cells through the contrasts, not by eye.",
        "",
        "## The grid — `gc_action_loss` ↓",
        "",
        "| | stock recipe (ctx 3 · stride 1 · 96×96) | improved recipe (ctx 3 · stride 3 · 160×120) |",
        "|---|---|---|",
        f"| **30 epochs** | ctx03<br>{cell(RULER_SEEDS)} | best-combined@30<br>{cell([BC30])} |",
        f"| **100 epochs** — primary | clean-stock@100<br>{cell(STOCK100_SEEDS)} | "
        f"best-combined@100<br>{cell([BC100])} |",
        f"| 100 epochs — secondary, confounded | vanilla@100 (`ema_99`)<br>"
        f"{cell([VANILLA100])}<br>*+26% train data, true 256 batch* | *not run — no "
        "improved-recipe counterpart on vanilla's data* |",
        "",
        "## Contrasts — `gc_action_loss`",
        "",
        "| contrast | Δ | paired SE | seed noise | combined | Δ / combined | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for contrast in CONTRASTS:
        delta, se, seed_noise, combined = contrast.evaluate(scores, PRIMARY, sigma30)
        lines.append(
            f"| **{contrast.label}** | {delta:+.5f} | ±{se:.5f} | ±{seed_noise:.5f} | "
            f"±{combined:.5f} | {abs(delta) / combined:.1f} | "
            f"{verdict(delta, combined, PRIMARY, contrast.directions)} |"
        )
    lines += [""]
    lines += [f"- **{c.label}** — {c.why}" for c in CONTRASTS]

    # Headline across every metric, each against its OWN seed ruler.
    lines += [
        "",
        "## The headline on every metric, each against its own ruler",
        "",
        "The Phase 4 ruler showed the secondary metrics are far more seed-sensitive than "
        "`gc_action_loss`, so each is judged against its own across-seed SD.",
        "",
        "| metric | best-combined@100 | clean-stock@100 | Δ | own σ (ctx03) | combined | "
        "Δ / combined | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for metric in METRICS:
        sigma = seed_sd(scores, RULER_SEEDS, metric)
        delta, _, _, combined = HEADLINE.evaluate(scores, metric, sigma)
        arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
        lines.append(
            f"| `{metric}` {arrow} | {fmt(scores[BC100][metric].mean(), metric)} | "
            f"{fmt(per_sample(scores, STOCK100_SEEDS, metric).mean(), metric)} | "
            f"{delta:+.5f} | {sigma:.5f} | ±{combined:.5f} | "
            f"{abs(delta) / combined:.1f} | {verdict(delta, combined, metric)} |"
        )

    lines += ["", "## Runs", "", "| run | checkpoint | sha256 | recipe |", "|---|---|---|---|"]
    for run in ALL_RUNS:
        manifest = manifests[run]
        checkpoint = manifest["checkpoint"]
        lines.append(
            f"| `{run}` | `{os.path.basename(os.path.dirname(checkpoint))}/"
            f"{os.path.basename(checkpoint)}` | `{manifest['checkpoint_sha256'][:12]}` | "
            f"{manifest['recipe']} |"
        )
    return lines, sigma30, sigma100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--write-md", metavar="PATH")
    parser.add_argument(
        "--reading",
        metavar="PATH",
        help="markdown file appended as the interpretation section",
    )
    args = parser.parse_args()

    scores, manifests = load_scores(args.scores, ALL_RUNS)
    lines, sigma30, _ = build_markdown(scores, manifests)

    print(f"ruler sigma (ctx03 @30, n=3) on {PRIMARY}: {sigma30:.5f}\n")
    header = f"{'contrast':<52}{'delta':>10}{'paired SE':>11}{'seed':>9}{'combined':>10}{'ratio':>7}  verdict"
    print(header)
    print("-" * (len(header) + 12))
    for contrast in CONTRASTS:
        delta, se, seed_noise, combined = contrast.evaluate(scores, PRIMARY, sigma30)
        print(
            f"{contrast.label:<52}{delta:>+10.5f}{se:>11.5f}{seed_noise:>9.5f}"
            f"{combined:>10.5f}{abs(delta) / combined:>7.1f}  "
            f"{verdict(delta, combined, PRIMARY, contrast.directions)}"
        )

    if args.write_md:
        if args.reading:
            # After Setup, so a reader knows what was measured before reading the answer,
            # and before the uncertainty and grid sections the answer refers to.
            with open(args.reading) as f:
                at = lines.index(UNCERTAINTY_HEADING)
                lines = lines[:at] + [f.read().strip(), ""] + lines[at:]
        with open(args.write_md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nwrote {args.write_md}")


if __name__ == "__main__":
    main()
