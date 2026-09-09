"""Read per-sample scores written by `eval_paired.py` and report them as a table.

Scoring and reporting are kept apart on purpose. Scoring a checkpoint costs a few GPU
minutes and its result is a fact about that checkpoint; re-reading those facts against a
different reference run, or with another run added, must cost nothing. So `eval_paired.py`
writes per-sample arrays to disk once, and this reads them as often as the write-up needs.

Every delta here is **paired**: the per-sample difference between two runs scored on
identical samples, carrying its own standard error. That error is much smaller than the
difference of the two marginal errors, because the sample-to-sample variance the two runs
share cancels.

That paired SE answers "could this gap be test-set sampling noise". It does **not** answer
"would a rerun reproduce it" -- for that you need to retrain the same recipe under a
different seed and see how far the two land apart. `--seed-group` computes that second,
larger error bar from a set of runs that differ only in `seed`, and `--judge` measures a
delta against it. The two must always be reported side by side: presenting the paired SE
alone would overstate what the numbers establish, by roughly an order of magnitude.

Run inside the container, from `/app/visualnav-transformer/train`:
    python ablation/report_capstone.py --scores /outputs/nomad_capstone/scores \
        --arm vanilla_ema29 --arm vanilla_ema49 --arm vanilla_ema69 --arm vanilla_ema99 \
        --reference vanilla_ema99
"""

import argparse
import json
import os

import numpy as np

from eval_paired import LOWER_IS_BETTER, METRICS, mean_and_standard_error, paired_delta

# How far a paired delta must exceed its own standard error before it is called at all.
# This is test-set estimation noise only; it says nothing about training-seed variance.
SIGNIFICANCE_SE = 2.0


def load_scores(scores_dir, arms):
    """Per-sample arrays and manifests for the named runs, checked for comparability."""
    scores, manifests = {}, {}
    for arm in arms:
        score_path = os.path.join(scores_dir, f"{arm}.npz")
        if not os.path.exists(score_path):
            raise SystemExit(f"FAIL: no scores for {arm!r} at {score_path}")
        scores[arm] = dict(np.load(score_path))
        with open(os.path.join(scores_dir, f"{arm}.json")) as f:
            manifests[arm] = json.load(f)

    counts = {manifests[arm]["n_samples"] for arm in arms}
    if len(counts) != 1:
        raise SystemExit(
            f"FAIL: runs were scored on different numbers of samples {counts}; "
            "the paired deltas would not be comparing like with like."
        )
    partial = [arm for arm in arms if manifests[arm]["max_batches"] is not None]
    if partial:
        raise SystemExit(
            f"FAIL: {', '.join(partial)} scored only part of the split (--max-batches); "
            "those runs are for the determinism check, not for results."
        )
    return scores, manifests


def verdict(delta, se, metric):
    """'better' / 'worse' / '~', in the direction the metric runs."""
    if abs(delta) <= SIGNIFICANCE_SE * se:
        return "~"
    improved = (delta < 0) if metric in LOWER_IS_BETTER else (delta > 0)
    return "better" if improved else "worse"


def seed_statistics(scores, seeds, metric):
    """Mean, across-seed SD and range for one metric over runs of an identical recipe.

    ddof=1: these seeds are a sample of the seeds that could have been drawn, not the
    population, and with only three of them the correction is not a rounding detail.
    """
    values = np.array([scores[seed][metric].mean() for seed in seeds])
    return values, float(values.mean()), float(values.std(ddof=1)), float(values.ptp())


def format_seed_group(label, seeds, scores):
    """The across-seed ruler: how far apart runs of the SAME recipe land."""
    header = (
        f"{'metric':<28}{'per-seed means':>44}{'mean':>12}"
        f"{'across-seed SD':>18}{'range':>12}"
    )
    lines = [
        f"Across-seed ruler: {label} ({len(seeds)} seeds -- identical recipe, "
        f"differing only in `seed`)",
        header,
        "-" * len(header),
    ]
    for metric in METRICS:
        values, mean, sd, spread = seed_statistics(scores, seeds, metric)
        rendered = ", ".join(f"{value:.5f}" for value in values)
        lines.append(
            f"{metric:<28}{rendered:>44}{mean:>12.5f}{sd:>18.5f}{spread:>12.5f}"
        )
    return "\n".join(lines)


def format_judgements(label, seeds, scores, metric, judgements):
    """Measure externally-supplied deltas against the ruler.

    A delta smaller than the across-seed SD is indistinguishable from having retrained
    the same recipe with a different seed, however many standard errors it clears on the
    test set.
    """
    _, _, sd, _ = seed_statistics(scores, seeds, metric)
    header = f"{'delta':<28}{'value':>12}{'/ seed SD':>12}   verdict"
    lines = [
        f"Judged against the {label} across-seed SD on `{metric}` "
        f"(SD = {sd:.5f}):",
        header,
        "-" * (len(header) + 24),
    ]
    for name, delta in judgements:
        ratio = abs(delta) / sd if sd else float("inf")
        if ratio >= 2.0:
            call = "exceeds the ruler"
        elif ratio >= 1.0:
            call = "comparable to seed noise"
        else:
            call = "WITHIN seed noise"
        lines.append(f"{name:<28}{delta:>+12.5f}{ratio:>12.1f}x   {call}")
    return "\n".join(lines)


def format_table(arms, scores, reference):
    metric_width, cell_width = 28, 26
    lines = []

    def block(title, columns, rows):
        header = f"{title:<{metric_width}}" + "".join(
            f"{column:>{cell_width}}" for column in columns
        )
        out = [header, "-" * len(header)]
        out += [
            f"{name:<{metric_width}}" + "".join(f"{cell:>{cell_width}}" for cell in cells)
            for name, cells in rows
        ]
        return out

    marginal = []
    for metric in METRICS:
        cells = []
        for arm in arms:
            mean, se = mean_and_standard_error(scores[arm][metric])
            cells.append(f"{mean:.5f} +-{se:.5f}")
        marginal.append((f"{metric} {'v' if metric in LOWER_IS_BETTER else '^'}", cells))
    lines += block("metric", arms, marginal)

    compared = [arm for arm in arms if arm != reference]
    if compared:
        deltas = []
        for metric in METRICS:
            cells = []
            for arm in compared:
                delta, se = paired_delta(scores, arm, reference, metric)
                cells.append(f"{delta:+.5f} +-{se:.5f} {verdict(delta, se, metric)}")
            deltas.append((metric, cells))
        lines += ["", f"Paired per-sample difference vs {reference} (identical samples):"]
        lines += block("metric", compared, deltas)
        lines += [
            "",
            f"  '~' = within {SIGNIFICANCE_SE:g} SE of {reference}; "
            "'better'/'worse' = beyond it.",
        ]
    return "\n".join(lines)


def write_markdown(path, arms, scores, manifests, reference, title, note,
                   seed_label=None, seed_arms=(), judgements=(), judge_metric=None):
    compared = [arm for arm in arms if arm != reference]
    n_samples = manifests[arms[0]]["n_samples"]
    lines = [
        f"# {title}",
        "",
        f"- Test samples per run: **{n_samples:,}**, the aligned set "
        f"(`index_context_size = {manifests[arms[0]]['index_context_size']}`); identical "
        "across runs, which is what makes the paired deltas below valid.",
        f"- Reference run for the paired deltas: `{reference}`.",
        "- Produced by `ablation/report_capstone.py` from per-sample scores written by "
        "`ablation/eval_paired.py`; re-rendering does not re-score.",
        "",
        "## Runs",
        "",
        "| run | checkpoint | recipe |",
        "|---|---|---|",
    ]
    lines += [
        f"| `{arm}` | `{os.path.basename(manifests[arm]['checkpoint'])}` "
        f"(`{manifests[arm]['checkpoint_sha256'][:12]}`) | {manifests[arm]['recipe']} |"
        for arm in arms
    ]

    lines += [
        "",
        "## Marginal means ±SE",
        "",
        "| metric | " + " | ".join(f"`{arm}`" for arm in arms) + " |",
        "|---" * (len(arms) + 1) + "|",
    ]
    for metric in METRICS:
        arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
        cells = [
            "{:.5f} ±{:.5f}".format(*mean_and_standard_error(scores[arm][metric]))
            for arm in arms
        ]
        lines.append(f"| `{metric}` {arrow} | " + " | ".join(cells) + " |")

    if compared:
        lines += [
            "",
            f"## Paired per-sample difference vs `{reference}`",
            "",
            "| metric | " + " | ".join(f"`{arm}`" for arm in compared) + " |",
            "|---" * (len(compared) + 1) + "|",
        ]
        for metric in METRICS:
            cells = []
            for arm in compared:
                delta, se = paired_delta(scores, arm, reference, metric)
                call = verdict(delta, se, metric)
                mark = f"**{call}**" if call != "~" else "~"
                cells.append(f"{delta:+.5f} ±{se:.5f} {mark}")
            lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
        lines += [
            "",
            f"`~` = within {SIGNIFICANCE_SE:g} SE of `{reference}`; "
            "**better**/**worse** = beyond it.",
        ]

    if seed_arms:
        _, _, sd, _ = seed_statistics(scores, seed_arms, "gc_action_loss")
        lines += [
            "",
            f"## Across-seed ruler — `{seed_label}`, {len(seed_arms)} seeds",
            "",
            "These runs share one recipe and differ only in `seed`. How far apart they land",
            "is how much of any effect could be a reseed rather than the knob under test.",
            "It is a different, and usually larger, uncertainty than the paired SE above.",
            "",
            "| metric | per-seed means | mean | across-seed SD | range |",
            "|---|---|---|---|---|",
        ]
        for metric in METRICS:
            values, mean, metric_sd, spread = seed_statistics(scores, seed_arms, metric)
            rendered = ", ".join(f"{value:.5f}" for value in values)
            lines.append(
                f"| `{metric}` | {rendered} | {mean:.5f} | **{metric_sd:.5f}** | {spread:.5f} |"
            )
        lines += [
            "",
            f"With {len(seed_arms)} seeds the SD has only {len(seed_arms) - 1} degrees of "
            "freedom, so it is itself a rough estimate: the 95% interval for the true "
            f"sigma spans roughly {0.52 * sd:.5f} to {6.29 * sd:.5f} on `gc_action_loss`. "
            "Read it as an order-of-magnitude bar, and treat an effect clearing it by only "
            "a small factor as unproven.",
        ]

    if judgements:
        _, _, sd, _ = seed_statistics(scores, seed_arms, judge_metric)
        lines += [
            "",
            f"## Effects judged against the ruler — `{judge_metric}` (SD = {sd:.5f})",
            "",
            "| effect | delta | / seed SD | verdict |",
            "|---|---|---|---|",
        ]
        for name, delta in judgements:
            ratio = abs(delta) / sd if sd else float("inf")
            if ratio >= 2.0:
                call = "exceeds the ruler"
            elif ratio >= 1.0:
                call = "comparable to seed noise"
            else:
                call = "**within seed noise**"
            lines.append(f"| {name} | {delta:+.5f} | {ratio:.1f}x | {call} |")

    if note:
        lines += ["", "## Reading", "", note]
    lines += [
        "",
        "These standard errors describe estimation noise on this test set. They are **not**",
        "training-seed variance, and a gap beyond 2 SE here is not a claim that a rerun with",
        "another seed would reproduce it. The across-seed ruler is built in Phase 4.",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="directory eval_paired.py wrote to")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="repeatable; reported in the order given",
    )
    parser.add_argument(
        "--reference",
        help="run the paired deltas are taken against (default: the last --arm)",
    )
    parser.add_argument(
        "--seed-group",
        metavar="LABEL=ARM,ARM,...",
        help="runs of one identical recipe differing only in seed; reports the "
        "across-seed SD, the bar an effect must clear to be more than a reseed",
    )
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        metavar="LABEL=DELTA",
        help="repeatable; measure an externally-computed delta against the seed ruler",
    )
    parser.add_argument(
        "--judge-metric",
        default="gc_action_loss",
        help="metric the seed ruler judges deltas on",
    )
    parser.add_argument("--title", default="Capstone scores")
    parser.add_argument("--note", help="a 'Reading' paragraph for the markdown output")
    parser.add_argument("--write-md", metavar="PATH", help="also write a markdown table")
    args = parser.parse_args()

    arms = args.arm
    reference = args.reference or arms[-1]
    if reference not in arms:
        raise SystemExit(f"FAIL: --reference {reference!r} is not among the --arm runs")

    seed_label, seed_arms = None, []
    if args.seed_group:
        seed_label, _, joined = args.seed_group.partition("=")
        seed_arms = [arm for arm in joined.split(",") if arm]
        missing = [arm for arm in seed_arms if arm not in arms]
        if missing:
            raise SystemExit(
                f"FAIL: --seed-group names {missing} which are not among the --arm runs"
            )
        if len(seed_arms) < 2:
            raise SystemExit("FAIL: --seed-group needs at least two runs to have a spread")

    judgements = []
    for entry in args.judge:
        name, _, value = entry.partition("=")
        try:
            judgements.append((name, float(value)))
        except ValueError:
            raise SystemExit(f"FAIL: --judge {entry!r} is not LABEL=DELTA")
    if judgements and not seed_arms:
        raise SystemExit("FAIL: --judge needs a --seed-group to judge against")

    scores, manifests = load_scores(args.scores, arms)
    n_samples = manifests[arms[0]]["n_samples"]

    print()
    print(args.title)
    print(f"Test samples per run: {n_samples:,}")
    print()
    print(format_table(arms, scores, reference))
    print()
    if seed_arms:
        print(format_seed_group(seed_label, seed_arms, scores))
        print()
    if judgements:
        print(
            format_judgements(
                seed_label, seed_arms, scores, args.judge_metric, judgements
            )
        )
        print()

    if args.write_md:
        write_markdown(
            args.write_md, arms, scores, manifests, reference, args.title, args.note,
            seed_label, seed_arms, judgements, args.judge_metric,
        )
        print(f"wrote {args.write_md}")


if __name__ == "__main__":
    main()
