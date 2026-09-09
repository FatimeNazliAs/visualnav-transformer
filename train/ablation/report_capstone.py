"""Read per-sample scores written by `eval_paired.py` and report them as a table.

Scoring and reporting are kept apart on purpose. Scoring a checkpoint costs a few GPU
minutes and its result is a fact about that checkpoint; re-reading those facts against a
different reference run, or with another run added, must cost nothing. So `eval_paired.py`
writes per-sample arrays to disk once, and this reads them as often as the write-up needs.

Every delta here is **paired**: the per-sample difference between two runs scored on
identical samples, carrying its own standard error. That error is much smaller than the
difference of the two marginal errors, because the sample-to-sample variance the two runs
share cancels.

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


def write_markdown(path, arms, scores, manifests, reference, title, note):
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
    parser.add_argument("--title", default="Capstone scores")
    parser.add_argument("--note", help="a 'Reading' paragraph for the markdown output")
    parser.add_argument("--write-md", metavar="PATH", help="also write a markdown table")
    args = parser.parse_args()

    arms = args.arm
    reference = args.reference or arms[-1]
    if reference not in arms:
        raise SystemExit(f"FAIL: --reference {reference!r} is not among the --arm runs")

    scores, manifests = load_scores(args.scores, arms)
    n_samples = manifests[arms[0]]["n_samples"]

    print()
    print(args.title)
    print(f"Test samples per run: {n_samples:,}")
    print()
    print(format_table(arms, scores, reference))
    print()

    if args.write_md:
        write_markdown(
            args.write_md, arms, scores, manifests, reference, args.title, args.note
        )
        print(f"wrote {args.write_md}")


if __name__ == "__main__":
    main()
