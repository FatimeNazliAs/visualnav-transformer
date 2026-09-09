import argparse
import csv
import os

import numpy as np
import torch

from eval_paired import (
    ALIGNED_INDEX_CONTEXT_SIZE,
    LOWER_IS_BETTER,
    METRICS as EVAL_METRICS,
    build_test_loader,
    enforce_determinism,
    load_arm_config,
    mean_and_standard_error,
    recipe_summary,
    score_checkpoint,
)

# The A1/A2 tables report the sampler metrics only. `eval_paired` also produces the
# deterministic denoising loss, which the capstone comparison leads with but which these
# tables predate -- filtering it out here keeps their shape, and keeps per-sample scores
# cached before that metric existed readable.
METRICS = [name for name in EVAL_METRICS if not name.endswith("diffusion_loss")]

# The three metrics the write-up leads with, in reporting order; the remaining METRICS
# are still emitted, after these.
HEADLINE_METRICS = ["gc_action_loss", "gc_action_waypts_cos_sim", "uc_action_loss"]
REPORT_METRICS = HEADLINE_METRICS + [m for m in METRICS if m not in HEADLINE_METRICS]


def load_config(run_dir, config_dir):
    """Recover the arm's config from its run name (ctx03_<timestamp> -> nomad_ctx03.yaml)."""
    arm = os.path.basename(os.path.normpath(run_dir)).split("_")[0]
    return load_arm_config(
        os.path.join(config_dir, f"nomad_{arm}.yaml"), ALIGNED_INDEX_CONTEXT_SIZE
    )


def arm_shape(config):
    """(context_size, context_stride, effective temporal window) for one arm.

    The window is how far back the context reaches, in waypoints. It is the axis the two
    ablations share, so it -- not context_size -- is what the table is ordered by.
    """
    context_size = config["context_size"]
    context_stride = config.get("context_stride", 1)
    return context_size, context_stride, context_size * context_stride


def format_arm_table(arms, shapes):
    """How each arm reaches its window: by spending tokens, by spreading them, or both."""
    header = (
        f"{'arm':<12}{'context_size':>14}{'context_stride':>16}"
        f"{'frames':>9}{'window':>9}"
    )
    lines = [header, "-" * len(header)]
    for arm in arms:
        context_size, context_stride, window = shapes[arm]
        lines.append(
            f"{arm:<12}{context_size:>14}{context_stride:>16}"
            f"{context_size + 1:>9}{window:>9}"
        )
    return "\n".join(lines)


def format_table(arms, scores, baseline):
    """Marginal means with standard errors, then paired deltas against the baseline arm."""
    metric_width = 28
    cell_width = 24

    def block(title, header_cells, rows):
        header = f"{title:<{metric_width}}" + "".join(
            f"{cell:>{cell_width}}" for cell in header_cells
        )
        lines = [header, "-" * len(header)]
        for name, cells in rows:
            lines.append(
                f"{name:<{metric_width}}"
                + "".join(f"{cell:>{cell_width}}" for cell in cells)
            )
        return lines

    marginal_rows = []
    for metric in METRICS:
        cells = []
        for arm in arms:
            mean, se = mean_and_standard_error(scores[arm][metric])
            cells.append(f"{mean:.5f} ±{se:.5f}")
        marginal_rows.append((metric, cells))

    compared = [arm for arm in arms if arm != baseline]
    delta_rows = []
    for metric in METRICS:
        cells = []
        for arm in compared:
            delta = scores[arm][metric] - scores[baseline][metric]
            mean, se = mean_and_standard_error(delta)
            verdict = "~"
            if abs(mean) > 2 * se:
                better = (mean < 0) if metric in LOWER_IS_BETTER else (mean > 0)
                verdict = "better" if better else "worse"
            cells.append(f"{mean:+.5f} ±{se:.5f} {verdict}")
        delta_rows.append((metric, cells))

    lines = block("metric", arms, marginal_rows)
    lines.append("")
    lines.append(f"Paired per-sample difference vs {baseline} (identical inputs):")
    lines.extend(block("metric", compared, delta_rows))
    lines.append("")
    lines.append("  '~' = within 2 SE of the baseline; 'better'/'worse' = beyond 2 SE.")
    return "\n".join(lines)


def one_line_finding(arms, scores, baseline, primary="gc_action_loss"):
    """State the trend on the primary metric, in the direction the metric runs."""
    compared = [arm for arm in arms if arm != baseline]
    if not compared:
        return "Only the baseline arm was scored; no comparison available."
    deltas = {}
    for arm in compared:
        deltas[arm] = mean_and_standard_error(
            scores[arm][primary] - scores[baseline][primary]
        )
    lower_better = primary in LOWER_IS_BETTER
    worse = [a for a, (m, se) in deltas.items() if abs(m) > 2 * se and ((m > 0) == lower_better)]
    better = [a for a, (m, se) in deltas.items() if abs(m) > 2 * se and ((m < 0) == lower_better)]
    parts = ", ".join(f"{a} {deltas[a][0]:+.5f}±{deltas[a][1]:.5f}" for a in compared)
    if worse and not better:
        verdict = "a wider temporal window HURTS"
    elif better and not worse:
        verdict = "a wider temporal window HELPS"
    elif not better and not worse:
        verdict = "a wider temporal window has NO EFFECT beyond test-set noise"
    else:
        verdict = "the effect depends on HOW the window is widened"
    return (
        f"On {primary} vs {baseline}, {verdict} ({parts}); "
        "n = 1 seed, so these SEs are test-set estimation noise, not seed variance."
    )


def headline_pair_finding(scores, shapes, arm, other, primary="gc_action_loss"):
    """Direct paired comparison between two arms that reach back about equally far.

    Taken from the per-sample scores rather than by subtracting two deltas, because the
    SE of a difference needs the covariance between the arms, not just their two SEs.
    """
    delta, se = mean_and_standard_error(scores[arm][primary] - scores[other][primary])
    verdict = "within test-set noise"
    if abs(delta) > 2 * se:
        better = (delta < 0) if primary in LOWER_IS_BETTER else (delta > 0)
        verdict = f"**{'better' if better else 'worse'}** ({abs(delta) / se:.1f} SE)"
    return (
        f"Headline — `{arm}` (window {shapes[arm][2]}, {shapes[arm][0] + 1} frames) vs "
        f"`{other}` (window {shapes[other][2]}, {shapes[other][0] + 1} frames) on "
        f"`{primary}`: {delta:+.5f} ±{se:.5f}, {verdict}. Near-identical reach, very "
        "different token budgets."
    )


def write_results_csv(path, arms, scores, configs, baseline, n_samples, epoch):
    """One row per arm: config, marginal mean/SE, and paired delta/SE vs the baseline."""
    header = ["arm", "config", "n_samples", "checkpoint", "baseline"]
    for metric in REPORT_METRICS:
        header += [
            f"{metric}_mean",
            f"{metric}_se",
            f"{metric}_delta_vs_{baseline}",
            f"{metric}_delta_se",
        ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for arm in arms:
            row = [arm, recipe_summary(configs[arm]), n_samples, f"ema_{epoch}.pth", baseline]
            for metric in REPORT_METRICS:
                mean, se = mean_and_standard_error(scores[arm][metric])
                if arm == baseline:
                    delta, delta_se = "", ""
                else:
                    d, d_se = mean_and_standard_error(
                        scores[arm][metric] - scores[baseline][metric]
                    )
                    delta, delta_se = f"{d:.5f}", f"{d_se:.5f}"
                row += [f"{mean:.5f}", f"{se:.5f}", delta, delta_se]
            writer.writerow(row)


def write_results_md(
    path, arms, scores, configs, shapes, baseline, n_samples, epoch, title, headline_pair
):
    compared = [arm for arm in arms if arm != baseline]
    lines = [
        f"# {title}",
        "",
        f"- Test samples per arm: **{n_samples:,}** (identical inputs across arms, so the "
        "paired deltas below are valid)",
        f"- Checkpoint: `ema_{epoch}.pth` · baseline arm: `{baseline}` · "
        "**n = 1 seed per arm**",
        "- Rows are ordered by **effective temporal window** (`context_size × "
        "context_stride`), the axis the two ablations share.",
        "",
        "## Arms",
        "",
        "| arm | config |",
        "|---|---|",
    ]
    lines += [f"| `{arm}` | {recipe_summary(configs[arm])} |" for arm in arms]

    lines += [
        "",
        "## Marginal means ±SE",
        "",
        "| metric | " + " | ".join(f"`{arm}`" for arm in arms) + " |",
        "|---" * (len(arms) + 1) + "|",
    ]
    for metric in REPORT_METRICS:
        arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
        cells = []
        for arm in arms:
            mean, se = mean_and_standard_error(scores[arm][metric])
            cells.append(f"{mean:.5f} ±{se:.5f}")
        lines.append(f"| `{metric}` {arrow} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"## Paired per-sample difference vs `{baseline}`",
        "",
        "| metric | " + " | ".join(f"`{arm}`" for arm in compared) + " |",
        "|---" * (len(compared) + 1) + "|",
    ]
    for metric in REPORT_METRICS:
        cells = []
        for arm in compared:
            d, se = mean_and_standard_error(
                scores[arm][metric] - scores[baseline][metric]
            )
            verdict = "~"
            if abs(d) > 2 * se:
                better = (d < 0) if metric in LOWER_IS_BETTER else (d > 0)
                verdict = "**better**" if better else "**worse**"
            cells.append(f"{d:+.5f} ±{se:.5f} {verdict}")
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "`~` = within 2 SE of the baseline; **better**/**worse** = beyond 2 SE.",
        "",
        "## Finding",
        "",
        one_line_finding(arms, scores, baseline),
    ]
    if headline_pair and all(arm in scores for arm in headline_pair):
        lines += ["", headline_pair_finding(scores, shapes, *headline_pair)]
    lines += [
        "",
        "Standard errors describe estimation noise on this test set, not training-seed",
        "variance. With n = 1 seed per arm, a gap larger than 2 SE means it exceeds",
        "test-set estimation noise; it is not a seed-level significance claim.",
        "",
        "Regenerate with `ablation/evaluate_sweep.py --write-results <dir>` "
        "(reads the cached per-sample scores; does not re-score).",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="ARM=RUN_DIR",
        help="Repeatable, e.g. --run ctx03=/outputs/nomad_ctx_ablation/ctx03_2026_08_31",
    )
    parser.add_argument("--epoch", type=int, default=29, help="EMA checkpoint epoch to score")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--baseline", default="ctx03", help="arm the paired deltas are taken against")
    parser.add_argument(
        "--cache-dir",
        default="/outputs/nomad_ctx_ablation/eval_cache",
        help="where per-sample scores are cached, so re-running the table is free",
    )
    parser.add_argument("--no-cache", action="store_true", help="re-evaluate even if cached")
    parser.add_argument(
        "--write-results",
        metavar="DIR",
        help="also write results.csv and results.md into DIR",
    )
    parser.add_argument("--title", default="Ablation results", help="heading used in results.md")
    parser.add_argument(
        "--headline-pair",
        metavar="ARM:OTHER",
        help="two arms to compare directly in the finding, e.g. stride3:ctx10",
    )
    args = parser.parse_args()

    enforce_determinism()
    runs = dict(entry.split("=", 1) for entry in args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores, sample_counts, shapes, configs = {}, set(), {}, {}
    for arm, run_dir in runs.items():
        config = load_config(run_dir, args.config_dir)
        shapes[arm] = arm_shape(config)
        configs[arm] = config

        # An arm's per-sample scores are fully determined by its checkpoint, its config
        # and the seed, so they are cached: a failure late in a sweep then costs only the
        # arm that failed, and re-rendering the table against another baseline is free.
        cache_path = os.path.join(args.cache_dir, f"{arm}_ema{args.epoch}.npz")
        if os.path.exists(cache_path) and not args.no_cache:
            print(f"{arm}: reusing cached per-sample scores from {cache_path}")
            scores[arm] = dict(np.load(cache_path))
        else:
            dataset, loader = build_test_loader(config, args.batch_size)
            checkpoint = os.path.join(run_dir, f"ema_{args.epoch}.pth")
            try:
                scores[arm] = score_checkpoint(
                    checkpoint, config, loader, device, args.seed, desc=arm
                )
            finally:
                # The arms are scored one after another in this process, and each needs
                # its own dataset because the context shape differs. LMDB reader slots
                # are per-process, so the previous arm's cache has to be released before
                # the next arm opens the same file.
                dataset.close()
            os.makedirs(args.cache_dir, exist_ok=True)
            np.savez(cache_path, **scores[arm])
            print(f"{arm}: cached per-sample scores to {cache_path}")

        sample_counts.add(len(scores[arm]["gc_action_loss"]))

    if len(sample_counts) != 1:
        raise SystemExit(
            f"FAIL: arms scored different numbers of samples {sample_counts}; "
            "the paired comparison is only valid on identical inputs."
        )

    # Ordered by effective temporal window, so arms that see equally far back sit
    # together regardless of which ablation produced them.
    arms = sorted(runs, key=lambda arm: (shapes[arm][2], shapes[arm][0]))
    n_samples = sample_counts.pop()
    print()
    print(f"Test samples per arm: {n_samples:,}   (n = 1 seed per arm)")
    print(f"EMA checkpoint: ema_{args.epoch}.pth")
    print()
    print(format_arm_table(arms, shapes))
    print()
    print(format_table(arms, scores, args.baseline))
    print()
    print(
        "Standard errors describe estimation noise on this test set, not training-seed\n"
        "variance. With n = 1 seed per arm a gap larger than 2 SE means it exceeds\n"
        "test-set estimation noise; it is not a seed-level significance claim."
    )

    if args.write_results:
        headline_pair = tuple(args.headline_pair.split(":", 1)) if args.headline_pair else None
        os.makedirs(args.write_results, exist_ok=True)
        csv_path = os.path.join(args.write_results, "results.csv")
        md_path = os.path.join(args.write_results, "results.md")
        write_results_csv(
            csv_path, arms, scores, configs, args.baseline, n_samples, args.epoch
        )
        write_results_md(
            md_path, arms, scores, configs, shapes, args.baseline, n_samples,
            args.epoch, args.title, headline_pair,
        )
        print()
        print(f"wrote {csv_path}")
        print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
