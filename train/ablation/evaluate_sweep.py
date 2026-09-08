"""Post-hoc evaluation and comparison table for the context_size ablation.

Why post-hoc rather than reading the training logs: the in-training eval only computes
`_compute_losses_nomad` when `i % print_log_freq == 0` and records `logger.latest()`, so
each epoch's number comes from a single shuffled batch. That is far too noisy to compare
arms. Here every arm is scored on the *whole* test split instead.

Because all arms share `index_context_size`, the sample index is identical across arms.
With `shuffle=False`, `num_workers=0` and a pinned seed, every arm therefore sees
bit-identical inputs — the same samples, the same sampled goals, the same negatives, and
the same denoising noise. That makes a *paired* comparison valid: the per-sample
difference between two arms cancels sample-to-sample variance and is far more sensitive
than comparing two independent means.

Metric definitions are taken verbatim from `_compute_losses_nomad` so the numbers mean
the same thing the repo's own logging means; the only change is that they are kept
per-sample instead of reduced to a scalar.

Run inside the container:
    CUDA_VISIBLE_DEVICES=1 python ablation/evaluate_sweep.py \
        --run ctx03=/outputs/nomad_ctx_ablation/ctx03_<stamp> \
        --run ctx10=/outputs/nomad_ctx_ablation/ctx10_<stamp> \
        --run ctx20=/outputs/nomad_ctx_ablation/ctx20_<stamp>
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.training.train_utils import model_output, ACTION_STATS

# Primary metric first; the table is ordered by this list.
METRICS = [
    "gc_action_loss",
    "gc_action_waypts_cos_sim",
    "uc_action_loss",
    "uc_action_waypts_cos_sim",
    "gc_dist_loss",
]
LOWER_IS_BETTER = {"gc_action_loss", "uc_action_loss", "gc_dist_loss"}

# The three metrics the write-up leads with, in reporting order; the remaining
# METRICS are still emitted, after these.
HEADLINE_METRICS = ["gc_action_loss", "gc_action_waypts_cos_sim", "uc_action_loss"]
REPORT_METRICS = HEADLINE_METRICS + [m for m in METRICS if m not in HEADLINE_METRICS]

IMAGENET_TRANSFORM = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def load_config(run_dir, config_dir):
    """Recover the arm's config from its run name (ctx03_<timestamp> -> nomad_ctx03.yaml)."""
    arm = os.path.basename(os.path.normpath(run_dir)).split("_")[0]
    with open(os.path.join(config_dir, "defaults.yaml")) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(config_dir, f"nomad_{arm}.yaml")) as f:
        config.update(yaml.safe_load(f))
    return config


def build_model(config, device):
    vision_encoder = replace_bn_with_gn(
        NoMaD_ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    )
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=DenseNetwork(embedding_dim=config["encoding_size"]),
    )
    return model.to(device).eval()


def build_test_loader(config, batch_size):
    data_config = config["datasets"]["go_stanford"]
    dataset = ViNT_Dataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config["test"],
        dataset_name="go_stanford",
        image_size=config["image_size"],
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
        min_dist_cat=config["distance"]["min_dist_cat"],
        max_dist_cat=config["distance"]["max_dist_cat"],
        min_action_distance=config["action"]["min_dist_cat"],
        max_action_distance=config["action"]["max_dist_cat"],
        negative_mining=data_config["negative_mining"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        context_size=config["context_size"],
        context_type=config["context_type"],
        index_context_size=config["index_context_size"],
        end_slack=data_config["end_slack"],
        goals_per_obs=data_config["goals_per_obs"],
        normalize=config["normalize"],
        goal_type=config["goal_type"],
    )
    # shuffle=False and num_workers=0 keep the sample order, the sampled goals and the
    # negatives identical across arms, which is what makes the paired comparison valid.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    return dataset, loader


def per_sample_metrics(actions, labels):
    """The `_compute_losses_nomad` formulas, kept per-sample."""
    squared_error = F.mse_loss(actions, labels, reduction="none")
    while squared_error.dim() > 1:
        squared_error = squared_error.mean(dim=-1)
    cos_sim = F.cosine_similarity(actions[:, :, :2], labels[:, :, :2], dim=-1).mean(dim=-1)
    return squared_error, cos_sim


def evaluate_checkpoint(checkpoint_path, config, loader, device, seed):
    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    results = {name: [] for name in METRICS}
    results["action_mask"] = []

    # Pinned here, immediately before iteration: the dataset samples goals and negatives
    # with numpy at __getitem__ time, so this fixes the inputs identically for every arm.
    np.random.seed(seed)

    with torch.no_grad():
        for batch_index, data in enumerate(
            tqdm.tqdm(loader, desc=os.path.basename(checkpoint_path), dynamic_ncols=True)
        ):
            obs_image, goal_image, actions, distance, _, _, action_mask = data

            obs_images = torch.split(obs_image, 3, dim=1)
            batch_obs_images = torch.cat(
                [IMAGENET_TRANSFORM(obs) for obs in obs_images], dim=1
            ).to(device)
            batch_goal_images = IMAGENET_TRANSFORM(goal_image).to(device)
            labels = actions.to(device)

            # Same denoising noise for every arm on every batch.
            torch.manual_seed(seed + batch_index)
            outputs = model_output(
                model,
                noise_scheduler,
                batch_obs_images,
                batch_goal_images,
                pred_horizon=labels.shape[1],
                action_dim=labels.shape[2],
                num_samples=1,
                device=device,
            )

            gc_loss, gc_cos = per_sample_metrics(outputs["gc_actions"], labels)
            uc_loss, uc_cos = per_sample_metrics(outputs["uc_actions"], labels)
            dist_error = (
                outputs["gc_distance"].squeeze(-1) - distance.to(device).float()
            ) ** 2

            results["gc_action_loss"].append(gc_loss.cpu().numpy())
            results["gc_action_waypts_cos_sim"].append(gc_cos.cpu().numpy())
            results["uc_action_loss"].append(uc_loss.cpu().numpy())
            results["uc_action_waypts_cos_sim"].append(uc_cos.cpu().numpy())
            results["gc_dist_loss"].append(dist_error.cpu().numpy())
            results["action_mask"].append(action_mask.numpy())

    return {name: np.concatenate(values) for name, values in results.items()}


def close_dataset(dataset):
    """Release the dataset's LMDB environment.

    LMDB is explicit that one process must not hold the same database open twice.
    Each arm needs its own dataset (the observation tensor has 3*(context_size+1)
    channels), so the envs must be closed as we go rather than left to the garbage
    collector -- otherwise the third open fails with
    `mdb_txn_begin: Invalid argument`.
    """
    cache = getattr(dataset, "_image_cache", None)
    if cache is not None:
        cache.close()
        dataset._image_cache = None


def mean_and_standard_error(values):
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


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


def config_summary(config):
    """One-cell description of what distinguishes this arm."""
    effective = config["batch_size"] * config.get("gradient_accumulation_steps", 1)
    return (
        f"context_size={config['context_size']}, "
        f"index_context_size={config['index_context_size']}, "
        f"image_size={config['image_size'][0]}x{config['image_size'][1]}, "
        f"eff_batch={effective} ({config['batch_size']}x{config.get('gradient_accumulation_steps', 1)}), "
        f"epochs={config['epochs']}, lr={config['lr']}, seed={config['seed']}"
    )


def one_line_finding(arms, scores, baseline, primary="gc_action_loss"):
    """State the trend on the primary metric, in the direction the metric runs."""
    compared = [a for a in arms if a != baseline]
    if not compared:
        return "Only the baseline arm was scored; no comparison available."
    deltas = {}
    for arm in compared:
        mean, se = mean_and_standard_error(scores[arm][primary] - scores[baseline][primary])
        deltas[arm] = (mean, se)
    lower_better = primary in LOWER_IS_BETTER
    worse = [a for a, (m, se) in deltas.items() if abs(m) > 2 * se and ((m > 0) == lower_better)]
    better = [a for a, (m, se) in deltas.items() if abs(m) > 2 * se and ((m < 0) == lower_better)]
    parts = ", ".join(f"{a} {deltas[a][0]:+.5f}±{deltas[a][1]:.5f}" for a in compared)
    if worse and not better:
        verdict = "more context HURTS"
    elif better and not worse:
        verdict = "more context HELPS"
    elif not better and not worse:
        verdict = "more context has NO EFFECT beyond test-set noise"
    else:
        verdict = "more context has a MIXED effect"
    return (
        f"On {primary} vs {baseline}, {verdict} ({parts}); "
        "n = 1 seed, so these SEs are test-set estimation noise, not seed variance."
    )


def write_results_csv(path, arms, scores, configs, baseline, n_samples, epoch):
    """One row per arm: config, marginal mean/SE, and paired delta/SE vs the baseline."""
    header = ["arm", "config", "n_samples", "checkpoint", "baseline"]
    for metric in REPORT_METRICS:
        header += [f"{metric}_mean", f"{metric}_se",
                   f"{metric}_delta_vs_{baseline}", f"{metric}_delta_se"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for arm in arms:
            row = [arm, config_summary(configs[arm]), n_samples, f"ema_{epoch}.pth", baseline]
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


def write_results_md(path, arms, scores, configs, baseline, n_samples, epoch, title):
    compared = [a for a in arms if a != baseline]
    lines = [
        f"# {title}",
        "",
        f"- Test samples per arm: **{n_samples:,}** (identical inputs across arms, so the "
        "paired deltas below are valid)",
        f"- Checkpoint: `ema_{epoch}.pth` · baseline arm: `{baseline}` · **n = 1 seed per arm**",
        "",
        "## Arms",
        "",
        "| arm | config |",
        "|---|---|",
    ]
    lines += [f"| `{a}` | {config_summary(configs[a])} |" for a in arms]

    lines += ["", "## Marginal means ±SE", "",
              "| metric | " + " | ".join(f"`{a}`" for a in arms) + " |",
              "|---" * (len(arms) + 1) + "|"]
    for metric in REPORT_METRICS:
        arrow = "↓" if metric in LOWER_IS_BETTER else "↑"
        cells = []
        for arm in arms:
            mean, se = mean_and_standard_error(scores[arm][metric])
            cells.append(f"{mean:.5f} ±{se:.5f}")
        lines.append(f"| `{metric}` {arrow} | " + " | ".join(cells) + " |")

    lines += ["", f"## Paired per-sample difference vs `{baseline}`", "",
              "| metric | " + " | ".join(f"`{a}`" for a in compared) + " |",
              "|---" * (len(compared) + 1) + "|"]
    for metric in REPORT_METRICS:
        cells = []
        for arm in compared:
            d, se = mean_and_standard_error(scores[arm][metric] - scores[baseline][metric])
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
    parser.add_argument(
        "--title",
        default="Ablation results",
        help="heading used in results.md",
    )
    args = parser.parse_args()

    runs = dict(entry.split("=", 1) for entry in args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scores, sample_counts, contexts, configs = {}, set(), {}, {}
    for arm, run_dir in runs.items():
        config = load_config(run_dir, args.config_dir)
        contexts[arm] = config["context_size"]
        configs[arm] = config

        cache_path = os.path.join(args.cache_dir, f"{arm}_ema{args.epoch}.npz")
        if os.path.exists(cache_path) and not args.no_cache:
            print(f"{arm}: reusing cached per-sample scores from {cache_path}")
            scores[arm] = dict(np.load(cache_path))
        else:
            dataset, loader = build_test_loader(config, args.batch_size)
            checkpoint = os.path.join(run_dir, f"ema_{args.epoch}.pth")
            try:
                scores[arm] = evaluate_checkpoint(
                    checkpoint, config, loader, device, args.seed
                )
            finally:
                close_dataset(dataset)
            os.makedirs(args.cache_dir, exist_ok=True)
            np.savez(cache_path, **scores[arm])
            print(f"{arm}: cached per-sample scores to {cache_path}")

        sample_counts.add(len(scores[arm]["gc_action_loss"]))

    if len(sample_counts) != 1:
        raise SystemExit(
            f"FAIL: arms scored different numbers of samples {sample_counts}; "
            "the paired comparison is only valid on identical inputs."
        )

    arms = sorted(runs, key=lambda arm: contexts[arm])
    n_samples = sample_counts.pop()
    print()
    print(f"Test samples per arm: {n_samples:,}   (n = 1 seed per arm)")
    print(f"context_size: " + ", ".join(f"{arm}={contexts[arm]}" for arm in arms))
    print(f"EMA checkpoint: ema_{args.epoch}.pth")
    print()
    print(format_table(arms, scores, args.baseline))
    print()
    print(
        "Standard errors describe estimation noise on this test set, not training-seed\n"
        "variance. With n = 1 seed per arm a gap larger than 2 SE means it exceeds\n"
        "test-set estimation noise; it is not a seed-level significance claim."
    )

    if args.write_results:
        os.makedirs(args.write_results, exist_ok=True)
        csv_path = os.path.join(args.write_results, "results.csv")
        md_path = os.path.join(args.write_results, "results.md")
        write_results_csv(
            csv_path, arms, scores, configs, args.baseline, n_samples, args.epoch
        )
        write_results_md(
            md_path, arms, scores, configs, args.baseline, n_samples, args.epoch, args.title
        )
        print()
        print(f"wrote {csv_path}")
        print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
