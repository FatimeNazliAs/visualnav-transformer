"""Deterministic paired per-sample evaluation of NoMaD checkpoints.

This is the single scoring definition for the whole thesis: the ablation sweeps
(`evaluate_sweep.py`) and the capstone comparison both go through it, so every number
in the write-up means the same thing.

Why paired, and why per-sample
------------------------------
The in-training eval only computes `_compute_losses_nomad` when `i % print_log_freq == 0`
and records `logger.latest()`, so each epoch's number comes from a single shuffled batch.
That is far too noisy to compare runs. Here every run is scored on the *whole* test split.

Because all runs are indexed with the same `index_context_size`, the sample index is
identical across them. With `shuffle=False`, `num_workers=0` and a pinned seed, every run
therefore sees the same samples, the same sampled goals and the same negatives -- only the
pixels are resized differently and the context frames spaced differently, as each run's own
recipe dictates. That makes a *paired* comparison valid: the per-sample difference between
two runs cancels sample-to-sample variance and is far more sensitive than comparing two
independent means.

The two families of metric
--------------------------
`*_action_loss` / `*_action_waypts_cos_sim` / `gc_dist_loss` are the repo's own
`_compute_losses_nomad` formulas kept per-sample, so they mean what the repo's logging
means. They come out of the full 10-step reverse diffusion, whose noise is pinned per
batch (see `_seed_sampler`).

`*_diffusion_loss` is the training objective itself -- the noise-prediction error --
averaged over **all** `num_diffusion_iters` timesteps with **fixed per-sample noise**.
It touches no sampler, so it is both lower-variance and independent of how the samples
happen to be grouped into batches. It is the deterministic low-variance loss the capstone
comparison leads with.

Determinism
-----------
`enforce_determinism()` pins cuDNN algorithm selection and forbids nondeterministic CUDA
kernels; the per-sample denoising noise comes from per-sample generators keyed on the
sample's global index, and the sampler's noise from a per-batch seed. Scoring the same
checkpoint twice therefore reproduces every per-sample number bit for bit. Verify with
`check_eval_determinism.py`.

Run inside the container, from `/app/visualnav-transformer/train`:

    CUDA_VISIBLE_DEVICES=0 python ablation/eval_paired.py \
        --arm vanilla=config/nomad.yaml=/outputs/nomad/nomad_2026_06_13_18_04_23/ema_99.pth \
        --out /outputs/nomad_capstone/scores
"""

# cuBLAS reads this when it creates its handle, which happens the first time torch
# touches CUDA -- so it has to be in the environment before torch is imported. Without
# it, torch.use_deterministic_algorithms(True) refuses to run any cuBLAS GEMM.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json

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
from vint_train.training.train_utils import (
    ACTION_STATS,
    get_delta,
    model_output,
    normalize_data,
)

# Every run is indexed with this many leading timesteps skipped, whatever it trained on,
# so the sample index -- and therefore the pairing -- is the same for all of them.
ALIGNED_INDEX_CONTEXT_SIZE = 20

# Primary metric first; tables are ordered by this list.
METRICS = [
    "gc_diffusion_loss",
    "gc_action_loss",
    "gc_action_waypts_cos_sim",
    "uc_diffusion_loss",
    "uc_action_loss",
    "uc_action_waypts_cos_sim",
    "gc_dist_loss",
]
LOWER_IS_BETTER = {
    "gc_diffusion_loss",
    "uc_diffusion_loss",
    "gc_action_loss",
    "uc_action_loss",
    "gc_dist_loss",
}

# Kept alongside the metrics: the weight `_compute_losses_nomad` applies when it reduces
# to a scalar. Carried per-sample so a masked variant can be computed later without
# re-scoring; the marginal means below are unweighted, as in the ablation tables.
PER_SAMPLE_ARRAYS = METRICS + ["action_mask"]

IMAGENET_TRANSFORM = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def enforce_determinism():
    """Make a scoring pass reproduce itself bit for bit.

    cuDNN benchmarking picks a convolution algorithm by timing candidates at runtime, so
    it can pick differently on two runs of the same code and change the last digits of
    the result. Turning it off and demanding deterministic kernels removes both sources.
    Inference-only, so the cost is negligible.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_arm_config(config_path, index_context_size=ALIGNED_INDEX_CONTEXT_SIZE):
    """Read a run's training config, layered on defaults.yaml, ready for scoring.

    `index_context_size` is overridden rather than read from the config: it selects which
    samples are scored, and every run has to be scored on the same ones. The vanilla run
    saved no config at all, which is why the config is named explicitly instead of being
    guessed from the run directory.
    """
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(os.path.join(config_dir, "defaults.yaml")) as f:
        config = yaml.safe_load(f)
    with open(config_path) as f:
        config.update(yaml.safe_load(f))

    config["index_context_size"] = index_context_size
    config.setdefault("context_stride", 1)
    config.setdefault("gradient_accumulation_steps", 1)

    look_back = config["context_size"] * config["context_stride"]
    if index_context_size < look_back:
        raise SystemExit(
            f"FAIL: index_context_size ({index_context_size}) is smaller than this run's "
            f"look-back of context_size x context_stride ({look_back}); it could not be "
            "scored on the aligned sample index."
        )
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
    """The aligned test split, fed at this run's own native input recipe.

    Each run reads its own `image_size`, `context_size` and `context_stride` -- scoring a
    model on inputs it was never trained to read would measure the mismatch, not the
    model. `index_context_size` is what is held fixed instead, and that is what makes the
    sample order, the sampled goals and the negatives identical across runs.
    """
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
        context_stride=config["context_stride"],
        index_context_size=config["index_context_size"],
        end_slack=data_config["end_slack"],
        goals_per_obs=data_config["goals_per_obs"],
        normalize=config["normalize"],
        goal_type=config["goal_type"],
    )
    # shuffle=False and num_workers=0 keep the sample order, the sampled goals and the
    # negatives identical across runs, which is what makes the paired comparison valid.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    return dataset, loader


def per_sample_metrics(actions, labels):
    """The `_compute_losses_nomad` formulas, kept per-sample."""
    squared_error = F.mse_loss(actions, labels, reduction="none")
    while squared_error.dim() > 1:
        squared_error = squared_error.mean(dim=-1)
    cos_sim = F.cosine_similarity(
        actions[:, :, :2], labels[:, :, :2], dim=-1
    ).mean(dim=-1)
    return squared_error, cos_sim


def _seed_sampler(seed, batch_index):
    """Pin the reverse-diffusion noise for one batch.

    `model_output` draws its starting noise and its ancestral noise from the global RNG,
    so seeding it here fixes the sampler for every run alike. The draws depend on how
    samples are grouped, so the batch size is part of the scoring recipe and is recorded
    in the manifest; `*_diffusion_loss` carries no such dependency.
    """
    torch.manual_seed(seed + batch_index)


def _fixed_noise(first_index, count, shape, seed):
    """One fixed noise tensor per sample, keyed on the sample's position in the split.

    Keying on the global index rather than on the RNG stream makes a sample's noise the
    same no matter which batch it lands in or how many runs were scored before it -- so
    the denoising loss is a property of the sample and the checkpoint alone.
    """
    noise = torch.empty((count, *shape))
    for row in range(count):
        generator = torch.Generator().manual_seed(seed + first_index + row)
        noise[row] = torch.randn(shape, generator=generator)
    return noise


def _denoising_loss(model, noise_scheduler, cond, naction, noise):
    """Mean noise-prediction error over every diffusion timestep, per sample.

    This is the objective the model was trained on (`diffusion_loss` in `train_nomad`),
    except that training draws one random timestep per sample where this averages over
    all of them with the noise held fixed -- which is what makes it a deterministic,
    low-variance measure of the same thing.
    """
    total = torch.zeros(naction.shape[0], device=naction.device)
    for step in range(noise_scheduler.config.num_train_timesteps):
        timesteps = torch.full(
            (naction.shape[0],), step, device=naction.device, dtype=torch.long
        )
        noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)
        noise_pred = model(
            "noise_pred_net",
            sample=noisy_action,
            timestep=timesteps,
            global_cond=cond,
        )
        total += F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2))
    return total / noise_scheduler.config.num_train_timesteps


def score_checkpoint(
    checkpoint_path, config, loader, device, seed, max_batches=None, desc=None
):
    """Per-sample scores for one checkpoint on one loader. Deterministic."""
    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    results = {name: [] for name in PER_SAMPLE_ARRAYS}

    # Pinned here, immediately before iteration: the dataset samples goals and negatives
    # with numpy at __getitem__ time, so this fixes the inputs identically for every run.
    np.random.seed(seed)

    batches = tqdm.tqdm(
        loader,
        desc=desc or os.path.basename(checkpoint_path),
        total=max_batches or len(loader),
        dynamic_ncols=True,
    )
    first_index = 0
    with torch.no_grad():
        for batch_index, data in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            obs_image, goal_image, actions, distance, _, _, action_mask = data

            obs_images = torch.split(obs_image, 3, dim=1)
            batch_obs_images = torch.cat(
                [IMAGENET_TRANSFORM(obs) for obs in obs_images], dim=1
            ).to(device)
            batch_goal_images = IMAGENET_TRANSFORM(goal_image).to(device)
            labels = actions.to(device)
            batch_size = labels.shape[0]

            # The diffusion model works on normalised position deltas, not on the
            # waypoints the dataset returns -- same conversion as `train_nomad`.
            naction = torch.from_numpy(
                normalize_data(get_delta(actions.numpy()), ACTION_STATS)
            ).float().to(device)
            noise = _fixed_noise(
                first_index, batch_size, tuple(naction.shape[1:]), seed
            ).to(device)

            # Goal masked (uc) vs goal visible (gc), as in `model_output`.
            masked_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=torch.ones(batch_size, dtype=torch.long, device=device),
            )
            visible_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=torch.zeros(batch_size, dtype=torch.long, device=device),
            )
            uc_diffusion = _denoising_loss(
                model, noise_scheduler, masked_cond, naction, noise
            )
            gc_diffusion = _denoising_loss(
                model, noise_scheduler, visible_cond, naction, noise
            )

            _seed_sampler(seed, batch_index)
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

            batch_results = {
                "gc_diffusion_loss": gc_diffusion,
                "gc_action_loss": gc_loss,
                "gc_action_waypts_cos_sim": gc_cos,
                "uc_diffusion_loss": uc_diffusion,
                "uc_action_loss": uc_loss,
                "uc_action_waypts_cos_sim": uc_cos,
                "gc_dist_loss": dist_error,
            }
            for name, values in batch_results.items():
                results[name].append(values.cpu().numpy())
            results["action_mask"].append(action_mask.numpy())
            first_index += batch_size

    return {name: np.concatenate(values) for name, values in results.items()}


def mean_and_standard_error(values):
    """Mean of a per-sample array, with the standard error of that mean."""
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def paired_delta(scores, arm, reference, metric):
    """Per-sample difference between two runs on one metric, with its own SE.

    Taken from the per-sample arrays rather than by subtracting two marginal SEs: the
    error on a difference depends on the covariance between the two runs, which is large
    here precisely because they were scored on identical samples. Subtracting marginal
    SEs would overstate it by roughly a factor of two.
    """
    return mean_and_standard_error(scores[arm][metric] - scores[reference][metric])


def checkpoint_digest(path):
    """sha256 of the weights, so a score can always be traced back to what produced it."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_summary(config):
    """One-line description of the input recipe a run was trained and scored at."""
    context_size = config["context_size"]
    context_stride = config["context_stride"]
    accumulation = config["gradient_accumulation_steps"]
    return (
        f"context_size={context_size}, context_stride={context_stride}, "
        f"frames={context_size + 1}, window={context_size * context_stride}, "
        f"image_size={config['image_size'][0]}x{config['image_size'][1]}, "
        f"eff_batch={config['batch_size'] * accumulation} "
        f"({config['batch_size']}x{accumulation}), "
        f"epochs={config['epochs']}, lr={config['lr']}, seed={config['seed']}"
    )


def write_arm_scores(out_dir, name, scores, manifest):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{name}.npz"), **scores)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def parse_arm(spec):
    """`NAME=CONFIG_PATH=CHECKPOINT_PATH` -> the three parts."""
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise SystemExit(
            f"FAIL: --arm {spec!r} is not NAME=CONFIG_PATH=CHECKPOINT_PATH"
        )
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=CONFIG=CHECKPOINT",
        help="Repeatable, e.g. --arm vanilla=config/nomad.yaml=/outputs/.../ema_99.pth",
    )
    parser.add_argument("--out", required=True, help="directory for the per-sample scores")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the EVAL seed -- fixes goal sampling, sampler noise and denoising noise. "
        "Unrelated to a run's training seed, and held at 0 for every reported result, "
        "so that all runs are scored on one identical set of inputs.",
    )
    parser.add_argument(
        "--index-context-size",
        type=int,
        default=ALIGNED_INDEX_CONTEXT_SIZE,
        help="leading timesteps skipped when indexing; held equal across runs",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="score only the first N batches (for the determinism check, not for results)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-score arms that already have scores in --out",
    )
    args = parser.parse_args()

    enforce_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for spec in args.arm:
        name, config_path, checkpoint_path = parse_arm(spec)
        score_path = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(score_path) and not args.overwrite:
            print(f"{name}: already scored at {score_path} (--overwrite to redo)")
            continue

        config = load_arm_config(config_path, args.index_context_size)
        dataset, loader = build_test_loader(config, args.batch_size)
        try:
            scores = score_checkpoint(
                checkpoint_path,
                config,
                loader,
                device,
                args.seed,
                max_batches=args.max_batches,
                desc=name,
            )
        finally:
            # Runs are scored one after another in this process, and each needs its own
            # dataset because the input shape differs. LMDB reader slots are per-process,
            # so the previous run's cache has to be released before the next one opens
            # the same file.
            dataset.close()

        manifest = {
            "arm": name,
            "config_path": os.path.abspath(config_path),
            "checkpoint": os.path.abspath(checkpoint_path),
            "checkpoint_sha256": checkpoint_digest(checkpoint_path),
            "recipe": recipe_summary(config),
            "index_context_size": args.index_context_size,
            "n_samples": int(len(scores["gc_action_loss"])),
            "batch_size": args.batch_size,
            "seed": args.seed,
            "max_batches": args.max_batches,
        }
        write_arm_scores(args.out, name, scores, manifest)
        print(f"{name}: {manifest['n_samples']:,} samples -> {score_path}")
        print(f"  recipe: {manifest['recipe']}")
        for metric in METRICS:
            print(f"  {metric:<28} {scores[metric].mean():.6f}")


if __name__ == "__main__":
    main()
