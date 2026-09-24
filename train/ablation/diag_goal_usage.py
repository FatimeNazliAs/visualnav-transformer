"""Phase 2 diagnostic: does the policy use its goal at all -- and if not, why?

The sanity check (sanity_clip_gap.py) found the CLIP-goal model's actions nearly identical
with the goal visible, fed as a word, or masked, and its distance head nearly constant.
This separates two explanations:

  (a) goal sensitivity. The same measurement for vanilla V1 and the CLIP model: if V1's
      actions also barely change when its goal is masked, GoStanford is predictable from
      the camera frames alone; if V1's do change, the CLIP goal token specifically is not
      contributing. Measured on the 20 sanity rows (goal 10 steps ahead) and on a seeded
      sample of the test split (every goal distance, plus negatives).
  (b) goal-token scale. Norm of the goal token next to the observation tokens and the
      positional encoding that is added to it before the transformer's pre-norm
      LayerNorm, and the first layer's attention onto the goal position. A goal token far
      smaller than the positional encoding reaches attention as little more than its
      position, whatever its content.

Read-only apart from its own output folder. Run inside the container, from
/app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 python ablation/diag_goal_usage.py
"""
# Must be set before torch creates its cuBLAS handle; see eval_paired.py.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader, Subset

from eval_paired import (
    IMAGENET_TRANSFORM,
    build_model,
    build_test_loader,
    enforce_determinism,
    load_arm_config,
)
from sanity_clip_gap import load_case_inputs, read_cases, sample_waypoints

ARMS = {
    "v1": ("config/nomad_ctx03.yaml",
           "/outputs/nomad_ctx_ablation/ctx03_2026_08_31_17_47_18/ema_29.pth"),
    "clip": ("config/nomad_clip.yaml",
             "/outputs/nomad_clip_v2/clip_vitb32_2026_09_23_17_58_18/ema_29.pth"),
}
DEFAULT_OUTPUT_DIR = "/outputs/nomad_clip_v2/diag_goal_usage"


def goal_vec_or_none(vec):
    """Image-goal datasets return an empty goal_vec; the model wants None for those."""
    return vec if vec.numel() > 0 else None


def mean_l2(a, b):
    """Mean per-waypoint L2 between two [..., T, 2] trajectories, per sample."""
    return np.linalg.norm(a - b, axis=-1).mean(axis=-1)


class TokenProbe:
    """Records the token norms entering the transformer and layer-0 attention to the goal."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.obs_norms, self.goal_norms, self.goal_attention = [], [], []
        goal_module = encoder.clip_goal_proj if encoder.goal_type == "clip" else encoder.compress_goal_enc
        encoder.compress_obs_enc.register_forward_hook(
            lambda m, i, out: self.obs_norms.append(out.norm(dim=-1).flatten().cpu()))
        goal_module.register_forward_hook(
            lambda m, i, out: self.goal_norms.append(out.norm(dim=-1).flatten().cpu()))
        encoder.positional_encoding.register_forward_hook(self._attention_hook)
        self.active = False

    def _attention_hook(self, module, inputs, tokens):
        if not self.active:
            return
        layer = self.encoder.sa_encoder.layers[0]
        x = layer.norm1(tokens)  # norm_first=True: attention sees the normalised tokens
        _, weights = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
        # Attention mass the observation positions put on the last (goal) position.
        self.goal_attention.append(weights[:, :-1, -1].mean(dim=-1).cpu())

    def summary(self):
        obs, goal = torch.cat(self.obs_norms), torch.cat(self.goal_norms)
        pe = self.encoder.positional_encoding.pos_enc[0, -1].norm().item()
        attention = torch.cat(self.goal_attention) if self.goal_attention else torch.zeros(1)
        return {
            "obs_token_norm": float(obs.mean()),
            "goal_token_norm": float(goal.mean()),
            "goal_token_norm_std": float(goal.std()),
            "goal_position_pe_norm": pe,
            "goal_over_pe": float(goal.mean()) / pe,
            "goal_over_obs": float(goal.mean() / obs.mean()),
            "layer0_attention_on_goal": float(attention.mean()),
            "uniform_attention": 1.0 / self.encoder.positional_encoding.pos_enc.shape[1],
        }


def load_arm(name, device):
    config_path, checkpoint = ARMS[name]
    config = load_arm_config(config_path)
    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return config, model


def make_scheduler(config):
    return DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )


@torch.no_grad()
def sanity_rows(model, config, scheduler, cases, goal_offset, seed, device):
    """Goal visible vs masked on the sanity rows, same frames and noise as sanity_clip_gap."""
    dataset, _ = build_test_loader(config, batch_size=1)
    visible = torch.zeros(1, dtype=torch.long, device=device)
    masked = torch.ones(1, dtype=torch.long, device=device)
    diffs, gc_err, uc_err, dist = [], [], [], []
    try:
        for i, case in enumerate(cases):
            obs, goal_image, vec, gt = load_case_inputs(dataset, case, goal_offset, device)
            vec = goal_vec_or_none(vec)
            gc_cond = model("vision_encoder", obs_img=obs, goal_img=goal_image, input_goal_mask=visible, goal_vec=vec)
            uc_cond = model("vision_encoder", obs_img=obs, goal_img=goal_image, input_goal_mask=masked, goal_vec=vec)
            gc = sample_waypoints(model, scheduler, gc_cond, config["len_traj_pred"], seed + i, device)
            uc = sample_waypoints(model, scheduler, uc_cond, config["len_traj_pred"], seed + i, device)
            diffs.append(mean_l2(gc, uc))
            gc_err.append(mean_l2(gc, gt))
            uc_err.append(mean_l2(uc, gt))
            dist.append(model("dist_pred_net", obsgoal_cond=gc_cond).item())
    finally:
        dataset.close()
    return {
        "gc_vs_uc": float(np.mean(diffs)),
        "gc_err": float(np.mean(gc_err)),
        "uc_err": float(np.mean(uc_err)),
        "dist_pred_mean": float(np.mean(dist)),
        "dist_pred_std": float(np.std(dist)),
    }


@torch.no_grad()
def test_sample(model, config, scheduler, n_samples, batch_size, seed, device, probe):
    """Goal visible vs masked on a seeded sample of the test split (all goal distances)."""
    dataset, _ = build_test_loader(config, batch_size)
    indices = np.random.RandomState(seed).choice(len(dataset), n_samples, replace=False)
    loader = DataLoader(Subset(dataset, sorted(indices)), batch_size=batch_size, shuffle=False, num_workers=0)
    np.random.seed(seed)  # goals and negatives are sampled at __getitem__ time
    per = {k: [] for k in ("gc_vs_uc", "gc_err", "uc_err", "action_mask", "dist_pred", "distance")}
    try:
        for batch_index, data in enumerate(loader):
            obs_image, goal_image, actions, distance, _, _, action_mask, goal_vec = data
            obs = torch.cat([IMAGENET_TRANSFORM(o) for o in torch.split(obs_image, 3, dim=1)], dim=1).to(device)
            goal = IMAGENET_TRANSFORM(goal_image).to(device)
            vec = goal_vec_or_none(goal_vec.to(device))
            b = obs.shape[0]
            probe.active = True
            gc_cond = model("vision_encoder", obs_img=obs, goal_img=goal,
                            input_goal_mask=torch.zeros(b, dtype=torch.long, device=device), goal_vec=vec)
            probe.active = False
            uc_cond = model("vision_encoder", obs_img=obs, goal_img=goal,
                            input_goal_mask=torch.ones(b, dtype=torch.long, device=device), goal_vec=vec)
            gc = batch_waypoints(model, scheduler, gc_cond, config["len_traj_pred"], seed + batch_index, device)
            uc = batch_waypoints(model, scheduler, uc_cond, config["len_traj_pred"], seed + batch_index, device)
            gt = actions[:, :, :2].numpy().astype(np.float64)
            per["gc_vs_uc"].append(mean_l2(gc, uc))
            per["gc_err"].append(mean_l2(gc, gt))
            per["uc_err"].append(mean_l2(uc, gt))
            per["action_mask"].append(action_mask.numpy())
            per["dist_pred"].append(model("dist_pred_net", obsgoal_cond=gc_cond).squeeze(-1).cpu().numpy())
            per["distance"].append(distance.numpy().astype(np.float64))
    finally:
        dataset.close()
    per = {k: np.concatenate(v) for k, v in per.items()}
    valid = per["action_mask"] > 0
    return {
        "n": int(len(per["gc_vs_uc"])),
        "n_action_valid": int(valid.sum()),
        "gc_vs_uc": float(per["gc_vs_uc"][valid].mean()),
        "gc_err": float(per["gc_err"][valid].mean()),
        "uc_err": float(per["uc_err"][valid].mean()),
        "dist_pred_std": float(per["dist_pred"].std()),
        "dist_true_std": float(per["distance"].std()),
        "dist_corr": float(np.corrcoef(per["dist_pred"], per["distance"])[0, 1]),
        "dist_mse": float(((per["dist_pred"] - per["distance"]) ** 2).mean()),
    }


def batch_waypoints(model, scheduler, cond, pred_horizon, seed, device):
    """sample_waypoints for a whole batch: [B, T, 2]."""
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn((cond.shape[0], pred_horizon, 2), generator=generator, device=device)
    from vint_train.training.train_utils import ACTION_STATS, get_action
    for k in scheduler.timesteps:
        noise_pred = model("noise_pred_net", sample=sample,
                           timestep=k.unsqueeze(-1).repeat(sample.shape[0]).to(device), global_cond=cond)
        sample = scheduler.step(model_output=noise_pred, timestep=k, sample=sample, generator=generator).prev_sample
    return get_action(sample, ACTION_STATS).cpu().numpy().astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="ablation/sanity_words.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--goal-offset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary_path = os.path.join(args.output_dir, "summary.txt")
    if os.path.exists(summary_path):
        raise SystemExit(f"FAIL: {summary_path} exists")

    enforce_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = read_cases(args.csv)

    results = {}
    for name in ARMS:
        config, model = load_arm(name, device)
        scheduler = make_scheduler(config)
        probe = TokenProbe(model.vision_encoder)
        results[name] = {
            "sanity_rows": sanity_rows(model, config, scheduler, cases, args.goal_offset, args.seed, device),
            "test_sample": test_sample(model, config, scheduler, args.n_samples, args.batch_size,
                                       args.seed, device, probe),
            "tokens": probe.summary(),
        }
        del model
        torch.cuda.empty_cache()

    lines = [
        f"arms: " + "; ".join(f"{k}={v[1]}" for k, v in ARMS.items()),
        f"units: normalised waypoints (1 = 0.12 m); gc = goal visible, uc = goal masked",
        "",
        f"{'':<34} {'v1':>10} {'clip':>10}",
    ]

    def row(label, section, key, fmt="{:>10.3f}"):
        lines.append(f"{label:<34} " + " ".join(fmt.format(results[a][section][key]) for a in ARMS))

    lines.append(f"(a) sanity rows ({len(cases)}, goal {args.goal_offset} steps ahead)")
    row("  gc vs uc waypoint L2", "sanity_rows", "gc_vs_uc")
    row("  gc error vs GT", "sanity_rows", "gc_err")
    row("  uc error vs GT", "sanity_rows", "uc_err")
    row("  dist_pred mean", "sanity_rows", "dist_pred_mean")
    row("  dist_pred std", "sanity_rows", "dist_pred_std")
    lines.append(f"(a') test sample (n={args.n_samples}, seed {args.seed}; action metrics on action-valid only)")
    row("  action-valid samples", "test_sample", "n_action_valid", "{:>10d}")
    row("  gc vs uc waypoint L2", "test_sample", "gc_vs_uc")
    row("  gc error vs GT", "test_sample", "gc_err")
    row("  uc error vs GT", "test_sample", "uc_err")
    row("  dist_pred std (true std)", "test_sample", "dist_pred_std")
    lines[-1] += f"   (true {results['v1']['test_sample']['dist_true_std']:.3f})"
    row("  corr(dist_pred, true distance)", "test_sample", "dist_corr")
    row("  dist MSE", "test_sample", "dist_mse")
    lines.append("(b) token scale (test sample, goal visible)")
    row("  obs token norm", "tokens", "obs_token_norm")
    row("  goal token norm", "tokens", "goal_token_norm")
    row("  goal token norm std", "tokens", "goal_token_norm_std")
    row("  pos-enc norm at goal position", "tokens", "goal_position_pe_norm")
    row("  goal / pos-enc", "tokens", "goal_over_pe")
    row("  goal / obs", "tokens", "goal_over_obs")
    row("  layer-0 attention on goal", "tokens", "layer0_attention_on_goal")
    row("  (uniform attention)", "tokens", "uniform_attention")
    summary = "\n".join(lines)
    print(summary)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
