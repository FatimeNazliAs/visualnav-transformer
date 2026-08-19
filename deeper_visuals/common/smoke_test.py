# deeper_visuals/common/smoke_test.py
"""
Phase S smoke test — prove the scaffold can do a real forward pass.

Loads the confirmed checkpoint, pushes one scene through the full NoMaD stack
(psi/phi encoders -> transformer -> context vector -> K-step diffusion), and
prints the shapes at each boundary. No figures, no page — this only answers
"is the plumbing sound?" so that P1–P5 can assume it is.

Run inside the container:

    source /opt/conda/etc/profile.d/conda.sh && conda activate vint_train
    cd /app/visualnav-transformer
    python -m deeper_visuals.common.smoke_test
    python -m deeper_visuals.common.smoke_test --sample left_turn --checkpoint ema
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from deeper_visuals.common import settings
from deeper_visuals.common.config import config_from_dict
from deeper_visuals.common.data import load_sample, to_model_input
from deeper_visuals.common.model import load_model

SEP = "─" * 66


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="deeper_visuals scaffold smoke test")
    ap.add_argument("--sample", default="gentle_left",
                    help="a key from common/samples.yaml (default: gentle_left)")
    ap.add_argument("--checkpoint", default=settings.DEFAULT_CHECKPOINT,
                    choices=sorted(settings.CHECKPOINT_FILES),
                    help=f"which weight file to load "
                         f"(default: {settings.DEFAULT_CHECKPOINT})")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_dict(
        {"sample": args.sample, "checkpoint": args.checkpoint}, phase="smoke"
    )

    print(f"\n{SEP}")
    print("  deeper_visuals — scaffold smoke test")
    print(f"  {cfg.summary()}")
    print(f"  {cfg.description}")
    print(SEP)

    print("\n── Data ────────────────────────────────────────────────────────")
    sample = load_sample(cfg)
    obs_input  = to_model_input(sample["obs_raw"])          # (12, 96, 96)
    goal_input = to_model_input([sample["goal_raw"]])       # (3, 96, 96)
    print(f"  obs tensor  : {obs_input.shape}   (3 channels x {settings.N_OBS_FRAMES} frames)")
    print(f"  goal tensor : {goal_input.shape}")

    print("\n── Model ───────────────────────────────────────────────────────")
    model, info = load_model(cfg)
    device = info["device"]
    if info["missing_keys"] or info["unexpected_keys"]:
        raise SystemExit(
            f"  FAIL — state dict did not load cleanly: "
            f"{info['missing_keys']} missing, {info['unexpected_keys']} unexpected"
        )
    print("  State dict : loaded cleanly (0 missing, 0 unexpected)")

    obs_t  = torch.from_numpy(obs_input).unsqueeze(0).to(device)
    goal_t = torch.from_numpy(goal_input).unsqueeze(0).to(device)

    print("\n── Forward pass: encoders + transformer -> c_t ──────────────────")
    with torch.no_grad():
        # goal mask 0 = goal token visible (navigation)
        # goal mask 1 = goal token zeroed  (exploration)
        nav_mask     = torch.zeros(1, dtype=torch.long, device=device)
        explore_mask = torch.ones(1, dtype=torch.long, device=device)

        ct_nav = model("vision_encoder", obs_img=obs_t, goal_img=goal_t,
                       input_goal_mask=nav_mask)
        ct_exp = model("vision_encoder", obs_img=obs_t, goal_img=goal_t,
                       input_goal_mask=explore_mask)

    print(f"  c_t (navigation) : {tuple(ct_nav.shape)}")
    print(f"  c_t (exploration): {tuple(ct_exp.shape)}")
    delta = (ct_nav - ct_exp).abs().mean().item()
    print(f"  mean |nav - explore| : {delta:.4f}")
    if delta == 0.0:
        raise SystemExit("  FAIL — goal masking had no effect on c_t")
    print("  Goal masking changes c_t -> masking is wired up correctly")

    print("\n── Forward pass: diffusion head ────────────────────────────────")
    scheduler = DDPMScheduler(
        num_train_timesteps=settings.K_DENOISING,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(settings.K_DENOISING)

    with torch.no_grad():
        naction = torch.randn(
            (1, settings.NUM_ACTIONS, settings.ACTION_DIM), device=device
        )
        for k in scheduler.timesteps:
            noise_pred = model(
                "noise_pred_net", sample=naction,
                timestep=k.unsqueeze(0).to(device), global_cond=ct_nav,
            )
            naction = scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            ).prev_sample

        dist = model("dist_pred_net", obsgoal_cond=ct_nav)

    actions = naction.cpu().numpy()[0]
    print(f"  denoised actions : {actions.shape}  "
          f"({settings.NUM_ACTIONS} steps x {settings.ACTION_DIM} dims, K={settings.K_DENOISING})")
    print(f"  action range     : [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  distance head    : {float(dist.item()):.3f}")
    if not np.isfinite(actions).all():
        raise SystemExit("  FAIL — denoised actions contain NaN/Inf")

    print(f"\n{SEP}")
    print("  PASS — checkpoint loads and a full forward pass runs clean.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
