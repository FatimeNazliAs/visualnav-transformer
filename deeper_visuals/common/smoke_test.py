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

from deeper_visuals.common import denoise, settings
from deeper_visuals.common.config import config_from_dict
from deeper_visuals.common.data import load_sample, to_model_input
from deeper_visuals.common.model import (
    GOAL_HIDDEN, GOAL_VISIBLE, encode_tokens, load_model)

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

    print("\n── Forward pass: encoders + transformer -> c_t ──────────────────")
    # Through encode_tokens, the same path every phase uses. Building the masks
    # and calling the model by hand here — as this test used to — meant the one
    # genuinely deep function in common/ was the only thing the smoke test never
    # touched, and its docstring says its frame-major unpack is "impossible to
    # notice when B is 1". That is precisely what a smoke test should cover.
    #
    # goal mask 0 = goal token visible          -> navigation
    # goal mask 1 = goal token hidden from attention -> exploration
    #
    # Hidden, not zeroed. NoMaD_ViNT.forward selects a src_key_padding_mask so
    # the goal token is excluded from attention entirely — it is still encoded,
    # the other tokens just cannot see it. The mean-pool is then rescaled over
    # the 4 remaining tokens.
    nav = encode_tokens(model, obs_input[None], goal_input[None], device,
                        goal_mask=GOAL_VISIBLE)
    exp = encode_tokens(model, obs_input[None], goal_input[None], device,
                        goal_mask=GOAL_HIDDEN)

    print(f"  obs tokens  : {nav.obs_tokens.shape}")
    print(f"  goal token  : {nav.goal_token.shape}")
    print(f"  tokens in   : {nav.tokens.shape}   (what the transformer sees)")
    print(f"  c_t (navigation) : {nav.context.shape}")
    print(f"  c_t (exploration): {exp.context.shape}")

    if not np.array_equal(nav.tokens, exp.tokens):
        raise SystemExit(
            "  FAIL — the goal mask changed the encoder tokens, which it cannot "
            "do. Masking hides the goal from attention, not from the encoder."
        )
    print("  Masking left the tokens untouched -> it acts on attention only")

    delta = float(np.abs(nav.context - exp.context).mean())
    print(f"  mean |nav - explore| : {delta:.4f}")
    if delta == 0.0:
        raise SystemExit("  FAIL — goal masking had no effect on c_t")
    print("  Goal masking changes c_t -> masking is wired up correctly")

    print("\n── Forward pass: diffusion head ────────────────────────────────")
    # Through common/denoise.py rather than a private copy of the K-step loop.
    result = denoise.denoise(model, nav.context, device, seed=0)
    actions = result.actions

    print(f"  denoised actions : {actions.shape}  "
          f"({settings.NUM_ACTIONS} steps x {settings.ACTION_DIM} dims, "
          f"K={result.n_steps})")
    print(f"  action range     : [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  distance head    : "
          f"{denoise.distance_to_goal(model, nav.context, device):.3f}")
    if not np.isfinite(actions).all():
        raise SystemExit("  FAIL — denoised actions contain NaN/Inf")
    if result.n_steps != settings.K_DENOISING:
        raise SystemExit(
            f"  FAIL — expected {settings.K_DENOISING} denoising steps, "
            f"got {result.n_steps}")

    print(f"\n{SEP}")
    print("  PASS — checkpoint loads and a full forward pass runs clean.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
