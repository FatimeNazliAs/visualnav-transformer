# debug/run_pipeline.py
"""
Entry point for the NoMaD inference pipeline debug trace.

Run from the repo root inside the container:
    conda activate vint_train
    cd /app/visualnav-transformer
    python -m debug.run_pipeline

To debug a different sample, edit TRAJ_NAME and FRAME_IDX in debug/config.py.
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from debug.config import (
    CHECKPOINT, DEVICE, CONTEXT_SIZE, NUM_ACTIONS,
    ENCODING_SIZE, DOWN_DIMS, K_DENOISING
)
from debug import stage1_data, stage2_obs_enc, stage3_goal_enc
from debug import stage4_transformer, stage5_diffusion

SEP = "═" * 60


def load_model():
    print(f"  Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    # Checkpoint may store state_dict directly or nested under a key
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=ENCODING_SIZE,
        context_size=CONTEXT_SIZE,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=ENCODING_SIZE,
        down_dims=DOWN_DIMS,
        cond_predict_scale=False,
    )
    dist_pred_net = DenseNetwork(embedding_dim=ENCODING_SIZE)

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,

    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING — missing keys  : {missing[:3]} ...")
    if unexpected:
        print(f"  WARNING — unexpected keys: {unexpected[:3]} ...")

    model = model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M")
    print(f"  Device     : {DEVICE}")
    return model


def main():
    print(f"\n{SEP}\n  NoMaD Inference Pipeline — debug trace\n{SEP}")

    model = load_model()

    print(f"\n── Stage 1: Data loading ──────────────────────────────")
    obs, goal = stage1_data.load_one_sample()

    print(f"\n── Stage 2: Observation encoder ψ ─────────────────────")
    obs_tokens = stage2_obs_enc.run_obs_encoder(model, obs)

    print(f"\n── Stage 3: Goal encoder φ ─────────────────────────────")
    goal_token = stage3_goal_enc.run_goal_encoder(model, obs, goal)

    print(f"\n── Stage 4: Transformer + Goal Masking ─────────────────")
    ct, dist = stage4_transformer.run_transformer(
        model, obs, goal, mask=False
    )
    stage4_transformer.compare_masked_vs_unmasked(model, obs, goal)

    print(f"\n── Stage 5: Diffusion denoising (K={K_DENOISING}) ─────────────")
    a_final = stage5_diffusion.run_diffusion(model, ct)

    print(f"\n── Final output ─────────────────────────────────────────")
    print(f"  Predicted action sequence (normalised velocities):")
    print(f"  {'step':>5}  {'linear_vel':>12}  {'angular_vel':>12}")
    print(f"  {'────':>5}  {'──────────':>12}  {'──────────':>12}")
    for h in range(NUM_ACTIONS):
        v  = a_final[0, h, 0].item()
        w  = a_final[0, h, 1].item()
        print(f"  {h:>5}  {v:>12.4f}  {w:>12.4f}")

    print(f"\n  dist_pred : {dist[0, 0].item():.2f} steps to goal")
    print(f"\n  Note: actions are normalised (normalize=True in yaml).")
    print(f"  Note: only step 0 would go to the robot in deployment.")
    print(f"  Run again → different action sequence (multimodality).\n")


if __name__ == "__main__":
    main()