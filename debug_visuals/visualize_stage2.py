# debug_visuals/visualize_stage2.py
"""
Stage 2 visualisation — "What do the encoders actually output?"

Loads the NoMaD checkpoint, runs encoder psi (obs_encoder) on the 4
observation frames and encoder phi (goal_encoder) on the goal frame,
then saves a bar chart of the resulting 256-dim tokens.

This module is a library of functions, driven by
debug_visuals/run_pipeline.py. It can also be run standalone:

    conda activate vint_train
    cd /app/visualnav-transformer
    python -m debug_visuals.visualize_stage2

To visualise a different sample, edit TRAJ_NAME and FRAME_IDX in
debug_visuals/config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed inside the container
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Make the repo root importable when running as `python -m debug_visuals…` ──
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "train"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from debug_visuals.config import (
    CHECKPOINT,
    TRAJ_NAME,
    FRAME_IDX,
    CONTEXT_SIZE,
    NUM_ACTIONS,
    ENCODING_SIZE,
    DOWN_DIMS,
    DEVICE,
    OUTPUTS_DIR,
)
from debug_visuals import visualize_stage1

SEP = "─" * 60


# ══════════════════════════════════════════════════════════════════════════════
# Model loading — copied from debug/run_pipeline.py, unchanged
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Token extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_obs_tokens(model, obs_raw_list) -> np.ndarray:
    """
    Encode each of the CONTEXT_SIZE+1 observation frames into its
    256-dim token via encoder psi (model.vision_encoder.obs_encoder).

    Mirrors debug/stage2_obs_enc.py: calling backbone.obs_encoder(x)
    directly would run EfficientNet's own forward() into its built-in
    1000-way ImageNet classifier, not the 256-dim NoMaD embedding — so
    we stop after pooling and apply compress_obs_enc ourselves, exactly
    like NoMaD_ViNT.forward() does internally.

    Returns a numpy array of shape (4, 256).
    """
    backbone = model.vision_encoder

    frames_norm = [visualize_stage1.normalise_frame(f) for f in obs_raw_list]
    obs_tensor = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1).float() for f in frames_norm
    ]).unsqueeze(0).to(DEVICE)                              # (1, 4, 3, 96, 96)

    B, P, C, H, W = obs_tensor.shape
    obs_flat = obs_tensor.view(B * P, C, H, W)              # (4, 3, 96, 96)

    with torch.no_grad():
        obs_encoded = backbone.obs_encoder.extract_features(obs_flat)
        obs_encoded = backbone.obs_encoder._avg_pooling(obs_encoded)
        obs_encoded = obs_encoded.flatten(start_dim=1)      # (4, 1280)
        obs_encoded = backbone.obs_encoder._dropout(obs_encoded)
        obs_encoded = backbone.compress_obs_enc(obs_encoded)  # (4, 256)

    return obs_encoded.view(P, ENCODING_SIZE).cpu().numpy()  # (4, 256)


def extract_goal_token(model, obs_raw_list, goal_raw) -> np.ndarray:
    """
    Encode the goal frame into a single 256-dim token via encoder phi
    (model.vision_encoder.goal_encoder).

    The goal encoder is NOT goal-only: NoMaD_ViNT builds it with
    in_channels=6 because it jointly encodes [most-recent obs frame ++
    goal frame] concatenated channel-wise (3+3). Mirrors
    debug/stage3_goal_enc.py — this is the token masked out when
    mask=True (exploration mode).

    Returns a numpy array of shape (256,).
    """
    backbone = model.vision_encoder

    last_obs_norm = visualize_stage1.normalise_frame(obs_raw_list[-1])
    goal_norm     = visualize_stage1.normalise_frame(goal_raw)

    last_obs_tensor = torch.from_numpy(last_obs_norm).permute(2, 0, 1).float()
    goal_tensor     = torch.from_numpy(goal_norm).permute(2, 0, 1).float()

    obsgoal_img = torch.cat([last_obs_tensor, goal_tensor], dim=0) \
                       .unsqueeze(0).to(DEVICE)             # (1, 6, 96, 96)

    with torch.no_grad():
        goal_token = backbone.goal_encoder.extract_features(obsgoal_img)
        goal_token = backbone.goal_encoder._avg_pooling(goal_token)
        goal_token = goal_token.flatten(start_dim=1)        # (1, 1280)
        goal_token = backbone.goal_encoder._dropout(goal_token)
        goal_token = backbone.compress_goal_enc(goal_token)  # (1, 256)

    return goal_token.squeeze(0).cpu().numpy()               # (256,)


# ══════════════════════════════════════════════════════════════════════════════
# Plot — token bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_token_barchart(obs_tokens: np.ndarray, goal_token: np.ndarray, save_path: Path) -> None:
    """
    Legacy bar-chart view — kept for standalone debugging but no longer
    called by run_pipeline.py. Use plot_token_heatmap() instead.
    """
    obs_idxs = list(range(FRAME_IDX - CONTEXT_SIZE, FRAME_IDX + 1))
    goal_idx = FRAME_IDX + NUM_ACTIONS

    n_panels = len(obs_tokens) + 1   # 4 obs + 1 goal

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(3.2 * n_panels, 4.2),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for i, token in enumerate(obs_tokens):
        ax = axes[i]
        ax.bar(range(256), token, color="#3498db", linewidth=0)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"obs[{i}]\nframe {obs_idxs[i]}", fontsize=9)
        ax.set_xlabel("dim", fontsize=8)
        ax.tick_params(labelsize=7)

    ax_goal = axes[n_panels - 1]
    ax_goal.bar(range(256), goal_token, color="#8e44ad", linewidth=0)
    ax_goal.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax_goal.set_title(f"GOAL\nframe {goal_idx}", fontsize=9, color="#8e44ad", fontweight="bold")
    ax_goal.set_xlabel("dim", fontsize=8)
    ax_goal.tick_params(labelsize=7)

    axes[0].set_ylabel("activation value", fontsize=9)

    fig.suptitle(
        f"Encoder output tokens — {TRAJ_NAME} t={FRAME_IDX}",
        fontsize=11,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


def plot_token_heatmap(obs_tokens: np.ndarray, goal_token: np.ndarray,
                       obs_raw_list: list, goal_raw: np.ndarray,
                       save_path: Path) -> None:
    """
    Frame thumbnails → heatmap strips: shows the image-to-vector
    transformation at a glance. Each row is one token; left column shows
    the original frame, right column shows its 256-dim embedding as a
    single-row heatmap.
    """
    import matplotlib.gridspec as gridspec

    obs_idxs = list(range(FRAME_IDX - CONTEXT_SIZE, FRAME_IDX + 1))
    goal_idx = FRAME_IDX + NUM_ACTIONS

    all_tokens = list(obs_tokens) + [goal_token]
    all_frames = obs_raw_list + [goal_raw]
    all_labels = [f"obs[{i}] (frame {obs_idxs[i]})" for i in range(len(obs_idxs))]
    all_labels.append(f"GOAL (frame {goal_idx})")
    is_goal = [False] * len(obs_tokens) + [True]

    n_rows = len(all_tokens)
    vmin = min(t.min() for t in all_tokens)
    vmax = max(t.max() for t in all_tokens)
    vabs = max(abs(vmin), abs(vmax))

    fig = plt.figure(figsize=(14, 1.8 * n_rows + 1.5))
    gs = gridspec.GridSpec(n_rows, 2, width_ratios=[1, 6], wspace=0.15, hspace=0.4)

    for row_idx in range(n_rows):
        token = all_tokens[row_idx]
        frame = all_frames[row_idx]
        label = all_labels[row_idx]
        goal_flag = is_goal[row_idx]

        ax_img = fig.add_subplot(gs[row_idx, 0])
        ax_img.imshow(frame)
        ax_img.axis("off")
        label_color = "#8e44ad" if goal_flag else "#2c3e50"
        ax_img.set_title(label, fontsize=9, color=label_color,
                         fontweight="bold" if goal_flag else "normal", pad=2)

        ax_heat = fig.add_subplot(gs[row_idx, 1])
        im = ax_heat.imshow(
            token.reshape(1, -1), aspect="auto", cmap="RdBu_r",
            vmin=-vabs, vmax=vabs,
        )
        ax_heat.set_yticks([])
        ax_heat.set_xlabel("embedding dimension" if row_idx == n_rows - 1 else "", fontsize=8)
        ax_heat.tick_params(labelsize=7)

        arrow_color = "#8e44ad" if goal_flag else "#3498db"
        ax_img.annotate(
            "", xy=(1.15, 0.5), xycoords="axes fraction",
            xytext=(1.02, 0.5), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
        )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="activation value")

    fig.suptitle(
        f"Visual Encoders: Images → 256-dim Token Embeddings\n"
        f"Trajectory: {TRAJ_NAME}  frame t={FRAME_IDX}",
        fontsize=12, y=1.0,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{SEP}")
    print("  NoMaD — Stage 2 Visualisation (encoder tokens)")
    print(SEP)

    print("\n[1/4] Loading model …")
    model = load_model()

    print("\n[2/4] Loading frames from disk …")
    sample = visualize_stage1.load_sample_frames()

    print("\n[3/4] Running encoders …")
    obs_tokens = extract_obs_tokens(model, sample["obs_raw"])
    goal_token = extract_goal_token(model, sample["obs_raw"], sample["goal_raw"])
    print(f"  obs_tokens : {obs_tokens.shape}")
    print(f"  goal_token : {goal_token.shape}")

    print("\n[4/4] Saving bar chart …")
    run_dir = OUTPUTS_DIR / f"{TRAJ_NAME}_f{FRAME_IDX}"
    plot_token_barchart(obs_tokens, goal_token, save_path=run_dir / "stage2_obs_tokens.png")

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    main()
