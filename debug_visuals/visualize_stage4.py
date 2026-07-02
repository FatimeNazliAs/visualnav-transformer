# debug_visuals/visualize_stage4.py
"""
Stage 4 visualisation — "What does the Transformer + goal masking do?"

Runs the 4-layer self-attention encoder (model.vision_encoder.sa_encoder)
twice on the same 5 input tokens (obs[0..3] + goal): once in navigation
mode (goal token active) and once in exploration mode (goal token masked
via src_key_padding_mask). run_stage4() saves 2 presentation PNGs:

  1. stage4_attention.png       — Layer 4 attention: the 4-head overview
     (nav vs exploration) plus a Head 2 zoom, in one figure. Shows where
     attention collapses under masking and how the goal pathway is severed.
  2. stage4_ct_comparison.png   — the measurable effect on ct (the vector
     actually handed to the diffusion model), as stacked heatmap strips.

Both forward passes use the same checkpoint, same input frames, same
goal image — only the mask differs. That's intentional, not a bug: it's
how this script isolates the effect of masking.

Run from the repo root inside the container:

    conda activate vint_train
    cd /app/visualnav-transformer
    python -m debug_visuals.visualize_stage4

To visualise a different sample, edit TRAJ_NAME and FRAME_IDX in
debug_visuals/config.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed inside the container
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# Repo root / train are put on sys.path by debug_visuals/__init__.py.
from debug_visuals.config import (
    TRAJ_NAME,
    FRAME_IDX,
    ENCODING_SIZE,
    DEVICE,
    RUN_DIR,
)
from debug_visuals.model import load_model
from debug_visuals.viz_utils import save_fig
from debug_visuals import visualize_stage1
from debug_visuals import visualize_stage2

SEP = "─" * 60
TOKEN_LABELS = ["F0", "F1", "F2", "F3", "Goal"]
LAYER_IDX_FOR_PLOT = 3   # Layer 4 (0-indexed) — final, most-refined attention
HEAD_IDX_FOR_ZOOM = 1    # Head 2 (0-indexed) — the goal-direction specialist


def _subtitle(extra: str = "") -> str:
    return f"Trajectory: {TRAJ_NAME} | Frame: {FRAME_IDX}{extra}"


# ══════════════════════════════════════════════════════════════════════════════
# Attention capture
#
# Two things stand between us and real attention weights, both confirmed via
# inspect.getsource on this torch install before writing this code:
#
# 1. In eval mode with no_grad, TransformerEncoderLayer.forward() takes a
#    fused fast path (torch._transformer_encoder_layer_fwd) that bypasses
#    self_attn.forward() ENTIRELY — hooks on self_attn never fire. We disable
#    it for the duration of the pass via torch.backends.mha.set_fastpath_enabled
#    (restored afterwards), forcing the plain Python _sa_block path.
#
# 2. Once on that path, TransformerEncoderLayer._sa_block hardcodes
#    need_weights=False when it calls self_attn, so a plain forward_hook
#    would still only ever see attn_weights=None. We use a forward_pre_hook
#    (with_kwargs=True, supported torch>=1.13) to force need_weights=True /
#    average_attn_weights=False on the call, and a forward_hook to capture
#    the resulting per-head (1, num_heads, 5, 5) weights.
#
# We capture and assert on all 4 layers — only the plotting step narrows
# down to Layer 4. We don't need a separate hook on the last layer's full
# output: sa_encoder(...) (nn.TransformerEncoder.forward) already returns
# exactly that — the post-MLP output of the final layer in the stack — so
# output_nav/output_explore below already are the "after Transformer"
# token values FILE 4 needs.
# ══════════════════════════════════════════════════════════════════════════════

def _print_model_structure(model) -> None:
    sa_encoder = model.vision_encoder.sa_encoder
    self_attn0 = sa_encoder.layers[0].self_attn
    print("  Transformer layers found at: model.vision_encoder.sa_encoder.layers")
    print(f"    len(layers)            : {len(sa_encoder.layers)}")
    print(f"    layer[i].self_attn type: {type(self_attn0).__name__}")
    print(f"    self_attn.num_heads    : {self_attn0.num_heads}")
    print(f"    self_attn.embed_dim    : {self_attn0.embed_dim}")
    print(f"    self_attn.batch_first  : {self_attn0.batch_first}")


def _register_attention_hooks(sa_encoder):
    captured = {}
    handles = []

    def make_pre_hook():
        def pre_hook(module, args, kwargs):
            kwargs = dict(kwargs)
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return args, kwargs
        return pre_hook

    def make_post_hook(layer_idx):
        def post_hook(module, args, output):
            captured[layer_idx] = output[1].detach().cpu()
        return post_hook

    for layer_idx, layer in enumerate(sa_encoder.layers):
        self_attn = layer.self_attn
        handles.append(self_attn.register_forward_pre_hook(make_pre_hook(), with_kwargs=True))
        handles.append(self_attn.register_forward_hook(make_post_hook(layer_idx)))

    return captured, handles


def run_transformer_pass(model, tokens_in: torch.Tensor, mask_goal: bool):
    """
    Runs tokens_in through positional encoding + the self-attention encoder,
    exactly as NoMaD_ViNT.forward() does internally, capturing per-layer
    attention weights via hooks.

    mask_goal=False -> navigation mode  (src_key_padding_mask=None)
    mask_goal=True  -> exploration mode (src_key_padding_mask[0, 4]=True)

    Returns (output_tokens (1, 5, 256), attn_weights dict keyed by layer
    index 0..3, each (1, num_heads, 5, 5)).
    """
    sa_encoder = model.vision_encoder.sa_encoder
    captured, handles = _register_attention_hooks(sa_encoder)

    if mask_goal:
        n_tokens = tokens_in.shape[1]
        src_key_padding_mask = torch.zeros(1, n_tokens, dtype=torch.bool, device=DEVICE)
        src_key_padding_mask[0, 4] = True
    else:
        src_key_padding_mask = None

    prev_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            pos_tokens = model.vision_encoder.positional_encoding(tokens_in)
            output_tokens = sa_encoder(pos_tokens, src_key_padding_mask=src_key_padding_mask)
    finally:
        torch.backends.mha.set_fastpath_enabled(prev_fastpath)
        for h in handles:
            h.remove()

    if len(captured) != len(sa_encoder.layers):
        raise RuntimeError(
            "Attention hooks captured nothing for one or more layers. "
            f"Searched model.vision_encoder.sa_encoder.layers[i].self_attn "
            f"({len(sa_encoder.layers)} layers found, {len(captured)} captured). "
            "The hook/pre-hook pairing or the MultiheadAttention call signature "
            "may have changed in this torch version."
        )

    return output_tokens, captured


def pool_ct(output_tokens: torch.Tensor, mask_goal: bool) -> torch.Tensor:
    """
    Mean-pool the 5 output tokens into ct, matching NoMaD_ViNT.forward()'s
    own avg_pool_mask logic: in exploration mode the masked goal token's
    output is excluded from the average entirely (not naively averaged in)
    — confirmed empirically against the previously verified
    mean |ct_nav - ct_explore| ≈ 0.6994, which only reproduces when the
    masked token is excluded rather than included via a plain mean(dim=1).
    """
    if mask_goal:
        return output_tokens[:, :-1, :].mean(dim=1)
    return output_tokens.mean(dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# Attention subplot helper (shared by the combined attention figure below)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_attention_subplot(ax, mat: np.ndarray, head_label: str) -> None:
    im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(TOKEN_LABELS, fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels(TOKEN_LABELS, fontsize=9)
    for r in range(5):
        for c in range(5):
            val = mat[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if val < 0.5 else "black")
    ax.set_title(head_label, fontsize=13, fontweight="bold")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Transformer attention: navigation vs exploration (stage4_attention.png)
# ══════════════════════════════════════════════════════════════════════════════

def plot_attention_combined(attn_nav: dict, attn_explore: dict, save_path: Path) -> None:
    """
    Combined attention figure: 2×4 grid (all 4 heads, nav vs exploration) on
    the left, a Head 2 zoom on the right — the whole goal-masking story in a
    single presentation-ready image.
    """
    nav_attn = attn_nav[LAYER_IDX_FOR_PLOT][0].numpy()
    explore_attn = attn_explore[LAYER_IDX_FOR_PLOT][0].numpy()

    fig = plt.figure(figsize=(30, 12))
    gs = gridspec.GridSpec(2, 6, wspace=0.4, hspace=0.35)

    # Left: 2×4 grid (4 heads × nav/explore)
    for head_idx in range(4):
        ax_nav = fig.add_subplot(gs[0, head_idx])
        _draw_attention_subplot(ax_nav, nav_attn[head_idx], f"Head {head_idx + 1}")
        ax_exp = fig.add_subplot(gs[1, head_idx])
        _draw_attention_subplot(ax_exp, explore_attn[head_idx], f"Head {head_idx + 1}")

    fig.text(0.02, 0.72, "Navigation\n(goal visible)", rotation=90, fontsize=13,
             fontweight="bold", ha="center", va="center", color="#2471a3")
    fig.text(0.02, 0.28, "Exploration\n(goal masked)", rotation=90, fontsize=13,
             fontweight="bold", ha="center", va="center", color="#c0392b")

    # Right: Head 2 zoom (nav on top, explore on bottom)
    nav_mat = nav_attn[HEAD_IDX_FOR_ZOOM]
    explore_mat = explore_attn[HEAD_IDX_FOR_ZOOM]

    for row, (mat, label, color) in enumerate([
        (nav_mat, "Navigation", "#2471a3"),
        (explore_mat, "Exploration", "#c0392b"),
    ]):
        ax = fig.add_subplot(gs[row, 4:6])
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(TOKEN_LABELS, fontsize=11)
        ax.set_yticks(range(5))
        ax.set_yticklabels(TOKEN_LABELS, fontsize=11)
        ax.set_xlabel("Token being attended TO", fontsize=10)
        ax.set_ylabel("Token doing the attending", fontsize=10)
        for r in range(5):
            for c in range(5):
                val = mat[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=13, color="white" if val < 0.5 else "black")
        ax.set_title(f"Head 2 — {label}", fontsize=13, fontweight="bold", color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    f0_goal_nav = nav_mat[0, 4]
    f0_goal_explore = explore_mat[0, 4]
    fig.text(
        0.5, 0.02,
        f"Head 2 is the goal-direction specialist: F0→Goal = {f0_goal_nav*100:.0f}% in navigation, "
        f"{f0_goal_explore*100:.0f}% in exploration.  "
        "One boolean completely severs the goal information pathway.",
        ha="center", fontsize=12, color="#c0392b", fontweight="bold",
    )

    fig.suptitle(
        "Transformer Attention — Navigation vs Exploration (Layer 4)",
        fontsize=18, y=0.99,
    )
    fig.text(0.5, 0.955, _subtitle(), ha="center", fontsize=12, style="italic", color="#555555")

    save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Effect of goal masking on ct (stage4_ct_comparison.png)
# ══════════════════════════════════════════════════════════════════════════════

def plot_ct_heatmap(ct_nav: torch.Tensor, ct_explore: torch.Tensor,
                    mean_abs_diff: float, save_path: Path) -> None:
    """
    Presentation-ready ct comparison as heatmaps: three stacked strips
    (nav, explore, difference) with scalar summary metrics.
    """
    nav = ct_nav[0].detach().cpu().numpy()
    explore = ct_explore[0].detach().cpu().numpy()
    diff = np.abs(nav - explore)

    cosine_sim = np.dot(nav, explore) / (np.linalg.norm(nav) * np.linalg.norm(explore) + 1e-8)
    l2_dist = np.linalg.norm(nav - explore)

    vabs = max(abs(nav).max(), abs(explore).max())

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(3, 2, width_ratios=[20, 1], wspace=0.05, hspace=0.6)

    data_list = [
        (nav, "ct — Navigation mode (goal visible)", "RdBu_r", -vabs, vabs),
        (explore, "ct — Exploration mode (goal masked)", "RdBu_r", -vabs, vabs),
        (diff, "Absolute difference |nav − explore|", "Reds", 0, diff.max()),
    ]

    for row_idx, (data, title, cmap, vmin, vmax) in enumerate(data_list):
        ax = fig.add_subplot(gs[row_idx, 0])
        im = ax.imshow(data.reshape(1, -1), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
        if row_idx == 2:
            ax.set_xlabel("embedding dimension (256-D)", fontsize=9)
        else:
            ax.set_xticks([])
        ax.tick_params(labelsize=7)

        cax = fig.add_subplot(gs[row_idx, 1])
        fig.colorbar(im, cax=cax)

    fig.text(
        0.98, 0.02,
        f"Cosine similarity: {cosine_sim:.4f}    L2 distance: {l2_dist:.2f}    mean|Δ|: {mean_abs_diff:.4f}",
        ha="right", va="bottom", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#555555", alpha=0.9),
    )

    fig.suptitle(
        "Effect of Goal Masking on Context Vector ct",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.text(0.5, 0.97, _subtitle(" — ct is the input to the diffusion model"),
             ha="center", fontsize=11, style="italic", color="#555555")

    save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration — shared by main() (standalone run) and run_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

def run_stage4(model, obs_tokens_np: np.ndarray, goal_token_np: np.ndarray, save_dir: Path) -> dict:
    """
    Builds tokens_in from the already-encoded obs/goal tokens, runs the
    navigation + exploration Transformer passes, pools ct for both, saves
    the 2 PNGs (stage4_attention.png, stage4_ct_comparison.png) into
    save_dir, and returns the key tensors/values.
    """
    _print_model_structure(model)

    obs_tokens = torch.from_numpy(obs_tokens_np).unsqueeze(0).to(DEVICE)   # (1, 4, 256)
    goal_token = torch.from_numpy(goal_token_np).unsqueeze(0).to(DEVICE)  # (1, 256)
    tokens_in  = torch.cat([obs_tokens, goal_token.unsqueeze(1)], dim=1)  # (1, 5, 256)

    print(f"  tokens_in shape : {tuple(tokens_in.shape)}")
    assert tokens_in.shape == (1, 5, ENCODING_SIZE), f"tokens_in shape mismatch: {tokens_in.shape}"
    print("  [OK] tokens_in.shape == (1, 5, 256)")

    output_nav, attn_nav = run_transformer_pass(model, tokens_in, mask_goal=False)
    output_explore, attn_explore = run_transformer_pass(model, tokens_in, mask_goal=True)
    print(f"  output_tokens (navigation)  : {tuple(output_nav.shape)}")
    print(f"  output_tokens (exploration) : {tuple(output_explore.shape)}")
    assert output_nav.shape == (1, 5, ENCODING_SIZE), f"output_tokens shape mismatch: {output_nav.shape}"
    print("  [OK] output_tokens.shape == (1, 5, 256)")

    assert len(attn_nav) == 4, f"expected 4 attention layers, got {len(attn_nav)}"
    print("  [OK] len(attn_nav) == 4")
    for i in range(4):
        assert attn_nav[i].shape == (1, 4, 5, 5), f"attn_nav[{i}] shape mismatch: {attn_nav[i].shape}"
    print("  [OK] attn_nav[3].shape == (1, 4, 5, 5)  (and layers 0-2 also checked)")

    ct_nav = pool_ct(output_nav, mask_goal=False)
    ct_explore = pool_ct(output_explore, mask_goal=True)
    print(f"  ct_nav     : {tuple(ct_nav.shape)}")
    print(f"  ct_explore : {tuple(ct_explore.shape)}")
    assert ct_nav.shape == (1, ENCODING_SIZE), f"ct_nav shape mismatch: {ct_nav.shape}"
    print("  [OK] ct_nav.shape == (1, 256)")
    assert ct_explore.shape == (1, ENCODING_SIZE), f"ct_explore shape mismatch: {ct_explore.shape}"
    print("  [OK] ct_explore.shape == (1, 256)")

    mean_abs_diff = (ct_nav - ct_explore).abs().mean().item()
    print(f"  Mean |ct_nav - ct_explore| = {mean_abs_diff:.4f}")

    plot_attention_combined(attn_nav, attn_explore, save_dir / "stage4_attention.png")
    plot_ct_heatmap(ct_nav, ct_explore, mean_abs_diff, save_dir / "stage4_ct_comparison.png")

    return {
        "tokens_in": tokens_in,
        "output_nav": output_nav,
        "output_explore": output_explore,
        "attn_nav": attn_nav,
        "attn_explore": attn_explore,
        "ct_nav": ct_nav,
        "ct_explore": ct_explore,
        "mean_abs_diff": mean_abs_diff,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{SEP}")
    print("  NoMaD — Stage 4 Visualisation (Transformer + goal masking)")
    print(SEP)

    print("\n[1/4] Loading model …")
    model = load_model()

    print("\n[2/4] Loading frames from disk …")
    sample = visualize_stage1.load_sample_frames()

    print("\n[3/4] Encoding obs + goal tokens …")
    obs_tokens_np = visualize_stage2.extract_obs_tokens(model, sample["obs_raw"])
    goal_token_np = visualize_stage2.extract_goal_token(model, sample["obs_raw"], sample["goal_raw"])
    print(f"  obs_tokens  : {obs_tokens_np.shape}")
    print(f"  goal_token  : {goal_token_np.shape}")

    print(f"\n[4/4] Running Transformer passes and saving 2 PNG files to {RUN_DIR} …")
    run_stage4(model, obs_tokens_np, goal_token_np, save_dir=RUN_DIR)

    print(f"\n{SEP}")
    print(f"Stage 4 complete. 2 files in {RUN_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
