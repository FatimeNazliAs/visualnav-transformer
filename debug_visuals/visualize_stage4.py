# debug_visuals/visualize_stage4.py
"""
Stage 4 visualisation — "What does the Transformer + goal masking do?"

Runs the 4-layer self-attention encoder (model.vision_encoder.sa_encoder)
twice on the same 5 input tokens (obs[0..3] + goal): once in navigation
mode (goal token active) and once in exploration mode (goal token masked
via src_key_padding_mask). Tells one story across 4 PNGs:

  1. stage4_attention_compare.png    — Layer 4, all 4 heads, nav vs
     exploration side by side: where attention collapses under masking.
  2. stage4_attention_head2_zoom.png — Head 2 alone, large and annotated:
     the single clearest example of the goal pathway being severed.
  3. stage4_ct_comparison.png        — the measurable effect on ct, the
     vector actually handed to the diffusion model.
  4. stage4_transformer_effect.png   — each token before vs after the
     4 Transformer layers, navigation mode: what attention actually added.

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

from debug_visuals.config import (
    TRAJ_NAME,
    FRAME_IDX,
    ENCODING_SIZE,
    DEVICE,
    OUTPUTS_DIR,
)
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
# Plot 1 — Layer 4 attention: navigation vs exploration, all 4 heads
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


def plot_attention_compare(attn_nav: dict, attn_explore: dict, save_path: Path) -> None:
    nav_attn = attn_nav[LAYER_IDX_FOR_PLOT][0].numpy()       # (4 heads, 5, 5)
    explore_attn = attn_explore[LAYER_IDX_FOR_PLOT][0].numpy()

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    for head_idx in range(4):
        _draw_attention_subplot(axes[0, head_idx], nav_attn[head_idx], f"Head {head_idx + 1}")
    for head_idx in range(4):
        _draw_attention_subplot(axes[1, head_idx], explore_attn[head_idx], f"Head {head_idx + 1}")

    fig.text(0.06, 0.70, "Navigation →", rotation=90, fontsize=14,
              fontweight="bold", ha="center", va="center")
    fig.text(0.06, 0.28, "Exploration →", rotation=90, fontsize=14,
              fontweight="bold", ha="center", va="center")

    head2_nav_col = nav_attn[HEAD_IDX_FOR_ZOOM][:, 4]       # all queries -> Goal key
    head2_explore_col = explore_attn[HEAD_IDX_FOR_ZOOM][:, 4]
    fig.text(
        0.5, 0.015,
        f"Head 2 Goal column: Navigation={head2_nav_col.min():.2f}–{head2_nav_col.max():.2f} "
        f"(bright yellow) vs Exploration={head2_explore_col.max():.2f} (black). "
        "Same head, one boolean, completely different behavior.",
        ha="center", fontsize=11, color="#c0392b", fontweight="bold",
    )

    fig.suptitle("Layer 4 Attention Weights: Navigation vs Exploration", fontsize=18, y=0.985)
    fig.text(0.5, 0.945, _subtitle(), ha="center", fontsize=12, style="italic", color="#555555")

    plt.tight_layout(rect=[0.04, 0.06, 1, 0.92])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Head 2 zoom: the goal-direction specialist
# ══════════════════════════════════════════════════════════════════════════════

def plot_head2_zoom(attn_nav: dict, attn_explore: dict, save_path: Path) -> None:
    nav_mat = attn_nav[LAYER_IDX_FOR_PLOT][0, HEAD_IDX_FOR_ZOOM].numpy()
    explore_mat = attn_explore[LAYER_IDX_FOR_PLOT][0, HEAD_IDX_FOR_ZOOM].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, mat, label in [(axes[0], nav_mat, "Navigation"), (axes[1], explore_mat, "Exploration")]:
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(TOKEN_LABELS, fontsize=12)
        ax.set_yticks(range(5))
        ax.set_yticklabels(TOKEN_LABELS, fontsize=12)
        ax.set_xlabel("Token being attended TO", fontsize=11)
        ax.set_ylabel("Token doing the attending", fontsize=11)
        for r in range(5):
            for c in range(5):
                val = mat[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=14, color="white" if val < 0.5 else "black")
        ax.set_title(label, fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    f0_goal_nav = nav_mat[0, 4]
    goal_goal_nav = nav_mat[4, 4]
    f0_goal_explore = explore_mat[0, 4]

    fig.text(
        0.27, 0.11,
        f"F0 pays {f0_goal_nav * 100:.0f}% attention to Goal: oldest frame is almost\n"
        "entirely focused on goal direction",
        ha="center", fontsize=10,
    )
    fig.text(
        0.27, 0.03,
        f"Goal token attends {goal_goal_nav * 100:.0f}% to itself: self-referential,\n"
        "it already carries goal information",
        ha="center", fontsize=10,
    )
    fig.text(
        0.73, 0.07,
        f"Goal column zeroed (value={f0_goal_explore:.2f}): no token can receive goal\n"
        "information — pathway completely severed",
        ha="center", fontsize=10,
    )

    fig.suptitle("Head 2, Layer 4: The Goal-Direction Specialist Head", fontsize=16, y=1.02)
    fig.text(0.5, 0.97, _subtitle(), ha="center", fontsize=12, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.18, 1, 0.93])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — ct: navigation vs exploration
# ══════════════════════════════════════════════════════════════════════════════

def plot_ct_comparison(ct_nav: torch.Tensor, ct_explore: torch.Tensor,
                        mean_abs_diff: float, save_path: Path) -> None:
    nav     = ct_nav[0].detach().cpu().numpy()
    explore = ct_explore[0].detach().cpu().numpy()
    diff    = (ct_nav - ct_explore).abs()[0].detach().cpu().numpy()

    top10_idx = np.argsort(diff)[-10:]
    bar_colors = np.full(256, "#e74c3c", dtype=object)
    bar_colors[top10_idx] = "#e67e22"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].bar(range(256), nav, color="#3498db", linewidth=0)
    axes[0].set_title("ct — Navigation mode", fontsize=12)
    axes[0].text(0.5, -0.18, "Goal token active → ct pulled toward goal direction",
                 transform=axes[0].transAxes, ha="center", fontsize=9, style="italic")

    axes[1].bar(range(256), explore, color="#2ecc71", linewidth=0)
    axes[1].set_title("ct — Exploration mode", fontsize=12)
    axes[1].text(0.5, -0.18, "Goal token masked → ct reflects environment only",
                 transform=axes[1].transAxes, ha="center", fontsize=9, style="italic")

    axes[2].bar(range(256), diff, color=list(bar_colors), linewidth=0)
    axes[2].set_title("Change caused by masking", fontsize=12)
    axes[2].text(
        0.97, 0.95, f"mean|Δ| = {mean_abs_diff:.4f}",
        transform=axes[2].transAxes, ha="right", va="top",
        fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    axes[2].text(
        0.97, 0.83,
        f"Each of 256 dimensions shifted by {mean_abs_diff:.2f} on average.\n"
        "The diffusion model receives substantially different\n"
        "input → different robot actions.",
        transform=axes[2].transAxes, ha="right", va="top", fontsize=8.5,
    )

    for ax in axes:
        ax.set_xlabel("dim", fontsize=10)
    axes[0].set_ylabel("value", fontsize=10)

    fig.suptitle("Effect of Goal Masking on ct (input to diffusion model)", fontsize=15, y=1.08)
    fig.text(0.5, 1.0, _subtitle(" | mean|Δ| proves masking is effective"),
              ha="center", fontsize=11, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Before vs after the Transformer, per token
# ══════════════════════════════════════════════════════════════════════════════

def plot_transformer_effect(obs_tokens_np: np.ndarray, goal_token_np: np.ndarray,
                             output_nav: torch.Tensor, save_path: Path) -> None:
    before = np.concatenate([obs_tokens_np, goal_token_np[None, :]], axis=0)   # (5, 256)
    after  = output_nav[0].detach().cpu().numpy()                              # (5, 256)
    diff   = after - before

    fig, axes = plt.subplots(5, 3, figsize=(18, 20))

    for i, name in enumerate(TOKEN_LABELS):
        ax_before, ax_after, ax_diff = axes[i]

        ax_before.bar(range(256), before[i], color="#3498db", linewidth=0)
        ax_before.set_ylabel(name, fontsize=13, fontweight="bold", rotation=0, labelpad=28, va="center")
        if i == 0:
            ax_before.set_title("Before Transformer", fontsize=12)

        ax_after.bar(range(256), after[i], color="#a9dfbf", linewidth=0)
        if i == 0:
            ax_after.set_title("After Transformer (nav)", fontsize=12)

        d = diff[i]
        bar_colors = np.where(d >= 0, "#e74c3c", "#3498db")
        ax_diff.bar(range(256), d, color=list(bar_colors), linewidth=0)
        if i == 0:
            ax_diff.set_title("What the Transformer added", fontsize=12)
        ax_diff.text(
            0.97, 0.92, f"mean|Δ| = {np.abs(d).mean():.3f}",
            transform=ax_diff.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        for ax in (ax_before, ax_after, ax_diff):
            ax.tick_params(labelsize=7)
        if i == 4:
            for ax in (ax_before, ax_after, ax_diff):
                ax.set_xlabel("dim", fontsize=9)

    fig.suptitle("What 4 Transformer Layers Added to Each Token", fontsize=17, y=0.995)
    fig.text(0.5, 0.975, _subtitle(), ha="center", fontsize=12, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration — shared by main() (standalone run) and run_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

def run_stage4(model, obs_tokens_np: np.ndarray, goal_token_np: np.ndarray, save_dir: Path) -> dict:
    """
    Builds tokens_in from the already-encoded obs/goal tokens, runs the
    navigation + exploration Transformer passes, pools ct for both, saves
    the 4 PNGs into save_dir, and returns the key tensors/values.
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

    plot_attention_compare(attn_nav, attn_explore, save_dir / "stage4_1_attention_compare.png")
    plot_head2_zoom(attn_nav, attn_explore, save_dir / "stage4_2_attention_head2_zoom.png")
    plot_ct_comparison(ct_nav, ct_explore, mean_abs_diff, save_dir / "stage4_3_ct_comparison.png")
    plot_transformer_effect(obs_tokens_np, goal_token_np, output_nav, save_dir / "stage4_4_transformer_effect.png")

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
    model = visualize_stage2.load_model()

    print("\n[2/4] Loading frames from disk …")
    sample = visualize_stage1.load_sample_frames()

    print("\n[3/4] Encoding obs + goal tokens …")
    obs_tokens_np = visualize_stage2.extract_obs_tokens(model, sample["obs_raw"])
    goal_token_np = visualize_stage2.extract_goal_token(model, sample["obs_raw"], sample["goal_raw"])
    print(f"  obs_tokens  : {obs_tokens_np.shape}")
    print(f"  goal_token  : {goal_token_np.shape}")

    run_dir = OUTPUTS_DIR / f"{TRAJ_NAME}_f{FRAME_IDX}"
    print(f"\n[4/4] Running Transformer passes and saving 4 PNG files to {run_dir} …")
    run_stage4(model, obs_tokens_np, goal_token_np, save_dir=run_dir)

    print(f"\n{SEP}")
    print("Stage 4 complete. 4 files in debug_visuals/outputs/")
    print(f"(actual path: {run_dir})")
    print(SEP)


if __name__ == "__main__":
    main()
