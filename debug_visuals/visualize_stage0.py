# debug_visuals/visualize_stage0.py
"""
Stage 0 visualisation — "What is NoMaD's architecture?"

Generates a single architecture flow diagram using matplotlib, with
color-coded regions separating Visual Processing (blue) from Action
Generation (green). This is a static diagram — no model inference needed.

Run from the repo root inside the container:

    conda activate vint_train
    cd /app/visualnav-transformer
    python -m debug_visuals.visualize_stage0
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from debug_visuals.config import OUTPUTS_DIR, TRAJ_NAME, FRAME_IDX

SEP = "─" * 60


def _add_box(ax, x, y, w, h, text, color, fontsize=9, text_color="white", alpha=0.9):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight="bold", zorder=4)
    return box


def _add_arrow(ax, x1, y1, x2, y2, color="#555555", style="->", lw=1.8):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, mutation_scale=15, zorder=2,
    )
    ax.add_patch(arrow)


def _add_label(ax, x, y, text, fontsize=7.5, color="#333333", style="italic"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, style=style, zorder=5)


def plot_architecture(save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Background regions ───────────────────────────────────────────────
    visual_bg = FancyBboxPatch(
        (0.3, 0.3), 9.4, 9.0,
        boxstyle="round,pad=0.3",
        facecolor="#d6eaf8", edgecolor="#2980b9", linewidth=2.5, alpha=0.35,
        zorder=0,
    )
    ax.add_patch(visual_bg)
    ax.text(5.0, 9.0, "Visual Processing", fontsize=14, fontweight="bold",
            color="#2471a3", ha="center", va="center", zorder=1)

    action_bg = FancyBboxPatch(
        (10.0, 0.3), 5.7, 9.0,
        boxstyle="round,pad=0.3",
        facecolor="#d5f5e3", edgecolor="#27ae60", linewidth=2.5, alpha=0.35,
        zorder=0,
    )
    ax.add_patch(action_bg)
    ax.text(12.85, 9.0, "Action Generation", fontsize=14, fontweight="bold",
            color="#1e8449", ha="center", va="center", zorder=1)

    # ── Visual Processing boxes ──────────────────────────────────────────

    # Obs frames input
    _add_box(ax, 2.0, 6.0, 2.4, 1.0, "Obs Frames\n(t-3 … t)", "#5dade2", fontsize=9)
    _add_label(ax, 2.0, 5.25, "4 × RGB 96×96", fontsize=7)

    # Goal frame input
    _add_box(ax, 2.0, 3.0, 2.4, 1.0, "Goal Frame\n(t + 8)", "#af7ac5", fontsize=9)
    _add_label(ax, 2.0, 2.25, "1 × RGB 96×96", fontsize=7)

    # Obs encoder
    _add_box(ax, 5.2, 6.0, 2.2, 1.0, "ψ Obs Encoder\n(EfficientNet)", "#2e86c1", fontsize=8)
    _add_label(ax, 5.2, 5.25, "→ 4 × 256-dim tokens", fontsize=7)

    # Goal encoder
    _add_box(ax, 5.2, 3.0, 2.2, 1.0, "φ Goal Encoder\n(EfficientNet)", "#7d3c98", fontsize=8)
    _add_label(ax, 5.2, 2.25, "→ 1 × 256-dim token", fontsize=7)

    # Transformer
    _add_box(ax, 8.5, 4.5, 2.2, 2.2, "Transformer\nSelf-Attention\n(4 layers × 4 heads)", "#1a5276", fontsize=8)

    # Goal masking annotation
    ax.text(8.5, 2.8, "Goal Masking", fontsize=8, ha="center", va="center",
            color="#c0392b", fontweight="bold", style="italic", zorder=5)
    ax.text(8.5, 2.3, "mask=False → Navigation\nmask=True  → Exploration",
            fontsize=7, ha="center", va="center", color="#922b21", zorder=5)

    # Context vector output
    _add_box(ax, 8.5, 7.5, 1.6, 0.7, "ct (256-dim)", "#1a5276", fontsize=8, alpha=0.7)
    _add_label(ax, 8.5, 7.0, "mean pooling", fontsize=7)

    # ── Action Generation boxes ──────────────────────────────────────────

    # Diffusion U-Net
    _add_box(ax, 12.5, 6.0, 2.6, 1.6, "Diffusion U-Net\n(ConditionalUnet1D)\nK=10 denoising steps", "#27ae60", fontsize=8)

    # Noise input
    _add_box(ax, 12.5, 8.0, 1.8, 0.6, "a^K ~ N(0, I)", "#7dcea0", fontsize=8, text_color="#1a5276")
    _add_label(ax, 12.5, 7.5, "pure noise", fontsize=7)

    # Action sequence output
    _add_box(ax, 12.5, 3.5, 2.6, 1.0, "Action Sequence\na⁰: 8 × (v, ω)", "#1e8449", fontsize=9)
    _add_label(ax, 12.5, 2.75, "linear_vel, angular_vel\n→ robot motion commands", fontsize=7)

    # Distance predictor (side branch)
    _add_box(ax, 12.5, 1.2, 2.0, 0.7, "Distance\nPredictor", "#82e0aa", fontsize=8, text_color="#1a5276")
    _add_label(ax, 14.8, 1.2, "→ dist\n   to goal", fontsize=7)

    # ── Arrows ───────────────────────────────────────────────────────────

    # Obs frames → Obs encoder
    _add_arrow(ax, 3.2, 6.0, 4.1, 6.0, color="#2e86c1")

    # Goal frame → Goal encoder
    _add_arrow(ax, 3.2, 3.0, 4.1, 3.0, color="#7d3c98")

    # Obs encoder → Transformer
    _add_arrow(ax, 6.3, 6.0, 7.4, 5.2, color="#2e86c1")

    # Goal encoder → Transformer
    _add_arrow(ax, 6.3, 3.0, 7.4, 3.9, color="#7d3c98")

    # Transformer → ct
    _add_arrow(ax, 8.5, 5.6, 8.5, 7.1, color="#1a5276")

    # ct → Diffusion U-Net
    _add_arrow(ax, 9.3, 7.5, 11.0, 6.5, color="#1a5276", lw=2.2)
    _add_label(ax, 10.2, 7.3, "conditioning", fontsize=7, color="#1a5276")

    # Noise → Diffusion U-Net
    _add_arrow(ax, 12.5, 7.7, 12.5, 6.8, color="#27ae60")

    # Diffusion U-Net → Action sequence
    _add_arrow(ax, 12.5, 5.2, 12.5, 4.0, color="#27ae60", lw=2.2)

    # ct → Distance predictor
    _add_arrow(ax, 9.3, 7.5, 11.3, 1.5, color="#82e0aa", style="-|>", lw=1.2)

    # ── Title ────────────────────────────────────────────────────────────
    fig.suptitle("NoMaD Architecture Overview", fontsize=18, fontweight="bold", y=0.98)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved : {save_path}")


def main() -> None:
    print(f"\n{SEP}")
    print("  NoMaD — Stage 0 Visualisation (architecture diagram)")
    print(SEP)

    run_dir = OUTPUTS_DIR / f"{TRAJ_NAME}_f{FRAME_IDX}"
    plot_architecture(save_path=run_dir / "stage0_architecture.png")

    print(f"\n{SEP}")
    print("  Done.")
    print(SEP)


if __name__ == "__main__":
    main()
