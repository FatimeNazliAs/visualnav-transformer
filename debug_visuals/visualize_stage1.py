# debug_visuals/visualize_stage1.py
"""
Stage 1 visualisation — "What does the model actually receive as input?"

This module is a library of functions, not a standalone script — it's
driven by debug_visuals/run_pipeline.py, which calls these in sequence
and decides where the output PNGs get saved. Run:

    conda activate vint_train
    cd /app/visualnav-transformer
    python -m debug_visuals.run_pipeline

Exposes:
  load_sample_frames()                    — loads obs context + goal frame
  plot_frame_strip(sample, save_path)     — frame strip PNG
  plot_pixel_distributions(sample, save_path) — raw vs normalised histograms

To visualise a different sample, edit TRAJ_NAME and FRAME_IDX in
debug_visuals/config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed inside the container
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import torchvision.transforms as T

# ── Make the repo root importable when running as `python -m debug_visuals…` ──
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "train"))

from debug_visuals.config import (
    RAW_DATA_DIR,
    TRAJ_NAME,
    FRAME_IDX,
    CONTEXT_SIZE,
    NUM_ACTIONS,
    IMAGE_SIZE,
    IMG_MEAN,
    IMG_STD,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N_OBS_FRAMES = CONTEXT_SIZE + 1   # 3 past + 1 current = 4
CHANNEL_COLOURS = {"R": "#e74c3c", "G": "#2ecc71", "B": "#3498db"}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_frame_rgb(traj_dir: Path, frame_idx: int) -> np.ndarray:
    """
    Load a single frame from disk and resize it to IMAGE_SIZE.

    Returns a uint8 numpy array of shape (H, W, 3), values in [0, 255].
    Frame files are named 0.jpg, 1.jpg, … (GoStanford2 convention).
    """
    img_path = traj_dir / f"{frame_idx}.jpg"
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def load_sample_frames() -> dict:
    """
    Load the observation context and goal frame for the configured sample.

    Returns a dict with:
        obs_raw   : list of N_OBS_FRAMES uint8 arrays (H, W, 3)
        goal_raw  : uint8 array (H, W, 3)
        obs_idxs  : list of frame indices loaded for obs
        goal_idx  : frame index loaded for goal
        traj_name : trajectory folder name (for plot labels)
    """
    traj_dir = RAW_DATA_DIR / TRAJ_NAME

    # obs  = [t - CONTEXT_SIZE, …, t]   (past context + current frame)
    # goal = t + NUM_ACTIONS             (NUM_ACTIONS steps into the future)
    obs_idxs = list(range(FRAME_IDX - CONTEXT_SIZE, FRAME_IDX + 1))
    goal_idx = FRAME_IDX + NUM_ACTIONS

    obs_raw  = [_load_frame_rgb(traj_dir, i) for i in obs_idxs]
    goal_raw = _load_frame_rgb(traj_dir, goal_idx)

    print(f"  Trajectory : {TRAJ_NAME}")
    print(f"  Frame idx  : {FRAME_IDX}  (current frame = obs[-1])")
    print(f"  Obs frames : {obs_idxs}")
    print(f"  Goal frame : {goal_idx}  ({NUM_ACTIONS} steps ahead)")
    print(f"  Image size : {IMAGE_SIZE}")

    return {
        "obs_raw":   obs_raw,
        "goal_raw":  goal_raw,
        "obs_idxs":  obs_idxs,
        "goal_idx":  goal_idx,
        "traj_name": TRAJ_NAME,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation helper
# ══════════════════════════════════════════════════════════════════════════════

def normalise_frame(frame_uint8: np.ndarray) -> np.ndarray:
    """
    Apply ImageNet normalisation to a uint8 (H, W, 3) frame.

    Steps:
        1. uint8 [0, 255]  →  float32 [0.0, 1.0]
        2. subtract per-channel IMG_MEAN
        3. divide by per-channel IMG_STD

    Output values are roughly in [-2.5, 2.5].
    A pixel exactly at the channel mean maps to 0.0.
    """
    frame_f32 = frame_uint8.astype(np.float32) / 255.0
    mean = np.array(IMG_MEAN, dtype=np.float32)
    std  = np.array(IMG_STD,  dtype=np.float32)
    return (frame_f32 - mean) / std   # broadcasts over H, W


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Frame strip
# ══════════════════════════════════════════════════════════════════════════════

def plot_frame_strip(sample: dict, save_path: Path) -> None:
    """
    Save a horizontal strip of all frames the model receives.

    Layout:  [obs t-3] [obs t-2] [obs t-1] [obs t=current] | [GOAL t+8]

    Orange border  = current observation frame (most recent).
    Purple border  = goal frame (encoder φ, can be masked in exploration mode).
    Vertical bar   = visual separator between obs context and goal.
    """
    obs_raw   = sample["obs_raw"]
    goal_raw  = sample["goal_raw"]
    obs_idxs  = sample["obs_idxs"]
    goal_idx  = sample["goal_idx"]
    traj_name = sample["traj_name"]

    n_panels = N_OBS_FRAMES + 1   # 4 obs + 1 goal

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(3.2 * n_panels, 3.8),
        gridspec_kw={"wspace": 0.12},
    )

    # ── Observation frames ────────────────────────────────────────────────────
    for col, (frame, fidx) in enumerate(zip(obs_raw, obs_idxs)):
        ax = axes[col]
        ax.imshow(frame)
        ax.axis("off")

        is_current = (fidx == FRAME_IDX)
        label = f"obs[{col}]\nframe {fidx}"
        if is_current:
            label += "\n← current (t)"
            for spine in ax.spines.values():
                spine.set_edgecolor("#f39c12")
                spine.set_linewidth(3)
                spine.set_visible(True)

        ax.set_title(label, fontsize=9, pad=4)

    # ── Goal frame ────────────────────────────────────────────────────────────
    ax_goal = axes[N_OBS_FRAMES]
    ax_goal.imshow(goal_raw)
    ax_goal.axis("off")
    ax_goal.set_title(
        f"GOAL\nframe {goal_idx}\n(t + {NUM_ACTIONS})",
        fontsize=9, pad=4, color="#8e44ad", fontweight="bold",
    )
    for spine in ax_goal.spines.values():
        spine.set_edgecolor("#8e44ad")
        spine.set_linewidth(3)
        spine.set_visible(True)

    # ── Vertical separator before goal ───────────────────────────────────────
    fig.text(
        (N_OBS_FRAMES / n_panels) - 0.005,
        0.5,
        "│",
        ha="center", va="center",
        fontsize=28, color="#7f8c8d", transform=fig.transFigure,
    )

    fig.suptitle(
        f"Model Input — What the Robot Sees\n"
        f"Trajectory: {traj_name}   frame t={FRAME_IDX}\n"
        f"Observation context (past + current) → Encoder ψ    |    "
        f"Goal (where to go) → Encoder φ (maskable)",
        fontsize=10, fontweight="bold", y=1.08,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Pixel distributions (raw vs normalised)
# ══════════════════════════════════════════════════════════════════════════════

def plot_pixel_distributions(sample: dict, save_path: Path) -> None:
    """
    Save per-channel pixel histograms for raw and normalised frames.

    Layout:  2 rows × (N_OBS_FRAMES + 1) columns
        Row 0 — RAW        : uint8,   x-axis [0, 255]
        Row 1 — NORMALISED : float32, x-axis ~ [-2.5, 2.5]

    The dashed vertical line in row 1 marks x=0 (the ideal post-normalisation
    centre). If normalisation was accidentally skipped, the row-1 distributions
    would be shifted far to the right — this plot catches that silent bug.
    """
    obs_raw   = sample["obs_raw"]
    goal_raw  = sample["goal_raw"]
    obs_idxs  = sample["obs_idxs"]
    goal_idx  = sample["goal_idx"]

    all_frames     = obs_raw + [goal_raw]
    all_frame_idxs = obs_idxs + [goal_idx]
    is_goal        = [False] * N_OBS_FRAMES + [True]

    n_cols = len(all_frames)
    n_rows = 2
    n_bins = 60

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 4.5),
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )

    for col, (frame_raw, fidx, goal_flag) in enumerate(
        zip(all_frames, all_frame_idxs, is_goal)
    ):
        frame_norm = normalise_frame(frame_raw)
        col_label  = f"frame {fidx}\n(GOAL)" if goal_flag else f"frame {fidx}"

        for ch_idx, (ch_name, ch_colour) in enumerate(CHANNEL_COLOURS.items()):

            # Row 0 — raw
            ax_raw = axes[0, col]
            ax_raw.hist(
                frame_raw[..., ch_idx].ravel(),
                bins=n_bins, range=(0, 255),
                color=ch_colour, alpha=0.55,
                histtype="stepfilled", label=ch_name,
            )
            ax_raw.set_xlim(0, 255)
            ax_raw.set_xlabel("pixel value (uint8)", fontsize=7)
            ax_raw.set_ylabel("count", fontsize=7)
            ax_raw.tick_params(labelsize=6)

            # Row 1 — normalised
            ax_nrm = axes[1, col]
            ax_nrm.hist(
                frame_norm[..., ch_idx].ravel(),
                bins=n_bins,
                color=ch_colour, alpha=0.55,
                histtype="stepfilled", label=ch_name,
            )
            ax_nrm.set_xlabel("normalised value (float32)", fontsize=7)
            ax_nrm.set_ylabel("count", fontsize=7)
            ax_nrm.tick_params(labelsize=6)
            ax_nrm.axvline(0, color="#2c3e50", linewidth=0.8,
                           linestyle="--", alpha=0.7)

        # Column title and goal border
        axes[0, col].set_title(col_label, fontsize=8,
                                color="#8e44ad" if goal_flag else "#2c3e50")
        if goal_flag:
            for row in range(n_rows):
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor("#8e44ad")
                    spine.set_linewidth(1.5)

        # Legend only on the leftmost column
        if col == 0:
            axes[0, 0].legend(fontsize=6, loc="upper right")
            axes[1, 0].legend(fontsize=6, loc="upper right")

    # Row labels
    axes[0, 0].annotate(
        "RAW\n(uint8)",
        xy=(-0.45, 0.5), xycoords="axes fraction",
        fontsize=9, ha="center", va="center",
        fontweight="bold", color="#c0392b", rotation=90,
    )
    axes[1, 0].annotate(
        "NORMALISED\n(float32)",
        xy=(-0.45, 0.5), xycoords="axes fraction",
        fontsize=9, ha="center", va="center",
        fontweight="bold", color="#27ae60", rotation=90,
    )

    fig.suptitle(
        f"Pixel distributions — {TRAJ_NAME}  t={FRAME_IDX}\n"
        f"ImageNet normalisation: mean={IMG_MEAN}  std={IMG_STD}",
        fontsize=9,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Raw vs normalised image comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_normalised_comparison(sample: dict, save_path: Path) -> None:
    """
    Save a 2×5 grid comparing each frame's raw image against its
    normalised counterpart, rendered with a diverging colourmap.

    Layout:  obs[0..3] then goal, same order as the frame strip.
        Row 0 — raw image          (ax.imshow(frame_uint8))
        Row 1 — normalised image   (RdBu_r, anchored to [-2.5, 2.5])

    Pixels at the channel mean show up as white/neutral; pixels far from
    the mean push toward red or blue, making normalisation skew visible
    at a glance (e.g. a uniformly blue row 1 would flag a channel-order bug).
    """
    obs_raw   = sample["obs_raw"]
    goal_raw  = sample["goal_raw"]
    obs_idxs  = sample["obs_idxs"]
    goal_idx  = sample["goal_idx"]

    all_frames     = obs_raw + [goal_raw]
    all_frame_idxs = obs_idxs + [goal_idx]
    is_goal        = [False] * N_OBS_FRAMES + [True]

    n_cols = len(all_frames)

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(3.0 * n_cols, 6.0),
        gridspec_kw={"hspace": 0.4, "wspace": 0.15},
    )

    for col, (frame_raw, fidx, goal_flag) in enumerate(
        zip(all_frames, all_frame_idxs, is_goal)
    ):
        frame_norm_clipped = np.clip(normalise_frame(frame_raw), -2.5, 2.5)

        ax_raw = axes[0, col]
        ax_raw.imshow(frame_raw)
        ax_raw.axis("off")

        ax_norm = axes[1, col]
        ax_norm.imshow(frame_norm_clipped, cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax_norm.axis("off")

        if goal_flag:
            title = f"GOAL\nframe {fidx}\n(t + {NUM_ACTIONS})"
            title_colour = "#8e44ad"
        else:
            title = f"obs[{col}]\nframe {fidx}"
            if fidx == FRAME_IDX:
                title += "\n← current (t)"
            title_colour = "#2c3e50"

        axes[0, col].set_title(
            title, fontsize=9, pad=4, color=title_colour,
            fontweight="bold" if goal_flag else "normal",
        )

        border_colour = None
        if goal_flag:
            border_colour = "#8e44ad"
        elif fidx == FRAME_IDX:
            border_colour = "#f39c12"

        if border_colour is not None:
            for row in range(2):
                for spine in axes[row, col].spines.values():
                    spine.set_edgecolor(border_colour)
                    spine.set_linewidth(3)
                    spine.set_visible(True)

    # Row labels
    axes[0, 0].annotate(
        "RAW",
        xy=(-0.35, 0.5), xycoords="axes fraction",
        fontsize=11, ha="center", va="center",
        fontweight="bold", color="#c0392b", rotation=90,
    )
    axes[1, 0].annotate(
        "NORMALISED",
        xy=(-0.35, 0.5), xycoords="axes fraction",
        fontsize=11, ha="center", va="center",
        fontweight="bold", color="#27ae60", rotation=90,
    )

    fig.suptitle(
        f"Raw vs normalised — {TRAJ_NAME} t={FRAME_IDX}",
        fontsize=10,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved : {save_path}")

