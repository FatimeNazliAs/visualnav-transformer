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

To visualise a different sample, edit TRAJ_NAME and FRAME_IDX in
debug_visuals/config.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed inside the container
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Repo root / train are put on sys.path by debug_visuals/__init__.py.
from debug_visuals.config import (
    RAW_DATA_DIR,
    TRAJ_NAME,
    FRAME_IDX,
    CONTEXT_SIZE,
    NUM_ACTIONS,
    IMAGE_SIZE,
    IMG_MEAN,
    IMG_STD,
    OBS_IDXS,
    GOAL_IDX,
)
from debug_visuals.viz_utils import save_fig

# ── Constants ─────────────────────────────────────────────────────────────────
N_OBS_FRAMES = CONTEXT_SIZE + 1   # 3 past + 1 current = 4


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
    # Both index lists are derived once, in config, so every stage agrees.
    obs_idxs = OBS_IDXS
    goal_idx = GOAL_IDX

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

    save_fig(fig, save_path)
