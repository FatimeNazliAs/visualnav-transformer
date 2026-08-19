# deeper_visuals/common/data.py
"""
Load the frames a phase's scene feeds into the model.

Adapted from debug_visuals/visualize_stage1.py's loading half, with the
module-level TRAJ_NAME/FRAME_IDX globals replaced by an explicit PhaseConfig
argument — that is what lets one library serve six phases.

The frame layout, for a scene at current frame t:

    obs  = [t-3, t-2, t-1, t]   -> encoder psi   (CONTEXT_SIZE + 1 frames)
    goal = t + 8                -> encoder phi   (NUM_ACTIONS steps ahead)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from deeper_visuals.common import settings
from deeper_visuals.common.config import PhaseConfig


def load_frame_rgb(traj_dir: Path, frame_idx: int) -> np.ndarray:
    """
    Load one frame and resize it to IMAGE_SIZE.

    Returns uint8 (H, W, 3) in [0, 255]. Frame files are named 0.jpg, 1.jpg, …
    (the GoStanford2 convention).
    """
    img_path = traj_dir / f"{frame_idx}.jpg"
    if not img_path.is_file():
        raise FileNotFoundError(f"frame not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    img = img.resize((settings.IMAGE_SIZE[1], settings.IMAGE_SIZE[0]), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def load_sample(cfg: PhaseConfig, *, verbose: bool = True) -> dict:
    """
    Load the observation context and goal frame for cfg's scene.

    Returns a dict with:
        obs_raw   : list of N_OBS_FRAMES uint8 arrays (H, W, 3)
        goal_raw  : uint8 array (H, W, 3)
        obs_idxs  : the frame indices loaded for obs
        goal_idx  : the frame index loaded for goal
        traj_name : trajectory folder name (for plot labels)
    """
    traj_dir = cfg.traj_dir
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"trajectory not found: {traj_dir}")

    # Guard the frame range up front — an out-of-range frame otherwise fails
    # deep inside the forward pass with a much less obvious error.
    n_frames = len(list(traj_dir.glob("*.jpg")))
    lo, hi = settings.CONTEXT_SIZE, n_frames - settings.NUM_ACTIONS - 1
    if not lo <= cfg.frame <= hi:
        raise ValueError(
            f"frame {cfg.frame} out of range for {cfg.traj} "
            f"({n_frames} frames): valid range is {lo}..{hi}"
        )

    obs_raw  = [load_frame_rgb(traj_dir, i) for i in cfg.obs_idxs]
    goal_raw = load_frame_rgb(traj_dir, cfg.goal_idx)

    if verbose:
        print(f"  Trajectory : {cfg.traj}  ({n_frames} frames)")
        print(f"  Frame idx  : {cfg.frame}  (current frame = obs[-1])")
        print(f"  Obs frames : {cfg.obs_idxs}")
        print(f"  Goal frame : {cfg.goal_idx}  ({settings.NUM_ACTIONS} steps ahead)")
        print(f"  Image size : {settings.IMAGE_SIZE}")

    return {
        "obs_raw":   obs_raw,
        "goal_raw":  goal_raw,
        "obs_idxs":  cfg.obs_idxs,
        "goal_idx":  cfg.goal_idx,
        "traj_name": cfg.traj,
    }


def normalise_frame(frame_uint8: np.ndarray) -> np.ndarray:
    """
    Apply ImageNet normalisation to a uint8 (H, W, 3) frame.

        1. uint8 [0, 255]  ->  float32 [0.0, 1.0]
        2. subtract per-channel IMG_MEAN
        3. divide by per-channel IMG_STD

    Output lands roughly in [-2.5, 2.5]; a pixel at the channel mean maps to 0.
    """
    frame_f32 = frame_uint8.astype(np.float32) / 255.0
    mean = np.array(settings.IMG_MEAN, dtype=np.float32)
    std  = np.array(settings.IMG_STD,  dtype=np.float32)
    return (frame_f32 - mean) / std   # broadcasts over H, W


def to_model_input(frames: list[np.ndarray]) -> "np.ndarray":
    """
    Stack raw frames into the channel-concatenated layout NoMaD_ViNT expects.

    Returns float32 (3 * len(frames), H, W) — the per-frame RGB channels
    concatenated along dim 0, which the encoder splits back apart internally.
    """
    normed = [normalise_frame(f).transpose(2, 0, 1) for f in frames]  # each (3,H,W)
    return np.concatenate(normed, axis=0).astype(np.float32)
