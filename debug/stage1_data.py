# debug/stage1_data.py
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from debug.config import (
    RAW_DATA_DIR, DEVICE, CONTEXT_SIZE,
    IMAGE_SIZE, NUM_ACTIONS, IMG_MEAN, IMG_STD,
    TRAJ_NAME, FRAME_IDX
)


def load_one_sample(traj_name=TRAJ_NAME, frame_idx=FRAME_IDX, verbose=True):
    """
    Loads one sample from a single trajectory.

    CONTEXT_SIZE is the number of PAST frames (matches the model's
    context_size param / context_size in nomad.yaml). The model expects
    CONTEXT_SIZE+1 total observation frames — the CONTEXT_SIZE past frames
    PLUS the current one — so that's what we collect here.

    Window (CONTEXT_SIZE=3 → 4 total obs frames):
        [t-3]  [t-2]  [t-1]  [t]      [t+8]
         obs    obs    obs   obs(now)  goal

    Parameters
    ----------
    traj_name : str   folder name inside RAW_DATA_DIR
    frame_idx : int   the "current" frame t
    """
    traj_path = RAW_DATA_DIR / traj_name
    assert traj_path.exists(), f"Trajectory not found: {traj_path}"

    # Sort image files numerically (0.jpg, 1.jpg, ..., not lexicographically)
    img_files = sorted(traj_path.glob("*.jpg"), key=lambda p: int(p.stem))
    T = len(img_files)

    # Validate frame index
    t = frame_idx
    assert t >= CONTEXT_SIZE, \
        f"frame_idx={t} too small. Need at least {CONTEXT_SIZE} past frames before it (context_size={CONTEXT_SIZE})."
    assert t + NUM_ACTIONS < T, \
        f"frame_idx={t} too large. Need goal frame at t+{NUM_ACTIONS}={t+NUM_ACTIONS}, but trajectory has only {T} frames."

    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    # Collect context frames: t-CONTEXT_SIZE ... t  (CONTEXT_SIZE+1 frames total)
    obs_frames = [
        tf(Image.open(img_files[i]).convert("RGB"))
        for i in range(t - CONTEXT_SIZE, t + 1)
    ]
    obs_input  = torch.stack(obs_frames).unsqueeze(0).to(DEVICE)  # (1, 4, 3, 96, 96)
    goal_input = tf(Image.open(img_files[t + NUM_ACTIONS]).convert("RGB")) \
                   .unsqueeze(0).to(DEVICE)                        # (1, 3, 96, 96)

    if verbose:
        print(f"  Trajectory : {traj_name}")
        print(f"  Frames used: obs={list(range(t - CONTEXT_SIZE + 1, t + 1))}  goal={t + NUM_ACTIONS}")
        print(f"  obs_input  : {tuple(obs_input.shape)}  (batch, context, C, H, W)")
        print(f"  goal_input : {tuple(goal_input.shape)}  (batch, C, H, W)")
        print(f"  Pixel range: [{obs_input.min():.2f}, {obs_input.max():.2f}] (normalised)")

    return obs_input, goal_input