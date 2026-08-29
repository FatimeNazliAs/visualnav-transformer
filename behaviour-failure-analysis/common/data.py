# behaviour-failure-analysis/common/data.py
"""
Load the frames a phase's scene feeds into the model.

Copied from deeper_visuals/common/data.py (branch feature/deeper-visualization).
Two changes, both because a Case varies what a PhaseConfig held fixed:

  1. It takes a Case, not a PhaseConfig. Same interface — traj, frame,
     traj_dir, obs_idxs, goal_idx — so the bodies are otherwise unchanged.
  2. The frame-range guard in load_sample now allows for a goal that sits
     further ahead than the prediction horizon. PhaseConfig pinned the goal at
     frame + NUM_ACTIONS, so guarding NUM_ACTIONS covered both; a Case may put
     the goal up to GOAL_DISTANCE_MAX ahead, and the guard has to cover
     whichever reaches further.

The frame layout, for a case at current frame t with goal distance d:

    obs  = [t-3, t-2, t-1, t]   -> encoder psi   (CONTEXT_SIZE + 1 frames)
    goal = t + d                -> encoder phi   (d in GOAL_DISTANCE_MIN..MAX)
    truth= t+1 … t+8            -> the recorded route (always NUM_ACTIONS long)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from common import actions as action_space
from common import settings
from common.config import Case


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


@dataclass(frozen=True, eq=False)
class Scene:
    """
    One loaded scene: the frames, their indices, and the model input they make.

    This closes a gap that used to sit between two modules. `data.py` stopped at
    `to_model_input` and `model.py` started at `obs_batch`, so the two lines that
    join them were written out in five places — p2 (behind a local wrapper), p3,
    p4, occlusion.py and smoke_test.py — and had already drifted apart in whether
    they stated `goal_mask` explicitly. Batching a scene is not phase-specific
    knowledge, so it lives here with the scene it batches.

    It also gives the frame-order convention a home. `obs_raw[-1]` is the current
    frame, a fact six files previously had to know independently and that was
    documented in comments and enforced nowhere; `now` says it once.
    """

    obs_raw: list[np.ndarray]   # N_OBS_FRAMES uint8 (H, W, 3), oldest first
    goal_raw: np.ndarray        # uint8 (H, W, 3)
    obs_idxs: list[int]
    goal_idx: int
    traj_name: str

    @property
    def now(self) -> np.ndarray:
        """The current frame — the last observation, not the goal."""
        return self.obs_raw[-1]

    @property
    def token_labels(self) -> list[str]:
        """
        What the five tokens are called, in the order NoMaD_ViNT assembles them.

        These exact words appear on P1's frame strip, P2's encoder rows and P3's
        attention axes, and the payoff of threading one scene through every phase
        is that the axes need no glossary. P3 previously wrote them out as a
        literal list with a comment saying they were "deliberately the words P1
        and P2 already used" — an assertion of agreement rather than agreement.
        It also baked CONTEXT_SIZE into the list, so raising the context size in
        settings.py would have left P3 silently mislabelling every panel of its
        attention figure while P1 and P2 followed along correctly.

        Derived here instead, from the one place the context size is set.
        """
        past = [f"t − {back}" for back in range(settings.CONTEXT_SIZE, 0, -1)]
        return [*past, "now", "goal"]

    @property
    def obs_input(self) -> np.ndarray:
        """The observation stack, unbatched — (3 * N_OBS_FRAMES, H, W)."""
        return to_model_input(self.obs_raw)

    @property
    def goal_input(self) -> np.ndarray:
        """The goal frame, unbatched — (3, H, W)."""
        return to_model_input([self.goal_raw])

    @property
    def obs_batch(self) -> np.ndarray:
        """The observation stack as a batch of one, ready for encode_tokens."""
        return self.obs_input[None]

    @property
    def goal_batch(self) -> np.ndarray:
        """The goal frame as a batch of one, ready for encode_tokens."""
        return self.goal_input[None]


def load_sample(cfg: Case, *, verbose: bool = True) -> Scene:
    """
    Load the observation context and goal frame for cfg's scene.
    """
    traj_dir = cfg.traj_dir
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"trajectory not found: {traj_dir}")

    # Guard the frame range up front — an out-of-range frame otherwise fails
    # deep inside the forward pass with a much less obvious error.
    n_frames = len(list(traj_dir.glob("*.jpg")))
    # The case needs frames in both directions: CONTEXT_SIZE behind for the
    # observation stack, and ahead for whichever of the goal frame or the
    # ground-truth horizon reaches further. Which one that is depends on the
    # goal distance, so it is a max rather than a constant.
    reach = max(cfg.goal_distance, settings.NUM_ACTIONS)
    lo, hi = settings.CONTEXT_SIZE, n_frames - reach - 1
    if not lo <= cfg.frame <= hi:
        raise ValueError(
            f"frame {cfg.frame} out of range for {cfg.traj} "
            f"({n_frames} frames): valid range is {lo}..{hi} "
            f"(needs {settings.CONTEXT_SIZE} behind and {reach} ahead)"
        )

    obs_raw  = [load_frame_rgb(traj_dir, i) for i in cfg.obs_idxs]
    goal_raw = load_frame_rgb(traj_dir, cfg.goal_idx)

    if verbose:
        print(f"  Trajectory : {cfg.traj}  ({n_frames} frames)")
        print(f"  Frame idx  : {cfg.frame}  (current frame = obs[-1])")
        print(f"  Obs frames : {cfg.obs_idxs}")
        print(f"  Goal frame : {cfg.goal_idx}  ({cfg.goal_distance} steps ahead)")
        print(f"  Image size : {settings.IMAGE_SIZE}")

    return Scene(
        obs_raw=obs_raw,
        goal_raw=goal_raw,
        obs_idxs=cfg.obs_idxs,
        goal_idx=cfg.goal_idx,
        traj_name=cfg.traj,
    )


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


def load_ground_truth(cfg: Case) -> dict:
    """
    The route the robot actually drove from cfg's frame, as the model sees it.

    Every earlier phase needed only pixels. This one needs the answer as well:
    diffusion is trained by corrupting a REAL action sequence and learning to
    undo that, so a figure about training has nothing to show without one.

    Returns:
        deltas    (NUM_ACTIONS, ACTION_DIM) — normalised to [-1, 1], the space
                  the diffusion head actually operates in. This is the tensor
                  noise gets added to.
        waypoints (NUM_ACTIONS, ACTION_DIM) — the same route in metres, ready
                  to plot beside anything denoise.to_waypoints produces.

    The chain reproduces vint_dataset._compute_actions followed by
    train_utils.get_delta and normalize_data, which together are what the
    checkpoint was fitted against. to_local_coords is imported from upstream
    rather than reimplemented — it is the definition of the robot frame, and a
    second copy of it would be a second definition.
    """
    import pickle

    from vint_train.data.data_utils import to_local_coords

    with open(cfg.traj_dir / "traj_data.pkl", "rb") as fh:
        traj = pickle.load(fh)

    # NUM_ACTIONS steps ahead, plus the current pose to measure them from.
    stop = cfg.frame + settings.NUM_ACTIONS + 1
    positions = np.asarray(traj["position"][cfg.frame:stop], dtype=np.float64)
    yaw = np.asarray(traj["yaw"][cfg.frame:stop], dtype=np.float64).squeeze()
    if len(positions) < settings.NUM_ACTIONS + 1:
        raise ValueError(
            f"{cfg.traj} ends at frame {len(traj['position'])}; frame "
            f"{cfg.frame} leaves fewer than {settings.NUM_ACTIONS} steps of "
            f"ground truth. Pick an earlier frame in config.yaml."
        )

    # Into the robot's own frame at the current instant, then out of metres and
    # into waypoint units — exactly the two lines vint_dataset applies.
    local = to_local_coords(positions, positions[0], yaw[0])
    actions = local[1:] / settings.METRIC_WAYPOINT_SPACING

    # The head predicts differences between waypoints, not the waypoints. Both
    # the differencing and the squash live in common/actions.py, so this is the
    # same arithmetic the descent is decoded with rather than a second copy of
    # it pointed the other way.
    return {
        "deltas":    action_space.to_normalised_deltas(actions),
        "waypoints": actions * settings.METRIC_WAYPOINT_SPACING,
    }
