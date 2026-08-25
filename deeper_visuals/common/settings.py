# deeper_visuals/common/settings.py
"""
Constants shared by every deeper_visuals phase.

Two kinds of thing live here:

  1. Container paths — where the data and the trained runs are mounted.
  2. Model hyperparameters — these MUST stay in sync with the values the
     checkpoint was trained under (train/config/nomad.yaml). They are not
     tunable knobs; changing one here without retraining silently produces
     a model that loads but is wrong.

Everything that IS meant to vary per phase (which scene, which weight file)
lives in the phase's config.yaml and is resolved by common/config.py.
"""

from __future__ import annotations

from pathlib import Path

# ── Container paths ───────────────────────────────────────────────────────────
# Host -> container mounts (see deeper_visuals/README.md for the docker run):
#   /mnt/shared_disk/nazli/nomad_data    -> /data
#   /mnt/shared_disk/nazli/nomad_outputs -> /outputs
#   /home/nazli/projects/nomad           -> /app/visualnav-transformer
RAW_DATA_DIR = Path("/data/raw/go_stanford/go_stanford")
RUNS_DIR     = Path("/outputs/nomad")

# The confirmed training run: 100/100 epochs (nomad.yaml sets epochs: 100), a
# clean ~14 h run holding 0.pth–99.pth, ema_0–ema_99 and latest.pth. The 15
# aborted same-day runs have been deleted, so this is the only run on disk.
DEFAULT_RUN = "nomad_2026_06_13_18_04_23"

# What `checkpoint:` in a phase's config.yaml resolves to. Fixed map, not a
# search — the run is settled, so a phase switches weights by editing one word.
CHECKPOINT_FILES = {
    "ema":    "ema_99.pth",   # EMA at the final epoch
    "latest": "latest.pth",   # raw epoch-99 weights
}

# EMA is the default: it is what NoMaD's own evaluate_nomad runs on, and it
# gives cleaner trajectories in the advisor figures.
DEFAULT_CHECKPOINT = "ema"

# ── Model hyperparameters (must match train/config/nomad.yaml) ────────────────
CONTEXT_SIZE  = 3                # nomad.yaml: context_size
NUM_ACTIONS   = 8                # nomad.yaml: len_traj_pred
IMAGE_SIZE    = (96, 96)         # (H, W)
ENCODING_SIZE = 256              # nomad.yaml: encoding_size
DOWN_DIMS     = [64, 128, 256]   # ConditionalUnet1D down_dims
ACTION_DIM    = 2                # learn_angle=False -> (linear_vel, angular_vel)
K_DENOISING   = 10               # nomad.yaml: num_diffusion_iters

# Transformer geometry (NoMaD_ViNT defaults used at training time)
MHA_NUM_ATTENTION_HEADS  = 4
MHA_NUM_ATTENTION_LAYERS = 4
MHA_FF_DIM_FACTOR        = 4

# go_stanford's average spacing between consecutive frames, in metres, from
# train/vint_train/data/data_config.yaml. It is a DISTANCE, not a time — the
# repo records no frame rate anywhere, so the goal's offset can be stated in
# metres but never in seconds.
METRIC_WAYPOINT_SPACING = 0.12

# The range the action deltas were squashed into before training, from
# train/vint_train/data/data_config.yaml's `action_stats`. Every value the
# diffusion head emits lives in [-1, 1] and has to be mapped back through these
# to mean anything; see common/denoise.to_waypoints for the arithmetic.
#
# Order is (forward, left) — vint_dataset._compute_actions builds the waypoints
# with to_local_coords, a yaw rotation into the robot's own frame, so +x is
# straight ahead and +y is to its left. A negative second number is a right turn.
ACTION_MIN = (-2.5, -4.0)   # (min_dx, min_dy), in waypoint units
ACTION_MAX = (5.0, 4.0)     # (max_dx, max_dy), in waypoint units

# ── ImageNet normalisation (must match training pre-processing) ───────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── Derived ───────────────────────────────────────────────────────────────────
N_OBS_FRAMES = CONTEXT_SIZE + 1   # 3 past + 1 current = 4 frames into psi
N_TOKENS     = N_OBS_FRAMES + 1   # 4 obs tokens + 1 goal token = 5
