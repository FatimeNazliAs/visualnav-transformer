# behaviour-failure-analysis/common/settings.py
"""
Constants shared by every behaviour-analysis task.

Lifted verbatim from deeper_visuals/common/settings.py (branch
feature/deeper-visualization), where every value was checked against the
checkpoint's train/config/nomad.yaml. Re-verified on 2026-08-25 against
train/config/nomad.yaml and train/vint_train/data/data_config.yaml at commit
32ef15e, which is the config the run was trained under.

Two kinds of thing live here:

  1. Container paths — where the data and the trained runs are mounted.
  2. Model hyperparameters — these MUST stay in sync with the values the
     checkpoint was trained under (train/config/nomad.yaml). They are not
     tunable knobs; changing one here without retraining silently produces
     a model that loads but is wrong.

Everything that IS meant to vary per phase (which scene, which weight file)
lives in a task's config.yaml and is resolved by common/config.py.
"""

from __future__ import annotations

from pathlib import Path

# ── Container paths ───────────────────────────────────────────────────────────
# Host -> container mounts (see behaviour-failure-analysis/README.md):
#   /mnt/shared_disk/nazli/nomad_data    -> /data
#   /mnt/shared_disk/nazli/nomad_outputs -> /outputs
#   /home/nazli/projects/nomad           -> /app/visualnav-transformer
RAW_DATA_DIR = Path("/data/raw/go_stanford/go_stanford")
RUNS_DIR     = Path("/outputs/nomad")

# The train/test split lists, one trajectory name per line. This project reads
# them; deeper_visuals never did, and that is the single most important
# difference between the two. Its curated samples.yaml turned out to be nine
# parts training data — fine for explaining an architecture, disqualifying for
# a behaviour analysis, where a case drawn from `train` measures memorisation
# rather than behaviour. Every case here is selected from TEST_SPLIT.
SPLITS_DIR  = Path("/data/splits/go_stanford")
TRAIN_SPLIT = SPLITS_DIR / "train" / "traj_names.txt"
TEST_SPLIT  = SPLITS_DIR / "test"  / "traj_names.txt"

# Where this project writes. On the shared disk, not the repo: the host root
# filesystem is at 99% capacity, and results are large and regenerable.
RESULTS_ROOT = Path("/outputs/behaviour_analysis")

# The confirmed training run: 100/100 epochs (nomad.yaml sets epochs: 100), a
# clean ~14 h run holding 0.pth–99.pth, ema_0–ema_99 and latest.pth. The 15
# aborted same-day runs have been deleted, so this is the only run on disk.
DEFAULT_RUN = "nomad_2026_06_13_18_04_23"

# What `checkpoint:` in a task's config.yaml resolves to. Fixed map, not a
# search — the run is settled, so a task switches weights by editing one word.
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
# The two action channels are NORMALISED POSITION DELTAS, not velocities.
# `learn_angle: False`, so each of the NUM_ACTIONS rows is (dx, dy) in the
# robot's frame, squashed into [-1, 1] by ACTION_MIN/ACTION_MAX below. Decoding
# them means un-normalise -> cumsum -> x METRIC_WAYPOINT_SPACING, which is what
# common/actions.to_waypoints does. Reading them as (linear, angular) velocity
# through a unicycle integrator — as the frozen debug_visuals/visualize_stage5.py
# does — is wrong, and wrong in a way that changes the SHAPE of the path rather
# than only its size, because the affine step does not commute with the sum.
ACTION_DIM    = 2
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

# How far ahead the goal frame may sit, in frames — nomad.yaml's `action:
# min_dist_cat / max_dist_cat`. During training the goal was sampled uniformly
# from this range, so any value in it is in-distribution and NUM_ACTIONS (8) is
# merely the middle of it, not a required value.
#
# Goal distance and the prediction horizon are independent: the head always
# emits NUM_ACTIONS waypoints (`len_traj_pred: 8`) regardless of how far away
# the goal is. deeper_visuals' PhaseConfig tied them together — goal_idx was
# hard-coded to frame + NUM_ACTIONS — because a phase had exactly one scene and
# never needed to vary one without the other. This project varies goal distance
# deliberately, so common/config.py keeps them apart.
GOAL_DISTANCE_MIN = 3
GOAL_DISTANCE_MAX = 20

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
