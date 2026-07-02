# debug_visuals/config.py
"""
Shared configuration for all debug_visuals scripts.

To visualise a different sample, change SAMPLE below to one of the keys in
SAMPLES (or add your own entry). TRAJ_NAME / FRAME_IDX are derived from it.

TRAJ_NAME must be a folder inside RAW_DATA_DIR.
FRAME_IDX is the "current" frame t.
  Valid range: >= CONTEXT_SIZE (3)  and  <= total_frames - NUM_ACTIONS - 1
"""

from pathlib import Path
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
# In-container mount of the host data dir (/mnt/shared_disk/nazli/nomad_data).
RAW_DATA_DIR = Path("/data/raw/go_stanford/go_stanford")

# ── Sample library ────────────────────────────────────────────────────────────
# Curated go_stanford samples, each verified to sit in the valid frame range and
# to show clear, interpretable motion — good for explaining the pipeline.
# Grouped by the behaviour the future (red) path shows.
#   key : (TRAJ_NAME, FRAME_IDX, description)
SAMPLES = {
    # straight ahead
    "hallway_straight":  ("no9vc_46_0",  189, "clean corridor, robot drives straight ahead"),
    "person_straight":   ("no6vc_125_2", 126, "hallway with a person ahead, straight motion"),
    "lobby_straight":    ("no10vc_6_0",   55, "straight into a lit lobby / entrance"),
    # gentle curves
    "gentle_left":       ("no12vc_17_0",  15, "gentle left curve through a red-carpet lobby"),
    "gentle_right":      ("no10vc_22_0", 142, "gentle right curve down a hallway"),
    # sharp turns
    "left_turn":         ("no31vc_25_0",  11, "sharp left turn in a corridor"),
    "left_turn_person":  ("no9vc_54_0",   19, "left turn in a store aisle, person ahead"),
    "right_turn":        ("no3vc_62_0",    6, "right turn near a stairwell"),
    "right_turn_doors":  ("no11vc_9_1",   54, "right turn toward glass doors"),
    # original
    "original":          ("no10vc_10_0",  30, "the original sample"),
}

# ── Sample selection  — change this one line to try a different sample ────────
SAMPLE = "gentle_left"
TRAJ_NAME, FRAME_IDX, _SAMPLE_DESC = SAMPLES[SAMPLE]

# To use a sample not in the table, just override the two lines directly, e.g.:
#   TRAJ_NAME, FRAME_IDX = "no2vc_21_0", 1

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model hyperparameters (must match nomad.yaml) ─────────────────────────────
CONTEXT_SIZE  = 3
NUM_ACTIONS   = 8
IMAGE_SIZE    = (96, 96)
ENCODING_SIZE = 256              # encoding_size in yaml
DOWN_DIMS     = [64, 128, 256]   # down_dims in yaml
ACTION_DIM    = 2                # learn_angle=False -> (linear_vel, angular_vel) only
K_DENOISING   = 10               # num_diffusion_iters in yaml

# ── ImageNet normalisation (must match training pre-processing) ───────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── Checkpoint  — switch EMA vs. latest by commenting / uncommenting ──────────
# Both weight files live in the same run folder. To switch, move the "#" so that
# exactly one CHECKPOINT line is active:
#     ema_*.pth   → PNGs go to debug_visuals/outputs_ema/
#     latest.pth  → PNGs go to debug_visuals/outputs_latest/
# The output folder is tagged by the file name (below), so the two never
# overwrite each other and you can compare EMA vs. latest side by side.
_CKPT_DIR  = Path("/outputs/nomad/nomad_2026_06_13_18_04_23")
# CHECKPOINT = _CKPT_DIR / "ema_99.pth"      # <-- active: EMA weights
CHECKPOINT = _CKPT_DIR / "latest.pth"    # <-- swap the "#" to use these instead

# ── Output directory ──────────────────────────────────────────────────────────
_CKPT_TAG   = "ema" if CHECKPOINT.stem.lower().startswith("ema") else "latest"
OUTPUTS_DIR = Path(__file__).resolve().parent / f"outputs_{_CKPT_TAG}"

# ── Derived values (single source of truth for the whole pipeline) ────────────
# The per-sample output folder and the frame indices the pipeline loads.
# Centralised here so run_pipeline and every stage agree without recomputing.
RUN_DIR  = OUTPUTS_DIR / f"{TRAJ_NAME}_f{FRAME_IDX}"
OBS_IDXS = list(range(FRAME_IDX - CONTEXT_SIZE, FRAME_IDX + 1))   # [t-3 … t]
GOAL_IDX = FRAME_IDX + NUM_ACTIONS                                # t + NUM_ACTIONS
