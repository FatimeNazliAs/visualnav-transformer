# debug/config.py
from pathlib import Path
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
# Inside the container, the dataset is mounted at /data/ (not /mnt/shared_disk)
RAW_DATA_DIR   = Path("/data/raw/go_stanford/go_stanford")
SPLIT_FILE     = Path("/data/splits/go_stanford/train/traj_names.txt")
CHECKPOINT = Path("/outputs/nomad/nomad_2026_06_13_18_04_23/latest.pth")

# ── Sample selection ───────────────────────────────────────────────────────────
# Change these two lines to debug a different sample.
# TRAJ_NAME must be a folder name inside RAW_DATA_DIR.
# FRAME_IDX is the "current" frame t. Valid range: >= CONTEXT_SIZE (3) and <= T-9
TRAJ_NAME  = "no10vc_10_0"
FRAME_IDX  = 30

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model hyperparameters — must match nomad.yaml exactly ─────────────────────
CONTEXT_SIZE   = 3      # P: past observation frames (context_size in yaml)
IMAGE_SIZE     = (96, 96)
ENCODING_SIZE  = 256    # encoding_size in yaml
NUM_ACTIONS    = 8      # len_traj_pred in yaml
ACTION_DIM     = 2      # learn_angle=False → (linear_vel, angular_vel) only
K_DENOISING    = 10     # num_diffusion_iters in yaml
LEARN_ANGLE    = False  # learn_angle in yaml
DOWN_DIMS      = [64, 128, 256]  # down_dims in yaml

# ── ImageNet normalisation (must match training preprocessing) ─────────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]