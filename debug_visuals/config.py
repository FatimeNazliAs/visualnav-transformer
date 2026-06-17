# debug_visuals/config.py
"""
Shared configuration for all debug_visuals scripts.

To visualise a different sample, edit the two lines below.

TRAJ_NAME must be a folder inside RAW_DATA_DIR.
FRAME_IDX is the "current" frame t.
  Valid range: >= CONTEXT_SIZE (3)  and  <= total_frames - NUM_ACTIONS - 1
"""

from pathlib import Path
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = Path("/data/raw/go_stanford/go_stanford")

# ── Sample selection  — change these two lines to try a different sample ──────
TRAJ_NAME = "no10vc_10_0"
FRAME_IDX = 30

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model hyperparameters (must match nomad.yaml) ─────────────────────────────
CONTEXT_SIZE  = 3
NUM_ACTIONS   = 8
IMAGE_SIZE    = (96, 96)
ENCODING_SIZE = 256              # encoding_size in yaml
DOWN_DIMS     = [64, 128, 256]   # down_dims in yaml

# ── ImageNet normalisation (must match training pre-processing) ───────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
