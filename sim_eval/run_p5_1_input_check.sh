#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p5_1_input_check.sh — is the model being fed what it was trained on?
#
# Puts one simulator frame and one GoStanford frame through the same transform
# the policy uses, and reports channel order, value range, aspect and crop,
# field of view and camera height side by side. It writes
# sim_eval/outputs/p5_1_input_check.png (the two sources at every stage of the
# pipeline, plus a strip of each) and p5_1_input_check.json (the numbers).
#
# It changes nothing. This is evidence for a decision, not a fix.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p5_1_input_check.sh [args for the python script]
#
#     --checkpoint NAME   whose image_size to resize to (default: best_combined)
#     --frames N          frames per source in the strip (default: 6)
#     --seed N            which frames and poses get sampled (default: 0)
#     --dataset PATH      GoStanford root (default: /data/raw/go_stanford/go_stanford)
#
# Pick the GPU with SIM_GPU=N. Check `nvidia-smi` first — shared machine.
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

echo "==> GPU $SIM_GPU, before:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# sim_exec takes a command string, so each argument is re-quoted before it is
# flattened into one — otherwise bash -lc re-splits them and any path with a
# space breaks.
args=""
for arg in "$@"; do
    args="$args $(printf '%q' "$arg")"
done

echo "==> comparing the two sources"
# -u: unbuffered, so the report appears as it is produced rather than in one
# block when the process exits.
sim_exec "python -u sim_eval/p5_1_input_check.py$args"

# The container runs as root, so anything it writes into the bind-mounted repo
# lands root-owned and cannot be deleted from the host. Hand it back.
sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
