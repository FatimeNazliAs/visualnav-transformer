#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p0_smoke_test.sh — P0's proof that the simulator renders.
#
# Drives the LoCoBot forward in Rs and writes sim_eval/outputs/p0_1_camera.png.
# Exits non-zero if the frame comes back blank.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p0_smoke_test.sh [extra args passed to the python script]
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

sim_exec "python sim_eval/p0_1_smoke_test.py$args"

# The container runs as root, so anything it writes into the bind-mounted repo
# lands root-owned and cannot be deleted from the host. Hand it back.
sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
