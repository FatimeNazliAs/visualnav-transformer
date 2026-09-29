#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p2_build_test.sh — P2's proof that the topomap builder works.
#
# Builds one reference-path trail from sim_eval/configs/topomap.yaml and writes
# it, a top-down path plot and a thumbnail strip to sim_eval/outputs/.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p2_build_test.sh [args for the python script]
#
#     --config PATH   build from a different topomap config
#     --output PATH   write the trail somewhere else
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

echo "==> building the reference-path topomap"
sim_exec "python sim_eval/p2_1_build_test.py$args"

# The container runs as root, so anything it writes into the bind-mounted repo
# lands root-owned and cannot be deleted from the host. Hand it back.
sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
