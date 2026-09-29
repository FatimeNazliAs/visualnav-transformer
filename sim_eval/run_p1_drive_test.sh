#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p1_drive_test.sh — P1's proof that the bridge drives.
#
# Runs one NoMaD checkpoint closed-loop along the hand-made topomap and writes
# per-tick frames plus a GIF to sim_eval/outputs/p1_1_drive_test/.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p1_drive_test.sh [--make-topomap] [args for the python script]
#
#     --make-topomap   drive the reference route first and rebuild the topomap
#                      in sim_eval/outputs/p1_0_topomap/ (needed once)
#
# Pick the GPU with SIM_GPU=N. Check `nvidia-smi` first — shared machine.
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

MAKE_TOPOMAP=0
if [ "${1:-}" = "--make-topomap" ]; then
    MAKE_TOPOMAP=1
    shift
fi

echo "==> GPU $SIM_GPU, before:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# sim_exec takes a command string, so each argument is re-quoted before it is
# flattened into one — otherwise bash -lc re-splits them and any path with a
# space breaks.
args=""
for arg in "$@"; do
    args="$args $(printf '%q' "$arg")"
done

if [ "$MAKE_TOPOMAP" = 1 ]; then
    echo "==> building the reference topomap"
    sim_exec "python sim_eval/p1_0_make_topomap.py"
fi

echo "==> driving"
sim_exec "python sim_eval/p1_1_drive_test.py$args"

# The container runs as root, so anything it writes into the bind-mounted repo
# lands root-owned and cannot be deleted from the host. Hand it back.
sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
