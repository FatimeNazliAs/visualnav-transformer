#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p3_score_test.sh — P3's proof that the scorer scores.
#
# Builds a small task set (3 tasks in one house), runs every one of them as an
# episode on one checkpoint, and writes the per-episode metrics table to
# sim_eval/outputs/p3_1_score_test.csv. It then re-derives every metric in that
# table from the table itself and fails if anything disagrees.
#
# It runs unattended: nothing here waits for a person.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p3_score_test.sh [args for the python script]
#
#     --checkpoint NAME   score a different arm (default: best_combined)
#     --tasks N           score N tasks instead of 3
#     --rebuild-tasks     rebuild the task set even if one is already there
#     --quiet             one line per episode instead of one per tick
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

echo "==> scoring"
# -u: unbuffered, so a long run can be followed with `tail -f` and a crash
# does not take the last few KB of output with it. docker exec hands Python a
# pipe, not a tty, so without this stdout is block-buffered.
sim_exec "python -u sim_eval/p3_1_score_test.py$args"

# The container runs as root, so anything it writes into the bind-mounted repo
# lands root-owned and cannot be deleted from the host. Hand it back.
sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
