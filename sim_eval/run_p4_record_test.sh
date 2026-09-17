#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p4_record_test.sh — P4's proof that a run can be watched.
#
# Runs one episode twice on the same task and seed: once filmed, once not. The
# filmed run writes a three-panel MP4 to
# sim_eval/outputs/videos/<checkpoint>/<task_id>.mp4 — robot camera, top-down
# map, and the subgoal/waypoint overlay, all from the same tick. The unfilmed
# run proves the recorder changed none of P3's metrics.
#
# It runs unattended: nothing here waits for a person.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p4_record_test.sh [args for the python script]
#
#     --checkpoint NAME   film a different arm (default: best_combined)
#     --fps N             video frame rate (4 = real time; default: config)
#     --skip-off-check    film only, and skip the second, unfilmed run
#     --rebuild-tasks     rebuild the one-task set even if one is already there
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

echo "==> recording"
# -u: unbuffered, so a long run can be followed with `tail -f` and a crash
# does not take the last few KB of output with it.
sim_exec "python -u sim_eval/p4_1_record_test.py$args"

# The container runs as root, so anything it writes into the bind-mounted repo
# lands root-owned and cannot be deleted from the host. Hand it back.
sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
