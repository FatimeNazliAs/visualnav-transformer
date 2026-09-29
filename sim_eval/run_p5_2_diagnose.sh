#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p5_2_diagnose.sh — a few episodes, filmed and taken apart.
#
# Runs a small task set (3 by default, one scene, one checkpoint) with the
# recorder on, then reads the traces back and names why each episode ended
# where it did. Everything lands in sim_eval/outputs/p5_2_diagnostics/:
# the metrics table, the traces, one flat per-tick CSV per episode, the
# three-panel videos, and the diagnosis table.
#
# It tunes nothing. The driver settings, the episode rules and the task set are
# exactly the config's — a run of this and a run of `run_eval.sh --record` over
# the same tasks write the same rows.
#
# It runs unattended, but it is slow (a few minutes per episode, plus the
# reference drives the first time). Run it inside `screen -r nomad_sim` if the
# SSH connection is unreliable.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p5_2_diagnose.sh [args for the python script]
#
#     --checkpoint NAME   which arm to diagnose (default: best_combined)
#     --tasks N           how many tasks in the scene (default: 3)
#     --rebuild-tasks     rebuild the task set even if one is already there
#     --no-record         skip the videos; the numbers are unchanged
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

echo "==> diagnosing"
# -u: unbuffered, so a long run can be followed with `tail -f` and a crash
# does not take the last few KB of output with it.
sim_exec "python -u sim_eval/p5_2_diagnose.py$args"

# The container runs as root, so anything it writes into the bind-mounted repo
# lands root-owned and cannot be deleted from the host. Hand it back.
sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
