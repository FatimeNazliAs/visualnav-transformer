#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_eval.sh — score one checkpoint over the whole task set.
#
# The full version of what run_p3_score_test.sh proves in miniature: build the
# shared task set if it is not there, then run every task on one checkpoint and
# append one row per episode to sim_eval/outputs/p3_2_metrics/<checkpoint>.csv.
#
# Long and unattended. Run it inside the screen session (`screen -r nomad_sim`)
# so it survives the SSH connection dropping.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_eval.sh --checkpoint best_combined [more args]
#
#     --checkpoint NAME   which arm (default: the first in configs/eval.yaml)
#     --build-only        build the shared task set and stop
#     --resume            append to an existing CSV instead of restarting it
#     --quiet             one line per episode instead of one per tick
#
# One checkpoint per GPU is how P6 runs the headline pair in parallel:
#     SIM_GPU=0 ./sim_eval/run_eval.sh --checkpoint best_combined
#     SIM_GPU=1 ./sim_eval/run_eval.sh --checkpoint clean_stock
# Both read the same task set, and neither writes to it. Check `nvidia-smi`
# first, and only take a GPU that is free — shared machine.
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

echo "==> GPU $SIM_GPU, before:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

args=""
for arg in "$@"; do
    args="$args $(printf '%q' "$arg")"
done

echo "==> evaluating"
# -u: unbuffered, so a run that takes an hour can be followed with `tail -f`.
# docker exec hands Python a pipe, not a tty, so stdout is block-buffered.
sim_exec "python -u sim_eval/run_eval.py$args"

sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
