#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p7_4_film.sh — re-film the P7 clips (configs/p7_4_clips.yaml).
#
# Needs a GPU: every listed (arm, task, seed) episode is run again with
# recording on, and checked against the row its scoring run logged. Pin the
# GPU with SIM_GPU (lib.sh) after checking nvidia-smi.
#
# Usage (from the repo root, on the host):
#     SIM_GPU=1 ./sim_eval/run_p7_4_film.sh [--config sim_eval/configs/p7_4_clips.yaml]
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

args=""
for arg in "$@"; do
    args="$args $(printf '%q' "$arg")"
done

sim_exec "python -u sim_eval/p7_4_film.py$args"

sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
