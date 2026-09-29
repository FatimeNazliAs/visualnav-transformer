#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p7_contrasts.sh — paired contrasts and the additivity check.
#
# GPU-free: reads the tables a run's config names, refuses if they are not a
# fair comparison (plan §7), and writes <name>_paired.md/.csv,
# <name>_paired_tasks.csv and, when the config asks, <name>_additivity.md/.csv
# beside them. Every statistic takes the task as its unit (task_stats.py).
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p7_contrasts.sh --config sim_eval/configs/p7_2_e1.yaml
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

args=""
for arg in "$@"; do
    args="$args $(printf '%q' "$arg")"
done

sim_exec "python -u sim_eval/p7_contrasts.py$args"

sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
