#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p6_2_compare.sh — roll the headline run's tables into one.
#
# GPU-free: reads sim_eval/outputs/p6_1_metrics/<checkpoint>.csv, refuses if
# they are not a fair comparison (plan §7), and writes p6_2_comparison.md/.csv
# and p6_2_episodes.csv beside them. run_p6_headline.sh ends by running this.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p6_2_compare.sh [--config sim_eval/configs/p6_headline.yaml]
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

args=""
for arg in "$@"; do
    args="$args $(printf '%q' "$arg")"
done

sim_exec "python -u sim_eval/p6_2_compare.py$args"

sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
