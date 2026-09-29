#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_lg7_0_port_check.sh — language-goal G7.0: re-score one stored
# episode (ctx03, Rs_00, seed offset 0) and compare its row byte for byte.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_lg7_0_port_check.sh
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

echo "==> G7.0 port check"
status=0
sim_exec "python -u sim_eval/lg7_0_port_check.py$args" || status=$?

sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
exit $status
