#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_lg7.sh — run one language-goal Phase 7 script in the sim container.
#
# Every lg7_* script (and run_eval.py in object-goal mode) goes through here,
# so all of them run in naz_nomad_clip_sim, pinned to $SIM_GPU (lib.sh).
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_lg7.sh lg7_1_annotations.py --scene Rs
#     SIM_GPU=0 ./sim_eval/run_lg7.sh run_eval.py --config sim_eval/configs/word_goal.yaml ...
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

[ $# -ge 1 ] || die "usage: $0 <script.py> [args]"
script=$1
shift

echo "==> GPU $SIM_GPU, before:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

args=""
for arg in "$@"; do
    args="$args $(printf '%q' "$arg")"
done

status=0
sim_exec "python -u sim_eval/$script$args" || status=$?

sim_exec "chown -R $(id -u):$(id -g) sim_eval/outputs"
exit $status
