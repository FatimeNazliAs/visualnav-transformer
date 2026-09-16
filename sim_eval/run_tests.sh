#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_tests.sh — the GPU-free unit tests.
#
# They need no GPU, no iGibson, no simulator and no checkpoint, but they do need
# pytest and numpy, which live in the container's conda env and not on the host.
# So they run the same way everything else here does.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_tests.sh [extra pytest args]
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

args=""
for arg in "$@"; do
    args="$args $(printf '%q' "$arg")"
done

sim_exec "python -m pytest sim_eval/tests -q$args"
