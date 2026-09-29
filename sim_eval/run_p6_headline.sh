#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p6_headline.sh — the headline run: best_combined vs clean_stock
# on P6's task set. run_arms.sh with P6's config; see it for how the arms are
# split across GPUs, resumed and rolled up.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p6_headline.sh [run_eval args, e.g. --record-tasks 3]
# ==============================================================================
exec "$(dirname "$0")/run_arms.sh" sim_eval/configs/p6_headline.yaml "$@"
