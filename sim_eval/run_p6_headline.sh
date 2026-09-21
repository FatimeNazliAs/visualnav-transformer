#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_p6_headline.sh — the headline run: both checkpoints, one task set.
#
#   1. find the GPUs that are free right now (shared machine — never assume)
#   2. build the fixed, seeded task set once, before anything is scored
#   3. score each checkpoint against it: one per GPU in parallel when two are
#      free, one after the other on the single free GPU otherwise
#   4. roll both tables up into the comparison (p6_2_compare.py)
#
# Resumable. Every scoring run is `--resume`: episodes already in a checkpoint's
# table are skipped, so rerunning this after anything stops it picks up where it
# left off. A run that ends with episodes unscored (a crashed rollout, a killed
# process) is retried, up to MAX_ATTEMPTS times per checkpoint.
#
# Long and unattended — launch it inside the existing screen session:
#     screen -S nomad_sim -X screen -t p6 bash -c \
#         'cd /home/nazli/projects/nomad && ./sim_eval/run_p6_headline.sh; exec bash'
# Progress: sim_eval/outputs/p6_1_metrics/<checkpoint>.log, one line per episode.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_p6_headline.sh [run_eval args, e.g. --record-tasks 3]
#
#     P6_GPUS=1 ./sim_eval/run_p6_headline.sh     pin the GPUs instead of probing
#     MAX_ATTEMPTS=5 ...                           retries per checkpoint (default 3)
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

CONFIG=sim_eval/configs/p6_headline.yaml
CHECKPOINTS=(best_combined clean_stock)   # GPU order: first arm on first GPU
LOG_DIR=sim_eval/outputs/p6_1_metrics
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
# A GPU is free if nobody runs a compute job on it and it holds less than this.
# GPU 0 always holds ~250 MiB of X server and desktop — that is not a job.
FREE_MEMORY_MIB=1024

# Indices of the GPUs nobody is computing on, space-separated.
free_gpus() {
    local busy
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
    nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits |
        while IFS=', ' read -r index uuid used; do
            if [ "$used" -lt "$FREE_MEMORY_MIB" ] && ! grep -q "$uuid" <<<"$busy"; then
                printf '%s ' "$index"
            fi
        done
}

stamp() { date '+%F %T'; }

# Score one checkpoint on one GPU until its table is complete, or out of tries.
score_arm() {
    local gpu=$1 checkpoint=$2
    shift 2
    local log="$LOG_DIR/$checkpoint.log"
    for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
        echo "[$(stamp)] $checkpoint on GPU $gpu, attempt $attempt/$MAX_ATTEMPTS" | tee -a "$log"
        if SIM_GPU=$gpu ./sim_eval/run_eval.sh --config "$CONFIG" \
                --checkpoint "$checkpoint" --resume --quiet "$@" 2>&1 | tee -a "$log"; then
            echo "[$(stamp)] $checkpoint complete" | tee -a "$log"
            return 0
        fi
        echo "[$(stamp)] $checkpoint stopped with episodes unscored; resuming" | tee -a "$log"
    done
    echo "[$(stamp)] $checkpoint INCOMPLETE after $MAX_ATTEMPTS attempts" | tee -a "$log"
    return 1
}

echo "==> GPUs:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
read -r -a GPUS <<<"${P6_GPUS:-$(free_gpus)}"
[ "${#GPUS[@]}" -gt 0 ] || die "no free GPU right now — check nvidia-smi and try later."
echo "==> using GPU(s): ${GPUS[*]}"

mkdir -p "$LOG_DIR"

echo "==> [$(stamp)] task set (built once, shared by every checkpoint)"
SIM_GPU=${GPUS[0]} ./sim_eval/run_eval.sh --config "$CONFIG" --build-only

status=0
if [ "${#GPUS[@]}" -ge 2 ]; then
    echo "==> [$(stamp)] parallel: ${CHECKPOINTS[0]} on GPU ${GPUS[0]}, ${CHECKPOINTS[1]} on GPU ${GPUS[1]}"
    score_arm "${GPUS[0]}" "${CHECKPOINTS[0]}" "$@" > /dev/null &
    first=$!
    score_arm "${GPUS[1]}" "${CHECKPOINTS[1]}" "$@" > /dev/null &
    second=$!
    wait "$first" || status=1
    wait "$second" || status=1
else
    echo "==> [$(stamp)] sequential on GPU ${GPUS[0]}"
    for checkpoint in "${CHECKPOINTS[@]}"; do
        score_arm "${GPUS[0]}" "$checkpoint" "$@" || status=1
    done
fi

[ "$status" -eq 0 ] || die "a checkpoint is incomplete — see $LOG_DIR/*.log, then rerun this to resume."

echo "==> [$(stamp)] comparison"
./sim_eval/run_p6_2_compare.sh
