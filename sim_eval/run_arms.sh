#!/usr/bin/env bash
# ==============================================================================
# sim_eval/run_arms.sh — score every checkpoint a config names, on one task set.
#
#   1. find the GPUs that are free right now (shared machine — never assume)
#   2. build the fixed, seeded task set once, before anything is scored
#   3. score each checkpoint against it: arms are dealt round-robin across the
#      free GPUs (arm 1 on the first, arm 2 on the second, arm 3 on the first
#      again, ...), each GPU scoring its arms one after the other; with one
#      free GPU, every arm runs there in turn
#   4. roll the tables up: the comparison (p6_2_compare.py), then the paired
#      contrasts and additivity check if the config asks (p7_contrasts.py)
#
# The arms, the seeds, the output directory — all of it comes from the config,
# so P6's headline pair and P7's four-arm 2x2 are one script. P6's launcher
# (run_p6_headline.sh) is this with P6's config.
#
# Resumable. Every scoring run is `--resume`: episodes already in a checkpoint's
# table are skipped, so rerunning this after anything stops it picks up where it
# left off. A run that ends with episodes unscored (a crashed rollout, a killed
# process) is retried, up to MAX_ATTEMPTS times per checkpoint. Resume keys on
# (checkpoint, task, seed) and cannot see *how* a row was scored — so a run
# whose scoring changed (P7's stride fix) must name a fresh `output_dir`.
#
# Long and unattended — launch it inside the existing screen session:
#     screen -S nomad_sim -X screen -t p7 bash -c \
#         'cd /home/nazli/projects/nomad && ./sim_eval/run_arms.sh <config>; exec bash'
# Progress: <output_dir>/<checkpoint>.log, one line per episode.
#
# Usage (from the repo root, on the host):
#     ./sim_eval/run_arms.sh <config> [run_eval args, e.g. --record-tasks 3]
#
#     ARM_GPUS=1 ./sim_eval/run_arms.sh ...     pin the GPUs instead of probing
#     MAX_ATTEMPTS=5 ...                        retries per checkpoint (default 3)
# ==============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source sim_eval/lib.sh

[ $# -ge 1 ] || die "usage: $0 <config> [run_eval args]"
CONFIG=$1
shift
[ -f "$CONFIG" ] || die "no config at $CONFIG"
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

# `checkpoints` and `output_dir` as the config resolves them, `extends` and all.
config_value() {
    sim_exec "cd sim_eval && python -c 'import run_eval; \
c = run_eval.EvalConfig.from_yaml(\"../$CONFIG\"); \
print($1)'" | tail -n 1
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

# Every arm dealt to one GPU, one after the other. Fails if any of them did.
score_lane() {
    local gpu=$1 status=0
    shift
    local arms=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do arms+=("$1"); shift; done
    shift
    for checkpoint in "${arms[@]}"; do
        score_arm "$gpu" "$checkpoint" "$@" > /dev/null || status=1
    done
    return "$status"
}

read -r -a CHECKPOINTS <<<"$(config_value '" ".join(c.checkpoint_names)')"
[ "${#CHECKPOINTS[@]}" -gt 0 ] || die "$CONFIG names no checkpoints."
LOG_DIR=sim_eval/$(config_value 'c.output_dir.relative_to(run_eval.SIM_EVAL_DIR)')
echo "==> config: $CONFIG — arms: ${CHECKPOINTS[*]} — logs: $LOG_DIR"

echo "==> GPUs:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
read -r -a GPUS <<<"${ARM_GPUS:-$(free_gpus)}"
[ "${#GPUS[@]}" -gt 0 ] || die "no free GPU right now — check nvidia-smi and try later."
echo "==> using GPU(s): ${GPUS[*]}"

mkdir -p "$LOG_DIR"

echo "==> [$(stamp)] task set (built once, shared by every checkpoint)"
SIM_GPU=${GPUS[0]} ./sim_eval/run_eval.sh --config "$CONFIG" --build-only

pids=()
for lane in "${!GPUS[@]}"; do
    arms=()
    for index in "${!CHECKPOINTS[@]}"; do
        [ $((index % ${#GPUS[@]})) -eq "$lane" ] && arms+=("${CHECKPOINTS[$index]}")
    done
    [ "${#arms[@]}" -gt 0 ] || continue
    echo "==> [$(stamp)] GPU ${GPUS[$lane]}: ${arms[*]}"
    score_lane "${GPUS[$lane]}" "${arms[@]}" -- "$@" &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" -eq 0 ] || die "a checkpoint is incomplete — see $LOG_DIR/*.log, then rerun this to resume."

echo "==> [$(stamp)] comparison"
./sim_eval/run_p6_2_compare.sh --config "$CONFIG"
echo "==> [$(stamp)] contrasts"
./sim_eval/run_p7_contrasts.sh --config "$CONFIG"
