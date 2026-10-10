#!/bin/bash
# Phase 2 drive generation on both GPUs: N workers per GPU, each in its own screen session (survives SSH drops).
#
#   mapmad/scripts/run_p2_gen.sh pilot 2      # 2 workers per GPU -> 4 workers, screens mapmad_p2_pilot_w0..w3
#   mapmad/scripts/run_p2_gen.sh full 3       # screens mapmad_p2_full_w0..w5
#   GPUS="1" mapmad/scripts/run_p2_gen.sh full 3   # only GPU1 (e.g. another user's job on GPU0)
#   OUT_NAME=bench_w8 mapmad/scripts/run_p2_gen.sh pilot 4   # write to <mapmad_data>/bench_w8 (speed tests)
#
# Worker k runs on GPU GPUS[k % n_gpus]. Run nvidia-smi first (plan §0.4). Restarting is safe: finished homes are
# skipped. Logs: <outputs>/p2_datagen/<dataset>/screen_w<k>.log and progress_w<k>of<n>.jsonl.
set -euo pipefail

DATASET="${1:?pilot or full}"
PER_GPU="${2:-2}"
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../docker/paths.env
source "$HERE/../docker/paths.env"
LOGS="$NOMAD_OUTPUTS/mapmad/p2_datagen/${OUT_NAME:-$DATASET}"
EXTRA=${OUT_NAME:+--out-name $OUT_NAME}
mkdir -p "$LOGS"

WORKERS=$(( PER_GPU * ${#GPU_LIST[@]} ))
for (( k = 0; k < WORKERS; k++ )); do
    gpu="${GPU_LIST[$(( k % ${#GPU_LIST[@]} ))]}"
    name="mapmad_p2_${OUT_NAME:-$DATASET}_w$k"
    cmd="docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/generate_drives.py \
--dataset $DATASET --gpu $gpu --worker $k --workers $WORKERS $EXTRA"
    echo "+ screen -dmS $name -L -Logfile $LOGS/screen_w$k.log bash -c \"$cmd\""
    screen -dmS "$name" -L -Logfile "$LOGS/screen_w$k.log" bash -c "$cmd"
done
