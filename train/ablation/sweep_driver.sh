#!/usr/bin/env bash
# Runs one GPU's queue of image_size arms sequentially inside the container.
# Invoked by run_sweep.sh as: sweep_driver.sh <gpu> <config>...
#
# Arms on the same GPU are sequential because they share it; stop-on-failure means a
# diverged or OOM-ing arm does not leave the rest of that queue burning hours behind it.
# The two GPUs' queues are independent, so a failure on one does not stop the other.
set -euo pipefail

GPU="$1"
shift
CONFIGS=("$@")
REPO_TRAIN_DIR=/app/visualnav-transformer/train
LOG_DIR=/outputs/nomad_img_ablation/sweep_logs

cd "${REPO_TRAIN_DIR}"
mkdir -p "${LOG_DIR}"

# `use_wandb: False` is not a usable option here: train_nomad calls wandb.log()
# unconditionally, so disabling wandb crashes the run. Fall back to offline logging when
# no API key is configured, which keeps the console log as the live monitor and leaves
# the runs syncable later with `wandb sync train/wandb/offline-run-*`.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" /root/.netrc; then
    export WANDB_MODE=offline
    echo "No wandb API key found -> WANDB_MODE=offline."
fi

for config in "${CONFIGS[@]}"; do
    echo "================================================================"
    echo "[$(date '+%F %T')] START ${config} on GPU${GPU}"
    echo "================================================================"

    log_file="${LOG_DIR}/${config%.yaml}_$(date '+%Y%m%d_%H%M%S').log"
    echo "Log: ${log_file}"

    # CUDA_VISIBLE_DEVICES is pinned here rather than via gpu_ids, because train.py sets
    # it after torch has already initialised CUDA, which makes its masking a no-op.
    # PYTHONUNBUFFERED keeps step logs flowing through tee in real time.
    set +e
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
        python train.py -c "config/${config}" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [ "${status}" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED ${config} (exit ${status}) — stopping GPU${GPU}'s queue." >&2
        echo "See ${log_file}" >&2
        exit 1
    fi

    echo "[$(date '+%F %T')] DONE ${config}"
done

echo "[$(date '+%F %T')] GPU${GPU} queue complete: ${CONFIGS[*]}"
