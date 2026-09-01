#!/usr/bin/env bash
# Runs the context_stride arms sequentially inside the container, on one GPU.
# Invoked by run_stride_sweep.sh; not meant to be called directly from the host.
#
# The arms are sequential because they share a single GPU, and stop-on-failure means a
# diverged or OOM-ing arm does not leave the remaining arms burning hours behind it.
#
# There is no stride-1 arm: Ablation A1's ctx03 run is the stride-1 baseline, and it is
# reused rather than re-run. Both ablations pin index_context_size = 20, so every arm in
# both studies sees the identical set of (trajectory, curr_time) pairs.
set -euo pipefail

if [ "$#" -gt 0 ]; then
    CONFIGS=("$@")
else
    CONFIGS=(nomad_stride2.yaml nomad_stride3.yaml)
fi
REPO_TRAIN_DIR=/app/visualnav-transformer/train
LOG_DIR=/outputs/nomad_stride_ablation/sweep_logs

cd "${REPO_TRAIN_DIR}"
mkdir -p "${LOG_DIR}"

# `use_wandb: False` is not a usable option here: train_nomad calls wandb.log()
# unconditionally, so disabling wandb crashes the run. Fall back to offline logging when
# no API key is configured, which keeps the console log as the live monitor and leaves
# the runs syncable later with `wandb sync train/wandb/offline-run-*`.
if [ -z "${WANDB_MODE:-}" ] && [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" /root/.netrc; then
    export WANDB_MODE=offline
    echo "No wandb API key found -> WANDB_MODE=offline."
    echo "For live wandb monitoring instead: docker exec -it \$(hostname) wandb login"
fi

for config in "${CONFIGS[@]}"; do
    echo "================================================================"
    echo "[$(date '+%F %T')] START ${config}"
    echo "================================================================"

    # GPU0 only: GPU1 belongs to Ablation A1. Pinned here rather than via gpu_ids,
    # because train.py sets CUDA_VISIBLE_DEVICES after torch has already initialised
    # CUDA, which makes its in-process masking a no-op.
    log_file="${LOG_DIR}/${config%.yaml}_$(date '+%Y%m%d_%H%M%S').log"
    echo "Log: ${log_file}"

    # PIPESTATUS is needed because the pipe through tee would otherwise mask a failure.
    set +e
    # PYTHONUNBUFFERED keeps the step logs flowing through the tee pipe in real time;
    # without it stdout is block-buffered and the log lags by thousands of steps.
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train.py -c "config/${config}" 2>&1 | tee "${log_file}"
    status=${PIPESTATUS[0]}
    set -e

    if [ "${status}" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAILED ${config} (exit ${status}) — stopping the sweep." >&2
        echo "See ${log_file}" >&2
        exit 1
    fi

    echo "[$(date '+%F %T')] DONE ${config}"
done

echo "[$(date '+%F %T')] Sweep complete: ${CONFIGS[*]}"
