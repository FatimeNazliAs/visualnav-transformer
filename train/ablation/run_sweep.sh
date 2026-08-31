#!/usr/bin/env bash
# Launch the context_size ablation sweep. Run this from the HOST, from the repo root.
#
#   ./train/ablation/run_sweep.sh                      # all three arms
#   ./train/ablation/run_sweep.sh nomad_ctx03.yaml     # one arm (e.g. a sanity run)
#
# Starts one detached screen driving one container, running the arms sequentially on
# GPU1. Attach with:  screen -r nomad_ctx_ablation
set -euo pipefail

CONTAINER=naz_nomad_ctx_ablation
SCREEN=nomad_ctx_ablation
IMAGE=nomad:latest
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if screen -list | grep -q "\.${SCREEN}[[:space:]]"; then
    echo "Screen '${SCREEN}' is already running. Attach with: screen -r ${SCREEN}" >&2
    exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
        echo "Starting existing container ${CONTAINER}..."
        docker start "${CONTAINER}" >/dev/null
    else
        echo "Creating container ${CONTAINER}..."
        docker run -d --name "${CONTAINER}" --gpus all --shm-size=8g \
            -v "${REPO_ROOT}:/app/visualnav-transformer" \
            -v /mnt/shared_disk/nazli/nomad_data:/data \
            -v /mnt/shared_disk/nazli/nomad_outputs:/outputs \
            -v /mnt/shared_disk/nazli/wandb_config:/root/.config/wandb \
            "${IMAGE}" sleep infinity >/dev/null
    fi
fi

echo "Launching sweep in screen '${SCREEN}' (container ${CONTAINER}, GPU1)..."
screen -dmS "${SCREEN}" \
    docker exec "${CONTAINER}" bash /app/visualnav-transformer/train/ablation/sweep_driver.sh "$@"

echo "Attach with:  screen -r ${SCREEN}"
echo "Watch GPU:    watch -n5 nvidia-smi"
