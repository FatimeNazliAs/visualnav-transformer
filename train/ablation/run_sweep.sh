#!/usr/bin/env bash
# Launch the image_size ablation sweep (Ablation B). Run from the HOST, from the repo root.
#
#   ./train/ablation/run_sweep.sh            # both GPUs, all three new arms
#   ./train/ablation/run_sweep.sh --status
#
# img096 is NOT run here: it is A1's ctx03, reused as the baseline (see
# B_IMAGE_SIZE_README.md). The three new arms are split across the two GPUs by
# projected runtime, longest arm alone:
#
#   GPU0 : img160x120                 ~5h52m
#   GPU1 : img112 -> img128x96        ~8h03m   <- makespan
#
# Each GPU's queue runs detached inside the container (docker exec -d), so it is
# parented by the container rather than by this shell and survives SSH disconnect.
set -euo pipefail

CONTAINER=naz_nomad_img_ablation
IMAGE=nomad:latest
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_OUTPUTS=/mnt/shared_disk/nazli/nomad_outputs
SWEEP_LOG_DIR="${HOST_OUTPUTS}/nomad_img_ablation/sweep_logs"

# GPU -> the arms it runs, in order.
GPU0_ARMS="nomad_img160x120.yaml"
GPU1_ARMS="nomad_img112.yaml nomad_img128x96.yaml"

if [ "${1:-}" = "--status" ]; then
    echo "== container process table =="
    docker top "${CONTAINER}" 2>/dev/null || echo "container ${CONTAINER} is not running"
    echo
    echo "== GPU =="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
    echo
    echo "== newest log per arm =="
    for arm in img112 img128x96 img160x120; do
        latest=$(ls -t "${SWEEP_LOG_DIR}"/nomad_${arm}_*.log 2>/dev/null | head -1 || true)
        if [ -n "${latest}" ]; then
            printf '%-12s %s\n' "${arm}" "$(tr '\r' '\n' < "${latest}" | grep -E 'Train Batch|FINISHED' | tail -1)"
        else
            printf '%-12s %s\n' "${arm}" "not started"
        fi
    done
    exit 0
fi

if docker exec "${CONTAINER}" pgrep -f "train.py -c config/" >/dev/null 2>&1; then
    echo "A training process is already running in ${CONTAINER}. Refusing to start a second." >&2
    echo "Check it with: $0 --status" >&2
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
            -v "${HOST_OUTPUTS}:/outputs" \
            -v /mnt/shared_disk/nazli/wandb_config:/root/.config/wandb \
            "${IMAGE}" sleep infinity >/dev/null
    fi
fi

docker exec "${CONTAINER}" mkdir -p /outputs/nomad_img_ablation/sweep_logs
docker exec "${CONTAINER}" bash -lc 'ln -sfn /outputs /app/visualnav-transformer/train/logs'

stamp=$(date '+%Y%m%d_%H%M%S')
for gpu in 0 1; do
    arms_var="GPU${gpu}_ARMS"
    arms="${!arms_var}"
    driver_log="/outputs/nomad_img_ablation/sweep_logs/sweep_gpu${gpu}_${stamp}.log"
    # -d detaches: the queue is parented by the container, not by this shell.
    docker exec -d "${CONTAINER}" bash -c \
        'exec bash /app/visualnav-transformer/train/ablation/sweep_driver.sh "$@" >>"'"${driver_log}"'" 2>&1' \
        _ "${gpu}" ${arms}
    echo "GPU${gpu}: queued ${arms}"
    echo "  driver log: ${SWEEP_LOG_DIR}/sweep_gpu${gpu}_${stamp}.log"
done

echo
echo "  Per-arm    : ${SWEEP_LOG_DIR}/nomad_img*.log"
echo "  Status     : $0 --status"
echo "  Stop       : docker exec ${CONTAINER} pkill -f train.py"
