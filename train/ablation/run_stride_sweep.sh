#!/usr/bin/env bash
# Launch the context_stride ablation sweep (A2). Run this from the HOST, from the
# ~/projects/nomad-a2 worktree root.
#
#   ./train/ablation/run_stride_sweep.sh                       # both arms
#   ./train/ablation/run_stride_sweep.sh nomad_stride2.yaml     # one arm
#
# A2 runs on GPU0 in its own container, so it is fully independent of Ablation A1, which
# may still be running on GPU1 out of ~/projects/nomad. The two share only the read-only
# dataset and the /outputs volume, and they write to different subdirectories of it.
#
# The sweep runs detached inside the container (docker exec -d), so it is parented by
# the container rather than by this shell. It survives SSH disconnects, terminal death
# and host logouts; only stopping the container stops it.
#
# Status:  ./train/ablation/run_stride_sweep.sh --status
set -euo pipefail

CONTAINER=naz_nomad_stride_ablation
IMAGE=nomad:latest
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_OUTPUTS=/mnt/shared_disk/nazli/nomad_outputs
SWEEP_LOG_DIR="${HOST_OUTPUTS}/nomad_stride_ablation/sweep_logs"

if [ "${1:-}" = "--status" ]; then
    echo "== container process table =="
    docker top "${CONTAINER}" 2>/dev/null || echo "container ${CONTAINER} is not running"
    echo
    echo "== GPU (0 = A2, 1 = A1) =="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
    echo
    echo "== newest sweep log =="
    latest=$(ls -t "${SWEEP_LOG_DIR}"/*.log 2>/dev/null | head -1 || true)
    if [ -n "${latest}" ]; then
        echo "${latest}"
        tr '\r' '\n' < "${latest}" | tail -5
    else
        echo "no logs yet in ${SWEEP_LOG_DIR}"
    fi
    exit 0
fi

# The code directory must be this worktree, never the A1 folder. Printed rather than
# assumed, so a sweep launched from the wrong tree is obvious in the log.
echo "Code directory: ${REPO_ROOT}"

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

# train.py writes checkpoints to logs/<project_name>/<run_name>, which has to land on the
# /outputs volume rather than inside the repo. Created here so a fresh worktree is
# reproducible without a manual setup step; the path is gitignored.
docker exec "${CONTAINER}" ln -sfn /outputs /app/visualnav-transformer/train/logs

stamp=$(date '+%Y%m%d_%H%M%S')
driver_log="/outputs/nomad_stride_ablation/sweep_logs/sweep_${stamp}.log"
host_driver_log="${SWEEP_LOG_DIR}/sweep_${stamp}.log"

docker exec "${CONTAINER}" mkdir -p /outputs/nomad_stride_ablation/sweep_logs

# -d detaches: the process is parented by the container, not by this shell.
docker exec -d "${CONTAINER}" bash -c \
    'exec bash /app/visualnav-transformer/train/ablation/stride_sweep_driver.sh "$@" >>"'"${driver_log}"'" 2>&1' \
    _ "$@"

echo "Sweep launched detached in ${CONTAINER} (GPU0)."
echo
echo "  Driver log : ${host_driver_log}"
echo "  Per-arm    : ${SWEEP_LOG_DIR}/nomad_stride*.log"
echo "  Follow     : tail -f ${host_driver_log}"
echo "  Status     : $0 --status"
echo "  Stop       : docker exec ${CONTAINER} pkill -f train.py"
