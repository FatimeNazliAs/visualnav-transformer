#!/usr/bin/env bash
# Launch the Phase 2 CLIP-goal training run (config/nomad_clip.yaml). Run this from the
# HOST, from the repo root.
#
#   ./train/ablation/run_clip_train.sh            # launch in screen "nomad_clip"
#   ./train/ablation/run_clip_train.sh --status   # process, GPU and newest log
#
# The run lives in a detached screen session that holds a foreground docker exec, so it
# survives SSH disconnects. train.py creates the run folder itself:
#   /outputs/nomad_clip_v2/clip_vitb32_<timestamp>/   (checkpoints, ema_*.pth)
# and the console log goes to /outputs/nomad_clip_v2/logs/train_<stamp>.log.
set -euo pipefail

CONTAINER=naz_nomad_clip
SCREEN_NAME=nomad_clip
CONFIG=config/nomad_clip.yaml
HOST_OUTPUTS=/mnt/shared_disk/nazli/nomad_outputs
LOG_DIR=/outputs/nomad_clip_v2/logs
HOST_LOG_DIR="${HOST_OUTPUTS}/nomad_clip_v2/logs"

if [ "${1:-}" = "--status" ]; then
    echo "== training processes in ${CONTAINER} =="
    docker exec "${CONTAINER}" pgrep -af "train.py -c ${CONFIG}" || echo "none"
    echo
    echo "== GPU =="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
    echo
    echo "== newest log =="
    latest=$(ls -t "${HOST_LOG_DIR}"/*.log 2>/dev/null | head -1 || true)
    if [ -n "${latest}" ]; then
        echo "${latest}"
        tr '\r' '\n' < "${latest}" | tail -5
    else
        echo "no logs yet in ${HOST_LOG_DIR}"
    fi
    exit 0
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
    echo "Container ${CONTAINER} is not running. Not starting or creating it from here." >&2
    exit 1
fi

if screen -ls | grep -q "[0-9]\.${SCREEN_NAME}[[:space:]]"; then
    echo "A screen named ${SCREEN_NAME} already exists. Refusing to start a second." >&2
    exit 1
fi

if docker exec "${CONTAINER}" pgrep -f "train.py -c " >/dev/null 2>&1; then
    echo "A training process is already running in ${CONTAINER}. Refusing to start a second." >&2
    exit 1
fi

stamp=$(date '+%Y%m%d_%H%M%S')
log_file="${LOG_DIR}/train_${stamp}.log"
docker exec "${CONTAINER}" mkdir -p "${LOG_DIR}"

# GPU1 only: GPU0 drives the display. Pinned here rather than via gpu_ids, because
# train.py sets CUDA_VISIBLE_DEVICES after torch has already initialised CUDA, which
# makes its in-process masking a no-op (checked: gpu_ids [0] + this lands on GPU1).
# PYTHONUNBUFFERED keeps the log live through tee; PIPESTATUS keeps train.py's exit code.
inner="cd /app/visualnav-transformer/train && \
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1 \
python train.py -c ${CONFIG} 2>&1 | tee ${log_file}; \
echo \"[\$(date '+%F %T')] train.py exited with \${PIPESTATUS[0]}\" | tee -a ${log_file}"

# printf %q quotes $inner for the screen's own bash, so it reaches the container's bash
# verbatim ($(date) and PIPESTATUS expand there, not on the host). The trailing
# `exec bash` keeps the screen open after training ends, so the last output stays
# readable with `screen -r nomad_clip`.
screen -dmS "${SCREEN_NAME}" bash -c "docker exec ${CONTAINER} bash -c $(printf '%q' "${inner}"); exec bash"

echo "Launched in screen ${SCREEN_NAME} (container ${CONTAINER}, GPU1)."
echo
echo "  Log     : ${HOST_LOG_DIR}/train_${stamp}.log"
echo "  Run dir : ${HOST_OUTPUTS}/nomad_clip_v2/clip_vitb32_<timestamp>/"
echo "  Attach  : screen -r ${SCREEN_NAME}   (detach: Ctrl-a d)"
echo "  Status  : $0 --status"
echo "  Stop    : docker exec ${CONTAINER} pkill -f 'train.py -c ${CONFIG}'"
