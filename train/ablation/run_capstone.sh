#!/usr/bin/env bash
# Launch capstone training runs. Run this from the HOST, from the repo root.
#
#   ./train/ablation/run_capstone.sh 0 nomad_ctx03_s1.yaml nomad_best_combined.yaml
#   ./train/ablation/run_capstone.sh --status
#   ./train/ablation/run_capstone.sh --stop
#
# One GPU lane per invocation. The configs given are trained SEQUENTIALLY on that GPU, in
# the order listed, stopping the lane if one fails -- so a diverged or OOM-ing run does
# not leave the rest of the queue burning hours behind it.
#
# Two layers of screen, both on the host:
#
#   capstone_gpu<N>   the lane supervisor: walks the queue, waits, checks exit status.
#   <run-tag>         one per training run, named from the config's `run_name`, alive
#                     only while that run is. `screen -ls` therefore always answers
#                     "what is training right now"; attach to one to watch it live.
#
# Both survive SSH disconnects and terminal death. Only killing the screens, or stopping
# the container, stops the training.
#
# wandb runs offline unconditionally. `train_nomad` calls wandb.log() on every step, and a
# network stall in the middle of an unattended 19-hour run is a risk with no upside here:
# every number in the write-up comes from the post-hoc eval, not from training-time
# logging. The offline runs stay syncable later with `wandb sync train/wandb/offline-run-*`.
set -euo pipefail

CONTAINER=naz_nomad_capstone
IMAGE=nomad:latest
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_OUTPUTS=/mnt/shared_disk/nazli/nomad_outputs
CAPSTONE_DIR="${HOST_OUTPUTS}/nomad_capstone"
LOG_DIR="${CAPSTONE_DIR}/logs"
STATUS_DIR="${CAPSTONE_DIR}/status"
CONTAINER_TRAIN_DIR=/app/visualnav-transformer/train

run_tag() {  # the config's own run_name, which is also its output directory prefix
    grep -E '^run_name:' "${REPO_ROOT}/train/config/$1" | head -1 | awk '{print $2}'
}

show_status() {
    echo "== training screens =="
    screen -ls | grep -E 'capstone_gpu|^\s+[0-9]+\.' || true
    echo
    echo "== training processes in ${CONTAINER} =="
    docker exec "${CONTAINER}" pgrep -af "train.py -c config/" 2>/dev/null || echo "none"
    echo
    echo "== GPU =="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
    echo
    echo "== newest lines per run log =="
    for log in "${LOG_DIR}"/*.log; do
        [ -e "${log}" ] || continue
        echo "-- $(basename "${log}")"
        tr '\r' '\n' < "${log}" | grep -v '^\s*$' | tail -2
    done
}

case "${1:-}" in
    --status) show_status; exit 0 ;;
    --stop)
        echo "Stopping every capstone training process and lane supervisor."
        docker exec "${CONTAINER}" pkill -f "train.py -c config/" 2>/dev/null || true
        screen -ls | grep -oE 'capstone_gpu[0-9]+' | xargs -r -n1 screen -S -X quit || true
        exit 0 ;;
    "") echo "usage: $0 <gpu> <config.yaml> [config.yaml ...] | --status | --stop" >&2; exit 2 ;;
esac

GPU="$1"; shift
[ "$#" -gt 0 ] || { echo "give at least one config" >&2; exit 2; }

for config in "$@"; do
    [ -f "${REPO_ROOT}/train/config/${config}" ] || {
        echo "no such config: train/config/${config}" >&2; exit 2; }
done

# Refuse to stack two lanes on one GPU: they would fit in memory and quietly halve each
# other's throughput, turning a 19-hour run into a 38-hour one.
if screen -ls | grep -q "capstone_gpu${GPU}\b"; then
    echo "A lane is already running on GPU ${GPU} (screen capstone_gpu${GPU})." >&2
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

mkdir -p "${LOG_DIR}" "${STATUS_DIR}"

# The supervisor is written out rather than passed as a string: it has to survive this
# shell exiting, and quoting a loop this size through `screen -dm bash -c` is a trap.
supervisor="${STATUS_DIR}/lane_gpu${GPU}.sh"
{
    echo '#!/usr/bin/env bash'
    echo 'set -uo pipefail'
    for config in "$@"; do
        tag="$(run_tag "${config}")"
        [ -n "${tag}" ] || { echo "config ${config} has no run_name" >&2; exit 2; }
        stamp='$(date "+%Y%m%d_%H%M%S")'
        cat <<LANE
echo "[\$(date '+%F %T')] START ${config} as ${tag} on GPU ${GPU}"
log="${LOG_DIR}/${tag}_${stamp}.log"
status="${STATUS_DIR}/${tag}.status"
rm -f "\${status}"
screen -dmS "${tag}" bash -c "docker exec ${CONTAINER} bash -lc 'cd ${CONTAINER_TRAIN_DIR} && CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 WANDB_MODE=offline python train.py -c config/${config}' 2>&1 | tee \${log}; echo \\\${PIPESTATUS[0]} > \${status}"
# Poll the status file rather than the screen list: a screen can vanish for reasons
# other than the run finishing, and the exit code is what decides whether to continue.
while [ ! -f "\${status}" ]; do sleep 30; done
code="\$(cat "\${status}")"
if [ "\${code}" != "0" ]; then
    echo "[\$(date '+%F %T')] FAILED ${tag} (exit \${code}) -- stopping GPU ${GPU} lane." >&2
    echo "See \${log}" >&2
    exit 1
fi
echo "[\$(date '+%F %T')] DONE ${tag}"
LANE
    done
    echo "echo \"[\$(date '+%F %T')] GPU ${GPU} lane complete\""
} > "${supervisor}"
chmod +x "${supervisor}"

lane_log="${LOG_DIR}/lane_gpu${GPU}.log"
screen -dmS "capstone_gpu${GPU}" bash -c "'${supervisor}' >>'${lane_log}' 2>&1"

echo "GPU ${GPU} lane launched. Queue:"
for config in "$@"; do echo "  ${config}  ->  $(run_tag "${config}")"; done
echo
echo "  Lane log : ${lane_log}"
echo "  Run logs : ${LOG_DIR}/<run-tag>_<stamp>.log"
echo "  Status   : $0 --status"
echo "  Stop     : $0 --stop"
