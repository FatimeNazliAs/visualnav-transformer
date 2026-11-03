#!/bin/bash
# Start Phase 14b-1's unattended part (kickoff §2): screen p14b1_chain (run_p14b1_chain.sh) with a host watcher
# p14b1_chain_watch that writes STOP_reason.txt when the chain's screen ends without CHAIN_DONE.txt or STOP_reason.txt,
# and the GPU watcher p14b1_gpuwatch (mapmad_p10/gpu_watch.py, 20-min rule) if it is not running.
#
#   bash train/scripts/mapmad_p14/launch_p14b1.sh
#   P14_DIR=<host p14b1_dry> P14_CFG=/app/visualnav-transformer/mapmad/configs/p14b1_dry.yaml P14_ALLOW_DIRTY=1 \
#       P14_GPUS="0 1" P14_EPISODES="<ids>" SHARDS_PER_GPU=1 P14B_VIDEO_PICKS=<container picks json> \
#       bash .../launch_p14b1.sh dry
set -euo pipefail
NAME="p14b1_chain"
[[ "${1:-}" == dry ]] && NAME="p14b1_dry"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
P14="${P14_DIR:-$OUT_MOUNT/mapmad/p14b1_marks}"
GW="$OUT_MOUNT/mapmad/p14b1_marks/driver"
mkdir -p "$P14/driver" "$GW"
if ! screen -ls | grep -q '[.]p14b1_gpuwatch[[:space:]]'; then
    screen -dmS p14b1_gpuwatch env MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER RECENT_S=3600 python3 "$HERE/../mapmad_p10/gpu_watch.py" "$GW"
    echo "started screen p14b1_gpuwatch (free GPUs need 20 min of checks from now)"
fi
screen -dmS "$NAME" env P14_DIR="$P14" ${P14_CFG:+P14_CFG="$P14_CFG"} ${P14_ALLOW_DIRTY:+P14_ALLOW_DIRTY=1} \
    ${P14_GPUS:+P14_GPUS="$P14_GPUS"} ${SILENT_S:+SILENT_S="$SILENT_S"} ${P14_EPISODES:+P14_EPISODES="$P14_EPISODES"} \
    ${SHARDS_PER_GPU:+SHARDS_PER_GPU="$SHARDS_PER_GPU"} ${BENCH_WORKERS:+BENCH_WORKERS="$BENCH_WORKERS"} \
    ${P14B_VIDEO_PICKS:+P14B_VIDEO_PICKS="$P14B_VIDEO_PICKS"} P14_GPU_FREE="${P14_GPU_FREE:-$GW/gpu_free.json}" \
    bash "$HERE/run_p14b1_chain.sh"
screen -dmS "${NAME}_watch" bash -c "sleep 5; while screen -ls | grep -q '[.]$NAME[[:space:]]'; do sleep 30; done; \
if [[ -f '$P14/driver/CHAIN_DONE.txt' && '$P14/driver/CHAIN_DONE.txt' -nt '$P14/driver/chain_state.json.start' ]]; then msg='$NAME ended: done'; \
elif [[ -f '$P14/STOP_reason.txt' ]]; then msg='$NAME ended: STOP_reason.txt present'; \
else msg='$NAME ended WITHOUT CHAIN_DONE or STOP_reason (kill -9?)'; echo \"\$msg\" > '$P14/STOP_reason.txt'; \
P14_PHASE=14b-1 P14_LOGNAME=phase14b1_log.md python3 '$HERE/p14_status.py' '$P14' chain --set reason=\"\$msg\" --state STOPPED; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$P14/driver/watch.log'"
touch "$P14/driver/chain_state.json.start"
echo "started screens $NAME + ${NAME}_watch (log $P14/driver/chain.log, $P14/STATUS.md)"
