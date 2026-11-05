#!/bin/bash
# Start Phase 14b-3's unattended part (kickoff §0 / Part 2): screen p14b3_chain (run_p14b3_chain.sh) with a host watcher
# p14b3_chain_watch that writes STOP_reason.txt when the chain's screen ends without CHAIN_DONE.txt or STOP_reason.txt,
# and the GPU watcher p14b3_gpuwatch (mapmad_p10/gpu_watch.py, 20-min rule) if it is not running.
#
#   bash train/scripts/mapmad_p14/launch_p14b3.sh
#   P14_DIR=<host p14b3_dry/chain> P14_CFG=/app/visualnav-transformer/mapmad/configs/p14b3_dry.yaml P14_ALLOW_DIRTY=1 \
#       P14_GPUS="0 1" P14_EPISODES="<ids>" P14B3_ROLL_MAP=<container dry rollout cfg> P14B3_ROLL_HID=<...> \
#       P14B3_TRAIN_CFG=<container dry train cfg> P14B3_R2_SUFFIX=_dry P14B3_REC_N=20 P14B_VIDEO_PICKS=<json> \
#       P14B3_VIDEO_PICKS_EXAMB=<json> CL_CHECK_S=30 SILENT_S=600 bash .../launch_p14b3.sh dry
set -euo pipefail
NAME="p14b3_chain"
[[ "${1:-}" == dry ]] && NAME="p14b3_dry"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
P14="${P14_DIR:-$OUT_MOUNT/mapmad/p14b3_v3}"
GW="$OUT_MOUNT/mapmad/p14b3_v3/driver"
mkdir -p "$P14/driver" "$GW"
if ! screen -ls | grep -q '[.]p14b3_gpuwatch[[:space:]]'; then
    screen -dmS p14b3_gpuwatch env MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER RECENT_S=3600 python3 "$HERE/../mapmad_p10/gpu_watch.py" "$GW"
    echo "started screen p14b3_gpuwatch (free GPUs need 20 min of checks from now)"
fi
PASS=()
for v in P14_CFG P14_ALLOW_DIRTY P14_GPUS SILENT_S CL_CHECK_S P14_EPISODES P14B_VIDEO_PICKS P14B3_ROLL_MAP P14B3_ROLL_HID \
         P14B3_TRAIN_CFG P14B3_TRAIN_EXTRA P14B3_R2_SUFFIX P14B3_REC_N P14B3_PAIRS P14B3_LABEL_WORKERS \
         P14B3_VIDEO_PICKS_EXAMB P14_TRAIN_GPUS; do
    [[ -n "${!v:-}" ]] && PASS+=("$v=${!v}")
done
screen -dmS "$NAME" env P14_DIR="$P14" "${PASS[@]}" P14_GPU_FREE="${P14_GPU_FREE:-$GW/gpu_free.json}" \
    bash "$HERE/run_p14b3_chain.sh"
screen -dmS "${NAME}_watch" bash -c "sleep 5; while screen -ls | grep -q '[.]$NAME[[:space:]]'; do sleep 30; done; \
if [[ -f '$P14/driver/CHAIN_DONE.txt' && '$P14/driver/CHAIN_DONE.txt' -nt '$P14/driver/chain_state.json.start' ]]; then msg='$NAME ended: done'; \
elif [[ -f '$P14/STOP_reason.txt' ]]; then msg='$NAME ended: STOP_reason.txt present'; \
else msg='$NAME ended WITHOUT CHAIN_DONE or STOP_reason (kill -9?)'; echo \"\$msg\" > '$P14/STOP_reason.txt'; \
P14_PHASE=14b-3 P14_LOGNAME=phase14b3_log.md python3 '$HERE/p14_status.py' '$P14' chain --set reason=\"\$msg\" --state STOPPED; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$P14/driver/watch.log'"
touch "$P14/driver/chain_state.json.start"
echo "started screens $NAME + ${NAME}_watch (log $P14/driver/chain.log, $P14/STATUS.md)"
