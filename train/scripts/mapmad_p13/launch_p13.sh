#!/bin/bash
# Start Phase 13's unattended part (kickoff §2b Part 2): screen p13_chain (run_p13_chain.sh) with a host watcher
# p13_chain_watch that writes STOP_reason.txt when the chain's screen ends without CHAIN_DONE.txt or STOP_reason.txt
# (e.g. kill -9), and the GPU watcher p13_gpuwatch (mapmad_p10/gpu_watch.py, MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER,
# RECENT_S=3600 = §5 rule 11) if it is not running. The driver screen lives on the host (it restarts naz_mapmad).
#
#   bash train/scripts/mapmad_p13/launch_p13.sh
#   P13_DIR=<DRY folder> P13_CFG=<container config> P13_ALLOW_DIRTY=1 P13_GPUS="0 1" bash .../launch_p13.sh dry
set -euo pipefail
NAME="p13_chain"
[[ "${1:-}" == dry ]] && NAME="p13_dry"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
P13="${P13_DIR:-$OUT_MOUNT/mapmad/p13_claim}"
GW="$OUT_MOUNT/mapmad/p13_claim/driver"
mkdir -p "$P13/driver" "$GW"
if ! screen -ls | grep -q '[.]p13_gpuwatch[[:space:]]'; then
    screen -dmS p13_gpuwatch env MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER RECENT_S=3600 python3 "$HERE/../mapmad_p10/gpu_watch.py" "$GW"
    echo "started screen p13_gpuwatch (free GPUs need 20 min of checks from now)"
fi
screen -dmS "$NAME" env P13_DIR="$P13" ${P13_CFG:+P13_CFG="$P13_CFG"} ${P13_ALLOW_DIRTY:+P13_ALLOW_DIRTY=1} \
    ${P13_GPUS:+P13_GPUS="$P13_GPUS"} ${CL_CHECK_S:+CL_CHECK_S="$CL_CHECK_S"} ${SILENT_S:+SILENT_S="$SILENT_S"} \
    ${P13_GPU_FREE:+P13_GPU_FREE="$P13_GPU_FREE"} ${P13_HIDDEN_CFG:+P13_HIDDEN_CFG="$P13_HIDDEN_CFG"} ${P13_V21B_BASE:+P13_V21B_BASE="$P13_V21B_BASE"} \
    ${P13_TRAIN_EXTRA:+P13_TRAIN_EXTRA="$P13_TRAIN_EXTRA"} ${P13_TRAIN_GPUS:+P13_TRAIN_GPUS="$P13_TRAIN_GPUS"} ${P13_VLFM_N:+P13_VLFM_N="$P13_VLFM_N"} \
    bash "$HERE/run_p13_chain.sh"
screen -dmS "${NAME}_watch" bash -c "sleep 5; while screen -ls | grep -q '[.]$NAME[[:space:]]'; do sleep 30; done; \
if [[ -f '$P13/driver/CHAIN_DONE.txt' && '$P13/driver/CHAIN_DONE.txt' -nt '$P13/driver/chain_state.json.start' ]]; then msg='$NAME ended: done'; \
elif [[ -f '$P13/STOP_reason.txt' ]]; then msg='$NAME ended: STOP_reason.txt present'; \
else msg='$NAME ended WITHOUT CHAIN_DONE or STOP_reason (kill -9?)'; echo \"\$msg\" > '$P13/STOP_reason.txt'; \
python3 '$HERE/p13_status.py' '$P13' chain --set reason=\"\$msg\" --state STOPPED; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$P13/driver/watch.log'"
touch "$P13/driver/chain_state.json.start"
echo "started screens $NAME + ${NAME}_watch (log $P13/driver/chain.log, $P13/STATUS.md)"
