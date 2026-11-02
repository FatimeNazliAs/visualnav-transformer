#!/bin/bash
# Start Phase 14a's unattended part (kickoff §2b Part 2): screen p14a_chain (run_p14a_chain.sh) with a host watcher
# p14a_chain_watch that writes STOP_reason.txt when the chain's screen ends without CHAIN_DONE.txt or STOP_reason.txt
# (e.g. kill -9), and the GPU watcher p14a_gpuwatch (mapmad_p10/gpu_watch.py, MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER,
# RECENT_S=3600 = §6 rule 11) if it is not running. The driver screen lives on the host.
#
#   bash train/scripts/mapmad_p14/launch_p14a.sh
#   P14_DIR=<host p14a_dry/chain> P14_CFG=<container DRY config> P14_ALLOW_DIRTY=1 P14_GPUS="0 1" \
#       P14_EPISODES="<ids>" P14_DIAG_SAMPLE=4 CL_CHECK_S=30 SILENT_S=600 bash .../launch_p14a.sh dry
set -euo pipefail
NAME="p14a_chain"
[[ "${1:-}" == dry ]] && NAME="p14a_dry"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
P14="${P14_DIR:-$OUT_MOUNT/mapmad/p14a_finish}"
GW="$OUT_MOUNT/mapmad/p14a_finish/driver"
mkdir -p "$P14/driver" "$GW"
if ! screen -ls | grep -q '[.]p14a_gpuwatch[[:space:]]'; then
    screen -dmS p14a_gpuwatch env MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER RECENT_S=3600 python3 "$HERE/../mapmad_p10/gpu_watch.py" "$GW"
    echo "started screen p14a_gpuwatch (free GPUs need 20 min of checks from now)"
fi
screen -dmS "$NAME" env P14_DIR="$P14" ${P14_CFG:+P14_CFG="$P14_CFG"} ${P14_ALLOW_DIRTY:+P14_ALLOW_DIRTY=1} \
    ${P14_GPUS:+P14_GPUS="$P14_GPUS"} ${CL_CHECK_S:+CL_CHECK_S="$CL_CHECK_S"} ${SILENT_S:+SILENT_S="$SILENT_S"} \
    ${P14_GPU_FREE:+P14_GPU_FREE="$P14_GPU_FREE"} ${P14_EPISODES:+P14_EPISODES="$P14_EPISODES"} \
    ${P14_DIAG_SAMPLE:+P14_DIAG_SAMPLE="$P14_DIAG_SAMPLE"} P14_GPU_FREE="${P14_GPU_FREE:-$GW/gpu_free.json}" \
    bash "$HERE/run_p14a_chain.sh"
screen -dmS "${NAME}_watch" bash -c "sleep 5; while screen -ls | grep -q '[.]$NAME[[:space:]]'; do sleep 30; done; \
if [[ -f '$P14/driver/CHAIN_DONE.txt' && '$P14/driver/CHAIN_DONE.txt' -nt '$P14/driver/chain_state.json.start' ]]; then msg='$NAME ended: done'; \
elif [[ -f '$P14/STOP_reason.txt' ]]; then msg='$NAME ended: STOP_reason.txt present'; \
else msg='$NAME ended WITHOUT CHAIN_DONE or STOP_reason (kill -9?)'; echo \"\$msg\" > '$P14/STOP_reason.txt'; \
python3 '$HERE/p14_status.py' '$P14' chain --set reason=\"\$msg\" --state STOPPED; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$P14/driver/watch.log'"
touch "$P14/driver/chain_state.json.start"
echo "started screens $NAME + ${NAME}_watch (log $P14/driver/chain.log, $P14/STATUS.md)"
