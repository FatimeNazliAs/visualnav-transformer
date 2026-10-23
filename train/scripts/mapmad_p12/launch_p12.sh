#!/bin/bash
# Start Phase 12's unattended part (kickoff §3 Part 2): screen p12_chain (run_p12_chain.sh) with a host watcher
# p12_chain_watch that writes STOP_reason.txt when the chain's screen ends without CHAIN_DONE.txt or STOP_reason.txt
# (e.g. kill -9; it reads the NEWEST start line of chain.log, Phase 11 lesson), and the GPU watcher p12_gpuwatch
# (mapmad_p10/gpu_watch.py, MAPMAD_OWNER_ENV=MAPMAD_P12_OWNER, RECENT_S=3600 = §8 rule 11) if it is not running.
#
#   bash train/scripts/mapmad_p12/launch_p12.sh
#   P12_DIR=<DRY folder> P12_CFG=<container config> P12_ALLOW_DIRTY=1 P12_GPUS="0 1" bash .../launch_p12.sh dry
set -euo pipefail
NAME="p12_chain"
[[ "${1:-}" == dry ]] && NAME="p12_dry"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
P12="${P12_DIR:-$OUT_MOUNT/mapmad/p12_search_exam}"
GW="$OUT_MOUNT/mapmad/p12_search_exam/driver"
mkdir -p "$P12/driver" "$GW"
if ! screen -ls | grep -q '[.]p12_gpuwatch[[:space:]]'; then
    screen -dmS p12_gpuwatch env MAPMAD_OWNER_ENV=MAPMAD_P12_OWNER RECENT_S=3600 python3 "$HERE/../mapmad_p10/gpu_watch.py" "$GW"
    echo "started screen p12_gpuwatch (free GPUs need 20 min of checks from now)"
fi
screen -dmS "$NAME" env P12_DIR="$P12" ${P12_CFG:+P12_CFG="$P12_CFG"} ${P12_ALLOW_DIRTY:+P12_ALLOW_DIRTY=1} \
    ${P12_GPUS:+P12_GPUS="$P12_GPUS"} ${CL_CHECK_S:+CL_CHECK_S="$CL_CHECK_S"} ${SILENT_S:+SILENT_S="$SILENT_S"} \
    ${P12_GPU_FREE:+P12_GPU_FREE="$P12_GPU_FREE"} ${P12_SAME_IGNORE:+P12_SAME_IGNORE="$P12_SAME_IGNORE"} \
    bash "$HERE/run_p12_chain.sh"
screen -dmS "${NAME}_watch" bash -c "sleep 5; while screen -ls | grep -q '[.]$NAME[[:space:]]'; do sleep 30; done; \
if [[ -f '$P12/driver/CHAIN_DONE.txt' && '$P12/driver/CHAIN_DONE.txt' -nt '$P12/driver/chain_state.json.start' ]]; then msg='$NAME ended: done'; \
elif [[ -f '$P12/STOP_reason.txt' ]]; then msg='$NAME ended: STOP_reason.txt present'; \
else msg='$NAME ended WITHOUT CHAIN_DONE or STOP_reason (kill -9?)'; echo \"\$msg\" > '$P12/STOP_reason.txt'; \
python3 '$HERE/p12_status.py' '$P12' chain --set reason=\"\$msg\" --state STOPPED; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$P12/driver/watch.log'"
touch "$P12/driver/chain_state.json.start"
echo "started screens $NAME + ${NAME}_watch (log $P12/driver/chain.log, $P12/STATUS.md)"
