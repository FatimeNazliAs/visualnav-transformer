#!/bin/bash
# Start the Phase 14b-3 addendum (M21bh completion; run_p14b3_addendum.sh) in screen p14b3_addm, with a host watcher
# p14b3_addm_watch that writes addendum/STOP_reason.txt when the screen ends without ADDENDUM_DONE.txt or a stop reason,
# and the GPU watcher p14b3_gpuwatch with the kickoff's server rules (4 checks x 5 min, no other user's process in the
# last 60 min) if it is not running.
#
#   bash train/scripts/mapmad_p14/launch_p14b3_addendum.sh
set -euo pipefail
NAME="p14b3_addm"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
MAIN="$OUT_MOUNT/mapmad/p14b3_v3"
P14="$MAIN/addendum"
mkdir -p "$P14/driver"
if ! screen -ls | grep -q '[.]p14b3_gpuwatch[[:space:]]'; then
    screen -dmS p14b3_gpuwatch env MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER RECENT_S=3600 FREE_CHECKS=4 CHECK_S=300 \
        python3 "$HERE/../mapmad_p10/gpu_watch.py" "$MAIN/driver"
    echo "started screen p14b3_gpuwatch (server rules: free GPUs need 20 min of checks and 60 min without another user)"
fi
screen -dmS "$NAME" bash "$HERE/run_p14b3_addendum.sh"
screen -dmS "${NAME}_watch" bash -c "sleep 5; while screen -ls | grep -q '[.]$NAME[[:space:]]'; do sleep 30; done; \
if [[ -f '$P14/driver/ADDENDUM_DONE.txt' && '$P14/driver/ADDENDUM_DONE.txt' -nt '$P14/driver/start' ]]; then msg='$NAME ended: done'; \
elif [[ -f '$P14/STOP_reason.txt' ]]; then msg='$NAME ended: STOP_reason.txt present'; \
else msg='$NAME ended WITHOUT ADDENDUM_DONE or STOP_reason (kill -9?)'; echo \"\$msg\" > '$P14/STOP_reason.txt'; \
P14_PHASE='14b-3 addendum (M21bh)' P14_LOGNAME=../phase14b3_log.md python3 '$HERE/p14_status.py' '$P14' chain --set reason=\"\$msg\" --state STOPPED; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$P14/driver/watch.log'"
touch "$P14/driver/start"
echo "started screens $NAME + ${NAME}_watch (log $P14/driver/chain.log, $P14/STATUS.md)"
