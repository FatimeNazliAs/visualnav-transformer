#!/bin/bash
# Start the Phase 8 driver in screen `mapmad_p8_<dataset>` and a host watcher in `mapmad_p8_<dataset>_watch` that logs
# when the driver's screen ends without DONE.txt or STOP_reason.txt (e.g. kill -9). Environment passes through
# (COUNTS, W1, W0, TOKEN, CHECK_S). Needs gpu_free_watch.sh running (screen mapmad_p8_gpuwatch).
#
#   bash mapmad/scripts/p8_launch.sh pilot
set -euo pipefail
DATASET="${1:?pilot or full}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HABITAT=naz_mapmad_habitat
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' "$HABITAT")"
RUN="$OUT_MOUNT/mapmad/p8_search_data/$DATASET${TAG:+_$TAG}"
[[ -n "${OUT_NAME:-}" ]] && RUN="$OUT_MOUNT/mapmad/p8_search_data/dry/$OUT_NAME"
mkdir -p "$RUN"
rm -f "$RUN/STOP_reason.txt"
NAME="mapmad_p8_${OUT_NAME:-$DATASET${TAG:+_$TAG}}"
screen -dmS "$NAME" bash -c "CONFIG='${CONFIG:-}' TAG='${TAG:-}' OUT_NAME='${OUT_NAME:-}' HOMES='${HOMES:-}' COUNTS='${COUNTS:-}' W1='${W1:-12}' W0='${W0:-6}' TOKEN='${TOKEN:-}' CHECK_S='${CHECK_S:-300}' FREE_JSON='${FREE_JSON:-}' bash '$HERE/run_p8_gen.sh' $DATASET"
screen -dmS "${NAME}_watch" bash -c "sleep 5; while screen -ls | grep -q '[.]$NAME[[:space:]]'; do sleep 30; done; \
if [[ -f '$RUN/DONE.txt' ]]; then msg='driver screen ended: DONE.txt present'; \
elif [[ -f '$RUN/STOP_reason.txt' ]]; then msg='driver screen ended: STOP_reason.txt present'; \
else msg='driver screen ended WITHOUT DONE.txt or STOP_reason.txt (kill -9?)'; echo \"\$msg\" > '$RUN/STOP_reason.txt'; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$RUN/watch.log'"
echo "started screens $NAME and ${NAME}_watch; log $RUN/driver.log"
