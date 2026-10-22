#!/bin/bash
# Start Phase 11's unattended part (kickoff §3 Part 2): screens p11_chain (run_p11_chain.sh) and p11_twin
# (run_p11_twin.sh), each with a host watcher (p11_chain_watch / p11_twin_watch) that writes the stop file when its
# screen ends without its done file (e.g. kill -9), and the GPU watcher p11_gpuwatch (mapmad_p10/gpu_watch.py with
# MAPMAD_OWNER_ENV=MAPMAD_P11_OWNER) if it is not running.
#
#   bash train/scripts/mapmad_p11/launch_p11.sh            # chain + twin
#   bash train/scripts/mapmad_p11/launch_p11.sh chain      # chain only
set -euo pipefail
WHAT="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
P11="${P11_DIR:-$OUT_MOUNT/mapmad/p11_dagger}"
mkdir -p "$P11/driver"
if ! screen -ls | grep -q '[.]p11_gpuwatch[[:space:]]'; then
    screen -dmS p11_gpuwatch env MAPMAD_OWNER_ENV=MAPMAD_P11_OWNER python3 "$HERE/../mapmad_p10/gpu_watch.py" "$P11/driver"
    echo "started screen p11_gpuwatch (free GPUs need 20 min of checks from now)"
fi
watch_cmd() {  # screen name, done file, stop file, status owner
    echo "sleep 5; while screen -ls | grep -q '[.]$1[[:space:]]'; do sleep 30; done; \
if [[ -f '$2' ]]; then msg='$1 ended: done'; elif [[ -f '$3' ]]; then msg='$1 ended: stop file present'; \
else msg='$1 ended WITHOUT a done or stop file (kill -9?)'; echo \"\$msg\" > '$3'; \
python3 '$HERE/p11_status.py' '$P11' '$4' --set reason=\"\$msg\" --state STOPPED; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$P11/driver/watch.log'"
}
if [[ "$WHAT" == all || "$WHAT" == chain ]]; then
    screen -dmS p11_chain bash "$HERE/run_p11_chain.sh"
    screen -dmS p11_chain_watch bash -c "$(watch_cmd p11_chain "$P11/driver/CHAIN_DONE.txt" "$P11/STOP_reason.txt" chain)"
    echo "started screens p11_chain + p11_chain_watch (log driver/chain.log, STATUS.md)"
fi
if [[ "$WHAT" == all || "$WHAT" == twin ]]; then
    screen -dmS p11_twin bash "$HERE/run_p11_twin.sh"
    screen -dmS p11_twin_watch bash -c "$(watch_cmd p11_twin "$P11/driver/TWIN_DONE.txt" "$P11/driver/STOP_twin_reason.txt" twin)"
    echo "started screens p11_twin + p11_twin_watch (log driver/twin.log)"
fi
