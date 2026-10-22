#!/bin/bash
# Start Phase 10's unattended part (kickoff §2-3): screens p10_chain (run_p10_chain.sh) and p10_twin
# (run_p10_twin.sh), each with a host watcher (p10_chain_watch / p10_twin_watch) that writes STOP_reason.txt when its
# screen ends without CHAIN_DONE.txt / a twin DONE state or a STOP file (e.g. kill -9); and the GPU watcher
# p10_gpuwatch (gpu_watch.py) if it is not running.
#
#   bash train/scripts/mapmad_p10/launch_p10.sh            # chain + twin
#   bash train/scripts/mapmad_p10/launch_p10.sh chain      # chain only
#   bash train/scripts/mapmad_p10/launch_p10.sh c          # run C (p10_rules_v4.md): screens p10_c + p10_c_watch
#   bash train/scripts/mapmad_p10/launch_p10.sh c-resume <run C folder>   # run C resume 2 (p10_rules_v5.md)
set -euo pipefail
WHAT="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
P10="$OUT_MOUNT/mapmad/p10_v2_train"
mkdir -p "$P10/driver"
if ! screen -ls | grep -q '[.]p10_gpuwatch[[:space:]]'; then
    screen -dmS p10_gpuwatch python3 "$HERE/gpu_watch.py" "$P10/driver"
    echo "started screen p10_gpuwatch (free GPUs need 20 min of checks from now)"
fi
watch_cmd() {  # screen name, done file, stop file [, status owner (default: the name without p10_)]
    local who="${4:-${1#p10_}}"
    echo "sleep 5; while screen -ls | grep -q '[.]$1[[:space:]]'; do sleep 30; done; \
if [[ -f '$2' ]]; then msg='$1 ended: done'; elif [[ -f '$3' ]]; then msg='$1 ended: stop file present'; \
else msg='$1 ended WITHOUT a done or stop file (kill -9?)'; echo \"\$msg\" > '$3'; \
python3 '$HERE/p10_status.py' '$P10' '$who' --set reason=\"\$msg\" --state STOPPED; fi; \
echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] \$msg\" >> '$P10/driver/watch.log'"
}
if [[ "$WHAT" == all || "$WHAT" == chain ]]; then
    screen -dmS p10_chain bash "$HERE/run_p10_chain.sh"
    screen -dmS p10_chain_watch bash -c "$(watch_cmd p10_chain "$P10/driver/CHAIN_DONE.txt" "$P10/STOP_reason.txt")"
    echo "started screens p10_chain + p10_chain_watch (log driver/chain.log, STATUS.md)"
fi
if [[ "$WHAT" == c || "$WHAT" == c-resume ]]; then
    if [[ "$WHAT" == c ]]; then screen -dmS p10_c bash "$HERE/run_p10_c.sh"
    else screen -dmS p10_c bash "$HERE/run_p10_c.sh" resume "$2"; fi
    screen -dmS p10_c_watch bash -c "$(watch_cmd p10_c "$P10/driver/C_DONE.txt" "$P10/STOP_reason.txt" chain)"
    echo "started screens p10_c + p10_c_watch (log driver/c.log, STATUS.md)"
fi
if [[ "$WHAT" == all || "$WHAT" == twin ]]; then
    screen -dmS p10_twin bash "$HERE/run_p10_twin.sh"
    screen -dmS p10_twin_watch bash -c "$(watch_cmd p10_twin "$P10/driver/TWIN_DONE.txt" "$P10/driver/STOP_twin_reason.txt")"
    echo "started screens p10_twin + p10_twin_watch (log driver/twin.log)"
fi
