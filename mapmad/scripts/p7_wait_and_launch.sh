#!/bin/bash
# Phase 7 launcher (Naz 2a, 2026-10-16): waits until GPUs are free of other users' jobs, re-runs the two rendering
# tests on a free GPU, starts the sweep driver and watches its screen session. Run on the host in screen:
#
#   screen -dmS mapmad_p7_launch bash -c 'P7_COMMIT=<hash> bash mapmad/scripts/p7_wait_and_launch.sh'
#
# Every CHECK_S seconds (nvidia-smi): both GPUs free -> P7_PAIRS="0 0 0 1 1 1"; only one free -> 3 pairs on it (the
# results do not depend on the number of pairs). Before the launch: tests/test_lens.py (lens off = byte-identical
# pictures; known 3D point = predicted pixel) on that GPU; a failure stops here (no sweep). Then the driver runs in
# screen `mapmad_p7_sweep`; this script stays and logs when that session ends (a kill -9 leaves no STOP_reason.txt).
# Log: <p7>/driver/launch.log. Never commits; touches only our containers.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HABITAT=naz_mapmad_habitat
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' "$HABITAT")"
P7="$OUT_MOUNT/mapmad/p7_camera"
LOG="$P7/driver/launch.log"
CHECK_S="${CHECK_S:-300}"
: "${P7_COMMIT:?set P7_COMMIT to the committed sweep code}"
mkdir -p "$P7/driver"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; echo "- [launcher $(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$P7/phase7_log.md"; }

free_gpus() {  # indices of GPUs without another user's compute process (display server ignored)
    local me idx uuid pid user
    me="$(id -un)"
    declare -A busy
    while IFS=', ' read -r pid uuid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        user="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        [[ -n "$user" && "$user" != "$me" ]] && busy[$uuid]=1
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
    while IFS=', ' read -r idx uuid; do [[ -z "${busy[$uuid]:-}" ]] && echo "$idx"; done \
        < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
}

say "launcher start (commit $P7_COMMIT, check every $CHECK_S s)"
last=""
while true; do
    nvidia-smi >> "$P7/driver/nvidia_smi.log" 2>&1
    mapfile -t free < <(free_gpus)
    if (( ${#free[@]} >= 2 )); then pairs="0 0 0 1 1 1"; gpu=0; break; fi
    if (( ${#free[@]} == 1 )); then g=${free[0]}; pairs="$g $g $g"; gpu=$g; break; fi
    state="both GPUs used by other users' jobs"
    [[ "$state" != "$last" ]] && say "waiting: $state"
    last="$state"
    sleep "$CHECK_S"
done
say "free GPU(s): ${free[*]} -> P7_PAIRS=\"$pairs\"; rendering tests on GPU $gpu"
if ! docker exec -e MAPMAD_TEST_GPU="$gpu" -w /app/visualnav-transformer/mapmad \
        -e PYTHONPATH=/app/visualnav-transformer/mapmad/src:/app/visualnav-transformer/mapmad/tests "$HABITAT" \
        python -m pytest -q -p no:cacheprovider tests/test_lens.py >> "$LOG" 2>&1; then
    say "STOP: rendering tests failed on GPU $gpu (see $LOG); sweep not started"
    exit 1
fi
say "rendering tests passed; nvidia-smi + launch: screen mapmad_p7_sweep, P7_PAIRS=\"$pairs\""
nvidia-smi >> "$P7/driver/nvidia_smi.log" 2>&1
screen -dmS mapmad_p7_sweep bash -c "P7_COMMIT=$P7_COMMIT P7_PAIRS='$pairs' bash '$REPO/mapmad/scripts/run_p7_sweep.sh'"
sleep 5
while screen -ls | grep -q "mapmad_p7_sweep"; do sleep 60; done
if [[ -f "$P7/DONE.txt" ]]; then say "sweep screen ended: DONE.txt present"
elif [[ -f "$P7/STOP_reason.txt" ]]; then say "sweep screen ended: STOP_reason.txt present"
else say "sweep screen ended WITHOUT DONE.txt or STOP_reason.txt (killed with -9?) -- see driver/run_p7_sweep.log"; fi
