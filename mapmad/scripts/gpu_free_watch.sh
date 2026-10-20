#!/bin/bash
# GPU free-check watcher (plan §0.4 "free GPU = 20 min", v2 plan §8). Runs on the host in screen, never touches a GPU:
#
#   screen -dmS mapmad_p8_gpuwatch bash mapmad/scripts/gpu_free_watch.sh /outputs-dir/p8_search_data/driver
#
# Every CHECK_S seconds (default 300): which GPUs hold a compute process of another user. Writes
# <dir>/gpu_free.json = {"time": ..., "streak": {"0": n, "1": n}, "free": [GPUs with streak >= FREE_CHECKS]} (atomic)
# and logs every change of the free set to <dir>/gpu_watch.log. A launcher reads gpu_free.json instead of waiting
# 20 min itself. A GPU is free only after FREE_CHECKS (4) consecutive checks without another user's process.
set -uo pipefail

DIR="${1:?output folder}"
CHECK_S="${CHECK_S:-300}"
FREE_CHECKS="${FREE_CHECKS:-4}"
mkdir -p "$DIR"
LOG="$DIR/gpu_watch.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

others_gpus() {  # GPU uuids with a compute process of another user
    local me pid uuid user
    me="$(id -un)"
    while IFS=', ' read -r pid uuid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        user="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        [[ -n "$user" && "$user" != "$me" ]] && echo "$uuid"  # same rule as p7_wait_and_launch.sh
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
}

say "watcher start: check every $CHECK_S s, free = $FREE_CHECKS consecutive checks"
declare -A streak
last=""
while true; do
    mapfile -t busy < <(others_gpus)
    free=()
    parts=()
    while IFS=', ' read -r idx uuid; do
        if [[ " ${busy[*]:-} " == *" $uuid "* ]]; then streak[$idx]=0; else streak[$idx]=$(( ${streak[$idx]:-0} + 1 )); fi
        (( streak[$idx] >= FREE_CHECKS )) && free+=("$idx")
        parts+=("\"$idx\": ${streak[$idx]}")
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    free_json="$(printf '%s,' "${free[@]:-}" | sed 's/,$//; s/^,//')"
    tmp="$DIR/.gpu_free.json.tmp"
    printf '{"time": "%s", "streak": {%s}, "free": [%s]}\n' "$(date '+%Y-%m-%d %H:%M:%S')" \
        "$(IFS=,; echo "${parts[*]}")" "$free_json" > "$tmp" && mv "$tmp" "$DIR/gpu_free.json"
    state="free=[${free[*]:-}] streak=$(IFS=,; echo "${parts[*]}")"
    [[ "${free[*]:-}" != "$last" ]] && say "$state"
    last="${free[*]:-}"
    sleep "$CHECK_S"
done
