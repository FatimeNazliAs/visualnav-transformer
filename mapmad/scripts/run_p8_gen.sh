#!/bin/bash
# Phase 8 drive generation driver (host, in screen). Launch with p8_launch.sh, which also starts the watcher:
#
#   bash mapmad/scripts/p8_launch.sh pilot                        # screens mapmad_p8_pilot + mapmad_p8_pilot_watch
#   COUNTS='{"search": 40, "explore": 4}' bash mapmad/scripts/p8_launch.sh full    # a pro-rata cut (block item 5)
#
# - GPUs (v2 plan §8): reads <p8>/driver/gpu_free.json from gpu_free_watch.sh (free = 4 consecutive 5-min checks
#   without another user's process). Waits until GPU 1 is free, starts W1 (12) workers on it; every 5 min, if GPU 0 has
#   also been free for 20 min, adds W0 (6) workers on GPU 0 from the same queue (once). Never starts a worker on a GPU
#   that is not free by that rule. Every GPU event goes to the log.
# - Workers: generate_drives_v2.py in naz_mapmad_habitat, one shared home queue (claims under _claims/<token>/), drives
#   seeded by id -> identical results for any number of workers. Total workers <= CPU cores - 4.
# - Full run: refuses uncommitted code (§0.8). Pilot: saves a code snapshot (base commit + patch + sha256).
# - EXIT trap: any exit that is not a normal finish writes <run>/STOP_reason.txt (exit code + last log line); a normal
#   finish writes DONE.txt. Long waits run in the background (`wait`), so TERM is handled within ~1 s. Resumable:
#   finished homes are skipped (_homes/<home>.json).
set -uo pipefail

DATASET="${1:?pilot or full}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HABITAT=naz_mapmad_habitat
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' "$HABITAT")"
P8="$OUT_MOUNT/mapmad/p8_search_data"
OUT_NAME="${OUT_NAME:-}"   # dry runs: write to <mapmad_data>/<OUT_NAME>, logs to <p8>/dry/<OUT_NAME> (marked DRY)
HOMES="${HOMES:-}"         # dry runs: only these homes (5-digit prefixes)
CONFIG="${CONFIG:-}"       # another config (e.g. a pilot fix attempt, p8_pilot_fix1.yaml), relative to the repo
TAG="${TAG:-}"             # ... and its run folder <p8>/<dataset>_<TAG>
RUN="$P8/$DATASET${TAG:+_$TAG}"
[[ -n "$OUT_NAME" ]] && RUN="$P8/dry/$OUT_NAME"
LOG="$RUN/driver.log"
W1="${W1:-12}"
W0="${W0:-6}"
CHECK_S="${CHECK_S:-300}"
TOKEN="${TOKEN:-$(date +%Y%m%d-%H%M%S)}"
COUNTS="${COUNTS:-}"
mkdir -p "$RUN"
rm -f "$RUN/DONE.txt"
[[ -n "$OUT_NAME" ]] && echo "DRY RUN: never results" > "$RUN/DRY.txt"
FINISHED=0

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; echo "- [driver $(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$P8/phase8_log.md"; }
on_exit() {
    local code=$?
    if [[ $FINISHED -ne 1 ]]; then
        { echo "time: $(date '+%Y-%m-%d %H:%M:%S')"; echo "exit code: $code"; echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$RUN/STOP_reason.txt"
        say "STOP (exit $code): see $RUN/STOP_reason.txt"
        [[ -n "${PIDS:-}" ]] && kill $PIDS 2>/dev/null
        docker exec "$HABITAT" pkill -f "generate_drives_v2.py .*--token $TOKEN" 2>/dev/null  # our workers in our container
    fi
}
trap on_exit EXIT
trap 'exit 143' TERM INT

free_list() { python3 -c "import json,sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['free'])))" "$P8/driver/gpu_free.json" 2>/dev/null; }
is_free() { [[ " $(free_list) " == *" $1 "* ]]; }

say "driver start: dataset=$DATASET token=$TOKEN W1=$W1 W0=$W0 counts=${COUNTS:-config} config=${CONFIG:-p8_datagen.yaml} tag=${TAG:-none}"
CORES=$(nproc)
(( W1 + W0 <= CORES - 4 )) || { say "too many workers for $CORES cores"; exit 2; }
cd "$REPO" || exit 2
if [[ "$DATASET" == "full" ]]; then
    if [[ -n "$(git status --porcelain -- mapmad train)" ]]; then say "uncommitted code under mapmad/ or train/: commit first (§0.8)"; exit 3; fi
else
    SNAP="$RUN/code_snapshot"; mkdir -p "$SNAP"
    git rev-parse HEAD > "$SNAP/base_commit.txt"
    { git diff HEAD -- mapmad train; git ls-files --others --exclude-standard -- mapmad train | while read -r f; do git diff --no-index /dev/null "$f"; done; } > "$SNAP/patch.diff"
    sha256sum "$SNAP/patch.diff" > "$SNAP/patch.diff.sha256"
    say "code snapshot: base $(cat "$SNAP/base_commit.txt"), patch sha256 $(cut -c1-16 "$SNAP/patch.diff.sha256")"
fi

until is_free 1; do say "waiting for GPU 1 (free list: [$(free_list)])"; sleep "$CHECK_S" & wait $!; done
nvidia-smi >> "$RUN/nvidia_smi.log" 2>&1
PIDS=""
start_workers() {  # gpu, count, first index
    local gpu=$1 n=$2 k0=$3
    for (( k = k0; k < k0 + n; k++ )); do
        docker exec -w /app/visualnav-transformer "$HABITAT" python mapmad/scripts/generate_drives_v2.py \
            --dataset "$DATASET" --gpu "$gpu" --token "$TOKEN" --worker "w$k" ${COUNTS:+--counts "$COUNTS"} \
            ${OUT_NAME:+--out-name "$OUT_NAME"} ${HOMES:+--homes $HOMES} ${CONFIG:+--config "$CONFIG"} \
            >> "$RUN/worker_w$k.log" 2>&1 &
        PIDS="$PIDS $!"
    done
    say "GPU $gpu: started $n workers (w$k0..w$((k0 + n - 1)))"
}
start_workers 1 "$W1" 0
ADDED0=0
while true; do
    alive=0
    for p in $PIDS; do kill -0 "$p" 2>/dev/null && alive=1; done
    (( alive )) || break
    if (( ! ADDED0 && W0 > 0 )) && is_free 0; then
        nvidia-smi >> "$RUN/nvidia_smi.log" 2>&1
        start_workers 0 "$W0" "$W1"
        ADDED0=1
    fi
    sleep "$CHECK_S" & wait $!
done
FAIL=0
for p in $PIDS; do wait "$p" || FAIL=1; done
(( FAIL )) && { say "a worker exited with an error (see worker_w*.log)"; exit 4; }
say "all workers finished"
FINISHED=1
date '+%Y-%m-%d %H:%M:%S' > "$RUN/DONE.txt"
say "DONE"
