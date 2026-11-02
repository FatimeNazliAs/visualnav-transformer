#!/usr/bin/env bash
# 14.0 practice-set extension: p14d candidates on the unused MP3D train homes (one worker per home, both GPUs), then
# freeze p14d + the practice set halves (make_p14_sets.py). Host side; work runs in naz_mapmad_habitat.
set -euo pipefail
WORKERS=${WORKERS:-11}
LOG=${P14_DIR:-/mnt/shared_disk/nazli/nomad_outputs/mapmad/p14a_finish}/sets_logs
mkdir -p "$LOG"
pids=()
for w in $(seq 0 $((WORKERS - 1))); do
  docker exec naz_mapmad_habitat bash -lc "cd /app/visualnav-transformer && python mapmad/scripts/make_p14_sets.py candidates --gpu $((w % 2)) --worker $w --workers $WORKERS" > "$LOG/candidates_w$w.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || { echo "a candidate worker failed (see $LOG)"; exit 1; }
docker exec naz_mapmad_habitat bash -lc "cd /app/visualnav-transformer && python mapmad/scripts/make_p14_sets.py freeze" > "$LOG/freeze.log" 2>&1
echo "SETS DONE"
