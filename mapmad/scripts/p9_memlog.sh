#!/bin/bash
# Phase 9 memory log (host, in screen through p9_job.sh): every INTERVAL s (default 30) one line per source, so a
# recogniser server that dies has its cause on record. Never touches a GPU or another user's process (read-only).
#   host RAM + swap (free -m) | each mapmad_recog.server: RSS, CPU, state | GPU memory per GPU (nvidia-smi)
#   | new kernel lines about OOM / killed processes / segfaults (dmesg)
#
#   bash mapmad/scripts/p9_job.sh memlog <p9>/driver bash mapmad/scripts/p9_memlog.sh <p9>/driver/memlog.txt
set -uo pipefail
OUT="${1:?output file}"
INTERVAL="${INTERVAL:-30}"
REC=naz_mapmad_recog
seen=$(dmesg 2>/dev/null | wc -l)
while true; do
    t=$(date '+%F %T')
    free -m | awk -v t="$t" '/^Mem:/ {printf "%s ram used_mb=%s free_mb=%s avail_mb=%s", t, $3, $4, $7} /^Swap:/ {printf " swap_used_mb=%s/%s\n", $3, $2}' >> "$OUT"
    docker exec "$REC" ps -eo pid,rss,pcpu,stat,args 2>/dev/null | awk -v t="$t" '/mapmad_recog.server/ && !/bash -c/ {printf "%s server pid=%s rss_mb=%.0f cpu=%s stat=%s model=%s\n", t, $1, $2/1024, $3, $4, $9}' >> "$OUT"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | awk -v t="$t" '{print t " gpu " $0}' >> "$OUT"
    n=$(dmesg 2>/dev/null | wc -l)
    if (( n > seen )); then
        dmesg 2>/dev/null | tail -n $((n - seen)) | grep -iE "out of memory|oom|killed process|segfault|general protection" | sed "s/^/$t dmesg /" >> "$OUT"
        seen=$n
    fi
    sleep "$INTERVAL"
done
