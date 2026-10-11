#!/bin/bash
# usage: gpu_mem_log.sh <run on GPU0> <run on GPU1> [csv name]. Every 10 s: per GPU the memory nvidia-smi reports
# (PyTorch reserved + CUDA context) and the epoch the run on it is in (last "Start MapMaD epoch" of its newest
# run_<label>*.log). Phase 4 ran it as gpu_mem_log.sh (A B -> gpu_mem.csv) and gpu_mem_log_v2.sh (C D -> gpu_mem_v2.csv).
: "${MAPMAD_P4_HOST_DIR:?set MAPMAD_P4_HOST_DIR}"
D=$MAPMAD_P4_HOST_DIR; R0=$1; R1=$2; CSV=${3:-gpu_mem.csv}
[ -f $D/$CSV ] || echo "time,gpu,run,epoch_index,mem_used_mib" > $D/$CSV
epoch_of() { log=$(ls -t $D/run_$1*.log | head -1); tr '\r' '\n' < $log | grep -a -o "Start MapMaD epoch [0-9]*" | tail -1 | awk '{print $4}'; }
while true; do
  t=$(date '+%F %T'); e0=$(epoch_of $R0); e1=$(epoch_of $R1)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | while IFS=', ' read -r g m; do
    if [ "$g" = 0 ]; then echo "$t,0,$R0,$e0,$m"; else echo "$t,1,$R1,$e1,$m"; fi
  done >> $D/$CSV
  sleep 10
done
