#!/bin/bash
# usage: run_supervised.sh <label C|D> <config>. p4_rules_v2.md §crashes: train; if it did not finish, resume ONCE from
# the run's latest.pth; if that does not finish either (or there is no latest.pth), write STOP_<label>.txt.
# Paths from the environment (values used on cukurovaai: scripts/mapmad_p4/README.md).
: "${MAPMAD_P4_HOST_DIR:?set MAPMAD_P4_HOST_DIR}" "${MAPMAD_P4_CONTAINER_DIR:?set MAPMAD_P4_CONTAINER_DIR}"
H=$MAPMAD_P4_HOST_DIR; C=$MAPMAD_P4_CONTAINER_DIR; L=$1; CFG=$2
log() { echo "- $(date '+%F %T') [run $L] $*" | tee -a $H/phase4_log.md; }
train() { docker exec -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MPLCONFIGDIR=/tmp/mpl naz_mapmad bash -lc "cd /app/visualnav-transformer/train && python train.py -c $CFG $1" > $2 2>&1; }
finished() { d=$(ls -td $H/${L}_2* 2>/dev/null | head -1); [ -n "$d" ] && [ -f $d/ema_19.pth ] && tr '\r' '\n' < $1 | grep -a -q "FINISHED TRAINING"; }
log "start: python train.py -c $CFG (expandable_segments:True)"
train "" $H/run_$L.log
if finished $H/run_$L.log; then log "finished"; exit 0; fi
d=$(ls -td $H/${L}_2* 2>/dev/null | head -1)
if [ -z "$d" ] || [ ! -f $d/latest.pth ]; then log "CRASH before any latest.pth (run_$L.log) -> STOP"; echo STOP > $H/STOP_$L.txt; exit 1; fi
log "CRASH 1 (run_$L.log) -> RESUME 1 from ${d/$H/$C}/latest.pth"
train "--resume ${d/$H/$C}/latest.pth" $H/run_${L}_resume1.log
if finished $H/run_${L}_resume1.log; then log "finished after resume 1"; exit 0; fi
log "CRASH 2 (run_${L}_resume1.log) -> STOP, decide with Naz"; echo STOP > $H/STOP_$L.txt; exit 1
