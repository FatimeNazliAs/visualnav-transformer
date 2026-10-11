#!/bin/bash
# Phase 4 confirmation post-run (p4_rules_v2.md), Naz's conditions at launch:
# - C's final eval + the G4' table run as soon as C ends (never wait on D); STOP_C.txt -> no eval, POST_V2_STOPPED.txt;
# - D's eval runs only if D finishes (no STOP_D.txt, ema_19 present);
# - the GPU-memory logger stops when both runs have ended. No selection.
# Paths from the environment (values used on cukurovaai: scripts/mapmad_p4/README.md).
: "${MAPMAD_P4_HOST_DIR:?set MAPMAD_P4_HOST_DIR}" "${MAPMAD_P4_CONTAINER_DIR:?set MAPMAD_P4_CONTAINER_DIR}" "${MAPMAD_OFFLINE_LIST:?set MAPMAD_OFFLINE_LIST}" "${MAPMAD_OFFICIAL_WEIGHTS:?set MAPMAD_OFFICIAL_WEIGHTS}"
H=$MAPMAD_P4_HOST_DIR; C=$MAPMAD_P4_CONTAINER_DIR
log() { echo "- $(date '+%F %T') [p4_post_v2] $*" | tee -a $H/phase4_log.md; }
dx() { docker exec -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e MPLCONFIGDIR=/tmp/mpl naz_mapmad bash -lc "cd /app/visualnav-transformer/train && $1"; }
alive() { screen -ls | grep -q "mapmad_p4_$1"; }
evalone() {  # $1 gpu, $2 checkpoint, $3 out dir
  dx "CUDA_VISIBLE_DEVICES=$1 python -m vint_train.mapmad.offline_eval score --config config/mapmad_p4.yaml --checkpoint $2 --list $MAPMAD_OFFLINE_LIST --out $3 && python -m vint_train.mapmad.offline_eval compare --scores $3/scores.npz --out $3 && CUDA_VISIBLE_DEVICES=$1 python -m vint_train.mapmad.offline_eval gostanford --config config/mapmad_p4.yaml --checkpoint $2 --official $MAPMAD_OFFICIAL_WEIGHTS --out $3 && python -m vint_train.mapmad.training_checks gostanford-report --scores $3/gostanford_scores.npz --subset $C/subsets/gostanford_subset.json --out $3/gostanford_report.json"
}
gate() {  # G4' rows from C's final eval -> final_eval_v2/g4_prime.json (row 4 report only)
  dx "python - <<'PY'
import json
d='$C/final_eval_v2/C_ema_19'
c=json.load(open(d+'/compare.json')); g=json.load(open(d+'/gostanford.json'))
r=c['row4']
out={'checkpoint': g['checkpoint'],
     'row2': {'diff': r['diff'], 'diff_ci_drive': r['diff_ci_drive'], 'R': r['R'], 'R_ci_drive': r['R_ci_drive'],
              'R_minus_0690': r['R'] - 0.690, 'R_minus_B_0767': r['R'] - 0.767,
              'pass': bool(r['diff_ci_drive'][1] < 0 and r['R'] > 0.677)},
     'row3': {'official': g['official'], 'mapmad': g['mapmad'], 'diff': g['diff'], 'diff_ci_sample': g['diff_ci_sample'],
              'pass': bool(g['mapmad'] <= g['official'] + 0.10)},
     'row4_report': {'attention_share_map_goal': c['attention_share'].get('map_goal'), 'step0_share': 0.354}}
json.dump(out, open('$C/final_eval_v2/g4_prime.json', 'w'), indent=1)
print(json.dumps(out))
PY"
}
mkdir -p $H/final_eval_v2
log "waiting for run C (screen mapmad_p4_C)"
while alive C; do sleep 60; done
dC=$(ls -td $H/C_2* 2>/dev/null | head -1)
if [ -f $H/STOP_C.txt ] || [ -z "$dC" ] || [ ! -f $dC/ema_19.pth ]; then
  log "run C STOPPED (STOP_C.txt or no ema_19) -> no eval; tell Naz"; echo "STOP: run C did not finish" > $H/POST_V2_STOPPED.txt
else
  sha256sum $dC/ema_19.pth > $H/final_eval_v2/C_ema_19.sha256
  log "final eval C ema_19 on GPU0 -> final_eval_v2/C_ema_19"
  evalone 0 ${dC/$H/$C}/ema_19.pth $C/final_eval_v2/C_ema_19 > $H/final_eval_v2/C_ema_19.log 2>&1 && gate > $H/final_eval_v2/g4_prime.log 2>&1
  log "C eval done (exit $?); G4' rows -> final_eval_v2/g4_prime.json"
  echo "C done" > $H/POST_V2_C_DONE.txt
fi
log "waiting for run D (screen mapmad_p4_D)"
while alive D; do sleep 60; done
screen -S mapmad_p4_gpumem -X quit && log "both runs ended -> stopped screen mapmad_p4_gpumem"
[ -f $H/POST_V2_STOPPED.txt ] && exit 0
dD=$(ls -td $H/D_2* 2>/dev/null | head -1)
if [ -f $H/STOP_D.txt ] || [ -z "$dD" ] || [ ! -f $dD/ema_19.pth ]; then
  log "run D did not finish (STOP_D.txt or no ema_19) -> no D eval (report the reason)"
  dx "python -m vint_train.mapmad.plot_curves --run C=${dC/$H/$C} --out $C/curves_v2" > $H/post_v2_curves.log 2>&1
else
  sha256sum $dD/ema_19.pth > $H/final_eval_v2/D_ema_19.sha256
  log "final eval D ema_19 on GPU1 -> final_eval_v2/D_ema_19"
  evalone 1 ${dD/$H/$C}/ema_19.pth $C/final_eval_v2/D_ema_19 > $H/final_eval_v2/D_ema_19.log 2>&1
  dx "python -m vint_train.mapmad.plot_curves --run C=${dC/$H/$C} --run D=${dD/$H/$C} --out $C/curves_v2" > $H/post_v2_curves.log 2>&1
fi
log "done (final_eval_v2/, curves_v2/)"; echo DONE > $H/POST_V2_DONE.txt
