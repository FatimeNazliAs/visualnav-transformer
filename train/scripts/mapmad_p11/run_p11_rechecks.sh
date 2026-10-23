#!/bin/bash
# Phase 11 §0.8 re-checks at the phase end (kickoff §8 item 2), host side, one GPU (default 0; never the twin's):
#   1. G10 rows 1-2 on ema_9 (CPU) again          -> checks/recheck/g10_rows_1_2_cpu.json, same pass fields as offline/v21_ema9;
#   2. ema_9 re-scored on the frozen search list   -> checks/recheck/search/scores.npz identical to offline/v21_ema9/search;
#   3. determinism episode, arm v21, twice on GPU  -> logs identical (cmp);
#   4. 2 rollouts regenerated (seed 11801 sample, the GPU of the originals) + relabelled
#                                                   -> frames, maps, labels identical to the real drives (timing excluded).
# Everything is written below checks/recheck (config mapmad/configs/p11_recheck.yaml) and <mapmad_data>/p11_recheck_dagger;
# the real runs and drives are only read. Result: checks/recheck/rechecks.json (+ log driver/rechecks.log).
#
#   GPU=0 bash train/scripts/mapmad_p11/run_p11_rechecks.sh p11r-00313-tv_monitor-009 p11r-00251-bed-003
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
P11="${P11_DIR:-$OUT_MOUNT/mapmad/p11_dagger}"
C="${P11/$OUT_MOUNT//outputs}"
WHO=chain
LOG="$P11/driver/rechecks.log"
# shellcheck source=p11_lib.sh
source "$REPO/train/scripts/mapmad_p11/p11_lib.sh"
G="${GPU:-0}"
RCFG=/app/visualnav-transformer/mapmad/configs/p11_recheck.yaml
R="$P11/checks/recheck" RC="$C/checks/recheck" O="$C/offline/v21_ema9"
mkdir -p "$R"
say "re-checks start on GPU $G (commit $(git -C "$REPO" rev-parse --short HEAD)); rollouts: $*"

# 1. G10 rows 1-2 (CPU) + 2. re-score on the frozen search list
dx "" "python -m vint_train.mapmad.checks_g10 --config config/mapmad_v2.yaml --v1-config config/mapmad_p4_C.yaml \
    --reference /outputs/mapmad/p10_v2_train/checks/reference_forward_cpu.pt --out $RC/g10_rows_1_2_cpu.json" >> "$LOG" 2>&1
ck="$(python3 -c "import json,sys; print('/outputs/mapmad/' + json.load(open(sys.argv[1]))['path'])" "$P11/V21.json")"
dx "$G" "python -m vint_train.mapmad.offline_eval score --config config/mapmad_v2.yaml --checkpoint $ck \
    --list /outputs/mapmad/p10_v2_train/offline/lists/search_list.json --out $RC/search" >> "$LOG" 2>&1

# 3. determinism episode (arm v21), twice on GPU $G
e="$(python3 "$SCRIPTS/p11_preflight.py" first-dev "$REPO/mapmad/configs/p11_dagger.yaml" "$P11")"
for r in a b; do
    GPU="$G" TOKEN="recheck_det_${r}_$(date +%H%M%S)" OWNER="$WHO" PAIRS_PER_GPU=1 PORT0=5620 \
        bash "$REPO/mapmad/scripts/run_p11_pairs.sh" --config "$RCFG" --subdir "determinism_$r" --set dev --arms v21 \
        --episodes "$e" --seeds 0 >> "$LOG" 2>&1
done

# 4. two rollouts regenerated (arm v2, seed 0, drives saved) and relabelled
GPU="$G" TOKEN="recheck_roll_$(date +%H%M%S)" OWNER="$WHO" PAIRS_PER_GPU=2 PORT0=5630 \
    bash "$REPO/mapmad/scripts/run_p11_pairs.sh" --config "$RCFG" --subdir rollouts --set rollout --arms v2 \
    --episodes "$@" --seeds 0 --save-drives >> "$LOG" 2>&1
dh "python mapmad/scripts/p11_labels.py label --config $RCFG --workers 2" >> "$LOG" 2>&1

dh "python mapmad/scripts/p11_recheck_compare.py --p11 $C --episode $e --drives /mapmad_data/habitat_mapmad_v2_dagger \
    --redrives /mapmad_data/p11_recheck_dagger --out $RC/rechecks.json $*" >> "$LOG" 2>&1
say "re-checks done: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(' '.join(f'{k}={v[\"pass\"]}' for k, v in d.items()))" "$R/rechecks.json")"
