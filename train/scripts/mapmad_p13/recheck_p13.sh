#!/usr/bin/env bash
# Phase 13 Part 3, §0.8 re-check after the clean-up (one GPU placement module, video fixes): the determinism episode of
# G13 row 6a (M21b, exam B, seed 0) once more on GPU 0 with the current code -> byte-identical with the chain's two logs?
#   bash train/scripts/mapmad_p13/recheck_p13.sh        (host, in screen; GPU 0 must be usable for owner part3)
set -u
REPO=/home/nazli/projects/nomad-mapmad
P13=/mnt/shared_disk/nazli/nomad_outputs/mapmad/p13_claim
CFG=/app/visualnav-transformer/mapmad/configs/p13_claim.yaml
LOG=$P13/driver/recheck_part3.log
E=p13b-2azQ1b91cZZ-bed-1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

nvidia-smi >> "$LOG" 2>&1
python3 "$REPO/train/scripts/mapmad_p10/gpu_place.py" clean 0 --owner part3 >> "$LOG" 2>&1 || { say "GPU 0 not clean: no re-check"; exit 1; }
say "determinism re-check: $E, M21b, seed 0, GPU 0 (current code)"
GPU=0 TOKEN="recheck_$(date +%H%M%S)" OWNER=part3 PAIRS_PER_GPU=1 PORT0=5600 RPORT0=5750 RECOG=yoloe \
    bash "$REPO/mapmad/scripts/run_p13_pairs.sh" --config "$CFG" --subdir checks/determinism_part3 --set examb \
    --arms M21b --episodes "$E" --seeds 0 >> "$LOG" 2>&1
new=$P13/checks/determinism_part3/logs/M21b/${E}__s0.jsonl
for r in a b; do
    if cmp "$new" "$P13/checks/determinism_$r/logs/M21b/${E}__s0.jsonl" >> "$LOG" 2>&1; then
        say "re-check vs determinism_$r: byte-identical"
    else
        say "re-check vs determinism_$r: DIFFERENT"
    fi
done
say "RECHECK DONE"
