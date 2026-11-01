#!/usr/bin/env bash
# Phase 13, after the gate: arm VF's videos for the M21b / M21bh rule picks (videos_rule/<set>/rule_list.json), from
# the exam logs (runs/hm3d, runs/examb; no new runs). Then `check` per folder for M21b, M21bh and VF.
#   bash train/scripts/mapmad_p13/render_vf_videos.sh        (host, in screen; GPUs 0 and 1, one shard each)
# VF's replay runs the VF agent as in the exam, so each shard has its own YOLOE server (port 5790 + shard) AND its own
# BLIP-2 ITM server (port 5795 + shard); both servers take one client at a time.
set -u
CFG=mapmad/configs/p13_claim.yaml
C=/outputs/mapmad/p13_claim
LOG=/mnt/shared_disk/nazli/nomad_outputs/mapmad/p13_claim/driver
WHO=part3
NSH=2
KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/videos_vf.log"; }

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >> "$LOG/videos_vf.log"
for k in 0 1; do
    docker exec -d -e MAPMAD_SIDECAR_KEY="$KEY" -e CUDA_VISIBLE_DEVICES="$k" -e HOME=/model_cache/home \
        -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog python -m mapmad_recog.server --model yoloe \
        --port $(( 5790 + k )) --text-reuse-out "$C/videos_rule/text_reuse_yoloe_vf_shard$k.json"
    docker exec -d -e MAPMAD_SIDECAR_KEY="$KEY" -e CUDA_VISIBLE_DEVICES="$k" -e OMP_NUM_THREADS=4 \
        -e HOME=/model_cache/home -e HF_HOME=/model_cache/hf -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog \
        bash -c "python -m mapmad_recog.itm_server --port $(( 5795 + k )) >> $C/videos_rule/itm_vf_shard$k.log 2>&1"
done
say "2 YOLOE servers (5790-5791) + 2 ITM servers (5795-5796), GPU k = shard k"

shard() {  # k -> shard k/NSH of both rule lists, arm VF
    local k=$1
    run() { docker exec -e MAPMAD_SIDECAR_KEY="$KEY" -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer \
        naz_mapmad_habitat python mapmad/scripts/p13_videos.py render-plan --config "$CFG" --gpu "$k" \
        --recog-port $(( 5790 + k )) --itm-port $(( 5795 + k )) --shard "$k/$NSH" --arms VF "$@" \
        >> "$LOG/videos_vf_shard$k.log" 2>&1; }
    run --subdir runs/hm3d --set hm3d --list "$C/videos_rule/hm3d/rule_list.json" --out "$C/videos_rule/hm3d" || return 1
    run --subdir runs/examb --set examb --list "$C/videos_rule/examb/rule_list.json" --out "$C/videos_rule/examb" || return 1
}
pids=()
for k in 0 1; do shard "$k" & pids+=($!); done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
say "render shards done (any failed: $bad)"
docker exec naz_mapmad_recog pkill -f -- "mapmad_recog.server --model yoloe --port 579" 2>/dev/null
docker exec naz_mapmad_recog pkill -f -- "mapmad_recog.itm_server --port 579" 2>/dev/null

for s in hm3d examb; do
    docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p13_videos.py check \
        --config "$CFG" --out "$C/videos_rule/$s" --list "$C/videos_rule/$s/rule_list.json" --arms M21b M21bh VF \
        --report "$C/checks/video_rule_${s}_check.json" >> "$LOG/videos_vf.log" 2>&1
    say "check videos_rule/$s (M21b, M21bh, VF) -> exit $?"
done
say "VIDEOS VF DONE"
