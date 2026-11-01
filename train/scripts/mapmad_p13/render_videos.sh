#!/usr/bin/env bash
# Phase 13 Part 3: render the 13.1 videos the chain's render left out + the M21b / M21bh rule-pick videos.
#   bash train/scripts/mapmad_p13/render_videos.sh        (host, in screen; GPUs 0 and 1, 2 shards each)
# One YOLOE server PER SHARD (port 5790 + shard): the server takes one client at a time, so two shards on one server
# made the second one time out (Phase 13 chain: shards 0 and 3 = every M21 and F video lost). Each shard renders, in
# order: the main plan (videos.list x videos.render_arms, resumable: done videos are skipped), then the rule lists of
# videos_rule/<set>/rule_list.json (p13_videos.py pick-rules) for M21b + M21bh. Then `check` per folder (with --config:
# every planned video must exist).
set -u
REPO=/home/nazli/projects/nomad-mapmad
CFG=mapmad/configs/p13_claim.yaml
C=/outputs/mapmad/p13_claim
LOG=/mnt/shared_disk/nazli/nomad_outputs/mapmad/p13_claim/driver
WHO=part3
NSH=4
KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/videos_part3.log"; }

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >> "$LOG/videos_part3.log"
for k in 0 1 2 3; do
    g=$(( k / 2 ))
    docker exec -d -e MAPMAD_SIDECAR_KEY="$KEY" -e CUDA_VISIBLE_DEVICES="$g" -e HOME=/model_cache/home \
        -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog python -m mapmad_recog.server --model yoloe \
        --port $(( 5790 + k )) --text-reuse-out "$C/videos/text_reuse_yoloe_shard$k.json"
done
say "4 YOLOE servers (ports 5790-5793; GPU 0: shards 0-1, GPU 1: shards 2-3)"

shard() {  # k -> renders shard k/NSH of the main plan and of both rule lists
    local k=$1 g=$(( $1 / 2 )) port=$(( 5790 + $1 ))
    run() { docker exec -e MAPMAD_SIDECAR_KEY="$KEY" -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer \
        naz_mapmad_habitat python mapmad/scripts/p13_videos.py render-plan --config "$CFG" --gpu "$g" \
        --recog-port "$port" --shard "$k/$NSH" "$@" >> "$LOG/videos_part3_shard$k.log" 2>&1; }
    run --subdir videos_rerun --out "$C/videos" || return 1
    run --subdir runs/hm3d --set hm3d --list "$C/videos_rule/hm3d/rule_list.json" --arms M21b M21bh \
        --out "$C/videos_rule/hm3d" || return 1
    run --subdir runs/examb --set examb --list "$C/videos_rule/examb/rule_list.json" --arms M21b M21bh \
        --out "$C/videos_rule/examb" || return 1
}
pids=()
for k in 0 1 2 3; do shard "$k" & pids+=($!); done
bad=0
for p in "${pids[@]}"; do wait "$p" || bad=1; done
say "render shards done (any failed: $bad)"
docker exec naz_mapmad_recog pkill -f -- "mapmad_recog.server --model yoloe --port 579" 2>/dev/null

chk() { docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p13_videos.py check \
    --config "$CFG" "$@" >> "$LOG/videos_part3.log" 2>&1; say "check $* -> exit $?"; }
chk --out "$C/videos" --report "$C/checks/video_check.json"
chk --out "$C/videos_rule/hm3d" --list "$C/videos_rule/hm3d/rule_list.json" --arms M21b M21bh \
    --report "$C/checks/video_rule_hm3d_check.json"
chk --out "$C/videos_rule/examb" --list "$C/videos_rule/examb/rule_list.json" --arms M21b M21bh \
    --report "$C/checks/video_rule_examb_check.json"
say "VIDEOS PART3 DONE"
