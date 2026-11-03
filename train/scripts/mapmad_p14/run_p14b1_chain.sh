#!/bin/bash
# Phase 14b-1 chained driver (kickoff .p14b1_kickoff.md §2-§4, p14b1_rules.md). Host side, in screen p14b1_chain (start
# with launch_p14b1.sh). No training, no driving, no exam: the Phase 14a practice runs are replayed and re-scored.
# Steps:
#   0. pre-flight: committed code, rules sha256 (outputs = repo copy), hashes of every 14a log / evidence / dets file
#      (checks/hashes_before.json, gate row 6), 14b1.2 temperature fit (CPU, seconds);
#   1. 14b1.1 replay of the 1,095 baseline runs: per usable GPU SHARDS_PER_GPU shards, each with its own YOLOE server;
#      a run replays on the GPU that first drove it (byte-identical, kickoff §3);
#   2. the 1,971 driving-test runs replayed in the background (same layout) while the CPU runs 14b1.3: score every
#      variant -> pick (rows 1-7) -> c_cat runs -> final (row 8, keep rule, recommended, truth ceiling) -> marks;
#   3. step 7: driving-test re-score (recommended variant);
#   4. 14b1.5: the 8 mark-check videos with old vs new marks (one YOLOE server per shard);
#   5. hashes after, G14b1 draft -> CHAIN DONE.
# Automatic stops (kickoff §7) -> STOP_reason.txt, STATUS.md STOPPED, wait. EXIT trap: any other exit writes
# STOP_reason.txt. Resumable: every step skips what is done (replay files, scored runs). The body is one function.
# DRY (never results): P14_DIR=<host p14b1_dry>, P14_CFG=<container p14b1_dry.yaml>, P14_ALLOW_DIRTY=1, P14_GPUS="0 1",
# P14_EPISODES="<ids>", SHARDS_PER_GPU=1, P14B_VIDEO_PICKS=<picks json (container path)>.
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P14="${P14_DIR:-$OUT_MOUNT/mapmad/p14b1_marks}"
    C="${P14/$OUT_MOUNT//outputs}"
    WHO=chain
    LOG="$P14/driver/chain.log"
    mkdir -p "$P14/driver" "$P14/checks"
    exec 2>> "$LOG"
    export P14_LOGNAME=phase14b1_log.md P14_PHASE=14b-1
    export P14_PASTE="continue Phase 14b-1: read STATUS.md and the chain log, then finish the tables, the G14b1 table and the summary"
    export P14_CFG="${P14_CFG:-/app/visualnav-transformer/mapmad/configs/p14b1_marks.yaml}"
    # shellcheck source=p14_lib.sh
    source "$REPO/train/scripts/mapmad_p14/p14_lib.sh"
    export P14_OUTPUTS="$OUT_MOUNT/mapmad" P14_EPISODES="${P14_EPISODES:-}"
    local RUNS_CFG=/app/visualnav-transformer/mapmad/configs/p14a_finish.yaml
    local S="${SHARDS_PER_GPU:-6}"
    local REPLAY_GPU_S=$(( 12 * 3600 ))   # 14b1.1: <= 8 GPU-h; stop > 1.5x
    local BENCH_S=$(( 6 * 3600 ))         # 14b1.3: <= 4 h CPU; stop > 1.5x
    local DRIVE_S=$(( 3 * 3600 / 2 ))     # 14b1.4: <= 1 h CPU; stop > 1.5x
    local VIDEO_S=$(( 3 * 3600 / 2 ))     # 14b1.5: <= 1 h; stop > 1.5x
    FINISHED=0
    BG_PID=""
    DRV_PID=""
    REPLAY_USED_S=0
    [[ -f "$P14/STOP_reason.txt" ]] && mv "$P14/STOP_reason.txt" "$P14/driver/STOP_reason_$(date +%m%d_%H%M%S).txt"

    stop() {  # reason [question] -> automatic stop; waits for Naz
        { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "reason: $1"
          echo "question: ${2:-how to go on? (recommended: read the log, fix, relaunch: every step resumes)}"
          echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P14/STOP_reason.txt"
        status step="${STEP:-?}" reason="$1" error="$1" --state STOPPED
        say "AUTOMATIC STOP: $1"
        FINISHED=1
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        [[ -n "${DRV_PID:-}" ]] && kill "$DRV_PID" 2>/dev/null
        for pat in "p14b_replay.py" "mapmad_recog.server --model yoloe --port 58" "p14b_bench" "p14b_videos.py"; do
            kill_inside "$pat"; done  # our own processes only (patterns of this chain)
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "exit code: $code"
              echo "question: relaunch launch_p14b1.sh after reading driver/chain.log? (recommended: yes, it resumes)"
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P14/STOP_reason.txt"
            status reason="driver exit $code" error="driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    fallback() { status fallback="$1"; say "FALLBACK: $1"; }
    bench() { dx "" "export P14_EPISODES='$P14_EPISODES' OMP_NUM_THREADS=1; python -m vint_train.mapmad.closed_loop.p14b_bench $* --config $CFG"; }
    n_logs() {  # subdir arms... -> number of 14a runs to replay (P14_EPISODES limits DRY runs)
        local sub="$1"; shift
        python3 - "$OUT_MOUNT/mapmad/p14a_finish/runs/$sub" "$P14_EPISODES" "$@" <<'EOF'
import os, sys
root, only, arms = sys.argv[1], set(sys.argv[2].split()), sys.argv[3:]
n = 0
for a in arms:
    for f in os.listdir(os.path.join(root, "logs", a)):
        if f.endswith(".jsonl") and (not only or f.rsplit("__s", 1)[0] in only):
            n += 1
print(n)
EOF
    }
    servers_gone() {  # pattern -> waits (<= 90 s) until no matching process is left in naz_mapmad_recog
        local k
        for k in $(seq 1 18); do
            docker exec naz_mapmad_recog pgrep -f -- "$1" > /dev/null 2>&1 || return 0
            pause 5
        done
        say "note: '$1' still running 90 s after the kill"
    }
    n_done() {  # subdir arms... -> replay files written (npz + json)
        local sub="$1" n=0 a; shift
        for a in "$@"; do n=$(( n + $(ls "$P14/replay/$sub/$a/" 2>/dev/null | grep -c '\.json$') )); done
        echo "$n"
    }
    replay_step() {  # label subdir arms... -> 0 when every run is replayed; stops on budget / hang / second failure
        local label="$1" sub="$2"; shift 2
        local arms=("$@") total done gs ngpu g s port round=0 t0 last_done=-1 last_t key
        local -a pids=()
        total=$(n_logs "$sub" "${arms[@]}")
        while true; do
            done=$(n_done "$sub" "${arms[@]}")
            (( done >= total )) && { say "$label: all $total runs replayed"; return 0; }
            round=$(( round + 1 ))
            (( round > 2 )) && stop "$label: $done of $total runs replayed after two rounds (driver/replay_${sub}_*.log)"
            gs=""
            status --state WAITING step="$STEP" note="$label: waiting for a free GPU"
            until [[ -n "$gs" ]]; do gs="$(usable_now)"; [[ -n "$gs" ]] || pause "$CHECK_S"; done
            ngpu=$(echo $gs | wc -w)
            local tries=0
            while (( ngpu == 1 && tries < 10 )); do  # a GPU just released by our previous step: give it 5 min
                pause 30; tries=$(( tries + 1 ))
                gs="$(usable_now)"; [[ -n "$gs" ]] || gs="$(free_for | cut -d' ' -f1)"
                ngpu=$(echo $gs | wc -w)
            done
            nvidia-smi >> "$P14/driver/nvidia_smi.log" 2>&1
            key="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
            local orig_flag=1
            if (( ngpu == 1 )); then
                fallback "$label: only GPU $gs usable -> every run replayed on it (other-GPU runs: score diff <= 1e-4 accepted, else F1-box rows)"
                orig_flag=0
            fi
            pids=()
            for g in $gs; do
                for (( s = 0; s < S; s++ )); do
                    port=$(( 5800 + 10 * g + s ))
                    docker exec -d -e MAPMAD_SIDECAR_KEY="$key" -e CUDA_VISIBLE_DEVICES="$g" -e HOME=/model_cache/home \
                        -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog bash -c \
                        "exec python -m mapmad_recog.server --model yoloe --port $port >> $C/driver/recog_${sub}_g${g}_s${s}.log 2>&1"
                    local og=() ep=()
                    (( orig_flag )) && og=(--orig-gpu "$g")
                    [[ -n "$P14_EPISODES" ]] && ep=(--episodes $P14_EPISODES)
                    docker exec -e MAPMAD_SIDECAR_KEY="$key" -e MAPMAD_P13_OWNER="$WHO" -e OMP_NUM_THREADS=2 \
                        -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p14b_replay.py \
                        --config "$CFG" --subdir "runs/$sub" --arms "${arms[@]}" --gpu "$g" --recog-port "$port" \
                        "${og[@]}" --shard "$s/$S" "${ep[@]}" >> "$P14/driver/replay_${sub}_g${g}_s${s}.log" 2>&1 &
                    pids+=($!)
                done
                say "GPU event: GPU $g taken by $label ($S shards, YOLOE ports $(( 5800 + 10 * g ))..$(( 5800 + 10 * g + S - 1 )))"
            done
            status --state RUNNING gpus="$gs" mode="Alone" note="$label: $done/$total replayed"
            t0=$SECONDS; last_t=$SECONDS
            while :; do
                local alive=0 p
                for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
                done=$(n_done "$sub" "${arms[@]}")
                if (( done != last_done )); then last_done=$done; last_t=$SECONDS; fi
                REPLAY_USED_S=$(( REPLAY_USED_S + 60 * ngpu ))
                status note="$label: $done/$total replayed (round $round); GPU-time used $(( REPLAY_USED_S / 60 )) min of 720"
                (( REPLAY_USED_S > REPLAY_GPU_S )) && { kill_inside "p14b_replay.py"; stop "budget: replay > 1.5 x 8 GPU-h"; }
                if (( SECONDS - last_t > ${SILENT_S:-1800} )); then
                    say "HANG: no replay file for > ${SILENT_S:-1800} s in $sub -> killing this round"
                    kill_inside "p14b_replay.py --config $CFG --subdir runs/$sub"; break
                fi
                for g in $gs; do
                    gpu_clean "$g" || say "GPU event: another user's process on GPU $g during $label (logged; our current runs finish)"
                done
                (( alive )) || break
                pause 60
            done
            for p in "${pids[@]}"; do wait "$p" 2>/dev/null || say "$label: a replay shard exited non-zero (driver/replay_${sub}_*.log)"; done
            kill_inside "mapmad_recog.server --model yoloe --port 58"
            servers_gone "mapmad_recog.server --model yoloe --port 58"
            say "$label round $round: $(n_done "$sub" "${arms[@]}")/$total replayed in $(( (SECONDS - t0) / 60 )) min"
        done
    }

    # --- 0. pre-flight ----------------------------------------------------------------------------------------------
    STEP="0 pre-flight"
    status --reset step="$STEP" started="$(date '+%a %d %b %H:%M')" mode="Alone" --state RUNNING
    say "chain start (P14=$P14, CFG=$CFG, shards/GPU $S, episodes '${P14_EPISODES}')"
    if [[ -z "${P14_ALLOW_DIRTY:-}" ]] && [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
        stop "uncommitted changes in tracked files (launcher commit first)"
    fi
    say "code at $(git -C "$REPO" rev-parse HEAD)"
    if [[ -z "${P14_ALLOW_DIRTY:-}" ]]; then
        local want
        want="$(cut -d' ' -f1 "$P14/p14b1_rules.sha256")"
        [[ "$(sha256sum "$P14/p14b1_rules.md" | cut -d' ' -f1)" == "$want" ]] || stop "p14b1_rules.md sha256 mismatch"
        [[ "$(sha256sum "$REPO/mapmad/configs/p14b1_rules.md" | cut -d' ' -f1)" == "$want" ]] || stop "repo copy of p14b1_rules.md differs"
    fi
    [[ -f "$P14/checks/hashes_before.json" ]] || run_bg bench hashes --tag before >> "$P14/driver/bench.log" 2>&1 \
        || stop "hashing the 14a runs failed (driver/bench.log)"
    [[ -f "$P14/calib/calib.json" ]] || run_bg bench calib >> "$P14/driver/bench.log" 2>&1 \
        || stop "temperature fit failed (driver/bench.log)"
    say "pre-flight done; T = $(python3 -c "import json; print(json.load(open('$P14/calib/calib.json'))['T'])")"

    # --- 1. replay of the baseline runs ---------------------------------------------------------------------------
    STEP="1 replay baseline (14b1.1)"
    status step="$STEP"
    replay_step "replay baseline" baseline M21b M21bh N F VF
    python3 - "$P14/replay/baseline" > "$P14/checks/replay_check_baseline.txt" <<'EOF'
import glob, json, sys
r = [json.load(open(f)) for f in glob.glob(sys.argv[1] + "/*/*.json")]
s = lambda k: sum(x[k] for x in r)
same = [x for x in r if x.get("orig_gpu") in (None, x["gpu"])]
print(json.dumps({"runs": len(r), "same_gpu": len(same), "max_raw_diff": max(x["max_raw_diff"] for x in r),
                  "fallback_rows": s("fallback_rows"), "det_rows": s("det_rows"), "det_base_same": s("det_base_same"),
                  "true_rows": s("true_rows"), "true_base_same": s("true_base_same"),
                  "seconds_per_run": s("seconds") / max(len(r), 1)}))
EOF
    say "replay check (baseline): $(cat "$P14/checks/replay_check_baseline.txt")"
    local eta
    eta="$(python3 - "$P14/checks/replay_check_baseline.txt" "$(n_logs a1 M21b_a1 M21bh_a1 N_a1)" "$S" <<'EOF'
import json, sys, time
r = json.load(open(sys.argv[1]))
left = 3 * int(sys.argv[2]) * r["seconds_per_run"] / (2 * int(sys.argv[3]))  # 3 switch folders, 2 GPUs x S shards
print(time.strftime("%a %H:%M", time.localtime(time.time() + left + 45 * 60)))
EOF
)"
    status come_back="~$eta (driving replay + bench + videos + gate draft)"
    say "estimate: CHAIN DONE ~$eta"

    # --- 2. driving-test replay (background) + stop bench (CPU) ---------------------------------------------------
    STEP="2 bench (14b1.3) + driving replay"
    status step="$STEP"
    ( STEP="2b replay driving tests"; replay_step "replay a1" a1 M21b_a1 M21bh_a1 N_a1 && \
      replay_step "replay w4" w4 M21b_w4 M21bh_w4 N_w4 && replay_step "replay a1w4" a1w4 M21b_a1w4 M21bh_a1w4 N_a1w4 ) \
        >> "$P14/driver/replay_driving.log" 2>&1 &
    DRV_PID=$!
    local bt0=$SECONDS
    local W="${BENCH_WORKERS:-8}"
    run_limit "$BENCH_S" "p14b_bench score" bench score --subdir runs/baseline --workers "$W" >> "$P14/driver/bench.log" 2>&1 \
        || stop "stop bench scoring failed (driver/bench.log)"
    run_bg bench pick >> "$P14/driver/bench.log" 2>&1 || stop "stop bench pick failed (driver/bench.log)"
    run_limit $(( BENCH_S - (SECONDS - bt0) )) "p14b_bench score" bench score --subdir runs/baseline --ccat \
        --workers "$W" >> "$P14/driver/bench.log" 2>&1 || stop "c_cat scoring failed (driver/bench.log)"
    run_bg bench final >> "$P14/driver/bench.log" 2>&1 || stop "stop bench final failed (driver/bench.log)"
    run_limit $(( BENCH_S - (SECONDS - bt0) )) "p14b_bench marks" bench marks --workers "$W" >> "$P14/driver/bench.log" 2>&1 \
        || stop "mark statistics failed (driver/bench.log)"
    say "stop bench done in $(( (SECONDS - bt0) / 60 )) min: stopbench/stopbench.md, stopbench_truth/truth.md, tables/marks.md"
    STEP="2b wait for the driving-test replay"
    status step="$STEP"
    wait "$DRV_PID" || stop "driving-test replay failed: $(grep '^reason:' "$P14/STOP_reason.txt" 2>/dev/null | cut -c9-) (driver/replay_driving.log)"
    DRV_PID=""

    # --- 3. driving-test re-score ---------------------------------------------------------------------------------
    STEP="3 driving re-score (step 7)"
    status step="$STEP" gpus="" --state RUNNING
    run_limit "$DRIVE_S" "p14b_bench drive" bench drive --workers 20 >> "$P14/driver/bench.log" 2>&1 \
        || stop "driving re-score failed (driver/bench.log)"

    # --- 4. videos --------------------------------------------------------------------------------------------------
    STEP="4 videos (14b1.5)"
    status step="$STEP" --state RUNNING
    local VM="$C/videos_marks_fixed" picks="${P14B_VIDEO_PICKS:-/outputs/mapmad/p14a_finish/videos_marks/picks.json}"
    if [[ ! -f "$P14/videos_marks_fixed/video_check.json" ]]; then
        local gs="" vg k=0 vpids=() nsh vkey vt0=$SECONDS vfail=0
        until [[ -n "$gs" ]]; do gs="$(usable_now)"; [[ -n "$gs" ]] || pause "$CHECK_S"; done
        nsh=$(( $(echo $gs | wc -w) * 2 ))
        vkey="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
        for vg in $gs; do
            for s in 0 1; do  # one YOLOE server PER SHARD
                docker exec -d -e MAPMAD_SIDECAR_KEY="$vkey" -e CUDA_VISIBLE_DEVICES="$vg" -e HOME=/model_cache/home \
                    -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog bash -c \
                    "exec python -m mapmad_recog.server --model yoloe --port $(( 5850 + k )) >> $C/driver/recog_videos_$k.log 2>&1"
                docker exec -e MAPMAD_SIDECAR_KEY="$vkey" -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer \
                    naz_mapmad_habitat python mapmad/scripts/p14b_videos.py --config "$RUNS_CFG" --marks-config "$CFG" \
                    --picks "$picks" --gpu "$vg" --recog-port $(( 5850 + k )) --out "$VM" --shard "$k/$nsh" \
                    >> "$P14/driver/videos_s$k.log" 2>&1 &
                vpids+=($!)
                k=$(( k + 1 ))
            done
            say "GPU event: GPU $vg for the old-vs-new videos (2 shards)"
        done
        for p in "${vpids[@]}"; do
            while kill -0 "$p" 2>/dev/null; do
                (( SECONDS - vt0 > VIDEO_S )) && { kill_inside "p14b_videos.py"; stop "budget: videos > 1.5 x 1 h"; }
                pause 30
            done
            wait "$p" || vfail=1
        done
        kill_inside "mapmad_recog.server --model yoloe --port 585"
        servers_gone "mapmad_recog.server --model yoloe --port 585"
        if (( vfail )); then
            fallback "14b1.5: a video shard failed (driver/videos_s*.log): the gate lists the videos that exist"
        else
            run_bg dh "python mapmad/scripts/p13_videos.py check --out $VM" >> "$P14/driver/videos.log" 2>&1 \
                || fallback "14b1.5: video check reported a problem (videos_marks_fixed/video_check.json)"
        fi
    fi

    # --- 5. hashes after + G14b1 draft --------------------------------------------------------------------------------
    STEP="5 gate draft"
    status step="$STEP" gpus="" --state RUNNING
    run_bg bench hashes --tag after >> "$P14/driver/bench.log" 2>&1 || stop "hashing after failed (driver/bench.log)"
    run_bg bench gate >> "$P14/driver/bench.log" 2>&1 || stop "gate draft failed (driver/bench.log)"
    FINISHED=1
    status step="$STEP" note="draft G14b1: G14b1_table_draft.md; stopbench/stopbench.md; videos_marks_fixed/" eta="" \
        --state "CHAIN DONE"
    date '+%F %T' > "$P14/driver/CHAIN_DONE.txt"
    say "CHAIN DONE"
}

main "$@"
exit $?
