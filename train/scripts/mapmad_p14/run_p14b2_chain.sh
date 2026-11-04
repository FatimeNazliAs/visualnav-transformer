#!/bin/bash
# Phase 14b-2 chained driver (kickoff .p14b2_kickoff.md, p14b2_rules.md). Host side, in screen p14b2_chain (start with
# launch_p14b2.sh). No exam: MP3D practice set + HM3D train homes (labelled, minus the photo-test homes) only. Steps:
#   0. pre-flight: committed code, rules sha256, training-home lists + contamination check (gate row 6), hashes of the
#      14a files (read only), the dev choices file;
#   1. R1 training frames: render every pool home's candidate poses (SHARDS_PER_GPU per GPU), select ~60k, YOLO dataset;
#   2. R2 fine-tune YOLOE: LP and FT in parallel (one GPU each), models file, Phase 9 photo test of each (raw, threshold,
#      timing, scores), refitted temperature per detector, the pick (photo/choice.json);
#   3. R3 replay of the 1,095 14a baseline runs with each fine-tuned detector (one sidecar per shard) -> new evidence;
#      stop bench per detector in the background during R4;
#   4. R4 driving: A-mark (base YOLOE), picked detector, both (W4 on) x M21b, M21bh, N on the practice set; the
#      determinism twins and the switch-off runs per GPU; fairness / determinism / switch-off checks;
#   5. R5 tables (photo, R3, R4, plain), 8 videos (picked detector + A-mark), hashes after, draft G14b2 -> CHAIN DONE.
# Automatic stops -> STOP_reason.txt, STATUS.md STOPPED, wait. A pool step whose worker dies is relaunched once
# (kickoff §2 fallback); a second death stops. Resumable: every step skips what is done.
# DRY (never results): P14_DIR=<host p14b2_dry>, P14_CFG=<container p14b2_dry.yaml>, P14_ALLOW_DIRTY=1, P14_GPUS="0 1",
# P14_EPISODES="<ids>", SHARDS_PER_GPU=1.
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P14="${P14_DIR:-$OUT_MOUNT/mapmad/p14b2_recog}"
    C="${P14/$OUT_MOUNT//outputs}"
    WHO=chain
    LOG="$P14/driver/chain.log"
    mkdir -p "$P14/driver" "$P14/checks"
    exec 2>> "$LOG"
    export P14_LOGNAME=phase14b2_log.md P14_PHASE=14b-2
    export P14_PASTE="continue Phase 14b-2: read STATUS.md and the chain log, then finish the tables, the G14b2 table and the summary, and close the gate if rows 6-8 pass"
    export P14_CFG="${P14_CFG:-/app/visualnav-transformer/mapmad/configs/p14b2_recog.yaml}"
    # shellcheck source=p14_lib.sh
    source "$REPO/train/scripts/mapmad_p14/p14_lib.sh"
    export P14_OUTPUTS="$OUT_MOUNT/mapmad" P14_EPISODES="${P14_EPISODES:-}"
    local S="${SHARDS_PER_GPU:-6}"
    local R1_S=$(( 6 * 3600 ))        # R1 <= 4 GPU-h; stop > 1.5x (GPU-time = wall x GPUs)
    local R2_S=$(( 12 * 3600 ))       # R2 <= 8 GPU-h
    local R3_S=$(( 6 * 3600 ))        # R3 <= 4 GPU-h
    local R4_S=$(( 9 * 3600 ))        # R4 <= 6 GPU-h
    FINISHED=0
    BG_PID=""
    B3_PID=""
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
        [[ -n "${B3_PID:-}" ]] && kill "$B3_PID" 2>/dev/null
        for pat in "p14b2_trainset.py" "mapmad_recog.finetune" "mapmad_recog.server --model yoloe" "p14b_replay.py" \
                   "p14b_bench" "p14b2_tables" "p9_photo_test.py" "p14_videos.py"; do
            kill_inside "$pat"; done  # our own processes only (patterns of this chain)
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "exit code: $code"
              echo "question: relaunch launch_p14b2.sh after reading driver/chain.log? (recommended: yes, it resumes)"
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P14/STOP_reason.txt"
            status reason="driver exit $code" error="driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    plan() { python3 "$SCRIPTS/p14_plan.py" "$HCFG" "$@"; }
    fallback() { status fallback="$1"; say "FALLBACK: $1"; }
    dxe() { dx "$1" "export P14_EPISODES='$P14_EPISODES' P14_OUTPUTS=/outputs/mapmad; ${*:2}"; }
    dh_g() { docker exec -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer naz_mapmad_habitat bash -lc "$*"; }
    drec() {  # gpu command... -> docker exec in naz_mapmad_recog, pinned + marked
        local g="$1"; shift
        docker exec -e CUDA_VISIBLE_DEVICES="$g" -e MAPMAD_P13_OWNER="$WHO" -e HOME=/model_cache/home \
            -e PYTHONPATH=/app/visualnav-transformer/mapmad/src -w /app/visualnav-transformer naz_mapmad_recog bash -lc "$*"
    }
    pool_retry() {  # label command... -> run a pool step; a worker death (BrokenProcessPool) -> relaunch once; else stop
        local label="$1"; shift
        "$@" && return 0
        if grep -q "BrokenProcessPool" "$P14/driver/bench.log" 2>/dev/null; then
            say "$label: a pool worker died -> relaunch once (kickoff §2 fallback)"
            "$@" && return 0
        fi
        stop "$label failed (driver/bench.log)"
    }
    servers_gone() {  # pattern -> waits (<= 90 s) until no matching process is left in naz_mapmad_recog
        local k
        for k in $(seq 1 18); do
            docker exec naz_mapmad_recog pgrep -f -- "$1" > /dev/null 2>&1 || return 0
            pause 5
        done
        say "note: '$1' still running 90 s after the kill"
    }
    take_gpus() {  # label -> echoes the usable GPUs (waits; one GPU only after 5 min of trying for two)
        local gs="" tries=0
        until [[ -n "$gs" ]]; do gs="$(usable_now)"; [[ -n "$gs" ]] || pause "$CHECK_S"; done
        while (( $(echo $gs | wc -w) == 1 && tries < 10 )); do
            pause 30; tries=$(( tries + 1 )); local g2; g2="$(usable_now)"; [[ -n "$g2" ]] && gs="$g2"
        done
        nvidia-smi >> "$P14/driver/nvidia_smi.log" 2>&1
        echo "$gs"
    }
    n_files() { compgen -G "$1" | wc -l; }  # glob -> number of matching files (progress of a step)
    watch_pids() {  # label budget pattern ngpu pid... -> waits; GPU-time = wall x GPUs; over budget -> stop;
        # PROGRESS_GLOB set: no new matching file for WATCHDOG_S (30 min) -> the step's processes are killed (return 2)
        local label="$1" budget="$2" pattern="$3" ngpu="$4"; shift 4
        local p alive t0=$SECONDS last_n=-1 last_t=$SECONDS n
        while :; do
            alive=0
            for p in "$@"; do kill -0 "$p" 2>/dev/null && alive=1; done
            (( alive )) || break
            USED_S=$(( USED_S + 60 * ngpu ))
            status note="$label: GPU-time used $(( USED_S / 60 )) min of $(( budget / 60 ))"
            (( USED_S > budget )) && { kill_inside "$pattern"; stop "budget: $label > 1.5 x its GPU-hours"; }
            if [[ -n "${PROGRESS_GLOB:-}" ]]; then
                n=$(n_files "$PROGRESS_GLOB")
                (( n != last_n )) && { last_n=$n; last_t=$SECONDS; }
                if (( SECONDS - last_t > ${WATCHDOG_S:-1800} )); then
                    say "WATCHDOG: $label made no progress for $(( (SECONDS - last_t) / 60 )) min -> killing it"
                    kill_inside "$pattern"
                    for p in "$@"; do wait "$p" 2>/dev/null; done
                    return 2
                fi
            fi
            pause 60
        done
        local bad=0
        for p in "$@"; do wait "$p" 2>/dev/null || bad=1; done
        say "$label: done in $(( (SECONDS - t0) / 60 )) min (GPU-time used $(( USED_S / 60 )) min)"
        return $bad
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
        want="$(cut -d' ' -f1 "$P14/p14b2_rules.sha256")"
        [[ "$(sha256sum "$P14/p14b2_rules.md" | cut -d' ' -f1)" == "$want" ]] || stop "p14b2_rules.md sha256 mismatch"
        [[ "$(sha256sum "$REPO/mapmad/configs/p14b2_rules.md" | cut -d' ' -f1)" == "$want" ]] || stop "repo copy of p14b2_rules.md differs"
    fi
    run_bg dh_g "python mapmad/scripts/p14b2_trainset.py homes --config $CFG" >> "$P14/driver/r1.log" 2>&1 \
        || stop "CONTAMINATION check failed or home lists incomplete (driver/r1.log, trainset/homes.json)" \
                "gate row 6 cannot pass: investigate the home lists? (recommended: yes)"
    [[ -f "$P14/P12_DEV_CHOICES.json" ]] || cp -p "$OUT_MOUNT/mapmad/p14a_finish/P12_DEV_CHOICES.json" "$P14/"
    [[ -f "$P14/checks/hashes_before.json" ]] || run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b_bench hashes --tag before --config $CFG" \
        >> "$P14/driver/bench.log" 2>&1 || stop "hashing the 14a runs failed (driver/bench.log)"
    say "pre-flight done: $(tail -n 1 "$P14/driver/r1.log")"

    # --- 1. R1 training frames --------------------------------------------------------------------------------------
    STEP="1 R1 training frames"
    status step="$STEP"
    USED_S=0
    local r1round
    for r1round in 1 2; do
        [[ -f "$P14/trainset/selection.json" ]] && break
        (( $(n_files "$P14/trainset/cand/*.json") >= 135 || r1round == 1 )) || say "R1: restarting the render once (watchdog)"
        local gs pids=() g s gi=0 code
        gs="$(take_gpus R1)"
        for g in $gs; do
            for (( s = 0; s < S; s++ )); do
                local k=$(( gi * S + s )) n=$(( $(echo $gs | wc -w) * S ))
                dh_g "python mapmad/scripts/p14b2_trainset.py render --config $CFG --gpu $g --shard $k/$n" \
                    >> "$P14/driver/r1_render_g${g}_s${s}.log" 2>&1 &
                pids+=($!)
            done
            say "GPU event: GPU $g taken by R1 render ($S shards)"
            gi=$(( gi + 1 ))
        done
        status --state RUNNING gpus="$gs"
        PROGRESS_GLOB="$P14/trainset/cand/*.json" watch_pids "R1 render" "$R1_S" "p14b2_trainset.py render" \
            "$(echo $gs | wc -w)" "${pids[@]}"
        code=$?
        (( code == 2 )) && { (( r1round == 2 )) && stop "R1 render hung twice (driver/r1_render_*.log)"; continue; }
        (( code == 0 )) || stop "R1 render: a shard failed (driver/r1_render_*.log)"
        run_bg dh_g "python mapmad/scripts/p14b2_trainset.py select --config $CFG" >> "$P14/driver/r1.log" 2>&1 \
            || stop "R1 select failed (driver/r1.log)"
    done
    say "R1 selection: $(python3 -c "import json; print(json.dumps(json.load(open('$P14/trainset/selection.json'))['stats'])[:400])")"
    python3 -c "import json,sys; sys.exit(0 if json.load(open('$P14/trainset/selection.json'))['stats']['fallback_fewer_than_20000'] else 1)" \
        && fallback "R1: fewer than 20,000 usable frames -> trained on what exists"
    [[ -f "$P14/yolo/prep.json" ]] || run_bg drec "" "python -m mapmad_recog.finetune prep --config $CFG" >> "$P14/driver/r2.log" 2>&1 \
        || stop "R1 YOLO dataset (prep) failed (driver/r2.log)"

    # --- 2. R2 fine-tune + photo test -------------------------------------------------------------------------------
    STEP="2 R2 fine-tune"
    status step="$STEP"
    USED_S=0
    local gs v g pids=() vs=(lp ft) i=0
    gs=($(take_gpus R2))
    pids=()
    for v in "${vs[@]}"; do
        [[ -f "$P14/train/$v/result.json" ]] && continue
        g="${gs[$(( i % ${#gs[@]} ))]}"; i=$(( i + 1 ))
        if (( ${#gs[@]} == 1 && ${#pids[@]} == 1 )); then wait "${pids[0]}"; fi  # one GPU: one after the other
        drec "$g" "python -m mapmad_recog.finetune train --config $CFG --variant $v" > "$P14/driver/train_$v.log" 2>&1 &
        pids+=($!)
        say "GPU event: GPU $g taken by R2 training $v"
    done
    (( ${#pids[@]} )) && { watch_pids "R2 training" "$R2_S" "mapmad_recog.finetune train" "${#gs[@]}" "${pids[@]}" || true; }
    [[ -f "$P14/train/ft/result.json" ]] || fallback "R2: FT failed (driver/train_ft.log) -> LP only"
    [[ -f "$P14/train/lp/result.json" ]] || stop "R2: LP training failed (driver/train_lp.log)"
    run_bg drec "" "python -m mapmad_recog.finetune models --config $CFG" >> "$P14/driver/r2.log" 2>&1 || stop "R2 models file failed"
    local key pkey ptest pt
    STEP="2 R2 photo test"
    status step="$STEP"
    pkey="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    for v in "${vs[@]}"; do
        [[ -f "$P14/train/$v/result.json" ]] || continue
        key="yoloe_$v"
        [[ -f "$P14/photo_test/$key/scores.json" ]] && continue
        g="${gs[0]}"
        docker exec -d -e MAPMAD_SIDECAR_KEY="$pkey" -e CUDA_VISIBLE_DEVICES="$g" -e HOME=/model_cache/home \
            -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog bash -c \
            "exec python -m mapmad_recog.server --model $key --port 5890 --config $C/models.yaml >> $C/driver/recog_photo_$key.log 2>&1"
        local E="-e MAPMAD_SIDECAR_KEY=$pkey -e MAPMAD_PHOTO_TEST_DIR=$C/photo_test -e MAPMAD_RECOG_MODELS=$C/models.yaml -e MAPMAD_P13_OWNER=$WHO"
        docker exec $E -w /app/visualnav-transformer naz_mapmad_habitat bash -lc \
            "python mapmad/scripts/p9_photo_test.py raw --model $key --port 5890 && python mapmad/scripts/p9_score.py tune --model $key" \
            >> "$P14/driver/photo_$key.log" 2>&1 || stop "R2 photo test raw / tune of $key failed (driver/photo_$key.log)"
        pt="$(python3 -c "import json; print(json.load(open('$P14/photo_test/$key/threshold.json'))['threshold'])")"
        docker exec $E -w /app/visualnav-transformer naz_mapmad_habitat bash -lc \
            "python mapmad/scripts/p9_photo_test.py timing --model $key --port 5890 --threshold $pt && python mapmad/scripts/p9_score.py score --model $key" \
            >> "$P14/driver/photo_$key.log" 2>&1 || stop "R2 photo test timing / score of $key failed (driver/photo_$key.log)"
        kill_inside "mapmad_recog.server --model $key --port 5890"
        servers_gone "mapmad_recog.server --model $key --port 5890"
        say "R2 photo test $key: $(tail -n 1 "$P14/driver/photo_$key.log")"
    done
    for v in "${vs[@]}"; do
        key="yoloe_$v"
        [[ -f "$P14/photo_test/$key/labels.json" && ! -f "$P14/calib_$key/calib.json" ]] || continue
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b_bench calib --config $CFG --det $key --labels ${C#/outputs/mapmad/}/photo_test/$key/labels.json" \
            >> "$P14/driver/bench.log" 2>&1 || stop "temperature fit for $key failed (driver/bench.log)"
    done
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b2_tables photo --config $CFG" >> "$P14/driver/tables.log" 2>&1 \
        || stop "R2 choice failed (driver/tables.log)"
    local PICK PICK_T PICK_P99 BASE_ONLY
    read -r PICK PICK_T PICK_P99 BASE_ONLY < <(python3 -c "
import json; c = json.load(open('$P14/photo/choice.json')); print(c['picked'], c['threshold'], c['p99'], int(c['fallback_base']))")
    say "R2 pick: $PICK (t $PICK_T, p99 $PICK_P99)"
    (( BASE_ONLY )) && fallback "R2: no fine-tuned detector beats base YOLOE on photo-test half A -> R4 runs the A-mark cell only"

    # --- 3. R3 replay with each fine-tuned detector -----------------------------------------------------------------
    STEP="3 R3 replay"
    status step="$STEP"
    USED_S=0
    local sub=baseline arms=(M21b M21bh N F VF) total
    total="$(python3 - "$OUT_MOUNT/mapmad/p14a_finish/runs/$sub" "$P14_EPISODES" "${arms[@]}" <<'EOF'
import os, sys
root, only, arms = sys.argv[1], set(sys.argv[2].split()), sys.argv[3:]
print(sum(1 for a in arms for f in os.listdir(os.path.join(root, "logs", a))
          if f.endswith(".jsonl") and (not only or f.rsplit("__s", 1)[0] in only)))
EOF
)"
    for v in "${vs[@]}"; do
        key="yoloe_$v"
        [[ -f "$P14/photo_test/$key/scores.json" ]] || continue
        local t p99 done_n
        read -r t p99 < <(python3 -c "import json; s=json.load(open('$P14/photo_test/$key/scores.json')); print(s['threshold'], s['p99_kept_half_a'])")
        done_n=$(ls "$P14/replay_$key/$sub"/*/*.json 2>/dev/null | wc -l)
        local r3round
        for r3round in 1 2; do
            (( done_n < total )) || break
            (( r3round == 1 )) || say "R3 replay $key: restarting once ($done_n/$total done)"
            gs=($(take_gpus R3))
            local rkey; rkey="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
            pids=()
            local gi=0
            for g in "${gs[@]}"; do
                for (( s = 0; s < S; s++ )); do
                    local port=$(( 5800 + 10 * g + s )) ep=() shard=$(( gi * S + s ))
                    [[ -n "$P14_EPISODES" ]] && ep=(--episodes $P14_EPISODES)
                    docker exec -d -e MAPMAD_SIDECAR_KEY="$rkey" -e CUDA_VISIBLE_DEVICES="$g" -e HOME=/model_cache/home \
                        -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog bash -c \
                        "exec python -m mapmad_recog.server --model $key --port $port --config $C/models.yaml >> $C/driver/recog_r3_${key}_g${g}_s${s}.log 2>&1"
                    docker exec -e MAPMAD_SIDECAR_KEY="$rkey" -e MAPMAD_P13_OWNER="$WHO" -e OMP_NUM_THREADS=2 \
                        -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p14b_replay.py \
                        --config "$CFG" --subdir "runs/$sub" --arms "${arms[@]}" --gpu "$g" --recog-port "$port" \
                        --shard "$shard/$(( ${#gs[@]} * S ))" "${ep[@]}" --detector "$key" \
                        --det-threshold "$t" --det-p99 "$p99" >> "$P14/driver/r3_${key}_g${g}_s${s}.log" 2>&1 &
                    pids+=($!)
                done
                say "GPU event: GPU $g taken by R3 replay $key ($S shards, ports $(( 5800 + 10 * g ))..)"
                gi=$(( gi + 1 ))
            done
            [[ ${#gs[@]} -eq 1 ]] && fallback "R3 $key: one GPU only (no exactness needed: new rows, not matched)"
            status --state RUNNING gpus="${gs[*]}"
            PROGRESS_GLOB="$P14/replay_$key/$sub/*/*.json" watch_pids "R3 replay $key" "$R3_S" "p14b_replay.py" \
                "${#gs[@]}" "${pids[@]}" || say "R3 replay $key: a shard exited non-zero or hung (driver/r3_${key}_*.log)"
            kill_inside "mapmad_recog.server --model $key --port 58"
            servers_gone "mapmad_recog.server --model $key --port 58"
            done_n=$(ls "$P14/replay_$key/$sub"/*/*.json 2>/dev/null | wc -l)
        done
        (( done_n >= total )) || stop "R3 replay $key: $done_n of $total runs after two rounds (driver/r3_${key}_*.log)"
        say "R3 replay $key: $done_n/$total runs"
    done
    local W="${BENCH_WORKERS:-6}"
    ( for v in "${vs[@]}"; do
          key="yoloe_$v"; [[ -f "$P14/photo_test/$key/scores.json" ]] || continue
          pool_retry "R3 bench $key" run_limit 14400 "p14b_bench score" dxe "" \
              "OMP_NUM_THREADS=1 python -m vint_train.mapmad.closed_loop.p14b_bench score --config $CFG --det $key --workers $W"
      done ) >> "$P14/driver/bench.log" 2>&1 &
    B3_PID=$!

    # --- 4. R4 driving ----------------------------------------------------------------------------------------------
    STEP="4 R4 driving"
    status step="$STEP"
    local R4CFG="$C/r4_config.yaml"
    printf 'extends: %s\nrecognisers:\n  picked: {model: %s, threshold: %s, p99: %s, vlm: false}\n' \
        "$CFG" "$PICK" "$PICK_T" "$PICK_P99" > "$P14/r4_config.yaml"
    export RECOG_CONFIG="$C/models.yaml"
    export PAIRS_PER_GPU="${R4_PAIRS:-4}"  # 14a drove 657 W4 runs in ~3.2 h on 3 pairs per GPU: 4 pairs to fit 1,971 runs
    (( BASE_ONLY )) && export P14B2_BASE_ONLY=1
    local budget=$R4_S label subdir set recog seeds arms_s n eps code
    while IFS='|' read -r label subdir set recog seeds arms_s n eps; do
        [[ -n "$label" ]] || continue
        STEP="4 R4: $label"
        status step="$STEP" --state RUNNING
        [[ "$recog" == picked ]] && recog="$PICK"
        say "$label: $n runs ($subdir, set $set, arms $arms_s, recogniser $recog)"
        CL_CFG="$R4CFG" CL_EP_PREFIX="${set}-" CL_EXTRA="${eps:+--episodes $eps}" closed_loop "$label" "$subdir" "$set" \
            "$n" "$budget" "$recog" "$seeds" $arms_s
        code=$?
        budget=$(( budget - CL_USED_S ))
        (( code == 1 )) && stop "budget: R4 $label exceeded 1.5 x 6 GPU-h"
        (( code == 2 )) && stop "$label: runners exit without progress (a possible bug; driver/${label}_gpu*.log)"
        (( code == 3 )) && stop "$label: recogniser server down after 3 restarts ($subdir/recog_restarts)"
        CL_EP_PREFIX="${set}-" crash_rate_ok "$subdir" "$n" $arms_s || stop "$label: crash rate > 1% ($subdir)"
    done < <(plan calls r4)
    STEP="4 R4 checks"
    status step="$STEP" gpus="" --state WAITING
    if [[ ! -f "$P14/checks/det_ok" ]]; then
        local dg
        dg="$(wait_gpu)"
        status gpus="$dg" --state RUNNING
        while IFS='|' read -r label subdir set recog seeds arms_s n eps; do
            [[ "$label" == det_* ]] || continue
            GPU="$dg" TOKEN="${label}_$(date +%m%d_%H%M%S)" OWNER="$WHO" PORT0=5560 RPORT0=5700 RECOG="$PICK" PAIRS_PER_GPU=1 \
                run_limit 3600 "run_search .*--subdir $subdir " bash "$REPO/mapmad/scripts/run_p13_pairs.sh" --config "$R4CFG" \
                --subdir "$subdir" --set "$set" --arms $arms_s --seeds 0 --episodes $eps >> "$P14/driver/${label}.log" 2>&1 \
                || stop "determinism runs $label failed (driver/${label}.log)"
        done < <(plan calls r4det)
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_checks det --stage r4det --config $R4CFG" \
            >> "$P14/driver/checks.log" 2>&1 || stop "determinism mismatch on the same GPU (checks/checks.json)" "investigate? (recommended: yes)"
        touch "$P14/checks/det_ok"
        say "determinism: $(tail -n 1 "$P14/driver/checks.log")"
    fi
    if [[ ! -f "$P14/checks/off.json" ]]; then
        local og
        for og in 0 1; do  # switch off = 14a byte for byte: each run vs the 14a run driven on the same GPU
            while IFS='|' read -r label subdir set recog seeds arms_s n eps; do
                [[ "$label" == off ]] || continue
                until gpu_clean "$og"; do pause 60; done
                GPU="$og" TOKEN="off_g${og}_$(date +%m%d_%H%M%S)" OWNER="$WHO" PORT0=5560 RPORT0=5700 RECOG=yoloe PAIRS_PER_GPU=1 \
                    RECOG_CONFIG="" run_limit 3600 "run_search .*--subdir checks/off_g$og " bash "$REPO/mapmad/scripts/run_p13_pairs.sh" \
                    --config "$R4CFG" --subdir "checks/off_g$og" --set "$set" --arms $arms_s --seeds 0 --episodes $eps \
                    >> "$P14/driver/off_g$og.log" 2>&1 || stop "switch-off runs on GPU $og failed (driver/off_g$og.log)"
            done < <(plan calls r4det)
        done
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b2_tables off --config $R4CFG" >> "$P14/driver/checks.log" 2>&1 \
            || stop "switch off differs from the 14a runs on the same GPU (checks/off.json)" "investigate? (recommended: yes)"
    fi
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_checks fairness --stages r4 --config $R4CFG" \
        >> "$P14/driver/checks.log" 2>&1 || stop "fairness check failed (checks/checks.json)" "investigate? (recommended: yes)"
    say "checks: $(tail -n 3 "$P14/driver/checks.log" | tr '\n' ' ' | cut -c1-400)"

    # --- 5. R5 tables, videos, gate draft ---------------------------------------------------------------------------
    STEP="5 R5 tables"
    status step="$STEP" gpus="" --state RUNNING
    wait "$B3_PID" || stop "R3 stop bench failed (driver/bench.log)"
    B3_PID=""
    for c in r3 r4 plain; do
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b2_tables $c --config $R4CFG --workers 12" \
            >> "$P14/driver/tables.log" 2>&1 || stop "table $c failed (driver/tables.log)"
    done
    STEP="5 R5 videos"
    status step="$STEP"
    if [[ ! -f "$P14/videos_detam/video_check.json" ]]; then
        local vg varm="M21b_w4_d_am" vsub="runs/detam" vkey vpids=() k=0 vfail=0
        (( BASE_ONLY )) && { varm="M21b_w4_am"; vsub="runs/am"; }
        vg="$(wait_gpu)"
        vkey="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
        for s in 0 1; do
            docker exec -d -e MAPMAD_SIDECAR_KEY="$vkey" -e CUDA_VISIBLE_DEVICES="$vg" -e HOME=/model_cache/home \
                -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog bash -c \
                "exec python -m mapmad_recog.server --model $PICK --port $(( 5850 + s )) --config $C/models.yaml >> $C/driver/recog_videos_$s.log 2>&1"
            docker exec -e MAPMAD_SIDECAR_KEY="$vkey" -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer naz_mapmad_habitat \
                python mapmad/scripts/p14_videos.py render --zoom --config "$R4CFG" --subdir "$vsub" --arm "$varm" \
                --picks "${P14B_VIDEO_PICKS:-/outputs/mapmad/p14a_finish/videos_marks/picks.json}" --gpu "$vg" \
                --recog-port $(( 5850 + s )) --out "$C/videos_detam" --shard "$s/2" >> "$P14/driver/videos_s$s.log" 2>&1 &
            vpids+=($!)
        done
        say "GPU event: GPU $vg for the 8 videos ($varm, 2 shards)"
        for p in "${vpids[@]}"; do wait "$p" || vfail=1; done
        kill_inside "mapmad_recog.server --model $PICK --port 585"
        servers_gone "mapmad_recog.server --model $PICK --port 585"
        if (( vfail )); then
            fallback "R5: a video shard failed (driver/videos_s*.log): the gate lists the videos that exist"
        else
            run_bg dh_g "python mapmad/scripts/p13_videos.py check --out $C/videos_detam" >> "$P14/driver/videos.log" 2>&1 \
                || fallback "R5: video check reported a problem (videos_detam/video_check.json)"
        fi
    fi
    STEP="5 gate draft"
    status step="$STEP"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b_bench hashes --tag after --config $CFG" >> "$P14/driver/bench.log" 2>&1 \
        || stop "hashing after failed (driver/bench.log)"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b2_tables gate --config $R4CFG" >> "$P14/driver/tables.log" 2>&1 \
        || stop "gate draft failed (driver/tables.log)"
    FINISHED=1
    status step="$STEP" note="draft G14b2: G14b2_table_draft.md; tables/plain.md, photo.md, r3.md, r4.md; videos_detam/" eta="" \
        --state "CHAIN DONE"
    date '+%F %T' > "$P14/driver/CHAIN_DONE.txt"
    say "CHAIN DONE"
}

main "$@"
exit $?
