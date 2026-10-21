#!/bin/bash
# Phase 10 chained driver (kickoff 10.6, p10_rules.md). Host side, in screen p10_chain (start with launch_p10.sh, which
# also starts the watcher screen p10_chain_watch). Steps:
#   1. frontier cache (CPU, 20 workers; resumable, every drive re-checked against the teacher's logged counts) +
#      20-frame check (cache = shared function) + Habitat-side parity (naz_mapmad_habitat, CPU);
#   2. smoke: 1 tiny epoch (mapmad_v2_smoke.yaml) + smoke_v2 checks (tiny and full split) on the first GPU free for us;
#   3. timing: 200 steps (mapmad_v2_timing.yaml) -> p10_timing.py decision -> memory / speed fallbacks -> estimate;
#   4. main run seed 1 (launch overlay launch/main.yaml on mapmad_v2.yaml), one resume; ema_19 sha256;
#   5. offline checks + GoStanford no-harm on ema_19, pre-training report, draft G10 table, curves;
#   6. CHAIN DONE.
# Automatic stops (p10_rules.md §7) -> STOP_reason.txt (+ driver/STOP_reason.txt), STATUS.md STOPPED, fix nothing.
# EXIT trap: any other exit writes STOP_reason.txt with the exit code and the last log line. Long steps run in the
# background (wait), so TERM is handled within ~1 s. Refuses uncommitted code and a changed rules file. The body is
# one function, so editing this file cannot change a running driver.
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P10="$OUT_MOUNT/mapmad/p10_v2_train"
    C=/outputs/mapmad/p10_v2_train
    WHO=chain
    LOG="$P10/driver/chain.log"
    mkdir -p "$P10/driver" "$P10/launch"
    # shellcheck source=p10_lib.sh
    source "$REPO/train/scripts/mapmad_p10/p10_lib.sh"
    FINISHED=0
    BG_PID=""
    rm -f "$P10/STOP_reason.txt"

    stop() {  # reason -> automatic stop (no fix)
        { echo "time: $(date '+%F %T')"; echo "reason: $1"; echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P10/STOP_reason.txt"
        status step="${STEP:-?}" reason="$1" --state STOPPED
        say "AUTOMATIC STOP: $1"
        FINISHED=1  # a planned stop, not a crash of the driver
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "exit code: $code"; echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P10/STOP_reason.txt"
            status reason="driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    STEP="0 start"
    status step="$STEP" started="$(date '+%F %T')" --state RUNNING
    cd "$REPO" || exit 2
    if [[ -n "$(git status --porcelain -- train mapmad)" ]]; then stop "uncommitted code under train/ or mapmad/ (commit first, §0.8)"; fi
    COMMIT="$(git rev-parse HEAD)"
    want="$(cut -d' ' -f1 "$P10/p10_rules.sha256")"
    [[ "$(sha256sum "$P10/p10_rules.md" | cut -d' ' -f1)" == "$want" && "$(sha256sum mapmad/configs/p10_rules.md | cut -d' ' -f1)" == "$want" ]] \
        || stop "p10_rules.md differs from p10_rules.sha256 or from the repo copy"
    say "chain start: commit $COMMIT, rules sha256 ${want:0:16}"

    # --- 1. frontier cache --------------------------------------------------------------------------------------
    STEP="1 frontier cache"
    status step="$STEP"
    local t0=$SECONDS
    dx "" "python -m vint_train.mapmad.frontier_cache build --config config/mapmad_v2.yaml --workers 20 \
        --out-json $C/checks/frontier_cache_build.json" > "$P10/driver/frontier_cache.log" 2>&1 &
    BG_PID=$!
    local cache_dir=/mnt/shared_disk/nazli/mapmad_data/p10_frontier_cache
    while kill -0 "$BG_PID" 2>/dev/null; do
        (( SECONDS - t0 > 16200 )) && { kill "$BG_PID"; stop "frontier cache > 4.5 h"; }
        (( $(du -sb "$cache_dir" | cut -f1) > 90000000000 )) && { kill "$BG_PID"; stop "frontier cache > 90 GB"; }
        pause 30
    done
    wait "$BG_PID"; local code=$?
    (( code == 3 )) && stop "frontier-cache mismatch (build: a checked frame differs from the teacher's count)"
    (( code == 0 )) || stop "frontier cache build failed (exit $code, driver/frontier_cache.log)"
    say "frontier cache: $(( SECONDS - t0 )) s, $(du -sh "$cache_dir" | cut -f1); $(grep -a -E '"(drives|mismatched_frames|checked_frames)"' "$P10/driver/frontier_cache.log" | tr -d ' \n')"
    run_bg dx "" "python -m vint_train.mapmad.frontier_cache check --config config/mapmad_v2.yaml --frames 20 \
        --out-json $C/checks/frontier_cache_check.json" >> "$P10/driver/frontier_cache.log" 2>&1
    code=$?; (( code == 0 )) || stop "frontier-cache mismatch (check: cache != teacher_frontier_at, exit $code)"
    run_bg docker exec -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p10_frontier_parity.py \
        --frames-json "$C/checks/frontier_cache_check.json" --cache-dir /mapmad_data/p10_frontier_cache/habitat_mapmad_v2 \
        --out "$C/checks/frontier_parity.json" >> "$P10/driver/frontier_cache.log" 2>&1
    code=$?; (( code == 0 )) || stop "frontier-cache mismatch (Habitat-side parity, exit $code)"
    say "frontier check (20 frames) and Habitat parity: all equal"

    # --- 2. smoke ---------------------------------------------------------------------------------------------------
    STEP="2 smoke (waiting for a GPU free for 20 min)"
    status step="$STEP" --state WAITING
    G="$(wait_gpu)"
    nvidia-smi >> "$P10/driver/nvidia_smi.log" 2>&1
    say "GPU event: GPU $G taken by the chain (free for the chain per gpu_free.json + instant check)"
    STEP="2 smoke"
    status step="$STEP" gpu="$G" --state RUNNING
    run_bg dx "$G" "python train.py -c config/mapmad_v2_smoke.yaml" > "$P10/driver/smoke.log" 2>&1
    code=$?
    [[ $code -eq 0 ]] && tr '\r' '\n' < "$P10/driver/smoke.log" | grep -a -q "FINISHED TRAINING" \
        || stop "smoke fails (train.py exit $code, driver/smoke.log)"
    run_bg dx "" "python -m vint_train.mapmad.smoke_v2 --config config/mapmad_v2_smoke.yaml --out $C/checks/smoke --no-sheet && \
        python -m vint_train.mapmad.smoke_v2 --config config/mapmad_v2.yaml --out $C/checks/smoke" >> "$P10/driver/smoke.log" 2>&1
    code=$?; (( code == 0 )) || stop "smoke fails (smoke_v2 checks exit $code, checks/smoke/)"
    say "smoke passed (1 tiny epoch + checks; waypoint gap 0.25 m; mode shares within 2 points)"

    # --- 3. timing + fallbacks ------------------------------------------------------------------------------------------
    STEP="3 timing"
    status step="$STEP"
    local batch=256 accum=1 workers=12 raised="" k=0 dec
    while true; do
        k=$(( k + 1 ))
        overlay "$P10/launch/timing_$k.yaml" "$TRAIN_DIR/config/mapmad_v2_timing.yaml" "batch_size: $batch" \
            "grad_accum_steps: $accum" "num_workers: $workers" "run_name: timing_$k"
        ( while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G")"; sleep 5; done ) \
            > "$P10/driver/timing_${k}_mem.csv" &
        local memlog=$!
        run_bg dx "$G" "python train.py -c $C/launch/timing_$k.yaml" > "$P10/driver/timing_$k.log" 2>&1
        kill "$memlog" 2>/dev/null
        dec="$(python3 "$SCRIPTS/p10_timing.py" "$(ls -td "$P10"/timing_DRY/timing_${k}_2* 2>/dev/null | head -1)" \
            "$P10/driver/timing_${k}_mem.csv" "$P10/driver/timing_$k.log" --accum "$accum" $raised)"
        say "timing $k (batch $batch x accum $accum, workers $workers): $dec"
        case "$dec" in
            *'"decision": "ok"'*) break ;;
            *'"decision": "memory_fallback"'*)
                [[ $accum -ne 1 ]] && stop "out-of-memory after the accumulation fallback (timing $k)"
                batch=128; accum=2; say "memory fallback: batch 128 x grad_accum_steps 2 (deviation, logged)" ;;
            *'"decision": "more_workers"'*) workers=20; raised="--workers-raised"; say "loader > 15% of the step: num_workers 20" ;;
            *'"decision": "budget_stop"'*) stop "budget: estimate > 10.5 h ($dec)" ;;
            *) stop "timing failed (driver/timing_$k.log)" ;;
        esac
    done
    local est; est="$(echo "$dec" | python3 -c 'import json,sys; print(json.load(sys.stdin)["estimate_h"])')"
    overlay "$P10/launch/main.yaml" "$TRAIN_DIR/config/mapmad_v2.yaml" "batch_size: $batch" "grad_accum_steps: $accum" \
        "num_workers: $workers"
    status estimate_h="$est" note="main run setting: batch $batch x accum $accum, workers $workers; estimate $est h"
    say "main run setting: batch $batch x accum $accum, workers $workers; estimate $est h (launch/main.yaml)"

    # --- 4. main run ------------------------------------------------------------------------------------------------------
    STEP="4 main run (seed 1)"
    if ! gpu_clean "$G"; then
        say "GPU $G now holds another process: waiting for a GPU free for the chain"
        status step="$STEP (waiting for a GPU)" --state WAITING
        G="$(wait_gpu)"; say "GPU event: GPU $G taken by the chain"
    fi
    status step="$STEP" gpu="$G" --state RUNNING
    supervised_train v2_s1 "$G" "$C/launch/main.yaml" "$accum" || stop "$STOP_WHY"
    cp "$P10/launch/main.yaml" "$RUN_DIR/launch_main.yaml"
    mkdir -p "$P10/offline/main_ema19"
    sha256sum "$RUN_DIR/ema_19.pth" > "$P10/offline/main_ema19/ema_19.sha256"
    [[ -s "$P10/offline/main_ema19/ema_19.sha256" ]] && cp "$P10/offline/main_ema19/ema_19.sha256" "$RUN_DIR/ema_19.sha256" \
        || stop "checkpoint sha256 not written"
    say "GPU event: main run done on GPU $G; ema_19 sha256 $(cut -c1-16 "$P10/offline/main_ema19/ema_19.sha256")"

    # --- 5. offline checks ------------------------------------------------------------------------------------------------
    STEP="5 offline checks (waiting for a GPU)"
    status step="$STEP" offline_h=1.5 --state WAITING
    G="$(wait_gpu "$G")"
    STEP="5 offline checks"
    status step="$STEP" gpu="$G" --state RUNNING
    say "GPU event: GPU $G for the offline checks"
    offline_eval "$G" main "${RUN_DIR/$P10/$C}/ema_19.pth" || stop "offline checks failed (driver/offline_main.log)"
    python3 "$SCRIPTS/g10_table.py" "$P10" main >> "$LOG" 2>&1
    pretrain_report "$G" || stop "pre-training report failed (driver/offline_pretrain.log)"
    python3 "$SCRIPTS/g10_table.py" "$P10" main >> "$LOG" 2>&1
    run_bg dx "" "python -m vint_train.mapmad.plot_curves --run v2_s1=${RUN_DIR/$P10/$C} --out $C/curves" \
        >> "$P10/driver/curves.log" 2>&1 || say "curves failed (report only, driver/curves.log)"
    say "GPU event: chain released GPU $G"

    # --- 6. done ------------------------------------------------------------------------------------------------------------
    STEP="6 done"
    FINISHED=1
    status step="$STEP" note="draft G10 table: offline/main_ema19/g10_table_draft.md" --state "CHAIN DONE"
    date '+%F %T' > "$P10/driver/CHAIN_DONE.txt"
    say "CHAIN DONE"
}

main "$@"
exit $?
