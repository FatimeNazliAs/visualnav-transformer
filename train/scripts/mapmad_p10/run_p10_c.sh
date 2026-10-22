#!/bin/bash
# Phase 10 run C driver (p10_rules_v4.md). Host side, in screen p10_c (start: launch_p10_c.sh). Steps:
#   1. wait for the first GPU free for us (gpu_watch.py; Phase 9 / other users = busy; never interrupts Phase 9);
#   2. run C: mapmad_v2_seen.yaml (seed 1, seen-before-out-of-view oversampling 0.25) with run 1's launch setting, one
#      resume; ema_19 sha256;
#   3. offline checks on ema_19 (run-1 harness -> offline/c_ema19), seen list, guard vs run 1, G10 rows 1-2 (CPU),
#      draft G10 table, curves, p10_c_decide.py -> verdict C_IS_V2 | RUN1_IS_V2 | STOP (STOP = automatic stop for Naz);
#   4. twin (report only): seed 2 of the recipe that became v2, after Phase 9 no longer needs a GPU, same checks.
# `run_p10_c.sh resume <run C folder>` (p10_rules_v5.md): instead of step 2, resume run C ONCE from its latest.pth on
# the first free GPU, logging free RAM and swap at the start and after every epoch; if it does not finish -> no stop:
# verdict RUN1_IS_V2 (crash), the run-1 recipe's twin follows.
# `run_p10_c.sh rerun` (p10_rules_v6.md): instead of step 2, train run C again from nomad.pth in a new folder; a crash
# before 2 complete epochs -> stop (health check); after that one resume, a second crash -> stop; writes
# driver/RUN_C_PAST_EPOCH2.txt after epoch 2; RAM / swap logged per epoch.
# EXIT trap as run_p10_chain.sh. Refuses uncommitted code and changed rules files. Body in one function.
set -uo pipefail

main() {
    local mode="${1:-new}" resume_dir="${2:-}"
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P10="${P10_DRY_DIR:-$OUT_MOUNT/mapmad/p10_v2_train}"  # P10_DRY_* = DRY replay only (p10_scratch/c_DRY)
    C="${P10_DRY_C:-/outputs/mapmad/p10_v2_train}"
    P9_STATUS="${P9_STATUS:-$OUT_MOUNT/mapmad/p9_recognisers/STATUS.md}"
    WHO=chain
    LOG="$P10/driver/c.log"
    mkdir -p "$P10/driver" "$P10/launch" "$P10/offline/c_ema19"
    exec 2>> "$LOG"
    # shellcheck source=p10_lib.sh
    source "$REPO/train/scripts/mapmad_p10/p10_lib.sh"
    [[ -n "${P10_DRY_STUBS:-}" ]] && source "$P10_DRY_STUBS"  # DRY replay: stubbed dx / training / GPU waits
    FINISHED=0
    BG_PID=""
    rm -f "$P10/STOP_reason.txt"

    stop() {
        { echo "time: $(date '+%F %T')"; echo "reason: $1"; echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P10/STOP_reason.txt"
        status step="${STEP:-?}" reason="$1" --state STOPPED
        say "AUTOMATIC STOP: $1"
        FINISHED=1
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        kill $(jobs -p) 2>/dev/null  # background watchers (RAM log, epoch-2 marker)
        for pat in "launch/c.yaml" "c_ema19" "launch/twin_c.yaml" "twin_ema19" "checks_g10" "target_groups"; do
            kill_inside "$pat"; done
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "exit code: $code"; echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P10/STOP_reason.txt"
            status reason="run C driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    STEP="C0 start"
    status step="$STEP" started="$(date '+%F %T')" main_run="" offline_h=0.5 --state RUNNING
    cd "$REPO" || exit 2
    if [[ -n "$(git status --porcelain -- train mapmad)" ]]; then stop "uncommitted code under train/ or mapmad/"; fi
    local want; want="$(cut -d' ' -f1 "$P10/p10_rules_v4.sha256")"
    [[ "$(sha256sum "$P10/p10_rules_v4.md" | cut -d' ' -f1)" == "$want" && "$(sha256sum mapmad/configs/p10_rules_v4.md | cut -d' ' -f1)" == "$want" ]] \
        || stop "p10_rules_v4.md differs from p10_rules_v4.sha256 or from the repo copy"
    say "run C driver start ($mode): commit $(git rev-parse HEAD), rules v4 sha256 ${want:0:16}"
    if [[ "$mode" == resume ]]; then
        local want5; want5="$(cut -d' ' -f1 "$P10/p10_rules_v5.sha256")"
        [[ "$(sha256sum "$P10/p10_rules_v5.md" | cut -d' ' -f1)" == "$want5" && "$(sha256sum mapmad/configs/p10_rules_v5.md | cut -d' ' -f1)" == "$want5" ]] \
            || stop "p10_rules_v5.md differs from p10_rules_v5.sha256 or from the repo copy"
        [[ -f "$resume_dir/latest.pth" ]] || stop "resume: $resume_dir/latest.pth missing"
        say "resume mode: $resume_dir (rules v5 sha256 ${want5:0:16})"
    fi
    if [[ "$mode" == rerun ]]; then
        local want6; want6="$(cut -d' ' -f1 "$P10/p10_rules_v6.sha256")"
        [[ "$(sha256sum "$P10/p10_rules_v6.md" | cut -d' ' -f1)" == "$want6" && "$(sha256sum mapmad/configs/p10_rules_v6.md | cut -d' ' -f1)" == "$want6" ]] \
            || stop "p10_rules_v6.md differs from p10_rules_v6.sha256 or from the repo copy"
        [[ -f "$P10/launch/c.yaml" ]] || stop "rerun: launch/c.yaml missing"
        rm -f "$P10/driver/RUN_C_PAST_EPOCH2.txt"
        say "rerun mode (rules v6 sha256 ${want6:0:16})"
    fi
    local batch accum workers
    batch="$(grep '^batch_size:' "$P10/launch/main.yaml" | awk '{print $2}')"
    accum="$(grep '^grad_accum_steps:' "$P10/launch/main.yaml" | awk '{print $2}')"
    workers="$(grep '^num_workers:' "$P10/launch/main.yaml" | awk '{print $2}')"
    [[ "$mode" != new ]] || overlay "$P10/launch/c.yaml" "$TRAIN_DIR/config/mapmad_v2_seen.yaml" \
        "batch_size: $batch" "grad_accum_steps: $accum" "num_workers: $workers"

    epochs_done() { local n; n="$(grep -c '"train_seconds"' "$1/metrics.jsonl" 2>/dev/null)"; echo "${n:-0}"; }
    past_epoch2_watch() {  # old newest run -> waits for the new run folder; RAM per epoch; marker after epoch 2
        local old="$1" d="" marked=0
        while true; do
            d="$(newest_run v2c_s1)"
            if [[ -n "$d" && "$d" != "$old" ]]; then break; fi
            pause 30
        done
        status main_run="$d"
        mem_watch "$d" &
        local mw=$!
        trap 'kill $mw 2>/dev/null; exit 0' TERM INT
        while (( marked == 0 )); do
            if (( $(epochs_done "$d") >= 2 )); then
                date '+%F %T' > "$P10/driver/RUN_C_PAST_EPOCH2.txt"
                say "run C rerun is past epoch 2 (health check passed): Phase 9 may run its held checks 3-4 on the other GPU"
                status note="rerun past epoch 2: health check passed"
                marked=1
            fi
            pause 30
        done
        wait "$mw"
    }
    rerun_train() {  # gpu -> RUN_DIR; rules v6 §3 (health check, one resume after epoch 2); STOP_WHY on failure
        local g="$1" old log d n
        old="$(newest_run v2c_s1)"
        log="$P10/driver/train_v2c_s1_rerun.log"
        STOP_WHY=""
        mem_log start
        past_epoch2_watch "$old" &
        local pw=$!
        say "train v2c_s1 rerun on GPU $g: python train.py -c $C/launch/c.yaml (from nomad.pth, new folder)"
        train_once "$g" "$C/launch/c.yaml" "$log"
        d="$(newest_run v2c_s1)"
        [[ "$d" == "$old" ]] && d=""
        RUN_DIR="$d"
        if finished_ok "$log" "$d"; then kill "$pw" 2>/dev/null; mem_log end; say "v2c_s1 rerun finished"; return 0; fi
        n="$( [[ -n "$d" ]] && epochs_done "$d" || echo 0)"
        mem_log "at crash (epochs done $n)"
        if grep -a -q "non-finite loss" "$log"; then kill "$pw" 2>/dev/null; STOP_WHY="NaN loss (rerun, $log)"; return 1; fi
        if (( n < 2 )); then
            kill "$pw" 2>/dev/null
            STOP_WHY="health check: run C rerun crashed before 2 complete epochs ($n done; $log) -- server problem; v2 = run 1 ema_19 stays"
            return 1
        fi
        say "CRASH 1 of the rerun after $n epochs ($log, last line: $(tr '\r' '\n' < "$log" | tail -n 1 | cut -c1-150)) -> resume once"
        if ! gpu_clean "$g"; then say "GPU $g now holds another process: waiting for a free GPU to resume"; g="$(wait_gpu)"; fi
        status gpu="$g" note="rerun resume 1"
        train_once "$g" "$C/launch/c.yaml" "${log%.log}_resume1.log" "--resume ${d/$P10/$C}/latest.pth"
        kill "$pw" 2>/dev/null
        mem_log end
        if finished_ok "${log%.log}_resume1.log" "$d"; then say "v2c_s1 rerun finished after resume 1"; return 0; fi
        STOP_WHY="second crash of the rerun (${log%.log}_resume1.log)"; return 1
    }

    # --- 1-2. run C -----------------------------------------------------------------------------------------------
    STEP="C1 run C (waiting for a GPU free for 20 min; Phase 9 first)"
    status step="$STEP" --state WAITING
    G="$(wait_gpu)"
    nvidia-smi >> "$P10/driver/nvidia_smi.log" 2>&1
    say "GPU event: GPU $G taken by run C (free for the chain per gpu_free.json + instant check)"
    STEP="C2 run C (seed 1, seen oversampling 0.25)"
    status step="$STEP" gpu="$G" --state RUNNING
    local cdir crashed=0
    if [[ "$mode" == resume ]]; then
        cdir="$resume_dir"
        STEP="C2 run C resume 2 (rules v5, from latest.pth)"
        status step="$STEP" main_run="$cdir" note="resume 2 (rules v5)"
        mem_log start
        mem_watch "$cdir" &
        local memw=$!
        local rlog="$P10/driver/train_v2c_s1_resume2.log"
        say "train v2c_s1 resume 2 on GPU $G: python train.py -c $C/launch/c.yaml --resume ${cdir/$P10/$C}/latest.pth"
        train_once "$G" "$C/launch/c.yaml" "$rlog" "--resume ${cdir/$P10/$C}/latest.pth"
        kill "$memw" 2>/dev/null
        mem_log end
        if ! finished_ok "$rlog" "$cdir"; then
            crashed=1
            say "CRASH in resume 2 ($rlog, last line: $(tr '\r' '\n' < "$rlog" | tail -n 1 | cut -c1-150)) -> rules v5: v2 = run 1 ema_19 (limitation), run-1 twin"
        else
            say "v2c_s1 finished after resume 2"
        fi
    elif [[ "$mode" == rerun ]]; then
        rerun_train "$G" || stop "$STOP_WHY"
        cdir="$RUN_DIR"
    else
        supervised_train v2c_s1 "$G" "$C/launch/c.yaml" "$accum" || stop "$STOP_WHY"
        cdir="$RUN_DIR"
    fi
    local v
    if [[ $crashed -eq 0 ]]; then
    cp "$P10/launch/c.yaml" "$cdir/launch_c.yaml"
    sha256sum "$cdir/ema_19.pth" > "$P10/offline/c_ema19/ema_19.sha256"
    [[ -s "$P10/offline/c_ema19/ema_19.sha256" ]] && cp "$P10/offline/c_ema19/ema_19.sha256" "$cdir/ema_19.sha256" \
        || stop "checkpoint sha256 not written"
    say "GPU event: run C done on GPU $G; ema_19 sha256 $(cut -c1-16 "$P10/offline/c_ema19/ema_19.sha256")"

    # --- 3. checks + decision ---------------------------------------------------------------------------------------
    STEP="C3 checks (waiting for a GPU)"
    status step="$STEP" --state WAITING
    G="$(wait_gpu "$G")"
    STEP="C3 checks"
    status step="$STEP" gpu="$G" --state RUNNING
    say "GPU event: GPU $G for run C's checks"
    local ck="${cdir/$P10/$C}/ema_19.pth" o="$C/offline/c_ema19"
    offline_eval "$G" c "$ck" || stop "offline checks failed (driver/offline_c.log)"
    run_limit 7200 "c_ema19/seen" dx "$G" "python -m vint_train.mapmad.offline_eval score --config config/mapmad_v2.yaml \
        --checkpoint $ck --list $C/offline/lists/seen_list.json --arms maps,target_zeroed,target_shifted --out $o/seen && \
        python -m vint_train.mapmad.target_groups seen-report --scores $o/seen/scores.npz --out $o/seen" \
        >> "$P10/driver/offline_c.log" 2>&1 || stop "seen-list scoring failed (driver/offline_c.log)"
    say "GPU event: run C checks released GPU $G"
    run_bg dx "" "python -m vint_train.mapmad.target_groups guard --scores $o/search/scores.npz \
        --reference $C/offline/main_ema19/search/scores.npz --out $o/guard.json && \
        python -m vint_train.mapmad.checks_g10 --config config/mapmad_v2.yaml --v1-config config/mapmad_p4_C.yaml \
        --reference $C/checks/reference_forward_cpu.pt --out $o/g10_rows_1_2_cpu.json" >> "$P10/driver/offline_c.log" 2>&1 \
        || stop "guard / G10 rows 1-2 failed to run (driver/offline_c.log)"
    python3 "$SCRIPTS/g10_table.py" "$P10" c >> "$LOG" 2>&1
    run_bg dx "" "python -m vint_train.mapmad.plot_curves --run v2c_s1=${cdir/$P10/$C} --out $C/curves" \
        >> "$P10/driver/curves.log" 2>&1 || say "curves failed (report only)"
    v="$(python3 "$SCRIPTS/p10_c_decide.py" "$P10" | tee -a "$LOG" | tail -n 1)"
    say "run C verdict: $v (offline/c_ema19/decision.md)"
    else
        v=RUN1_IS_V2_CRASH
        say "GPU event: run C released GPU $G (crashed); run C verdict: $v (rules v5)"
    fi
    case "$v" in
        C_IS_V2) echo "v2 = run C ema_19 ${cdir}/ema_19.pth" > "$P10/V2_CHOICE.txt"; local twin_cfg=mapmad_v2_seen_twin.yaml twin_label=v2c_s2 ;;
        RUN1_IS_V2) echo "v2 = run 1 ema_19 (run C failed the seen-list rule; limitation)" > "$P10/V2_CHOICE.txt"
                    local twin_cfg=mapmad_v2_twin.yaml twin_label=v2_s2 ;;
        RUN1_IS_V2_CRASH) echo "v2 = run 1 ema_19 (run C crashed again in resume 2, rules v5; limitation: no result for run C)" > "$P10/V2_CHOICE.txt"
                    local twin_cfg=mapmad_v2_twin.yaml twin_label=v2_s2 ;;
        *) stop "run C verdict STOP: row 4, the guard or G10 failed (offline/c_ema19/decision.md) -- waiting for Naz" ;;
    esac
    status step="C3 verdict $v; twin next" note="$(cat "$P10/V2_CHOICE.txt")"

    # --- 4. twin (report only) -----------------------------------------------------------------------------------------
    phase9_done() { grep -q -i -E "AT GATE|STOPPED|photo test done|photo test: done|PHASE 9 DONE" "$P9_STATUS" 2>/dev/null; }
    STEP="C4 twin $twin_label (waiting: Phase 9 done with GPUs, then a free GPU)"
    status step="$STEP" --state WAITING
    while ! phase9_done; do pause "$CHECK_S"; done
    G="$(wait_gpu)"
    overlay "$P10/launch/twin_c.yaml" "$TRAIN_DIR/config/$twin_cfg" "batch_size: $batch" "grad_accum_steps: $accum" \
        "num_workers: $workers"
    nvidia-smi >> "$P10/driver/nvidia_smi.log" 2>&1
    say "GPU event: GPU $G taken by the twin ($twin_label, report only)"
    STEP="C4 twin $twin_label"
    status step="$STEP" gpu="$G" --state RUNNING
    supervised_train "$twin_label" "$G" "$C/launch/twin_c.yaml" "$accum" || stop "twin: $STOP_WHY (report only; v2 choice stands)"
    sha256sum "$RUN_DIR/ema_19.pth" > "$P10/offline/twin_ema19.sha256"
    G="$(wait_gpu "$G")"
    local tk="${RUN_DIR/$P10/$C}/ema_19.pth" to="$C/offline/twin_ema19"
    offline_eval "$G" twin "$tk" || stop "twin offline checks failed (report only)"
    mkdir -p "$P10/offline/twin_ema19"
    cp "$P10/offline/twin_ema19.sha256" "$P10/offline/twin_ema19/ema_19.sha256"
    run_limit 7200 "twin_ema19/seen" dx "$G" "python -m vint_train.mapmad.offline_eval score --config config/mapmad_v2.yaml \
        --checkpoint $tk --list $C/offline/lists/seen_list.json --arms maps,target_zeroed,target_shifted --out $to/seen && \
        python -m vint_train.mapmad.target_groups seen-report --scores $to/seen/scores.npz --out $to/seen" \
        >> "$P10/driver/offline_twin.log" 2>&1 || say "twin seen list failed (report only)"
    python3 "$SCRIPTS/g10_table.py" "$P10" twin >> "$LOG" 2>&1
    say "GPU event: twin released GPU $G; twin done (report only)"

    FINISHED=1
    status step="C done: verdict $v, twin $twin_label done" --state "CHAIN DONE"
    date '+%F %T' > "$P10/driver/C_DONE.txt"
    say "RUN C CHAIN DONE"
}

main "$@"
exit $?
