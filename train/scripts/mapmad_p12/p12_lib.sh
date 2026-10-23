#!/bin/bash
# Shared functions of the Phase 12 chain (run_p12_chain.sh). Sourced, never run. The caller sets: REPO (worktree), P12
# (host p12_search_exam folder), C (the same as the containers see it), WHO (chain), LOG (its log file). Every GPU job is
# pinned (CUDA_VISIBLE_DEVICES / --gpu) and marked MAPMAD_P12_OWNER=$WHO, so the GPU watcher (mapmad_p10/gpu_watch.py with
# MAPMAD_OWNER_ENV=MAPMAD_P12_OWNER, RECENT_S=3600) tells our processes apart from everyone else's (§8 rules 1-11).

TRAIN_DIR=/app/visualnav-transformer/train
CFG="${P12_CFG:-/app/visualnav-transformer/mapmad/configs/p12_exam.yaml}"
HCFG="$REPO/mapmad/configs/${CFG##*/}"   # the same config, host path (p12_plan.py)
SCRIPTS="$REPO/train/scripts/mapmad_p12"
WATCH="$REPO/train/scripts/mapmad_p10/gpu_watch.py"
CHECK_S="${CHECK_S:-300}"
export MAPMAD_OWNER_ENV=MAPMAD_P12_OWNER
export P12_OUTPUTS="${P12_OUTPUTS:-${OUT_MOUNT}/mapmad}"

say() {  # message -> own log + phase12_log.md
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
    echo "- $(date '+%m-%d %H:%M') [$WHO] $*" >> "$P12/phase12_log.md"
}

status() {  # key=value ... [--state S] -> driver/chain_state.json + STATUS.md
    python3 "$SCRIPTS/p12_status.py" "$P12" "$WHO" --set "$@" >> "$LOG" 2>&1
}

pause() { sleep "$1" & wait $!; }  # a sleep that TERM can interrupt at once

free_for() {  # -> GPUs free for us (20-min rule + rule 11: gpu_watch.py's gpu_free.json; DRY runs: P12_GPU_FREE)
    if [[ -n "${P12_GPUS:-}" ]]; then echo "$P12_GPUS"; return; fi  # DRY runs only: fixed GPUs
    python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['free'][sys.argv[2]]))" \
        "${P12_GPU_FREE:-$P12/driver/gpu_free.json}" "$WHO" 2>/dev/null
}

gpu_clean() {  # live nvidia-smi right now: nobody else on GPU $1 (§8 rule 9)
    python3 "$WATCH" --check "$1" "$WHO" >> "$LOG" 2>&1
}

wait_gpu() {  # [preferred gpu] -> echoes a GPU free for 20 min AND clean right now; waits in CHECK_S
    local prefer="${1:-}" g list
    while true; do
        list="$(free_for)"
        for g in $prefer $list; do
            if [[ " $list " == *" $g "* ]] && gpu_clean "$g"; then echo "$g"; return 0; fi
        done
        pause "$CHECK_S"
    done
}

dx() {  # gpu ("" = CPU) command... -> docker exec in naz_mapmad, pinned + marked, from train/
    local g="$1"; shift
    docker exec -e CUDA_VISIBLE_DEVICES="$g" -e MAPMAD_P12_OWNER="$WHO" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e MPLCONFIGDIR=/tmp/mpl -e P12_OUTPUTS=/outputs/mapmad \
        -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad bash -lc "cd $TRAIN_DIR && $*"
}

dh() {  # command... -> docker exec in naz_mapmad_habitat (CPU unless the command names a GPU), marked
    docker exec -e MAPMAD_P12_OWNER="$WHO" -w /app/visualnav-transformer naz_mapmad_habitat bash -lc "$*"
}

run_bg() {  # command... (stdout/stderr already redirected by the caller) -> waits, TERM-safe; returns its code
    "$@" &
    BG_PID=$!
    wait "$BG_PID"
}

kill_inside() {  # pattern -> kill the matching processes INSIDE our three containers (killing docker exec does not)
    docker exec naz_mapmad pkill -f -- "$1" 2>/dev/null
    docker exec naz_mapmad_habitat pkill -f -- "$1" 2>/dev/null
    docker exec naz_mapmad_recog pkill -f -- "$1" 2>/dev/null
    return 0
}

run_limit() {  # seconds pattern command... -> like run_bg, but killed (inside the containers too) after `seconds`
    local limit="$1" pattern="$2" t0=$SECONDS; shift 2
    "$@" &
    BG_PID=$!
    while kill -0 "$BG_PID" 2>/dev/null; do
        if (( SECONDS - t0 > limit )); then
            say "TIME LIMIT: '$pattern' ran > $limit s -> killed"
            kill_inside "$pattern"; kill "$BG_PID" 2>/dev/null; wait "$BG_PID" 2>/dev/null
            return 124
        fi
        pause 20
    done
    wait "$BG_PID"
}

recog_restarts() {  # subdir -> the largest number of restarts of any one recogniser server of this step
    local d="$P12/$1/recog_restarts" m=0 n f
    [[ -d "$d" ]] || { echo 0; return; }
    for f in "$d"/*; do [[ -f "$f" ]] || continue; n=$(wc -l < "$f"); (( n > m )) && m=$n; done
    echo "$m"
}

newest_progress_age() {  # subdir -> seconds since the newest progress / server log line of the step (silent-log watchdog)
    python3 - "$P12/$1" <<'EOF'
import glob, os, sys, time
files = glob.glob(os.path.join(sys.argv[1], "progress.*.log")) + glob.glob(os.path.join(sys.argv[1], "logs", "*", "*.partial"))
print(int(time.time() - max((os.path.getmtime(f) for f in files), default=time.time())))
EOF
}

# --- closed-loop runs on a shared queue (block item 9, §8 rules 2-5, 9, 11) -------------------------------------------
closed_loop() {  # label subdir set total budget_s recog seeds arms... -> 0 all done, 1 budget, 2 no progress, 3 recogniser
    # Pairs (PAIRS_PER_GPU, default 3) on every GPU free for us (20-min rule + rule 11 + live check right before), all on
    # one run queue; a GPU that gets another user's process is withdrawn (STOP file: its runners finish their current run,
    # take no more); a GPU free later is added to the same queue. When every runner has exited and runs are left, a new
    # launch token restarts the queue. Budget = GPU-running time scaled by the share of the two GPUs in use (waiting is
    # not counted; CL_USED_S returns it). A recogniser restarted > 3 times -> 3; no new run for SILENT_S -> the runners
    # are killed (a hang, logged) and the round restarts.
    local label="$1" subdir="$2" set="$3" total="$4" budget="$5" recog="$6" seeds="$7"; shift 7
    local arms=("$@") round=0 used=0 t_last token g done crashed pids_alive progress_at_round=0 launched=0 seed_args=()
    [[ -n "$seeds" ]] && seed_args=(--seeds $seeds)
    declare -A PIDS=() ENDED=()
    while true; do
        read -r done crashed < <(python3 "$SCRIPTS/p12_status.py" "$P12" count "$subdir" "${arms[@]}")
        if (( done + crashed >= total )); then
            say "$label: all $total runs done ($crashed crashed); used $(( used / 60 )) min of GPU-running time"
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            CL_USED_S=$used
            return 0
        fi
        if (( $(recog_restarts "$subdir") > 3 )); then
            say "$label: a recogniser sidecar was restarted more than 3 times ($subdir/recog_restarts)"
            for g in "${!PIDS[@]}"; do touch "$P12/driver/STOP_gpu${g}_$token"; done
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            CL_USED_S=$used
            return 3
        fi
        pids_alive=0
        for g in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$g]}" 2>/dev/null; then
                pids_alive=1
                if [[ ! -f "$P12/driver/STOP_gpu${g}_$token" ]] && ! gpu_clean "$g"; then
                    touch "$P12/driver/STOP_gpu${g}_$token"
                    nvidia-smi >> "$P12/driver/nvidia_smi.log" 2>&1
                    say "GPU event: another user's process on GPU $g -> withdrawing $label pairs there (current runs finish)"
                fi
            else
                wait "${PIDS[$g]}" 2>/dev/null
                unset "PIDS[$g]"
                ENDED[$g]="$token"
            fi
        done
        if (( pids_alive == 1 )) && (( $(newest_progress_age "$subdir") > ${SILENT_S:-1800} )); then
            say "HANG: no run finished and no log written in $subdir for > ${SILENT_S:-1800} s -> killing this step's runners"
            kill_inside "run_search .*--subdir $subdir"
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            pids_alive=0
        fi
        if (( pids_alive == 0 )); then
            if (( round > 0 && launched > 0 && done + crashed == progress_at_round )); then
                say "$label: every runner of round $round exited without a new run (a possible bug; driver/${label}_gpu*.log)"
                CL_USED_S=$used
                return 2
            fi
            round=$(( round + 1 )); token="${label}_$(date +%m%d_%H%M%S)_r$round"
            progress_at_round=$(( done + crashed )); launched=0
        fi
        for g in $(free_for); do
            [[ -n "${PIDS[$g]:-}" || "${ENDED[$g]:-}" == "$token" ]] && continue
            [[ -f "$P12/driver/STOP_gpu${g}_$token" ]] && continue
            gpu_clean "$g" || continue
            nvidia-smi >> "$P12/driver/nvidia_smi.log" 2>&1
            say "GPU event: GPU $g taken by $label (free 20 min, rule 11, live check); ${PAIRS_PER_GPU:-3} pairs, recogniser $recog, token $token"
            GPU="$g" TOKEN="$token" OWNER="$WHO" STOP_FILE="$C/driver/STOP_gpu${g}_$token" PORT0="${CL_PORT0:-5560}" \
                RPORT0="${CL_RPORT0:-5700}" RECOG="$recog" PAIRS_PER_GPU="${PAIRS_PER_GPU:-3}" \
                bash "$REPO/mapmad/scripts/run_p12_pairs.sh" --config "$CFG" --subdir "$subdir" --set "$set" \
                --arms "${arms[@]}" "${seed_args[@]}" ${CL_EXTRA:-} >> "$P12/driver/${label}_gpu${g}.log" 2>&1 &
            PIDS[$g]=$!
            launched=$(( launched + 1 ))
        done
        status gpus="${!PIDS[*]}" subdir="$subdir" arms="${arms[*]}" total="$total" step_label="$label"
        t_last=$SECONDS
        pause "${CL_CHECK_S:-300}"
        used=$(( used + (SECONDS - t_last) * ${#PIDS[@]} / 2 ))
        if (( used > budget )); then
            say "$label: budget exceeded ($(( used / 60 )) min > $(( budget / 60 )) min of GPU-running time)"
            for g in "${!PIDS[@]}"; do touch "$P12/driver/STOP_gpu${g}_$token"; done
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            CL_USED_S=$used
            return 1
        fi
    done
}

crash_rate_ok() {  # subdir total arms... -> 0 if crash records <= 1% of the step's runs
    local subdir="$1" total="$2"; shift 2
    local done crashed
    read -r done crashed < <(python3 "$SCRIPTS/p12_status.py" "$P12" count "$subdir" "$@")
    (( crashed * 100 <= total ))
}
