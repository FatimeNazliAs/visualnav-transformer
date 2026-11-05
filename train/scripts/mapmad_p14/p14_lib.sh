#!/bin/bash
# Shared functions of the Phase 14a chain (run_p14a_chain.sh) = Phase 13's p13_lib.sh with Phase 14a names (P14 folder,
# p14_status.py, phase14a_log.md; marks stay MAPMAD_P13_OWNER, run_p13_pairs.sh unchanged) and one change: closed_loop
# counts only the runs whose episode id starts with CL_EP_PREFIX (the two practice parts p13d / p14d share a run folder).
# Sourced, never run. The caller sets: REPO (worktree), P14 (host p14a_finish folder), C (the same as the containers see
# it), WHO (chain), LOG (its log file).
# Every GPU job is pinned (CUDA_VISIBLE_DEVICES / --gpu) and marked MAPMAD_P13_OWNER=$WHO, so the GPU watcher
# (mapmad_p10/gpu_watch.py with MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER, RECENT_S=3600) tells our processes apart (§5).

TRAIN_DIR=/app/visualnav-transformer/train
CFG="${P14_CFG:-/app/visualnav-transformer/mapmad/configs/p14a_finish.yaml}"
HCFG="$REPO/mapmad/configs/${CFG##*/}"   # the same config, host path (p12_plan.py)
SCRIPTS="$REPO/train/scripts/mapmad_p14"
WATCH="$REPO/train/scripts/mapmad_p10/gpu_watch.py"
PLACE="$REPO/train/scripts/mapmad_p10/gpu_place.py"
CHECK_S="${CHECK_S:-300}"
export MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER
export P14_OUTPUTS="${P14_OUTPUTS:-${OUT_MOUNT}/mapmad}"

say() {  # message -> own log + phase14a_log.md
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
    echo "- $(date '+%m-%d %H:%M') [$WHO] $*" >> "$P14/${P14_LOGNAME:-phase14a_log.md}"
}

status() {  # key=value ... [--state S] -> driver/chain_state.json + STATUS.md
    python3 "$SCRIPTS/p14_status.py" "$P14" "$WHO" --set "$@" >> "$LOG" 2>&1
}

pause() { sleep "$1" & wait $!; }  # a sleep that TERM can interrupt at once

# GPU placement: one module (mapmad_p10/gpu_place.py) answers "which GPUs may $WHO take now?" -- the watcher's free
# list (20-min rule + rule 11; a stale file = nothing free; DRY runs: P14_GPUS / P14_GPU_FREE) and the live check.
place() { python3 "$PLACE" "$@" --owner "$WHO" 2>> "$LOG"; }
place_free() { place "$1" --free-file "${P14_GPU_FREE:-$P14/driver/gpu_free.json}" --fixed "${P14_GPUS:-}" "${@:2}"; }
free_for() { place_free free; }        # -> GPUs free for us by the watcher (no live check)
usable_now() { place_free usable; }    # -> GPUs free for us AND clean right now
gpu_clean() { place clean "$1"; }      # live nvidia-smi right now: nobody else on GPU $1 (§8 rule 9)

wait_gpu() {  # [preferred gpu ...] -> echoes a GPU free for 20 min AND clean right now; waits in CHECK_S
    local g
    until g="$(place_free pick --prefer "$@")"; do pause "$CHECK_S"; done
    echo "$g"
}

dx() {  # gpu ("" = CPU) command... -> docker exec in naz_mapmad, pinned + marked, from train/
    local g="$1"; shift
    docker exec -e CUDA_VISIBLE_DEVICES="$g" -e MAPMAD_P13_OWNER="$WHO" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e MPLCONFIGDIR=/tmp/mpl -e P14_OUTPUTS=/outputs/mapmad \
        -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad bash -lc "cd $TRAIN_DIR && $*"
}

dh() {  # command... -> docker exec in naz_mapmad_habitat (CPU unless the command names a GPU), marked
    docker exec -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer naz_mapmad_habitat bash -lc "$*"
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
    local d="$P14/$1/recog_restarts" m=0 n f
    [[ -d "$d" ]] || { echo 0; return; }
    for f in "$d"/*; do [[ -f "$f" ]] || continue; n=$(wc -l < "$f"); (( n > m )) && m=$n; done
    echo "$m"
}

newest_progress_age() {  # subdir -> seconds since the newest progress / server log line of the step (silent-log watchdog)
    python3 - "$P14/$1" <<'EOF'
import glob, os, sys, time
files = glob.glob(os.path.join(sys.argv[1], "progress.*.log")) + glob.glob(os.path.join(sys.argv[1], "logs", "*", "*.partial"))
print(int(time.time() - max((os.path.getmtime(f) for f in files), default=time.time())))
EOF
}

# --- closed-loop runs on a shared queue (block item 9, §8 rules 2-5, 9, 11) -------------------------------------------
closed_loop() {  # label subdir set total budget_s recog seeds arms... -> 0 all done, 1 budget, 2 no progress, 3 recogniser
    # Phase 13: CL_CFG = another config (hidden-DAgger rollouts), CL_PREFIX = its out_dir below $P14 ("hidden_dagger/")
    # Pairs (PAIRS_PER_GPU, default 3) on every GPU free for us (20-min rule + rule 11 + live check right before), all on
    # one run queue; a GPU that gets another user's process is withdrawn (STOP file: its runners finish their current run,
    # take no more); a GPU free later is added to the same queue. When every runner has exited and runs are left, a new
    # launch token restarts the queue. Budget = GPU-running time scaled by the share of the two GPUs in use (waiting is
    # not counted; CL_USED_S returns it; Phase 14b-3: CL_UNIT_DIV=1 counts wall time x GPUs in use = the GPU-hours of
    # every other step, one unit). A recogniser restarted > 3 times -> 3; no new run for SILENT_S -> the runners
    # are killed (a hang, logged) and the round restarts.
    local label="$1" subdir="$2" set="$3" total="$4" budget="$5" recog="$6" seeds="$7"; shift 7
    local arms=("$@") round=0 used=0 t_last token g done crashed pids_alive progress_at_round=0 launched=0 seed_args=()
    [[ -n "$seeds" ]] && seed_args=(--seeds $seeds)
    declare -A PIDS=() ENDED=()
    while true; do
        read -r done crashed < <(python3 "$SCRIPTS/p14_status.py" "$P14" count "${CL_PREFIX:-}$subdir" "${arms[@]}")
        if (( done + crashed >= total )); then
            say "$label: all $total runs done ($crashed crashed); used $(( used / 60 )) min of GPU-running time"
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            CL_USED_S=$used
            return 0
        fi
        if (( $(recog_restarts "${CL_PREFIX:-}$subdir") > 3 )); then
            say "$label: a recogniser sidecar was restarted more than 3 times ($subdir/recog_restarts)"
            for g in "${!PIDS[@]}"; do touch "$P14/driver/STOP_gpu${g}_$token"; done
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            CL_USED_S=$used
            return 3
        fi
        pids_alive=0
        for g in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$g]}" 2>/dev/null; then
                pids_alive=1
                if [[ ! -f "$P14/driver/STOP_gpu${g}_$token" ]] && ! gpu_clean "$g"; then
                    touch "$P14/driver/STOP_gpu${g}_$token"
                    nvidia-smi >> "$P14/driver/nvidia_smi.log" 2>&1
                    say "GPU event: another user's process on GPU $g -> withdrawing $label pairs there (current runs finish)"
                fi
            else
                wait "${PIDS[$g]}" 2>/dev/null
                unset "PIDS[$g]"
                ENDED[$g]="$token"
            fi
        done
        if (( pids_alive == 1 )) && (( $(newest_progress_age "${CL_PREFIX:-}$subdir") > ${SILENT_S:-1800} )); then
            say "HANG: no run finished and no log written in $subdir for > ${SILENT_S:-1800} s -> killing this step's runners"
            kill_inside "run_search .*--subdir $subdir "
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
            [[ -f "$P14/driver/STOP_gpu${g}_$token" ]] && continue
            gpu_clean "$g" || continue  # live check only for GPUs not already ours (not usable_now: fewer nvidia-smi)
            nvidia-smi >> "$P14/driver/nvidia_smi.log" 2>&1
            say "GPU event: GPU $g taken by $label (free 20 min, rule 11, live check); ${PAIRS_PER_GPU:-3} pairs, recogniser $recog, token $token"
            GPU="$g" TOKEN="$token" OWNER="$WHO" STOP_FILE="$C/driver/STOP_gpu${g}_$token" PORT0="${CL_PORT0:-5560}" \
                RPORT0="${CL_RPORT0:-5700}" IPORT0="${CL_IPORT0:-5960}" RECOG="$recog" PAIRS_PER_GPU="${PAIRS_PER_GPU:-3}" \
                bash "$REPO/mapmad/scripts/run_p13_pairs.sh" --config "${CL_CFG:-$CFG}" --subdir "$subdir" --set "$set" \
                --arms "${arms[@]}" "${seed_args[@]}" ${CL_EXTRA:-} >> "$P14/driver/${label}_gpu${g}.log" 2>&1 &
            PIDS[$g]=$!
            launched=$(( launched + 1 ))
        done
        status gpus="${!PIDS[*]}" subdir="${CL_PREFIX:-}$subdir" arms="${arms[*]}" total="$total" step_label="$label"
        t_last=$SECONDS
        pause "${CL_CHECK_S:-300}"
        used=$(( used + (SECONDS - t_last) * ${#PIDS[@]} / ${CL_UNIT_DIV:-2} ))  # 14b-3: CL_UNIT_DIV=1 -> wall x GPUs
        if (( used > budget )); then
            say "$label: budget exceeded ($(( used / 60 )) min > $(( budget / 60 )) min of GPU-running time)"
            for g in "${!PIDS[@]}"; do touch "$P14/driver/STOP_gpu${g}_$token"; done
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            CL_USED_S=$used
            return 1
        fi
    done
}

crash_rate_ok() {  # subdir total arms... -> 0 if crash records <= 1% of the step's runs
    local subdir="$1" total="$2"; shift 2
    local done crashed
    read -r done crashed < <(python3 "$SCRIPTS/p14_status.py" "$P14" count "$subdir" "$@")
    (( crashed * 100 <= total ))
}

# --- training (Phase 11's p11_lib.sh, unchanged except the folder) -----------------------------------------------------
STALL_S="${STALL_S:-1800}"

train_gpu() { wait_gpu ${P14_TRAIN_GPUS:-1}; }  # -> a usable GPU, preferring P14_TRAIN_GPUS (kickoff: GPU 1)

monitor_while() {  # pid log pattern -> refresh STATUS.md every 60 s; a log silent for STALL_S -> kill (= a crash)
    local pid="$1" log="$2" pattern="$3"
    while kill -0 "$pid" 2>/dev/null; do
        status
        if [[ -f "$log" ]] && (( $(date +%s) - $(stat -c %Y "$log") > STALL_S )); then
            say "HANG: $log silent for > $STALL_S s -> killing '$pattern' (counts as a crash)"
            kill_inside "$pattern"
            return 0
        fi
        pause 60
    done
}

train_once() {  # gpu config(container) log resume_arg -> train.py exit code
    local g="$1" cfg="$2" log="$3" resume="${4:-}"
    dx "$g" "python train.py -c $cfg $resume" > "$log" 2>&1 &
    BG_PID=$!
    monitor_while "$BG_PID" "$log" "train.py -c $cfg" &
    local mon=$!
    wait "$BG_PID"; local code=$?
    kill "$mon" 2>/dev/null
    return "$code"
}

newest_run() { ls -td "$1/runs/$2"_2* 2>/dev/null | head -1; }  # train.py: <logs_root>/runs/<run>_<time>

finished_ok() {  # log run_dir last_epoch -> FINISHED TRAINING and ema_<last>.pth
    [[ -n "$2" && -f "$2/ema_$3.pth" ]] && tr '\r' '\n' < "$1" | grep -a -q "FINISHED TRAINING"
}

supervised_train() {  # label gpu config(container) last_epoch root(host) -> RUN_DIR; one resume; STOP_WHY on failure
    local label="$1" g="$2" cfg="$3" last="$4" root="$5" log d
    log="$P14/driver/train_${label}.log"
    STOP_WHY=""
    say "train $label on GPU $g: python train.py -c $cfg"
    train_once "$g" "$cfg" "$log"
    d="$(newest_run "$root" "$label")"; RUN_DIR="$d"
    status run="$d"
    if finished_ok "$log" "$d" "$last"; then say "$label finished"; return 0; fi
    if grep -a -q "non-finite loss" "$log"; then STOP_WHY="NaN loss ($label, $log)"; return 1; fi
    if [[ -z "$d" || ! -f "$d/latest.pth" ]]; then STOP_WHY="training crash before any latest.pth ($label, $log)"; return 1; fi
    say "CRASH 1 of $label ($log, last line: $(tail -n 1 "$log" | tr -d '\r' | cut -c1-150)) -> resume once from latest.pth"
    if ! gpu_clean "$g"; then say "GPU $g now holds another process: waiting for a free GPU to resume"; g="$(wait_gpu)"; fi
    status gpus="$g" note="resume 1"
    train_once "$g" "$cfg" "${log%.log}_resume1.log" "--resume ${d/$P14/$C}/latest.pth"
    if finished_ok "${log%.log}_resume1.log" "$d" "$last"; then say "$label finished after resume 1"; return 0; fi
    if grep -a -q "non-finite loss" "${log%.log}_resume1.log"; then STOP_WHY="NaN loss ($label, after resume)"; return 1; fi
    STOP_WHY="second training crash ($label, ${log%.log}_resume1.log)"; return 1
}

overlay() {  # file base(container path) key: value ... -> a launch overlay config (the containers see it under $C)
    local file="$1" base="$2"; shift 2
    mkdir -p "$(dirname "$file")"
    { echo "# Phase 13 launch overlay written by the $WHO driver $(date '+%F %T')"; echo "base_config: $base"
      echo "gpu_ids: [0]"; for kv in "$@"; do echo "$kv"; done; } > "$file"
}

offline_checks() {  # gpu label ckpt(container) train_cfg(container) -> offline/<label>/: G10 rows 3-4, seen list
    local g="$1" label="$2" ckpt="$3" cfg="$4" o="$C/offline/$2" lists=/outputs/mapmad/p10_v2_train/offline/lists
    say "offline checks of $label on GPU $g -> offline/$label"
    run_limit 10800 "offline/$label" dx "$g" "python -m vint_train.mapmad.offline_eval gostanford --config $cfg --checkpoint $ckpt \
        --official /outputs/mapmad/weights/official/nomad.pth --out $o/gostanford && \
        python -m vint_train.mapmad.offline_eval score --config $cfg --checkpoint $ckpt --list $lists/search_list.json --out $o/search && \
        python -m vint_train.mapmad.offline_eval compare --scores $o/search/scores.npz --out $o/search && \
        python -m vint_train.mapmad.offline_eval score --config $cfg --checkpoint $ckpt --list $lists/explore_list.json \
        --arms explore_maps,hidden --out $o/explore && \
        python -m vint_train.mapmad.offline_eval compare --scores $o/explore/scores.npz --out $o/explore && \
        python -m vint_train.mapmad.offline_eval score --config $cfg --checkpoint $ckpt --list $lists/seen_list.json \
        --arms maps,target_zeroed,target_shifted --out $o/seen && \
        python -m vint_train.mapmad.target_groups seen-report --scores $o/seen/scores.npz --out $o/seen" \
        >> "$P14/driver/offline_${label}.log" 2>&1
}
