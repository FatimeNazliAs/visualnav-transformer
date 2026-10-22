#!/bin/bash
# Shared functions of the Phase 11 chain (run_p11_chain.sh) and twin launcher (run_p11_twin.sh). Sourced, never run.
# The caller sets: REPO (worktree), P11 (host p11_dagger folder), C (the same as the containers see it,
# /outputs/mapmad/p11_dagger), WHO (chain | twin), LOG (its log file). Every GPU job is pinned (CUDA_VISIBLE_DEVICES /
# --gpu) and marked MAPMAD_P11_OWNER=$WHO, so the GPU watcher (mapmad_p10/gpu_watch.py with
# MAPMAD_OWNER_ENV=MAPMAD_P11_OWNER) tells our processes apart from everyone else's.

TRAIN_DIR=/app/visualnav-transformer/train
CFG=/app/visualnav-transformer/mapmad/configs/p11_dagger.yaml
SCRIPTS="$REPO/train/scripts/mapmad_p11"
WATCH="$REPO/train/scripts/mapmad_p10/gpu_watch.py"
CHECK_S="${CHECK_S:-300}"
export MAPMAD_OWNER_ENV=MAPMAD_P11_OWNER

say() {  # message -> own log + phase11_log.md
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
    echo "- $(date '+%m-%d %H:%M') [$WHO] $*" >> "$P11/phase11_log.md"
}

status() {  # key=value ... [--state S] -> driver/<who>_state.json + STATUS.md
    python3 "$SCRIPTS/p11_status.py" "$P11" "$WHO" --set "$@" >> "$LOG" 2>&1
}

pause() { sleep "$1" & wait $!; }  # a sleep that TERM can interrupt at once

free_for() {  # owner -> GPUs free for it for 20 min (gpu_watch.py's gpu_free.json; DRY runs: P11_GPU_FREE)
    python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['free'][sys.argv[2]]))" \
        "${P11_GPU_FREE:-$P11/driver/gpu_free.json}" "$1" 2>/dev/null
}

gpu_clean() { python3 "$WATCH" --check "$1" "$WHO" >> "$LOG" 2>&1; }  # live nvidia-smi: nobody else on GPU $1

wait_gpu() {  # [preferred gpu] [excluded gpu] -> echoes a GPU free for 20 min AND clean right now; waits in CHECK_S
    local prefer="${1:-}" exclude="${2:-}" g list
    while true; do
        list="$(free_for "$WHO")"
        for g in $prefer $list; do
            [[ "$g" == "$exclude" ]] && continue
            if [[ " $list " == *" $g "* ]] && gpu_clean "$g"; then echo "$g"; return 0; fi
        done
        pause "$CHECK_S"
    done
}

dx() {  # gpu ("" = CPU) command... -> docker exec in naz_mapmad, pinned + marked, from train/
    local g="$1"; shift
    docker exec -e CUDA_VISIBLE_DEVICES="$g" -e MAPMAD_P11_OWNER="$WHO" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e MPLCONFIGDIR=/tmp/mpl \
        -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad bash -lc "cd $TRAIN_DIR && $*"
}

dh() {  # command... -> docker exec in naz_mapmad_habitat (CPU unless the command names a GPU), marked
    docker exec -e MAPMAD_P11_OWNER="$WHO" -w /app/visualnav-transformer naz_mapmad_habitat bash -lc "$*"
}

run_bg() {  # command... (stdout/stderr already redirected by the caller) -> waits, TERM-safe; returns its code
    "$@" &
    BG_PID=$!
    wait "$BG_PID"
}

kill_inside() {  # pattern -> kill the matching processes INSIDE our two containers (killing docker exec does not)
    docker exec naz_mapmad pkill -f -- "$1" 2>/dev/null
    docker exec naz_mapmad_habitat pkill -f -- "$1" 2>/dev/null
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

mem_log() {  # label -> one log line with free RAM and swap (MiB)
    say "RAM $1: $(free -m | awk '/^Mem:/ {printf "mem used %d / total %d, available %d", $3, $2, $7} /^Swap:/ {printf "; swap used %d / %d", $3, $2}') MiB"
}

# --- closed-loop runs on a shared queue (block item 10, §8 rules 2-5, 9) -------------------------------------------
closed_loop() {  # label subdir total budget_s args... -> 0 when every run has a summary or crash record, 1 = budget
    # Pairs (3 per GPU) on every GPU free for us (20-min rule + live check right before), all on one run queue; a GPU
    # that gets another user's process is withdrawn (STOP file: its runners finish their current run, take no more);
    # a GPU free later is added to the same queue. When every runner has exited and runs are left (withdrawal, a
    # crashed runner), a new launch token restarts the queue (claims of unfinished runs do not block). The budget
    # counts running time scaled by the share of the two GPUs in use (one GPU for 1 h = 0.5 h; waiting, also for a
    # second GPU, is not counted, p11_rules.md §9).
    # Returns 0 (all runs done), 1 (budget exceeded) or 2 (runners keep exiting without progress: a possible bug).
    local label="$1" subdir="$2" total="$3" budget="$4"; shift 4
    local round=0 used=0 t_last token g done crashed pids_alive progress_at_round=0 launched=0
    declare -A PIDS=() ENDED=()
    while true; do
        read -r done crashed < <(python3 "$SCRIPTS/p11_status.py" "$P11" count "$subdir" ${CL_ARMS})
        if (( done + crashed >= total )); then
            say "$label: all $total runs done ($crashed crashed); used $(( used / 60 )) min of GPU-running time"
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            CL_USED_S=$used
            return 0
        fi
        pids_alive=0
        for g in "${!PIDS[@]}"; do
            if kill -0 "${PIDS[$g]}" 2>/dev/null; then
                pids_alive=1
                if [[ ! -f "$P11/driver/STOP_gpu${g}_$token" ]] && ! gpu_clean "$g"; then
                    touch "$P11/driver/STOP_gpu${g}_$token"
                    say "GPU event: another user's process on GPU $g -> withdrawing $label pairs there (current runs finish)"
                fi
            else
                wait "${PIDS[$g]}" 2>/dev/null
                unset "PIDS[$g]"
                ENDED[$g]="$token"  # its runners found the queue empty (or were withdrawn): not again this round
            fi
        done
        if (( pids_alive == 0 )); then  # nothing running: a new token for the next launch round
            if (( round > 0 && launched > 0 && done + crashed == progress_at_round )); then
                say "$label: every runner of round $round exited without a new run (a possible bug; see driver/${label}_gpu*.log)"
                CL_USED_S=$used
                return 2
            fi
            round=$(( round + 1 )); token="${label}_$(date +%m%d_%H%M%S)_r$round"
            progress_at_round=$(( done + crashed )); launched=0
        fi
        for g in $(free_for "$WHO"); do
            [[ -n "${PIDS[$g]:-}" || "${CL_EXCLUDE:-}" == "$g" || "${ENDED[$g]:-}" == "$token" ]] && continue
            [[ -f "$P11/driver/STOP_gpu${g}_$token" ]] && continue
            gpu_clean "$g" || continue
            nvidia-smi >> "$P11/driver/nvidia_smi.log" 2>&1
            say "GPU event: GPU $g taken by $label (free 20 min + live check); 3 pairs, token $token"
            GPU="$g" TOKEN="$token" OWNER="$WHO" STOP_FILE="$C/driver/STOP_gpu${g}_$token" PORT0="${CL_PORT0:-5560}" \
                bash "$REPO/mapmad/scripts/run_p11_pairs.sh" --config "$CFG" --subdir "$subdir" "$@" \
                >> "$P11/driver/${label}_gpu${g}.log" 2>&1 &
            PIDS[$g]=$!
            launched=$(( launched + 1 ))
        done
        status gpus="${!PIDS[*]}" subdir="$subdir" arms="${CL_ARMS}" total="$total"
        t_last=$SECONDS
        pause "${CL_CHECK_S:-300}"
        used=$(( used + (SECONDS - t_last) * ${#PIDS[@]} / 2 ))  # GPU-running time x share of the 2 GPUs in use
        if (( used > budget )); then
            say "$label: budget exceeded ($(( used / 60 )) min > $(( budget / 60 )) min of GPU-running time)"
            for g in "${!PIDS[@]}"; do touch "$P11/driver/STOP_gpu${g}_$token"; done
            for g in "${!PIDS[@]}"; do wait "${PIDS[$g]}" 2>/dev/null; done
            CL_USED_S=$used
            return 1
        fi
    done
}

# --- supervised training (one resume; Phase 10) ---------------------------------------------------------------------
STALL_S="${STALL_S:-3600}"  # a training log that does not change for this long = a hung run (Phase 10 lesson)

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
    log="$P11/driver/train_${label}.log"
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
    train_once "$g" "$cfg" "${log%.log}_resume1.log" "--resume ${d/$P11/$C}/latest.pth"
    if finished_ok "${log%.log}_resume1.log" "$d" "$last"; then say "$label finished after resume 1"; return 0; fi
    if grep -a -q "non-finite loss" "${log%.log}_resume1.log"; then STOP_WHY="NaN loss ($label, after resume)"; return 1; fi
    STOP_WHY="second training crash ($label, ${log%.log}_resume1.log)"; return 1
}

overlay() {  # file base(container path) key: value ... -> a launch overlay config (the containers see it under $C)
    local file="$1" base="$2"; shift 2
    mkdir -p "$(dirname "$file")"
    { echo "# Phase 11 launch overlay written by the $WHO driver $(date '+%F %T')"; echo "base_config: $base"
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
        >> "$P11/driver/offline_${label}.log" 2>&1
}
