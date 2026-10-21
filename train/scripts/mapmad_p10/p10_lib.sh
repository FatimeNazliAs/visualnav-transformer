#!/bin/bash
# Shared functions of the Phase 10 chain (run_p10_chain.sh) and twin launcher (run_p10_twin.sh). Sourced, never run.
# The caller sets: REPO (worktree), P10 (host p10_v2_train folder), C (the same as the container sees it, /outputs/...),
# WHO (chain | twin), LOG (its log file). Every GPU job is pinned (docker exec -e CUDA_VISIBLE_DEVICES=<g>) and marked
# MAPMAD_P10_OWNER=$WHO (gpu_watch.py tells our processes apart by it).

TRAIN_DIR=/app/visualnav-transformer/train
SCRIPTS="$REPO/train/scripts/mapmad_p10"
CHECK_S="${CHECK_S:-300}"

say() {  # message -> own log + phase10_log.md
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
    echo "- $(date '+%m-%d %H:%M') [$WHO] $*" >> "$P10/phase10_log.md"
}

status() {  # key=value ... [--state S] -> driver/<who>_state.json + STATUS.md
    python3 "$SCRIPTS/p10_status.py" "$P10" "$WHO" --set "$@" >> "$LOG" 2>&1
}

pause() { sleep "$1" & wait $!; }  # a sleep that TERM can interrupt at once

free_for() {  # owner -> GPUs free for it (gpu_watch.py's gpu_free.json)
    python3 -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['free'][sys.argv[2]]))" \
        "$P10/driver/gpu_free.json" "$1" 2>/dev/null
}

gpu_clean() { python3 "$SCRIPTS/gpu_watch.py" --check "$1" "$WHO" >> "$LOG" 2>&1; }

wait_gpu() {  # [preferred gpu] [extra condition function] -> echoes the GPU; waits in CHECK_S steps
    local prefer="${1:-}" cond="${2:-}" g list
    while true; do
        list="$(free_for "$WHO")"
        if [[ -z "$cond" ]] || "$cond"; then
            for g in $prefer $list; do
                if [[ " $list " == *" $g "* ]] && gpu_clean "$g"; then echo "$g"; return 0; fi
            done
        fi
        pause "$CHECK_S"
    done
}

dx() {  # gpu ("" = CPU) command... -> docker exec in naz_mapmad, pinned + marked, from train/
    local g="$1"; shift
    docker exec -e CUDA_VISIBLE_DEVICES="$g" -e MAPMAD_P10_OWNER="$WHO" -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e MPLCONFIGDIR=/tmp/mpl naz_mapmad \
        bash -lc "cd $TRAIN_DIR && $*"
}

run_bg() {  # command... (stdout/stderr already redirected by the caller) -> waits, TERM-safe; returns its code
    "$@" &
    BG_PID=$!
    wait "$BG_PID"
}

newest_run() { ls -td "$P10/runs/$1"_2* 2>/dev/null | head -1; }  # train.py: <logs_root>/runs/<run>_<time>

overlay() {  # file base(container path) key: value ... -> a launch overlay config (container sees it under $C)
    local file="$1" base="$2"; shift 2
    mkdir -p "$(dirname "$file")"
    { echo "# Phase 10 launch overlay written by the $WHO driver $(date '+%F %T')"; echo "base_config: $base"
      echo "gpu_ids: [0]"; for kv in "$@"; do echo "$kv"; done; } > "$file"
}

monitor_while() {  # pid -> refresh STATUS.md every 60 s while it runs
    while kill -0 "$1" 2>/dev/null; do status; pause 60; done
}

train_once() {  # gpu config(container) log resume_arg -> train.py exit code
    local g="$1" cfg="$2" log="$3" resume="${4:-}"
    dx "$g" "python train.py -c $cfg $resume" > "$log" 2>&1 &
    BG_PID=$!
    monitor_while "$BG_PID" &
    local mon=$!
    wait "$BG_PID"; local code=$?
    kill "$mon" 2>/dev/null
    return "$code"
}

finished_ok() {  # log run_dir -> FINISHED TRAINING and ema_19.pth
    [[ -n "$2" && -f "$2/ema_19.pth" ]] && tr '\r' '\n' < "$1" | grep -a -q "FINISHED TRAINING"
}

supervised_train() {  # label gpu config(container) accum -> RUN_DIR; one resume; returns 0 / stop reason in STOP_WHY
    local label="$1" g="$2" cfg="$3" accum="$4" log="$P10/driver/train_${label}.log" d
    STOP_WHY=""
    say "train $label on GPU $g: python train.py -c $cfg (expandable_segments:True)"
    train_once "$g" "$cfg" "$log"
    d="$(newest_run "$label")"; RUN_DIR="$d"
    status main_run="$d"
    if finished_ok "$log" "$d"; then say "$label finished"; return 0; fi
    if grep -a -q "non-finite loss" "$log"; then STOP_WHY="NaN loss ($label, $log)"; return 1; fi
    if grep -a -q -i "out of memory" "$log" && [[ "$accum" != "1" ]]; then STOP_WHY="out-of-memory after the fallback ($label)"; return 1; fi
    if [[ -z "$d" || ! -f "$d/latest.pth" ]]; then STOP_WHY="crash before any latest.pth ($label, $log)"; return 1; fi
    say "CRASH 1 of $label ($log, last line: $(tail -n 1 "$log" | tr -d '\r' | cut -c1-150)) -> resume once from latest.pth"
    if ! gpu_clean "$g"; then say "GPU $g now holds another process: waiting for a free GPU to resume"; g="$(wait_gpu)"; fi
    status gpu="$g" note="resume 1"
    train_once "$g" "$cfg" "${log%.log}_resume1.log" "--resume ${d/$P10/$C}/latest.pth"
    if finished_ok "${log%.log}_resume1.log" "$d"; then say "$label finished after resume 1"; return 0; fi
    if grep -a -q "non-finite loss" "${log%.log}_resume1.log"; then STOP_WHY="NaN loss ($label, after resume)"; return 1; fi
    STOP_WHY="second crash ($label, ${log%.log}_resume1.log)"; return 1
}

offline_eval() {  # gpu label ckpt(container) -> offline/<label>_ema19/{search,explore,gostanford}; returns 0 / 1
    local g="$1" label="$2" ckpt="$3" o="$C/offline/${2}_ema19" lists="$C/offline/lists" cfg="config/mapmad_v2.yaml"
    local olog="$P10/driver/offline_${label}.log"
    say "offline checks of $label ema_19 on GPU $g -> offline/${label}_ema19"
    run_bg dx "$g" "python -m vint_train.mapmad.offline_eval gostanford --config $cfg --checkpoint $ckpt \
        --official /outputs/mapmad/weights/official/nomad.pth --out $o/gostanford && \
        python -m vint_train.mapmad.training_checks gostanford-report --scores $o/gostanford/gostanford_scores.npz \
        --subset /outputs/mapmad/p4_train/subsets/gostanford_subset.json --out $o/gostanford/gostanford_report.json && \
        python -m vint_train.mapmad.offline_eval score --config $cfg --checkpoint $ckpt --list $lists/search_list.json --out $o/search && \
        python -m vint_train.mapmad.offline_eval compare --scores $o/search/scores.npz --out $o/search && \
        python -m vint_train.mapmad.offline_eval score --config $cfg --checkpoint $ckpt --list $lists/explore_list.json \
        --arms explore_maps,hidden --out $o/explore && \
        python -m vint_train.mapmad.offline_eval compare --scores $o/explore/scores.npz --out $o/explore" >> "$olog" 2>&1
}

pretrain_report() {  # gpu -> offline/pretrain/{search,explore} (untrained v2 = official + fresh encoders)
    local g="$1" o="$C/offline/pretrain" lists="$C/offline/lists" cfg="config/mapmad_v2.yaml"
    say "pre-training report (untrained v2) on GPU $g -> offline/pretrain"
    run_bg dx "$g" "python -m vint_train.mapmad.offline_eval score --config $cfg --init-official --list $lists/search_list.json \
        --arms maps,hidden --out $o/search && python -m vint_train.mapmad.offline_eval compare --scores $o/search/scores.npz --out $o/search && \
        python -m vint_train.mapmad.offline_eval score --config $cfg --init-official --list $lists/explore_list.json \
        --arms explore_maps,hidden --out $o/explore && \
        python -m vint_train.mapmad.offline_eval compare --scores $o/explore/scores.npz --out $o/explore" \
        >> "$P10/driver/offline_pretrain.log" 2>&1
}
