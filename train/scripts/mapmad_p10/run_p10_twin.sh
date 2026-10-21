#!/bin/bash
# Phase 10 twin launcher (report only; p10_rules.md §1, §8). Host side, in screen p10_twin (launch_p10.sh). Waits,
# checking every 5 min, until ALL of:
#   - the chain's main run has started (priority: Phase 10 main > Phase 9 > twin);
#   - Phase 9 no longer needs a GPU: its STATUS.md (read only) shows the photo test done, AT GATE or STOPPED;
#   - a GPU is free for the twin (gpu_watch.py: no other process for 20 min; the chain's GPU is never free for it);
# then trains seed 2 (mapmad_v2_twin.yaml with the main run's batch / accumulation / workers from launch/main.yaml,
# one resume) and runs the same offline checks on its ema_19 -> offline/twin_ema19/ (+ a draft table). It never holds
# the gate. EXIT trap -> driver/STOP_twin_reason.txt. The chain never starts it.
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P10="$OUT_MOUNT/mapmad/p10_v2_train"
    C=/outputs/mapmad/p10_v2_train
    P9_STATUS="${P9_STATUS:-$OUT_MOUNT/mapmad/p9_recognisers/STATUS.md}"
    WHO=twin
    LOG="$P10/driver/twin.log"
    mkdir -p "$P10/driver" "$P10/launch"
    # shellcheck source=p10_lib.sh
    source "$REPO/train/scripts/mapmad_p10/p10_lib.sh"
    FINISHED=0
    BG_PID=""
    rm -f "$P10/driver/STOP_twin_reason.txt"
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        for pat in "launch/twin.yaml" "twin_ema19"; do kill_inside "$pat"; done  # our own processes only
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "exit code: $code"; echo "reason: ${WHY:-unexpected exit}";
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P10/driver/STOP_twin_reason.txt"
            status reason="${WHY:-exit $code}" --state STOPPED
            say "twin STOP (exit $code): ${WHY:-see driver/STOP_twin_reason.txt}"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    cd "$REPO" || exit 2
    if [[ -n "$(git status --porcelain -- train mapmad)" ]]; then WHY="uncommitted code"; exit 3; fi
    status step="waiting (main run started? Phase 9 done with GPUs? second GPU free?)" --state WAITING
    say "twin launcher start (commit $(git rev-parse HEAD)); Phase 9 status file: $P9_STATUS"

    main_started() { python3 -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get('main_run') else 1)" \
        "$P10/driver/chain_state.json" 2>/dev/null; }
    phase9_done() { grep -q -i -E "AT GATE|STOPPED|photo test done|photo test: done" "$P9_STATUS" 2>/dev/null; }
    ready() {
        local why=""
        main_started || why="main run not started"
        phase9_done || why="${why:+$why; }Phase 9 still needs a GPU"
        [[ -n "$(free_for twin)" ]] || why="${why:+$why; }no GPU free for the twin"
        status note="${why:-starting}"
        [[ -z "$why" ]]
    }
    G="$(wait_gpu "" ready)"
    [[ -f "$P10/launch/main.yaml" ]] || { WHY="launch/main.yaml missing"; exit 4; }
    local accum batch workers
    batch="$(grep '^batch_size:' "$P10/launch/main.yaml" | awk '{print $2}')"
    accum="$(grep '^grad_accum_steps:' "$P10/launch/main.yaml" | awk '{print $2}')"
    workers="$(grep '^num_workers:' "$P10/launch/main.yaml" | awk '{print $2}')"
    overlay "$P10/launch/twin.yaml" "$TRAIN_DIR/config/mapmad_v2_twin.yaml" "batch_size: $batch" \
        "grad_accum_steps: $accum" "num_workers: $workers"
    nvidia-smi >> "$P10/driver/nvidia_smi.log" 2>&1
    say "GPU event: GPU $G taken by the twin (seed 2, batch $batch x accum $accum, workers $workers)"
    status step="twin run (seed 2)" gpu="$G" --state RUNNING
    supervised_train v2_s2 "$G" "$C/launch/twin.yaml" "$accum" || { WHY="$STOP_WHY"; exit 5; }
    mkdir -p "$P10/offline/twin_ema19"
    sha256sum "$RUN_DIR/ema_19.pth" > "$P10/offline/twin_ema19/ema_19.sha256"
    [[ -s "$P10/offline/twin_ema19/ema_19.sha256" ]] || { WHY="checkpoint sha256 not written"; exit 6; }
    status step="twin offline checks (waiting for a GPU)" --state WAITING
    G="$(wait_gpu "$G")"
    status step="twin offline checks" gpu="$G" --state RUNNING
    offline_eval "$G" twin "${RUN_DIR/$P10/$C}/ema_19.pth" || { WHY="twin offline checks failed"; exit 7; }
    python3 "$SCRIPTS/g10_table.py" "$P10" twin >> "$LOG" 2>&1
    say "GPU event: twin released GPU $G; twin done (report only)"
    FINISHED=1
    date '+%F %T' > "$P10/driver/TWIN_DONE.txt"
    status step="twin done" --state DONE
}

main "$@"
exit $?
