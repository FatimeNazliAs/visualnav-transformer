#!/bin/bash
# Phase 11 twin of the final v2 (block item 8; report only, never holds the gate). Host side, in screen p11_twin
# (launch_p11.sh). Waits until the chain has picked v2 (V2_FINAL.json), restarted naz_mapmad and taken its GPU for the
# fine-tune (driver/FINETUNE_GPU), then trains seed 2 with the winner's Phase 10 recipe (20 epochs:
# mapmad_v2_twin.yaml or mapmad_v2_seen_twin.yaml) on a GPU free for the twin (20-min rule + live check; a GPU the
# chain uses never counts as free for the twin), one resume; then G10 rows 3-4 + the seen list on its ema_19
# (offline/twin_ema19). Priority (kickoff §5): the chain never waits for the twin; the twin only takes a GPU the chain
# does not hold. Stops -> driver/STOP_twin_reason.txt (the chain's STOP_reason.txt is never touched).
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P11="${P11_DIR:-$OUT_MOUNT/mapmad/p11_dagger}"
    C="${P11/$OUT_MOUNT//outputs}"
    WHO=twin
    LOG="$P11/driver/twin.log"
    mkdir -p "$P11/driver" "$P11/launch" "$P11/twin"
    exec 2>> "$LOG"
    # shellcheck source=p11_lib.sh
    source "$REPO/train/scripts/mapmad_p11/p11_lib.sh"
    FINISHED=0
    BG_PID=""

    stop() {
        { echo "time: $(date '+%F %T')"; echo "reason: $1"; } > "$P11/driver/STOP_twin_reason.txt"
        status step="${STEP:-?}" reason="$1" --state STOPPED
        say "TWIN STOP: $1"
        FINISHED=1
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        for pat in "launch/twin.yaml" "offline/twin_ema19"; do kill_inside "$pat"; done
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "exit code: $code"; } > "$P11/driver/STOP_twin_reason.txt"
            status reason="twin driver exit $code" --state STOPPED
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    STEP="waiting for the chain (v2 pick, restart, fine-tune GPU)"
    status step="$STEP" --state WAITING
    while [[ ! -f "$P11/driver/FINETUNE_GPU" ]]; do
        [[ -f "$P11/STOP_reason.txt" ]] && stop "the chain stopped before the fine-tune: no twin"
        pause "$CHECK_S"
    done
    local winner cfg label g
    winner="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['arm'])" "$P11/V2_FINAL.json")"
    cfg=mapmad_v2_twin.yaml
    [[ "$winner" == runC ]] && cfg=mapmad_v2_seen_twin.yaml
    label="$(grep -E '^run_name:' "$REPO/train/config/$cfg" | awk '{print $2}')"
    say "twin: v2 = $winner -> $cfg (seed 2, 20 epochs); waiting for a GPU free for the twin"
    STEP="waiting for a GPU (not the chain's)"
    status step="$STEP" --state WAITING
    g="$(wait_gpu "" "$(cat "$P11/driver/FINETUNE_GPU")")"
    overlay "$P11/launch/twin.yaml" "$TRAIN_DIR/config/$cfg" "logs_root: $C/twin"
    STEP="training $label"
    status step="$STEP" gpus="$g" epochs=20 --state RUNNING
    nvidia-smi >> "$P11/driver/nvidia_smi.log" 2>&1
    say "GPU event: GPU $g taken by the twin"
    supervised_train "$label" "$g" "$C/launch/twin.yaml" 19 "$P11/twin" || stop "$STOP_WHY"
    mkdir -p "$P11/offline/twin_ema19"
    sha256sum "$RUN_DIR/ema_19.pth" > "$P11/offline/twin_ema19/ema_19.sha256"
    STEP="offline checks (rows 3-4, seen list)"
    status step="$STEP" run=""
    gpu_clean "$g" || g="$(wait_gpu)"
    offline_checks "$g" twin_ema19 "${RUN_DIR/$P11/$C}/ema_19.pth" config/mapmad_v2.yaml \
        || stop "twin offline checks failed (driver/offline_twin_ema19.log)"
    python3 "$SCRIPTS/g11_table.py" "$P11" >> "$LOG" 2>&1
    say "GPU event: twin released GPU $g; twin done"
    FINISHED=1
    status step="done" --state DONE
    date '+%F %T' > "$P11/driver/TWIN_DONE.txt"
}

main "$@"
exit $?
