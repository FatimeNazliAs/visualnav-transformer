#!/bin/bash
# Run one long Phase 9 job in screen `mapmad_p9_<name>` with the EXIT trap (plan §0.4): a non-normal exit writes
# <dir>/STOP_reason.txt (exit code + last log line); a normal finish writes <dir>/DONE.txt. The job's output goes to
# <dir>/<name>.log. The command runs in the background so a kill reaches the trap within ~1 s.
#
#   bash mapmad/scripts/p9_job.sh <name> <dir> <command ...>
set -euo pipefail
NAME="${1:?name}"; DIR="${2:?dir}"; shift 2
mkdir -p "$DIR"
LOG="$DIR/$NAME.log"
if [[ "${P9_JOB_INNER:-}" != 1 ]]; then
    rm -f "$DIR/STOP_reason.txt" "$DIR/DONE.txt"
    screen -dmS "mapmad_p9_$NAME" env P9_JOB_INNER=1 bash "$0" "$NAME" "$DIR" "$@"
    echo "started screen mapmad_p9_$NAME; log $LOG"
    exit 0
fi

finish() {
    code=$?
    if [[ $code -ne 0 || ! -f "$DIR/DONE.txt" ]]; then
        printf 'exit code %s at %s\nlast log line: %s\n' "$code" "$(date '+%F %T')" "$(tail -n 1 "$LOG" 2>/dev/null)" \
            > "$DIR/STOP_reason.txt"
    fi
}
trap finish EXIT
trap 'kill "$child" 2>/dev/null; exit 143' TERM INT HUP
echo "[$(date '+%F %T')] start: $*" >> "$LOG"
"$@" >> "$LOG" 2>&1 &
child=$!
wait "$child"
echo "[$(date '+%F %T')] done" >> "$LOG"
date '+%F %T' > "$DIR/DONE.txt"
