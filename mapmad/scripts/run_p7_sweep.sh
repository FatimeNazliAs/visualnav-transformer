#!/bin/bash
# Phase 7 unattended driver (p7_rules.md §7-10): checks -> sweep -> tables + choice -> report-only extra -> videos ->
# camera_fair.yaml. Run on the host inside screen (no sudo, our containers only), after Naz's OK at S-sweep:
#
#   screen -dmS mapmad_p7_sweep bash mapmad/scripts/run_p7_sweep.sh
#
# Steps (each leaves <p7>/driver/<step>.done; a re-launch skips finished steps, the runners skip finished runs):
#   checks       p7_rules.md sha256 = p7_rules.sha256; episode fingerprint = the rules'; nomad.pth sha256; git HEAD =
#                P7_COMMIT and no changes in mapmad/ or train/vint_train/; GPUs free of other users' jobs
#   determinism  B_fish025, first in-view episode, seed 0, twice -> byte-identical (G7 row 5); compatibility (report):
#                B_pin018 vs B_limo (Phase 5 camera path) on the same run -> equal apart from the header's camera field
#   sweep        16 arms (A_/B_ x 8 cameras) x 3 seeds, 6 pairs (3 per GPU), arm by arm -> sweep/runs/
#   tables       p7_tables (stats, paired vs pin018, the choice) -> sweep/tables.md, choice.md, choice.json
#   extra        the winner with a context frame every 3rd step (X_<winner>_cs3, seed 0) -> sweep/extra.md
#   videos       one video, all 8 cameras side by side, seeded in-view episode -> sweep/videos/
#   camera       mapmad/configs/camera_fair.yaml (the winner)
# Any failed check, a runner stopping on a non-infrastructure error, > 2% crashed runs or a determinism failure writes
# <p7>/STOP_reason.txt and stops; nothing is fixed or rerun. Never commits.
# Any other exit that is not a finished run (a crash, an empty variable under `set -u`, a kill by TERM/INT/HUP) also
# writes STOP_reason.txt, with the exit code and the last log line (trap on EXIT; Naz, 2026-10-16, after the first dry
# run died silently on an empty variable). kill -9 cannot be trapped: watch the screen session too.
# Environment: P7_COMMIT (required: the committed sweep code), P7_PAIRS (default "0 0 0 1 1 1"), P7_SKIP (steps).
# Dry run only: P7_CONFIG=p7_dryrun.yaml (9 episodes, 15 steps, own folder) P7_ALLOW_DIRTY=1 (no commit check).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HABITAT=naz_mapmad_habitat
NOMAD=naz_mapmad
APP=/app/visualnav-transformer
CFG=$APP/mapmad/configs/${P7_CONFIG:-p7_sweep.yaml}
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' "$HABITAT")"
P7C="$(docker exec "$HABITAT" python -c "from mapmad_sim.run_layout import RunLayout; print(RunLayout.load('$CFG').base)")"
P7="$OUT_MOUNT${P7C#/outputs}"
[[ -d "$P7" ]] || { echo "outputs folder $P7 not found" >&2; exit 2; }
PAIRS6="${P7_PAIRS:-0 0 0 1 1 1}"
DRV="$P7/driver"
PHASE_LOG="$P7/phase7_log.md"
CAMS="pin018 fish018 pin025 fish025 pin045 fish045 pin055 fish055"
mkdir -p "$DRV"
exec > >(tee -a "$DRV/run_p7_sweep.log") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" >&2; }
note() { say "$*"; echo "- [driver $(ts)] $*" >> "$PHASE_LOG"; }
CHILD=""
run() {  # long steps run in the background + wait, so a TERM / INT / HUP reaches the trap at once (bash runs traps only
    note "+ $*"  # between foreground commands)
    "$@" &
    CHILD=$!
    wait "$CHILD"
    local rc=$?
    CHILD=""
    return $rc
}
stop_ours() {  # our own processes only: the running step, simulator servers, runners, video renders
    [[ -n "$CHILD" ]] && kill "$CHILD" 2>/dev/null
    docker exec "$HABITAT" pkill -f "mapmad_sim.sidecar_server" 2>/dev/null || true
    docker exec "$HABITAT" pkill -f "p7_videos.py" 2>/dev/null || true
    docker exec "$NOMAD" pkill -f "vint_train.mapmad.closed_loop" 2>/dev/null || true
}
on_exit() {  # every exit that is not a finished run leaves STOP_reason.txt (unless stop() already wrote it)
    local rc=$?
    trap - EXIT
    if [[ $rc -ne 0 && ! -f "$P7/STOP_reason.txt" ]]; then
        sleep 1  # let tee flush the log
        { echo "STOP $(ts)"; echo "driver exited unexpectedly (exit code $rc: crash, empty variable or kill)";
          echo "last log line: $(tail -n 1 "$DRV/run_p7_sweep.log" 2>/dev/null)"; echo;
          echo "Driver log: $DRV/run_p7_sweep.log. Nothing was fixed or rerun."; } > "$P7/STOP_reason.txt"
        echo "- [driver $(ts)] STOP: unexpected exit $rc" >> "$PHASE_LOG"
        stop_ours
    fi
    exit $rc
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
[[ "${P7_TEST_EXIT:-}" == unbound ]] && echo "$P7_TEST_UNSET_VARIABLE"  # trap test only: an empty variable
[[ "${P7_TEST_EXIT:-}" == sleep ]] && run sleep 30  # trap test only: a long step to kill the driver in

stop() {
    say "STOP: $1"
    { echo "STOP $(ts)"; echo "$1"; echo; echo "Driver log: $DRV/run_p7_sweep.log. Nothing was fixed or rerun."; } > "$P7/STOP_reason.txt"
    echo "- [driver $(ts)] STOP: $1" >> "$PHASE_LOG"
    stop_ours
    exit 1
}
done_step() { [[ -f "$DRV/$1.done" || " ${P7_SKIP:-} " == *" $1 "* ]]; }
mark() { date -Is > "$DRV/$1.done"; note "step $1 done"; }
nomad() { docker exec -e MPLCONFIGDIR=/tmp/mpl -e PYTHONPATH=$APP/mapmad/src -w $APP/train "$NOMAD" python -m "$@"; }
habitat() { docker exec -w $APP "$HABITAT" python "$@"; }
runs() { PAIRS="$1" BASE_PORT=5560 HANDOUT=dynamic "$REPO/mapmad/scripts/run_p5.sh" --config "$CFG" "${@:2}"; }

gpus_free() {  # every process on the GPUs this driver uses (P7_PAIRS), other than the display server, must be ours
    nvidia-smi >> "$DRV/nvidia_smi.log" 2>&1
    local me bad=0 pid uuid user name idx
    me="$(id -un)"
    declare -A index
    while IFS=', ' read -r idx uuid; do index[$uuid]=$idx; done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    while IFS=', ' read -r pid uuid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        [[ " $PAIRS6 " == *" ${index[$uuid]} "* ]] || continue  # a GPU we do not use
        name="$(ps -o comm= -p "$pid" 2>/dev/null)"; user="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        case "$name" in Xorg|gnome-shell) continue ;; esac
        if [[ -n "$user" && "$user" != "$me" ]]; then say "GPU ${index[$uuid]}: process $pid ($name) belongs to $user"; bad=1; fi
    done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
    return $bad
}
need_gpus() { gpus_free || stop "a GPU is used by another user's job before step $1 (see $DRV/nvidia_smi.log)"; }
check_code() {
    if [[ -n "${P7_ALLOW_DIRTY:-}" ]]; then note "TEST MODE: code check skipped (P7_ALLOW_DIRTY, dry run only)"; return; fi
    [[ -n "${P7_COMMIT:-}" ]] || stop "P7_COMMIT is not set (the committed sweep code)"
    [[ "$(git -C "$REPO" rev-parse HEAD)" == "$P7_COMMIT"* ]] || stop "git HEAD is not P7_COMMIT=$P7_COMMIT (before $1)"
    [[ -z "$(git -C "$REPO" status --porcelain -- mapmad train/vint_train)" ]] || stop "uncommitted changes in mapmad/ or train/vint_train/ (before $1)"
}
layout_py() {
    habitat -c "from pathlib import Path; from mapmad_sim.run_layout import RunLayout, run_id
L = RunLayout.load(Path('$CFG'), subdir='$1'); print($2)"
}
host() { echo "$OUT_MOUNT${1#/outputs}"; }
crash_share_ok() {  # > 2% crashed runs = a suspected bug
    local sub="$1" n="$2" crashed
    crashed=$(ls "$P7/$sub"/summaries/*/*.crash.json 2>/dev/null | wc -l)
    note "crashed runs in $sub: $crashed of $n"
    (( crashed * 100 <= 2 * n ))
}

[[ -f "$P7/STOP_reason.txt" ]] && { say "STOP_reason.txt exists: review it and move it away before re-launching"; exit 1; }
note "driver start (pid $$, repo $REPO, outputs $P7, commit ${P7_COMMIT:-unset})"
FIRST_IV=$(python3 -c "import json; print(next(e['episode_id'] for e in json.load(open('$P7/episodes/episodes.json'))['episodes'] if e['type'] == 'in_view'))")
N_IV=$(python3 -c "import json; print(sum(e['type'] == 'in_view' for e in json.load(open('$P7/episodes/episodes.json'))['episodes']))")
N_ALL=$(python3 -c "import json; print(len(json.load(open('$P7/episodes/episodes.json'))['episodes']))")

# --- checks --------------------------------------------------------------------------------------------------------
if ! done_step checks; then
    want=$(cut -d' ' -f1 "$P7/p7_rules.sha256"); have=$(sha256sum "$P7/p7_rules.md" | cut -d' ' -f1)
    [[ "$want" == "$have" ]] || stop "p7_rules.md sha256 $have != p7_rules.sha256 $want"
    fp=$(cut -d' ' -f1 "$P7/episodes/episodes.sha256")
    grep -q "$fp" "$P7/p7_rules.md" || stop "episode fingerprint $fp is not the one in p7_rules.md"
    w=$(sha256sum "$OUT_MOUNT/mapmad/weights/official/nomad.pth" | cut -d' ' -f1)  # the file the runners load
    [[ "$w" == 70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333 ]] || stop "nomad.pth sha256 $w"
    check_code checks
    need_gpus checks
    note "checks passed: rules $have, episodes $fp, weights ok, code ${P7_COMMIT:-unset}"
    mark checks
fi

# --- determinism + compatibility -----------------------------------------------------------------------------------
if ! done_step determinism; then
    check_code determinism; need_gpus determinism
    for r in a b; do
        runs "1" --arms B_fish025 --seeds 0 --episodes "$FIRST_IV" --subdir "checks/determinism_$r" || stop "determinism run $r failed"
    done
    runs "1" --arms B_pin018 B_limo --seeds 0 --episodes "$FIRST_IV" --subdir checks/compat || stop "compatibility runs failed"
    la="$(host "$(layout_py checks/determinism_a "L.log('B_fish025', run_id('$FIRST_IV', 0))")")"
    lb="$(host "$(layout_py checks/determinism_b "L.log('B_fish025', run_id('$FIRST_IV', 0))")")"
    if cmp -s "$la" "$lb"; then det=true; else det=false; fi
    p="$(host "$(layout_py checks/compat "L.log('B_pin018', run_id('$FIRST_IV', 0))")")"
    q="$(host "$(layout_py checks/compat "L.log('B_limo', run_id('$FIRST_IV', 0))")")"
    compat=$(python3 - "$p" "$q" <<'PY'
import json, sys
a, b = (open(f).read().splitlines() for f in sys.argv[1:3])
ha, hb = json.loads(a[0]), json.loads(b[0])
for h in (ha, hb):
    h.pop("camera", None); h.pop("arm", None)
print(str(ha == hb and a[1:-1] == b[1:-1] and {k: v for k, v in json.loads(a[-1]).items() if k != "arm"}
          == {k: v for k, v in json.loads(b[-1]).items() if k != "arm"}).lower())
PY
)
    echo "{\"determinism_pass\": $det, \"episode\": \"$FIRST_IV\", \"arm\": \"B_fish025\", \"log_sha256\": [\"$(sha256sum "$la" | cut -d' ' -f1)\", \"$(sha256sum "$lb" | cut -d' ' -f1)\"], \"compat_pin018_equals_phase5_camera\": $compat}" \
        > "$P7/checks/determinism.json"
    note "determinism $det, compatibility $compat ($FIRST_IV)"
    [[ $det == true ]] || stop "determinism failed: B_fish025 $FIRST_IV seed 0 differs on re-run (G7 row 5)"
    mark determinism
fi

# --- sweep ---------------------------------------------------------------------------------------------------------
if ! done_step sweep; then
    check_code sweep; need_gpus sweep
    arms=(); for c in $CAMS; do arms+=("B_$c" "A_$c"); done
    run runs "$PAIRS6" --arms "${arms[@]}" --arm-major --subdir sweep/runs \
        || stop "a sweep runner exited with an error: see sweep/runs/progress.*.log and $DRV/run_p7_sweep.log"
    crash_share_ok sweep/runs $((8 * N_ALL * 3)) || stop "more than 2% of sweep runs crashed"
    mark sweep
fi

# --- tables + choice -----------------------------------------------------------------------------------------------
if ! done_step tables; then
    run nomad vint_train.mapmad.closed_loop.p7_tables --config "$CFG" > "$DRV/tables.out" 2>&1 || stop "p7_tables failed (see $DRV/tables.out)"
    [[ "$(python3 -c "import json; print(json.load(open('$P7/sweep/choice.json'))['complete'])")" == True ]] \
        || stop "sweep incomplete: not every B_ arm has $N_IV x 3 runs (see sweep/tables.md)"
    mark tables
fi
WIN=$(python3 -c "import json; print(json.load(open('$P7/sweep/choice.json'))['winner'])")

# --- report-only extra ---------------------------------------------------------------------------------------------
if ! done_step extra; then
    check_code extra; need_gpus extra
    run runs "$PAIRS6" --arms "X_${WIN}_cs3" --seeds 0 --subdir sweep/runs || stop "extra runner exited with an error"
    mark extra
fi

# --- videos --------------------------------------------------------------------------------------------------------
if ! done_step videos; then
    need_gpus videos
    run habitat mapmad/scripts/p7_videos.py --config "$CFG" --subdir sweep/runs --pick 7710 > "$DRV/video.log" 2>&1 \
        || note "WARNING: video failed (see $DRV/video.log)"
    mark videos
fi

# --- camera_fair.yaml + final tables (with extra.md) -----------------------------------------------------------------
CAM_OUT=$APP/mapmad/configs/camera_fair.yaml; CAM_HOST=$REPO/mapmad/configs/camera_fair.yaml
if [[ -n "${P7_ALLOW_DIRTY:-}" ]]; then  # a dry run never writes the repo's config
    CAM_OUT=$P7C/camera_fair_dryrun.yaml; CAM_HOST=$P7/camera_fair_dryrun.yaml
fi
run nomad vint_train.mapmad.closed_loop.p7_tables --config "$CFG" --write-camera --camera-out "$CAM_OUT" > "$DRV/tables_final.out" 2>&1 \
    || stop "final p7_tables failed (see $DRV/tables_final.out)"
mark camera
DRY=""; [[ -n "${P7_ALLOW_DIRTY:-}" ]] && DRY="DRY RUN (test only, never a result): "
{ echo "${DRY}Phase 7 driver finished $(ts). Winner: $WIN"; echo "Tables: $P7/sweep/tables.md"; echo "Choice: $P7/sweep/choice.md";
  echo "Extra: $P7/sweep/extra.md"; echo "Videos: $P7/sweep/videos/"; echo "Determinism: $P7/checks/determinism.json";
  echo "camera file: $CAM_HOST"; } > "$P7/DONE.txt"
note "driver finished: winner $WIN (see DONE.txt; gate and commit wait for Naz)"
