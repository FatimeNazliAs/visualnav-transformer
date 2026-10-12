#!/bin/bash
# Phase 5 unattended driver (Naz, 2026-10-11): everything after S-dev up to the G5 table, in order, without this
# chat. Run on the host inside screen (no sudo, our containers only):
#
#   screen -dmS mapmad_p5_rest bash mapmad/scripts/run_p5_rest.sh
#
# Steps (each leaves <p5>/driver/<step>.done; a re-launch skips finished steps, the runners skip finished runs):
#   snapshot    the exact code (HEAD + tracked.patch + untracked.tar, sha256) -> <p5>/code_snapshot/
#   dev_1a      dev goal photos re-chosen with rule 1a -> dev/set_1a; arms B, F on all 99 dev episodes (seed 0,
#               report only) -> runs/dev_1a, tables dev/tables_1a
#   rules       p5_rules.md + p5_rules.sha256, written before the exam set is drawn (rules version 3)
#   exam_set    candidates (8 workers, resumable) + freeze -> exam_set/ (build_report.md, goal_photos_sheet.png)
#   set_facts   p5_rules_set.md + sha256 (+ p5_rules_choices.json): fingerprint, video and determinism episodes
#   sfreeze     Naz's pre-approved S-freeze conditions -> exam_set/sfreeze.md; any FAIL stops here
#   exam        arms A-F x 3 seeds on 6 pairs (3 per GPU) -> runs/
#   determinism the pre-registered C episode again -> checks/determinism_exam/determinism.json
#   tables      exam_tables + fairness (gate row 1) -> tables/
#   figures     exam_figures -> figures/
#   videos      8 seeded + 4 picked by rule -> videos/
#   gate        exam_gate -> tables/gate.md
# A suspected bug (block items 13, 13b: a runner stopping on a non-infrastructure error, > 2% of episodes removed
# for crashes, fairness or determinism failing, code changed since the snapshot, a GPU used by another user)
# writes <p5>/STOP_reason.txt and stops; nothing is fixed or rerun. Never commits.
# Paths: the outputs folder is read from the naz_mapmad_habitat /outputs mount and the config's out_dir.
# Test settings (environment): P5_CONFIG (default p5_exam.yaml; p5_dryrun.yaml = 3 dev episodes, short runs),
# P5_PAIRS (default "0 0 0 1 1 1"), P5_SKIP (steps to skip, e.g. "dev_1a exam_set").
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HABITAT=naz_mapmad_habitat
NOMAD=naz_mapmad
APP=/app/visualnav-transformer
CFG_EXAM=$APP/mapmad/configs/${P5_CONFIG:-p5_exam.yaml}
CFG_DEV1A=$APP/mapmad/configs/p5_dev_1a.yaml
OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' "$HABITAT")"
P5C="$(docker exec "$HABITAT" python -c "from mapmad_sim.run_layout import RunLayout; print(RunLayout.load('$CFG_EXAM').base)")"
P5="$OUT_MOUNT${P5C#/outputs}"   # host side of the container's <outputs>/<out_dir>
[[ -d "$P5" ]] || { echo "outputs folder $P5 not found" >&2; exit 2; }
PAIRS6="${P5_PAIRS:-0 0 0 1 1 1}"
NPAIRS=$(wc -w <<< "$PAIRS6")
WORKERS=8
DRV="$P5/driver"
PHASE_LOG="$P5/phase5_log.md"
mkdir -p "$DRV"
exec > >(tee -a "$DRV/run_p5_rest.log") 2>&1   # driver log (screen shows the same)

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" >&2; }  # stderr: never into a redirected output file
note() { say "$*"; echo "- [driver $(ts)] $*" >> "$PHASE_LOG"; }  # also into the phase command log
run() { note "+ $*"; "$@"; }

stop() {
    local reason="$1"
    say "STOP: $reason"
    { echo "STOP $(ts)"; echo "$reason"; echo; echo "Driver log: $DRV/run_p5_rest.log. Nothing was fixed or rerun."; } \
        > "$P5/STOP_reason.txt"
    echo "- [driver $(ts)] STOP: $reason" >> "$PHASE_LOG"
    docker exec "$HABITAT" pkill -f "mapmad_sim.sidecar_server" 2>/dev/null || true  # our own servers only
    exit 1
}

done_step() { [[ -f "$DRV/$1.done" || " ${P5_SKIP:-} " == *" $1 "* ]]; }
mark() { date -Is > "$DRV/$1.done"; note "step $1 done"; }

nomad() {  # run a python module in the NoMaD container
    docker exec -e MPLCONFIGDIR=/tmp/mpl -e PYTHONPATH=$APP/mapmad/src -w $APP/train "$NOMAD" python -m "$@"
}
habitat() { docker exec -w $APP "$HABITAT" python "$@"; }

gpus_free() {
    # nvidia-smi first (CLAUDE.md): every GPU process other than the display server must be ours
    nvidia-smi >> "$DRV/nvidia_smi.log" 2>&1
    local me bad=0 pid user name
    me="$(id -un)"
    while read -r pid; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        name="$(ps -o comm= -p "$pid" 2>/dev/null)"
        user="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
        case "$name" in Xorg|gnome-shell) continue ;; esac
        if [[ -n "$user" && "$user" != "$me" ]]; then
            say "GPU process $pid ($name) belongs to $user"
            bad=1
        fi
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader; nvidia-smi | awk '$3 == "N/A" && $4 == "N/A" {print $5}')
    return $bad
}
need_gpus() { gpus_free || stop "a GPU is used by another user's job before step $1 (see $DRV/nvidia_smi.log)"; }

code_sha() {  # the current tracked diff + untracked file contents, one hash
    { git -C "$REPO" diff HEAD --binary
      git -C "$REPO" ls-files --others --exclude-standard | grep -v '^CLAUDE.md$' | sort | while read -r f; do
          echo "== $f"; cat "$REPO/$f"; done; } | sha256sum | cut -d' ' -f1
}
check_code() { [[ "$(code_sha)" == "$(cat "$P5/code_snapshot/CODE_SHA")" ]] || stop "code changed since the snapshot (before $1)"; }

layout_py() {  # print something from the run layout of <subdir> (mapmad_sim.run_layout owns run ids and file names)
    habitat -c "from pathlib import Path; from mapmad_sim.run_layout import RunLayout, run_id
L = RunLayout.load(Path('$CFG_EXAM'), subdir='$1'); print($2)"
}
host() { echo "$OUT_MOUNT${1#/outputs}"; }  # container path -> host path

crash_share_ok() {  # block item 13b: > 2% of the episodes removed = a bug
    local sub="$1" n="$2" crashed
    crashed=$(layout_py "$sub" "len(L.crashed_episodes())")
    note "crashed episodes in $sub: $crashed of $n"
    (( crashed * 100 <= 2 * n ))
}

[[ -f "$P5/STOP_reason.txt" ]] && { say "STOP_reason.txt exists: review it and move it away before re-launching"; exit 1; }
note "driver start (pid $$, repo $REPO, outputs $P5)"

# --- snapshot ------------------------------------------------------------------------------------------------------
if ! done_step snapshot; then
    S="$P5/code_snapshot"
    mkdir -p "$S"
    git -C "$REPO" rev-parse HEAD > "$S/COMMIT"
    git -C "$REPO" status --porcelain > "$S/STATUS"
    run git -C "$REPO" diff HEAD --binary > "$S/tracked.patch"
    git -C "$REPO" ls-files --others --exclude-standard | grep -v '^CLAUDE.md$' | sort > "$S/untracked.txt"
    run tar -C "$REPO" -cf "$S/untracked.tar" -T "$S/untracked.txt"
    (cd "$S" && sha256sum tracked.patch untracked.tar > SHA256SUMS)
    code_sha > "$S/CODE_SHA"
    mark snapshot
fi
check_code dev_1a

# --- dev B and F with the 1a photos (report only) ------------------------------------------------------------------
if ! done_step dev_1a; then
    if [[ ! -f "$P5/dev/set_1a/episodes.json" ]]; then
        need_gpus rephoto
        run habitat mapmad/scripts/make_p5_sets.py rephoto --set dev --out dev/set_1a --gpu 1 || stop "rephoto failed"
    fi
    need_gpus dev_1a
    PAIRS="$PAIRS6" BASE_PORT=5560 run "$REPO/mapmad/scripts/run_p5.sh" --config "$CFG_DEV1A" --arms B F --seeds 0 \
        --subdir runs/dev_1a || stop "dev B/F runner exited with an error (block item 13): see runs/dev_1a"
    crash_share_ok runs/dev_1a 99 || stop "dev B/F: more than 2% of episodes crashed (block item 13b)"
    # same 99 episodes: the other dev arms' runs (photo fields do not reach them) sit next to B and F for the tables
    for a in A C_wp1 D E; do
        for d in logs summaries; do
            [[ -e "$P5/runs/dev_1a/$d/$a" ]] || ln -s "../../dev/$d/$a" "$P5/runs/dev_1a/$d/$a"
        done
    done
    run nomad vint_train.mapmad.closed_loop.exam_tables --config "$CFG_DEV1A" --subdir runs/dev_1a --out dev/tables_1a \
        --arms A B C_wp1 D E F --paired C_wp1-B C_wp1-F F-B C_wp1-D D-A C_wp1-E --title "Phase 5 dev (1a photos)" \
        > "$DRV/dev_tables_1a.out" 2>&1 || stop "dev tables failed"
    mark dev_1a
fi

# --- rules (version 3: written before the exam set is drawn; Naz 2026-10-12) ---------------------------------------
if ! done_step rules; then
    [[ -f "$P5/p5_rules.md" ]] || run habitat mapmad/scripts/p5_freeze.py rules --config "$CFG_EXAM" || stop "writing p5_rules.md failed"
    mark rules
fi

# --- exam set ------------------------------------------------------------------------------------------------------
if ! done_step exam_set; then
    if [[ ! -f "$P5/exam_set/episodes.json" ]]; then
        need_gpus exam_set
        pids=()
        for ((w = 0; w < WORKERS; w++)); do
            run habitat mapmad/scripts/make_p5_sets.py candidates --config "$CFG_EXAM" --set exam --gpu $((w % 2)) --worker $w \
                --workers $WORKERS > "$DRV/candidates_w$w.log" 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid" || stop "exam-set candidates worker failed (see $DRV/candidates_w*.log)"; done
        run habitat mapmad/scripts/make_p5_sets.py freeze --config "$CFG_EXAM" --set exam > "$DRV/freeze.log" 2>&1 \
            || stop "exam-set freeze failed (see $DRV/freeze.log)"
    fi
    mark exam_set
fi

# --- set values + S-freeze ----------------------------------------------------------------------------------------
if ! done_step set_facts; then
    [[ -f "$P5/p5_rules_set.md" ]] || run habitat mapmad/scripts/p5_freeze.py set-facts --config "$CFG_EXAM" \
        --pairs "$NPAIRS" || stop "writing p5_rules_set.md failed"
    mark set_facts
fi
if ! done_step sfreeze; then
    free=1
    gpus_free || free=0
    run habitat mapmad/scripts/p5_freeze.py check --config "$CFG_EXAM" --pairs "$NPAIRS" --gpus-free $free > "$DRV/sfreeze.out" 2>&1
    rc=$?
    cat "$DRV/sfreeze.out"
    [[ $rc -eq 0 ]] || stop "an S-freeze condition failed: see exam_set/sfreeze.md (exit $rc)"
    mark sfreeze
fi

# --- exam runs -----------------------------------------------------------------------------------------------------
N=$(python3 -c "import json; print(len(json.load(open('$P5/exam_set/episodes.json'))['episodes']))")
if ! done_step exam; then
    check_code exam
    need_gpus exam
    PAIRS="$PAIRS6" BASE_PORT=5560 run "$REPO/mapmad/scripts/run_p5.sh" --config "$CFG_EXAM" --arms A B C D E F \
        --subdir runs || stop "an exam runner exited with an error (block item 13): see runs/progress.*.log and $DRV/run_p5_rest.log"
    crash_share_ok runs "$N" || stop "more than 2% of exam episodes removed for crashes (block item 13b)"
    mark exam
fi

# --- determinism (gate row 6) --------------------------------------------------------------------------------------
if ! done_step determinism; then
    check_code determinism
    need_gpus determinism
    DET=$(python3 -c "import json; print(json.load(open('$P5/p5_rules_choices.json'))['determinism_episode'])")
    PAIRS="0" BASE_PORT=5560 run "$REPO/mapmad/scripts/run_p5.sh" --config "$CFG_EXAM" --arms C --seeds 0 \
        --episodes "$DET" --subdir checks/determinism_exam || stop "determinism re-run failed"
    a="$(host "$(layout_py runs "L.log('C', run_id('$DET', 0))")")"
    b="$(host "$(layout_py checks/determinism_exam "L.log('C', run_id('$DET', 0))")")"
    if [[ -f "$a" && -f "$b" ]] && cmp -s "$a" "$b"; then pass=true; else pass=false; fi
    sa=$(sha256sum "$a" 2>/dev/null | cut -d' ' -f1); sb=$(sha256sum "$b" 2>/dev/null | cut -d' ' -f1)
    echo "{\"pass\": $pass, \"episode\": \"$DET\", \"seed_index\": 0, \"exam_log_sha256\": \"$sa\", \"rerun_log_sha256\": \"$sb\", \"note\": \"C $DET seed 0: $([[ $pass == true ]] && echo byte-identical || echo DIFFERENT)\"}" \
        > "$P5/checks/determinism_exam/determinism.json"
    note "determinism: $pass ($DET; $sa vs $sb)"
    [[ $pass == true ]] || stop "determinism check failed: $DET C seed 0 log differs on re-run (gate row 6)"
    mark determinism
fi

# --- tables + fairness ---------------------------------------------------------------------------------------------
if ! done_step tables; then
    run nomad vint_train.mapmad.closed_loop.exam_tables --config "$CFG_EXAM" --subdir runs --out tables \
        --title "Phase 5 exam" > "$DRV/tables.out" 2>&1 || stop "exam_tables failed (see $DRV/tables.out)"
    run nomad vint_train.mapmad.closed_loop.exam_gate --config "$CFG_EXAM" --subdir runs --tables tables \
        > "$DRV/gate_pre.out" 2>&1 || stop "exam_gate failed (see $DRV/gate_pre.out)"
    fair=$(python3 -c "import json; print(json.load(open('$P5/tables/gate.json'))['fairness']['pass'])")
    [[ "$fair" == "True" ]] || stop "fairness (gate row 1) failed: see tables/gate.md"
    mark tables
fi

# --- figures -------------------------------------------------------------------------------------------------------
if ! done_step figures; then
    run nomad vint_train.mapmad.closed_loop.exam_figures --tables $P5C/tables --out $P5C/figures \
        > "$DRV/figures.out" 2>&1 || note "WARNING: figures failed (see $DRV/figures.out); continuing"
    mark figures
fi

# --- videos --------------------------------------------------------------------------------------------------------
if ! done_step videos; then
    need_gpus videos
    habitat mapmad/scripts/p5_freeze.py pick-videos --config "$CFG_EXAM" --tables tables > "$DRV/video_list.tsv" 2> "$DRV/video_list.err" \
        || note "WARNING: picking the video episodes failed (see $DRV/video_list.err); rendering what was listed"
    i=0
    vpids=()
    while IFS=$'\t' read -r eid label; do
        i=$((i + 1))
        name=$(printf "%02d_%s" "$i" "$eid")
        [[ -f "$P5/videos/$name.mp4" ]] && continue
        run habitat mapmad/scripts/p5_videos.py --config "$CFG_EXAM" --subdir runs --episode "$eid" \
            --seed-index 0 --arms A B C D E F --label "$label" --out "videos/$name.mp4" --gpu $((i % 2)) \
            > "$DRV/video_$name.log" 2>&1 &
        vpids+=($!)
        if (( ${#vpids[@]} >= 4 )); then wait "${vpids[0]}" || note "WARNING: video failed (see $DRV/video_*.log)"; vpids=("${vpids[@]:1}"); fi
    done < "$DRV/video_list.tsv"
    for pid in "${vpids[@]}"; do wait "$pid" || note "WARNING: video failed (see $DRV/video_*.log)"; done
    cp "$DRV/video_list.tsv" "$P5/videos/video_list.tsv"
    mark videos
fi

# --- gate ----------------------------------------------------------------------------------------------------------
run nomad vint_train.mapmad.closed_loop.exam_gate --config "$CFG_EXAM" --subdir runs --tables tables \
    > "$DRV/gate.out" 2>&1 || stop "exam_gate failed (see $DRV/gate.out)"
cat "$DRV/gate.out"
mark gate
{ echo "Phase 5 driver finished $(ts)."; echo "G5 table: $P5/tables/gate.md"; echo "Tables: $P5/tables/tables.md";
  echo "Figures: $P5/figures/"; echo "Videos: $P5/videos/"; echo "S-freeze: $P5/exam_set/sfreeze.md";
  echo "Dev B/F with 1a photos: $P5/dev/tables_1a/tables.md"; } > "$P5/DONE.txt"
note "driver finished: see DONE.txt (phase-end steps wait for Naz)"
