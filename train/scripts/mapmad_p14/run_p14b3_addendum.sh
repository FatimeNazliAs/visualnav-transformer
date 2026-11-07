#!/bin/bash
# Phase 14b-3 addendum: complete arm M21bh (v2.1b, maps hidden, P*) on exam B -- the three calls the budget fallback cut
# (p14b3_rules.md v1.1 §6). Same labels, config, seeds and run folder as the main exam (runs/examb); runs already
# finished there count (resume). Host side, in screen p14b3_addm; state in <p14b3_v3>/addendum (driver/, STATUS.md);
# messages also go to phase14b3_log.md. Steps:
#   0. pre-flight: committed code; the files changed since the exam's commit (3e6f1e7) are logged (none on the run path
#      is expected); exam rules sha256 = the frozen value (key addendum_rules_before);
#   1. contamination re-check of the training splits + round-2 drives (key addendum_contamination);
#   2. a GPU drives addendum runs only after it reproduced one M21b + P* exam-B run of this phase byte-identically
#      (addrepro_g<g>; checked before each call for every GPU free by the server rules); a mismatch -> automatic stop;
#   3. ex_m21bh_s0, s1, s2 (closed_loop); budget 8 GPU-h (wall x GPUs), hard stop at 12 GPU-h (1.5x): the running call
#      and the later ones stop and are reported as unfinished (what finished is reported; no automatic stop);
#   4. fairness (--stages addm, key addendum_fairness), exam rules sha256 after (key addendum_rules_after), the
#      addendum table (tables/addendum_m21bh.md) -> ADDENDUM DONE.
# GPU rule: the server rules of the kickoff (watcher: 4 checks x 5 min + no other user's process in the last 60 min, live
# check right before; a GPU that gets another user's process is withdrawn).
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    local MAIN="$OUT_MOUNT/mapmad/p14b3_v3"
    P14="${P14_DIR:-$MAIN/addendum}"
    C="${P14/$OUT_MOUNT//outputs}"
    local CM="${MAIN/$OUT_MOUNT//outputs}"
    WHO=chain
    LOG="$P14/driver/chain.log"
    mkdir -p "$P14/driver"
    exec 2>> "$LOG"
    export P14_LOGNAME=../phase14b3_log.md P14_PHASE="14b-3 addendum (M21bh)"
    export P14_PASTE="continue the Phase 14b-3 addendum: read addendum/STATUS.md and its chain log, then write the Addendum section, update STATUS.md and the gate footnote, and commit"
    export P14_CFG="${P14_CFG:-/app/visualnav-transformer/mapmad/configs/p14b3_v3.yaml}"
    # shellcheck source=p14_lib.sh
    source "$REPO/train/scripts/mapmad_p14/p14_lib.sh"
    export P14_OUTPUTS="$OUT_MOUNT/mapmad" P14_EPISODES="" CL_UNIT_DIV=1  # budgets: wall x GPUs (as the main chain)
    export P14_GPU_FREE="${P14_GPU_FREE:-$MAIN/driver/gpu_free.json}"
    export RECOG_CONFIG=/outputs/mapmad/p14b2_recog/models.yaml
    export PAIRS_PER_GPU="${P14B3_PAIRS:-4}"
    local BUDGET_S=$(( 8 * 3600 )) HARD_S=$(( 12 * 3600 ))
    FINISHED=0
    BG_PID=""
    REPRO_OK=""
    REPRO_TRIED=""
    [[ -f "$P14/STOP_reason.txt" ]] && mv "$P14/STOP_reason.txt" "$P14/driver/STOP_reason_$(date +%m%d_%H%M%S).txt"

    stop() {  # reason [question] -> automatic stop; waits for Naz
        { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "reason: $1"
          echo "question: ${2:-how to go on? (recommended: read the log, fix, relaunch: every step resumes)}"
          echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P14/STOP_reason.txt"
        status step="${STEP:-?}" reason="$1" error="$1" --state STOPPED
        say "ADDENDUM AUTOMATIC STOP: $1"
        FINISHED=1
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        for pat in "run_search .*--subdir runs/examb " "run_search .*--subdir checks/addrepro" "mapmad_recog.server" \
                   "sidecar_server" "p14b3_checks" "p14b3_tables"; do
            kill_inside "$pat"; done  # our own processes only (patterns of this driver)
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "exit code: $code"
              echo "question: relaunch launch_p14b3_addendum.sh after reading addendum/driver/chain.log? (recommended: yes, it resumes)"
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P14/STOP_reason.txt"
            status reason="driver exit $code" error="driver exit $code" --state STOPPED
            say "ADDENDUM STOP (exit $code): see addendum/STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    plan() { python3 "$SCRIPTS/p14_plan.py" "$HCFG" "$@"; }
    dxe() { dx "$1" "export P14_EPISODES='' P14_OUTPUTS=/outputs/mapmad; ${*:2}"; }
    dh_g() { docker exec -e MAPMAD_P13_OWNER="$WHO" -e P14_EPISODES="" -w /app/visualnav-transformer naz_mapmad_habitat bash -lc "$*"; }
    gpu_used() { echo "GPU-time used $(( $1 / 60 )) min (budget $(( BUDGET_S / 60 )), hard stop $(( HARD_S / 60 )))"; }
    owners() {  # owner table + one decision line (kickoff §5) before a launch
        { echo "### $(date '+%m-%d %H:%M') owner check before: $1"; nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
          free -h | sed -n 2p; uptime; docker ps --format '{{.Names}}'; ps -eo user --no-headers | sort | uniq -c | tr '\n' ' '; echo; } \
          >> "$P14/driver/owners.log" 2>&1
        say "owner check before $1: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) GPU processes; decision: GPUs free by the server rules (20-min watcher + 60-min rule) AND reproduced: '${REPRO_OK:-none}', $PAIRS_PER_GPU pairs per GPU"
    }
    # closed_loop takes only GPUs that passed the reproduction check (and are free by the watcher, as always)
    free_for() { local g; for g in $(place_free free); do [[ " $REPRO_OK " == *" $g "* ]] && echo "$g"; done; }

    repro_new_gpus() {  # every GPU free by the server rules and not yet tried: reproduce one M21b + P* run on it first
        local g label subdir set recog seeds arms_s n eps
        for g in $(place_free free); do
            [[ " $REPRO_TRIED " == *" $g "* ]] && continue
            gpu_clean "$g" || continue
            IFS='|' read -r label subdir set recog seeds arms_s n eps < <(plan calls addrepro | grep "^addrepro_g$g|")
            [[ -n "$eps" ]] || { say "addendum: no M21b + P* reference run on GPU $g -> GPU $g not used"; REPRO_TRIED+=" $g"; continue; }
            STEP="2 reproduce one M21b + P* exam-B run on GPU $g"
            status step="$STEP" --state RUNNING
            owners "addendum reproduction on GPU $g"
            say "GPU event: GPU $g taken for the reproduction check ($eps, M21b_w4_d_am seed 0)"
            GPU="$g" TOKEN="${label}_$(date +%m%d_%H%M%S)" OWNER="$WHO" PORT0=5560 RPORT0=5700 RECOG=yoloe_ft PAIRS_PER_GPU=1 \
                run_limit 3600 "run_search .*--subdir $subdir " bash "$REPO/mapmad/scripts/run_p13_pairs.sh" --config "$CFG" \
                --subdir "$subdir" --set "$set" --arms $arms_s --seeds 0 --episodes $eps >> "$P14/driver/${label}.log" 2>&1 \
                || stop "reproduction run on GPU $g failed (addendum/driver/${label}.log)"
            run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks repro --config $CFG --gpu $g" \
                >> "$P14/driver/checks.log" 2>&1 \
                || stop "GPU $g: the M21b + P* exam-B run is NOT reproduced byte-identically (checks/checks.json addendum_repro_g$g)" \
                        "investigate before any addendum run? (recommended: yes; no M21bh run was made on GPU $g)"
            say "addendum: GPU $g reproduced the M21b + P* run byte-identically -> GPU $g may drive M21bh"
            REPRO_TRIED+=" $g"; REPRO_OK+=" $g"
        done
    }

    # --- 0. pre-flight ----------------------------------------------------------------------------------------------
    STEP="0 pre-flight"
    status --reset step="$STEP" --state RUNNING
    say "ADDENDUM start (M21bh completion; P14=$P14, CFG=$CFG, pairs/GPU $PAIRS_PER_GPU, budget 8 GPU-h, hard stop 12)"
    if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
        stop "uncommitted changes in the repo (the runs must name one commit)"
    fi
    say "code at $(git -C "$REPO" rev-parse HEAD); files changed since the exam's commit 3e6f1e7: $(git -C "$REPO" diff --name-only 3e6f1e7 HEAD | tr '\n' ' ')"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks rules --config $CFG --tag before_addendum \
        --key addendum_rules_before --repo-copy /app/visualnav-transformer/mapmad/configs/p14b3_rules.md" >> "$P14/driver/checks.log" 2>&1 \
        || stop "exam rules sha256 differs from the frozen value"

    # --- 1. contamination re-check ------------------------------------------------------------------------------------
    STEP="1 contamination re-check"
    status step="$STEP"
    local sp="/mapmad_data/splits"
    run_bg dh_g "python mapmad/scripts/make_p14b3_sets.py contamination --outputs /outputs/mapmad --out-dir $CM/checks/addm \
        --drives /mapmad_data/habitat_mapmad_v3_dagger_r2map /mapmad_data/habitat_mapmad_v3_dagger_r2hidden \
        $sp/habitat_mapmad_v2/train/traj_names.txt $sp/habitat_mapmad_v2_dagger/train/traj_names.txt \
        $sp/habitat_mapmad_v2_dagger_hidden/train/traj_names.txt $sp/habitat_mapmad_v3_dagger_r2map/train/traj_names.txt \
        $sp/habitat_mapmad_v3_dagger_r2hidden/train/traj_names.txt" >> "$P14/driver/checks.log" 2>&1 \
        || stop "CONTAMINATION re-check failed (checks/addm/checks/contamination.json)"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks contamination --config $CFG --dir addm/checks \
        --key addendum_contamination" >> "$P14/driver/checks.log" 2>&1 || stop "contamination merge failed"

    # --- 2 + 3. reproduction per GPU, then the three M21bh calls -------------------------------------------------------
    local used=0 cut=() over=0 label subdir set recog seeds arms_s n eps code
    [[ -f "$P14/driver/used_s" ]] && used="$(cat "$P14/driver/used_s")"
    while IFS='|' read -r label subdir set recog seeds arms_s n eps; do
        if (( over )); then cut+=("$label"); continue; fi
        STEP="3 M21bh: $label"
        status step="$STEP" --state WAITING budget="$(gpu_used "$used")"
        until repro_new_gpus; [[ -n "${REPRO_OK// /}" ]]; do pause "$CHECK_S"; done  # new free GPUs join between calls
        STEP="3 M21bh: $label"
        status step="$STEP" --state RUNNING budget="$(gpu_used "$used")"
        owners "addendum $label"
        CL_PREFIX="../" CL_CFG="$CFG" CL_EP_PREFIX="p13b-" CL_SEEDS="$seeds" \
            closed_loop "$label" "$subdir" "$set" "$n" "$(( HARD_S - used ))" yoloe_ft "$seeds" $arms_s
        code=$?
        used=$(( used + CL_USED_S ))
        echo "$used" > "$P14/driver/used_s"
        (( used > BUDGET_S )) && say "addendum: over the 8 GPU-h budget ($(gpu_used "$used")); continuing up to the hard stop"
        if (( code == 1 )); then
            over=1; cut+=("$label")
            say "addendum: HARD STOP at 12 GPU-h during $label -> it and every later call are reported as unfinished"
            continue
        fi
        (( code == 2 )) && stop "$label: runners exit without progress (addendum/driver/${label}_gpu*.log)"
        (( code == 3 )) && stop "$label: recogniser down after 3 restarts"
        CL_EP_PREFIX="p13b-" CL_SEEDS="$seeds" crash_rate_ok "../$subdir" "$n" $arms_s || stop "$label: crash rate > 1%"
    done < <(plan calls addm)
    python3 -c "import json,sys; json.dump(sys.argv[2:], open(sys.argv[1], 'w'))" "$P14/driver/cut.json" "${cut[@]}"
    say "addendum M21bh calls done ($(gpu_used "$used")); unfinished: ${cut[*]:-none}"

    # --- 4. checks, table ------------------------------------------------------------------------------------------------
    STEP="4 checks + table"
    status step="$STEP" gpus="" --state RUNNING
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks fairness --config $CFG --stages addm \
        --cut $C/driver/cut.json --key addendum_fairness" >> "$P14/driver/checks.log" 2>&1 \
        || stop "addendum fairness check failed (checks/checks.json addendum_fairness)"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks rules --config $CFG --tag after_addendum \
        --key addendum_rules_after --repo-copy /app/visualnav-transformer/mapmad/configs/p14b3_rules.md" >> "$P14/driver/checks.log" 2>&1 \
        || stop "exam rules sha256 changed during the addendum"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_tables addendum --config $CFG --workers ${P14B3_WORKERS:-8}" \
        >> "$P14/driver/tables.log" 2>&1 || stop "addendum table failed (addendum/driver/tables.log)"
    say "addendum table: $(grep -m1 '^| MapMaD v2.1b, maps shown + P\* −' "$MAIN/tables/addendum_m21bh.md")"
    date '+%F %T' > "$P14/driver/ADDENDUM_DONE.txt"
    status step="done" --state "ADDENDUM DONE"
    say "ADDENDUM DONE"
    FINISHED=1
}

main "$@"
