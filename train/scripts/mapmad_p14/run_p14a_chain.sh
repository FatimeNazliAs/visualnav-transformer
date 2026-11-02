#!/bin/bash
# Phase 14a chained driver (kickoff §2b Part 2, p14a_rules.md). Host side, in screen p14a_chain (start with
# launch_p14a.sh, which also starts the host watcher p14a_chain_watch; the GPU watcher p14a_gpuwatch runs since Part 1).
# Steps:
#   0. pre-flight: committed code, rules sha256 (outputs = repo copy), checkpoint sha256s + set fingerprints (the rules'
#      preflight block);
#   1. 14.0b baseline practice runs (M21b, M21bh, N, F, VF x p13d + p14d, seed 0; every run logs its stop evidence)
#      -> same-p13 check (p13d runs of M21b / M21bh / F / VF = Phase 13's dev runs byte for byte; row 6/7)
#      -> determinism (row 7: one p13d episode x M21b, M21b_a1w4, N_a1w4, twice on one GPU);
#   2. 14.1 stop bench (CPU): score every baseline run under every rule, automatic pick (§4b);
#   3. 14.3 diagnosis (CPU, background, 8 workers) -- waited for before the tables;
#   4. 14.2 Qwen3-VL check: one Qwen server + one worker per usable GPU (no pairs run meanwhile);
#   5. estimate of 14.4 from the baseline seconds per run -> "come back" time;
#   6. 14.4 driving tests (M21b, M21bh, N x A1, W4, A1 + W4) -> stop bench of the switch runs -> driving tables;
#   7. 14.5 mark check: code check, counts (CPU), 8 picked videos (one YOLOE server per shard), video check;
#   8. diagnosis report, Qwen table, fairness, tables + draft G14a -> CHAIN DONE.
# Automatic stops (kickoff §8) -> STOP_reason.txt, STATUS.md STOPPED, wait. EXIT trap: any other exit writes
# STOP_reason.txt. Long steps run in the background (TERM handled within ~1 s). The body is one function, so editing
# this file cannot change a running driver. Resumable: every step skips what is already done.
# DRY runs (never results): P14_CFG=<container DRY config>, P14_DIR=<host p14a_dry folder>, P14_ALLOW_DIRTY=1,
# P14_GPUS="0 1", P14_EPISODES="<ids>" (only these episodes), P14_DIAG_SAMPLE=4 (DAgger drives per set), CL_CHECK_S=30,
# SILENT_S=600.
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P14="${P14_DIR:-$OUT_MOUNT/mapmad/p14a_finish}"
    C="${P14/$OUT_MOUNT//outputs}"
    WHO=chain
    LOG="$P14/driver/chain.log"
    mkdir -p "$P14/driver" "$P14/checks"
    exec 2>> "$LOG"
    # shellcheck source=p14_lib.sh
    source "$REPO/train/scripts/mapmad_p14/p14_lib.sh"
    export P14_OUTPUTS="$OUT_MOUNT/mapmad" P14_EPISODES="${P14_EPISODES:-}"
    FINISHED=0
    BG_PID=""
    DIAG_PID=""
    [[ -f "$P14/STOP_reason.txt" ]] && mv "$P14/STOP_reason.txt" "$P14/driver/STOP_reason_$(date +%m%d_%H%M%S).txt"
    local BASE_BUDGET=$(( 6 * 3600 ))     # 14.0b: <= 6 h GPU; stop > 1.5x
    local DRIVE_BUDGET=$(( 16 * 3600 ))   # 14.4: <= 16 h GPU; stop > 1.5x
    local QWEN_LIMIT=$(( 6 * 3600 ))      # 14.2: <= 4 h GPU; stop > 1.5x
    local VIDEO_LIMIT=$(( 3 * 3600 ))     # 14.5: <= 2 h; stop > 1.5x
    local DIAG_LIMIT=$(( 9 * 3600 / 2 ))  # 14.3: <= 3 h CPU; stop > 1.5x

    stop() {  # reason [question] -> automatic stop (no fix); waits for Naz
        { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "reason: $1"
          echo "question: ${2:-how to go on? (recommended: read the log, fix, relaunch: every step resumes)}"
          echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P14/STOP_reason.txt"
        status step="${STEP:-?}" reason="$1" error="$1" --state STOPPED
        say "AUTOMATIC STOP: $1"
        FINISHED=1
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        [[ -n "${DIAG_PID:-}" ]] && kill "$DIAG_PID" 2>/dev/null
        for pat in "run_search" "sidecar_server" "mapmad_recog.server" "mapmad_recog.itm_server" "p14_qwen_check.py" \
                   "p14_diagnosis.py" "p14_videos.py" "p14_stopbench" "p14_tables"; do
            kill_inside "$pat"; done  # our own processes only (patterns of this chain)
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "exit code: $code"
              echo "question: relaunch launch_p14a.sh after reading driver/chain.log? (recommended: yes, it resumes)"
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P14/STOP_reason.txt"
            status reason="driver exit $code" error="driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    plan() { python3 "$SCRIPTS/p14_plan.py" "$HCFG" "$@"; }
    fallback() { status fallback="$1"; say "FALLBACK: $1"; }
    dxe() { dx "$1" "export P14_EPISODES='$P14_EPISODES' P14_OUTPUTS=/outputs/mapmad; ${*:2}"; }
    dhe() { dh "export P14_EPISODES='$P14_EPISODES'; $*"; }
    step_runs() {  # stage budget -> runs every call of a stage (p14_plan.py), with all automatic stops
        local stage="$1" budget="$2" label subdir set recog seeds arms n eps code
        while IFS='|' read -r label subdir set recog seeds arms n eps; do
            [[ -n "$label" ]] || continue
            STEP="$CURRENT_STEP: $label"
            status step="$STEP" --state RUNNING
            say "$label: $n runs ($subdir, set $set, arms $arms, recogniser $recog)"
            CL_EP_PREFIX="${set}-" CL_EXTRA="${eps:+--episodes $eps}" closed_loop "$label" "$subdir" "$set" "$n" \
                "$budget" "$recog" "$seeds" $arms
            code=$?
            budget=$(( budget - CL_USED_S ))
            (( code == 1 )) && stop "budget: $label exceeded the remaining GPU-running time (> 1.5x)" \
                "go on with a larger budget? (recommended: yes if the rest of the plan fits; else stop the stage)"
            (( code == 2 )) && stop "$label: runners exit without progress (a possible bug; driver/${label}_gpu*.log)"
            (( code == 3 )) && stop "$label: recogniser or ITM server down after 3 restarts ($subdir/recog_restarts)"
            CL_EP_PREFIX="${set}-" crash_rate_ok "$subdir" "$n" $arms || stop "$label: crash rate > 1% ($subdir)"
        done < <(plan calls "$stage")
        STAGE_LEFT=$budget
    }

    # --- 0. pre-flight ---------------------------------------------------------------------------------------------------
    STEP="0 pre-flight"
    status --reset step="$STEP" started="$(date '+%F %T')" mode="see phase14a_log.md" --state RUNNING
    cd "$REPO" || exit 2
    if [[ -z "${P14_ALLOW_DIRTY:-}" && -n "$(git status --porcelain -- train mapmad)" ]]; then
        stop "uncommitted code under train/ or mapmad/ (commit first, §0.8)"
    fi
    COMMIT="$(git rev-parse HEAD)"
    local rules="$OUT_MOUNT/mapmad/p14a_finish/p14a_rules.md" want
    want="$(cut -d' ' -f1 "$OUT_MOUNT/mapmad/p14a_finish/p14a_rules.sha256")"
    [[ "$(sha256sum "$rules" | cut -d' ' -f1)" == "$want" && "$(sha256sum mapmad/configs/p14a_rules.md | cut -d' ' -f1)" == "$want" ]] \
        || stop "p14a_rules.md differs from p14a_rules.sha256 or from the repo copy"
    run_bg python3 "$REPO/train/scripts/mapmad_p11/p11_preflight.py" check "$rules" "$OUT_MOUNT" >> "$LOG" 2>&1 \
        || stop "pre-flight: a checkpoint sha256 or a frozen set fingerprint differs from p14a_rules.md (driver/chain.log)"
    say "chain start: commit $COMMIT, rules sha256 ${want:0:16}; config $CFG${P14_EPISODES:+; DRY episodes $P14_EPISODES}"

    # --- 1. 14.0b baseline -----------------------------------------------------------------------------------------------
    CURRENT_STEP="1 baseline (14.0b)"
    status eta="baseline ~5 h on 6 pairs"
    step_runs baseline $(( BASE_BUDGET * 3 / 2 ))
    say "baseline done ($(( (BASE_BUDGET * 3 / 2 - STAGE_LEFT) / 60 )) min of GPU-running time)"
    STEP="1 same-p13 check"
    status step="$STEP" gpus="" subdir="" arms=""
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_checks same-p13 --config $CFG" >> "$P14/driver/checks.log" 2>&1 \
        || stop "determinism mismatch: p13d baseline runs differ from Phase 13's dev runs (checks/checks.json)" \
                "investigate before any judged step? (recommended: yes; nothing is judged on non-identical runs)"
    say "same-p13: $(tail -n 1 "$P14/driver/checks.log")"
    STEP="1 determinism (row 7)"
    status step="$STEP" --state WAITING
    if [[ ! -f "$P14/checks/det_ok" ]]; then
        local dg label subdir set recog seeds arms n eps
        dg="$(wait_gpu)"
        status gpus="$dg" --state RUNNING
        say "GPU event: GPU $dg for the determinism runs (one pair, both calls on this GPU)"
        while IFS='|' read -r label subdir set recog seeds arms n eps; do
            GPU="$dg" TOKEN="${label}_$(date +%m%d_%H%M%S)" OWNER="$WHO" PORT0=5560 RPORT0=5700 RECOG=yoloe PAIRS_PER_GPU=1 \
                run_limit 3600 "run_search .*--subdir $subdir " bash "$REPO/mapmad/scripts/run_p13_pairs.sh" --config "$CFG" \
                --subdir "$subdir" --set "$set" --arms $arms --seeds 0 --episodes $eps >> "$P14/driver/${label}.log" 2>&1 \
                || stop "determinism runs $label failed (driver/${label}.log)"
        done < <(plan calls det)
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_checks det --config $CFG" >> "$P14/driver/checks.log" 2>&1 \
            || stop "determinism mismatch on the same GPU (checks/checks.json)" "investigate? (recommended: yes)"
        touch "$P14/checks/det_ok"
        say "determinism: $(tail -n 1 "$P14/driver/checks.log")"
    fi

    # --- 2. 14.1 stop bench ------------------------------------------------------------------------------------------------
    STEP="2 stop bench (14.1)"
    status step="$STEP" gpus="" subdir="" arms="" eta="~15 min CPU" --state RUNNING
    run_limit 10800 "p14_stopbench" dxe "" "python -m vint_train.mapmad.closed_loop.p14_stopbench score --config $CFG \
        --subdir runs/baseline --arms M21b M21bh N F VF --workers 16 && python -m vint_train.mapmad.closed_loop.p14_stopbench \
        pick --config $CFG --subdir runs/baseline" >> "$P14/driver/stopbench.log" 2>&1 \
        || stop "stop bench failed (driver/stopbench.log)"
    say "stop bench: $(grep -a 'pick' "$P14/stopbench/stopbench.md" | head -2 | tr '\n' ' ')"

    # --- 3. 14.3 diagnosis (CPU, background) ------------------------------------------------------------------------------
    STEP="3 diagnosis (14.3, background)"
    status step="$STEP"
    dhe "python mapmad/scripts/p14_diagnosis.py practice --config $CFG --subdir runs/baseline --arms M21b M21bh F \
        --workers 8 && python mapmad/scripts/p14_diagnosis.py dagger --config $CFG --workers 8 --sample ${P14_DIAG_SAMPLE:-0}" \
        > "$P14/driver/diagnosis.log" 2>&1 &
    DIAG_PID=$!
    DIAG_T0=$SECONDS
    say "diagnosis started in the background (8 CPU workers)"

    # --- 4. 14.2 Qwen3-VL check ---------------------------------------------------------------------------------------------
    STEP="4 Qwen3-VL check (14.2)"
    status step="$STEP" --state WAITING
    if [[ ! -f "$P14/qwen/done" ]]; then
        local gs="" qg k=0 qpids=() nq
        until [[ -n "$gs" ]]; do gs="$(usable_now)"; [[ -n "$gs" ]] || pause "$CHECK_S"; done
        nq=$(echo $gs | wc -w)
        status gpus="$gs" eta="Qwen check ~1-2 h" --state RUNNING
        : "${QKEY:=$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"
        mkdir -p "$P14/qwen"
        for qg in $gs; do
            say "GPU event: GPU $qg for the Qwen3-VL check (one Qwen server, port $(( 5990 + qg )), one worker)"
            docker exec -d -e MAPMAD_SIDECAR_KEY="$QKEY" -e CUDA_VISIBLE_DEVICES="$qg" -e HOME=/model_cache/home \
                -e HF_HOME=/model_cache/hf -e OMP_NUM_THREADS=4 -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog bash -c \
                "python -m mapmad_recog.server --model qwen3vl --port $(( 5990 + qg )) --text-reuse-out $C/qwen/text_reuse_gpu$qg.json \
                 >> $C/qwen/server_gpu$qg.log 2>&1"
            docker exec -e MAPMAD_SIDECAR_KEY="$QKEY" -e MAPMAD_P13_OWNER="$WHO" -e P14_EPISODES="$P14_EPISODES" \
                -w /app/visualnav-transformer naz_mapmad_habitat python mapmad/scripts/p14_qwen_check.py --config "$CFG" \
                --subdir runs/baseline --arms M21b M21bh N F VF --qwen-port $(( 5990 + qg )) --gpu "$qg" --worker "$k" \
                --workers "$nq" ${P14_EPISODES:+--episodes $P14_EPISODES} >> "$P14/driver/qwen_w$k.log" 2>&1 &
            qpids+=($!)
            k=$(( k + 1 ))
        done
        local qt0=$SECONDS qfail=0
        for p in "${qpids[@]}"; do
            while kill -0 "$p" 2>/dev/null; do
                (( SECONDS - qt0 > QWEN_LIMIT )) && { kill_inside "p14_qwen_check.py"; stop "budget: the Qwen check ran > 1.5 x 4 h"; }
                pause 30
            done
            wait "$p" || qfail=1
        done
        kill_inside "mapmad_recog.server --model qwen3vl"
        if (( qfail )); then
            fallback "14.2: a Qwen check worker failed (driver/qwen_w*.log, qwen/server_gpu*.log): reported as not done"
        else
            touch "$P14/qwen/done"
            say "Qwen check done ($(( (SECONDS - qt0) / 60 )) min)"
        fi
    fi

    # --- 5. estimate of 14.4 -------------------------------------------------------------------------------------------------
    STEP="5 estimate (14.4)"
    status step="$STEP" gpus=""
    local est_s
    est_s="$(python3 - "$P14/runs/baseline/timing" "$(plan total drive)" <<'EOF'
import glob, json, sys
w = {}
for f in glob.glob(sys.argv[1] + "/*.jsonl"):
    arm = f.rsplit("/", 1)[1].split(".")[0]
    w.setdefault(arm, []).extend(json.loads(x)["wall_s"] for x in open(f))
mean = sum(sum(w[a]) / len(w[a]) for a in ("M21b", "M21bh", "N") if w.get(a)) / 3
print(int(mean * int(sys.argv[2]) / 6))
EOF
)" || est_s=$(( 10 * 3600 ))
    (( est_s * 2 > DRIVE_BUDGET * 3 )) && stop "budget: 14.4 estimate $(( est_s / 3600 )) h > 1.5 x 16 h"
    status come_back="$(date -d "+$(( est_s + 5400 )) seconds" '+%a %d %b ~%H:%M')" \
        estimate="14.4 ~$(( est_s / 3600 )) h on 6 pairs + mark check + tables ~1.5 h"
    say "14.4 estimate: $(( est_s / 60 )) min on 6 pairs (baseline seconds per run of M21b, M21bh, N)"

    # --- 6. 14.4 driving tests ----------------------------------------------------------------------------------------------
    CURRENT_STEP="6 driving tests (14.4)"
    step_runs drive $(( DRIVE_BUDGET * 3 / 2 ))
    say "driving tests done ($(( (DRIVE_BUDGET * 3 / 2 - STAGE_LEFT) / 3600 )) h of GPU-running time)"
    STEP="6 driving tables"
    status step="$STEP" gpus="" subdir="" arms="" eta=""
    local sub
    for sub in a1 w4 a1w4; do
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_stopbench score --config $CFG --subdir runs/$sub \
            --arms M21b_$sub M21bh_$sub N_$sub --workers 16" >> "$P14/driver/stopbench.log" 2>&1 \
            || stop "stop bench of runs/$sub failed (driver/stopbench.log)"
    done
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_tables drive --config $CFG" >> "$P14/driver/tables.log" 2>&1 \
        || stop "driving tables failed (driver/tables.log)"

    # --- 7. 14.5 mark check -------------------------------------------------------------------------------------------------
    STEP="7 mark check (14.5)"
    status step="$STEP" --state RUNNING
    local VM="$C/videos_marks"
    run_bg dhe "python mapmad/scripts/p14_videos.py code-check --out $VM && python mapmad/scripts/p14_videos.py counts \
        --config $CFG --subdir runs/baseline --arms M21b M21bh --workers 16 --out $VM && python mapmad/scripts/p14_videos.py \
        pick --config $CFG --subdir runs/baseline --out $VM" >> "$P14/driver/marks.log" 2>&1 || stop "mark counts failed (driver/marks.log)"
    if [[ ! -f "$P14/videos_marks/video_check.json" ]]; then
        local gs="" vg k=0 vpids=() nsh
        status --state WAITING
        until [[ -n "$gs" ]]; do gs="$(usable_now)"; [[ -n "$gs" ]] || pause "$CHECK_S"; done
        nsh=$(echo $gs | wc -w)
        status gpus="$gs" --state RUNNING
        : "${VKEY:=$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"
        for vg in $gs; do  # one YOLOE server PER SHARD (Phase 13 lesson)
            say "GPU event: GPU $vg for the mark-check videos (shard $k/$nsh, YOLOE port $(( 5790 + k )))"
            docker exec -d -e MAPMAD_SIDECAR_KEY="$VKEY" -e CUDA_VISIBLE_DEVICES="$vg" -e HOME=/model_cache/home \
                -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog python -m mapmad_recog.server --model yoloe \
                --port $(( 5790 + k )) --text-reuse-out "$VM/text_reuse_yoloe_shard$k.json"
            docker exec -e MAPMAD_SIDECAR_KEY="$VKEY" -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer \
                naz_mapmad_habitat python mapmad/scripts/p14_videos.py render --config "$CFG" --subdir runs/baseline \
                --gpu "$vg" --recog-port $(( 5790 + k )) --out "$VM" --shard "$k/$nsh" >> "$P14/driver/videos_s$k.log" 2>&1 &
            vpids+=($!)
            k=$(( k + 1 ))
        done
        local vt0=$SECONDS vfail=0
        for p in "${vpids[@]}"; do
            while kill -0 "$p" 2>/dev/null; do
                (( SECONDS - vt0 > VIDEO_LIMIT )) && { kill_inside "p14_videos.py render"; stop "budget: mark-check videos > 1.5 x 2 h"; }
                pause 30
            done
            wait "$p" || vfail=1
        done
        kill_inside "mapmad_recog.server --model yoloe --port 579"
        if (( vfail )); then
            fallback "14.5: video render failed (driver/videos_s*.log): counts only"
        else
            run_bg dh "python mapmad/scripts/p13_videos.py check --out $VM" >> "$P14/driver/marks.log" 2>&1 \
                || fallback "14.5: video check reported a problem (videos_marks/video_check.json)"
        fi
    fi

    # --- 8. tables ----------------------------------------------------------------------------------------------------------
    STEP="8 tables"
    status step="$STEP" gpus="" --state RUNNING
    while kill -0 "$DIAG_PID" 2>/dev/null; do
        (( SECONDS - DIAG_T0 > DIAG_LIMIT )) && { kill_inside "p14_diagnosis.py"; stop "budget: diagnosis ran > 1.5 x 3 h"; }
        pause 30
    done
    wait "$DIAG_PID" || fallback "14.3: diagnosis failed (driver/diagnosis.log): the report covers what finished"
    DIAG_PID=""
    run_bg dhe "python mapmad/scripts/p14_diagnosis.py report --config $CFG" >> "$P14/driver/tables.log" 2>&1 \
        || stop "diagnosis report failed (driver/tables.log)"
    if [[ -f "$P14/qwen/done" ]]; then
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_tables qwen --config $CFG" >> "$P14/driver/tables.log" 2>&1 \
            || stop "Qwen table failed (driver/tables.log)"
    fi
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_checks fairness --config $CFG --stages baseline drive" \
        >> "$P14/driver/checks.log" 2>&1 || fallback "fairness check found problems (checks/checks.json) -- row 6 to judge in Part 3"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14_tables gate --config $CFG" >> "$P14/driver/tables.log" 2>&1 \
        || stop "gate tables failed (driver/tables.log)"
    say "tables written (tables/tables.md); draft G14a: G14a_table_draft.md"
    STEP="8 G14a draft"
    FINISHED=1
    status step="$STEP" note="draft G14a: G14a_table_draft.md; tables/tables.md; videos_marks/" eta="" gpus="" --state "CHAIN DONE"
    date '+%F %T' > "$P14/driver/CHAIN_DONE.txt"
    say "CHAIN DONE"
}

main "$@"
exit $?
