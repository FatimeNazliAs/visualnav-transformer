#!/bin/bash
# Phase 12 chained driver (kickoff §3 item 12.4, p12_rules.md). Host side, in screen p12_chain (start with
# launch_p12.sh, which also starts the GPU watcher p12_gpuwatch and the host watcher p12_chain_watch). Steps:
#   0. pre-flight: committed code, rules sha256 (outputs = repo copy), checkpoint sha256s + dev-set fingerprint against
#      the rules' preflight block; F check (F_truth = the Phase 8 teacher on 2 Phase 8 dev drives; else STOP);
#   1. dev runs: v2 + v2.1 maps hidden (3 seeds, sim truth); refactor check (seed 0 of v2 maps hidden = Phase 11's log);
#      headline (block item 1) -> P12_DEV_CHOICES.json (headline); headline + YOLOE (3 seeds); determinism episode
#      (headline + YOLOE, twice, same GPU, cmp);
#   2. tau (block item 4) -> P12_DEV_CHOICES.json (headline, deltas, tau);
#   3. exam sets: candidates (exam + OVON, rendering on free GPUs) -> freeze (a bin < 40 -> STOP) -> fingerprints;
#   4. 20-run timing (the first runs of the headline pair) -> estimate -> trims (block item 9 order) -> budget stop if
#      still > 1.5 x 48 h (6 pairs; 96 h when only one GPU is usable);
#   5. main exam: headline pair -> N -> V1 -> F -> other pair -> truth arm -> N-fish (headline interim in STATUS.md);
#   6. OVON; 7. exploration; 8. unseeable; 9. Qwen3-VL subset (own GPU, 1 pair per GPU); 10. scores + fairness +
#   tables; 11. videos (pick + render + small copies); 12. G12 draft -> CHAIN DONE.
# Automatic stops (block item 9) -> STOP_reason.txt, STATUS.md STOPPED, fix nothing. EXIT trap: any other exit writes
# STOP_reason.txt with the exit code and the last log line. Long steps run in the background (TERM handled within ~1 s);
# a silent step (no run finished and no log written for SILENT_S) is killed and restarted. The body is one function, so
# editing this file cannot change a running driver. Resumable: every step skips what is already done.
# DRY runs (never results): P12_CFG=<container path of a DRY config> P12_DIR=<host DRY folder> P12_ALLOW_DIRTY=1
# (uncommitted code allowed, rules not checked) P12_GPUS="0 1" (no free-check wait) CL_CHECK_S=30 SILENT_S=600.
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P12="${P12_DIR:-$OUT_MOUNT/mapmad/p12_search_exam}"
    C="${P12/$OUT_MOUNT//outputs}"
    WHO=chain
    LOG="$P12/driver/chain.log"
    mkdir -p "$P12/driver" "$P12/checks"
    exec 2>> "$LOG"
    # shellcheck source=p12_lib.sh
    source "$REPO/train/scripts/mapmad_p12/p12_lib.sh"
    FINISHED=0
    BG_PID=""
    [[ -f "$P12/STOP_reason.txt" ]] && mv "$P12/STOP_reason.txt" "$P12/driver/STOP_reason_$(date +%m%d_%H%M%S).txt"  # kept
    local DEV_BUDGET=$(( 12 * 3600 ))        # dev runs: 1.5 x 8 h of GPU-running time (logged decision)
    local EXAM_BUDGET=$(( 48 * 3600 ))       # §8: <= 48 h on 6 pairs (both GPUs); stop > 1.5x
    local EXAM_USED=0

    stop() {  # reason -> automatic stop (no fix)
        { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "reason: $1"
          echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P12/STOP_reason.txt"
        status step="${STEP:-?}" reason="$1" --state STOPPED
        say "AUTOMATIC STOP: $1"
        FINISHED=1
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        for pat in "run_search" "sidecar_server" "mapmad_recog.server" "make_p12_sets.py" "p12_f_check.py" "p12_videos.py"; do
            kill_inside "$pat"; done  # our own processes only (patterns of this chain)
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "exit code: $code"
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P12/STOP_reason.txt"
            status reason="driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    step_runs() {  # stage budget_var(name) -> runs every call of a stage (p12_plan.py), with all automatic stops
        local stage="$1" budget="$2" line label subdir set recog seeds arms n code
        while IFS='|' read -r label subdir set recog seeds arms n; do
            [[ -n "$label" ]] || continue
            STEP="$CURRENT_STEP: $label"
            status step="$STEP" --state RUNNING
            say "$label: $n runs ($subdir, set $set, arms $arms, recogniser $recog${seeds:+, seeds $seeds})"
            closed_loop "$label" "$subdir" "$set" "$n" "$budget" "$recog" "$seeds" $arms
            code=$?
            STEP_USED=$(( ${STEP_USED:-0} + CL_USED_S ))
            budget=$(( budget - CL_USED_S ))
            (( code == 1 )) && stop "budget: $label exceeded the remaining GPU-running time"
            (( code == 2 )) && stop "$label: runners exit without progress (a possible bug; driver/${label}_gpu*.log)"
            (( code == 3 )) && stop "$label: recogniser sidecar down after 3 restarts ($subdir/recog_restarts)"
            crash_rate_ok "$subdir" "$n" $arms || stop "$label: crash rate > 1% ($subdir)"
            [[ "$label" == exam_pair ]] && headline_interim
        done < <(python3 "$SCRIPTS/p12_plan.py" "$HCFG" calls "$stage")
        STAGE_LEFT=$budget
    }
    headline_interim() {  # headline pair numbers -> driver/headline_interim.txt (STATUS.md)
        run_bg dx "" "python -m vint_train.mapmad.closed_loop.p12_score score --config $CFG --subdir runs/exam --set exam \
            --arms $(python3 "$SCRIPTS/p12_plan.py" "$HCFG" pair) && \
            python -m vint_train.mapmad.closed_loop.p12_tables --config $CFG --out tables_interim" >> "$P12/driver/interim.log" 2>&1 \
            && grep -a -E '^\| (M2|M2h|M21|M21h) |headline' "$P12/tables_interim/tables.md" | head -4 | tr '\n' ' ' \
            > "$P12/driver/headline_interim.txt"
        status note="headline pair interim numbers in STATUS.md"
    }

    # --- 0. pre-flight ------------------------------------------------------------------------------------------------
    STEP="0 pre-flight"
    status --reset step="$STEP" started="$(date '+%F %T')" --state RUNNING
    cd "$REPO" || exit 2
    if [[ -z "${P12_ALLOW_DIRTY:-}" && -n "$(git status --porcelain -- train mapmad)" ]]; then
        stop "uncommitted code under train/ or mapmad/ (commit first, §0.8)"
    fi
    COMMIT="$(git rev-parse HEAD)"
    local rules="$P12/p12_rules.md" want="DRY"
    if [[ -z "${P12_ALLOW_DIRTY:-}" ]]; then
        want="$(cut -d' ' -f1 "$P12/p12_rules.sha256")"
        [[ "$(sha256sum "$rules" | cut -d' ' -f1)" == "$want" && "$(sha256sum mapmad/configs/p12_rules.md | cut -d' ' -f1)" == "$want" ]] \
            || stop "p12_rules.md differs from p12_rules.sha256 or from the repo copy"
        run_bg python3 "$REPO/train/scripts/mapmad_p11/p11_preflight.py" check "$rules" "$OUT_MOUNT" >> "$LOG" 2>&1 \
            || stop "pre-flight: a checkpoint sha256 or a frozen set fingerprint differs from p12_rules.md (driver/chain.log)"
    fi
    say "chain start: commit $COMMIT, rules sha256 ${want:0:16}; config $CFG"
    if [[ ! -f "$P12/checks/f_check.json" ]] || ! grep -q '"pass": true' "$P12/checks/f_check.json"; then
        STEP="0 F check"; status step="$STEP" --state WAITING
        local g; g="$(wait_gpu)"
        status gpus="$g" --state RUNNING
        say "GPU event: GPU $g for the F check (2 Phase 8 dev drives, pose replay)"
        run_limit 3600 "p12_f_check.py" dh "python mapmad/scripts/p12_f_check.py --gpu $g --config $CFG" \
            >> "$P12/driver/f_check.log" 2>&1
        local code=$?
        (( code == 3 )) && stop "F check failed: F_truth's decisions differ from the Phase 8 teacher (checks/f_check.json)"
        (( code == 0 )) || stop "F check did not run (exit $code, driver/f_check.log)"
        say "F check passed: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print([(r['drive'], r['frames']) for r in d['drives']])" "$P12/checks/f_check.json")"
    fi

    # --- 1. dev runs ---------------------------------------------------------------------------------------------------
    CURRENT_STEP="1 dev"
    STEP_USED=0
    step_runs dev_hidden "$DEV_BUDGET"
    run_bg dx "" "python -m vint_train.mapmad.closed_loop.p12_score same-as-p11 --config $CFG --p12 dev/hidden/logs/v2h_dev \
        --p11 ../p11_dagger/step0/logs/run1_hidden --seed 0 ${P12_SAME_IGNORE:+--ignore $P12_SAME_IGNORE} \
        --out $C/checks/same_as_p11.json" >> "$P12/driver/dev.log" 2>&1
    code=$?
    (( code == 3 )) && stop "refactor check: v2 maps hidden seed 0 differs from Phase 11's log (checks/same_as_p11.json)"
    (( code == 0 )) || stop "refactor check did not run (exit $code, driver/dev.log)"
    say "refactor check passed: v2 maps hidden seed 0 = Phase 11's run1_hidden logs (checks/same_as_p11.json)"
    if [[ ! -f "$P12/P12_DEV_CHOICES.json" ]]; then
        run_bg dx "" "python -m vint_train.mapmad.closed_loop.p12_score dev-choices --config $CFG --headline-only" \
            >> "$P12/driver/dev.log" 2>&1
        code=$?
        (( code == 3 )) && stop "dev decisions: $(cat "$P12/STOP_dev.txt" 2>/dev/null)"
        (( code == 0 )) || stop "headline decision failed (exit $code, driver/dev.log)"
        cp "$P12/P12_HEADLINE.json" "$P12/P12_DEV_CHOICES.json"
    fi
    local hl; hl="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['headline'])" "$P12/P12_DEV_CHOICES.json")"
    say "headline checkpoint (block item 1): $hl ($(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['deltas'], 'both Delta <= 0' if d['both_delta_le_0'] else '')" "$P12/P12_DEV_CHOICES.json"))"
    status note="headline = $hl"
    step_runs dev_tau "$(( DEV_BUDGET - STEP_USED ))"
    STEP="1 determinism episode"
    status step="$STEP" --state WAITING
    g="$(wait_gpu)"
    status gpus="$g" --state RUNNING
    local e=p11d-00081-bed-2624
    for r in a b; do
        if [[ ! -f "$P12/checks/determinism_$r/logs/hl_yoloe_dev/${e}__s0.jsonl" ]]; then
            GPU="$g" TOKEN="det_${r}_$(date +%H%M%S)" OWNER="$WHO" PAIRS_PER_GPU=1 PORT0=5600 RPORT0=5750 RECOG=yoloe \
                run_limit 1800 "checks/determinism_$r" bash "$REPO/mapmad/scripts/run_p12_pairs.sh" --config "$CFG" \
                --subdir "checks/determinism_$r" --set dev --arms hl_yoloe_dev --episodes "$e" --seeds 0 \
                >> "$P12/driver/determinism.log" 2>&1
        fi
    done
    if cmp "$P12/checks/determinism_a/logs/hl_yoloe_dev/${e}__s0.jsonl" "$P12/checks/determinism_b/logs/hl_yoloe_dev/${e}__s0.jsonl" >> "$LOG" 2>&1; then
        echo "{\"pass\": true, \"summary\": \"$e, headline ($hl) + YOLOE, seed 0, twice on GPU $g: byte-identical logs\"}" \
            > "$P12/checks/determinism.json"
    else
        echo "{\"pass\": false, \"summary\": \"$e twice on GPU $g: logs differ\"}" > "$P12/checks/determinism.json"
        stop "determinism episode not identical ($e, GPU $g, checks/determinism_a|b)"
    fi
    say "determinism episode $e identical on GPU $g (headline + YOLOE)"

    # --- 2. tau ---------------------------------------------------------------------------------------------------------
    STEP="2 tau"
    status step="$STEP"
    run_bg dx "" "python -m vint_train.mapmad.closed_loop.p12_score score --config $CFG --subdir dev/tau --set dev \
        --arms hl_yoloe_dev && python -m vint_train.mapmad.closed_loop.p12_score dev-choices --config $CFG" \
        >> "$P12/driver/dev.log" 2>&1
    code=$?
    (( code == 3 )) && stop "dev decisions: $(cat "$P12/STOP_dev.txt" 2>/dev/null)"
    (( code == 0 )) || stop "tau decision failed (exit $code, driver/dev.log)"
    say "dev choices: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('headline', d['headline'], 'deltas', d['deltas'], 'tau', d['tau'])" "$P12/P12_DEV_CHOICES.json")"
    grep -q '"both_delta_le_0": true' "$P12/P12_DEV_CHOICES.json" && say "both Delta <= 0 (rule applied anyway, block item 1)"
    status note="dev choices done (see Dev choices)"
    STEP="2 live rule-stop check"
    local tau e2; tau="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tau'])" "$P12/P12_DEV_CHOICES.json")"
    e2="$(python3 "$SCRIPTS/p12_plan.py" "$HCFG" episodes dev | grep -v -x "$e" | head -1)"
    [[ -n "$e2" ]] || stop "live rule-stop check: no second dev episode"
    for mode in full live; do
        if [[ ! -d "$P12/checks/live_rule_$mode/summaries" ]]; then
            GPU="$g" TOKEN="live_${mode}_$(date +%H%M%S)" OWNER="$WHO" PAIRS_PER_GPU=1 PORT0=5600 RPORT0=5750 RECOG=yoloe                 run_limit 3600 "checks/live_rule_$mode" bash "$REPO/mapmad/scripts/run_p12_pairs.sh" --config "$CFG"                 --subdir "checks/live_rule_$mode" --set dev --arms hl_yoloe_dev --episodes "$e" "$e2" --seeds 0                 $([[ $mode == live ]] && echo "--live-rule-tau $tau") >> "$P12/driver/live_rule.log" 2>&1
        fi
    done
    run_bg dx "" "python -m vint_train.mapmad.closed_loop.p12_score live-check --config $CFG --full checks/live_rule_full         --live checks/live_rule_live --arm hl_yoloe_dev --tau $tau --out $C/checks/live_rule.json" >> "$P12/driver/live_rule.log" 2>&1
    code=$?
    (( code == 3 )) && stop "rule stop scored from the log differs from the live rule stop (checks/live_rule.json)"
    (( code == 0 )) || stop "live rule-stop check did not run (exit $code, driver/live_rule.log)"
    say "live rule-stop check passed on $e, $e2 (GPU $g, tau $tau)"

    # --- 3. exam sets ------------------------------------------------------------------------------------------------------
    STEP="3 exam sets"
    status step="$STEP" --state WAITING
    if [[ ! -f "$P12/sets.json" ]]; then
        local gs pids=() k w=0 nw
        gs="$(free_for)"
        while [[ -z "$gs" ]]; do pause "$CHECK_S"; gs="$(free_for)"; done
        nw=$(( 3 * $(echo $gs | wc -w) ))
        status gpus="$gs" --state RUNNING
        say "GPU event: GPU(s) $gs for the exam-set candidates ($nw workers)"
        candidates() {  # every worker (3 per GPU), exam then OVON candidates; 1 if any worker failed
            local g k w=0 p bad=0 pids=()
            for g in $gs; do
                gpu_clean "$g" || continue
                for k in 0 1 2; do
                    dh "python mapmad/scripts/make_p12_sets.py exam-candidates --config $CFG --gpu $g --worker $w --workers $nw && \
                        python mapmad/scripts/make_p12_sets.py ovon-candidates --config $CFG --gpu $g --worker $w --workers $nw" \
                        >> "$P12/driver/sets_w$w.log" 2>&1 &
                    pids+=($!); w=$(( w + 1 ))
                done
            done
            for p in "${pids[@]}"; do wait "$p" || bad=1; done
            return $bad
        }
        run_limit 10800 "make_p12_sets.py" candidates
        code=$?
        (( code == 124 )) && stop "set candidates ran > 3 h (driver/sets_w*.log)"
        (( code == 0 )) || stop "set candidates failed (driver/sets_w*.log)"
        run_bg dh "python mapmad/scripts/make_p12_sets.py freeze --config $CFG" >> "$P12/driver/sets.log" 2>&1
        code=$?
        (( code == 3 )) && stop "$(cat "$P12/STOP_sets.txt" 2>/dev/null)"
        (( code == 0 )) || stop "set freeze failed (exit $code, driver/sets.log)"
    fi
    say "exam sets frozen: $(tr -d '\n ' < "$P12/sets.json")"

    # --- 4. 20-run timing + estimate + trims ---------------------------------------------------------------------------
    STEP="4 timing"
    status step="$STEP"
    local first_eps hm
    hm="$(python3 "$SCRIPTS/p12_plan.py" "$HCFG" pair)"
    first_eps="$(python3 -c "import json,sys; print(' '.join(e['episode_id'] for e in json.load(open(sys.argv[1]))['episodes'][:4]))" "$P12/exam_set/episodes.json")"
    local n_first pair_runs n_exam t_total
    n_first=$(echo $first_eps | wc -w)
    pair_runs="$(python3 "$SCRIPTS/p12_plan.py" "$HCFG" calls exam | head -1 | cut -d'|' -f7)"
    n_exam="$(python3 "$SCRIPTS/p12_plan.py" "$HCFG" episodes exam | wc -l)"
    t_total=$(( n_first * pair_runs / n_exam ))  # the headline pair's runs on the first 4 episodes (24)
    if [[ ! -f "$P12/driver/timing.json" ]]; then
        CL_EXTRA="--episodes $first_eps" closed_loop timing runs/exam exam "$t_total" "$EXAM_BUDGET" yoloe "" $hm
        code=$?
        CL_EXTRA=""
        EXAM_USED=$(( EXAM_USED + CL_USED_S ))
        (( code == 0 )) || stop "timing runs failed (code $code)"
        python3 - "$P12" "$hm" > "$P12/driver/timing.json" <<'EOF' || stop "timing estimate failed"
import glob, json, os, sys
p12, arms = sys.argv[1], sys.argv[2].split()
rows = [json.loads(l) for a in arms for f in glob.glob(os.path.join(p12, "runs/exam/timing", f"{a}.shard*.jsonl")) for l in open(f)]
wall = sorted(r["wall_s"] for r in rows)
print(json.dumps({"runs": len(rows), "mean_wall_s": sum(wall) / len(wall), "median_wall_s": wall[len(wall) // 2],
                  "rec_ms_mean": sum(r.get("rec_ms_mean", 0) for r in rows) / len(rows)}))
EOF
    fi
    local per_run pairs_now est
    per_run="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['mean_wall_s'])" "$P12/driver/timing.json")"
    pairs_now=$(( 3 * $(echo $(free_for) | wc -w) ))
    (( pairs_now > 0 )) || pairs_now=3
    local budget_now=$EXAM_BUDGET
    (( pairs_now <= 3 )) && budget_now=$(( 96 * 3600 ))  # §8: only one GPU usable -> 3 pairs, budget <= 96 h
    python3 "$SCRIPTS/p12_plan.py" "$HCFG" trim "$per_run" "$pairs_now" "$budget_now" >> "$LOG" 2>&1 || stop "trim plan failed"
    est="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['estimate_s'], d['estimate_untrimmed_s'], ' '.join(d['trims']) or 'none')" "$P12/driver/trims.json")"
    read -r est_s est0_s trims <<< "$est"
    say "timing: $per_run s per run ($t_total runs), $pairs_now pairs -> exam stages $(( est0_s / 3600 )) h untrimmed, $(( est_s / 3600 )) h after trims ($trims); budget $(( budget_now / 3600 )) h"
    (( est_s * 2 > budget_now * 3 )) && stop "budget: estimate $(( est_s / 3600 )) h > 1.5 x $(( budget_now / 3600 )) h even after all trims"
    local qwen_s; qwen_s=$(( $(python3 "$SCRIPTS/p12_plan.py" "$HCFG" total qwen) * 750 / 2 ))
    local eta_s=$(( est_s + qwen_s + 3600 ))
    status come_back="$(date -d "+$eta_s seconds" '+%a %d %b ~%H:%M')" estimate="exam stages ~$(( est_s / 3600 )) h ($pairs_now pairs, trims: $trims) + Qwen3-VL ~$(( qwen_s / 3600 )) h + tables/videos ~1 h"
    [[ "$pairs_now" -le 3 ]] && status note="only one GPU usable at the timing: 3 pairs, budget 96 h"

    # --- 5-8. main exam, OVON, exploration, unseeable ----------------------------------------------------------------------
    local left=$(( budget_now * 3 / 2 - EXAM_USED ))  # the 1.5x stop line, shared by steps 5-8
    for stage in exam ovon explore unseeable; do
        CURRENT_STEP="$(case $stage in exam) echo 5 main exam;; ovon) echo 6 OVON;; explore) echo 7 exploration;; unseeable) echo 8 unseeable;; esac)"
        STEP_USED=0
        step_runs "$stage" "$left"
        left=$STAGE_LEFT
        say "$CURRENT_STEP done ($(( STEP_USED / 60 )) min of GPU-running time; $(( left / 3600 )) h left before the 1.5x stop)"
    done

    # --- 9. Qwen3-VL subset (own GPU: one pair per GPU, the main queue is done) ---------------------------------------------
    CURRENT_STEP="9 Qwen3-VL subset"
    STEP_USED=0
    PAIRS_PER_GPU=1 CL_RPORT0=5780 step_runs qwen $(( 30 * 3600 ))

    # --- 10. scores + fairness + tables ------------------------------------------------------------------------------------
    STEP="10 tables"
    status step="$STEP" gpus="" subdir="" arms=""
    local stage line label subdir set recog seeds arms n
    for stage in exam ovon explore unseeable qwen; do
        while IFS='|' read -r label subdir set recog seeds arms n; do
            run_bg dx "" "python -m vint_train.mapmad.closed_loop.p12_score score --config $CFG --subdir $subdir --set $set \
                --arms $arms" >> "$P12/driver/tables.log" 2>&1 || stop "scoring $label failed (driver/tables.log)"
        done < <(python3 "$SCRIPTS/p12_plan.py" "$HCFG" calls "$stage")
    done
    run_bg dx "" "python -m vint_train.mapmad.closed_loop.p12_score fairness --config $CFG && \
        python -m vint_train.mapmad.closed_loop.p12_tables --config $CFG && \
        python -m vint_train.mapmad.closed_loop.p12_figures --config $CFG" >> "$P12/driver/tables.log" 2>&1 \
        || stop "tables failed (driver/tables.log)"
    say "tables written (tables/tables.md, figures/); fairness: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['summary'])" "$P12/checks/fairness.json")"

    # --- 11. videos ---------------------------------------------------------------------------------------------------------
    STEP="11 videos"
    status step="$STEP" --state WAITING
    g="$(wait_gpu)"
    status gpus="$g" --state RUNNING
    say "GPU event: GPU $g for the videos (one YOLOE server, port 5790)"
    docker exec -d -e MAPMAD_SIDECAR_KEY="${VKEY:=$(python3 -c 'import secrets; print(secrets.token_hex(32))')}" \
        -e CUDA_VISIBLE_DEVICES="$g" -e HOME=/model_cache/home -e MAPMAD_P12_OWNER="$WHO" naz_mapmad_recog \
        python -m mapmad_recog.server --model yoloe --port 5790 --text-reuse-out "$C/videos/text_reuse_yoloe.json"
    run_limit 21600 "p12_videos.py" docker exec -e MAPMAD_SIDECAR_KEY="$VKEY" -e MAPMAD_P12_OWNER="$WHO" \
        -w /app/visualnav-transformer naz_mapmad_habitat bash -lc "python mapmad/scripts/p12_videos.py pick --config $CFG && \
        python mapmad/scripts/p12_videos.py render --config $CFG --gpu $g --recog-port 5790" >> "$P12/driver/videos.log" 2>&1
    code=$?
    kill_inside "mapmad_recog.server --model yoloe --port 5790"
    (( code == 0 )) || say "videos failed (exit $code, driver/videos.log): report, no stop (rendered from logs, can be redone)"

    # --- 12. G12 draft ------------------------------------------------------------------------------------------------------
    STEP="12 G12 draft"
    FINISHED=1
    status step="$STEP" note="draft G12: G12_table_draft.md; tables/tables.md; videos/small/" eta="" gpus="" --state "CHAIN DONE"
    date '+%F %T' > "$P12/driver/CHAIN_DONE.txt"
    say "CHAIN DONE"
}

main "$@"
exit $?
