#!/bin/bash
# Phase 11 chained driver (kickoff §3 item 11.5, p11_rules.md). Host side, in screen p11_chain (start with
# launch_p11.sh, which also starts the GPU watcher p11_gpuwatch and the host watcher p11_chain_watch). Steps:
#   0. pre-flight: committed code, rules sha256 (outputs = repo copy), checkpoint sha256s, frozen set fingerprints;
#      label check (CPU, 50 Phase 8 test-split drives; a failure stops the chain BEFORE the rollouts, after step 2);
#   1. step-0 dev check: run 1 + run C (3 seeds) + run 1 maps hidden (1 seed, report only) on the 120 dev episodes, 3
#      Habitat + policy pairs per free GPU, one shared run queue; determinism episode (same GPU, twice, cmp);
#   2. pick mechanically (search_tables pick) -> V2_FINAL.json / .txt;
#   3. rollouts: the winner drives itself on the 1,448 rollout episodes, recorded as Phase 8 drives; labels in
#      parallel on finished drives, a final pass, label stats (teacher-fail share stop);
#   4. data index: DAgger splits, frontier cache (checked), sample index + LMDB + union shares, 10-state sheet;
#   5. restart naz_mapmad (nothing else in it), wait until healthy; RESTART_DONE (the twin may start);
#   6. fine-tune -> v2.1 ema_9 (supervised, one resume, hang watchdog, budget on the measured epoch time);
#   7. offline checks on ema_9: G10 rows 1-2 (CPU), rows 3-4, seen list, v2.1 - v2, DAgger-state loss;
#   8. dev check v2.1 (3 seeds; v2 = its step-0 runs) -> dev_v21/table;
#   9. draft G11 table -> CHAIN DONE.
# Automatic stops (block item 9) -> STOP_reason.txt, STATUS.md STOPPED, fix nothing. EXIT trap: any other exit writes
# STOP_reason.txt with the exit code and the last log line. Long steps run in the background (TERM handled within
# ~1 s). The body is one function, so editing this file cannot change a running driver.
# DRY runs (never results): P11_CONFIG=<dry config in mapmad/configs> P11_DIR=<host DRY folder> P11_ALLOW_DIRTY=1
# (uncommitted code allowed, rules not checked), P11_V21_BASE=<container path of the v2.1 base config>,
# P11_TRAIN_EXTRA="key: value|key: value" (extra launch-overlay lines, e.g. a tiny epoch).
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P11="${P11_DIR:-$OUT_MOUNT/mapmad/p11_dagger}"
    C="${P11/$OUT_MOUNT//outputs}"
    WHO=chain
    LOG="$P11/driver/chain.log"
    mkdir -p "$P11/driver" "$P11/launch"
    exec 2>> "$LOG"  # bash errors (e.g. an unbound variable) land in the log, not only on the screen
    # shellcheck source=p11_lib.sh
    source "$REPO/train/scripts/mapmad_p11/p11_lib.sh"
    [[ -n "${P11_CONFIG:-}" ]] && CFG="/app/visualnav-transformer/mapmad/configs/$P11_CONFIG"
    FINISHED=0
    BG_PID=""
    LABEL_FAIL=""
    rm -f "$P11/STOP_reason.txt"

    stop() {  # reason -> automatic stop (no fix)
        { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "reason: $1"
          echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P11/STOP_reason.txt"
        status step="${STEP:-?}" reason="$1" --state STOPPED
        say "AUTOMATIC STOP: $1"
        FINISHED=1  # a planned stop, not a crash of the driver
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        for pat in "run_search" "sidecar_server" "p11_labels.py" "launch/v21.yaml" "offline/v21_ema9" "dataset_dagger"; do
            kill_inside "$pat"; done  # our own processes only (patterns of this chain)
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "exit code: $code"
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P11/STOP_reason.txt"
            status reason="driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    # --- 0. pre-flight ------------------------------------------------------------------------------------------------
    STEP="0 pre-flight"
    status step="$STEP" started="$(date '+%F %T')" --state RUNNING
    cd "$REPO" || exit 2
    if [[ -z "${P11_ALLOW_DIRTY:-}" && -n "$(git status --porcelain -- train mapmad)" ]]; then
        stop "uncommitted code under train/ or mapmad/ (commit first, §0.8)"
    fi
    COMMIT="$(git rev-parse HEAD)"
    local rules="$OUT_MOUNT/mapmad/p11_dagger/p11_rules.md" want="DRY"
    if [[ -z "${P11_ALLOW_DIRTY:-}" ]]; then
        want="$(cut -d' ' -f1 "$OUT_MOUNT/mapmad/p11_dagger/p11_rules.sha256")"
        [[ "$(sha256sum "$rules" | cut -d' ' -f1)" == "$want" && "$(sha256sum mapmad/configs/p11_rules.md | cut -d' ' -f1)" == "$want" ]] \
            || stop "p11_rules.md differs from p11_rules.sha256 or from the repo copy"
        run_bg python3 "$SCRIPTS/p11_preflight.py" check "$rules" "$OUT_MOUNT" >> "$LOG" 2>&1 \
            || stop "pre-flight: a checkpoint sha256 or a frozen set fingerprint differs from p11_rules.md (driver/chain.log)"
    fi
    local n_step0 n_roll n_dev
    read -r n_step0 n_roll n_dev < <(python3 "$SCRIPTS/p11_preflight.py" totals "$REPO/mapmad/configs/${CFG##*/}" "$P11")
    [[ -n "$n_dev" ]] || stop "run counts not readable (p11_preflight.py totals)"
    say "chain start: commit $COMMIT, rules sha256 ${want:0:16}, checkpoints + set fingerprints match the rules; runs: step 0 $n_step0, rollouts $n_roll, dev v2.1 $n_dev"

    STEP="0 label check"
    status step="$STEP"
    run_limit 5400 "p11_labels.py check" dh "python mapmad/scripts/p11_labels.py check --config $CFG --workers 16" \
        > "$P11/driver/label_check.log" 2>&1
    local code=$?
    if (( code == 3 )); then
        LABEL_FAIL="label check: median error above the limit (labels/label_check.json)"
        say "LABEL CHECK FAILED -> the chain stops before the rollouts (after the v2 pick): $LABEL_FAIL"
    elif (( code != 0 )); then
        stop "label check did not run (exit $code, driver/label_check.log)"
    else
        say "label check passed: $(grep -a -E '"(median_m|p95_m|states)"' "$P11/driver/label_check.log" | tr -d ' \n')"
    fi

    # --- 1. step-0 dev check ------------------------------------------------------------------------------------------
    STEP="1 step-0 dev check"
    status step="$STEP" eta="step 0 ~3 h on 6 pairs"
    CL_ARMS="run1 runC run1_hidden"
    closed_loop step0 step0 "$n_step0" 21600 --set dev --arms run1 runC run1_hidden
    code=$?
    (( code == 1 )) && stop "budget: step-0 dev check > 1.5 x 4 h of GPU-running time"
    (( code == 2 )) && stop "step-0 runners exit without progress (a possible bug; driver/step0_gpu*.log)"
    say "step 0 runs done ($(( CL_USED_S / 60 )) min)"
    STEP="1 determinism episode"
    status step="$STEP"
    local g e
    g="$(wait_gpu)"
    e="$(python3 "$SCRIPTS/p11_preflight.py" first-dev "$REPO/mapmad/configs/${CFG##*/}" "$P11")"
    [[ -n "$e" ]] || stop "determinism episode id not readable"
    for r in a b; do
        GPU="$g" TOKEN="det_${r}_$(date +%H%M%S)" OWNER="$WHO" PAIRS_PER_GPU=1 PORT0=5600 \
            run_limit 1800 "checks/determinism_$r" bash "$REPO/mapmad/scripts/run_p11_pairs.sh" --config "$CFG" \
            --subdir "checks/determinism_$r" --set dev --arms run1 --episodes "$e" --seeds 0 \
            >> "$P11/driver/determinism.log" 2>&1
    done
    cmp "$P11/checks/determinism_a/logs/run1/${e}__s0.jsonl" "$P11/checks/determinism_b/logs/run1/${e}__s0.jsonl" \
        >> "$LOG" 2>&1 || stop "determinism episode not identical ($e, GPU $g, checks/determinism_a|b)"
    say "determinism episode $e identical on GPU $g"

    # --- 2. pick ----------------------------------------------------------------------------------------------------------
    STEP="2 v2 pick"
    status step="$STEP"
    run_bg dx "" "python -m vint_train.mapmad.closed_loop.search_tables pick --config $CFG --subdir step0" \
        >> "$P11/driver/pick.log" 2>&1
    code=$?
    (( code == 3 )) && stop "step 0: $(cat "$P11/step0/STOP_pick.txt" 2>/dev/null)"
    (( code == 0 )) || stop "v2 pick failed to run (exit $code, driver/pick.log)"
    local winner
    winner="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['arm'])" "$P11/V2_FINAL.json")"
    say "v2 pick: $(cat "$P11/V2_FINAL.txt")"
    status note="v2 = $winner"
    [[ -n "$LABEL_FAIL" ]] && stop "$LABEL_FAIL"

    # --- 3. rollouts + labels --------------------------------------------------------------------------------------------
    STEP="3 rollouts + labels"
    status step="$STEP" eta="rollouts ~5 h on 6 pairs"
    (  # labels on finished drives while the rollouts run (CPU, 8 workers), every 10 min
        while [[ ! -f "$P11/driver/ROLLOUTS_DONE" ]]; do
            dh "python mapmad/scripts/p11_labels.py label --config $CFG --workers 8 --ready $C/rollouts/summaries/v2" \
                >> "$P11/driver/labels_partial.log" 2>&1
            sleep 600
        done
    ) &
    local labeller=$!
    CL_ARMS="v2"
    closed_loop rollouts rollouts "$n_roll" 32400 --set rollout --arms v2 --seeds 0 --save-drives
    code=$?
    touch "$P11/driver/ROLLOUTS_DONE"
    kill "$labeller" 2>/dev/null; kill_inside "p11_labels.py label"
    (( code == 1 )) && stop "budget: rollouts > 1.5 x 6 h of GPU-running time"
    (( code == 2 )) && stop "rollout runners exit without progress (a possible bug; driver/rollouts_gpu*.log)"
    local done crashed
    read -r done crashed < <(python3 "$SCRIPTS/p11_status.py" "$P11" count rollouts v2)
    (( crashed * 100 > n_roll )) && stop "rollouts: crash rate $crashed / $n_roll > 1%"
    say "rollouts done: $done drives, $crashed crashed ($(( CL_USED_S / 60 )) min); final label pass"
    local left=$(( 32400 - CL_USED_S ))  # rollouts + labels share one budget (1.5 x 6 h)
    (( left > 600 )) || stop "budget: rollouts + labels > 1.5 x 6 h before the final label pass"
    run_limit "$left" "p11_labels.py label" dh "python mapmad/scripts/p11_labels.py label --config $CFG --workers 24" \
        > "$P11/driver/labels.log" 2>&1
    code=$?
    (( code == 124 )) && stop "budget: rollouts + labels > 1.5 x 6 h (final label pass cut)"
    (( code == 3 )) && stop "teacher-fail states > 40% of the visited states (labels/label_stats.json)"
    (( code == 0 )) || stop "labelling failed (exit $code, driver/labels.log)"
    say "labels: $(grep -a -E '"(teacher_fail_share|frames)"' "$P11/driver/labels.log" | tr -d ' \n')"

    # --- 4. data index --------------------------------------------------------------------------------------------------
    STEP="4 data index"
    status step="$STEP"
    local v21cfg="mapmad_v21.yaml" extra=()
    [[ "$winner" == runC ]] && v21cfg="mapmad_v21_seen.yaml"
    IFS='|' read -r -a extra <<< "${P11_TRAIN_EXTRA:-}"
    overlay "$P11/launch/v21.yaml" "${P11_V21_BASE:-$TRAIN_DIR/config/$v21cfg}" "logs_root: $C" "${extra[@]}"
    local v21launch="$C/launch/v21.yaml"
    run_bg dh "python mapmad/scripts/p11_labels.py splits --config $CFG" >> "$P11/driver/index.log" 2>&1 \
        || stop "DAgger splits failed (driver/index.log)"
    run_limit 7200 "dataset_dagger frontier" dx "" "python -m vint_train.mapmad.dataset_dagger frontier --config $v21launch \
        --workers 20 --out $C/labels/frontier_cache.json" >> "$P11/driver/index.log" 2>&1
    code=$?
    (( code == 3 )) && stop "DAgger frontier cache: a frame differs from the live harness count (labels/frontier_cache.json)"
    (( code == 0 )) || stop "DAgger frontier cache failed (exit $code, driver/index.log)"
    run_limit 10800 "dataset_dagger prepare" dx "" "python -m vint_train.mapmad.dataset_dagger prepare --config $v21launch \
        --out $C/labels/union.json" >> "$P11/driver/index.log" 2>&1 || stop "DAgger index / LMDB / union failed (driver/index.log)"
    say "DAgger data: $(tr -d ' \n' < "$P11/labels/union.json")"
    run_bg dh "python mapmad/scripts/p11_labels.py sheet --config $CFG" >> "$P11/driver/index.log" 2>&1 \
        || say "label sheet failed (report only, driver/index.log)"

    # --- 5. restart naz_mapmad --------------------------------------------------------------------------------------------
    STEP="5 restart naz_mapmad"
    status step="$STEP"
    local k busy
    for k in $(seq 1 20); do
        busy="$(docker exec naz_mapmad ps -eo stat,pid,args | awk '$1 !~ /^Z/' | grep -E 'python|train.py' \
            | grep -v -E 'ps -eo|grep' || true)"  # zombies (debris the restart clears) do not count
        [[ -z "$busy" ]] && break
        say "naz_mapmad still runs: $(echo "$busy" | head -3 | tr '\n' ';') -> waiting 1 min"
        pause 60
    done
    [[ -z "$busy" ]] || stop "naz_mapmad still runs processes after 20 min; not restarted (driver/chain.log)"
    { echo "before $(date '+%F %T'): shm $(docker exec naz_mapmad df -h /dev/shm | tail -1), files $(docker exec naz_mapmad bash -c 'ls /dev/shm | wc -l'), zombies $(docker exec naz_mapmad ps -eo stat | grep -c Z)"; } \
        >> "$P11/driver/restart.log" 2>&1
    run_bg docker restart naz_mapmad >> "$P11/driver/restart.log" 2>&1 || stop "docker restart naz_mapmad failed"
    for k in $(seq 1 30); do
        docker exec naz_mapmad python -c 'import torch; assert torch.cuda.device_count() == 2' >> "$P11/driver/restart.log" 2>&1 && break
        pause 10
    done
    docker exec naz_mapmad python -c 'import torch; assert torch.cuda.device_count() == 2' >> "$P11/driver/restart.log" 2>&1 \
        || stop "naz_mapmad not healthy after the restart (driver/restart.log)"
    echo "after $(date '+%F %T'): files $(docker exec naz_mapmad bash -c 'ls /dev/shm | wc -l'), zombies $(docker exec naz_mapmad ps -eo stat | grep -c Z)" \
        >> "$P11/driver/restart.log"
    date '+%F %T' > "$P11/driver/RESTART_DONE"
    say "naz_mapmad restarted and healthy (driver/restart.log)"

    # --- 6. fine-tune -> v2.1 ---------------------------------------------------------------------------------------------
    STEP="6 fine-tune v2.1"
    status step="$STEP" --state WAITING
    g="$(train_gpu)"
    echo "$g" > "$P11/driver/FINETUNE_GPU"  # the twin may now take a GPU the chain does not hold
    cp "$P11/launch/v21.yaml" "$P11/driver/launch_v21_$(date +%H%M%S).yaml"
    status step="$STEP" gpus="$g" epochs=10 --state RUNNING
    say "GPU event: GPU $g for the fine-tune ($v21cfg, 10 epochs)"
    local label run
    label="$(grep -E '^run_name:' "$REPO/train/config/$v21cfg" | awk '{print $2}')"
    (  # budget: the estimate after epoch 1 (10 x minutes per epoch) must stay <= 1.5 x 4 h
        while true; do
            sleep 120
            run="$(newest_run "$P11" "$label")"
            [[ -n "$run" && -f "$run/metrics.jsonl" ]] || continue
            n="$(grep -c '"train_seconds"' "$run/metrics.jsonl")"
            if (( n >= 1 )); then
                est="$(python3 -c "import json,sys; e=[json.loads(l) for l in open(sys.argv[1]) if '\"train_seconds\"' in l]; print(int(10*(e[0]['train_seconds']+e[0].get('eval_seconds',0))))" "$run/metrics.jsonl")"
                if (( est > 21600 )); then echo "BUDGET $est" > "$P11/driver/FINETUNE_BUDGET"; kill_inside "launch/v21.yaml"; fi
                break
            fi
        done
    ) &
    local budget_watch=$!
    supervised_train "$label" "$g" "$v21launch" 9 "$P11"
    code=$?
    kill "$budget_watch" 2>/dev/null
    [[ -f "$P11/driver/FINETUNE_BUDGET" ]] && stop "budget: fine-tune estimate $(cut -d' ' -f2 "$P11/driver/FINETUNE_BUDGET") s > 1.5 x 4 h"
    (( code == 0 )) || stop "$STOP_WHY"
    mkdir -p "$P11/offline/v21_ema9"
    sha256sum "$RUN_DIR/ema_9.pth" > "$P11/offline/v21_ema9/ema_9.sha256"
    cp "$P11/offline/v21_ema9/ema_9.sha256" "$RUN_DIR/ema_9.sha256"
    python3 - "$RUN_DIR" "$P11" "$v21cfg" "$OUT_MOUNT" <<'EOF' || stop "V21.json not written"
import json, sys
run, p11, cfg, out = sys.argv[1:]
sha = open(f"{p11}/offline/v21_ema9/ema_9.sha256").read().split()[0]
rel = run.replace(out + "/mapmad/", "", 1) + "/ema_9.pth"  # below <outputs>/mapmad, as the run config's paths
json.dump({"path": rel, "sha256": sha, "config": f"repo:train/config/{cfg}"}, open(f"{p11}/V21.json", "w"), indent=1)
EOF
    say "GPU event: fine-tune done on GPU $g; ema_9 sha256 $(cut -c1-16 "$P11/offline/v21_ema9/ema_9.sha256")"

    # --- 7. offline checks ------------------------------------------------------------------------------------------------
    STEP="7 offline checks on ema_9"
    status step="$STEP" run="" --state WAITING
    g="$(wait_gpu "$g")"
    status step="$STEP" gpus="$g" --state RUNNING
    local ck="${RUN_DIR/$P11/$C}/ema_9.pth" o="$C/offline/v21_ema9" v2ck v2scores
    offline_checks "$g" v21_ema9 "$ck" config/mapmad_v2.yaml || stop "offline checks failed (driver/offline_v21_ema9.log)"
    v2ck="$(python3 -c "import json,sys; print('/outputs/mapmad/' + json.load(open(sys.argv[1]))['path'])" "$P11/V2_FINAL.json")"
    v2scores=/outputs/mapmad/p10_v2_train/offline/main_ema19/search/scores.npz
    [[ "$winner" == runC ]] && v2scores=/outputs/mapmad/p10_v2_train/offline/c_ema19/search/scores.npz
    run_bg dx "" "python -m vint_train.mapmad.checks_g10 --config config/mapmad_v2.yaml --v1-config config/mapmad_p4_C.yaml \
        --reference /outputs/mapmad/p10_v2_train/checks/reference_forward_cpu.pt --out $o/g10_rows_1_2_cpu.json && \
        python -m vint_train.mapmad.target_groups guard --scores $o/search/scores.npz --reference $v2scores \
        --out $o/v21_minus_v2.json" >> "$P11/driver/offline_v21_ema9.log" 2>&1 \
        || stop "G10 rows 1-2 / v2.1 - v2 failed to run (driver/offline_v21_ema9.log)"
    run_limit 5400 "dataset_dagger loss" dx "$g" "python -m vint_train.mapmad.dataset_dagger loss --config $v21launch \
        --checkpoint v2=$v2ck v21=$ck --out $o/dagger_loss.json" >> "$P11/driver/offline_v21_ema9.log" 2>&1 \
        || say "DAgger-state loss failed (report only, driver/offline_v21_ema9.log)"
    python3 "$SCRIPTS/g11_table.py" "$P11" >> "$LOG" 2>&1
    say "offline checks done (offline/v21_ema9)"

    # --- 8. dev check v2.1 ------------------------------------------------------------------------------------------------
    STEP="8 dev check v2.1"
    status step="$STEP" eta="dev check v2.1 ~1.5 h"
    CL_ARMS="v21"
    # P11_TRAIN_GPUS set (Naz, 10-23): GPU 1 always, other GPUs only when free (20-min rule + live check); then the twin
    P11_GPUS="${P11_TRAIN_GPUS:-${P11_GPUS:-}}" closed_loop dev_v21 dev_v21 "$n_dev" 10800 --set dev --arms v21
    code=$?
    (( code == 1 )) && stop "budget: dev check v2.1 > 1.5 x 2 h of GPU-running time"
    (( code == 2 )) && stop "dev v2.1 runners exit without progress (a possible bug; driver/dev_v21_gpu*.log)"
    date '+%F %T' > "$P11/driver/DEV_V21_DONE"  # the twin may now take the training GPU
    run_bg dx "" "python -m vint_train.mapmad.closed_loop.search_tables compare --config $CFG --subdir dev_v21 \
        --arms v2=step0/$winner v21 --out dev_v21/table" >> "$P11/driver/dev_v21.log" 2>&1 \
        || stop "dev v2.1 table failed (driver/dev_v21.log)"
    say "dev check v2.1: $(grep -a 'Success' "$P11/dev_v21/table.md" | head -1)"

    # --- 9. G11 draft --------------------------------------------------------------------------------------------------------
    STEP="9 G11 draft"
    python3 "$SCRIPTS/g11_table.py" "$P11" >> "$LOG" 2>&1
    FINISHED=1
    status step="$STEP" note="draft G11: G11_table_draft.md" eta="" --state "CHAIN DONE"
    date '+%F %T' > "$P11/driver/CHAIN_DONE.txt"
    say "CHAIN DONE"
}

main "$@"
exit $?
