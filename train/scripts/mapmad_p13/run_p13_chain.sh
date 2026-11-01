#!/bin/bash
# Phase 13 chained driver (kickoff §2b Part 2, p13_rules.md). Host side, in screen p13_chain (start with launch_p13.sh,
# which also starts the GPU watcher p13_gpuwatch and the host watcher p13_chain_watch). Steps:
#   0. pre-flight: committed code, rules sha256 (outputs = repo copy), checkpoint sha256s + set fingerprints against the
#      rules' preflight block; label check of the Phase 11 labeller (CPU, report only);
#   1. 13.1 videos: the 12 Phase 12 video episodes x 8 arms, seed 0, re-run with Phase 13 code (YOLOE boxes logged),
#      each run on its Phase 12 GPU -> same-run check (pose sequence = the Phase 12 log, else "not reproducible") ->
#      render + automatic checks (boxes, poses) -> small copies;
#   2. 13.3 hidden-DAgger rollouts (v2 ema_19, maps hidden, Phase 11 rollout set) + labels (in parallel, final pass)
#      -> splits -> data index (hidden + map DAgger, equal-count numbers);
#   3. restart naz_mapmad -> 13.4 fine-tune v2.1b (GPU 1 preferred) -> ema_9 -> V21B.json -> G10 rows 1-4 offline;
#      determinism (row 6a): M21b on one exam-B episode twice on the same GPU;
#   4. 13.6 dev sanity on the MP3D dev set (report only, seed 0);
#   5. 20-run timing (first 4 HM3D episodes of M21b / M21bh) + the dev runs' MP3D / VF seconds -> estimate -> budget
#      check (stop only > 1.5 x 40 h);
#   6. 13.7 exams in the kickoff's order (p13_plan.py calls exams);
#   7. VLFM port check (train/scripts/mapmad_p13/vlfm_port.sh, if set up; else "not done", reported); HSSD skipped;
#   8. scores + fairness + tables + figures + draft G13 -> CHAIN DONE.
# Automatic stops (kickoff §7) -> STOP_reason.txt, STATUS.md STOPPED, wait. EXIT trap: any other exit writes
# STOP_reason.txt. Long steps run in the background (TERM handled within ~1 s). The body is one function, so editing
# this file cannot change a running driver. Resumable: every step skips what is already done.
# DRY runs (never results): P13_CFG / P13_HCFG_HIDDEN / P13_V21B_BASE container paths of DRY configs, P13_DIR=<host DRY
# folder>, P13_ALLOW_DIRTY=1, P13_GPUS="0 1", CL_CHECK_S=30, SILENT_S=600, P13_TRAIN_EXTRA="key: value|..."
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P13="${P13_DIR:-$OUT_MOUNT/mapmad/p13_claim}"
    C="${P13/$OUT_MOUNT//outputs}"
    WHO=chain
    LOG="$P13/driver/chain.log"
    mkdir -p "$P13/driver" "$P13/checks" "$P13/launch"
    exec 2>> "$LOG"
    # shellcheck source=p13_lib.sh
    source "$REPO/train/scripts/mapmad_p13/p13_lib.sh"
    local HIDDEN_CFG="${P13_HIDDEN_CFG:-/app/visualnav-transformer/mapmad/configs/p13_hidden_dagger.yaml}"
    local V21B_BASE="${P13_V21B_BASE:-$TRAIN_DIR/config/mapmad_v21b.yaml}"
    FINISHED=0
    BG_PID=""
    [[ -f "$P13/STOP_reason.txt" ]] && mv "$P13/STOP_reason.txt" "$P13/driver/STOP_reason_$(date +%m%d_%H%M%S).txt"
    local EXAM_BUDGET=$(( 40 * 3600 ))       # 13.7: <= 40 h; stop only > 1.5x
    local ROLL_BUDGET=$(( 9 * 3600 ))        # 13.3: <= 6 h GPU (stop > 9 h, rollouts + final label pass)
    local DEV_BUDGET=$(( 27 * 3600 / 6 ))    # 13.6: <= 3 h (stop > 4.5 h)
    local VIDEO_BUDGET=$(( 3 * 3600 / 2 ))   # 13.1: <= 1 h (stop > 1.5 h of GPU-running time)

    stop() {  # reason [question] -> automatic stop (no fix); waits for Naz
        { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "reason: $1"
          echo "question: ${2:-how to go on? (recommended: read the log, fix, relaunch: every step resumes)}"
          echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P13/STOP_reason.txt"
        status step="${STEP:-?}" reason="$1" error="$1" --state STOPPED
        say "AUTOMATIC STOP: $1"
        FINISHED=1
        exit 10
    }
    on_exit() {
        local code=$?
        [[ -n "${BG_PID:-}" ]] && kill "$BG_PID" 2>/dev/null
        for pat in "run_search" "sidecar_server" "mapmad_recog.server" "mapmad_recog.itm_server" "p11_labels.py" \
                   "p13_videos.py" "launch/v21b.yaml" "offline/v21b_ema9" "dataset_dagger"; do
            kill_inside "$pat"; done  # our own processes only (patterns of this chain)
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "exit code: $code"
              echo "question: relaunch launch_p13.sh after reading driver/chain.log? (recommended: yes, it resumes)"
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P13/STOP_reason.txt"
            status reason="driver exit $code" error="driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    plan() { python3 "$SCRIPTS/p13_plan.py" "$HCFG" "$@"; }
    fallback() { status fallback="$1"; say "FALLBACK: $1"; }
    step_runs() {  # stage budget -> runs every call of a stage (p13_plan.py), with all automatic stops
        local stage="$1" budget="$2" label subdir set recog seeds arms n code
        while IFS='|' read -r label subdir set recog seeds arms n; do
            [[ -n "$label" ]] || continue
            STEP="$CURRENT_STEP: $label"
            status step="$STEP" --state RUNNING
            say "$label: $n runs ($subdir, set $set, arms $arms, recogniser $recog${seeds:+, seeds $seeds})"
            closed_loop "$label" "$subdir" "$set" "$n" "$budget" "$recog" "$seeds" $arms
            code=$?
            budget=$(( budget - CL_USED_S ))
            (( code == 1 )) && stop "budget: $label exceeded the remaining GPU-running time (> 1.5x)" \
                "go on with a larger budget? (recommended: yes if the rest of the plan fits; else stop the stage)"
            (( code == 2 )) && stop "$label: runners exit without progress (a possible bug; driver/${label}_gpu*.log)"
            (( code == 3 )) && stop "$label: recogniser or ITM server down after 3 restarts ($subdir/recog_restarts)"
            crash_rate_ok "$subdir" "$n" $arms || stop "$label: crash rate > 1% ($subdir)"
        done < <(plan calls "$stage")
        STAGE_LEFT=$budget
    }

    # --- 0. pre-flight ------------------------------------------------------------------------------------------------
    STEP="0 pre-flight"
    status --reset step="$STEP" started="$(date '+%F %T')" mode="see phase13_log.md" --state RUNNING
    cd "$REPO" || exit 2
    if [[ -z "${P13_ALLOW_DIRTY:-}" && -n "$(git status --porcelain -- train mapmad)" ]]; then
        stop "uncommitted code under train/ or mapmad/ (commit first, §0.8)"
    fi
    COMMIT="$(git rev-parse HEAD)"
    local rules="$P13/p13_rules.md" want="DRY"
    if [[ -z "${P13_ALLOW_DIRTY:-}" ]]; then
        want="$(cut -d' ' -f1 "$P13/p13_rules.sha256")"
        [[ "$(sha256sum "$rules" | cut -d' ' -f1)" == "$want" && "$(sha256sum mapmad/configs/p13_rules.md | cut -d' ' -f1)" == "$want" ]] \
            || stop "p13_rules.md differs from p13_rules.sha256 or from the repo copy"
        run_bg python3 "$REPO/train/scripts/mapmad_p11/p11_preflight.py" check "$rules" "$OUT_MOUNT" >> "$LOG" 2>&1 \
            || stop "pre-flight: a checkpoint sha256 or a frozen set fingerprint differs from p13_rules.md (driver/chain.log)"
    fi
    say "chain start: commit $COMMIT, rules sha256 ${want:0:16}; config $CFG"
    if [[ ! -f "$P13/hidden_dagger/labels/label_check.json" ]]; then
        STEP="0 label check (report only)"
        status step="$STEP"
        run_limit 5400 "p11_labels.py check" dh "python mapmad/scripts/p11_labels.py check --config $HIDDEN_CFG --workers 16" \
            > "$P13/driver/label_check.log" 2>&1
        say "label check (report only, exit $?): $(grep -a -E '"(median_m|p95_m)"' "$P13/driver/label_check.log" | tr -d ' \n')"
    fi

    # --- 1. 13.1 videos ---------------------------------------------------------------------------------------------------
    CURRENT_STEP="1 videos (13.1)"
    STEP="$CURRENT_STEP: re-runs"
    status step="$STEP" --state RUNNING
    if [[ ! -f "$P13/checks/video_same_run.json" ]]; then
        video_gpu() {  # gpu -> runs this GPU's video calls (each run on its Phase 12 GPU); waits <= 6 h for the GPU
            local g="$1" t0=$SECONDS line arm eps
            until gpu_clean "$g" && [[ " $(free_for) " == *" $g "* ]]; do
                (( SECONDS - t0 > 21600 )) && { echo "GPU $g not free for 6 h" > "$P13/driver/video_gpu${g}_skipped"; return 1; }
                pause "$CHECK_S"
            done
            while IFS='|' read -r line arm eps; do
                [[ "$line" == "$g" ]] || continue
                GPU="$g" TOKEN="video_${arm}_$(date +%m%d_%H%M%S)" OWNER="$WHO" PAIRS_PER_GPU=2 PORT0=5600 RPORT0=5750 \
                    RECOG=yoloe bash "$REPO/mapmad/scripts/run_p13_pairs.sh" --config "$CFG" --subdir videos_rerun \
                    --set video --arms "$arm" --episodes $eps --seeds 0 >> "$P13/driver/video_gpu$g.log" 2>&1
            done < <(plan video-calls)
        }
        video_gpu 0 & local v0=$!
        video_gpu 1 & local v1=$!
        wait "$v0"; wait "$v1"
        for g in 0 1; do
            [[ -f "$P13/driver/video_gpu${g}_skipped" ]] && fallback "13.1: GPU $g was not free for 6 h: its video runs marked not reproducible"
        done
        run_bg dh "python mapmad/scripts/p13_videos.py same-run --config $CFG --new $C/videos_rerun/logs \
            --old /outputs/mapmad/p12_search_exam/runs/exam/logs --out $C/checks/video_same_run.json" \
            >> "$P13/driver/videos.log" 2>&1 || say "same-run check failed to run (driver/videos.log): reported"
        fallback "13.1: Phase 12 logs hold no YOLOE boxes -> the 12 video episodes x 8 arms re-run (seed 0, Phase 12 GPU)"
    fi
    say "13.1 same-run: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('summary', d))" "$P13/checks/video_same_run.json" 2>/dev/null | cut -c1-300)"
    STEP="$CURRENT_STEP: render"
    status step="$STEP" --state WAITING
    if [[ ! -f "$P13/checks/video_check.json" ]]; then
        local gs="" vg
        until [[ -n "$gs" ]]; do  # every GPU free for us now (both: 2 render shards each)
            gs="$(usable_now)"
            [[ -n "$gs" ]] || pause "$CHECK_S"
        done
        status gpus="$gs" --state RUNNING
        local nsh=$(( 2 * $(echo $gs | wc -w) )) k=0 rpids=()
        # one YOLOE server PER SHARD (port 5790 + shard): the server takes one client at a time; two shards on one
        # server made the second time out (Phase 13 real chain: every M21 and F video lost)
        say "GPU event: GPU(s)$gs for the video render ($nsh shards; one YOLOE server per shard, port 5790 + shard)"
        : "${VKEY:=$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"
        for vg in $gs; do
            for s in 0 1; do
                docker exec -d -e MAPMAD_SIDECAR_KEY="$VKEY" -e CUDA_VISIBLE_DEVICES="$vg" -e HOME=/model_cache/home \
                    -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog python -m mapmad_recog.server --model yoloe \
                    --port $(( 5790 + k )) --text-reuse-out "$C/videos/text_reuse_yoloe_shard$k.json"
                k=$(( k + 1 ))
            done
        done
        k=0
        render_shards() {
            local vg s pids=() bad=0 p
            for vg in $gs; do
                for s in 0 1; do
                    docker exec -e MAPMAD_SIDECAR_KEY="$VKEY" -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer \
                        naz_mapmad_habitat python mapmad/scripts/p13_videos.py render-plan --config "$CFG" \
                        --subdir videos_rerun --gpu "$vg" --recog-port $(( 5790 + k )) --out "$C/videos" \
                        --shard "$k/$nsh" >> "$P13/driver/videos_shard$k.log" 2>&1 &
                    pids+=($!); k=$(( k + 1 ))
                done
            done
            for p in "${pids[@]}"; do wait "$p" || bad=1; done
            return $bad
        }
        run_limit $(( VIDEO_BUDGET * 2 )) "p13_videos.py" render_shards
        code=$?
        kill_inside "mapmad_recog.server --model yoloe --port 579"
        run_bg dh "python mapmad/scripts/p13_videos.py check --config $CFG --out $C/videos --report $C/checks/video_check.json" \
            >> "$P13/driver/videos.log" 2>&1
        (( code == 0 )) || say "videos: render failed or cut at the time limit (exit $code, driver/videos_shard*.log): reported, no stop"
        fallback "13.1: rendered 12 picks x the 4 review arms (M21, M21h, N, F) in the chain (budget); M2, M2h, V1, MT re-run and logged, renderable later"
    fi

    # --- 2. 13.3 hidden-DAgger rollouts + labels ---------------------------------------------------------------------------
    CURRENT_STEP="2 hidden-DAgger rollouts (13.3)"
    local n_roll HC="$C/hidden_dagger"
    n_roll="$(python3 -c "import json,os,sys,yaml; c=yaml.safe_load(open(sys.argv[1])); f=os.path.normpath(os.path.join(sys.argv[2], c['out_dir'], c['sets']['rollout']['folder'], 'episodes.json')); print(len(json.load(open(f))['episodes']))" "$REPO/mapmad/configs/${HIDDEN_CFG##*/}" "$OUT_MOUNT/mapmad")"
    [[ "$n_roll" =~ ^[0-9]+$ ]] || stop "rollout set size not readable ($HIDDEN_CFG)"
    STEP="$CURRENT_STEP"
    status step="$STEP" eta="rollouts ~6 h on 6 pairs"
    if [[ ! -f "$P13/hidden_dagger/labels/label_stats.json" || ! -f "$P13/driver/ROLLOUTS_DONE" ]]; then
        (  # labels on finished drives while the rollouts run (CPU, 8 workers), every 10 min
            while [[ ! -f "$P13/driver/ROLLOUTS_DONE" ]]; do
                dh "python mapmad/scripts/p11_labels.py label --config $HIDDEN_CFG --workers 8 --ready $HC/rollouts/summaries/v2h" \
                    >> "$P13/driver/labels_partial.log" 2>&1
                sleep 600
            done
        ) &
        local labeller=$!
        CL_CFG="$HIDDEN_CFG" CL_PREFIX="hidden_dagger/" CL_EXTRA="--save-drives" \
            closed_loop rollouts rollouts rollout "$n_roll" "$ROLL_BUDGET" none "0" v2h
        code=$?
        touch "$P13/driver/ROLLOUTS_DONE"
        kill "$labeller" 2>/dev/null; kill_inside "p11_labels.py label"
        (( code == 1 )) && stop "budget: hidden rollouts > 1.5 x 6 h of GPU-running time"
        (( code == 2 )) && stop "hidden rollout runners exit without progress (a possible bug; driver/rollouts_gpu*.log)"
        crash_rate_ok "hidden_dagger/rollouts" "$n_roll" v2h || stop "hidden rollouts: crash rate > 1%"
        say "hidden rollouts done ($(( CL_USED_S / 60 )) min); final label pass"
        run_limit 14400 "p11_labels.py label" dh "python mapmad/scripts/p11_labels.py label --config $HIDDEN_CFG --workers 24" \
            > "$P13/driver/labels.log" 2>&1
        code=$?
        (( code == 3 )) && fallback "13.3: teacher-fail share above Phase 11's 40% limit (recorded, no stop: hidden_dagger/labels/label_stats.json)"
        (( code == 0 || code == 3 )) || stop "labelling failed (exit $code, driver/labels.log)"
    fi
    say "hidden labels: $(grep -a -E '"(teacher_fail_share|frames)"' "$P13/driver/labels.log" 2>/dev/null | tr -d ' \n')"
    STEP="2 hidden-DAgger data index"
    status step="$STEP"
    local extra=()
    IFS='|' read -r -a extra <<< "${P13_TRAIN_EXTRA:-}"
    overlay "$P13/launch/v21b.yaml" "$V21B_BASE" "logs_root: $C" "${extra[@]}"
    local v21blaunch="$C/launch/v21b.yaml"
    if [[ ! -f "$P13/hidden_dagger/union.json" ]]; then
        run_bg dh "python mapmad/scripts/p11_labels.py splits --config $HIDDEN_CFG" >> "$P13/driver/index.log" 2>&1 \
            || stop "hidden-DAgger splits failed (driver/index.log)"
        run_limit 10800 "dataset_dagger prepare" dx "" "python -m vint_train.mapmad.dataset_dagger prepare --config $v21blaunch \
            --out $HC/union.json" >> "$P13/driver/index.log" 2>&1 || stop "hidden-DAgger index / LMDB / union failed (driver/index.log)"
    fi
    say "DAgger data (equal-count rule): $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('hidden'))" "$P13/hidden_dagger/union.json")"
    if python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['hidden']['hidden_repeat_factor'] > 1 else 1)" "$P13/hidden_dagger/union.json"; then
        fallback "13.4: hidden-DAgger set smaller than its draws -> drawn with replacement (repeat factor in hidden_dagger/union.json)"
    fi

    # --- 3. restart naz_mapmad -> fine-tune v2.1b -> offline checks ---------------------------------------------------------
    if [[ ! -f "$P13/V21B.json" ]]; then
        STEP="3 restart naz_mapmad"
        status step="$STEP"
        local k busy
        for k in $(seq 1 20); do
            busy="$(docker exec naz_mapmad ps -eo stat,pid,args | awk '$1 !~ /^Z/' | grep -E 'python|train.py' \
                | grep -v -E 'ps -eo|grep' || true)"
            [[ -z "$busy" ]] && break
            say "naz_mapmad still runs: $(echo "$busy" | head -3 | tr '\n' ';') -> waiting 1 min"
            pause 60
        done
        [[ -z "$busy" ]] || stop "naz_mapmad still runs processes after 20 min; not restarted"
        run_bg docker restart naz_mapmad >> "$P13/driver/restart.log" 2>&1 || stop "docker restart naz_mapmad failed"
        for k in $(seq 1 30); do
            docker exec naz_mapmad python -c 'import torch; assert torch.cuda.device_count() == 2' >> "$P13/driver/restart.log" 2>&1 && break
            pause 10
        done
        docker exec naz_mapmad python -c 'import torch; assert torch.cuda.device_count() == 2' >> "$P13/driver/restart.log" 2>&1 \
            || stop "naz_mapmad not healthy after the restart (driver/restart.log)"
        say "naz_mapmad restarted and healthy"
        STEP="3 fine-tune v2.1b (13.4)"
        status step="$STEP" --state WAITING
        g="$(train_gpu)"
        cp "$P13/launch/v21b.yaml" "$P13/driver/launch_v21b_$(date +%H%M%S).yaml"
        status step="$STEP" gpus="$g" epochs=10 eta="fine-tune ~2.5 h" --state RUNNING
        say "GPU event: GPU $g for the v2.1b fine-tune (10 epochs)"
        supervised_train v21b_s1 "$g" "$v21blaunch" 9 "$P13"
        code=$?
        (( code == 0 )) || stop "$STOP_WHY"
        mkdir -p "$P13/offline/v21b_ema9"
        sha256sum "$RUN_DIR/ema_9.pth" > "$P13/offline/v21b_ema9/ema_9.sha256"
        python3 - "$RUN_DIR" "$P13" "$OUT_MOUNT" <<'EOF' || stop "V21B.json not written"
import json, sys
run, p13, out = sys.argv[1:]
sha = open(f"{p13}/offline/v21b_ema9/ema_9.sha256").read().split()[0]
rel = run.replace(out + "/mapmad/", "", 1) + "/ema_9.pth"
json.dump({"path": rel, "sha256": sha, "config": "repo:train/config/mapmad_v21b.yaml"}, open(f"{p13}/V21B.json", "w"), indent=1)
EOF
        say "GPU event: fine-tune done on GPU $g; ema_9 sha256 $(cut -c1-16 "$P13/offline/v21b_ema9/ema_9.sha256")"
    fi
    if [[ ! -f "$P13/offline/v21b_ema9/g10_rows_1_2_cpu.json" ]]; then
        STEP="3 G10 rows 1-4 offline on ema_9"
        status step="$STEP" --state WAITING
        g="$(wait_gpu 1)"
        status gpus="$g" --state RUNNING
        local ck o="$C/offline/v21b_ema9"
        ck="/outputs/mapmad/$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['path'])" "$P13/V21B.json")"
        offline_checks "$g" v21b_ema9 "$ck" config/mapmad_v2.yaml || fallback "13.4: G10 rows 3-4 checks failed to run (recorded, exams still run)"
        run_bg dx "" "python -m vint_train.mapmad.checks_g10 --config config/mapmad_v2.yaml --v1-config config/mapmad_p4_C.yaml \
            --reference /outputs/mapmad/p10_v2_train/checks/reference_forward_cpu.pt --out $o/g10_rows_1_2_cpu.json" \
            >> "$P13/driver/offline_v21b_ema9.log" 2>&1 || stop "G10 rows 1-2 (a code check) failed (driver/offline_v21b_ema9.log)"
        say "offline checks done (offline/v21b_ema9)"
    fi
    if [[ ! -f "$P13/checks/determinism.json" ]]; then
        STEP="3 determinism (row 6a)"
        status step="$STEP" --state WAITING
        g="$(wait_gpu)"
        local e; e="$(plan episodes examb | head -1)"
        for r in a b; do
            [[ -f "$P13/checks/determinism_$r/logs/M21b/${e}__s0.jsonl" ]] && continue
            GPU="$g" TOKEN="det_${r}_$(date +%H%M%S)" OWNER="$WHO" PAIRS_PER_GPU=1 PORT0=5600 RPORT0=5750 RECOG=yoloe \
                run_limit 1800 "checks/determinism_$r" bash "$REPO/mapmad/scripts/run_p13_pairs.sh" --config "$CFG" \
                --subdir "checks/determinism_$r" --set examb --arms M21b --episodes "$e" --seeds 0 >> "$P13/driver/determinism.log" 2>&1
        done
        if cmp "$P13/checks/determinism_a/logs/M21b/${e}__s0.jsonl" "$P13/checks/determinism_b/logs/M21b/${e}__s0.jsonl" >> "$LOG" 2>&1; then
            echo "{\"pass\": true, \"summary\": \"$e, M21b + YOLOE, seed 0, twice on GPU $g: byte-identical logs\"}" > "$P13/checks/determinism.json"
        else
            echo "{\"pass\": false, \"summary\": \"$e twice on GPU $g: logs differ\"}" > "$P13/checks/determinism.json"
            stop "determinism mismatch on the same GPU ($e, GPU $g, checks/determinism_a|b)"
        fi
        say "determinism episode $e identical on GPU $g (M21b)"
    fi

    # --- 4. 13.6 dev sanity ------------------------------------------------------------------------------------------------
    CURRENT_STEP="4 dev sanity (13.6)"
    step_runs dev "$DEV_BUDGET"
    say "dev sanity done"

    # --- 5. timing + budget ------------------------------------------------------------------------------------------------
    STEP="5 timing"
    status step="$STEP"
    local tline teps label subdir set recog seeds arms n
    { read -r tline; read -r teps; } < <(plan timing)
    IFS='|' read -r label subdir set recog seeds arms n <<< "$tline"
    if [[ ! -f "$P13/driver/timing.json" ]]; then
        local t_total=$(( $(echo $teps | wc -w) * 2 * 3 ))
        CL_EXTRA="--episodes $teps" closed_loop timing "$subdir" "$set" "$t_total" "$EXAM_BUDGET" "$recog" "" $arms
        code=$?
        CL_EXTRA=""
        (( code == 0 )) || stop "timing runs failed (code $code)"
        python3 "$SCRIPTS/p13_estimate.py" "$P13" "$HCFG" > "$P13/driver/timing.json" || stop "timing estimate failed"
    fi
    local est_s pairs_now
    read -r est_s pairs_now < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(int(d['estimate_s']), d['pairs'])" "$P13/driver/timing.json")
    say "timing: $(cat "$P13/driver/timing.json" | tr -d '\n' | cut -c1-400)"
    (( est_s * 2 > EXAM_BUDGET * 3 )) && stop "budget: exam estimate $(( est_s / 3600 )) h > 1.5 x 40 h" \
        "run all exams anyway, or drop the report-only arms (VF, N on exam B)? (recommended: drop VF on HM3D first)"
    status come_back="$(date -d "+$(( est_s + 7200 )) seconds" '+%a %d %b ~%H:%M')" estimate="exams ~$(( est_s / 3600 )) h on $pairs_now pairs + port check + tables ~2 h"

    # --- 6. 13.7 exams -------------------------------------------------------------------------------------------------------
    CURRENT_STEP="6 exams (13.7)"
    step_runs exams $(( EXAM_BUDGET * 3 / 2 ))
    say "exams done ($(( (EXAM_BUDGET * 3 / 2 - STAGE_LEFT) / 3600 )) h of GPU-running time)"

    # --- 7. VLFM port check; HSSD ------------------------------------------------------------------------------------------
    STEP="7 VLFM port check"
    status step="$STEP" --state WAITING
    if [[ -x "$SCRIPTS/vlfm_port.sh" && ! -f "$P13/vlfm_port/run_100/summary.json" ]]; then
        g="$(wait_gpu)"
        status gpus="$g" --state RUNNING
        say "GPU event: GPU $g for the VLFM port check (100 HM3D v2 val episodes)"
        run_limit 32400 "vlfm" env MAPMAD_P13_OWNER="$WHO" bash "$SCRIPTS/vlfm_port.sh" "$g" "${P13_VLFM_N:-all}" "$P13/vlfm_port/run_100" \
            >> "$P13/driver/vlfm_port.log" 2>&1 || fallback "port check: failed to run (driver/vlfm_port.log): reported as not done"
    elif [[ ! -x "$SCRIPTS/vlfm_port.sh" ]]; then
        fallback "port check: the original VLFM stack could not be set up in Part 1: reported as not done"
    fi
    fallback "HSSD: not cheap (dataset + ObjectNav episodes + MP3D-style checks > 2 h of work): skipped (kickoff §3 item 10)"

    # --- 8. scores + fairness + tables ----------------------------------------------------------------------------------------
    STEP="8 tables"
    status step="$STEP" gpus="" subdir="" arms=""
    for stage in dev exams; do
        while IFS='|' read -r label subdir set recog seeds arms n; do
            run_bg dx "" "python -m vint_train.mapmad.closed_loop.p12_score score --config $CFG --subdir $subdir --set $set \
                --arms $arms" >> "$P13/driver/tables.log" 2>&1 || stop "scoring $label failed (driver/tables.log)"
        done < <(plan calls "$stage")
    done
    run_bg dx "" "python -m vint_train.mapmad.closed_loop.p13_tables fairness --config $CFG && \
        python -m vint_train.mapmad.closed_loop.p13_tables tables --config $CFG" >> "$P13/driver/tables.log" 2>&1 \
        || stop "tables failed (driver/tables.log)"
    say "tables written (tables/tables.md); G13 draft: G13_table_draft.md"
    STEP="8 G13 draft"
    FINISHED=1
    status step="$STEP" note="draft G13: G13_table_draft.md; tables/tables.md; videos/small/" eta="" gpus="" --state "CHAIN DONE"
    date '+%F %T' > "$P13/driver/CHAIN_DONE.txt"
    say "CHAIN DONE"
}

main "$@"
exit $?
