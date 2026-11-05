#!/bin/bash
# Phase 14b-3 chained driver (kickoff .p14b3_kickoff.md v1.0; rules p14b3_rules.md). Host side, in screen p14b3_chain
# (start with launch_p14b3.sh). Steps (every one resumes; budgets in GPU-hours = wall time x GPUs, one unit):
#   0. pre-flight: committed code, rules sha256 (+ the exam-rules value frozen here once), checkpoints, sets, dev choices;
#   1. V1 DAgger round 2: v2.1b drives the round-2 start set with maps shown, then hidden (FT YOLOE marks, W4 + A-mark
#      off, --save-drives); labels (Phase 11 labeller + W1 recovery labels) per mode; splits; contamination check;
#      budget 16 GPU-h -> rollouts stop AT the budget (map gets 8, hidden the rest), >= 50 % per mode needed;
#   2. data index (frontier cache of the round-2 map drives, sample index + LMDB, union / equal-count report);
#   3. restart naz_mapmad -> V2 fine-tune v3 (ema_9 fixed) -> V3.json; G10 rows 1-2 (must pass) + 3-4 (recorded);
#      recovery-label loss (falls? else fallback W1-turn: relabel turn-only, retrain once, reverse controller off);
#      past-move error report;
#   4. determinism (row 8a) + Phase 13 exam-B M21b reproduction and the 14b-2 reuse check per GPU (row 8b);
#   5. V3 practice check (M3_P, M3h_P, M3_Pnoam; seed 0) + table; forward/back oscillation > 20 % of reverse steps ->
#      fallback: reverse controller off for the exam arms;
#   6. V4 exam B: the exam-rules sha256 check, then the calls in the pre-set order until 1.5 x 18 GPU-h; cut calls
#      -> cut.json ("not run");
#   7. fairness (row 7), rules sha256 after, 12 videos, tables, draft G14b3 -> CHAIN DONE.
# Automatic stops -> STOP_reason.txt, STATUS.md STOPPED, wait. Progress watchdog: closed_loop's SILENT_S (30 min).
# DRY (never results): P14_DIR=<host p14b3_dry/chain>, P14_ALLOW_DIRTY=1, P14_GPUS="0 1", P14_EPISODES="<ids>",
# P14B3_DRY=1 (tiny rollout sets, 1-epoch tiny training: P14B3_TRAIN_EXTRA), CL_CHECK_S=30, SILENT_S=600.
set -uo pipefail

main() {
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    OUT_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    DATA_MOUNT="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/mapmad_data"}}{{.Source}}{{end}}{{end}}' naz_mapmad)"
    P14="${P14_DIR:-$OUT_MOUNT/mapmad/p14b3_v3}"
    C="${P14/$OUT_MOUNT//outputs}"
    WHO=chain
    LOG="$P14/driver/chain.log"
    mkdir -p "$P14/driver" "$P14/checks" "$P14/launch" "$P14/labels"
    exec 2>> "$LOG"
    export P14_LOGNAME=phase14b3_log.md P14_PHASE=14b-3
    export P14_PASTE="continue Phase 14b-3: read STATUS.md and the chain log, then finish the tables, the G14b3 table and the summary, and close the gate if the must-pass rows pass"
    export P14_CFG="${P14_CFG:-/app/visualnav-transformer/mapmad/configs/p14b3_v3.yaml}"
    # shellcheck source=p14_lib.sh
    source "$REPO/train/scripts/mapmad_p14/p14_lib.sh"
    export P14_OUTPUTS="$OUT_MOUNT/mapmad" P14_EPISODES="${P14_EPISODES:-}" CL_UNIT_DIV=1  # budgets: wall x GPUs
    local ROLL_MAP_CFG="${P14B3_ROLL_MAP:-/app/visualnav-transformer/mapmad/configs/p14b3_rollout_map.yaml}"
    local ROLL_HID_CFG="${P14B3_ROLL_HID:-/app/visualnav-transformer/mapmad/configs/p14b3_rollout_hidden.yaml}"
    local TRAIN_CFG="${P14B3_TRAIN_CFG:-$TRAIN_DIR/config/mapmad_v3.yaml}"
    local MODELS=/outputs/mapmad/p14b2_recog/models.yaml
    local R2="${P14B3_R2_SUFFIX:-}"            # DRY: "_dry" (own round-2 data folders and splits)
    local ROLL_S=$(( 16 * 3600 ))            # V1 <= 16 GPU-h: rollouts stop AT the budget (kickoff §3 fallback)
    local TRAIN_S=$(( 15 * 3600 / 2 ))       # V2 <= 5 GPU-h (stop > 1.5x)
    local PRAC_S=$(( 9 * 3600 ))             # V3 <= 6 GPU-h (stop > 1.5x)
    local EXAM_S=$(( 27 * 3600 ))            # V4 <= 18 GPU-h; calls cut at 1.5x (kickoff §3 fallback, no stop)
    export PAIRS_PER_GPU="${P14B3_PAIRS:-4}"
    FINISHED=0
    BG_PID=""
    LBL_PID=""
    [[ -f "$P14/STOP_reason.txt" ]] && mv "$P14/STOP_reason.txt" "$P14/driver/STOP_reason_$(date +%m%d_%H%M%S).txt"

    stop() {  # reason [question] -> automatic stop; waits for Naz
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
        [[ -n "${LBL_PID:-}" ]] && kill "$LBL_PID" 2>/dev/null
        for pat in "run_search" "sidecar_server" "mapmad_recog.server" "p11_labels.py" "dataset_dagger" \
                   "launch/v3" "offline/v3" "v3_offline" "p14b3_tables" "p14b3_checks" "p14_videos.py"; do
            kill_inside "$pat"; done  # our own processes only (patterns of this chain)
        if [[ $FINISHED -ne 1 ]]; then
            { echo "time: $(date '+%F %T')"; echo "step: ${STEP:-?}"; echo "exit code: $code"
              echo "question: relaunch launch_p14b3.sh after reading driver/chain.log? (recommended: yes, it resumes)"
              echo "last log line: $(tail -n 1 "$LOG" 2>/dev/null)"; } > "$P14/STOP_reason.txt"
            status reason="driver exit $code" error="driver exit $code" --state STOPPED
            say "STOP (exit $code): see STOP_reason.txt"
        fi
    }
    trap on_exit EXIT
    trap 'exit 143' TERM INT

    plan() { python3 "$SCRIPTS/p14_plan.py" "$HCFG" "$@"; }
    fallback() { status fallback="$1"; say "FALLBACK: $1"; echo "$1" >> "$P14/driver/fallbacks.txt"; }
    dxe() { dx "$1" "export P14_EPISODES='$P14_EPISODES' P14_OUTPUTS=/outputs/mapmad; ${*:2}"; }
    dh_g() { docker exec -e MAPMAD_P13_OWNER="$WHO" -e P14_EPISODES="$P14_EPISODES" -w /app/visualnav-transformer naz_mapmad_habitat bash -lc "$*"; }
    has_fb() { grep -q -- "$1" "$P14/driver/fallbacks.txt" 2>/dev/null; }
    idp() { [[ "$1" == examb ]] && echo "p13b-" || echo "$1-"; }  # a set's episode-id prefix (exam B ids: p13b-)
    gpu_used() { echo "GPU-time used $(( $1 / 60 )) min of $(( $2 / 60 ))"; }
    owners() {  # owner table + one decision line (kickoff §5) before a launch
        { echo "### $(date '+%m-%d %H:%M') owner check before: $1"; nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader
          free -h | sed -n 2p; uptime; docker ps --format '{{.Names}}'; ps -eo user --no-headers | sort | uniq -c | tr '\n' ' '; echo; } \
          >> "$P14/driver/owners.log" 2>&1
        say "owner check before $1: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) GPU processes; decision: free GPUs by the 20-min watcher, $PAIRS_PER_GPU pairs per GPU, Alone-mode thread cap"
    }

    # --- 0. pre-flight ----------------------------------------------------------------------------------------------
    STEP="0 pre-flight"
    status --reset step="$STEP" started="$(date '+%a %d %b %H:%M')" mode="Alone" --state RUNNING
    say "chain start (P14=$P14, CFG=$CFG, pairs/GPU $PAIRS_PER_GPU, episodes '${P14_EPISODES}')"
    if [[ -z "${P14_ALLOW_DIRTY:-}" ]] && [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
        stop "uncommitted changes in tracked files (launcher commit first)"
    fi
    say "code at $(git -C "$REPO" rev-parse HEAD)"
    local RULES="$OUT_MOUNT/mapmad/p14b3_v3/p14b3_rules.md" want
    want="$(cut -d' ' -f1 "$OUT_MOUNT/mapmad/p14b3_v3/p14b3_rules.sha256")"
    [[ "$(sha256sum "$RULES" | cut -d' ' -f1)" == "$want" ]] || stop "p14b3_rules.md sha256 mismatch"
    [[ "$(sha256sum "$REPO/mapmad/configs/p14b3_rules.md" | cut -d' ' -f1)" == "$want" ]] || stop "repo copy of p14b3_rules.md differs"
    [[ -f "$P14/checks/exam_rules_frozen.sha256" ]] || echo "$want  p14b3_rules.md (frozen before the first exam run)" \
        > "$P14/checks/exam_rules_frozen.sha256"
    if [[ "$P14" != "$OUT_MOUNT/mapmad/p14b3_v3" ]]; then  # DRY: the rules check and the contamination merge read here
        cp -p "$RULES" "$P14/p14b3_rules.md"
        cp -p "$OUT_MOUNT/mapmad/p14b3_v3/checks/contamination_rollout_set.json" "$P14/checks/"
    fi
    run_bg dh_g "python mapmad/scripts/p14b3_preflight.py --config $CFG --rules /outputs/mapmad/p14b3_v3/p14b3_rules.md" \
        >> "$P14/driver/preflight.log" 2>&1 || stop "pre-flight failed: checkpoint sha256 / set fingerprints (driver/preflight.log)"
    [[ -f "$P14/P12_DEV_CHOICES.json" ]] || cp -p "$OUT_MOUNT/mapmad/p14a_finish/P12_DEV_CHOICES.json" "$P14/"
    say "pre-flight done: $(tail -n 1 "$P14/driver/preflight.log")"

    # --- 1. V1 DAgger round 2 ---------------------------------------------------------------------------------------
    local used_roll=0 mode rcfg arm n_roll done crashed code budget share
    for mode in map hidden; do
        [[ $mode == map ]] && { rcfg="$ROLL_MAP_CFG"; arm=r2map; } || { rcfg="$ROLL_HID_CFG"; arm=r2hidden; }
        STEP="1 V1 rollouts ($mode)"
        status step="$STEP"
        n_roll="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['episodes']))" "$OUT_MOUNT/mapmad/p14b3_v3/rollouts_r2/set/episodes.json")"
        [[ -n "$P14_EPISODES" ]] && n_roll="$(echo $P14_EPISODES | tr ' ' '\n' | grep -c '^p11r-')"
        read -r done crashed < <(python3 "$SCRIPTS/p14_status.py" "$P14" count "rollouts_r2/$mode/rollouts" "$arm")
        if [[ ! -f "$P14/rollouts_r2/$mode/ROLLOUTS_DONE" ]]; then
            [[ $mode == map ]] && budget=$(( ROLL_S / 2 ))
            [[ $mode == hidden ]] && budget=$(( ROLL_S - used_roll ))
            owners "V1 rollouts $mode"
            RECOG_CONFIG="$MODELS" CL_CFG="$rcfg" CL_PREFIX="rollouts_r2/$mode/" CL_EP_PREFIX="p11r-" \
                CL_EXTRA="--save-drives${P14_EPISODES:+ --episodes $P14_EPISODES}" \
                closed_loop "roll_$mode" rollouts rollout "$n_roll" "$budget" yoloe_ft "0" "$arm"
            code=$?
            used_roll=$(( used_roll + CL_USED_S ))
            (( code == 2 )) && stop "roll_$mode: runners exit without progress (driver/roll_${mode}_gpu*.log)"
            (( code == 3 )) && stop "roll_$mode: recogniser down after 3 restarts"
            read -r done crashed < <(python3 "$SCRIPTS/p14_status.py" "$P14" count "rollouts_r2/$mode/rollouts" "$arm")
            (( crashed * 100 <= n_roll )) || stop "roll_$mode: crash rate > 1% ($crashed of $n_roll)"
            if (( code == 1 )); then
                (( done * 2 >= n_roll )) || stop "roll_$mode: budget reached with $done of $n_roll rollouts (< 50 %)" \
                    "the kickoff needs >= 50 % of the start set per mode: raise the budget? (recommended: yes, +4 GPU-h)"
                fallback "V1 $mode: rollouts stopped at the budget, $done of $n_roll used (kickoff §3)"
            fi
            echo "$done $(( CL_USED_S ))" > "$P14/rollouts_r2/$mode/ROLLOUTS_DONE"
            say "V1 $mode: $done rollouts ($crashed crashed); $(gpu_used "$used_roll" "$ROLL_S")"
        else
            used_roll=$(( used_roll + $(cut -d' ' -f2 "$P14/rollouts_r2/$mode/ROLLOUTS_DONE") ))
        fi
        # labels of this mode in the background (CPU) while the next mode drives
        [[ -n "$LBL_PID" ]] && { wait "$LBL_PID" || true; LBL_PID=""; }
        has_fb "W1-turn" && [[ -f "$P14/launch/rollout_${mode}_turn.yaml" ]] && rcfg="$C/launch/rollout_${mode}_turn.yaml"
        ( dh_g "python mapmad/scripts/p11_labels.py label --config $rcfg --workers ${P14B3_LABEL_WORKERS:-12}" \
              >> "$P14/driver/labels_$mode.log" 2>&1; echo $? > "$P14/driver/labels_$mode.code" ) &
        LBL_PID=$!
    done
    STEP="1 V1 labels + splits"
    status step="$STEP"
    [[ -n "$LBL_PID" ]] && { wait "$LBL_PID" || true; LBL_PID=""; }
    for mode in map hidden; do
        [[ $mode == map ]] && rcfg="$ROLL_MAP_CFG" || rcfg="$ROLL_HID_CFG"
        if has_fb "W1-turn" && [[ -f "$P14/launch/rollout_${mode}_turn.yaml" ]]; then  # resume after the fallback
            rcfg="$C/launch/rollout_${mode}_turn.yaml"; echo 1 > "$P14/driver/labels_$mode.code"
        fi
        code="$(cat "$P14/driver/labels_$mode.code" 2>/dev/null || echo 1)"
        if [[ "$code" != 0 ]]; then  # the first pass may have run with a half-written last drive: one full pass more
            run_bg dh_g "python mapmad/scripts/p11_labels.py label --config $rcfg --workers 24" >> "$P14/driver/labels_$mode.log" 2>&1
            code=$?
        fi
        (( code == 3 )) && fallback "V1 $mode: teacher-fail share > 0.40 (recorded, Phase 13 rule: no stop)"
        (( code == 0 || code == 3 )) || stop "V1 labels ($mode) failed (driver/labels_$mode.log)"
        run_bg dh_g "python mapmad/scripts/p11_labels.py splits --config $rcfg" >> "$P14/driver/labels_$mode.log" 2>&1 \
            || stop "V1 splits ($mode) failed"
        say "V1 labels $mode: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('drives', d['drives'], 'recovery share of labels', d['recovery_share_of_labels'], 'teacher fail', round(d['teacher_fail_share'],4), 'labelled', d['overall']['ok'])" "$P14/rollouts_r2/$mode/labels/label_stats.json")"
    done
    local sp="/mapmad_data/splits"
    run_bg dh_g "python mapmad/scripts/make_p14b3_sets.py contamination --outputs /outputs/mapmad --out-dir $C \
        --drives /mapmad_data/habitat_mapmad_v3_dagger_r2map$R2 /mapmad_data/habitat_mapmad_v3_dagger_r2hidden$R2 \
        $sp/habitat_mapmad_v2/train/traj_names.txt $sp/habitat_mapmad_v2_dagger/train/traj_names.txt \
        $sp/habitat_mapmad_v2_dagger_hidden/train/traj_names.txt $sp/habitat_mapmad_v3_dagger_r2map$R2/train/traj_names.txt \
        $sp/habitat_mapmad_v3_dagger_r2hidden$R2/train/traj_names.txt" >> "$P14/driver/checks.log" 2>&1 \
        || stop "CONTAMINATION check failed (checks/contamination.json)" "gate row 6 cannot pass: investigate? (recommended: yes)"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks contamination --config $CFG" >> "$P14/driver/checks.log" 2>&1 \
        || stop "contamination merge failed (checks/checks.json)"

    # --- 2. data index ------------------------------------------------------------------------------------------------
    STEP="2 data index"
    status step="$STEP"
    if [[ ! -f "$P14/labels/union.json" ]]; then
        run_bg dxe "" "python -m vint_train.mapmad.dataset_dagger frontier --config $TRAIN_CFG --block 0 --workers 20 \
            --out $C/labels/frontier_r2map.json" >> "$P14/driver/index.log" 2>&1 \
            || stop "frontier cache of the round-2 map drives failed or differs from the live count (labels/frontier_r2map.json)"
        run_bg dxe "" "python -m vint_train.mapmad.dataset_dagger prepare --config $TRAIN_CFG --out $C/labels/union.json" \
            >> "$P14/driver/index.log" 2>&1 || stop "data index (prepare) failed (driver/index.log)"
        run_bg dh_g "python mapmad/scripts/p11_labels.py sheet --config $ROLL_MAP_CFG" >> "$P14/driver/index.log" 2>&1 \
            || say "label sheet failed (report only)"
    fi
    say "data index: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('map-DAgger parts', d.get('dagger_part_samples'), 'hidden', {k: d['hidden'].get(k) for k in ('hidden_part_samples', 'hidden_q', 'expected_draws_per_epoch', 'hidden_repeat_factor')})" "$P14/labels/union.json")"

    # --- 3. restart, V2 fine-tune, offline ------------------------------------------------------------------------------
    local RUN_DIR g ck tag=v3 tcfg="$TRAIN_CFG" relabelled=0
    if has_fb "W1-turn"; then tag=v3t; fi
    train_v3() {  # tag base_config -> V3.json (restart, fine-tune, ema_9)
        local t="$1" base="$2" k busy
        STEP="3 restart naz_mapmad"
        status step="$STEP"
        for k in $(seq 1 20); do
            busy="$(docker exec naz_mapmad ps -eo stat,pid,args | awk '$1 !~ /^Z/' | grep -E 'python|train.py' \
                | grep -v -E 'ps -eo|grep' || true)"
            [[ -z "$busy" ]] && break
            say "naz_mapmad still runs: $(echo "$busy" | head -3 | tr '\n' ';') -> waiting 1 min"
            pause 60
        done
        [[ -z "$busy" ]] || stop "naz_mapmad still runs processes after 20 min; not restarted"
        run_bg docker restart naz_mapmad >> "$P14/driver/restart.log" 2>&1 || stop "docker restart naz_mapmad failed"
        for k in $(seq 1 30); do
            docker exec naz_mapmad python -c 'import torch; assert torch.cuda.device_count() == 2' >> "$P14/driver/restart.log" 2>&1 && break
            pause 10
        done
        docker exec naz_mapmad python -c 'import torch; assert torch.cuda.device_count() == 2' >> "$P14/driver/restart.log" 2>&1 \
            || stop "naz_mapmad not healthy after the restart (driver/restart.log)"
        say "naz_mapmad restarted and healthy"
        STEP="3 V2 fine-tune $t"
        status step="$STEP" --state WAITING
        owners "V2 fine-tune $t"
        g="$(train_gpu)"
        local extra=() kv
        IFS='|' read -r -a extra <<< "${P14B3_TRAIN_EXTRA:-}"
        overlay "$P14/launch/$t.yaml" "$base" "run_name: ${t}_s1" "run_label: ${t}_s1" "logs_root: $C" "${extra[@]}"
        status step="$STEP" gpus="$g" epochs=10 eta="fine-tune ~2.5 h" --state RUNNING
        say "GPU event: GPU $g for the $t fine-tune (10 epochs)"
        local t0=$SECONDS
        supervised_train "${t}_s1" "$g" "$C/launch/$t.yaml" 9 "$P14"
        code=$?
        (( code == 0 )) || stop "$STOP_WHY"
        (( SECONDS - t0 <= TRAIN_S )) || stop "budget: V2 fine-tune took $(( (SECONDS - t0) / 60 )) min > 1.5 x 5 GPU-h"
        mkdir -p "$P14/offline/$t"
        sha256sum "$RUN_DIR/ema_9.pth" > "$P14/offline/$t/ema_9.sha256"
        python3 - "$RUN_DIR" "$P14" "$OUT_MOUNT" "$base" <<'EOF' || stop "V3.json not written"
import json, sys
run, p14, out, base = sys.argv[1:]
sha = open(f"{p14}/offline/{run.rstrip('/').split('/')[-1].split('_s1_')[0]}/ema_9.sha256").read().split()[0]
rel = run.replace(out + "/mapmad/", "", 1) + "/ema_9.pth"
cfg = "repo:" + base.replace("/app/visualnav-transformer/", "", 1)
json.dump({"path": rel, "sha256": sha, "config": cfg}, open(f"{p14}/V3.json", "w"), indent=1)
EOF
        say "GPU event: fine-tune done on GPU $g; ema_9 sha256 $(cut -c1-16 "$P14/offline/$t/ema_9.sha256")"
    }
    if [[ ! -f "$P14/V3.json" ]]; then
        train_v3 "$tag" "$tcfg"
    fi
    offline_v3() {  # tag train_config -> offline/<tag>: G10 rows 1-4, recovery loss, past report
        local t="$1" cfgf="$2" o="$C/offline/$1"
        STEP="3 offline checks ($t)"
        status step="$STEP" --state WAITING
        g="$(wait_gpu 1)"
        status gpus="$g" --state RUNNING
        ck="/outputs/mapmad/$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['path'])" "$P14/V3.json")"
        if [[ ! -f "$P14/offline/$t/g10_rows_1_2_cpu.json" ]]; then
            run_bg dx "" "python -m vint_train.mapmad.checks_g10 --config $cfgf --v1-config config/mapmad_p4_C.yaml \
                --reference /outputs/mapmad/p10_v2_train/checks/reference_forward_cpu.pt \
                --weights /outputs/mapmad/weights/official/nomad.pth --out $o/g10_rows_1_2_cpu.json" \
                >> "$P14/driver/offline_$t.log" 2>&1 || stop "G10 rows 1-2 (a code check) failed to run (driver/offline_$t.log)"
        fi
        [[ "$t" == v3 ]] || cp -p "$P14/offline/$t/g10_rows_1_2_cpu.json" "$P14/offline/v3/g10_rows_1_2_cpu.json" 2>/dev/null
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks g10 --config $CFG" >> "$P14/driver/checks.log" 2>&1 \
            || stop "G10 rows 1-2 FAIL (gate row 5: a code bug; two honest fixes, then a safety stop)" \
                    "fix the past-token masking? (recommended: read offline/$t/g10_rows_1_2_cpu.json)"
        [[ -f "$P14/offline/$t/search/compare.json" ]] || offline_checks "$g" "$t" "$ck" "$cfgf" \
            || fallback "V2: G10 rows 3-4 checks failed to run (recorded, the phase continues)"
        [[ -f "$P14/offline/$t/recovery.json" ]] || run_bg dxe "$g" "python -m vint_train.mapmad.v3_offline recovery \
            --config $cfgf --checkpoint v21b=/outputs/mapmad/p13_claim/runs/v21b_s1_2026_10_30_17_02_10/ema_9.pth $t=$ck \
            --configs v21b=config/mapmad_v21b.yaml --n ${P14B3_REC_N:-1000} --out $o/recovery.json" \
            >> "$P14/driver/offline_$t.log" 2>&1 || stop "recovery-label loss failed to run (driver/offline_$t.log)"
        run_bg dxe "" "python -m vint_train.mapmad.v3_offline past --run ${RUN_DIR/$OUT_MOUNT//outputs} --out $o/past.json" \
            >> "$P14/driver/offline_$t.log" 2>&1 || say "past-move report failed (report only)"
        say "offline $t: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('recovery loss', d.get('means'), d.get('difference'), 'falls', d.get('falls'))" "$P14/offline/$t/recovery.json")"
    }
    [[ -n "${RUN_DIR:-}" ]] || RUN_DIR="$OUT_MOUNT/mapmad/$(python3 -c "import json,sys,os; print(os.path.dirname(json.load(open(sys.argv[1]))['path']))" "$P14/V3.json")"
    offline_v3 "$tag" "$tcfg"
    if [[ "$tag" == v3 ]] && ! python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('falls') else 1)" "$P14/offline/v3/recovery.json"; then
        fallback "W1-turn: the recovery-label loss does not fall (offline/v3/recovery.json) -> turn-only recovery labels, retrain once, reverse controller off"
        for mode in map hidden; do
            [[ $mode == map ]] && rcfg="$ROLL_MAP_CFG" || rcfg="$ROLL_HID_CFG"
            printf 'extends: %s\nrollouts:\n  recovery:\n    mode: turn\n' "$rcfg" > "$P14/launch/rollout_${mode}_turn.yaml"
            run_bg dh_g "python mapmad/scripts/p11_labels.py label --config $C/launch/rollout_${mode}_turn.yaml --workers 24" \
                >> "$P14/driver/labels_turn.log" 2>&1; code=$?
            (( code == 0 || code == 3 )) || stop "W1-turn relabel ($mode) failed"
        done
        mv "$P14/V3.json" "$P14/driver/V3_reverse_turn_$(date +%m%d_%H%M%S).json"
        tag=v3t
        train_v3 v3t "$tcfg"
        offline_v3 v3t "$tcfg"
    fi
    local V3CFG="$CFG"
    if has_fb "W1-turn" || has_fb "oscillation"; then  # the reverse controller off for every v3 arm
        printf 'extends: %s\narms:\n' "$CFG" > "$P14/launch/p14b3_noreverse.yaml"
        for a in M3_P M3h_P M3_Pnoam; do printf '  %s: {reverse: none}\n' "$a" >> "$P14/launch/p14b3_noreverse.yaml"; done
        V3CFG="$C/launch/p14b3_noreverse.yaml"
    fi

    # --- 4. determinism + reproductions -------------------------------------------------------------------------------
    export RECOG_CONFIG="$MODELS"
    local label subdir set recog seeds arms_s n eps og
    if [[ ! -f "$P14/checks/det_ok" ]]; then
        STEP="4 determinism (row 8a)"
        status step="$STEP" --state WAITING
        og="$(wait_gpu)"
        while IFS='|' read -r label subdir set recog seeds arms_s n eps; do
            GPU="$og" TOKEN="${label}_$(date +%m%d_%H%M%S)" OWNER="$WHO" PORT0=5560 RPORT0=5700 RECOG=yoloe_ft PAIRS_PER_GPU=1 \
                run_limit 3600 "run_search .*--subdir $subdir " bash "$REPO/mapmad/scripts/run_p13_pairs.sh" --config "$V3CFG" \
                --subdir "$subdir" --set "$set" --arms $arms_s --seeds 0 --episodes $eps >> "$P14/driver/${label}.log" 2>&1 \
                || stop "determinism runs $label failed (driver/${label}.log)"
        done < <(plan calls det)
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks det --config $V3CFG" >> "$P14/driver/checks.log" 2>&1 \
            || stop "determinism mismatch on the same GPU (checks/checks.json)" "investigate? (recommended: yes)"
        touch "$P14/checks/det_ok"
    fi
    if [[ ! -f "$P14/checks/off_ok" ]]; then
        STEP="4 Phase 13 reproduction + 14b-2 reuse check (row 8b)"
        status step="$STEP"
        for og in 0 1; do
            while IFS='|' read -r label subdir set recog seeds arms_s n eps; do
                [[ "$label" == *_g$og ]] || continue
                [[ -n "$eps" ]] || { say "$label: no reference run on GPU $og (skipped)"; continue; }
                until gpu_clean "$og"; do pause 60; done
                local rc=yoloe_ft rcf="$MODELS"
                [[ "$recog" == yoloe ]] && { rc=yoloe; rcf=""; }
                GPU="$og" TOKEN="${label}_$(date +%m%d_%H%M%S)" OWNER="$WHO" PORT0=5560 RPORT0=5700 RECOG="$rc" PAIRS_PER_GPU=1 \
                    RECOG_CONFIG="$rcf" run_limit 3600 "run_search .*--subdir $subdir " bash "$REPO/mapmad/scripts/run_p13_pairs.sh" \
                    --config "$CFG" --subdir "$subdir" --set "$set" --arms $arms_s --seeds 0 --episodes $eps \
                    >> "$P14/driver/$label.log" 2>&1 || stop "reproduction runs $label failed (driver/$label.log)"
            done < <(plan calls off)
        done
        run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks off --config $CFG" >> "$P14/driver/checks.log" 2>&1 \
            || stop "a Phase 13 / 14b-2 run is not reproduced on its GPU (checks/checks.json)" "investigate? (recommended: yes)"
        touch "$P14/checks/off_ok"
    fi

    # --- 5. V3 practice check ------------------------------------------------------------------------------------------
    local used_prac=0
    while IFS='|' read -r label subdir set recog seeds arms_s n eps; do
        STEP="5 V3 practice: $label"
        status step="$STEP" --state RUNNING
        owners "V3 $label"
        CL_CFG="$V3CFG" CL_EP_PREFIX="$(idp "$set")" CL_SEEDS="$seeds" CL_EXTRA="${eps:+--episodes $eps}" \
            closed_loop "$label" "$subdir" "$set" "$n" "$(( PRAC_S - used_prac ))" yoloe_ft "$seeds" $arms_s
        code=$?
        used_prac=$(( used_prac + CL_USED_S ))
        (( code == 1 )) && stop "budget: V3 practice exceeded 1.5 x 6 GPU-h"
        (( code == 2 )) && stop "$label: runners exit without progress (driver/${label}_gpu*.log)"
        (( code == 3 )) && stop "$label: recogniser down after 3 restarts"
        CL_EP_PREFIX="$(idp "$set")" crash_rate_ok "$subdir" "$n" $arms_s || stop "$label: crash rate > 1% ($subdir)"
    done < <(plan calls practice)
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_tables practice --config $V3CFG --workers 12" \
        >> "$P14/driver/tables.log" 2>&1 || stop "practice table failed (driver/tables.log)"
    if [[ "$V3CFG" == "$CFG" ]] && python3 - "$P14/tables/practice.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1]))["arms"]["all"]
osc = [a[k].get("reverse_osc_share") for k in ("M3_P", "M3h_P") if a.get(k, {}).get("runs")]
sys.exit(0 if any(x is not None and x > 0.20 for x in osc) else 1)
EOF
    then
        fallback "oscillation: forward / back > 20 % of reverse steps in the practice check -> reverse controller off for the exam (W1-turn at test time)"
        printf 'extends: %s\narms:\n' "$CFG" > "$P14/launch/p14b3_noreverse.yaml"
        for a in M3_P M3h_P M3_Pnoam; do printf '  %s: {reverse: none}\n' "$a" >> "$P14/launch/p14b3_noreverse.yaml"; done
        V3CFG="$C/launch/p14b3_noreverse.yaml"
    fi
    say "V3 practice done ($(gpu_used "$used_prac" "$PRAC_S"))"

    # --- 6. V4 exam B -----------------------------------------------------------------------------------------------------
    STEP="6 V4 exam rules check"
    status step="$STEP"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks rules --config $CFG --tag before_exam \
        --repo-copy /app/visualnav-transformer/mapmad/configs/p14b3_rules.md" >> "$P14/driver/checks.log" 2>&1 \
        || stop "exam rules sha256 changed since they were frozen (gate row 6)"
    local used_exam=0 cut=() over=0
    [[ -f "$P14/driver/exam_used_s" ]] && used_exam="$(cat "$P14/driver/exam_used_s")"
    while IFS='|' read -r label subdir set recog seeds arms_s n eps; do
        if (( over )); then cut+=("$label"); continue; fi
        STEP="6 V4 exam B: $label"
        status step="$STEP" --state RUNNING budget="$(gpu_used "$used_exam" "$EXAM_S")"
        owners "V4 $label"
        CL_CFG="$V3CFG" CL_EP_PREFIX="$(idp "$set")" CL_SEEDS="$seeds" CL_EXTRA="${eps:+--episodes $eps}" \
            closed_loop "$label" "$subdir" "$set" "$n" "$(( EXAM_S - used_exam ))" yoloe_ft "$seeds" $arms_s
        code=$?
        used_exam=$(( used_exam + CL_USED_S ))
        echo "$used_exam" > "$P14/driver/exam_used_s"
        if (( code == 1 )); then
            over=1; cut+=("$label")
            fallback "V4: exam budget 1.5 x 18 GPU-h reached during $label -> it and every later call reported as not run"
            continue
        fi
        (( code == 2 )) && stop "$label: runners exit without progress (driver/${label}_gpu*.log)"
        (( code == 3 )) && stop "$label: recogniser down after 3 restarts"
        CL_EP_PREFIX="$(idp "$set")" CL_SEEDS="$seeds" crash_rate_ok "$subdir" "$n" $arms_s || stop "$label: crash rate > 1% ($subdir)"
    done < <(plan calls examb)
    python3 -c "import json,sys; json.dump(sys.argv[2:], open(sys.argv[1], 'w'))" "$P14/driver/cut.json" "${cut[@]}"
    say "V4 exam B done ($(gpu_used "$used_exam" "$EXAM_S")); cut: ${cut[*]:-none}"

    # --- 7. checks, videos, tables, gate draft ------------------------------------------------------------------------
    STEP="7 checks"
    status step="$STEP" gpus="" --state RUNNING
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks fairness --config $V3CFG --stages practice examb \
        --cut $C/driver/cut.json" >> "$P14/driver/checks.log" 2>&1 || stop "fairness check failed (checks/checks.json)" "investigate? (recommended: yes)"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_checks rules --config $CFG --tag after_exam \
        --repo-copy /app/visualnav-transformer/mapmad/configs/p14b3_rules.md" >> "$P14/driver/checks.log" 2>&1 \
        || stop "exam rules sha256 changed during the exam (gate row 6)"
    STEP="7 tables"
    status step="$STEP"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_tables examb --config $V3CFG --workers 12" \
        >> "$P14/driver/tables.log" 2>&1 || stop "exam table failed (driver/tables.log)"
    STEP="7 videos"
    status step="$STEP"
    if [[ ! -f "$P14/videos/video_check.json" ]]; then
        local vg vkey vpids=() s vfail=0 vs vsub vpicks
        vg="$(wait_gpu)"
        vkey="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
        s=0
        for vs in practice examb; do
            [[ $vs == practice ]] && { vsub=runs/practice; vpicks="${P14B_VIDEO_PICKS:-/outputs/mapmad/p14a_finish/videos_marks/picks.json}"; } \
                || { vsub=runs/examb; vpicks="${P14B3_VIDEO_PICKS_EXAMB:-/outputs/mapmad/p14b3_v3/videos/picks_examb.json}"; }
            docker exec -d -e MAPMAD_SIDECAR_KEY="$vkey" -e CUDA_VISIBLE_DEVICES="$vg" -e HOME=/model_cache/home \
                -e MAPMAD_P13_OWNER="$WHO" naz_mapmad_recog bash -c \
                "exec python -m mapmad_recog.server --model yoloe_ft --port $(( 5850 + s )) --config $MODELS >> $C/driver/recog_videos_$s.log 2>&1"
            docker exec -e MAPMAD_SIDECAR_KEY="$vkey" -e MAPMAD_P13_OWNER="$WHO" -w /app/visualnav-transformer naz_mapmad_habitat \
                python mapmad/scripts/p14_videos.py render --zoom --config "$V3CFG" --subdir "$vsub" --arm M3_P \
                --picks "$vpicks" --gpu "$vg" --recog-port $(( 5850 + s )) --out "$C/videos/$vs" --shard "0/1" \
                >> "$P14/driver/videos_$vs.log" 2>&1 &
            vpids+=($!)
            s=$(( s + 1 ))
        done
        say "GPU event: GPU $vg for the 12 videos (M3_P, 2 shards)"
        for p in "${vpids[@]}"; do wait "$p" || vfail=1; done
        kill_inside "mapmad_recog.server --model yoloe_ft --port 585"
        (( vfail )) && fallback "V5: a video shard failed (driver/videos_*.log): the gate lists the videos that exist"
        for vs in practice examb; do
            run_bg dh_g "python mapmad/scripts/p13_videos.py check --out $C/videos/$vs" >> "$P14/driver/videos.log" 2>&1 \
                || fallback "V5: video check ($vs) reported a problem"
        done
        touch "$P14/videos/video_check.json"
    fi
    STEP="7 gate draft"
    status step="$STEP"
    run_bg dxe "" "python -m vint_train.mapmad.closed_loop.p14b3_tables gate --config $V3CFG" >> "$P14/driver/tables.log" 2>&1 \
        || stop "gate draft failed (driver/tables.log)"
    FINISHED=1
    status step="$STEP" note="draft G14b3: G14b3_table_draft.md; tables/practice.md, examb.md; videos/" eta="" \
        --state "CHAIN DONE"
    date '+%F %T' > "$P14/driver/CHAIN_DONE.txt"
    say "CHAIN DONE"
}

main "$@"
exit $?
