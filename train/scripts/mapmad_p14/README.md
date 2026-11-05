# scripts/mapmad_p14 — Phase 14a run scripts (host side)

Phase 14a = Goal 2, Step 1: cheap checks so the robot can finish the search — no training, no exam, MP3D practice set
only (p13d ∪ p14d, 219 episodes, halves A / B). Rules: `/outputs/mapmad/p14a_finish/p14a_rules.md` (+ `.sha256`, copy
in `mapmad/configs/p14a_rules.md`); config `mapmad/configs/p14a_finish.yaml` (DRY: `p14a_dry.yaml`). Outputs:
`/mnt/shared_disk/nazli/nomad_outputs/mapmad/p14a_finish/` (`STATUS.md`, `phase14a_log.md`, `sets/`,
`runs/{baseline,a1,w4,a1w4}/` (logs, dets, evidence, summaries), `stopbench/`, `qwen/`, `diagnosis/`, `videos_marks/`,
`tables/`, `checks/`, `G14a_table_draft.md`, `driver/`).

| Script | What it does |
|---|---|
| `sets_p14.sh` | 14.0: p14d candidates on the 12 unused MP3D train homes (one worker per home), then `make_p14_sets.py freeze` (p14d + practice halves) |
| `launch_p14a.sh [dry]` | starts screens `p14a_chain` + `p14a_chain_watch` (STOP_reason.txt if the chain dies without its done file) and `p14a_gpuwatch` (Phase 10's `gpu_watch.py`) if not running |
| `run_p14a_chain.sh` | the chain (header: pre-flight, 14.0b baseline + same-p13 + determinism, 14.1 stop bench, 14.3 diagnosis in the background, 14.2 Qwen check, 14.4 estimate + driving tests + tables, 14.5 mark check + videos, tables + draft G14a) |
| `p14_lib.sh` | Phase 13's `p13_lib.sh` with Phase 14a names; `closed_loop` counts only runs whose episode id starts with `CL_EP_PREFIX` (the practice parts share a run folder) |
| `p14_plan.py` | command line over `mapmad_sim/run_plan_p14.py` (calls per stage, run counts; `P14_EPISODES` limits DRY runs) |
| `p14_status.py` | `STATUS.md` (first line = what to paste on return); `count <subdir> <arms>` |

Closed-loop pairs: Phase 13's `mapmad/scripts/run_p13_pairs.sh` unchanged. Tools: `mapmad/scripts/make_p14_sets.py`,
`p14_qwen_check.py`, `p14_diagnosis.py`, `p14_videos.py`; NoMaD side `vint_train.mapmad.closed_loop.p14_stopbench`,
`p14_tables`, `p14_checks`, `drive_switches`; Habitat side `mapmad_sim/stop_evidence.py`, `stop_rules.py`,
`diagnosis.py`, `mark_check.py`, `run_plan_p14.py`.
Launch (after `nvidia-smi`): `bash train/scripts/mapmad_p14/launch_p14a.sh`. Watch: `cat .../p14a_finish/STATUS.md`,
`screen -r p14a_chain`. DRY: `P14_DIR=<host p14a_dry/chain> P14_CFG=/app/visualnav-transformer/mapmad/configs/p14a_dry.yaml
P14_ALLOW_DIRTY=1 P14_GPUS="0 1" P14_EPISODES="<2 p13d + 2 p14d ids>" P14_DIAG_SAMPLE=4 CL_CHECK_S=30 SILENT_S=600
bash launch_p14a.sh dry`.

## Phase 14b-1 — fix the target marks in the stop module (no training)

Re-scores the 14a practice runs (read only) with fixes F1 (object-level projection: median-depth filter / eroded mask +
DBSCAN largest cluster), F2 (temperature-scaled raw YOLOE scores, no 0.4 floor) and F3 (stop cell ahead and in view
now), inside the stop module only. Rules: `/outputs/mapmad/p14b1_marks/p14b1_rules.md` (+ `.sha256`, copy in
`mapmad/configs/p14b1_rules.md`); config `mapmad/configs/p14b1_marks.yaml` (`runs_from: p14a_finish`; DRY:
`p14b1_dry.yaml`). Outputs: `/mnt/shared_disk/nazli/nomad_outputs/mapmad/p14b1_marks/` (`STATUS.md`,
`phase14b1_log.md`, `replay/`, `calib/`, `stopbench/`, `stopbench_ccat/`, `stopbench_truth/`, `rescore_driving/`,
`videos_marks_fixed/`, `tables/`, `checks/`, `G14b1_table_draft.md`, `driver/`).

| Script | What it does |
|---|---|
| `launch_p14b1.sh [dry]` | starts screens `p14b1_chain` + `p14b1_chain_watch` and `p14b1_gpuwatch` if not running |
| `run_p14b1_chain.sh` | the chain: pre-flight (hashes before, temperature fit), replay of the baseline runs (6 shards per GPU, one YOLOE server each, every run on the GPU that drove it), driving-test replay in the background while the CPU runs the stop bench (score, pick, c_cat, final, marks), driving re-score, old-vs-new videos, hashes after + draft G14b1 |
| `p14_lib.sh`, `p14_status.py` | shared with 14a; `P14_LOGNAME`, `P14_PHASE`, `P14_PASTE` select the 14b-1 names |

Tools: Habitat side `mapmad_sim/mark_projection.py` (F1), `mapmad_sim/stop_variants.py` (F2 / F3 / tentative /
c_cat, grid scorer), `mapmad/scripts/p14b_replay.py`, `p14b_videos.py`; NoMaD side
`vint_train.mapmad.closed_loop.p14b_bench` (calib, score, pick, final, marks, drive, hashes, gate).
Launch (after `nvidia-smi`): `bash train/scripts/mapmad_p14/launch_p14b1.sh`. DRY: see the header of `launch_p14b1.sh`.

## Phase 14b-2 — better recognition + drive-to-the-mark

R1 renders detector training frames in the labelled HM3D train homes (minus the photo-test homes; HM3D val and MP3D
excluded, checked by id lists), R2 fine-tunes YOLOE-11-L-seg (LP and FT) and runs the Phase 9 photo test, R3 replays
the 14a baseline recordings with each fine-tuned detector (stop module only), R4 drives the practice set with the
picked detector and the A-mark switch (drive to an accepted target cluster, `mapmad_sim/amark.py`), R5 writes the
tables, 8 videos and the draft gate. Rules: `/outputs/mapmad/p14b2_recog/p14b2_rules.md` (+ `.sha256`, copy in
`mapmad/configs/p14b2_rules.md`); config `mapmad/configs/p14b2_recog.yaml` (DRY: `p14b2_dry.yaml`). Outputs:
`/mnt/shared_disk/nazli/nomad_outputs/mapmad/p14b2_recog/` (`STATUS.md`, `phase14b2_log.md`, `trainset/`, `yolo/`,
`train/`, `models.yaml`, `photo_test/`, `photo/`, `calib_yoloe_*`, `replay_yoloe_*`, `stopbench_yoloe_*`, `runs/{am,det,detam}`,
`checks/`, `tables/` (plain, photo, r3, r4), `videos_detam/`, `G14b2_table_draft.md`, `driver/`).

| Script | What it does |
|---|---|
| `launch_p14b2.sh [dry]` | starts screens `p14b2_chain` + `p14b2_chain_watch` and `p14b2_gpuwatch` if not running |
| `run_p14b2_chain.sh` | the chain: pre-flight (home lists + contamination check, 14a hashes), R1 render + select + YOLO dataset, R2 LP / FT training in parallel + photo test + temperature + pick, R3 replay per detector (bench in the background), R4 driving cells + determinism + switch-off + fairness checks, R5 tables, videos, hashes after, draft G14b2 |

Tools: Habitat side `mapmad/scripts/p14b2_trainset.py`, `p14b_replay.py --detector`, `p9_photo_test.py` /
`p9_score.py` (with `MAPMAD_PHOTO_TEST_DIR`, `MAPMAD_RECOG_MODELS`), `p14_videos.py render --arm`; `mapmad_sim/amark.py`;
recogniser side `mapmad_recog/finetune.py` (prep, train, models) and the fixed-vocabulary mode of `models.py`
(`weights`, `vocab`, `sha256` in a models file; `run_p13_pairs.sh` passes `RECOG_CONFIG`); NoMaD side
`vint_train.mapmad.closed_loop.p14b2_tables` (photo, r3, r4, extras, plain, off, gate), `p14b_bench --det`, `p14_checks det
--stage r4det`. Launch (after `nvidia-smi`): `bash train/scripts/mapmad_p14/launch_p14b2.sh`. DRY: see the header of
`run_p14b2_chain.sh`.

## Phase 14b-3 — retrain MapMaD v3 + final exam B

V1 DAgger round 2 (v2.1b drives the Phase 11 start set minus the 10 photo-test homes, maps shown then hidden, with the
fine-tuned YOLOE marks, W4 / A-mark off; Phase 11 labels + W1 recovery labels in stuck states), V2 one fine-tune ->
MapMaD v3 (round-2 data, reverse-capable recovery waypoints, past-move token + auxiliary head; `ema_9` fixed), G10
rows + recovery-label loss, V3 practice check (v3 vs the 14b-2 "both" cell), V4 exam B (second use, pipeline P* for
every arm, pre-set run order with the budget cut), V5 tables, videos, draft gate. Rules:
`/outputs/mapmad/p14b3_v3/p14b3_rules.md` (+ `.sha256`, copy in `mapmad/configs/p14b3_rules.md`); configs
`mapmad/configs/p14b3_v3.yaml`, `p14b3_rollout_{map,hidden}.yaml`, `train/config/mapmad_v3.yaml` (DRY: `p14b3_dry*.yaml`,
`mapmad_v3_dry.yaml`). Outputs: `/mnt/shared_disk/nazli/nomad_outputs/mapmad/p14b3_v3/` (`STATUS.md`,
`phase14b3_log.md`, `rollouts_r2/{set,map,hidden}`, `labels/`, `launch/`, `runs/v3_s1_*/`, `V3.json`, `offline/v3/`,
`runs/{practice,examb}`, `checks/`, `tables/`, `videos/`, `G14b3_table_draft.md`, `driver/`); round-2 drives in
`<mapmad_data>/habitat_mapmad_v3_dagger_r2{map,hidden}`.

| Script | What it does |
|---|---|
| `launch_p14b3.sh [dry]` | starts screens `p14b3_chain` + `p14b3_chain_watch` and `p14b3_gpuwatch` if not running |
| `run_p14b3_chain.sh` | the chain (header: pre-flight, V1 rollouts + labels + contamination, data index, restart + fine-tune + offline checks (+ the W1-turn fallback), determinism + Phase 13 / 14b-2 reproductions, V3 practice (+ the oscillation fallback), V4 exam B with the budget cut, fairness, rules sha256, videos, tables, draft gate) |
| `p14_lib.sh` | shared; `CL_UNIT_DIV=1` makes `closed_loop` count GPU-hours as wall time x GPUs (one unit) |
| `p14_plan.py` | picks `run_plan_p14b3.P14b3Plan` for `p14b3_*` configs (stages practice, det, off, examb) |
| `p14_status.py` | `CL_SEEDS` counts only runs of those seed indices (exam calls per seed share `runs/examb`) |

Tools: Habitat side `mapmad/scripts/make_p14b3_sets.py` (round-2 set, contamination), `p14b3_preflight.py`,
`mapmad_sim/recovery_labels.py` (W1), `mapmad_sim/run_plan_p14b3.py`, `arm_table.Arm.reverse`; NoMaD side
`vint_train/mapmad/past_moves.py` (B1), `v3_offline.py` (recovery-label loss, past-move error),
`closed_loop/p14b3_checks.py`, `closed_loop/p14b3_tables.py`, `closed_loop/deployment.ReverseSpec`.
Launch (after `nvidia-smi`): `bash train/scripts/mapmad_p14/launch_p14b3.sh`. DRY: see the header of `launch_p14b3.sh`.
