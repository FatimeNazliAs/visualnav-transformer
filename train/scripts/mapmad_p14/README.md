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
