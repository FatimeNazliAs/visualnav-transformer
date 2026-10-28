# scripts/mapmad_p12 — Phase 12 run scripts (host side)

Phase 12 is the v2 search exam: dev decisions by pre-set rules → exam-set freeze → all arms → tables → videos, unattended,
in `screen`. Everything judged is fixed beforehand in `/outputs/mapmad/p12_search_exam/p12_rules.md` (+ `p12_rules.sha256`,
copy in `mapmad/configs/p12_rules.md`); the arm table, sets and run settings are `mapmad/configs/p12_exam.yaml`.
Outputs: `/mnt/shared_disk/nazli/nomad_outputs/mapmad/p12_search_exam/` (= `/outputs/mapmad/p12_search_exam` in the
containers): `STATUS.md`, `P12_DEV_CHOICES.json`, `dev/`, `exam_set/`, `unseeable_set/`, `ovon_set/`, `explore_set/`,
`qwen_set/`, `sets.json`, `runs/<set>/` (logs, summaries, scores, timing), `checks/`, `tables/`, `figures/`, `videos/`
(+ `small/`), `G12_table_draft.md`, `driver/`, `phase12_log.md`.

| Script | What it does |
|---|---|
| `launch_p12.sh [dry]` | starts screens `p12_chain` + `p12_chain_watch` (host watcher: STOP_reason.txt if the chain dies without its done file) and `p12_gpuwatch` (Phase 10's `gpu_watch.py`, `MAPMAD_OWNER_ENV=MAPMAD_P12_OWNER`, `RECENT_S=3600` = §8 rule 11) if not running |
| `run_p12_chain.sh` | the chain: pre-flight (committed code, rules sha256, checkpoints, dev set) + F check → dev runs (maps hidden v2 / v2.1, refactor check, headline, headline + YOLOE, determinism) → τ → exam sets (candidates on free GPUs, freeze) → 24-run timing, estimate, trims → main exam → OVON → exploration → unseeable → Qwen3-VL subset → scores, fairness, tables, figures → videos → draft G12 → `CHAIN DONE`; automatic stops → `STOP_reason.txt` |
| `p12_lib.sh` | shared bash: pinned + marked `docker exec`, GPU waiting (20-min rule, rule 11, live check), `closed_loop` (3 pairs per free GPU on one run queue, add a GPU when free, withdraw on another user's process, budget on GPU-running time, recogniser-restart stop, silent-step watchdog, no-progress stop) |
| `p12_plan.py` | command line over `mapmad_sim/run_plan.py` (the run plan: closed-loop calls per stage in block item 9 order, run counts, the headline pair, and the trims in block item 9's order when the 24-run estimate is over budget, `driver/trims.json`); the fairness check, tables and video picks read the same plan |
| `p12_status.py` | `STATUS.md` (step, runs per arm, GPUs, come back at, dev choices, headline interim numbers, STOPPED / CHAIN DONE); `count <subdir> <arms>` |

Closed-loop pairs: `mapmad/scripts/run_p12_pairs.sh` (one GPU, `PAIRS_PER_GPU` Habitat servers + YOLOE / Qwen3-VL sidecar
servers + `run_search` runners, shared `TOKEN`, `STOP_FILE` for withdrawal, recogniser restarts counted). Sets:
`mapmad/scripts/make_p12_sets.py`; F check: `mapmad/scripts/p12_f_check.py`; videos: `mapmad/scripts/p12_videos.py`.
Scoring / tables / figures (NoMaD side): `vint_train.mapmad.closed_loop.p12_score`, `p12_tables`, `p12_figures`.
Every GPU job is pinned and marked `MAPMAD_P12_OWNER=chain`. Launch (after `nvidia-smi`):

    bash train/scripts/mapmad_p12/launch_p12.sh

Watch: `cat /mnt/shared_disk/nazli/nomad_outputs/mapmad/p12_search_exam/STATUS.md`, `screen -r p12_chain`.
DRY run (never results; `p12_dry/`, config `mapmad/configs/p12_dry.yaml`): `P12_DIR=<host p12_dry> P12_CFG=<container
p12_dry.yaml> P12_ALLOW_DIRTY=1 P12_GPU_FREE=<real gpu_free.json> CL_CHECK_S=30 SILENT_S=600 bash launch_p12.sh dry`.
