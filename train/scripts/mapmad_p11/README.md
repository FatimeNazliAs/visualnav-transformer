# scripts/mapmad_p11 — Phase 11 run scripts (host side)

Phase 11 picks the final v2 in closed loop (step 0), runs one DAgger round and fine-tunes v2.1, unattended, in
`screen`. Everything judged is fixed beforehand in `/outputs/mapmad/p11_dagger/p11_rules.md` (+ `p11_rules.sha256`,
copy in `mapmad/configs/p11_rules.md`). Outputs: `/mnt/shared_disk/nazli/nomad_outputs/mapmad/p11_dagger/`
(= `/outputs/mapmad/p11_dagger` in the containers): `STATUS.md`, `dev_set/`, `step0/`, `V2_FINAL.{json,txt}`,
`rollouts/` (set + run logs), `labels/`, `sheets/`, `runs/`, `offline/`, `dev_v21/`, `twin/`, `checks/`, `driver/`,
`launch/`, `G11_table_draft.md`, `phase11_log.md`; DAgger drives in `/mnt/shared_disk/nazli/mapmad_data/habitat_mapmad_v2_dagger/`
(splits `…/splits/habitat_mapmad_v2_dagger/`, frontier cache `…/p11_frontier_cache/`, index `…/p11_index/`).

| Script | What it does |
|---|---|
| `launch_p11.sh [all\|chain\|twin]` | starts screens `p11_chain` + `p11_chain_watch`, `p11_twin` + `p11_twin_watch`, and `p11_gpuwatch` (Phase 10's `gpu_watch.py` with `MAPMAD_OWNER_ENV=MAPMAD_P11_OWNER`) if not running |
| `run_p11_chain.sh` | the chain: pre-flight + label check → step-0 dev check (+ determinism episode) → v2 pick → rollouts + labels → data index (splits, frontier cache, sample index + LMDB, union shares, sheet) → `naz_mapmad` restart → fine-tune (ema_9) → offline checks → dev check v2.1 → draft G11 → `CHAIN DONE`; automatic stops → `STOP_reason.txt` |
| `run_p11_twin.sh` | the twin (seed 2, the winner's Phase 10 recipe, report only): starts after the chain's fine-tune holds its GPU, only on a GPU the chain does not hold; then G10 rows 3-4 + seen list |
| `run_p11_rechecks.sh <rollout ids>` | phase-end §0.8 re-checks on one GPU (`GPU=0`): G10 rows 1-2, re-score `ema_9`, determinism episode, 2 rollouts regenerated + relabelled; only into `checks/recheck` (config `mapmad/configs/p11_recheck.yaml`) and `<mapmad_data>/p11_recheck_dagger`; verdicts by `mapmad/scripts/p11_recheck_compare.py` -> `checks/recheck/rechecks.json` |
| `p11_lib.sh` | shared bash: pinned `docker exec`, GPU waiting, `closed_loop` (3 pairs per free GPU on one run queue, add a GPU when free, withdraw on another user's process, budget on GPU-running time, no-progress stop), supervised training with one resume + hang watchdog, offline checks |
| `p11_status.py` | `STATUS.md` from the chain/twin state, run counts, epochs; `count <subdir> <arms>` |
| `p11_preflight.py` | `check`: checkpoint sha256s + set fingerprints against the rules file's `json preflight` block; `totals`: run counts per step |
| `g11_table.py <p11>` | draft G11 table from the step outputs |

Closed-loop pairs: `mapmad/scripts/run_p11_pairs.sh` (one GPU, `PAIRS_PER_GPU` simulator servers + `run_search` runners,
shared `TOKEN`, `STOP_FILE` for withdrawal). Every GPU job is pinned and marked `MAPMAD_P11_OWNER=chain|twin`. The
chain refuses uncommitted code under `train/` or `mapmad/` and a rules file whose sha256 differs. Launch (after
`nvidia-smi`):

    bash train/scripts/mapmad_p11/launch_p11.sh

Watch: `cat /mnt/shared_disk/nazli/nomad_outputs/mapmad/p11_dagger/STATUS.md`, `screen -r p11_chain`.
DRY run (never results; tiny sets in `p11_dagger/DRY`): see the header of `run_p11_chain.sh` (`P11_DIR`, `P11_CONFIG=p11_dryrun.yaml`,
`P11_TRAIN_GPUS` (trainings only on these GPUs; the twin starts after the dev check v2.1), `P11_ALLOW_DIRTY`, `P11_V21_BASE`, `P11_TRAIN_EXTRA`, `P11_GPU_FREE`, `CL_CHECK_S`).
