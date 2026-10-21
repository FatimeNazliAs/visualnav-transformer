# scripts/mapmad_p10 — Phase 10 run scripts (host side)

Phase 10 trains MapMaD v2 and checks it offline, unattended, in `screen`. Everything judged is fixed beforehand in
`/outputs/mapmad/p10_v2_train/p10_rules.md` (+ `p10_rules.sha256`, copy in `mapmad/configs/p10_rules.md`). Outputs:
`/mnt/shared_disk/nazli/nomad_outputs/mapmad/p10_v2_train/` (= `/outputs/mapmad/p10_v2_train` in the containers):
`STATUS.md`, `driver/` (logs, state, GPU watcher), `launch/` (overlay configs actually used), `runs/`, `offline/`,
`checks/`, `curves/`, `smoke_DRY/`, `timing_DRY/`.

| Script | What it does |
|---|---|
| `launch_p10.sh [all\|chain\|twin]` | starts screens `p10_chain` + `p10_chain_watch`, `p10_twin` + `p10_twin_watch`, and `p10_gpuwatch` if not running |
| `run_p10_chain.sh` | the chain: frontier cache + parity → smoke → timing (+ fallbacks) → main run seed 1 → offline checks, GoStanford, pre-training report, draft G10 table, curves → `CHAIN DONE`; automatic stops → `STOP_reason.txt` |
| `run_p10_twin.sh` | the twin (seed 2, report only): waits for the main run, Phase 9 done with GPUs, and a second free GPU; then trains and runs the same offline checks |
| `p10_lib.sh` | shared bash functions (pinned `docker exec`, GPU wait, supervised training with one resume, offline checks) |
| `gpu_watch.py <dir>` | GPU free check every 5 min (free = 4 consecutive checks without anyone else's process; Phase 9 counts as busy; our processes carry `MAPMAD_P10_OWNER`); `--check <gpu> <owner>` = instant check |
| `p10_timing.py` | the timing step's decision: memory fallback (> 23.5 GiB or OOM), more workers (loader > 15%), budget stop (> 10.5 h) |
| `p10_status.py` | writes `STATUS.md` from the chain/twin state and the main run's `metrics.jsonl` |
| `g10_table.py <p10> main\|twin` | draft G10 table from the rows 1-2 check, the offline comparison and the GoStanford harness |

Every GPU job is pinned with `docker exec -e CUDA_VISIBLE_DEVICES=<g>` (configs keep `gpu_ids: [0]`: inside, the one
visible GPU is index 0). The drivers refuse uncommitted code under `train/` or `mapmad/`. Launch (after `nvidia-smi`):

    bash train/scripts/mapmad_p10/launch_p10.sh

Watch: `cat /mnt/shared_disk/nazli/nomad_outputs/mapmad/p10_v2_train/STATUS.md`, `screen -r p10_chain`.
