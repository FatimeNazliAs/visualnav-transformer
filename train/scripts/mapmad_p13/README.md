# scripts/mapmad_p13 — Phase 13 run scripts (host side)

Phase 13 = Goal 1, "make the map claim airtight": a maps-hidden DAgger round and v2.1b (DAgger control), a fresh
exam on MP3D (exam B), the VLFM-style arm VF + a port check of the original VLFM, and checkable videos — unattended,
in `screen`. Everything judged is fixed beforehand in `/outputs/mapmad/p13_claim/p13_rules.md` (+ `p13_rules.sha256`,
copy in `mapmad/configs/p13_rules.md`); configs `mapmad/configs/p13_claim.yaml`, `p13_hidden_dagger.yaml`,
`train/config/mapmad_v21b.yaml`. Outputs: `/mnt/shared_disk/nazli/nomad_outputs/mapmad/p13_claim/` (`STATUS.md`,
`phase13_log.md`, `mp3d_homes.json`, `examb_set/`, `dev_set/`, `sets_mp3d.json`, `videos_rerun/`, `videos/`,
`hidden_dagger/`, `runs/<v21b run>`, `offline/`, `runs/{dev,hm3d,examb}/`, `vlfm_port/`, `checks/`, `tables/`,
`G13_table_draft.md`, `driver/`).

| Script | What it does |
|---|---|
| `launch_p13.sh [dry]` | starts screens `p13_chain` + `p13_chain_watch` (STOP_reason.txt if the chain dies without its done file) and `p13_gpuwatch` (Phase 10's `gpu_watch.py`, `MAPMAD_OWNER_ENV=MAPMAD_P13_OWNER`, `RECENT_S=3600`) |
| `run_p13_chain.sh` | the chain (header: steps 0-8 = videos, hidden-DAgger rollouts + labels + index, restart, v2.1b fine-tune + G10 rows, determinism, dev sanity, timing, exams, port check, tables) |
| `p13_lib.sh` | Phase 12's `p12_lib.sh` with Phase 13 names (+ `CL_CFG` / `CL_PREFIX` for the hidden-DAgger config) and Phase 11's supervised training helpers |
| `p13_plan.py` | command line over `mapmad_sim/run_plan_p13.py` (calls per stage, run counts, timing call, the 13.1 re-runs grouped by their Phase 12 GPU) |
| `p13_estimate.py` | exam estimate from the 24 timing runs (HM3D) and the dev runs (MP3D, F, VF) |
| `p13_status.py` | `STATUS.md` (first line = what to paste on return; step, done/total, GPUs, mode, fallbacks, last error, STOP question); `count <subdir> <arms>` |
| `render_videos.sh` | Part 3: renders the 13.1 videos the chain's render lost (two shards shared one YOLOE server; now one server per shard, in the chain too) + the M21b / M21bh rule-pick videos (`p13_videos.py pick-rules`, `videos_rule/<set>/`); then `check --config` (every planned video must exist) |
| `vlfm_port.sh` | the original VLFM on 100 HM3D v2 val episodes in its own container (`mapmad/docker/Dockerfile.vlfm`), if it could be set up |

Closed-loop pairs: `mapmad/scripts/run_p13_pairs.sh` (= run_p12_pairs.sh, marks `MAPMAD_P13_OWNER`, `RECOG=yoloe+itm`
also starts one BLIP-2 ITM server per GPU for arm VF). Sets: `mapmad/scripts/make_p13_sets.py`; MP3D checks:
`p13_mp3d_check.py`, `p13_overlap_check.py`; videos: `p13_videos.py`. Tables: `vint_train.mapmad.closed_loop.p13_tables`.
Launch (after `nvidia-smi`): `bash train/scripts/mapmad_p13/launch_p13.sh`. Watch: `cat .../p13_claim/STATUS.md`,
`screen -r p13_chain`. DRY: `P13_DIR=<host p13_dry/chain> P13_CFG=<container DRY config> P13_HIDDEN_CFG=... P13_V21B_BASE=...
P13_ALLOW_DIRTY=1 P13_GPUS="0 1" CL_CHECK_S=30 SILENT_S=600 P13_TRAIN_EXTRA="epoch_samples: 512|epochs: 10" bash launch_p13.sh dry`.
