# p10_rules.md — Phase 10 · MapMaD v2 model + training (rules frozen before the main run)

Version 3, 2026-10-21 19:0x (adds §9, the target-layer decision rule; Naz's instruction after the gate). Version 2
(sha256 `ee070c1a…8e69`) is kept as `p10_rules_v2.md`, version 1 (`3f3398b8…32df`) as `p10_rules_v1.md`. Sources: v2 plan v2.2 §8.5 / §5 G10 / §8 / §9; Phase 10
confirmation block (Naz, 2026-10-21, items 1-11, no changes). sha256 of this file in `p10_rules.sha256`; a copy in
`mapmad/configs/p10_rules.md`. A rule changed after launch gets a new version with its own sha256 BEFORE the result
it judges (older versions archived as `p10_rules_v<N>.md`).

**Amendment v2 (pre-registered, 2026-10-21 11:50):** made during the main run's epoch 0, before any judged result
exists (no ema_19, no offline score of any trained checkpoint). G10 row 3 becomes maps − zeroed (§4, §5); the v1
row 3 (maps − hidden) becomes report only, printed next to its step-0 baseline. Why: maps − hidden is already
negative for the untrained v2 (two extra tokens + mean pooling, no map content), so it cannot show map use; maps −
zeroed keeps the token count and pooling the same and changes only the map content. The training run, its recipe
and every other rule are unchanged; the chain is not restarted (the zeroed arm is added by `offline_eval score`
whenever `maps` is scored, so it is computed on ema_19 and on the step-0 model).

## 1. Recipe (block item 1) — `train/config/mapmad_v2.yaml` (layered on `mapmad_p4_C.yaml`)
- Start: official `nomad.pth` (sha256 `70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333`), strict load;
  only `vision_encoder.map_encoder.{near,far}.*` (36 tensors, 1,532,736 weights) are new.
- 50/50 Habitat (`habitat_mapmad_v2`, 9,257 train / 551 test drives after `exclude_drives`) / GoStanford by sampling
  weight; `waypoint_spacing` 5 (0.25 m between waypoints); 20 epochs x 200,000 samples (782 optimizer steps of 256).
- Two AdamW groups: pretrained weights 1e-5, both map encoders 1e-4; warm-up per optimizer step in epoch 1
  (x 0.01 -> x 1), then cosine per epoch (Phase 4 `lr_schedule.py`); EMA as Phase 4; `ema_e.pth` per epoch + one
  resumable `latest.pth`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Main run: **seed 1**. **Checkpoint `ema_19.pth` of the main run is fixed in advance** (no selection of any kind).
- Per-epoch test loss on a fixed 1,000-sample subset per test set (`eval_samples: 1000`): curves only, never used to
  choose anything. In-training checks every 2 epochs (step-0 sanity, map effect on the 1,007-sample subset,
  GoStanford 1,000-item subset, attention share per map token, forward/turn split, per-category loss): report only,
  EXCEPT the step-0 sanity (maps hidden must equal the official model; else the run stops = a bug).
- Supervisor: one automatic resume from `latest.pth` after a crash; a second crash stops.

## 2. Map input (block items 2-4)
- Near 64 x 64 @ 10 cm + far 64 x 64 @ 40 cm (far = 4 x 4 summary: obstacle any, explored mean, frontier any,
  target max), robot-centred, forward = up, Phase 8's `map_v2.near_window` / `far_window` at frame t.
- Target layer: rebuilt at frame t from the sparse **noisy** updates (training input); clean only in a check arm.
- Frontier: the teacher's plain frontier at frame t, ONE shared function `map_v2.teacher_frontier_at` =
  frontier_clusters(obstacle_t | cells bumped before t, explored_t), clusters >= 4. Bumped cells (collisions and
  safety steps, `mapmad_frames.npz` `bump_events`, active from the frame after they were logged) are what the live
  teacher used; they never enter the obstacle layer. Read through the per-drive cache (`frontier_cache.py`).
  Part 1 result: 9,808 drives, 5,298,431 checked frames, 0 differ from the teacher's logged cluster count
  (without the bump cells 103 of 4,075 sampled frames differed).
- Modes (seeded per sample): search_map 45% (photo hidden, both maps, noisy target; search drives) · explore_map 20%
  (photo hidden, both maps, target empty; exploration drives) · explore 20% (photo + maps hidden; all drives) ·
  photo 15% (photo 0-20 waypoints ahead, maps hidden; all drives). GoStanford: maps hidden, photo 50/50. Distance loss
  only with the photo shown. No near-/far-only training mode.
- Model: two v1-design encoders (separate weights, 4 -> 32 -> 64 -> 128 -> 128, GroupNorm, flatten -> 256, last
  Linear zero-init); tokens [4 frames, goal, near, far] = 7; sine positional table 5 -> 7 rows (rows 0-4 = the
  checkpoint's). Both maps hidden -> the first 5 tokens only, NoMaD's own pooling (= NoMaD's computation, exactly).
  `map_input: false` -> byte-identical NoMaD; v1 (one token) unchanged.

## 3. Frozen sample lists (block item 5) — `/outputs/mapmad/p10_v2_train/offline/lists/`
| File | Content | sha256 (of the samples) |
|---|---|---|
| `search_list.json` | 5,000 samples, 272 held-out search drives (18-19 each), t-15 >= 0, t+40 <= last | `aa9ec40678d0748afbd2174e540847e730e0a515c2d104938c16e90c689d88bb` |
| `explore_list.json` | 2,000 samples, 249 held-out exploration drives (30 too short dropped) | `da883ed257e102ddc4a4141511254023d75b04b23f056710b03d3ccfd952ecf6` |
| `offline_subset.json` | in-training map-effect subset: 1,007 samples / 55 whole drives of search_list | `25c64fb9669e145b4d93e69910abf62c6fd8e9ac871868a587915d8dd3a9ca3c` (subset body sha256, checked at load) |
| Phase 4 `gostanford_subset.json` | in-training GoStanford subset (reused; harness items checked by sha256) | as frozen in Phase 4 |
The test split (`test/traj_names.txt` sha256 `b32d66e6…3c`) and `collision_drives.txt` (`40442cf5…af17`) are Phase 8's.

## 4. Offline checks (block items 5-6) — on `ema_19` of the main run (and of the twin, report only)
- Scoring: `offline_eval score` (full 10-step diffusion, every sample's noise from its own id, deterministic
  kernels); loss = `gc_action_loss` (MSE of the 8 waypoints, normalised units).
- **Primary (G10 row 3, v2):** search list (the same 5,000 samples), arm `maps` (noisy layer, both windows) minus arm
  `zeroed` (both map tokens shown, all four map layers zeroed, same mode and noise), drive-level paired bootstrap,
  10,000 resamples (seed 0), 95% percentile CI. Scored on ema_19 and on the step-0 model.
- Report only: maps − hidden (photo + maps hidden = NoMaD explore; the v1 row 3) next to its step-0 baseline; clean layer; near only; far only; target zeroed; frontier zeroed; target shifted 1 m (direction
  seeded per sample id); exploration list explore_maps vs hidden; per category; forward/turn split (turn =
  |teacher heading change over the 8 waypoints| > 30 deg) and the forward / sideways parts of the loss; attention
  share per map token (near, far), with the untrained model's (step-0) shares as the baseline.
- Pre-training report (report only, no reweighting): the untrained v2 (official weights + fresh encoders; its maps-
  hidden arm = the official NoMaD) on both lists: forward/turn split and per-category loss; v2 after training the same.
- GoStanford no-harm (G10 row 4): Phase 4 harness, full GoStanford test set, `ema_19` (maps hidden, photo shown) vs
  the official `nomad.pth`, same items and noise (paired).

## 5. Gate G10
| # | Check | Pass |
|---|---|---|
| 1 | Switch off | NoMaD forward pass byte-identical to the reference saved before any Phase 10 change (checked on the step-0 model; done in Part 1: pass on GPU1 and CPU) |
| 2 | Maps hidden | step-0 v2 output = official NoMaD, allclose atol 1e-6 rtol 0; token count 7; all nomad.pth keys load (done in Part 1: max diff 0.0) |
| 3 | Offline map effect | drive-level paired 95% CI of (maps − zeroed) < 0 (upper end < 0) on the search list (5,000 samples), noisy layer; zeroed = both map tokens shown, all map layers zeroed (v2) |
| 4 | GoStanford no-harm | mean(ema_19) ≤ mean(official) + 0.10 (paired, full test set) |
| 5 | Runs | `ema_19` of the main run (seed 1) fixed in advance; twin (seed 2) report only |
Report next to row 3 (no pass/fail): maps − zeroed for the untrained v2 (step 0; the zero-init encoders make it
exactly 0), and maps − hidden for ema_19 next to its step-0 value. v1 history: Part 1 found maps − hidden already
negative at step 0 (mini test, 40 samples: −1.03) because showing the maps adds two tokens and switches to plain
mean pooling; hence amendment v2.
Row 3 or 4 fails -> recorded as failed, stop for a decision with Naz (map alignment first: rotation sign, frame index,
far summary, target rebuild; then one change at a time). The twin never holds the gate (pending -> addendum).

## 6. Memory / speed fallbacks (block item 7) — chain step 3 (timing, 200 optimizer steps, DRY folder)
1. Peak GPU memory > 23.5 GiB (24,064 MiB, nvidia-smi, sampled every 5 s) or out-of-memory -> `batch_size 128 +
   grad_accum_steps 2` (effective 256; logged deviation; the model has GroupNorm only, so no batch statistics change),
   timing again; no stop. Out-of-memory after that -> automatic stop.
2. Loader wait > 15% of the step time -> `num_workers` 12 -> 20, timing again once (logged).
3. Estimate = 20 x (optimizer steps per epoch x seconds per step + 0.5 min eval) + 11 checks x 1 min; > 10.5 h ->
   automatic budget stop. (Budget 7 h per run; stop > 1.5x.)
The main run uses the setting the timing step ended with (launch overlay `launch/main.yaml`, saved with the run).

## 7. Automatic stops (block item 8) -> `STOP_reason.txt`, `STATUS.md` -> STOPPED, fix nothing
Smoke fails (training error, no FINISHED TRAINING, or a smoke_v2 check) · NaN loss · a second crash · out-of-memory
after the fallback · checkpoint sha256 not written · estimate > 10.5 h · frontier-cache mismatch (build: any checked
frame differs from the teacher's count; check: cache != shared function; Habitat parity: any difference) · cache
> 4.5 h or > 90 GB · any unexpected exit of the driver (EXIT trap, exit code + last log line).
Hangs (added 2026-10-21 11:3x, before any judged run, after the first smoke hung on a DataLoader worker that died while
importing torch): every DataLoader has a 900 s batch timeout (a dead worker -> an error); smoke > 30 min and a timing
run > 60 min -> killed = smoke fails / timing failed (stop); a training run whose log is silent > 60 min -> killed =
a crash (one resume, a second -> stop); offline checks > 5 h, pre-training report > 3 h -> killed, stop.

## 8. GPU rule (v2 plan §8 + block item 9)
- Free = no process of anyone else for 4 consecutive 5-min checks (`gpu_watch.py`; Phase 9's processes count as busy;
  our own processes are marked MAPMAD_P10_OWNER=chain|twin). Never start on or share a GPU that holds another process.
- Priority: Phase 10 main > Phase 9 > Phase 10 twin. Main run on the first GPU that is free for the chain. Twin only
  when a second GPU is free for it AND Phase 9's STATUS.md shows photo test done / AT GATE / STOPPED; never started by
  the chain (own launcher, screen p10_twin).
- A started run is never killed when another process appears; the event is logged. Every GPU event is logged.
- Every GPU job is pinned with `docker exec -e CUDA_VISIBLE_DEVICES=<g>` (configs keep gpu_ids [0]).

## 9. Target-layer decision (v3, pre-set by Naz after G10 passed; written before any score of this list)
- List (frozen before scoring): `offline/lists/seen_list.json`, sha256
  `bd44465c51a8842370e9beff4ca8e6dac01b6152983649ca7df6d9b2eb598c93`: EVERY frame t of the 272 held-out search drives
  (551 held-out drives = test split minus collision drives) with t − 15 ≥ 0, t + 40 ≤ last frame, a simulator-truth
  target detection (≥ 200 px at ≤ 4 m) at some frame < t and none at t ("seen before, now out of view"):
  2,151 frames from 107 drives.
- Scoring: `offline_eval score` on main `ema_19` (sha256 `6d2492f1…`), arms maps, target_zeroed, target_shifted (1 m,
  direction seeded per sample id) (the zeroed arm is added automatically, report only); per-sample noise from the id;
  CPU (GPU1 belongs to Phase 9 now; arms are paired within one run).
- Statistic: target_zeroed − maps and target_shifted − maps, drive-level paired bootstrap, 10,000 resamples (seed 0),
  95% percentile CI.
- **Rule:** 95% CI lower bound of (target_zeroed − maps) > 0 → **A** (go to Phase 11 with this ema_19); else → **C**:
  stop for Naz with a one-paragraph proposal of ONE training change (oversample seen-before-out-of-view frames, or
  reduce the false-mark rate); no new run without Naz's OK. target_shifted is report only.
- Report also: the share of training search samples (train-split search-drive index entries, collision drives
  excluded) that are seen-before-out-of-view.

