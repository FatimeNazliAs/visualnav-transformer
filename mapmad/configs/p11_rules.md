# Phase 11 rules — dev check (pick v2) + DAgger round → v2.1

Written and hashed before step 0 runs (`p11_rules.sha256`, copy in `mapmad/configs/p11_rules.md`). A later version
(`p11_rules_v2.md`, …) is allowed only before the result it judges. Source: Phase 11 confirmation block (app chat, final),
kickoff `.p11_kickoff.md`, v2 plan v2.4 §8.6 / §5 G11 / §8 / §9. Config: `mapmad/configs/p11_dagger.yaml`.

## 1. Frozen inputs (checked by the chain before step 0; a mismatch = automatic stop)

- **Dev set** (also Phase 12's dev set): `p11_dagger/dev_set/episodes.json`, **120 episodes**, fingerprint
  `c7e64fcedfd981864af8b84874e87a407f545f67eb9e9503ec9bfce7a4e067af` (file sha256 `2aa998cf…0efd`). Plan 8.7 builder on the
  10 Phase 1 dev homes (00746, 00440, 00272, 00330, 00241, 00732, 00081, 00534, 00397, 00401), ObjectNav v2 TRAIN episodes,
  300 examined per home (seed 11101): start snaps ≤ 0.2 m on the real floor; ≥ 1 usable view point of an instance of the
  category on the start storey reachable; no instance of the category visible (any pixel) in 8 headings at pin025; 3–15 m
  walking distance to the nearest usable view point; Phase 8 start filter (the category is seeable: some usable view point
  + heading shows ≥ 200 px at ≤ 4 m). Draw (seed 11201): ≤ 15 per home, aim 120, 3 distance bins (3–6 / 6–10 / 10–15 m,
  40 each), neighbour-bin fill. Ids `p11d-<home>-<category>-<ObjectNav id>`; duplicates refused. Target = **any** instance.
- **Rollout set**: `p11_dagger/rollouts/set/episodes.json`, **1,448 episodes** in the 145 labelled train homes (10 draws per
  home, seed 1101; 2 draws found no start), fingerprint `8e3428c74454f7a6f0a6021694c0e0eb8565c3c646d3ec5780b83930e187aa40`
  (file sha256 `990fdb01…45ec`). Phase 8 search-start rules (datagen_v2 `sample_search`, option A config): category uniform
  among the home's seeable categories, start on the real floor (wall starts as Phase 8's variations), 3–15 m, no
  instance visible in 8 headings. Ids `p11r-<home5>-<category>-<n:03d>`. The 100 episodes in the 10 dev homes form the
  **DAgger test split** (held out of training, like Phase 8's val-drive drives); the other 1,348 are trained on.
- **Checkpoints** (sha256 checked when every network is built and before the chain starts):
  run 1 `p10_v2_train/runs/v2_s1_2026_10_21_08_42_50/ema_19.pth` `6d2492f19f7f1a5ccadea3f120b66a78b04bb5e88d10ca701fc523e0f9aee984`
  (`mapmad_v2.yaml`); run C `p10_v2_train/runs/v2c_s1_2026_10_22_04_44_29/ema_19.pth`
  `aa30d133606094506e57ec10695e5d12ffe0c5443fe8bf7726127057cc502a5c` (`mapmad_v2_seen.yaml`).

```json preflight
{"checkpoints": {"p10_v2_train/runs/v2_s1_2026_10_21_08_42_50/ema_19.pth": "6d2492f19f7f1a5ccadea3f120b66a78b04bb5e88d10ca701fc523e0f9aee984",
                 "p10_v2_train/runs/v2c_s1_2026_10_22_04_44_29/ema_19.pth": "aa30d133606094506e57ec10695e5d12ffe0c5443fe8bf7726127057cc502a5c"},
 "sets": {"p11_dagger/dev_set/episodes.json": "c7e64fcedfd981864af8b84874e87a407f545f67eb9e9503ec9bfce7a4e067af",
          "p11_dagger/rollouts/set/episodes.json": "8e3428c74454f7a6f0a6021694c0e0eb8565c3c646d3ec5780b83930e187aa40"}}
```

## 2. Closed-loop harness and policy settings (every closed-loop run of this phase)

- Online map v2 in the simulator server (`mapmad_sim.search_v2.SearchMap`): Phase 8 builder (obstacles 0.05–0.55 m above the
  real floor, max 4 m, invalid = unknown, first-seen); near 64×64 @ 10 cm + far 64×64 @ 40 cm (4×4 summary any/mean/any/max);
  frontier = the teacher's plain frontier including the cells bumped before t (a forward move stopped short marks the cell
  0.3 m ahead); target layer = simulator truth with the Phase 8 noise (U(0.4, 1.0) per instance per frame, miss 0.2, false
  mark 0.05 as a 3×3 blob 0.3–0.7 on a visible obstacle cell), fused over views (equal-weight mean, no detection = 0).
  Verified before this file: the server's windows = the windows rebuilt from a Phase 8 generator drive at every frame of 2
  dev drives (574 + 1,097 frames, pose replay; `checks/map_parity.json`).
- Camera pin025 (`camera_fair.yaml`); LIMO floor map. Policy = `MapMaDV2Inference` through `MapMaDV2Policy`: context every
  5th step (t−15, −10, −5, t), missing past frames repeat the first, photo hidden, 8 samples / sample 0, **waypoint index
  1**, 0.25 m per normalised step, NoMaD's pd_controller, 0 ≤ v ≤ 0.2 m/s, |w| ≤ 0.4 rad/s, 4 Hz, no reversing. "Maps
  hidden" = both windows masked = the first 5 tokens only = NoMaD's computation.
- **Seeds:** noise seed = sha256(`mapmad-p11/<episode id>/<seed index>`) (first 8 hex): the same diffusion noise AND the
  same target-layer noise stream for every arm. 1,000 steps. **Oracle stop:** success = first entry into the success zone
  (walking distance ≤ 0.2 m to the nearest usable ObjectNav view point of ANY instance of the category).
- Reported per run (never decide): SPL (to that view point set), steps to the first truth sighting (search steps; ≥ 200 px
  at ≤ 4 m, any instance of the category), approach steps (after it), collisions, stuck streaks (≥ 100 consecutive
  collision steps).
- Shared run queue (claims), 3 Habitat + policy pairs per free GPU; results do not depend on the number of pairs.
  Infrastructure errors are retried twice with the same seed, then recorded as a crash; any other exception stops.

## 3. Step 0 — v2 pick (G11 row 0)

- Arms: **run1** (run 1 ema_19, maps on), **runC** (run C ema_19, maps on): 120 dev episodes × seed indices 0, 1, 2.
  Report only: **run1_hidden** (run 1, maps hidden), seed index 0.
- Per-episode success = mean over its 3 seeds. **Δ = mean over the dev episodes of (run C − run 1).**
- **Rule: run C becomes v2 only if Δ ≥ +0.03; otherwise run 1 (a tie → run 1).** (Computed in floating point: Δ ≥ 0.03 − 1e−12.)
  No CI. An episode with a crash record in either judged arm is dropped from both (and counted).
- Output: `step0/decision.{json,md}`, `V2_FINAL.json` / `V2_FINAL.txt` (path, sha256, train config, Δ, the rule).
- Determinism episode: run 1 on the first dev episode, seed index 0, twice on the same GPU (two fresh launches) → the step
  logs must be byte-identical (the logs hold no timing fields).

## 4. Rollouts and labels (G11 row 1)

- The winner ("v2") drives itself (pure student, no teacher mixing) on the 1,448 rollout episodes, seed index 0, 1,000
  steps or the first entry into the success zone. The server records each run in the Phase 8 drive format
  (`<mapmad_data>/habitat_mapmad_v2_dagger/<episode id>/`: frames, `traj_data.pkl`, map first-seen arrays, clean + noisy
  target-layer sparse updates, detections, bump events, `phase` = search until a usable instance is first seen, then
  approach; `teacher_mode` = search on every frame, `frontier_clusters` = the live count).
- **Label** at every visited frame t (`mapmad_sim.dagger_labels`): `teacher_step` on the robot's own map at t (obstacle /
  explored first seen ≤ t, bumped cells before t), its pose and the truth "seen" set (≥ 200 px, ≤ 4 m, sticky); the teacher
  state carried from frame to frame (the committed frontier stays committed until reached / gone, then re-chosen as
  usual); **the blacklist counter is frozen** (`blacklist_window_steps` → never: no frontier is dropped because the student
  ignored it; a reached goal is still dropped as in Phase 8). Option A teacher numbers (`p8_datagen_v3a.yaml`).
- **Label = kinematic look-ahead** on the frozen own map: the Phase 8 motion model (pure pursuit, 0.2 m/s, 0.4 rad/s, 4 Hz;
  a new choice > 30° off first turns on the spot; arrived: turn to ±15° of the instance and stand) for 40 control steps →
  the poses after every 5th step → 8 waypoints in the robot frame at t (/ 0.25 m = the network's normalised format).
  **Look-ahead variant (Naz, 2026-10-22, answer B): "re-ask"** — at every look-ahead step the teacher is asked again
  (`teacher_step`) from the simulated pose on the same frozen own map, its state carried, and the motion model follows
  that step's path for one step (as the Phase 8 drive loop re-asks the teacher every step).
- **No label:** teacher-fail states (no frontier, no path, blacklist exhausted) and states whose move is inside a collision
  streak ≥ 20 steps. **Order (Naz, 2026-10-22, answer 2a): collision first** — a state inside a collision streak is
  skipped as collision (excluded like Phase 8's collision drives); the **40% teacher-fail stop counts the teacher-fail
  states outside collision streaks**. Reported: both orders' shares (teacher-fail first / collision first) and the
  overlap (teacher-fail states inside collision streaks), overall and per category.
- **Label check (before the rollouts):** 50 Phase 8 TEST-split search drives (after `exclude_drives`; seed 11301); every frame
  with 40 future frames where the teacher does not fail: look-ahead label (the drive's own teacher spec, blacklist not
  frozen) vs the real future waypoints at t + 5, …, t + 40; error per state = mean of the 8 waypoint distances (m).
  **Pass: median over all states ≤ 0.25 m AND the median error of waypoint 1 (t + 5) ≤ 0.10 m** (Naz, 2026-10-22), else
  automatic stop (before the rollouts; step 0 and the pick still run). Reported: p95, per-waypoint medians + p95, per
  phase, and the share of states where the relabelled goal equals the logged one.
- Collision rule: DAgger states inside a collision streak ≥ 20 steps are never samples.
- 10-state sheet (seed 11401): storey map around the robot, the student's real next 40 steps, the teacher's label and path,
  the camera frame (`sheets/label_sheet.png`).

## 5. Union, sampling, fine-tune → v2.1 (G11 row 2 input)

- DAgger samples (labelled frames with t ≥ 15, train split) join the **search-with-map pool only**: a search_map draw is
  uniform over (Phase 8 search samples + DAgger samples); modes stay 45/20/20/15; GoStanford 50/50. The winner's
  `seen_oversample` (run 1: 0; run C: 0.25) applies over the union (seen pool = both sources' seen-before-out-of-view
  frames). The resulting DAgger share is reported (`labels/union.json`).
- Fine-tune from the winner's ema_19 (weights; the EMA starts from them), **10 epochs × 200,000 samples**, seed 1, batch 256,
  the same two lr groups (pretrained 1e-5, map encoders 1e-4), warm-up over epoch 1, cosine over epochs 2–10; config
  `mapmad_v21.yaml` (run 1) / `mapmad_v21_seen.yaml` (run C). In-training checks off (their step-0 sanity compares with
  the official weights). **Checkpoint `ema_9` fixed in advance = v2.1.** Supervisor with one auto-resume; DataLoader
  timeouts; a training log silent for 60 min = a hang (killed, counts as a crash). `naz_mapmad` is restarted right before
  (after checking that it runs nothing).

## 6. Offline checks on ema_9 (G11 row 2) and dev check v2.1 (G11 row 3)

- G10 rows 1–4 as in Phase 10 rules v2: row 1 switch off byte-identical; row 2 maps hidden = official (allclose 1e-6, 7
  tokens, all keys); row 3 drive-level CI of (maps − zeroed maps) < 0 on the held-out search list (noisy layer); row 4
  GoStanford ≤ official + 0.10 (paired). Report only: seen-list rule (target zeroed − maps, CI lower > 0), learned map
  effect (maps − hidden), v2.1 − v2 on the frozen search list, DAgger-state loss (2,000 held-out DAgger test states,
  seed 11601).
- Dev check v2.1 (report only): arm **v21** on the same dev set, seeds and settings as step 0; v2 = its step-0 runs.
  Oracle-stop success, SPL, stuck streaks, search / approach steps; difference with an episode bootstrap CI (seed 11501).

## 7. Twin (report-only addendum)

Seed 2 with the winner's Phase 10 recipe (20 epochs, `mapmad_v2_twin.yaml` / `mapmad_v2_seen_twin.yaml`), only on a GPU
the chain does not hold, starting after the chain's fine-tune has its GPU; G10 rows 3–4 + seen list on its ema_19. It never
delays the chain and never holds the gate.

## 8. Automatic stops (`STOP_reason.txt`, STATUS.md → STOPPED, fix nothing)

Checkpoint sha256 / set fingerprint mismatch; duplicate ids; crash rate > 1% of a step's runs; runners exiting without
progress; determinism episode not identical; label check above a limit (§4); skipped teacher-fail states outside collision
streaks > 40% of the visited states; DAgger frontier cache ≠ the live harness count; NaN loss; a second training crash; any budget estimate > 1.5×.
The online map ≠ generator map check ran before this file (§2: passed).

## 9. Budgets (pre-approved; GPU waiting not counted; stop at > 1.5×)

Step-0 dev check ≤ 4 h · rollouts + labels ≤ 6 h · fine-tune ≤ 4 h (estimate after epoch 1) · offline checks ≤ 1 h · dev
check v2.1 ≤ 2 h · twin ≤ 7 h.

## 10. GPUs (v2 plan §8 rules 1–10)

A GPU is free after 4 consecutive 5-min checks without another user's process (watcher `p11_gpuwatch`; our processes carry
MAPMAD_P11_OWNER); live `nvidia-smi` right before taking one; another user's process on a GPU we use → its pairs finish
their current run and take no more (training already started is not killed; logged); a second GPU joins the run queue
when free; never share a GPU. Priority: the chain > the twin; no mutual waiting. Claim flag if ever needed:
`/mnt/shared_disk/nazli/nomad_outputs/mapmad/P11_WANTS_GPU`.

## 11. Gate G11 (v2 plan v2.4)

| # | Check | Pass |
|---|---|---|
| 0 | v2 pick | dev check of run 1 vs run C applied mechanically from this file (tie → run 1); winner named |
| 1 | Rollouts + labels | done in the 145 labelled train homes; teacher labels checked on 10 random states (sheet in the summary); share of skipped (teacher-fail) states reported |
| 2 | Model checks | G10 rows 1–4 again on ema_9 (+ seen-list rule, report only) |
| 3 | Dev check (report only) | v2.1 vs v2 on the dev set, oracle-stop success |

A failed pass/fail row is recorded as failed and the phase stops for a decision with Naz.

## 12. Deviations from the confirmation block (decided by Naz before any judged result, 2026-10-22)

1. **Label-check limit 0.15 m → 0.25 m (+ waypoint-1 median ≤ 0.10 m), look-ahead "re-ask" instead of one frozen
   path.** Before this file, the look-ahead was measured against the real teacher on 16 Phase 8 TRAIN-split drives (2,227
   states; the 50 test drives untouched): frozen path median 0.235 m (p95 0.554), re-ask median 0.167 m (p95 0.546);
   per-waypoint medians (frozen path, 6 drives) 0.03 / 0.08 / 0.16 / 0.21 / 0.26 / 0.32 / 0.38 / 0.42 m — the error
   grows with the horizon because the real teacher replans as it sees new rooms and Phase 8's motion adds detours and
   kicks, which a label from the map at t cannot know; the relabelled goal equalled the logged goal on 100% of the
   frames. With 0.15 m the chain would have stopped at the label check. (A DRY run of the check, limit not applied, on
   4 test-split drives drawn with the same seed gave median 0.173 m, waypoint 1 0.012 m; those drives may be among the
   50.)
2. **Skip order collision first** (the 40% stop counts teacher-fail states outside collision streaks). On 7 DRY rollouts
   of run 1: teacher-fail first 32.7%, collision first 18.5% teacher-fail / 39.5% collision.
3. **Dev homes in the rollouts:** the 145 labelled train homes include the 10 dev homes; their 100 rollouts form the
   DAgger test split (never trained on), as Phase 8 held out the val-drive homes. (Claude Code decision, to confirm.)
4. **Map parity by pose replay:** "replayed actions" cannot match exactly because Phase 8 stores the commands as float32
   (pose drift ~1e-8 m); the pass/fail check replays the saved poses and applies the bump rule to the logged moves; the
   action replay is reported (51 / 9 frames differ). (Claude Code decision, to confirm.)
