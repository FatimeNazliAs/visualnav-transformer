# Phase 7 sweep rules (written before any sweep run; sha256 in p7_rules.sha256)

Sources: kickoff prompt + "Phase 7 confirmation" block (2026-10-16; items in brackets) + Naz's S-lens answers
(2026-10-16: 2a, 3b, 4b). Config: `mapmad/configs/p7_sweep.yaml`. Original NoMaD only; dev homes only; no training.

## 1. Settings [5] + Naz 3b — 8 cameras
| name | lens | height (m) | pitch | who |
|---|---|---|---|---|
| pin018 | pinhole 66.5° HFOV, 320 × 240 | 0.18 | −2° | LIMO (= the Phase 5 camera) |
| fish018 | fisheye | 0.18 | 0° | |
| pin025 / fish025 | pinhole / fisheye | 0.25 | −2° / 0° | SACSoN (Method A) |
| pin045 / fish045 | pinhole / fisheye | 0.45 | −2° / 0° | RECON (photo, Method A; replaces 0.95 m, Naz 3b) |
| pin055 / fish055 | pinhole / fisheye | 0.55 | −2° / 0° | GoStanford (Method A) |

- Fisheye [2] + Naz 2a: double sphere fitted to SACSoN's frames (`lens/sacson_lens.json`): fx = fy 117.5916, cx 82.1921,
  cy 54.0238, ξ 0.80, α 0.72, 160 × 120 (SACSoN's resolution, 4:3 → NoMaD's crop is a no-op), image circle 96.91 px
  (170°), black outside it. Pitch 0° (no accepted |pitch| ≥ 3° measurement: Method B dropped).
- All: camera 0.084 m ahead of the turning centre, camera height measured every step against the real floor, LIMO's
  collision body and floor map (radius 0.195 m, height 0.55 m; no camera is above 0.55 m, Naz 3b → no raised floor
  map). Depth camera unchanged (not used by NoMaD arms). Speed limits, 4 Hz, controller: Phase 1/5.
- Heights (7.4, `height/height_estimates.md`): Method B failed the RECON check (rule anchored at 0.95 m) and cannot
  work on SACSoN (no metric odometry) → dropped; Method A decides (GoStanford 0.55, SACSoN 0.24 → 0.25).

## 2. Episodes [6] + Naz 4b
- Frozen set `episodes/episodes.json`, fingerprint **`bddbb734157f37400b58218a342c06ca62ad0353a2d92c170ebf92b4f9f32189`**:
  **108 in-view + 30 out-of-view**, 10 dev homes (00746, 00440, 00272, 00330, 00241, 00732, 00081, 00534, 00397, 00401).
- Built from Phase 1 in-view 30 (`8ea943b2…351a`) + 30 (`f1916fa5…1e94`) + 60 new (Phase 1 rules, seed 1001,
  `p7x-iv-*`) and Phase 1's 30 out-of-view; Phase 5 view points and goal-photo rule (closest to 1.0 m, ≥ 5% at
  pinhole 0.18 m); goal photo = same pose for every camera, rendered with that camera. 0 dropped for the floor map (no
  raised floor map); 12 in-view episodes dropped because their start is already a primary success (walking distance
  ≤ 0.2 m to a view point, Naz 4b). Ids unique (freeze + runner refuse duplicates).
- Superseded set (0.95 m camera, before 3b/4b): `episodes/superseded_095/` (`0b7a09a8…901e`), never run.

## 3. Arms, seeds, runs [6]
- Main: `B_<camera>` = official NoMaD + goal photo on the 108 in-view episodes. Secondary (report only, never
  decides): `A_<camera>` = official NoMaD explore on the 30 out-of-view episodes.
- Official NoMaD exactly as Phase 1/5 arms A/B (audit `audit/audit_table.md`): `nomad.pth` sha256 `70f79b82…a333`
  checked at load; 4 context frames one per step, 4-frame wait, 8 samples / sample 0, waypoint 2, × 0.05 m, pd_controller.
- 3 seeds per episode; noise seed = first 8 hex of sha256("mapmad-p7/<episode_id>/<seed index>"), identical across
  cameras. Timeout 1,000 steps; oracle stop at primary success.
- Runs: 8 × (108 + 30) × 3 = **3,312**. Order arm by arm (one home load per camera); 6 Habitat + NoMaD pairs, 3 per
  GPU (only GPUs free at launch, `nvidia-smi`); results do not depend on the number of pairs.

## 4. Metrics [6]
- **Primary:** walking distance ≤ 0.2 m to the nearest usable ObjectNav view point of the target (oracle stop).
- Phase 1 metric: walking distance ≤ 1.0 m to the target point at any step. SPL (to the view point).
- Collisions per run (a command's move > 1 cm short); distance driven before the first collision (runs without a
  collision: their path length, reported separately as "no collision"); stuck streaks (≥ 100 consecutive collision steps).
- Target pixel share at the start and in the goal photo per camera (`episodes/shares.md`, report only).

## 5. Statistics [6]
- Per episode: mean over its 3 seeds. Setting success = mean over episodes.
- 95% CIs: episode-level bootstrap (10,000 resamples, seed 7700, percentile); home-clustered bootstrap (10 homes
  resampled, 10,000, seed 7701) also reported.
- Paired differences vs **pin018** (the Phase 5 camera) on the same episodes and seeds, both bootstraps.

## 6. Choice rule [7] (applied mechanically by `p7_tables.py`; Naz 3b: training-like uses 0.25 / 0.55 m)
1. S(c) = mean primary success of `B_c` over 108 episodes × 3 seeds. Best = the camera with the highest S.
2. Training-like T = the better (higher S) of **fish025** and **fish055** (tie → fish025, the SACSoN-like one, as SACSoN
   is NoMaD's larger indoor dataset). If S(T) ≥ S(Best) − 0.05 → **T wins**.
3. Otherwise: the candidates are every camera with S ≥ S(Best) − 0.03 (Best included); if Best is the only one, Best
   wins; else the candidate with the fewest collisions per run (mean over its `B_` runs) wins.
4. Still tied (candidates whose collisions per run are within 5% (relative) of the lowest) → fisheye over pinhole,
   then the height closer to an indoor estimate (|h − 0.25| or |h − 0.55|, whichever is smaller), then higher S,
   then the name in alphabetical order.
- Explore (`A_`) never decides. Winner → `mapmad/configs/camera_fair.yaml` (lens model + parameters, height, pitch,
  resolution, forward offset) = the camera for all later phases.

## 7. Determinism and compatibility (before the sweep, by the driver)
- Determinism (G7 row 5): `B_fish025` on the first in-view episode of the set, seed index 0, twice → byte-identical logs.
- Compatibility (report): `B_pin018` on the same episode equals the Phase 5 camera path (an arm without a camera, same
  seed) byte for byte apart from the log header's `camera` field.

## 8. Report-only extra [1, changed]
- After the choice: the winning camera with a context frame every **3rd** step (≈ 0.15 m; NoMaD waits until all 4
  spaced frames exist), 108 in-view episodes × seed index 0. Never changes the choice. Compared with the winner's
  seed-index-0 runs on the same episodes; > 5 points better → open issue for the next phase.

## 9. Unattended run, stop conditions (Phase 5 precedent) and bugs
- One driver script in `screen` (`mapmad_p7_sweep`), resumable (`driver/<step>.done`). Before the sweep it stops with
  `STOP_reason.txt` if any fails: this file's sha256 = `p7_rules.sha256`; episode fingerprint = §2; `nomad.pth`
  sha256; git HEAD = the committed sweep code and the worktree has no changes in `mapmad/` or `train/vint_train/`;
  the determinism check (§7).
- During the sweep: infrastructure errors retried twice (same seed), then a crash record; > 2% crashed runs, or any
  other exception → stop with `STOP_reason.txt`; nothing is fixed or rerun until Naz is back.
- Any other exit that is not a finished run — a crash, an empty variable (`set -u`), a kill (TERM / INT / HUP) — also
  writes `STOP_reason.txt` with the exit code and the last log line (trap on EXIT; long steps run in the background +
  wait so a kill is handled within seconds; the driver then stops only our own runners and servers). `kill -9` cannot
  be trapped: a watcher on the host reports when the screen session ends (Naz, 2026-10-16, after the first dry run
  died silently on an empty variable).
- Dry runs (`p7_dryrun.yaml`, folder `/outputs/mapmad/p7_dryrun/`, marked DRY) test the driver only; never reused or
  reported as results.
- Lens approval: the full sweep starts only after Naz's one-line S-lens verdict (her rule, 2026-10-16).
- Bug rule (§0.4): code that does something different from these rules, shown by a test failing before and passing
  after the fix → every run of every affected camera reruns from zero; superseded runs are kept and reported. Low
  success alone is never a bug.

## 10. Outputs
`/outputs/mapmad/p7_camera/sweep/` (`runs/logs|summaries/<arm>/`, `tables.md`, `choice.md`, `extra.md`, `videos/`
one per camera on the same episode, side by side), `mapmad/configs/camera_fair.yaml`, `phase7_summary.md`.
