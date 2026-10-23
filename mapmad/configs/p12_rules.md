# Phase 12 rules — the search exam (v1)

Written 2026-10-23 before the first dev run, from the Phase 12 confirmation block (Naz, 2026-10-23) and v2 plan v2.5
§8.7–8.9. Frozen with its sha256 (`/outputs/mapmad/p12_search_exam/p12_rules.sha256`); copy in `mapmad/configs/p12_rules.md`.
A rule may be replaced only before its result is judged, with a new rules version and sha256.
Run config: `mapmad/configs/p12_exam.yaml` (arm table, sets, run settings). Code: commit named in `driver/chain.log`.

## 1. What is judged
- **Headline (G12 row 1):** oracle-stop success of the headline pair, maps shown − maps hidden (same weights), on the
  main exam set; pass = difference > 0 with home-clustered one-sided p < 0.05. Row 1 failing → recorded, stop for Naz
  (no pre-set fallback).
- Everything else is reported (G12 row 2), with fairness (row 3), determinism (row 4) and videos (row 5) as gate rows.
- Exam homes are rendered only to build and run the frozen Phase 12 sets; nothing is tuned on them.

## 2. Headline checkpoint (block item 1; dev set only, before any exam run)
- Dev set = Phase 11's `dev_set/episodes.json`, fingerprint `c7e64fcedfd981864af8b84874e87a407f545f67eb9e9503ec9bfce7a4e067af`,
  120 episodes × 3 seeds, oracle stop, simulator-truth target layer with the Phase 8 noise, noise seeds of Phase 11
  (salt `mapmad-p11`).
- Δ(c) = success(maps) − success(maps hidden), per checkpoint c ∈ {v2, v2.1}; success = mean over episodes of the
  per-episode mean over 3 seeds. Maps arms = Phase 11's runs (`p11_dagger/step0` arm run1 = v2; `p11_dagger/dev_v21`
  arm v21 = v2.1). Maps hidden = rerun here (arms `v2h_dev`, `v21h_dev`, 3 seeds, 1,000 steps).
- **Headline = v2.1 unless Δ(v2.1) < Δ(v2) − 0.02, then v2.** Both pairs run in the exam; the truth arm (MT) and the
  Qwen3-VL arm (MR1) use the headline checkpoint. If both Δ ≤ 0: the rule is applied anyway, "both Δ ≤ 0" goes to
  STATUS.md, no stop.
- **Refactor check:** seed 0 of `v2h_dev` must equal Phase 11's `run1_hidden` log on every step and field Phase 11
  logged (Phase 11 runs ended at success, so its log is a prefix; header field `arm` excluded). Not equal → automatic stop.
- Written mechanically to `P12_DEV_CHOICES.json` (headline, both Δ, τ).

## 3. Recogniser (block item 2)
- **YOLOE-11-L-seg for every arm, every control step**, one category per episode, threshold 0.01, map score
  0.4 + 0.6·clip((s − 0.01)/(0.788 − 0.01), 0, 1) (Phase 9 p99 0.7878322374820708), detections through Phase 8's fusion.
  One YOLOE sidecar server per Habitat + policy pair, on the pair's GPU.
- **Qwen3-VL-8B** (rev `0c351dd0`, greedy, ≤ 256 new tokens, bf16, score 1.0) only on a seeded 60-episode main-exam
  subset (stratified by distance bin) × 1 seed, headline checkpoint with maps (arm MR1), report only, on a GPU not used by
  the pairs (one pair per GPU, after the main queue).
- The truth arm (MT) = upper limit. No switch to RF-DETR.

## 4. Arms (block item 3; arm table in `p12_exam.yaml`)

| Arm | Policy | Checkpoint (sha256) | Maps | Target layer | Seeds |
|---|---|---|---|---|---|
| M21 | MapMaD v2 settings | v2.1 `ema_9` (`ebaabe28c490567cbd453ae426332864ca1ef22961bf1fabae59bb62bdb7c74f`) | shown | YOLOE | 3 |
| M21h | same | v2.1 | hidden | YOLOE | 3 |
| M2 | same | v2 `ema_19` (`6d2492f19f7f1a5ccadea3f120b66a78b04bb5e88d10ca701fc523e0f9aee984`) | shown | YOLOE | 3 |
| M2h | same | v2 | hidden | YOLOE | 3 if the headline is v2, else 1 |
| MT | same | headline | shown | sim truth + Phase 8 noise | 3 |
| MR1 | same | headline | shown | Qwen3-VL | subset, 1 |
| N | official NoMaD explore, Phase 5 arm A settings | `nomad.pth` (`70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333`) | – | YOLOE | 3 |
| V1 | Phase 5 MapMaD arm, heat layer zero, one map token | Phase 4 C `ema_19` (`8093c6dba5f91bce48903e74ad9128fde50cccbf9a8f587f9aec8b288bffd9e2`) | v1 map | YOLOE | 3 |
| F | map + frontier | – | – | YOLOE | 1 (deterministic, counts for all 3) |
| N-fish | official NoMaD explore, Phase 7 fish045 camera | `nomad.pth` | – | none | 1, report only |

- MapMaD v2 settings (Phase 11): context every 5th step, missing frames repeat the first, 8 samples / sample 0,
  waypoint index 1, 0.25 m per step; photo hidden. Official NoMaD: context every step, 0.05 m, waypoint index 2,
  4-frame wait. All arms: 0.2 m/s, 0.4 rad/s, 4 Hz, no reversing; pin025 camera (N-fish: fish045).
- **F:** `explorer.teacher_step` (option A settings, `p8_datagen_v3a.yaml`) on the same online map (obstacle /
  explored as known + bumped cells), every control step, state carried. F's "seen" = any map cell ≥ τ seen from ≥ 2 views,
  **at any distance** (the 1.0 m / ±30° condition is for the stop only): the strongest such cell, not within 1 m of a
  mark F already went to → goal = the nearest free cell (explored, not within 0.195 m + half a cell of a known obstacle or
  bump) within 1 m of it, given to the teacher as one seen instance → path; steered as the v1 planner arm (path point
  0.5 m ahead → pd_controller; behind → turn on the spot). Teacher ARRIVED and facing the mark (±15°) → the mark is
  remembered as visited and F searches on. Teacher failure (no frontier, blacklist exhausted, no path) → turn on the spot
  (left), as the planner arm without a plan.
- **F check (automatic stop if not):** F with Phase 8's simulator-truth "seen" (arm F_truth), run inside the simulator
  server as in the exam, makes the same decision (mode, goal, next path point) as the Phase 8 teacher at every frame of
  2 seeded (seed 12301) Phase 8 search drives with an approach in the dev homes (test split), pose replay.
- **N-fish:** the Phase 7 fisheye renders the colour and semantic cameras as fisheye while the depth camera stays a
  pinhole, so no target layer and no truth "seen" can be drawn consistently: N-fish gets no target layer (`none`) and is
  scored with the oracle stop only (report only).

## 5. Stops: one run, two scores (block item 4)
- Every run: 1,000 control steps, no stop. Logged every step: the largest noisy-layer value among cells whose centre is
  ≤ 1.0 m from the robot's turning centre and within ±30° of its heading, seen from ≥ 2 views (the fusion's view count),
  and that cell.
- **Success zone:** walking distance ≤ 0.2 m to the nearest usable ObjectNav view point of any instance of the category
  (OVON: OVON's view points).
- **Oracle stop:** the first step in the zone (step 0 counts); SPL with the path driven up to it.
- **Rule stop:** the first step (≥ 1) with value ≥ τ; success if the robot is in the zone at that step; SPL with the path up
  to it; no such step = failure.
- **τ:** dev set, headline arm + YOLOE (arm `hl_yoloe_dev`, 3 seeds), one set of runs scored at every τ of 0.30–0.95 step
  0.05 from the logs; τ = the highest rule-stop success, tie → the higher τ; then fixed for all arms. Waypoint index stays 1.
- **Wrong-object stop:** the triggering cell is > 1.0 m (horizontally) from the box of every usable instance of the
  category. **Near-miss:** a rule stop outside the zone within 1.0 m more walking distance (walking distance to the
  nearest view point ≤ 1.2 m). **STOP precision** = rule-stop successes / rule stops.
- The same step-log rounding (6 digits) is used for the τ comparison everywhere. **Live check (automatic stop if not):**
  after τ is fixed, 2 dev episodes (the determinism episode + the next dev episode), headline + YOLOE, seed 0, run once
  without a stop and once stopping live at the first value ≥ τ, on the same GPU → the same stop step and success.

## 6. Sets (block items 2, 5–7; built and frozen in the chain, fingerprints in `sets.json` before any exam run)
- **Main exam:** every official ObjectNav HM3D v2 val episode of the 36 homes; official start + rotation; target = any
  instance on the start storey. Keep: start snaps ≤ 0.2 m onto the LIMO floor (on the real floor); an instance of the
  category reachable on the start storey; no instance visible (any pixel) in 8 headings at pin025; 3–15 m walking distance to
  the nearest usable view point; **start filter** (from ≥ 1 usable view point of ≥ 1 instance, one of 8 headings shows
  ≥ 200 px at ≤ 4 m). Draw: seeded (12201), ≤ 10 per home, aim 100 per bin (3–6 / 6–10 / 10–15 m), neighbour-bin fill
  (Phase 5 rule), within each bin round-robin over the categories (sorted category order, each category's seeded order).
  A short bin takes all its eligible episodes (logged, continue); **automatic stop only if a bin < 40** after the fill.
  Ids `p12e-<home5>-<category>-<ObjectNav id>`; duplicates refused.
- **Unseeable set (report only):** the eligible episodes dropped only by the start filter (counted per category), seeded
  (12202), ≤ 3 per home; ids `p12u-...`; headline pair + N, 1 seed, oracle stop.
- **OVON:** HM3D-OVON val (seen / seen-synonyms / unseen), same keep rules incl. the start filter; an OVON goal = the HM3D
  semantic object whose box centre is within 0.1 m of OVON's object centre (unmatched goals dropped, counted); success via
  OVON's own view points; `children_object_categories` count as the target too; the recogniser is asked with the episode's
  category name. 100 per split, seeded (12203), ≤ 5 per home; ids `p12o-<split>-<home5>-<category>-<id>` (non-alphanumeric
  characters → `_`). Arms: headline pair (3), N (3), F (1).
- **Exploration side test:** the main-exam start poses (ids `p12x-...`), no target (no target layer; F in explore mode),
  1,000 steps, 1 seed; headline pair, N, V1, F.
- **Qwen3-VL subset:** 60 main-exam episodes, seeded (12204), per distance bin proportional to the bin's size (largest
  remainder); ids unchanged.

## 7. Seeds
- Diffusion noise and target-layer noise seeded by (episode id, seed index): noise seed = first 8 hex of
  sha256("<salt>/<episode id>/<seed index>"), identical across arms. Salt: dev set `mapmad-p11` (reproduces Phase 11), all
  Phase 12 sets `mapmad-p12`. F is deterministic (seed index 0 stands for all 3).
- YOLOE scores differ across the two GPUs by float noise (≤ 1.4e-6, Phase 9): results are identical for a fixed GPU;
  which pair runs a run is decided by the shared queue. Determinism is checked on the same GPU.

## 8. Metrics and statistics (block item 8)
- Success, SPL (both stops), success@500 (both stops).
- Search phase: first sight = the first step where simulator truth sees an instance of the category (≥ 200 px at ≤ 4 m);
  steps to first sight; share found by 1,000. Approach phase: success given seen; steps from first sight to the stop.
- Mistakes: wrong-object stops, near-misses, STOP precision; collisions; stuck streaks (≥ 100 consecutive collision steps).
- House seen (search phase and exploration): a walkable cell (0.1 m, start storey, 8-connected to the start) is seen when
  its centre is ≤ 3.0 m from the robot, within the camera's horizontal field of view, and every point sampled every half cell
  on the straight line to it is walkable (line of sight on the true map); reported at 250 / 500 / 1,000 steps.
  Redundancy = 1 − distinct 0.5 m cells / 0.5 m cell entries.
- Per category, distance bin, home; OVON per split.
- Per-episode mean over seeds; paired differences over the episodes both arms ran (an episode with a crash record in either
  arm is dropped from that comparison); home-clustered bootstrap (36 homes, 10,000 resamples, seed 12601): one-sided p for
  the headline only, 95% CIs everywhere; episode-level bootstrap also reported.
- Recogniser driving test: YOLOE (headline arm) vs truth (MT), all episodes; Qwen3-VL (MR1) vs YOLOE (headline arm, seed 0)
  on the subset; Phase 9 photo-test F1 per category vs driving rule-stop success per category (Pearson r, report only).

## 9. Runner, order, budget, automatic stops (block item 9)
- Harness = Phase 11's (server `search_v2` + client + policy + runner), arm table; shared run queue, resumable (a run cut
  short is re-run from zero); 3 Habitat + policy pairs per free GPU, each with its own YOLOE sidecar server.
- Order: dev decisions → freeze → headline pair → N → V1 → F → other pair → truth arm → N-fish → OVON → exploration →
  unseeable → Qwen3-VL subset → tables → videos.
- **Budget:** exam stages (main exam, OVON, exploration, unseeable) ≤ 48 h on 6 pairs (≤ 96 h when only one GPU is usable
  at the timing), counted as GPU-running time scaled by the share of the two GPUs in use (Phase 11); the Qwen3-VL subset
  does not count. Dev runs: ≤ 8 h (stop at 12 h). The first 24 runs of the headline pair (4 exam episodes) are timed → the
  estimate. If the estimate exceeds the budget, trims in this order, each logged, until it fits: (1) drop N-fish;
  (2) OVON N → 1 seed; (3) drop exploration V1; (4) the non-headline maps-hidden arm → 1 seed. Still > 1.5 × budget →
  automatic stop. Running past 1.5 × budget → automatic stop.
- **Automatic stops** (`STOP_reason.txt`, STATUS.md STOPPED, fix nothing): checkpoint sha256 mismatch; duplicate ids; crash
  rate > 1% (per step); determinism episode not identical (same GPU); refactor check not equal; live rule-stop check not
  equal; recogniser sidecar down
  after 3 restarts; a bin < 40 after the fill; F check fails; a step whose runners exit without progress; a silent step
  (no run finished and no log written for 30 min) is killed and restarted (logged, not a stop).
- **Determinism (G12 row 4):** dev episode `p11d-00081-bed-2624`, headline + YOLOE, seed 0, twice on the same GPU →
  byte-identical log (the log holds no timing fields).
- DRY chain first (`p12_dry/`, never results) on 2 dev episodes through every step, incl. a recogniser-sidecar restart.

## 10. GPUs (block item 10)
- v2 plan §8 rules 1–11: free = no other user's process for 4 consecutive 5-min checks AND none in the last 60 min
  (rule 11); live `nvidia-smi` check right before taking a GPU; withdraw on another user's process (current runs finish,
  no new ones; re-queued); never share; no exception. Owner table + one decision line per launch (`phase12_log.md`).
  Only GPU 1 usable → 3 pairs there, stated in STATUS.md.

## 11. Videos (block item 11)
- 8 seeded random exam episodes (seed 12501) + 4 by rule (headline maps succeeds / maps hidden fails; headline stuck;
  headline wrong-object rule stop; F succeeds / headline fails), each the first qualifying episode of a seeded order;
  seed index 0; all main arms (M2, M2h, M21, M21h, N, V1, F, MT) side by side with camera, near + far windows, floor plan;
  captioned (arm, step, oracle and rule stop); rendered from the logs by replaying every logged pose through the exam's
  server code; small copies ≤ 1280 px wide, H.264 CRF 31, faststart. Naz reviews them at the gate.

## 12. G12
| # | Check | Pass |
|---|---|---|
| 1 | Headline | headline maps − maps hidden > 0 in oracle-stop success on the main exam, home-clustered one-sided p < 0.05 |
| 2 | Reported | M2 vs N, V1, F; M21 vs M2; rule-stop results; OVON splits; recogniser driving test; exploration side test; search/approach phases |
| 3 | Fairness | all arms × episodes × seeds on the frozen fingerprints; checkpoint sha256 matched every run; no run missing |
| 4 | Determinism | one headline episode twice, same seed, same GPU → identical log (timing fields excluded) |
| 5 | Videos | legible, captioned, reproducible from logs; reviewed by Naz |

## 13. Pre-flight (checked by the chain before the first run)
```json preflight
{"checkpoints": {"p10_v2_train/runs/v2_s1_2026_10_21_08_42_50/ema_19.pth": "6d2492f19f7f1a5ccadea3f120b66a78b04bb5e88d10ca701fc523e0f9aee984",
                 "p11_dagger/runs/v21_s1_2026_10_23_06_11_59/ema_9.pth": "ebaabe28c490567cbd453ae426332864ca1ef22961bf1fabae59bb62bdb7c74f",
                 "p4_train/C_2026_10_11_05_21_43/ema_19.pth": "8093c6dba5f91bce48903e74ad9128fde50cccbf9a8f587f9aec8b288bffd9e2",
                 "weights/official/nomad.pth": "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333"},
 "sets": {"p11_dagger/dev_set/episodes.json": "c7e64fcedfd981864af8b84874e87a407f545f67eb9e9503ec9bfce7a4e067af"}}
```
