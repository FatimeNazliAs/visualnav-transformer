# Phase 14b-1 rules (v1, 2026-11-03) — fix the target marks in the stop module (no training)

Frozen before any judged result (sha256 in `p14b1_rules.sha256`; copy in `mapmad/configs/p14b1_rules.md`). Kickoff:
`~/projects/nomad-mapmad/.p14b1_kickoff.md` v1.1. Config `mapmad/configs/p14b1_marks.yaml` (extends `p14a_finish.yaml`;
`runs_from: p14a_finish`). A rule may be replaced only before its result is judged (new version + sha256).

## 1. Scope
- **No training, no driving, no exam.** The Phase 14a practice runs (219 episodes, p13d ∪ p14d, fingerprint
  `8d7b623e739c…`, halves A 108 / B 111) are re-scored. Their logs, evidence and dets files are only read; sha256 of
  every one of them before and after the chain (`checks/hashes_{before,after}.json`) must be equal (gate row 6).
- Arms: M21b, M21bh, N, F, VF (baseline runs, 1,095); driving tests A1, W4, A1 + W4 × M21b, M21bh, N (1,971; step 7 only).
- The marks are rebuilt **only inside the stop module**. The network's map input stays as in Phase 14a.
- HM3D exam and exam B are never rendered or run.

## 2. Replay (14b1.1; kickoff §3 fallback "masks / depth", expected)
- The 14a evidence holds cells, not masks or depth. Every run is replayed pose by pose (`mapmad/scripts/p14b_replay.py`):
  Phase 13's replay (`sidecar_server.SimService`, `search_wire.arm_wire`, logged noise seed) with the arm's policy
  blocks removed (F's teacher, VF's value map, rule value, logging; none changes the map or the recogniser's input).
- **Exact poses:** frame t is rendered at the evidence's full-precision 2D pose and the navmesh height at that point
  (`pathfinder.snap_point`, kept only when within 1e-6 of the log's 6-digit height). Part 1 smoke test (one run): the
  log's rounded poses moved YOLOE scores by up to 9.8e-4 (two honest fixes: evidence poses, then navmesh height);
  with both, replay = original byte for byte (max |score diff| 0.0, 1,186 / 1,186 rows, base cells identical).
- Where `snap_point` disagrees with the log's height (slopes / steps) the log's 6-digit height is kept; such frames are
  counted (`height_log_frames`) and may fall back to F1-box rows (DRY: 2 of 5,902 rows, on two such frames).
- **Same GPU:** a run is replayed on the GPU that first drove it (progress logs: shard = 10 × GPU + pair).
- **Replay check per frame:** same detection count and every |raw − logged raw| ≤ 1e-4 → masks row by row. Else per
  logged row: the unused replay detection with box IoU ≥ 0.9 and |raw diff| ≤ 1e-4; none → **F1-box** (the logged box
  filled with pixels within 0.3 m of its median depth; its base cells stay the logged ones). Counted and reported.
- True instances: the evidence's true rows (simulator semantic picture), mask = the instance's pixels.
- One YOLOE server per shard; SHARDS_PER_GPU = 6 (Alone mode).

## 3. Fix F1 — where a mask lands (`mapmad_sim/mark_projection.py`)
- **base:** every mask pixel through its depth onto a 10 cm storey cell (today; must equal the 14a evidence cells).
- **F1-med** `med_d<d>`, d ∈ {0.2, 0.3, 0.5} m: keep mask pixels whose depth is within d of the median depth of the
  mask's pixels with depth > 0, then project.
- **F1-clu** `clu_k<k>_m<m>` (VLFM `object_point_cloud_map.py`; ConceptGraphs): erode the mask (3 × 3 kernel, k ∈ {1, 2,
  3} iterations); back-project the remaining pixels with 0 < depth ≤ 4 m; DBSCAN eps 0.2 m, min points m ∈ {20, 50,
  100} pixels; keep the largest cluster only (most pixels; empty → no mark from this look); mark the cells of its pixels.
  **DBSCAN on 5 cm voxels weighted by their pixel count** (kickoff fallback "DBSCAN too slow", taken up front: masks up
  to ~77k pixels, ~7 M mask pixels per run): a voxel is core when the pixels of the voxels within eps (itself included)
  are ≥ m; a border voxel joins its first core neighbour's cluster.
- **Tentative (VLFM, F1-clu only):** the mask's box lies in an outer third of the picture (x1 ≤ W/3 or x0 ≥ 2W/3), or
  the cluster's median depth > 0.95 × 4 m. A tentative look is pending; at a later frame where the cell is in view
  within 3 m of the robot and no detection covers it, every pending look on the cell becomes a miss.
- Rule 2's size factor keeps the original mask's pixel count (F1 moves marks, not the strength of a look).

## 4. Fix F2 — scores without the stretch
- **map** (today): map value 0.4 + 0.6 · (s − 0.01) / (0.788 − 0.01); Rule 1 reads the raw score (as 14a).
- **cal** (F2): s_cal = sigmoid(logit(s) / T) for the value **and** Rule 1. Logits recovered from the scores
  (logit = log(s / (1 − s))): the logs hold no raw logits (kickoff fallback).
- **T** fitted on the Phase 9 photo test, six categories (p9q items; TP / FP at box IoU ≥ 0.5 from Phase 9's labels,
  IGN rows left out), **half A** homes (5 labelled Phase 1 homes); grid 0.40–3.00 step 0.05; minimal ECE (10 equal
  bins on [0, 1]; ties → T nearest 1). ECE raw / today's map value / calibrated reported on half B.
- **c_cat** (ApexNav idea, separate row): detections with s_cal < c_cat(category) are dropped; per category c ∈ {0.1,
  0.2, 0.3, 0.4, 0.5} picked on practice half A (row 8, §6).

## 5. Fix F3 — stop geometry
- A stop cell must be ≤ 1.0 m from the robot, within ±30° of its heading **and** in view in the current frame (one
  of that frame's view cells = passes the depth test now). Applies to today's rule, Rule 1 and Rule 2 (Rule 2 without
  F3: ≤ 1.0 m in any direction, as 14a). Both F3 off and on are scored from the same looks.

## 6. Variants, grids, picks
- Grids (the 14a points are inside): today τ ∈ {0.05, 0.10, …, 0.95}; Rule 1 s ∈ {0.05, 0.1, 0.25, 0.4, 0.55} × n ∈
  {2, 3, 4, 5} × h ∈ {0.3, …, 0.7}; Rule 2 v ∈ {0.05, 0.10, …, 0.70} × g ∈ {1, 2, 3}. Definitions of today / Rule 1 /
  Rule 2 as `p14a_rules.md` §3.
- Rows per rule family (today / Rule 1 / Rule 2), one change at a time, then together:
  1. **base** (14a marks; key re-picked on the grids above) and **base (14a pick)** (τ 0.45 / `r1_s0.10_n2_h0.5` /
     `r2_v0.50_g1`, the 14a numbers);
  2. **+F3**; 3. **+F2**; 4. **+F1-med** (d and key picked jointly); 5. **+F1-clu** (k, m and key jointly);
  6. **+F1(best)+F2**; 7. **+F1(best)+F2+F3**; 8. **+F1(best)+F2+F3+c_cat**.
  F1(best) = of rows 4 and 5 the one with the higher half-A score (ties → F1-med), per family.
- **Pick (14a selection rule):** the grid point with the highest mean rule-stop success over the 5 arms on half A;
  ties → the higher threshold (today τ; Rule 1 h, then s; Rule 2 v), then fewer looks (n / g); between projection
  settings the milder one (F1-med larger d; F1-clu smaller k, then smaller m).
- **Row 8:** per category c_cat = the c with the highest half-A mean success over the 5 arms at row 7's key on that
  category's episodes (ties → the higher c); then the key re-picked on half A with those c.
- Reported on halves A, B and all.

## 7. Keep rule and recommendation (kickoff §3 item 9)
- A fix **helps** if on **half B** the mean rule-stop success over the 5 arms rises by ≥ +0.02 vs the same rule
  without it **and** the pooled STOP precision (correct stops / stops over the 5 arms) does not fall.
- Comparisons: +F3 / +F2 / +F1-med / +F1-clu vs base; +F1(best)+F2 vs +F2 (F1 on top) and vs +F1(best) (F2 on top);
  +F1+F2+F3 vs +F1+F2; +c_cat vs +F1+F2+F3.
- **Recommended for 14b-2** (report only, no exam): per family the row with the highest half-B mean success among the
  rows whose every added fix helps (a combination needs each of its steps); none → base. Overall = the family with the
  highest such half-B success.

## 8. True-mark ceiling
- The same projection variants with simulator-truth masks (every true instance in a frame = a look with score 1.0,
  MT-style; F2 does not apply), F3 off / on; reported at the YOLOE base / +F3 rows' keys (no re-pick) and at the truth
  variant's own half-A pick.

## 9. Metrics
- Per arm and mean, halves A / B / all: rule-stop success, SPL, oracle, capture (rule / oracle), STOP precision,
  wrong-object share, near-miss share (`p14a_rules.md` §3 definitions), **share of stops on cells not in view** at the
  stop frame, per category.
- **Mark statistics** (all baseline runs, per projection): per row (one detection or true instance in one frame) the
  median distance of its cells to the nearest usable target instance's box (0 inside) and its area (cells × 0.01 m²);
  rows split into YOLOE boxes on a true target (IoU ≥ 0.3), YOLOE boxes on nothing true, true instances.
- **Driving-test re-score (step 7, report only):** each family's recommended row vs today's rule τ 0.45 on the A1 /
  W4 / A1 + W4 runs, per arm, paired.
- **Videos (14b1.5):** the 8 Phase 14a mark-check runs (`videos_marks/picks.json`, M21b) re-rendered: the network's
  near window with the 14a marks beside the same window with the overall recommended variant's marks (today's fusion
  of that variant: mean over looks); Phase 13's box + pose check.

## 10. Statistics
- Paired by episode (mean over the 5 arms per episode), home-clustered bootstrap, 10,000 draws, seed 14401, 95 % CIs;
  one seed (seed index 0), stated.

## 11. Budgets and automatic stops
- Budgets (stop only > 1.5×): replay ≤ 8 GPU-h (stop > 12 GPU-h); temperature fit ≤ 0.5 h; bench ≤ 4 h CPU;
  driving re-score ≤ 1 h CPU; videos ≤ 1 h.
- Automatic stops (`STOP_reason.txt`): a replay incomplete after two rounds; a budget > 1.5×; a bench step failing;
  any run on an exam set. Fallbacks (logged, "Decisions to confirm"): only one GPU usable → every run replayed on it
  (other-GPU rows: ≤ 1e-4 accepted, else F1-box); F1-box rows (§2).

## 12. Gate G14b1
| # | Check | Pass |
|---|---|---|
| 1 | F1 / F2 / F3 one at a time and together, per rule and arm, halves A / B / all | reported |
| 2 | Keep rule (§7) applied; recommended variant for 14b-2 | reported |
| 3 | True-mark ceiling with each projection | reported |
| 4 | Calibration: T, ECE before / after (photo-test half B) | reported |
| 5 | Mark statistics old vs new; 8 videos | reported |
| 6 | Fairness: every baseline run re-scored, none missing; base = the 14a bench on every 14a grid point; trajectories untouched (hashes equal) | must pass |
| 7 | Replay check: boxes / scores equal the logged ones within §2 limits | must pass (else F1-box rows, reported) |
