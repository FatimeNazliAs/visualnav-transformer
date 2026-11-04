# Phase 14b-2 rules (v1, 2026-11-04) — better recognition + drive-to-the-mark

Frozen before any judged result (sha256 in `p14b2_rules.sha256`; copy in `mapmad/configs/p14b2_rules.md`). Kickoff:
`~/projects/nomad-mapmad/.p14b2_kickoff.md` v1.0. Config `mapmad/configs/p14b2_recog.yaml` (extends `p14b1_marks.yaml`
→ `p14a_finish.yaml`). Outputs `/outputs/mapmad/p14b2_recog/`. A rule may be replaced only before its result is
judged (new version + sha256).

## 1. Scope
- **No exam.** HM3D exam homes, exam B (MP3D) and the photo-test homes are never rendered, run or used for training.
- Training frames come only from HM3D **train** homes with HM3D Semantics labels, minus the 10 photo-test homes.
- Driving: the 14a MP3D practice set (219 episodes, p13d ∪ p14d, fingerprint `8d7b623e739c…`, halves A 108 / B 111),
  seed index 0; the 14a W4 runs are the reference cell (read only; sha256 before / after must be equal).

## 2. R1 · training frames (`mapmad/scripts/p14b2_trainset.py`)
- **Pool:** HM3D train homes with semantic labels (145) minus the photo-test homes (Phase 1 / Phase 9: the 10
  `val_drive.phase1_homes` = the homes of the Phase 9 frame set) → 135 homes. **Exclusions** (lists + sha256 frozen
  in `trainset/homes.json` before rendering): photo-test homes, every HM3D val home (100, exam), every MP3D scene (90:
  practice set + exam B). Check by full id and by scene id: pool ∩ each list = ∅ (gate row 6), else the chain stops.
- **Poses:** every Phase 8 drive of a pool home (habitat_mapmad_v2, train + test splits), thinned to poses ≥ 0.5 m or
  ≥ 30° apart (`photo_frames.thin_poses`); at most 1,000 per home (seeded subset; seed 1402).
- **Render:** Phase 9's renderer (`p9_build_frames.open_home` / `render_truth`: camera pin025, p8_datagen_v3a robot),
  320 × 240 RGB + semantics; frames over a mesh hole skipped (counted).
- **Labels (one picture, 6 classes):** Phase 9 tables (`photo_labels.six_tables`: Phase 8 mapping + near-synonym ignore
  list). An instance counted for a category with ≥ 50 px → that class; a counted instance < 50 px → ignore; a
  near-synonym → ignore, **except** near-synonyms with a 14a look-alike word (stool, bench, ottoman, footstool, pouf,
  pouffe) → background (they are the look-alikes the detector must learn to reject); everything else background.
  Ignore pixels are painted grey (114) in the training picture.
- **Selection:** positive frames (≥ 1 labelled instance) and target-free frames (no labelled or ignored target pixel);
  60,000 frames, 40 % target-free (fewer positives → all positives, free scaled to keep 40 %); target-free frames
  showing a look-alike drawn 3× as often (weighted sampling without replacement, seed 1402). Split by home 90 / 10
  (train / val; val for early checks only, never for picking). Fewer than 20,000 usable frames → use what exists,
  reported.

## 3. R2 · fine-tune YOLOE-11-L-seg (`mapmad_recog/finetune.py`; ultralytics 8.4.183, Phase 9's pinned version)
- Base weights: Phase 9's `yoloe-11l-seg.pt` (sha256 in `p9_recognisers/photo_test/weights.json`). Fixed vocabulary =
  the 6 Phase 9 prompts (chair, couch, bed, toilet, tv, potted plant), fused into the head (`YOLOEPESegTrainer`).
- **LP:** linear probing — ultralytics' recipe: every layer frozen except the head's class projections `cv3.*.2`.
  **FT:** full fine-tune (no freeze). Same fixed recipe: 20 epochs, image 320, batch = ultralytics auto-batch at 90 %
  of GPU memory (≈ 2.4 GB headroom; the value used is logged), seed 0, deterministic, default augmentation and
  optimiser; the **last epoch** is used (no picking on val). LP and FT train in parallel, one GPU each.
- Fallbacks: FT out of memory → auto-batch handles it; FT still failing → LP only (logged).
- **Photo test (Phase 9 rules unchanged, `p9_rules.md` §4–§5):** the 6-category items of the Phase 9 frame set, raw
  pass (top 10, score ≥ 0.01), threshold = the grid value with the highest half-A map F1 (ties → lowest), metrics on
  half B (AP50, precision / recall, false alarms per target-free item, map hit rate H, map recall R, map F1, per
  category), timing pass. Base YOLOE = Phase 9's scores (read only).
- **Pick:** of base / LP / FT, the highest **half-A** map F1 (ties → base, then LP). Base picked → fallback: R3 / R4
  run without a fine-tuned detector cell (A-mark still tested).
- Driving settings of a fine-tuned detector: its threshold t and p99 of kept half-A scores (map value 0.4 +
  0.6·(s − t)/(p99 − t), as Phase 9); one category per call = the asked class only (the fixed vocabulary).

## 4. R3 · replay re-score (stop module only; trajectories unchanged)
- Every 14a baseline run (1,095; M21b, M21bh, N, F, VF) replayed pose by pose (14b-1 replay, `p14b_replay.py
  --detector`) with **each** fine-tuned detector (LP, FT): every detection the detector keeps (score ≥ t) becomes an
  evidence row (raw score, map value, mask pixels, box, cells); the logged evidence's other arrays (poses, view cells,
  true rows, first-seen maps) are kept. Marks: 14a projection (base) and F1-med 0.5 (14b-1's F1 pick).
- Temperature T refitted per detector on photo-test half A (14b-1 recipe; ECE half B reported).
- Stop bench (14b-1 code): today's rule / Rule 1 / Rule 2 × {base, F1-med}, map scores (calibrated scores report
  only); keys picked on half A (14b-1 grid and tie rule, F3 off).
- **Keep rule** (14b-1): half B, mean over the 5 arms, correct stops (rule-stop success) ≥ +0.02 **and** pooled STOP
  precision not lower, vs the same rule and marks with base YOLOE (14b-1 bench). CIs: home-clustered bootstrap.
- Also reported: share of detections on a target (box IoU ≥ 0.3 with a true box of the frame; base 17 %), wrong-object
  stop share.

## 5. R4 · driving test (`mapmad_sim/amark.py`; practice set, seed 0, W4 on in every cell)
- **A-mark (drive to the mark):** each control step, when idle, an *accepted cluster* = the strongest noisy-layer cell
  with value ≥ τ (0.45, the dev choice), seen from ≥ 2 views, **in view now** and ≤ 3 m from the robot, not within
  0.5 m of a cluster already handled. Goal = the nearest free cell (explored, outside the inflated obstacles / bumps)
  within 0.9 m of it; none → rejected ("no free point"). The robot drives there with arm F's planner and steering
  (`explorer.teacher_step`, `FrontierPolicy`; the network is not asked), turns to face the cluster, and the stop rule
  is evaluated there: rule value ≥ τ → confirmed (a rule stop, scored offline as always); else rejected
  ("unconfirmed"). Planner failure → rejected ("no path"); > 120 steps → rejected ("timeout"). Every handled cluster
  is remembered and the network drives on. Same switch for M21b, M21bh, N.
- **Cells:** detector {base, picked} × A-mark {off, on} × {M21b, M21bh, N}; base + off = the 14a W4 runs (reused);
  new runs: base + on (`runs/am`), picked + off (`runs/det`), picked + on (`runs/detam`) = 3 × 657 = 1,971. With the
  picked detector the network's target layer also changes (marks from the new detector), the same for every arm.
- **Metrics** (today's rule, τ 0.45): oracle success, rule-stop success, SPL, STOP precision, wrong-object share,
  collisions, stuck runs, A-mark events; per arm and the mean over the 3 arms; each cell vs base + off on half B
  (paired by episode, home CI; keep rule as §4); **map effect M21b − M21bh per cell** (all and half B).

## 6. Statistics
- Paired by episode, home-clustered bootstrap, 10,000 draws, seed 14401, 95 % CIs; one seed (seed index 0), stated.

## 7. Budgets and automatic stops
- Budgets (stop only > 1.5×): R1 ≤ 4 GPU-h, R2 ≤ 8 GPU-h, R3 ≤ 4 GPU-h, R4 ≤ 6 GPU-h; GPU-time = wall time × GPUs.
- Automatic stops (`STOP_reason.txt`): contamination; a budget > 1.5×; checkpoint sha256 mismatch; duplicate ids;
  crash rate > 1 %; a determinism mismatch on the same GPU; a pool worker dying twice (once → relaunch); a failed
  must-pass row after two honest fixes. A progress watchdog restarts a silent step once (no new item for 30 min).

## 8. Gate G14b2
| # | Check | Pass |
|---|---|---|
| 1 | Photo test half B: base vs LP vs FT | reported |
| 2 | R3 stop bench with the fine-tuned detectors, keep rule | reported |
| 3 | R4 driving cells, map effect per cell | reported |
| 4 | Detection share on targets, wrong-object stops, calibration | reported |
| 5 | 8 videos (zoom panel; the 14a mark-check episodes, M21b + picked detector + A-mark; box + pose check) | reported |
| 6 | **No contamination:** training homes ∩ (photo test ∪ HM3D val ∪ MP3D) = ∅ (id lists + sha256) | must pass |
| 7 | **Fairness:** every R4 run present on the frozen practice fingerprints, checkpoint sha256 and switches matched; switch off = 14a byte-identical (M21b_w4, N_w4 on 2 p13d episodes, each vs the 14a run driven on the same GPU); 14a files unchanged | must pass |
| 8 | **Determinism:** one p13d episode twice on the same GPU (M21b and N with the picked detector + A-mark) identical; R3 replay check (poses equal, true rows re-projected equal) | must pass |

**Self-close** (kickoff): rows 6–8 pass → `/commit`, ff-merge into `mapmad`, `STATUS.md` → `CLOSED <hash>`; never push.
