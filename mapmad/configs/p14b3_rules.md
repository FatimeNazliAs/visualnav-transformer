# Phase 14b-3 rules (v1, 2026-11-05) — Retrain MapMaD v3 + final exam B

Frozen before any judged run and before any exam-B run (sha256 in `p14b3_rules.sha256`; copy in
`mapmad/configs/p14b3_rules.md`; the chain also stores the value in `checks/exam_rules_frozen.sha256` and re-checks it
right before the first exam run and after the exam: gate row 6). Kickoff `~/projects/nomad-mapmad/.p14b3_kickoff.md`
v1.0. Configs: `mapmad/configs/p14b3_v3.yaml` (arms, practice set, exam B; extends 14b-2), `p14b3_rollout_map.yaml`,
`p14b3_rollout_hidden.yaml` (V1), `train/config/mapmad_v3.yaml` (V2). Outputs `/outputs/mapmad/p14b3_v3/`. A rule may be
replaced only before its result is judged (new version + sha256). **Budgets are GPU-hours = wall time × GPUs in use,
one unit for every step.**

## 1. Scope and data
- **Exam:** exam B only (`p13_claim/examb_set`, fingerprint `98a8d920…`, 291 episodes, 26 MP3D homes), second use; the
  HM3D exam (`6fb61eb8…`) is never run or rendered; HSSD not in this phase.
- **Practice set:** the 14a set (p13d 120 + p14d 99 = 219 episodes, `p14a_finish/sets/practice.json` `8d7b623e…`,
  halves A 108 / B 111), seed index 0.
- **Training data homes:** HM3D train homes only. The round-2 start set = Phase 11's rollout set (`8e3428c7…`, 1,448
  episodes, 145 labelled homes) **minus the 10 photo-test homes** (= its 10 dev / test-split homes; gate row 6) →
  **1,348 episodes in 135 homes** (`rollouts_r2/set/episodes.json`, fingerprint `a6f172ac…`, Phase 11 ids). Exclusion
  lists = the frozen 14b-2 lists (`p14b2_recog/trainset/homes.json`: photo test 10, HM3D val 100, MP3D 90; sha256
  re-checked). Contamination (gate row 6): the rollout set, the round-2 drive folders and every training split list of
  v3 (Phase 8 train, Phase 11 map-DAgger train, Phase 13 hidden-DAgger train, round 2 maps shown / hidden train) ∩
  (photo test ∪ HM3D val ∪ MP3D) = ∅ by full id, scene id and home prefix (`make_p14b3_sets.py`).

## 2. V1 · DAgger round 2 (budget 16 GPU-h; rollouts stop AT the budget)
- **Driver:** v2.1b `ema_9` (`9ab0dbfb…`), once with both maps shown (`r2map`), once with both hidden (`r2hidden`);
  target layer from the fine-tuned YOLOE **FT** (`yoloe_ft`, `p14b2_recog/train/ft/weights/last.pt` `93c66d28…`,
  threshold 0.05, p99 0.9707, the 14b-2 driving settings); **W4 off, A-mark off, reverse controller off** (the network
  drives; NoMaD's controller); 1,000 steps max, stop at the success zone (Phase 11 rollout rule); seed index 0, noise
  salt **`mapmad-p14b3`**; one drive per episode saved in `<mapmad_data>/habitat_mapmad_v3_dagger_r2{map,hidden}`.
- **Order and budget:** maps shown first with at most 8 GPU-h, then maps hidden with the rest of the 16 GPU-h. A mode
  cut by its budget keeps its finished drives (fallback; ≥ 50 % of the 1,348 per mode required, else a safety stop).
- **Labels** (`p11_labels.py label`, `mapmad_sim/dagger_labels.py` + `recovery_labels.py`): the Phase 11 labeller
  unchanged (re-ask look-ahead 40 steps, every 5th → 8 waypoints, frozen blacklist) plus **W1 recovery labels**:
  - **stuck at frame t** (W4's rule on the record): ≥ 16 collision steps among the 20 moves before t, or < 0.10 m net
    movement over the last 40 steps; a stuck state gets a recovery label **before** the collision-streak and
    teacher-fail skips;
  - **direction:** the teacher's path (teacher_step at t) → the first path point ≥ 0.5 m from the backed-up position;
    teacher fail (no path) → the longest free ray on the robot's own map from the backed-up position (±180°, 10° steps,
    0.3 m start, 0.05 m march, 3.15 m reach; obstacles and bumped cells block; ties → smaller turn, then left);
  - **motion** (40 steps at 4 Hz): 10 steps backward at 0.1 m/s (0.25 m straight back), turn in place at ≤ 0.4 rad/s to
    the direction, then straight ahead at 0.2 m/s for the steps left (the waypoints hold positions only, so the turn
    must show as a move); poses after every 5th step → 8 waypoints in the robot frame (points behind allowed);
  - teacher-fail states that are not stuck: skipped (as before); collision-streak states that are not stuck: skipped.
  - Reported: share of recovery labels among all labels, per mode; teacher-fail share (> 0.40 recorded, no stop).
- Splits: every round-2 drive is in the train split (no round-2 drive in the dev / photo-test homes).

## 3. V2 · MapMaD v3 (budget 5 GPU-h; stop > 7.5)
- One fine-tune from v2.1b `ema_9`, the v2.1b recipe (`mapmad_v3.yaml` ← `mapmad_v21b.yaml` ← `mapmad_v21.yaml`): 10 ×
  200,000 samples, seed 1, lr groups pretrained 1e-5 / map encoders 1e-4, warm-up epoch 1 + cosine, batch 256, modes
  45/20/20/15, GoStanford 50/50. **`ema_9` fixed in advance; no picking.** `naz_mapmad` restarted first; one GPU
  (GPU 1 preferred), ~2 GB headroom.
- **Data:** map-DAgger = Phase 11 (524,409) ∪ round 2 maps shown, in search_map; hidden-DAgger = Phase 13 (search
  phase, 511,849) ∪ round 2 maps hidden (**search-phase states + every recovery state of the drive**), in explore.
  **Equal-count rule** (Phase 13) over the two unions: an explore draw takes a hidden-DAgger sample with probability
  q = p(search_map) · s / p(explore), s = the map-DAgger share of the search_map pool; repeat factor reported. Within a
  union every sample is equally likely (round 1 and round 2 pooled).
- **Reverse-capable waypoints:** labels behind the robot are kept as they are (nothing clips them in training).
- **B1 past moves** (`train/vint_train/mapmad/past_moves.py`): a new token after the two map windows = the last 8
  executed moves (every 5th control step; each (dx, dy) / 0.25 m in its start frame and dθ in rad; steps before 0 =
  standing), embedded by Linear 24→256, GELU, Linear 256→256 (last layer zero-initialised); own mask bit; an
  auxiliary head (256→256→24) predicts the 8 moves from the pooled context, MSE weight **λ_past = 0.1**. In training the
  token is hidden with probability **0.3** (Habitat samples); **GoStanford: token hidden, no past loss** (not the
  Habitat odometry; NoMaD's own data). Maps hidden + token hidden (or absent) = NoMaD's own computation. New
  parameters fresh, in the map-encoder lr group (1e-4); every other weight from v2.1b (positional table grown by one
  row, first rows equal).
- **Test time:** the runner passes the robot's pose history; v3 arms always show the token (W4 and A-mark steps
  included in the moves). v2.1b / NoMaD arms get no token.

## 4. Offline checks (after V2)
- **G10 row 1** (switch off byte-identical) and **row 2** (v3 architecture + official `nomad.pth`, maps hidden AND the
  past token hidden with non-zero past moves and live token weights = official NoMaD, allclose 1e-6; new keys only
  `map_encoder.{near,far,past,past_head}`; first positional rows equal): `checks_g10.py --config
  config/mapmad_v3.yaml --weights nomad.pth`, CPU. **Must pass (gate row 5):** a failure = a code bug → two honest
  fixes, then a safety stop; fallback after two fixes: v3 without B1 (round-2 data + W1 only), reported.
- **Rows 3–4** (Phase 10 rules v2, on `ema_9`, past token absent): maps − zeroed maps CI < 0 on held-out search drives;
  GoStanford ≤ official + 0.10. Failing → recorded, the phase continues.
- **Recovery-label loss** (`v3_offline.py recovery`): the offline action loss (fixed noise per state) on 1,000 seeded
  round-2 recovery states per mode (maps shown in arm "maps", hidden in arm "hidden"; no past token), v3 − v2.1b with a
  drive-bootstrap 95 % CI. **Falls** = mean lower and CI upper bound < 0. Does not fall → **fallback "W1-turn"**:
  recovery labels relabelled turn-only (no backing up; §2 motion without the 10 backward steps), v3 retrained once
  (same recipe, `v3t`), the reverse controller **off** for every v3 arm (W4 stays the reversing rule).
- **Past-move error:** the auxiliary head's MSE per epoch on the in-loop test sets (report).

## 5. Arms and the pipeline P\*
- **P\*** = the 14b-2 "both" cell, applied identically to every network arm: FT detector (driving layer + stop
  marks), **A-mark** (14b-2 Decision 8: trigger mark ≥ 0.45 seen ≥ 2×, in view, ≤ 3 m; goal = nearest free point
  ≤ 0.9 m; confirm if the rule value ≥ 0.45 on arrival; remembered 0.5 m; give up after 120 steps; W4 has priority),
  **W4** (14a), path 0 (sample 0, waypoint 1); **P\*'s stop = today's rule at τ 0.45** (cells ≤ 1.0 m ahead, ±30°,
  ≥ 2 views; scored offline from the log, first step ≥ 1 with the rule value ≥ 0.45).
- **Reverse controller** (v3's controller, arms M3_P, M3h_P, M3_Pnoam only): when the chosen waypoint lies behind the
  robot (|bearing| > 150°) the robot drives backward at ≤ 0.1 m/s toward it (turn rate = NoMaD's arctan(dy/dx)/dt,
  which points the rear at it); otherwise NoMaD's controller unchanged. Collision check as forward motion (the same
  navmesh step). With no behind-waypoints it equals v2.1b's controller (test).
- **Arms:** M21b_w4_d_am (v2.1b, maps shown, P\*), M21bh_w4_d_am (v2.1b, hidden, P\*), **M3_P** (v3, shown, P\* +
  reverse), **M3h_P** (v3, hidden, P\* + reverse), M3_Pnoam (v3, shown, P\* without A-mark; practice only),
  N_w4_d_am (official NoMaD explore, P\*), **F_d** (map + frontier planner, FT detector; its own driving, P\*'s stop).
  Checkpoint sha256 checked before every run.

## 6. V3 practice check and V4 exam B
- **V3 practice** (report only; budget 6 GPU-h, stop > 9): M3_P, M3h_P, M3_Pnoam on the 219 practice episodes, seed 0;
  v2.1b + P\* = the 14b-2 `runs/detam` runs, **reused** (valid if the 14b-2 runs are reproduced byte-identically on
  their GPU with this phase's code, gate row 8). **Oscillation rule:** if forward / back oscillation (a reverse step
  followed by a forward step within 2 steps and a reverse step again within 4) exceeds **20 % of the reverse steps**
  of M3_P or M3h_P → fallback: the reverse controller off for the exam arms (reported as "W1-turn at test time").
- **V4 exam B** (budget 18 GPU-h): network arms seed indices 0, 1, 2 (Phase 13 noise salt `mapmad-p13b`, so M21b's
  Phase 13 runs pair by episode and seed), F_d seed 0 (deterministic). **Run order** (one call per arm group and seed):
  M3_P + M3h_P s0, M21b_w4_d_am s0, M3_P + M3h_P s1, M21b s1, M3 s2, M21b s2 (= rows 1–3 first), then N s0, F_d, N s1,
  N s2, M21bh s0, s1, s2. **Budget fallback:** at 1.5 × 18 = 27 GPU-h the running call stops; it and every later call
  are reported as **not run** (their finished runs listed only, never compared); no stop.
- Old-pipeline reference: the Phase 13 exam-B logs of M21b (`p13_claim/runs/examb`, base YOLOE, no switch; read only).

## 7. Statistics and readings (exam B)
- Per episode, a metric is first averaged over its seed indices (F: its one run); paired differences over the episodes
  both arms ran (a crash record drops the episode); **home-clustered bootstrap**, 10,000 draws, seed **14301**; 95 %
  CIs; one-sided p = share of draws with mean difference ≤ 0; **Holm over rows 1–3**. Row 3 is a cross-run comparison
  on the same episodes and seed indices.
- **Primary metric: correct stops** = rule-stop success with P\*'s stop (a stop counts if the robot is within 0.2 m
  walking distance of a usable view point at the stop step). Reported besides: oracle success, SPL (oracle and rule),
  STOP precision (pooled, home CI), wrong-object share, collision steps, stuck runs, W4 events, A-mark detours,
  reverse steps, zigzag and flips per 100 turning steps; per distance bin, category, val / test homes.
- **G14b3 rows 1–3 (results):**
  1. map effect with the finished pipeline: M3_P − M3h_P > 0;
  2. retraining effect: M3_P − M21b_w4_d_am > 0;
  3. pipeline effect: M21b_w4_d_am − M21b (Phase 13, τ 0.45, base YOLOE) > 0.
  - **Pre-set reading:** raw one-sided p < 0.05 and Holm p < 0.05 → "passes: claim as stated"; else "no evidence";
    no extra runs either way.
- Practice comparisons (report only): paired by episode, home-clustered 95 % CIs, halves A / B / all.

## 8. Gate G14b3
| # | Check | Pass |
|---|---|---|
| 1 | Map effect M3 − M3h, exam B, correct stops | reported with p (raw + Holm) |
| 2 | Retraining effect M3 − M21b, exam B, correct stops | reported with p |
| 3 | Pipeline effect M21b P\* − M21b Phase 13, exam B, correct stops | reported with p |
| 4 | Oracle, SPL, precision, wrong-object, collisions, stuck, per bin / category; practice check; G10 rows 3–4; recovery share; past-move error | reported |
| 5 | **G10 rows 1–2** (switch off byte-identical; maps hidden + token hidden = official) | must pass |
| 6 | **No contamination** (§1) + exam B rendered only for its frozen set + exam rules sha256 unchanged since before the first exam run | must pass |
| 7 | **Fairness:** every planned arm × episode × seed of the practice check and of every exam call not cut by the budget fallback on the frozen fingerprints, checkpoint sha256 and switches matched, none missing (cut calls listed as not run) | must pass |
| 8 | **Determinism:** one p13d episode of M3_P twice on the same GPU identical (logs, dets, evidence); one Phase 13 exam-B M21b run per GPU reproduced byte-identically (log + dets) with every new switch off, on the GPU that drove it; one 14b-2 "both"-cell episode per GPU reproduced (the reuse) | must pass |

Rows 5–8 failing after two honest fixes → the only gate stop. Otherwise self-close: `/commit`, ff-merge into
`mapmad`, `STATUS.md` → `CLOSED <hash>`; never push.

## 9. Automatic stops (`STOP_reason.txt`) and watchdog
Checkpoint sha256 or set fingerprint mismatch (pre-flight); duplicate ids; crash rate > 1 % of a call; a same-GPU
determinism mismatch; a reproduction mismatch; contamination; exam rules changed; G10 rows 1–2 failing; runners
exiting without progress; a recogniser restarted > 3 times; a step over 1.5 × its GPU-h budget (except the pre-set V1
and V4 budget fallbacks); fewer than 50 % rollouts per mode. Progress watchdog: no finished run and no log line for
30 min → the step's runners are killed and the step restarts once; a second hang → stop. A pool worker that dies
raises (14b-1 fix).

## 10. Pre-flight
```json preflight
{
 "checkpoints": {
  "p13_claim/runs/v21b_s1_2026_10_30_17_02_10/ema_9.pth": "9ab0dbfb590b1ec481fb805e7b34f3b93cb858057bec377b1cdf6f4ddeec2c68",
  "weights/official/nomad.pth": "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333",
  "p14b2_recog/train/ft/weights/last.pt": "93c66d287b1ae11c351364045c640eaf9d92c5432f15d825ccfc5b6850e04fd3"
 },
 "sets": {
  "p14b3_v3/rollouts_r2/set/episodes.json": "a6f172ac455e9330b1475ab49ffed6e767a161dfc7a9713ec623ee0e5db8c5a4",
  "p11_dagger/rollouts/set/episodes.json": "8e3428c74454f7a6f0a6021694c0e0eb8565c3c646d3ec5780b83930e187aa40",
  "p13_claim/examb_set/episodes.json": "98a8d920e198903e41e1b6b82c16074049ab1f5bb098dd02d9c441024b74cf1a",
  "p13_claim/dev_set/episodes.json": "5745b34074faa0bb654fc8cb71938af533ac1d8382361531afdfc721df4ab085",
  "p14a_finish/sets/p14d/episodes.json": "68991c7e020b31ded7338bd5c67e39cf808ad3145013d604503694de728bc7df"
 }
}
```
Videos (report): M3_P on the same 8 practice episodes as 14a / 14b-2 (`p14a_finish/videos_marks/picks.json`) and on 4
exam-B episodes picked by `numpy default_rng(12501).choice(291, 4)` (`videos/picks_examb.json`, frozen with these rules).
