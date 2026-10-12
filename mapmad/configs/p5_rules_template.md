# Phase 5 exam rules (written before the exam-set draw and before any exam run; sha256 in p5_rules.sha256)

Written by `mapmad/scripts/p5_freeze.py rules` on $WRITTEN, before the exam set was drawn (version 3, see
`p5_rules_log.md`) and before the first exam run, from `mapmad/configs/p5_rules_template.md`. The values that depend
on the drawn set (fingerprint, counts, video and determinism episodes, GPU estimate) are written right after the
freeze into `p5_rules_set.md` (+ sha256) by the procedures fixed here. Source: Phase 5 confirmation block (2026-10-11) items 4-14, plan v1.9
"Phase 5", Naz's S-dev answers (2026-10-11: 1a 2a 3a, decisions 1-5 OK), dev rules `p5_dev_rules.md` (sha256
46e38280911fa70c65d1274112027fc812773abc2c8b040388173d6cd4674fde) and their outcome.

Code: commit $COMMIT (branch mapmad-phase-5-exam) **plus uncommitted changes** -- Naz asked not to commit before
the unattended run (2026-10-11); the exact code is saved in `code_snapshot/` (tracked.patch sha256 $PATCH_SHA,
untracked.tar sha256 $UNTRACKED_SHA; `git apply` the patch on $COMMIT and unpack the tar to rebuild it).
Config `mapmad/configs/p5_exam.yaml` sha256 $CONFIG_SHA.

## 1. Exam set (items 6, 8)
- `/outputs/mapmad/p5_exam/exam_set/episodes.json` (fingerprint, counts: `p5_rules_set.md`), from ObjectNav HM3D v2
  val (all 1,000 official episodes examined); rules `mapmad_sim/exam_set.py`; tied to the
  LIMO floor maps (navmesh sha256 per home in the file; the runner refuses a mismatch). Build report:
  `exam_set/build_report.md`; S-freeze checks: `exam_set/sfreeze.md`.
- **Episode ids:** `p5e-<home5>-<category>-<ObjectNav id>` (ObjectNav val reuses ids within a home, once per
  category); the freeze and the runner refuse a set with repeated ids. Noise seeds, goal-photo files, the video draw
  and the determinism draw all use these ids.
- **Draw (block item 6, unchanged):** `exam_set.draw`, seed 5202, 300 asked, at most 10 per home, aim 100 per bin
  (3-6 / 6-10 / 10-15 m); a short bin passes its shortfall to its neighbour(s); what is still missing is reported.
- **S-freeze history (rules log `p5_rules_log.md`):** (1) the first exam set (2026-10-11, fingerprint
  3659b361...a7f9) had 181 episodes and failed the S-freeze condition ">= 250 episodes, every bin >= 60". The real
  cause was a bug: episode ids were `p5e-<home5>-<ObjectNav id>`, so the 446 eligible episodes had only 181 distinct
  ids and the draw (which skips used ids) took one episode per id. (2) Based on a wrong diagnosis of that failure
  (draw quota moves vs the home cap), a scarcest-bin-first draw was adopted on 2026-10-12 morning; with it, the set
  again had repeated ids (300 episodes, 164 distinct) and the exam was stopped after 21 runs. That draw is withdrawn
  (Naz 2026-10-12, answer "fix, 2b"). With unique ids, the block's draw gives 278 episodes, bins 100 / 123 / 55.
  Both superseded sets, their rules and S-freeze records and the 21 runs are kept (`exam_set/superseded_181/`,
  `exam_set/superseded_dupids/`, `runs/superseded_dupids/`); no result from them is used.
- **S-freeze conditions (rules version 3):** >= 250 episodes from >= 30 homes, **every bin >= 50** (was >= 60; Naz
  2026-10-12); this rules file + `p5_rules_set.md` with matching sha256, this file older than the exam set, no exam
  log yet; checkpoint sha256s match; goal-photo target share median in [5%, 40%]; GPUs free of other users' jobs;
  GPU estimate <= 16 h. Any failure stops before the exam runs (`STOP_reason.txt`).
- **Goal photo, correction of block item 8 (Naz, S-dev answer 1a):** the block's "view point nearest the target"
  gave close-ups (dev: mean 60% target pixels, taken ~0.1 m from the box). Corrected to the plan's "~1 m" = Phase 1's
  rule: the usable view point whose horizontal distance to the box is closest to 1.0 m, facing the box centre,
  >= 5% target pixels, otherwise the next one in that order; none -> the episode is dropped (`no_goal_photo`).
  Config `set_rules.goal_photo_box_m: 1.0`.
- Categories (S-dev answer 2a): tv_monitor and plant are under-represented through `no_goal_photo` (LIMO's camera
  is 0.18 m high); accepted and reported (counts in the build report).
- Disclosure: minival homes 00800/00802/00803/00808 are exam homes; 00800 was viewed during development (Phase 0
  renders, horizon check, first Phase 1 debugging); nothing was tuned on it or any exam home.

## 2. Arms (item 9) -- same episodes, paired
| arm | policy | inputs |
|---|---|---|
| A | official NoMaD (`nomad.pth` 70f79b82...a333) | camera, goal masked (Phase 1 setup exactly) |
| B | official NoMaD | camera + goal photo |
| C | MapMaD `ema_19` (8093c6db...d9e2) | camera + map with oracle heat, photo masked -- headline |
| D | MapMaD `ema_19` | camera, map masked + photo masked (control) |
| E | map planner | own online whole-storey obstacle grid, A* to the target box (`planner_policy.py`) |
| F | MapMaD `ema_19` | camera + goal photo, map masked (reported, not gated) |
- MapMaD arms (item 4): context frames every 5th control step (t-15, t-10, t-5, t), missing past frames repeat the
  first picture; de-normalisation 0.25 m; 8 samples, sample 0; **waypoint index $WP** (dev rule, `p5_dev_rules.md`
  1: SR 0.465 at index 1 vs 0.485 at index 2, difference 2.0 pp < 3 pp -> index 1).
- Official arms: context every step, 0.05 m, waypoint index 2, no waypoint before 4 pictures (deployment and
  Phase 1). **Deviation from block item 4's wording** (confirmed by Naz at S-dev): the block says the first-picture
  repeat is "as in deployment / Phase 1"; deployment and Phase 1 in fact wait for 4 pictures. The repeat applies to
  the MapMaD arms only; the official arms keep Phase 1's waiting exactly.
- All arms: v 0-0.2 m/s, |w| <= 0.4 rad/s, 4 Hz, never reversing; LIMO floor map and camera as Phases 1-2;
  pose for the map (and the planner) = the true simulator pose (limitation; confirmed at S-dev).
- Planner (confirmed at S-dev): inflation 0.195 m, unknown = free, goal = within inflation + 1.5 cells of the
  target box (A* capped at 100,000 expanded cells; no path -> the reached cell nearest the box), replan every step,
  steer at the path point 0.5 m ahead, a point behind the robot -> turn on the spot.
- Checkpoint sha256 checked when every policy is built and written into every log (`weights_sha256`).

## 3. Seeds, limits and compute (items 7, 16)
- 3 seed indices (0, 1, 2) per episode; noise seed = first 8 hex digits of sha256("mapmad-p5/<episode_id>/<index>"),
  the same for every arm; starts fixed. The planner is deterministic: its index-0 run counts for all 3.
- Timeout 1,000 steps; success@500 also reported.
- 6 simulator/runner pairs, 3 per GPU (S-dev answer 3a); results do not depend on the number of pairs.
  Estimated wall time: `p5_rules_set.md` (from the dev timings).

## 4. Metrics (item 10)
- Primary success: walking distance (LIMO floor map) <= 0.2 m to the nearest usable ObjectNav view point of the
  target instance (snapped <= 0.2 m, real floor), oracle stop (the episode ends at the first step where it holds).
- SPL to that view point (shortest = its walking distance at the start).
- Secondary, reached at any step: target point (Phase 1 point) <= 1.0 m; any side = nearest view point <= 1.0 m;
  success@500.
- Collisions per run (move > 1 cm short), stuck streaks (>= 100 consecutive collision steps), final distance.
- Per home, per distance bin (3-6 / 6-10 / 10-15 m), per category.

## 5. Statistics (item 11)
- Per episode: mean over its 3 seed runs. Paired differences a - b per episode.
- Primary test: home-clustered bootstrap, resample the homes with replacement, 10,000 resamples, bootstrap seed 0;
  statistic = mean over the episodes of the drawn homes; one-sided p = share of resamples with mean difference <= 0.
  CIs: 95% percentile, two-sided. Episode-level bootstrap (seed 0) also reported.
- Only C - B gets the primary p-value; G5 rows 2 and 3 must both pass -> no multiplicity correction.
- Reported differences: C - B (primary), C - D, C - E, C - F, F - B, D - A.

## 6. Gate G5 (item 12)
1. Fairness: every arm ran every episode x seed index (planner: index 0) on the frozen set; every log's
   `weights_sha256` equals the frozen hash; no run missing after rule 8.
2. Headline: C - B primary success > 0 with home-clustered one-sided p < 0.05.
3. Control: C - D > 0 with the home-clustered 95% CI lower bound > 0.
4. Planner: C vs E reported (success, SPL, collisions, stuck streaks) per distance bin; where each wins discussed.
5. Videos legible, captioned, reproducible from the logs.
6. Determinism: one C episode, drawn seeded from the exam set (numpy default_rng(6006), uniform over the sorted
   episode ids; the id is in `p5_rules_set.md`), run again at the end with seed index 0 -> byte-identical log.
- If row 2 fails: report honestly; analyse by distance bin and failure type; decide next steps with Naz and the
  advisors. No re-run, no re-judging.

## 7. Failure types (labelled from the logs, item 12; first match in this order; confirmed at S-dev)
stuck (a stuck streak) -> wrong_instance (turning centre within 1.0 m horizontally of another instance's box of the
category, same storey: navmesh height 2.0 m below to 0.5 m above the box bottom) -> circling (> 50% of the 0.25 m
cell entries of the path revisit a cell, >= 10 entries) -> timeout_moving (the rest).
- Note for the thesis (Naz, S-dev): a common MapMaD stuck pattern is "pushes into a wall toward heat behind it"
  (example: dev episode p5d-00081-1004, arm C). It is counted as `stuck`; not a separate type.

## 8. Bugs and crashes (items 13, 13b)
- A bug = the code does something different from the plan / this file, shown by a test that fails before the fix and
  passes after; low success alone is never a bug. Then: stop, Naz approves, the fix and its reason go into
  `p5_rules_log.md`; every run of every affected arm is rerun on the full set (shared code: all arms); all runs kept.
- Infrastructure crash (simulator/bridge exception, or the simulator server process dying -- it is restarted
  automatically): the run is retried up to 2 times with the same seed; still failing -> the episode is removed
  from all arms (pairing kept) and listed; > 2% of episodes removed -> treated as a bug.
- Unattended run (Naz offline, 2026-10-11/12): any suspected bug (a runner stopping on a non-infrastructure error,
  > 2% removed, fairness or determinism failing) stops the driver with `STOP_reason.txt`; nothing is fixed or rerun
  until Naz is back.

## 9. Videos (item 14, chosen now)
- 8 episodes drawn with numpy default_rng(1414) from the frozen set, stratified 3 / 3 / 2 over the bins 3-6 / 6-10 /
  10-15 m (uniform within a bin, sorted ids); the ids are in `p5_rules_set.md`.
- Plus 4 picked after the runs to show typical cases. Unattended, they are picked by rule from the seed-0 runs in a
  default_rng(1515) order of the sorted episode ids, skipping the 8 above: (1) C stuck, (2) C circling (else C
  timeout while moving), (3) C success while B fails, (4) C fails while E succeeds; labelled "picked by rule";
  Naz may replace them by hand-picked ones.
- All arms side by side (camera + 64 x 64 map with heat + floor plan with path and heat), success stamps, seed index 0,
  rendered from the logs (`mapmad/scripts/p5_videos.py`).

## 10. Ablations (item 17)
None in the unattended run (Naz, 2026-10-11). Registered for later: perturbed heat at test (heat target shifted by
0.5 m and by 1.0 m in a seeded random direction per episode, same frozen episodes, seed index 0, arm C settings;
report only). B / D `ema_19` on arm C, heat-only map, FOV: not registered yet.
