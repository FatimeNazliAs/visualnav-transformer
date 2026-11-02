# Phase 14a rules (v1, 2026-11-02) — Goal 2, Step 1: cheap checks so the robot can finish the search

Frozen before any judged run (sha256 in `p14a_rules.sha256`; copy in `mapmad/configs/p14a_rules.md`). Kickoff:
`~/.claude/plans/phase14_plan_goal2.md` v1.0. A rule may be replaced only before its result is judged (new version +
sha256). Config: `mapmad/configs/p14a_finish.yaml` (extends `p13_claim.yaml`: Phase 13 harness numbers unchanged).

## 1. Scope
- No training, no exam: HM3D exam (`6fb61eb8…`) and exam B (`98a8d920…`) are never rendered or run in 14a.
- Arms: **M21b** (MapMaD v2.1b, maps shown), **M21bh** (v2.1b, maps hidden), **N** (official NoMaD explore) — the
  network arms, every switch; **F** (map + frontier planner) and **VF** (VLFM-style) — baseline runs only (14.0b,
  14.1, 14.2), their own driving. Checkpoints (sha256 before every run): v2.1b `9ab0dbfb…` (key `v21b`), official
  `70f79b82…`. Recogniser YOLOE-11-L-seg, threshold 0.01 (`p13_claim.yaml`), one sidecar per pair.
- Harness = Phase 13 exactly (1,000 steps, oracle + rule stop from one log, context every 5th step, 8 samples, waypoint
  index 1, 0.25 m per normalised step, 0.2 m/s, 0.4 rad/s, 4 Hz). Seed index 0 only (kickoff §3 item 12).
- Evidence logging (`run.log_evidence`): every run also writes `evidence/<arm>/<run>.npz` (mapmad_sim/stop_evidence.py:
  per frame pose, view cells, recogniser masks -> cells with raw score, map value, pixels, box; true target boxes ->
  cells; at the end first-seen maps + bump events). It only records: switch-off runs are byte-identical to Phase 13.

## 2. Practice set (14.0) — frozen
- p13d = Phase 13's MP3D dev set, `5745b34074faa0bb654fc8cb71938af533ac1d8382361531afdfc721df4ab085` (120 episodes, 44
  train homes), noise salt `mapmad-p13d` (Phase 13's seeds: M21b, M21bh, F, VF re-runs reproduce Phase 13's dev logs).
- p14d = new episodes from the **12 MP3D train homes without a p13d episode**, Phase 13's dev builder unchanged (start
  snap <= 0.2 m; a reachable instance on the start storey; nothing visible in 8 headings at pin025; walking 3-15 m;
  Phase 8 start filter; bins 3-6 / 6-10 / 10-15 m; round-robin over categories; per-home cap 12), 1,500 seeded official
  train episodes per home as candidates (candidate seed 14001, draw seed 14002; Phase 13 used 30), ids
  `p14d-<home>-<category>-<id>`, noise salt `mapmad-p14d`:
  `68991c7e020b31ded7338bd5c67e39cf808ad3145013d604503694de728bc7df`, **99 episodes** in 9 homes (3 homes have no
  passing episode; the cap and the bin fill limit the rest; bins 12 / 27 / 60). **Kickoff §3 fallback:** fewer than 180
  pass -> all are used; 99 < 100 -> practice = p13d + what exists, the A / B split is reported as small.
- Practice = p13d ∪ p14d = **219 episodes in 53 homes**, fingerprint
  `8d7b623e739cfc6b457f7a1382a8c86bef30040f866c90d079869379bbb63da6` (sha256 over the parts' fingerprints,
  `sets/practice.json`). Home-level halves: homes in a seeded order (seed 14003), each to the half with fewer episodes;
  first order with every category in both halves: **A = 108 episodes (30 homes), B = 111 (23 homes)**.

## 3. Stop rules (14.1), scored offline from the evidence (mapmad_sim/stop_rules.py)
A look = a frame in which a cell is in view (view cells ∪ cells under a detection). Its detection = the recogniser
detection covering the cell with the largest map value (ties: more mask pixels, then the first); none = a miss. Stop
step = the first step t >= 1 whose frame qualifies; the stop cell = the qualifying cell with the largest rule value
(Rule 1: hit share, then looks; ties: row-major order).
- **Today:** value = mean map value over looks (miss = 0; TargetFusion); stop if a cell <= 1.0 m ahead (±30° of the
  heading) seen from >= 2 looks has value >= τ = 0.45. Parity check: the rebuilt value = the logged `rule_value` on
  every step (|diff| <= 1e-5).
- **Rule 1 (voting):** a look is a hit if the largest **raw YOLOE score** covering the cell >= s (the map values start
  at 0.4, so s is read on the raw score); stop if a cell <= 1.0 m ahead (±30°) has looks >= n and hits / looks >= h.
  Grid s ∈ {0.1, 0.25, 0.4}, n ∈ {2, 3, 4, 5}, h ∈ {0.3, 0.4, 0.5, 0.6, 0.7} (60 points).
- **Rule 2 (weighted voting + close-and-confident):** a hit look weighs w = value × size × near (value = the map value
  0.4-1.0; size = min(mask pixels / 1,000, 1); near = 1 at <= 1.5 m falling linearly to 0.25 at 4 m, distance robot ->
  cell centre in that frame); a miss weighs near (size 1: the cell was in view) with value 0; cell value = Σ(w·value) /
  Σw; stop if the robot is <= 1.0 m from the cell (any direction), the value >= v and >= g **hit** looks had w >= 0.3.
  Grid v ∈ {0.30, 0.35, …, 0.70}, g ∈ {1, 2, 3} (27 points).
- **Rejected spots:** after a vetoed candidate (14.2) cells within 0.5 m of it are not used again (all variants
  support it; without a check no candidate is vetoed).
- **Metrics per arm** (practice halves A, B, all): rule-stop success (stop pose inside the 0.2 m zone), SPL (path up to
  the stop), capture = rule / oracle success, STOP precision = correct stops / stops, wrong-object share (stop cell >
  1.0 m from every usable instance's box), near-miss share (failed stop within 0.2 + 1.0 m walking), per category.

## 4. Pick (§4b of the kickoff, automatic)
- Per rule (Rule 1, Rule 2) separately: the grid point with the highest **mean rule-stop success over the 5 arms**
  (arm mean = mean over its episodes) on **half A**; ties -> the higher threshold (Rule 1: h, then s; Rule 2: v), then
  fewer looks (n / g). Its scores on half B (honest check) and on all are reported. **Main stop = Rule 2's pick.**
  Today's rule and Rule 1's pick are comparison rows. Per-arm best τ of today's rule (0.30-0.95, half A, ties -> the
  higher τ) and per-arm best Rule 2 point: report only.

## 5. Qwen3-VL check (14.2) on top of Rule 2's pick
- When Rule 2 fires on a cell (not within 0.5 m of a rejected spot): the robot asks at the first frame from that step
  on whose view holds the cell (within 20 steps; none -> it stops at the firing step). Pictures rendered at the logged
  pose with the arm's camera: a crop around the marked box (box × 1.5, >= 96 px) and the full frame with the box drawn;
  box = the frame's YOLOE box covering the cell with the largest map value, else the cell's projection (0.05-1.0 m
  above the floor, padded 30 %). Question: "Is there a {word} inside the marked box? Answer yes, no, or unsure."
  ({word} = the YOLOE prompt word). Qwen3-VL-8B-Instruct (Phase 9 weights, revision `0c351dd0…`), greedy, <= 8 tokens.
- Answer (first word): yes -> stop; no -> veto, remember the spot, driving goes on as logged; anything else = unsure
  -> asked again at the next frame holding the cell (within 20 steps of the last ask), <= 3 asks; no "no" within 3
  asks (or no frame within 20 steps) -> stop at the last ask's step.
- **Keep** if on half B the mean rule-stop success over the 5 arms rises by >= +0.02 **and** the pooled STOP precision
  (correct stops / stops over the 5 arms) does not fall; else dropped. Latency per call and calls per run reported.

## 6. Diagnosis (14.3), report only (mapmad_sim/diagnosis.py)
- States: search-phase steps (before the first true sight) of M21b and M21bh (+ F as the check) in the practice
  baseline runs; plus every labelled Phase 11 map-DAgger and Phase 13 hidden-DAgger rollout (HM3D train homes).
- Teacher: `explorer.teacher_step` on the robot's own map (first-seen <= t, cells bumped before t), state carried,
  frozen blacklist (the DAgger labeller's rule), nothing seen; DAgger rollouts: the stored labels (goal, path length,
  look-ahead waypoints).
- 20-step windows (last one kept if >= 10 steps). Progress = same goal (within 0.3 m) and path length fell by > 0.1 m,
  or another goal and the straight distance to the old goal fell by >= 0.1 m. Teacher direction = its path point 1.0
  m along (DAgger: the last look-ahead waypoint). Robot's cluster = the frontier cluster whose distance fell most
  (>= 0.1 m); the teacher's = the one nearest its goal. Classes, first match: **bad driving** if >= 10 collision steps
  (scaled by window length / 20); other_teacher (no teacher goal); other_still (net move < 0.1 m); **agree** if the
  robot's net move is within 30° of the teacher's direction and progress; **bad driving** if within 30° without
  progress; **wrong place** if > 60° off and the robot's cluster exists and differs from the teacher's; **other**.
- Report: shares of windows and of search steps per class, per arm / rollout set.

## 7. Driving switches (14.4), outside the network (closed_loop/drive_switches.py)
- **A1 "stick with your choice"** (BID ρ = 0.5; IMLE periodic reset C = 10): of the 8 samples, the one with the
  smallest Σ_k ρ^k |sample_k − previous_k| (k = 0..7, Euclidean per waypoint, metres), the previous chosen plan moved
  into the current robot frame by the executed command (unicycle, (v, w) for 0.25 s); the first step, every step with
  step % 10 == 0, and the first step after a W4 manoeuvre take sample 0. All 8 samples are still drawn (same noise).
- **W4 get-unstuck rule:** stuck = >= 16 collision steps in the last 20 steps, or < 0.10 m net movement in the last 40
  steps; then reverse 0.25 m at 0.1 m/s (10 steps; reversing is allowed only here: `step(..., reverse=True)`), turn in
  place towards the most open direction of the near map window (obstacle layer; rays every 10° in [−180°, 180°) from
  0.3 m in 0.05 m steps until an obstacle cell or the window edge; longest ray; ties: smaller |angle|, then left) at
  <= 0.4 rad/s, then hand back to the policy (histories cleared). <= 3 manoeuvres started per 100 steps. The motion
  model supports reversing (no "W4-turn" fallback needed). Counted and logged.
- Runs: M21b, M21bh, N × {A1, W4, A1 + W4} on all 219 practice episodes, seed 0 (arms `<arm>_a1`, `_w4`, `_a1w4`).

## 8. Driving reading (14.4), per switch (mean over M21b, M21bh, N; paired by episode vs the baseline runs)
- Metrics: oracle success, SPL, rule-stop success (today τ 0.45 and the main stop), left/right flips per 100 turning
  steps (|heading change| > 0.5°), zigzag °/step, collision steps per run, stuck runs (a streak of >= 100 consecutive
  collision steps), steps to the first true sight; map effect M21b − M21bh per switch.
- A switch **helps** if oracle success is not lower by more than 0.03 **and** its target metric improves: A1 flips per
  100 turning steps −25 % relative or more; W4 stuck runs −25 % relative or more **and** collision steps per run lower.
  A1 + W4 is read with W4's target. No switch is chosen for an exam here.

## 9. Mark check (14.5), report only
- Code check: does `p13_videos.py` draw the target layer (answer + line numbers in `videos_marks/code_check.json`).
- 8 videos (M21b practice runs, seed 14501): 4 runs with a YOLOE box on a true target (IoU >= 0.3), 4 runs whose rule
  stop (τ 0.45) is on a wrong object; Phase 13's renderer with every target-layer cell > 0 in a bright strength colour
  scale + legend, the step's max mark strength ahead printed, YOLOE boxes, rule-ready cells; Phase 13's box + pose check.
- Counts over all practice runs of M21b and M21bh (mapmad_sim/mark_check.py): per frame with a YOLOE box on a true
  target, the strength = largest layer value on the object's cells within that frame and the next 20 steps; marked =
  >= 0.45; distribution; the same for YOLOE boxes on nothing true (IoU < 0.3 with every true box).

## 10. Checks (gate rows 6-7)
- Fairness: every planned arm × episode on the frozen fingerprints (p13d, p14d), checkpoint sha256 and switches in
  every log header, evidence present, none missing / crashed.
- Determinism: (a) all p13d baseline runs of M21b, M21bh, F, VF byte-identical with Phase 13's dev runs (logs + dets;
  includes "2 Phase 13 dev episodes reproduced"; compared for runs on the same GPU as in Phase 13 (>= 2 required),
  other-GPU runs reported only: YOLOE float noise differs between the cards); (b) one p13d episode × M21b, M21b_a1w4, N_a1w4 twice on one GPU:
  logs, dets and evidence identical. A mismatch = automatic stop.

## 11. Statistics
Paired by episode; home-clustered bootstrap, 10,000 draws, seed 14401, 95 % CIs; one seed (stated).

## 12. Fallbacks (kickoff §3, pre-set; each taken one is logged and listed under "Decisions to confirm")
- Taken in Part 1: p14d has 99 episodes (< 180, < 100) -> practice = p13d + p14d (219), A / B reported as small.
- Taken in Part 1: depth replay is not byte-faithful by construction (Phase 13 dets files hold boxes, no masks, so the
  mask sizes and mask cells of Rule 2 cannot be rebuilt) -> every baseline practice run is re-run with full evidence
  logging (incl. Phase 13's p13d runs of M21b, M21bh, F, VF, checked byte-identical).
- Pre-set: Qwen3-VL cannot be served / a worker fails after the restarts -> 14.2 "not done", reported. Only one GPU
  usable -> 3 pairs on it, budget by GPU share. Renderer fails -> 14.5 counts only.

## 13. Budgets and automatic stops
Budgets (stop only > 1.5×): 14.0b <= 6 h GPU; 14.1 <= 3 h; 14.2 <= 4 h GPU; 14.3 <= 3 h CPU; 14.4 <= 16 h GPU; 14.5
<= 2 h. Automatic stops (`STOP_reason.txt`): checkpoint sha256 mismatch; duplicate ids; crash rate > 1 %; determinism
mismatch (§10); any run on an exam set; a budget > 1.5×.

## 14. Gate G14a
| # | Check | Pass |
|---|---|---|
| 1 | Stop bench reported | today / Rule 1 / Rule 2 per arm on halves A, B, all; Rule 2 pick by §4 |
| 2 | Qwen check | kept or dropped by §5, latency and calls per run (or "not done" + reason) |
| 3 | Diagnosis | class shares per arm (+ F check) |
| 4 | Driving tests | A1, W4, A1 + W4 vs baseline per arm, map effect per switch, §8 reading |
| 5 | Mark check | renderer answer, 8 videos, counts |
| 6 | Fairness | every arm × episode on the frozen practice fingerprint; sha256 matched; none missing; switch-off byte-identical to Phase 13 |
| 7 | Determinism | one episode twice on the same GPU identical; Phase 13 dev episodes reproduced byte-identical with Phase 14a code |
Rows 1-5 reported (no pass/fail on the numbers). Rows 6-7 must pass (else two honest fixes, then a safety stop).

```json preflight
{
 "checkpoints": {
  "p13_claim/runs/v21b_s1_2026_10_30_17_02_10/ema_9.pth": "9ab0dbfb590b1ec481fb805e7b34f3b93cb858057bec377b1cdf6f4ddeec2c68",
  "weights/official/nomad.pth": "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333"
 },
 "sets": {
  "p13_claim/dev_set/episodes.json": "5745b34074faa0bb654fc8cb71938af533ac1d8382361531afdfc721df4ab085",
  "p14a_finish/sets/p14d/episodes.json": "68991c7e020b31ded7338bd5c67e39cf808ad3145013d604503694de728bc7df"
 }
}
```
