# p8_rules_v2.md — Phase 8 · Search map + explorer teacher + drives (rules v2 = §1–9 of p8_rules.md + §10)

Written 2026-10-20 before the pilot, from the "Phase 8 confirmation" block (Naz, 2026-10-20). Items 1–7 are copied
verbatim from the block. §8 lists how the code implements them where the block leaves a detail open, all decided
during development on 4 dev homes (val-drive split: 00746, 00440, 00272, 00330), before this file was written.
§9 lists what a pilot fix attempt may change. Code: `mapmad/src/mapmad_sim/{explorer,datagen_v2,target_layer}.py`,
`train/vint_train/mapmad/map_v2.py`, numbers in `mapmad/configs/p8_datagen.yaml`.

## 1. Teacher rules (8.2)
- Uses only the robot's own online map (whole-storey, 10 cm), never the full house plan.
- Target not yet seen → nearest frontier by path length on the own map (obstacles inflated 0.195 m; path through explored free cells only). Frontier goal = nearest reachable cell of the frontier cluster.
- Commit to that frontier until robot ≤ 0.5 m from it or it is no longer a frontier. Tie (two frontiers within 0.5 m of equal path length) → the one more straight ahead.
- Blacklist: if the path length to the committed frontier shrinks by < 0.5 m within 40 steps (10 s), drop that frontier for the rest of the drive.
- Target seen (training only, simulator truth): an instance of the category has ≥ 200 px in view (320×240) at depth ≤ 4 m. "Seen" is sticky: stays in approach even if the object leaves view.
- Approach → shortest path on own map (unknown = free, replanned every step, as v1 planner arm) to that instance's nearest ObjectNav view point (snapped ≤ 0.2 m, Phase 2 rule). Several instances seen → the one with shortest path, re-evaluated when a new instance is seen.
- Arrived → spot-turn until the instance centre is within ±15° of heading, then stop.
- Search failure: no frontier left and target never seen → drive discarded, counted as failure.
- One function of the state (own map, pose, committed frontier, blacklist, seen instances) → next path point. Target ≤ 10 ms/step on CPU (not a gate; log median + p95). Entry point documented for Phase 11 (DAgger) and Phase 12 arm F.

## 2. Target layer for training (8.1–8.2)
- Detection in a frame exists for an instance only if it has ≥ 200 px at ≤ 4 m (same rule as "seen"). Its truth mask pixels are projected through depth onto 10 cm cells.
- Confidence per instance per frame ~ U(0.4, 1.0).
- Miss: per instance per frame, p = 0.2 → score 0.
- False mark: per frame, p = 0.05 → one 3×3-cell blob, value ~ U(0.3, 0.7), centred on a random visible obstacle cell.
- Fusion: for every cell in view (in FOV, ≤ 4 m, passes depth test): value = equal-weight mean of scores over all views that saw it; no detection = 0.
- Saved: clean (truth) and noisy fused layers, each as sparse updates (cell, frame, value); plus raw per-frame detections (instance, confidence, missed flag, false-mark position/value). One seeded noise realisation per drive (seed = f(drive id)).

## 3. Map v2 (8.1)
- Base: Phase 2/5 whole-storey builder, 10 cm, obstacles 0.05–0.55 m above real floor, max depth 4 m, invalid = unknown, first-seen indices; depth sensor at 0.25 m with the camera (pin025).
- Layers: obstacle · explored · frontier (explored free cell touching unknown; 8-connected clusters < 4 cells dropped) · target seen here (0–1).
- Near window 64×64 @ 10 cm (6.4 m); far window 64×64 @ 40 cm (25.6 m); far cell = 4×4 near cells: obstacle = any, explored = mean, frontier = any, target = max.
- Both robot-centred, rotating, forward = up. Teacher plans on the whole-storey map; windows are for the network only.
- Category name never on the map.
- v1 crop with new features off byte-identical (Phase 12 arm V1).

## 4. Drive types and starts
- Search drives: 145 labelled train homes; category uniform among the 6 ObjectNav categories with ≥ 1 reachable instance on the start storey; start on real floor; no instance of the category visible in any of 8 headings (pin025); 3–15 m walking distance to nearest view point.
- Exploration drives: all 800 train homes; random real-floor start on a random storey (Phase 2 rule); no target; until 1,000 steps or no frontier left (a 1,000-step end is complete, not discarded).
- Motion as Phase 2: pure pursuit (look-ahead 0.5 m, 0.3 m tight turns), 0.2 m/s, 0.4 rad/s, 4 Hz; detours ~30%, heading kicks ~15%, wall-recovery starts ~15%; never reverses.
- Discard: > 2,000 steps or ≥ 20 consecutive collision steps.
- Saved per drive: Phase 2 format + target history (item 2) + frontier choice per step + `phase` per frame (search / approach / explore) + category.

## 5. Drive mix and budget (8.3)
- 40 search drives × 145 homes (5,800) + 8 exploration drives × 800 homes (6,400).
- Same frozen val-drive split: mapmad/configs/p2_home_splits.json (10 Phase 1 homes + 30 unlabelled, seed 2610).
- ≤ 120 GB drives (+ loader cache), ≤ 4 h. If the pilot predicts either limit exceeded: first cut exploration drives down to 4/home, then cut both pro rata. Logged, no stop.
- Collision drives listed and excluded as in Phase 3.
- Data /mnt/shared_disk/nazli/mapmad_data/habitat_mapmad_v2/; splits + cache …/splits/habitat_mapmad_v2/; loader entry habitat_mapmad_v2 (metric_waypoint_spacing 0.05, waypoint_spacing 1).

## 6. Pilot + S-pilot
- Pilot: 20 labelled train homes × 10 drives (8 search + 2 exploration). Ids p8p-…; throwaway, not part of the full run.
- S-pilot sheet: teacher success + failures by cause (timeout / no frontier / blacklist exhaustion / collision), steps per drive, s/drive, GB/drive, predicted full-run GB + hours, any cut that will apply, spot turns by type, ms/teacher step (median, p95).
- 2 videos (1 search, 1 exploration): camera view · near + far window with all 4 layers (clean and noisy target) · teacher path on the house plan, phase label shown.
- If pilot success < 90%: ≤ 2 fix attempts, changing only the teacher parameters in item 1 (blacklist window, tie margin, facing angle, commit distance); show both versions at S-pilot.
- At S-pilot ask to commit the generator code (§0.8) before the full run.

## 7. Gate G8
| # | Check | Pass |
|---|---|---|
| 1 | Map v2 | pytest green: crop/rotation of both windows; far = summary of near; frontier on hand-made maps; false alarm fades; v1 crop with new features off byte-identical; online map = generator map on 2 dev drives |
| 2 | Pilot | S-pilot approved by Naz |
| 3 | Drives | teacher success on search drives ≥ 90% (over all attempted search drives, discarded included; success = success zone ≤ 0.2 m walking distance to nearest view point of any instance within 2,000 steps); collision drives < 2%; spot turns only of 4 allowed types (start, wall recovery, final face-the-object, re-target = starts ≤ 4 steps after a new frontier/target choice and ends at heading error < 30°), share per type reported; detours ≥ 25%; wall recoveries ≥ 10% |
| 4 | Format | unchanged NoMaD loader reads the data (maps ignored); one training step runs |
| 5 | Size | ≤ 120 GB (or reduced count, logged) |

## 8. Implementation notes (how the code reads items 1–7; fixed before the pilot)

**Map and frontier (item 3)**
1. "Touching unknown" = sharing an edge (4-neighbour) with a cell that is not explored. Cells outside the stored storey map count as known (the map reaches ≥ 1 m beyond the floor map, so its border is never a frontier).
2. Obstacle cells are always explored (Phase 2 builder), so a frontier cell is never an obstacle.

**Teacher (item 1)** — `explorer.teacher_step(own_map, pose, state, seen, spec, searching)`
3. Inflation: a cell is blocked if its centre is within 0.195 m + half a cell (0.245 m) of a known obstacle cell centre, as the v1 planner arm. Cells within 0.3 m of the robot are never blocked (its own position).
4. Blind zone: at pin025 the floor is first seen ~0.55 m ahead of the turning centre, so UNKNOWN cells within 0.6 m of the robot count as passable for the frontier search (otherwise no frontier is reachable at frame 0). Known obstacles there stay blocked.
5. Path length = 8-connected shortest path (steps 0.1 m and 0.1·√2 m; a diagonal step needs both side cells free) through the passable cells. All choices, commitments and the blacklist use this plain length.
6. A cluster's goal = its reachable cell with the shortest path length; goals within 0.5 m (Euclidean) of the robot are never chosen (they would count as reached at once). Tie: all goals within 0.5 m of the shortest length → the smallest |bearing| from the robot's heading to the goal cell (ties again → shorter, then lower cell index).
7. Commitment ends when the robot is ≤ 0.5 m (Euclidean) from the goal cell, the goal cell is no longer a frontier cell (of a kept cluster), or it becomes unreachable. Then a new goal is chosen in the same step.
8. Blacklist: the path length to the committed goal is logged every step since it was chosen; when 41 values exist and the oldest minus the newest < 0.5 m, the goal is dropped; no goal within 0.5 m of a dropped one is chosen again. A goal that was reached but is still a frontier cell (its unknown side cannot be seen, e.g. a mesh hole) is dropped with radius 0.2 m (the rest of its cluster stays).
9. Modes: no kept cluster has a reachable goal → `no_frontier`; reachable goals exist but all are dropped → `blacklist_exhausted`.
10. Path shape (not a rule): after the goal is chosen by plain length, the path handed to the drive loop is planned to that goal with Phase 2's clearance cost (step × (1 + 4 × shortfall below 0.445 m from the nearest known obstacle cell centre = 0.195 m + Phase 2's preferred 0.25 m)), so it keeps to the middle where there is room.
11. Bumps: a forward move stopped short by a collision (< half its length), or the safety step (note 16) finding the robot within 0.15 m of the floor map's edge with the teacher's path crossing blocked floor within 0.3 m, marks that cell (0.3 m ahead) as blocked for the teacher's planning (a bumper). Bumps never enter the network's map layers; they are saved per drive (`bump_events`) so Phase 11 can rebuild the teacher's state.
12. Approach: "seen" needs ≥ 200 pixels of the instance with depth in (0, 4] m; only instances with ≥ 1 usable view point (snapped ≤ 0.2 m, real floor, Phase 2 rule) become "seen" (detections and the target layer count every instance of the category). Approach plans with unknown = free, inflation as note 3; if no view point is reachable it plans once more without inflation; still none → `no_path` (failure). The chosen (instance, view point) is kept until the seen set grows. Arrived = ≤ 0.1 m (Euclidean) from the view point; then face = heading to the instance's box centre.
13. Timing: the teacher's wall time per step is saved per frame (`teacher_ms`); median and p95 are reported.

**Drive loop and motion (item 4)** — `datagen_v2.DriveRunnerV2`
14. Look-around: when the teacher reports `no_frontier` / `blacklist_exhausted`, the robot first turns on the spot (left, 0.4 rad/s) until the teacher has a frontier ≥ 1 m away by path, the target is seen, or one full turn is done; only then does "no frontier" end a search drive. At most one look-around per place (≥ 1 m apart). Before the first goal choice this is a start turn (allowed type "start"); later it is type `look_around` (not one of the 4 allowed types; counted against the 2%). Exploration drives look around only at the start; a later "no frontier / all dropped" ends them as complete.
15. Re-target turn: in the step a new goal is chosen, if the aim point (0.5 m along the path) is > 30° off, turn on the spot until it is < 30°, then pure pursuit. Spot turns of pure pursuit itself (aim > 90° off without a new choice) are logged as `forced` (outside the 4 types).
16. Safety step (motion, not the teacher's choice): if the teacher's path within the next 1.0 m comes closer than 0.20 m to the floor map's edge (LIMO navmesh, as Phase 2's motion), or the robot itself is closer than 0.15 m, the robot follows a short smoothed floor-map path (Phase 2 clearance cost) to the farthest safe point of that stretch (search cost ≤ 3 m); if that point is not reachable within it, the floor-map path to the teacher path's end (≤ 30 m). Steps that used it are saved per frame (`safety_path`).
17. Start: wall-recovery starts (Phase 2 rule) turn toward the path side, left when there is no path yet, until the centre depth > 1.0 m or the path is within 30°. Then start turn while the aim is > 10° off (Phase 2).
18. Kick (drives drawn as kick drives, 21% asked as Phase 2): due after a drawn 20–200 steps (an online path has no fixed length), on a straight stretch (≤ 25° turn over the next 1 m), aim ≤ 15° off, floor-map clearance ≥ 0.25 m, ≥ 1.5 m of path left; an arc of 20–40° at 0.15 m/s, then pure pursuit steers back (Phase 2 numbers).
19. Detour (drives drawn as detour drives, 35% asked, 1 or 2 detours): due after a drawn 10–150 steps (the second ≥ 60 steps after the first started), in search / explore mode with ≥ 3.0 m of teacher path: a sideways point 0.5–1.5 m off the teacher path at 1.25 m ahead (Phase 2 detour rules, floor map), then rejoin the teacher path 2.5 m ahead; ends within 0.3 m of the rejoin point, after 80 steps, or when the teacher switches to approach. Realised = the detour route leaves the teacher path by ≥ 0.4 m.
20. Face: spot turn toward the instance centre until within 15°, then the drive ends; success = walking distance (LIMO navmesh) from the final position to the nearest usable view point of any instance of the category ≤ 0.2 m.
21. Outcomes: search → success / not_at_viewpoint / no_frontier / blacklist_exhausted / no_path (failures) / timeout (> 2,000 steps) / collision (≥ 20 consecutive collision steps). Exploration → complete (1,000 steps or no frontier / all dropped) or collision. Only success / complete drives are kept. **One attempt per drive id**: failures and discards are counted, never redrawn. A search drive whose start sampling finds nothing (200 tries) is `no_sample` (not a teacher attempt; counted separately).

**Target layer (item 2)** — `target_layer.py`, `map_v2.TargetFusion`
22. Cells in view = the storey cells hit by a valid depth pixel (0 < depth ≤ 4 m) that lies no lower than 0.10 m below the real floor (any height above: a TV on a wall counts).
23. Clean layer: detected instances' cells score 1.0 (no misses, no false marks). Noisy layer: per detected instance confidence ~ U(0.4, 1.0) then miss draw p = 0.2; per frame false-mark draw p = 0.05, centre = a random cell in view holding an obstacle-band point this frame (no such cell → no mark), 3×3 cells (clipped to the map) of value ~ U(0.3, 0.7). Per cell per frame the maximum score counts.
24. Exploration drives have no target layer (empty arrays).

**Seeds, ids, homes (items 5–6, 10)**
25. Drive seed = first 8 bytes (little endian) of sha256("20261020:<drive id>"). Two streams: `default_rng([seed, 0])` for start sampling, variations, kick / detour timing; `default_rng([seed, 1])` for target-layer noise. Results do not depend on worker count or order.
26. Pilot homes: 20 labelled train homes (semantic labels + ObjectNav v2 train goals) outside the val-drive split, drawn with `numpy.random.default_rng(808).choice(…, 20, replace=False)`, sorted; frozen in `pilot/pilot_homes.json`.
27. Camera: `camera_fair.yaml` (pin025) applied to robot_limo.yaml; colour, depth and semantic cameras share the pose (checked: `checks/camera_check.json`).

## 9. Pilot fix attempts (block item 6)
If pilot teacher success < 90%: at most 2 fix attempts, each changing only these teacher parameters of item 1:
blacklist window (`teacher.blacklist_window_steps`, 40), tie margin (`teacher.tie_margin_m`, 0.5), facing angle
(`teacher.face_tolerance_deg`, 15), commit distance (`teacher.commit_reach_m`, 0.5). Both versions are shown at S-pilot.
Every other number above stays as written.

## 10. Rules v2 (Naz's S-pilot answers, 2026-10-20; written before the re-pilot)
Pilot results under §1–9: v1 128/160 = 80.0%, fix 1 121/160 = 75.6%, fix 2 131/160 = 81.9% (`pilot*/pilot_numbers.md`,
`pilot/pilot_compare.md`). Naz's answer, verbatim:

> S-pilot visually OK. 1: neither A nor B as written — do this: (a) keep fix 2 (window 80); (b) teacher ignores frontier
> cells within 0.7 m of its own past trajectory (save frontier both with and without this ring for Phase 10); (c) start
> filter: keep a search drive only if from ≥1 view point of ≥1 instance of the category, some of 8 headings at pin025
> shows ≥200 px at ≤4 m — log the drop rate per category; write p8_rules_v2.md + sha256, re-pilot once on the same 20
> homes. If ≥90%: G8 row 3 pass. If <90%: record G8 row 3 failed, run full with the v2 rules, no further stop, failed
> drives discarded. Report the success per category and per cause both times. 2A. 3A (put numbers in p8_rules_v2.md).
> 4 yes — commit the current code now, then the v2 changes as a second commit before the full run. 5A, default in
> Phase 10 = the same frontier definition the teacher used.

These items replace the matching parts of §1–9; everything else stays. Numbers: `mapmad/configs/p8_datagen_v2.yaml`
(extends `p8_datagen.yaml`).

**1a. Blacklist window** 80 steps (20 s) instead of 40 (fix 2). Rule of §1 / note 8 otherwise unchanged (41 → 81 values).

**1b. Trajectory ring** (`teacher.frontier_ignore_trajectory_m` = 0.7)
- Trajectory = the robot's (X, Y) positions at frames 0..t (the current one included), as saved in `traj_data.pkl`.
- A frontier cell (note 1) whose centre lies within 0.7 m of any trajectory position is ignored. The 8-connected
  clusters and the ≥ 4-cell filter are formed from the remaining cells (a cluster cut below 4 cells by the ring drops).
- The teacher plans with this frontier (choice, commitment, "still a frontier", blacklist, no_frontier).
- Saved for Phase 10: both frontiers are rebuilt exactly from the saved files: `map_v2.layers_at(map, t, …)` = plain
  frontier (item 3), `layers_at(map, t, …, positions=traj["position"], visited_m=meta["frontier_ignore_trajectory_m"])`
  = the teacher's. `mapmad_meta.json` stores `frontier_ignore_trajectory_m` per drive. Checked byte-identical against
  the live state at every step of 2 dev drives (`checks/map_parity_v2.json`). Phase 10 default = the teacher's (5A).
- Explanation (not a rule): the camera first sees the floor ~0.55 m ahead, so a ring of never-seen floor follows the
  robot; in the pilot 47% of the chosen goals lay within 0.7 m of earlier positions.

**1c. Start filter** (`sampling.seeable_filter`)
- Per labelled home, before any drive, for each category with ≥ 1 usable instance: the category passes if from ≥ 1
  usable view point (note 12) of ≥ 1 of its instances, at least one of 8 headings (0°, 45°, …, 315°, camera pin025 at
  that point) shows a usable instance of the category with ≥ 200 px at depth (0, 4] m (the "seen" rule, note 12).
  Fixed order (instances, view points, headings as stored), stops at the first pass; no random draws.
- Search drive sampling: the category is drawn uniformly as before; a category that fails is dropped
  (`notes.dropped_categories`) and drawn again (= uniform over the passing categories). No passing category in the
  home → `no_sample`. Dropped draws are not teacher attempts (not in the success denominator).
- Logged: per home and category pass/fail and render count (`_homes/<home>.json` `categories`); report per category =
  homes failing, dropped draws and drop rate = dropped draws / (dropped draws + search drives of the category).
- All instances of a passing category stay targets (success zone and "seen" as before).

**2A. Collision drives**: accepted above 2% if so; collision drives are discarded (not kept), as in Phase 3.

**3A. Detours**: asked in 65% of drives (`variations.detour.share` 0.65, was 0.35), 1 or 2 detours as before; due with
≥ 2.5 m of teacher path (`motion.detour_ahead_m` 2.0 + 0.5, was 3.0), rejoin the teacher path 2.0 m ahead (was 2.5);
other detour numbers of note 19 unchanged.

**Gate (G8 row 3) after the re-pilot** (same 20 homes, same drive ids p8p-…, dataset
`pilot_habitat_mapmad_v2_rules2`, outputs `pilot_rules2/`): teacher success ≥ 90% → row 3 success item passes; < 90%
→ recorded failed, the full run uses the v2 rules without a further stop, failed drives discarded. Report success per
category and failures per cause for fix 2 and the re-pilot.
