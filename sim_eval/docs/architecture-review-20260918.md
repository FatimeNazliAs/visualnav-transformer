# sim_eval — architecture review after P5

**Date:** 2026-09-18
**Branch:** `feature/sim-eval-p5-navigate`
**Scope reviewed:** all of `sim_eval/` after P5 (226 tests at review time, 228 after)
**Plan:** `.claude/plans/nomad-sim-evaluation.md`
**Predecessors:** `architecture-review-20260916.md` (after P1) · `architecture-review-20260917.md` (after P4)

Markdown companion to the HTML report. Unlike the earlier reviews, the HTML version
was written to `/tmp` and was not kept in the repo. This file holds the findings.

---

## What this review asked

The same question as before:

> Where will **P6** have to *change* this code rather than add to it?

This round had a second question as well. A correctness fix, the camera's field of
view, is waiting on a decision, so the review first checked every seam that fix
would go through.

Still no `CONTEXT.md` and no `docs/adr/`. Domain terms come from the plan and the
README (*trail · node · task set · episode · trace · arm*).

---

## Candidates

| | Candidate | Strength | Outcome |
|---|---|---|---|
| **A** | The task-set fingerprint hashes the world config's *path*, not its contents | Strong | **Implemented** |
| **B** | Scripts set up a scoring run by editing the shared config (was P4's C) | Worth exploring | Open |
| **C** | The trace's pose/tick alignment rule is re-derived by each reader | Worth exploring | Open |
| **D** | Duplicated runner scripts and GPU boilerplate (was P4's D) | Worth exploring | Open, now due |
| **E** | The input check's copy of training's crop has no tests | Speculative | Open |

### A — The fingerprint didn't cover the camera

**Strength: Strong. Implemented.**

**Problem.** `TaskSetConfig.fingerprint()` hashed `base_knobs()`, and those knobs
hold `scene_config: configs/locobot_rs_bridge.yaml`. That is the file's path. What
the file contains (the camera, the robot, the physics) never went into the hash. Yet
every trail image is rendered through that camera, and every episode reopens the
`world.yaml` saved beside its trail. For example, `p5_2_task_set/Rs_01/world.yaml`
fixes `vertical_fov: 45`. So if the FOV fix waiting on a decision were applied by
editing the bridge config:

- **Against an existing set:** the fingerprint still matched, the set was reused,
  and every rollout kept running at 45°. The fix would do nothing, and nothing
  would report it.
- **Against a new directory:** the fix would apply, but the new set would have the
  same fingerprint as the old one, so nothing would show which camera a table was
  scored under.

This was the only candidate that could make a reported number wrong. The others
make the code harder to read or change.

**What was done.**

- `TaskSetConfig.world()` returns the resolved world config (`scene_id` is left as
  the file has it, because the scene list is hashed separately), and
  `fingerprint()` now hashes it too.
- Two tests in `tests/test_task_set.py`. The first edits `vertical_fov` in a world
  file in place and checks that the fingerprint changes. The second checks that
  changing any other world knob (`image_width`) changes it as well.

**Cost, paid.** Every manifest written before this has a stale fingerprint, so
`p4_1`, `p5_2` and any other pre-review set now refuse to load until rebuilt
(checked: `p5_2_task_set` raised `TaskSetError`, `ba2021…` on disk against
`ea28d0…` now). Doing it now was cheap because P6 has not written a table yet.

### B — Scripts set up a scoring run by editing the shared config

**Strength: Worth exploring. Open.** Carried over from P4's C.

`evaluate` still takes 9 parameters, `score_checkpoint` 8 and `run_scene` 7, and
`evaluate` now has 4 callers. There are **13 places across 4 scripts** that set up
a run by assigning straight into `EvalConfig` (`config.tasks.tasks_per_scene = …`,
`config.recording.directory = …`). `p4_1` changes it between two runs in the same
process. A `ScoringRun` built from the config plus overrides would take the
pass-through parameters out of those signatures. P6's two-arm driver will be the
fifth caller, which makes it the natural point to do this.

### C — The trace's alignment rule is worked out by each reader

**Strength: Worth exploring. Open.** New in P5.

`poses` holds one more entry than there are ticks, and `poses[i]` is where tick *i*
started. `episode_runner.py:336` writes it that way. `diagnosis._displacement_m`
and `p5_2_diagnose.write_tick_log` each work the rule out again, and nothing tests
`write_tick_log`. P6's comparison would be a third reader. An `EpisodeTrace` module
that owns the rule would do for the trace what moving the window arithmetic onto
`PolicyStep` did for the distance-head readings.

### D — Duplicated runner scripts and GPU boilerplate

**Strength: Worth exploring. Open, now due.** Carried over from P4's D.

With the header and script name removed, 5 runners have identical bodies:
`run_eval`, `run_p3_score_test`, `run_p4_record_test`, `run_p5_1_input_check` and
`run_p5_2_diagnose`. On the Python side, from P4 to P5, the `select_gpu` prologue
went from 7 to 9 copies, the `--gpu` argument from 7 to 9, and the `__main__`
try/except from 4 to 6. The proposed fix is a `sim_run SCRIPT.py "$@"` function in
`lib.sh` plus one Python entry-point helper. Each runner would keep its header,
because the headers are real documentation.

### E — The input check's copy of training's crop has no tests

**Strength: Speculative. Open.**

`p5_1_input_check.aspect_crop` copies training's `resize_and_aspect_crop` on
purpose, and the input check's claim that the crop is a no-op rests on it. It has
no tests. That is missing coverage rather than an architecture problem: three
GPU-free tests would close it.

---

## Deletion test on P5's own code

| Thing | Verdict |
|---|---|
| `recorder.subgoal_distance` | **Deleted.** Since P5 it only forwarded to `step.subgoal_distance()`. The overlay now calls the step directly. |
| `diagnosis.py` | **Keep.** Without it, the ordered first-match logic would move into the script, where the 15 tests couldn't reach it. |
| `SimBody.render_at_vertical_fov` | **Keep.** The render and its `finally` restore belong together. Doing it at the call sites would mean reaching through the adapter, which is the leak P4 closed. |
| `PolicyStep.window_start` / `_score` | **Keep.** The only copy of the window-offset recovery. |
| Unused `rules` local, `episode_runner.py` | **Removed.** It existed before P5; this is lint, not architecture. |

---

## How it was checked that nothing moved

A and the two deletions touch the measuring instrument itself, so the check was
run in the simulator as well as in the unit tests:

1. GPU-free suite: 228 passed.
2. The P5 diagnostic set was rebuilt from the same seeds and re-scored, and its
   table was diffed against the one from before the review.
3. `p4_1_record_test.py` rebuilt its one-task set and re-ran the filmed/unfilmed
   comparison.

Result: the rebuilt P5 set scored **exactly** the same table as before the
review (the diff of all 3 rows came back empty: 1 success, 2 wedged, 838 ticks),
the P4 row was also identical, and filmed and unfilmed runs still agree on every
column. Rebuilding with the same seeds gave back the same trails, so the only
change on disk is the manifest fingerprint.
