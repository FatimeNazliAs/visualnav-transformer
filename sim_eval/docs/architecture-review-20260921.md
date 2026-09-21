# sim_eval — architecture review after P6

**Date:** 2026-09-21
**Branch:** `feature/sim-eval-p6-headline`
**Scope reviewed:** all of `sim_eval/` after P6 (287 tests at review time, 289 after)
**Plan:** `.claude/plans/nomad-sim-evaluation.md`
**Predecessors:** `architecture-review-20260916.md` (after P1) · `architecture-review-20260917.md` (after P4) · `architecture-review-20260918.md` (after P5)

This is the Markdown companion to the HTML report. The HTML was written to `/tmp` and is not kept in the repo; the findings are recorded here.

This review also created the harness's first domain glossary, `sim_eval/CONTEXT.md`, and its first decision record, `sim_eval/docs/adr/0001-per-arm-statistics-are-per-task-means.md`.

---

## What this review asked

1. The usual question: **where will P7 have to change this code rather than add to it?**
2. A new question, because P6 produced the first headline table: **does any module make a reported number mean something other than its label says?**

## Candidates

| | Candidate | Strength | Outcome |
|---|---|---|---|
| **A** | One statistic name, two definitions, two numbers | Strong | **Implemented** |
| **B** | A run is set up by mutating the shared config | Worth exploring | Open. P6 built the fix but it is not yet used |
| **C** | The trace's pose/tick rule is re-derived by each reader | Worth exploring | Open. 5 readers, not 3 |
| **D** | Duplicated runner scripts and GPU boilerplate | Worth exploring | Open, unchanged |
| **F** | The comparison works out the seed rule itself | Speculative | Open |
| **E** | The input check's copy of training's crop has no tests | — | **Closed** (3 tests in `tests/test_input_check.py:39-54`) |

Two predictions from the last review did not come true, which makes B and C less urgent:

- The two-arm driver was expected to become `evaluate`'s fifth caller. It isn't one: `run_p6_headline.sh` runs `run_eval.sh` once per arm.
- The comparison was expected to become the trace's third reader. It reads only the CSV.

### A — One statistic name, two definitions

**Strength: Strong. Implemented.**

**Problem.** Two functions rolled the same per-episode rows up and used the same statistic names:

- `metrics.aggregate` **pooled** across episodes: the sum of collision events over the sum of metres, and the sum of contact ticks over the sum of ticks.
- `EPISODE_STATISTICS` / `summarize_with_se` **averaged the per-episode column**.

The headline run printed both: each arm's log came from `aggregate`, and the comparison table from `summarize_with_se`. On P6's data, `best_combined`'s log said **76% of ticks in contact** while the table said **0.401**. Collisions per metre read 0.200 in the log and 0.248 in the table.

Two smaller problems sat in the same place:

- `final_distance_m` and `final_geodesic_distance_m` were one computation under two names.
- `aggregate`'s docstring claimed "P6 reports these", but P6 reported the other function's numbers.

This was the only candidate that could make a reader misread a number. The plan (§6) says only "collisions per episode / per meter" and never chose between the two definitions.

**Decision.** Every per-arm statistic is a mean over episodes of a per-episode value, never a pooled ratio. This is recorded in the ADR, and "per-arm statistic" is defined in `CONTEXT.md`. It is the only definition that can carry a standard error across tasks, which is what P6's table was asked to report. The pooled definitions were **deleted**, not renamed.

**What was done.**

- Deleted `metrics.aggregate` and `metrics._mean_distance`.
- Renamed `summarize_with_se` to `summarize`. With its rival gone, "with SE" no longer distinguished it from anything.
- `EPISODE_STATISTICS` is now the only definition of each name. The comment above it says why each statistic is a mean over tasks.
- `format_aggregate(checkpoint, aggregate(...))` became `format_summary(checkpoint, summarize(...))`. It prints the same block, now labelled "means per episode", with the contact line reworded to "of an episode's ticks". Updated its caller at `run_eval.py:357`, and the `summarize` call in `p6_2_compare.py:79`.
- Tests: the old test that pinned the pooled contact fraction (0.75) now pins the per-task mean (0.5). Its path lengths were chosen so that the two definitions of collisions per metre also differ (0.25 against 0.1). A new test checks that the end-of-run block prints the comparison table's means, and another covers a run with no episodes.

**How it was checked that nothing moved.**

- GPU-free suite: 289 passed.
- P6's comparison was regenerated from the same rows. All three output files, `p6_2_comparison.md`, `p6_2_comparison.csv` and `p6_2_episodes.csv`, are **byte-identical** to the originals. The table being interpreted did not change.
- The new end-of-run block on P6's data prints `best_combined` at 40% in contact and 0.248 per metre, matching the table (0.401 and 0.248), where the log had said 76% and 0.200.

**Left as is.** `outputs/p6_1_metrics/{best_combined,clean_stock}.log` were printed by the old code and still show the pooled numbers. They record what was printed, so they were not rewritten.

### B — A run is set up by mutating the shared config

**Strength: Worth exploring. Open.**

Four scripts set up a run in **14 places** by assigning into `EvalConfig` after it is built. The last review counted 13, but the count was already 14 then. `p4_1` still changes one config object between two `evaluate` calls in the same process.

Most of the parameters on these functions are only passed through:

| Function | Parameters |
|---|---|
| `evaluate` | 9 |
| `score_checkpoint` | 8 |
| `run_scene` | 7 |
| `score_task` (new in P6) | 7 |

P6 showed what the fix looks like: its run is configured entirely by `extends:` layering through `deep_merge`, with no mutation at all. Moving the four older scripts onto that path would let the passed-through parameters collapse into a single run object. **This is the next candidate to take on.**

### C — The trace's pose/tick rule is re-derived by each reader

**Strength: Worth exploring. Open.**

The rule is `len(poses) == ticks + 1`, where `poses[i]` is where tick *i* started. One writer produces it and **five** places work it out again:

- `metrics.path_length`
- `diagnosis._displacement_m`
- `diagnosis._first_contact`
- `p5_2_diagnose.write_tick_log`
- `recorder`, which keeps its own pose list and drops the last pose

The last review missed `_first_contact` and `recorder`. Only the writer asserts the rule, and `write_tick_log` has no tests. P6 added no reader.

### D — Duplicated runner scripts and GPU boilerplate

**Strength: Worth exploring. Open.**

Unchanged in substance. The same five scripts have byte-identical bodies, and P6's two new scripts are not among them. On the Python side:

- P6 took the copies of the `__main__` try/except from 9 to 10, and they catch six different sets of exceptions.
- 10 of the 11 scripts carry the same four-line block that re-quotes their arguments.

### F — The comparison works out the seed rule itself

**Strength: Speculative. Open.**

`p6_2_compare.expected_episodes` computes `task.seed + seed_offset` by hand, while P6's `EpisodeRunner.seed_for` owns the same rule. It is only speculative because a drift would fail loudly: `fairness_problems` compares the expected seeds against the rows actually scored.

---

## Deletion test on P6's own code

| Thing | Verdict |
|---|---|
| `run_eval.score_task` | **Keep.** It is the unit the crash isolation wraps. |
| `EpisodeRunner.seed_for` | **Keep.** The resume key and the seed each row records come from this one place. |
| `read_layered_yaml` / `deep_merge` | **Keep.** B wants more callers of these, not fewer. |
| `metrics.mean_se` | **Keep.** It is the only place the standard error is computed. |
| `metrics.aggregate` | **Deleted** (A). |
| `configs/p6_0_maben_probe.yaml` | **Deleted** before commit. It was a throwaway diagnostic, and what it found is recorded in `p6_headline.yaml`. |

## Smaller notes (test gaps, not architecture)

- `p6_2_compare.compare()` has no end-to-end test.
- `compare()` does not create its output directory; it assumes scoring already has. Run alone against a fresh config, it would fail with `FAILED:`.
- The header of `run_eval.sh:7` names `outputs/p3_2_metrics`, which is only correct for the default config.
