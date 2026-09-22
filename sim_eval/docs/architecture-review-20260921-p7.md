# sim_eval — architecture review after P7

**Date:** 2026-09-21
**Branch:** `feature/sim-eval-p7-stretch`
**Scope reviewed:** the P7 surface — `task_stats.py`, `p7_contrasts.py`, `run_arms.sh`, adoption in `task_set.py`, the `p7_*` configs — and how it sits in `sim_eval/` (340 tests before and after)
**Plan:** `.claude/plans/nomad-sim-evaluation.md`
**Predecessor:** `architecture-review-20260921.md` (after P6)

A structure-only pass: nothing it changed may move a reported number. Every comparison and contrast output P7 wrote (`p7_1_rescore_*`, `p7_2_e1_*`, `p7_3_e2_*`, `p7_3_epoch_*`, 26 files) was regenerated before and after the refactor and compared byte for byte: all 26 identical.

---

## What this review asked

1. Where does P7's code repeat, or reach across, what was already there?
2. After P7 changed what an episode *is* (several per task), do the domain glossary and the ADRs still say true things?

## Candidates

| | Candidate | Strength | Outcome |
|---|---|---|---|
| **A** | The fairness gate lived inside an entry-point script | Strong | **Implemented** |
| **B** | The glossary and ADR-0001 still said "each task exactly once" | Strong | **Implemented** |
| **C** | Two per-arm summaries: episode-level and task-level | Worth exploring | Open — changes a printed line |
| **D** | Adoption re-reads each source manifest once per task | Speculative | Open |
| **E** | Duplicated runner scripts (P6's D) | Worth exploring | Open, one more copy |
| **F** | The comparison works out the seed rule itself (P6's F) | — | **Closed** in P7 (`episode_runner.episode_seed`) |

---

### A — The fairness gate lived inside an entry-point script

**Strength: Strong. Implemented.**

`p7_contrasts.py` imported the `p6_2_compare.py` *script* as a library for seven things — the naming, the fairness gate, the seed expectation, the house list, the CSV writer, the error type, a constant — and then repeated the part that matters: open the task set, read every table, run the gate, raise. Two copies of the one check that stands between a table and a reported number, one of them with an extra step (`restrict_to_task_set`) the other did not have.

**Change:** a new module, `comparison.py`, with one entry, `load_fair_tables(eval_config, table_paths, restrict_to_task_set=False)`, plus what both scripts write with (`ComparisonNaming`, `houses`, `round_cell`, `write_csv`). `p6_2_compare.py` and `p7_contrasts.py` are now each one job: the side-by-side table, and the paired/additivity tables. `p7_contrasts.run` was split into `write_paired` and `write_additivity`. Two identical `_round` helpers became `comparison.round_cell`.

**Deletion test:** deleting `comparison.py` puts the gate back into two places — it concentrates. Tests moved with the functions (`tests/test_compare.py` and `tests/test_contrasts.py` now call `comparison.*` for the gate, the naming and the restriction).

### B — The domain model still said "each task exactly once"

**Strength: Strong. Implemented.**

`CONTEXT.md` defined a per-arm statistic as a mean over episodes that "because each arm runs each task exactly once" is a mean over tasks — ADR-0001 says the same. P7 made that premise false (three seed offsets per task) and the code chose the task (`task_stats`), which is the ADR's intent; the words had not followed.

**Change:** `CONTEXT.md` — *Per-arm statistic* now says seeds are averaged within a task first; *Arm* includes the context stride; new terms *Context stride*, *Seed offset*, *Adopted task*, *Contrast*, *Interaction*. `docs/adr/0002-under-several-seeds-the-task-stays-the-unit.md` records the decision (task as the unit; Wilcoxon on success fractions, not McNemar, under several seeds); ADR-0001 points to it.

### C — Two per-arm summaries

**Strength: Worth exploring. Open.**

`metrics.summarize` averages episodes; `task_stats.summarize` averages a task's seeds, then tasks. The comparison uses the second; the block `run_eval.py` prints at the end of a scoring run uses the first. For a complete table with the same number of seeds per task the means are identical, and the printed block shows means only — so nothing reported disagrees today. A *partial* multi-seed table (a run stopped mid-seed) weights tasks unequally in the printed block. This is ADR-0001's "one name, two definitions" in a milder form.

**Why not done here:** routing the printed block through `task_stats.summarize` and deleting `metrics.summarize` changes what a scoring run prints for a partial table. This pass was held to no behaviour change. It is the first thing to do next time `metrics.py` is touched.

### D — Adoption re-reads each source manifest once per task

**Strength: Speculative. Open.**

`TaskSetConfig.adoptable` calls `task_set.load(source)` for every task it is asked about — 60 tasks x 2 sources for E2. Correct, and cheap at this size (well under a second). Loading the sources once per build would be the shape to have if task sets grow; it does not change what is adopted.

### E — Duplicated runner scripts

**Strength: Worth exploring. Open, one more copy.**

P6's finding D. `run_p7_contrasts.sh` is another copy of the six-line "re-quote the arguments, `sim_exec` one Python script, chown the outputs" body. `run_arms.sh` did remove one: `run_p6_headline.sh` is now a one-line call into it.

---

## Deletion test on P7's own code

| Thing | Verdict |
|---|---|
| `task_stats.py` | **Keep.** The one place the unit of analysis is decided; deleting it spreads seed-averaging into every table. |
| `comparison.py` | **Keep.** See A. |
| `p7_contrasts.ContrastsConfig.restrict_to_task_set` | **Keep.** One user (the epoch contrast), off by default, and the only way to pair tables from two task sets that share trails without copying rows. |
| `task_set` adoption (`adopt_from`) | **Keep.** It is what makes P6's reused rows valid by construction rather than by assertion. |
| `run_p6_headline.sh` | **Keep** as P6's documented entry point; it is one line. |
| `configs/p7_3_probe_*.yaml` | **Delete** before commit (throwaway, as P6's Maben probe was). |
