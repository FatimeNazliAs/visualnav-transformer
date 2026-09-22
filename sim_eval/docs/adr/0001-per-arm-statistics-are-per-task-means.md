# Per-arm statistics are means over tasks, not pooled rates

_Extended by [ADR-0002](0002-under-several-seeds-the-task-stays-the-unit.md): once a task is run under several seeds, "a mean over episodes" and "a mean over tasks" part ways, and the task is the one kept._

Every per-arm statistic is the mean, over episodes, of a value computed per episode. Because each arm runs each task exactly once, that is a mean over tasks, and it comes with a standard error across tasks. This includes the two ratios, collision events per metre and the fraction of ticks in contact. It is never a ratio of totals pooled across episodes. We chose this because a comparison between arms is read against the spread across tasks, and a pooled ratio has no such spread. A pooled ratio also lets one long timeout outweigh several short successes: in P6's data, a 338-tick timeout counts about six times as much as a 58-tick success. The plan said only "collisions per episode / per meter" and did not choose.

## Considered Options

- **Pooled ratio of totals** (sum of collision events / sum of metres; sum of contact ticks / sum of ticks). It is the obvious reading of "per metre", and it is what the harness printed at the end of every scoring run until the P6 architecture review. It was dropped, not kept under another name. On the same P6 rows it gave a contact fraction of 0.76 where the per-task mean gives 0.40, and both numbers had been printed under one name.
- **Report both.** Rejected: the pooled number has no standard error across tasks to set beside a difference between arms, and a second definition is how the name came to mean two things.

## Consequences

Until the review, the block printed at the end of every scoring run (P3 onward) used the pooled definition, so older run logs show pooled numbers. Only the printed summaries are affected. Every per-episode row, and P6's comparison table, were already per-episode and did not change.
