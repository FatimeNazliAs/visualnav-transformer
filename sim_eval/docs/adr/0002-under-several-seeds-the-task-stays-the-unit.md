# Under several seeds, the task stays the unit

From P7 an arm runs each task under several seed offsets, so a task has several episodes. Every statistic averages a task's seeds into one value first, and takes its mean, standard error and paired tests across tasks. A task's seeds share a start, a goal, a trail and a house and differ only in diffusion noise, so treating them as independent draws would shrink every standard error by about the square root of the seed count, and credit the seeds with evidence only the tasks can give. With one seed per task this gives exactly the numbers ADR-0001 describes, and P6's comparison was reproduced to the byte.

Success between two arms is tested with a Wilcoxon signed-rank test (and a paired t test) on the per-task success fractions, not with McNemar's test. McNemar pairs single episodes, and under several seeds that would pair seeds. `task_stats.mcnemar` refuses a table with more than one episode per task, and the paired tables report McNemar only when every task has one seed (P7 step 1b).

## Considered Options

- **Episodes as the unit.** It is what the P6 code computed. It was rejected because it overstates the evidence as soon as a task is run more than once.
- **A mixed model with a random effect per task.** It would use the within-task spread instead of discarding it. It was not adopted: with 20 to 60 tasks and three seeds, the per-task mean is almost as efficient, and it is a statistic a reader can recompute from `<name>_tasks.csv` by hand.
