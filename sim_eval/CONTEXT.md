# Closed-loop simulation evaluation

Puts trained NoMaD checkpoints in a simulated house, lets each one drive itself along the same fixed set of navigation problems, and scores how they do. It exists so that two checkpoints can be compared on identical problems under identical conditions.

## Language

### The problem

**Trail**:
A sequence of goal images captured by driving a route once, which the model follows. The simulator's equivalent of a topomap recorded by hand on the real robot.
_Avoid_: topomap (the on-disk format), goal trail, route

**Node**:
One image of a trail. The model localizes itself against nodes.
_Avoid_: waypoint (that is a point the model predicts), frame

**Subgoal**:
The node the model is steering towards at a given moment.
_Avoid_: target, next goal

**Reference path**:
The shortest route through the house that a trail was driven along. Its length is the denominator of SPL.
_Avoid_: ground-truth path, optimal path

**Task**:
One navigation problem: a start pose, a trail to follow, and the goal pose at the trail's end.
_Avoid_: scenario, problem, trial

**Task set**:
The fixed, seeded list of tasks that every arm faces. It is built once and cannot change after any arm has been scored against it.
_Avoid_: benchmark, test set, dataset

**Adopted task**:
A task a new task set copies from an existing one instead of driving it again, because it is exactly the task the new set would build: same knobs, same seed, same world. Episodes already scored on it stay valid in the new set.
_Avoid_: reused trail, cached task

### A run

**Arm**:
One checkpoint as it takes part in a comparison, including the render resolution and the context stride it was trained with, which are part of what the arm is.
_Avoid_: model, condition, variant

**Context stride**:
How many ticks apart the frames of an arm's observation window are. It is the arm's own, read from its training run: stock NoMaD's is 1, the stride-3 arms' is 3.
_Avoid_: frame skip, context spacing

**Seed offset**:
What a run adds to every task's own seed. Each offset is one more episode of every task under different diffusion noise, and every arm in a comparison runs the same offsets.
_Avoid_: replicate seed, run seed

**Episode**:
One arm's attempt at one task. It ends on success or timeout, never on a collision.
_Avoid_: rollout, trial, run

**Trace**:
The tick-by-tick record of an episode: where the robot was and what it decided.
_Avoid_: log, trajectory, history

### Scoring

**Success**:
An episode that ends within the success radius of the goal pose, measured around obstacles rather than in a straight line.
_Avoid_: arrival, completion

**Collision event**:
One distinct collision. An unbroken stretch of contact counts once, and the episode continues.
_Avoid_: collision (on its own, it is ambiguous with a contact tick), crash, hit

**Contact tick**:
A tick during which the robot is touching something. One collision event can span many.
_Avoid_: collision tick

**Per-arm statistic**:
A number that describes an arm: success rate, SPL, collision events per episode and per metre, the fraction of ticks in contact, final distance to the goal, and time to goal. It is computed per episode, averaged over a task's seed offsets into one value per task, and then averaged over tasks, never a ratio of totals pooled across episodes. Every task counts equally, and the statistic has a standard error across tasks, not across episodes.
_Avoid_: pooled rate, overall rate, aggregate

**Contrast**:
A weighted difference between arms taken task by task, then averaged over tasks: one arm minus another, or an arm against a baseline. Its standard error is across tasks, and it is paired, because every arm faced the same tasks under the same seeds.
_Avoid_: delta (on its own), effect

**Interaction**:
The contrast that tests whether two changes add up: the combined arm, minus each single-change arm, plus the baseline. Near zero, the changes are additive.
_Avoid_: synergy, combination effect

**Time to goal**:
How long a successful episode took. It is averaged over successes only, so a failure has no time to goal rather than a long one.
_Avoid_: steps, episode length
