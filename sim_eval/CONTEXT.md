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

### A run

**Arm**:
One checkpoint as it takes part in a comparison, including the render resolution it was trained at, which is part of what the arm is.
_Avoid_: model, condition, variant

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
A number that describes an arm: success rate, SPL, collision events per episode and per metre, the fraction of ticks in contact, final distance to the goal, and time to goal. It is always a mean over episodes of a per-episode value, never a ratio of totals pooled across episodes. Because each arm runs each task exactly once, that makes it a mean over tasks, so every task counts equally and the statistic has a standard error across tasks.
_Avoid_: pooled rate, overall rate, aggregate

**Time to goal**:
How long a successful episode took. It is averaged over successes only, so a failure has no time to goal rather than a long one.
_Avoid_: steps, episode length
