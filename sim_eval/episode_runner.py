"""Run one episode to its end, and hand back what it did.

P1 drove the bridge until NoMaD said it had arrived, and P2 built the trails.
This is the piece that turns a drive into a *measurement*: a checkpoint, a task
and a seed go in, and an episode that ended for a stated reason comes out,
carrying everything `metrics.py` needs to score it.

**Two end conditions, and only two** (plan §6):

  * **success** — the agent is physically within `success_radius_m` of the
    task's goal pose.
  * **timeout** — the tick budget, tied to the reference path's length, is
    spent.

Collisions are logged every tick and are never terminal (count-and-continue).
That is a deliberate difference from the reference drive in P2, which rejects a
route that touches anything: the reference path is meant to be perfect, and an
agent following it is not.

**What is deliberately *not* an end condition** is NoMaD's own opinion. The
bridge stops when the distance head localizes onto the last node, which is
`navigate.py`'s rule on the real robot, and it is a claim about where the agent
*thinks* it is. Ending on it would let a lost agent declare victory, and worse,
would stop the odometer early and inflate its SPL. So the episode ends on where
the agent *is*; the tick it first claimed arrival is recorded as
`declared_arrival_tick`, a diagnostic, and the run continues either way.

The runner never opens a simulator or loads a checkpoint: it is handed both. So
`run_eval.py` can pay for a scene and a checkpoint once and run many episodes
through them, and `tests/test_episode_runner.py` can pin the stop conditions
with neither.
"""

import math

import numpy as np

import metrics

# The geodesic that measures how far a failed episode stopped from its goal.
# `scene.get_shortest_path` raises networkx's NoPath when the two ends are not
# connected, which happens for real: an agent that climbs onto furniture leaves
# the traversable component entirely. That is a fact about the episode, not an
# error, so it is caught and reported as "no geodesic" rather than raised.
NO_PATH_ERRORS = ("NetworkXNoPath", "NodeNotFound")

# How "within the goal radius" is measured. Geodesic is the default and is what
# `final_distance_to_goal` reports, so success and the distance beside it in the
# table mean the same thing; euclidean is the looser reading, kept because it is
# what a straight-line reading of plan §5 says and because it costs nothing.
# They disagree exactly where it matters — the first episode scored in P3 ended
# 0.97 m from its goal in a straight line and 1.08 m around the furniture, which
# is a success under one and a timeout under the other.
SUCCESS_METRICS = ("geodesic", "euclidean")


class EpisodeRules:
    """When an episode is over — the two end conditions, as numbers.

    `success_radius_m` is plan §5's "within ~1 m of the goal pose", measured
    from the pose the reference drive actually finished at (`goal_pose` in the
    task metadata), not from the point the planner aimed at. The goal *image*
    the model is chasing was taken from that pose, so it is the place the model
    is being asked to reach.

    The timeout is "tied to reference-path length" (plan §6), spelled out as:

        timeout_ticks = ceil(slack * geodesic_m / (max_v * dt)  +  turn_s / dt)

    The first term is how many ticks the shortest path would take at the
    robot's top speed — the fastest any agent could possibly finish — scaled by
    `slack` to leave room for steering, overshoot and correction. The second is
    a flat allowance for turning on the spot, which buys no distance at all: a
    differential drive starting off-heading spends up to `pi / max_w` seconds
    lining up before it moves, and on a short task that would otherwise eat a
    visible share of the budget. Both scale with the robot's own limits rather
    than being tick counts typed in by hand, so changing the control rate does
    not silently change what a timeout means.
    """

    def __init__(self, success_radius_m=1.0, success_metric="geodesic",
                 timeout_slack=4.0, turn_allowance_s=None, min_timeout_ticks=40):
        self.success_radius_m = float(success_radius_m)
        self.success_metric = str(success_metric)
        self.timeout_slack = float(timeout_slack)
        # None = derive it from the robot: a half turn at max_w. A number
        # overrides that, for a scene or a robot where it is not enough.
        self.turn_allowance_s = (None if turn_allowance_s is None
                                 else float(turn_allowance_s))
        # A floor, so a short task still gets a usable budget. It binds only
        # below ~0.5 m of geodesic, which the sampler's 3 m minimum excludes.
        self.min_timeout_ticks = int(min_timeout_ticks)

        if self.timeout_slack <= 0:
            raise ValueError("timeout_slack must be positive")
        if self.success_radius_m <= 0:
            raise ValueError("success_radius_m must be positive")
        if self.success_metric not in SUCCESS_METRICS:
            raise ValueError(
                "success_metric must be one of {}, not {!r}".format(
                    SUCCESS_METRICS, self.success_metric))

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))

    def turn_allowance(self, limits):
        """Seconds allowed for turning on the spot: half a turn at max_w."""
        if self.turn_allowance_s is not None:
            return self.turn_allowance_s
        return math.pi / limits.max_w

    def timeout_ticks(self, geodesic_length_m, limits):
        """The tick budget for a task whose shortest path is this long."""
        metres_per_tick = limits.max_v * limits.dt
        ticks_at_top_speed = float(geodesic_length_m) / metres_per_tick
        budget = math.ceil(self.timeout_slack * ticks_at_top_speed
                           + self.turn_allowance(limits) / limits.dt)
        return max(budget, self.min_timeout_ticks)

    def as_dict(self):
        return {
            "success_radius_m": self.success_radius_m,
            "success_metric": self.success_metric,
            "timeout_slack": self.timeout_slack,
            "turn_allowance_s": self.turn_allowance_s,
            "min_timeout_ticks": self.min_timeout_ticks,
        }

    def summary(self):
        turn = ("pi / max_w" if self.turn_allowance_s is None
                else "{} s".format(self.turn_allowance_s))
        return ("success within {} m ({}) · timeout = ceil({} x geodesic / "
                "(max_v x dt) + {} / dt), at least {} ticks".format(
                    self.success_radius_m, self.success_metric,
                    self.timeout_slack, turn, self.min_timeout_ticks))


def geodesic_distance(scene, floor, source_xy, goal_xy):
    """Geodesic distance between two points, or None if they are not connected.

    Plan §6 asks for the *geodesic* final distance-to-goal, and the difference
    matters exactly where the metric is read: an agent stopped a metre from the
    goal with a wall between them has not nearly arrived.

    `get_shortest_path` grafts an off-graph endpoint onto the graph as a new
    node with a single edge to its nearest neighbour, so calling it mutates the
    scene. That is safe here and it is worth saying why, because it is the
    reason this is not simply asked every tick: a node with one edge is a leaf,
    no shortest path ever routes *through* a leaf, and so no later query comes
    back shorter for having been called. The graph still grows by a node per
    call, which is why the callers below ask only at the end of an episode and
    on the handful of ticks that are already inside the radius.
    """
    try:
        _path, distance = scene.get_shortest_path(
            floor, np.asarray(source_xy, dtype=float)[:2],
            np.asarray(goal_xy, dtype=float)[:2], entire_path=False)
    except Exception as error:                       # noqa: BLE001 - see below
        # networkx is not imported here (it is iGibson's dependency, not ours),
        # so the exception is identified by name rather than by class. Anything
        # else is re-raised: a scene that cannot plan at all is a defect, and
        # swallowing it would turn every episode's distance into a silent blank.
        if type(error).__name__ not in NO_PATH_ERRORS:
            raise
        return None
    return float(distance)


def seed_episode(seed):
    """Fix everything stochastic in an episode, so two arms face the same one.

    Two generators are involved and both matter (plan §7): numpy's, which the
    task's reset samples a pose through, and torch's, which the diffusion head
    draws its 8 samples from. Seeding them with the *task's* seed rather than a
    per-run counter is what makes the comparison fair — the same task hands
    every checkpoint the same reset and the same noise.

    torch is imported here rather than at module scope so that importing this
    module stays cheap and GPU-free, the way bridge.py keeps itself free of
    iGibson.
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


class EpisodeResult:
    """One finished episode: how it ended, what it scored, and what it did."""

    def __init__(self, episode_metrics, records, poses):
        self.metrics = episode_metrics
        self.records = records
        self.poses = poses

    @property
    def success(self):
        return self.metrics.success

    def trace(self):
        """The episode as JSON — the evidence behind its row in the table.

        Frames are left out on purpose: they are large, and P4's recorder
        re-runs the episode to capture them. What is here is everything needed
        to redraw the path and read the decisions behind it.
        """
        row = self.metrics.as_row()
        row["poses"] = [[round(value, 4) for value in pose] for pose in self.poses]
        row["ticks_log"] = [
            {"tick": record.index,
             "node": record.closest_node,
             "subgoal": record.subgoal_node,
             "v": round(record.v, 4),
             "w": round(record.w, 4),
             "collided": bool(record.collided)}
            for record in self.records]
        return row


class EpisodeRunner:
    """Runs episodes for one checkpoint, in one already-open simulator.

    Holds no per-episode state: `run` is handed the task and the seed, so the
    same runner scores a whole scene's worth of tasks without anything leaking
    from one episode into the next except the simulator itself (which `reset`
    clears) and numpy's global RNG (which `seed_episode` overwrites).
    """

    def __init__(self, policy, body, rules=None, floor=0):
        self.policy = policy
        self.body = body
        self.rules = rules or EpisodeRules()
        self.floor = floor

    def _euclidean_distance(self, goal_xy):
        """How far the agent is from the goal right now, in a straight line."""
        position = np.asarray(self.body.pose[:2], dtype=float)
        return float(np.linalg.norm(position - np.asarray(goal_xy, dtype=float)))

    def _geodesic_distance(self, goal_xy):
        """How far the agent is from the goal around the furniture, or None."""
        return geodesic_distance(
            self.body.env.scene, self.floor, self.body.pose, goal_xy)

    def _reached(self, goal_xy):
        """Is the agent inside the goal radius — by whichever measure rules say.

        The straight-line distance is checked first, and under the geodesic
        rule that is a proof rather than an optimization: a path around the
        furniture is never shorter than the line through it, so anything
        outside the radius in a straight line is outside it geodesically too.
        Only the ticks that pass that gate pay for an A* — a handful per
        episode, at the very end of it.

        An agent with no geodesic to the goal has not reached it. That is not a
        technicality: it means the agent has left the traversable component,
        which in this scene means it has climbed onto the furniture.
        """
        if self._euclidean_distance(goal_xy) > self.rules.success_radius_m:
            return False
        if self.rules.success_metric == "euclidean":
            return True
        geodesic = self._geodesic_distance(goal_xy)
        return geodesic is not None and geodesic <= self.rules.success_radius_m

    def run(self, task, checkpoint_name, seed=None, on_tick=None):
        """Run `task` to success or timeout, and score it.

        `seed` defaults to the task's own, which is what the fairness protocol
        wants; it is an argument only so a rerun can deliberately vary it.
        """
        import bridge

        seed = task.seed if seed is None else int(seed)
        seed_episode(seed)

        rules = self.rules
        limits = self.body.limits
        timeout_ticks = rules.timeout_ticks(task.geodesic_length_m, limits)
        goal_xy = task.goal_xy

        # The floor's height belongs to the scene, not to the task, so it is
        # read from the open scene here — exactly as P2 does when it places the
        # robot for the reference drive.
        floor_height = float(self.body.env.scene.floor_heights[self.floor])

        runner = bridge.NomadBridge(self.policy, self.body)
        runner.start_episode(task.topomap(), task.start_pose(floor_height))

        records = []
        poses = []
        declared_arrival_tick = None
        success = False

        for _ in range(timeout_ticks):
            record = runner.tick()
            records.append(record)
            poses.append(record.pose)

            if declared_arrival_tick is None and runner.reached_goal:
                declared_arrival_tick = record.index
            if on_tick is not None:
                on_tick(record)

            # Measured after the tick, on the pose the tick produced — the
            # record holds the pose the tick *started* from.
            if self._reached(goal_xy):
                success = True
                break

        # The odometer has to include the leg of the last tick, which no record
        # holds: each one stores where its tick began.
        final_pose = self.body.pose
        poses.append(final_pose)

        episode_metrics = metrics.EpisodeMetrics(
            checkpoint=checkpoint_name,
            task=task,
            seed=seed,
            success=success,
            collision_ticks=sum(1 for record in records if record.collided),
            collision_events=metrics.count_collision_events(
                record.collided for record in records),
            path_length_m=metrics.path_length(poses),
            final_geodesic_distance_m=self._geodesic_distance(goal_xy),
            final_euclidean_distance_m=self._euclidean_distance(goal_xy),
            ticks=len(records),
            seconds=len(records) * limits.dt,
            timeout_ticks=timeout_ticks,
            success_radius_m=rules.success_radius_m,
            success_metric=rules.success_metric,
            declared_arrival_tick=declared_arrival_tick,
            final_node=records[-1].closest_node if records else 0,
        )
        return EpisodeResult(episode_metrics, records, poses)
