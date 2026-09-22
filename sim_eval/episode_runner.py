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

An episode is a **stream**, not a batch: iterate it and each tick arrives as it
is decided, with the frame the model saw. A consumer that wants numbers keeps
none of them; a consumer that wants a video encodes each one and drops it.
Nothing accumulates, which is what keeps a 439-tick episode from pinning 405 MB
of frames and a 20-task scene from pinning gigabytes.

The runner never opens a simulator or loads a checkpoint: it is handed both. So
`run_eval.py` can pay for a scene and a checkpoint once and run many episodes
through them, and `tests/test_episode_runner.py` can pin the stop conditions
with neither.
"""

import math

import numpy as np

import metrics

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


def _rounded(value, places=3):
    """Round a reading for the trace, keeping `None` as `None`.

    A distance the head could not be asked for is blank, not zero — the same
    rule the table already follows for a goal with no geodesic to it.
    """
    return None if value is None else round(float(value), places)


def episode_seed(task, seed_offset=0):
    """The seed one episode of `task` runs under — the one its row records.

    The task's own seed plus the run's offset. The one place this rule lives:
    the runner seeds an episode with it and resume keys on it, and the
    comparison uses it to say which episodes every arm must have run.
    """
    return task.seed + int(seed_offset)


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
    """One finished episode: what it scored, and the evidence behind it.

    Deliberately small. This used to also carry every `TickRecord`, which meant
    it carried every decoded camera frame — 405 MB for the longest episode that
    has actually run, and `run_eval` kept one of these per task for a whole
    scene. Nothing read them. The ticks are gone from here because the ticks
    are *streamed* now (see `Episode`): a consumer that wants a frame is handed
    it as it happens, and the result is what is left once the episode is over.
    """

    def __init__(self, episode_metrics, trace):
        self.metrics = episode_metrics
        self._trace = trace

    def trace(self):
        """The episode as JSON — the evidence behind its row in the table.

        Frames are not in it: they are large, and a consumer that wants them
        takes them off the stream. What is here is everything needed to redraw
        the path and read the decisions behind it.
        """
        return self._trace


class Episode:
    """One episode, as a stream of ticks that can be watched while it happens.

    Iterate it to run it; ask it for its `result()` once it is done:

        episode = runner.episode(task, "best_combined")
        for record in episode:
            print(record.summary())      # or encode a video frame, or both
        result = episode.result()

    **Why the consumer holds the loop and not the runner.** The work a consumer
    does per tick — write a frame to a video, draw an overlay, print a line —
    is the consumer's business, and a callback the runner invokes can only ever
    be given what the runner thought to pass. That is how the old `on_tick`
    became too weak for P4: it received a `TickRecord` and nothing else, no
    task, no output path, so every new need widened the runner's signature.
    Iterating inverts that. The runner keeps what only it can know — when the
    episode is over — and the consumer keeps everything else.

    **Nothing here retains a record.** Each one pins a 0.92 MB decoded frame,
    so an episode's worth is hundreds of megabytes and a scene's worth is
    gigabytes. What is kept instead is what the metrics need: the poses, the
    collision flags and the last step. The invariant is pinned by
    `tests/test_episode_runner.py`.
    """

    def __init__(self, runner, task, checkpoint_name, seed):
        self.runner = runner
        self.task = task
        self.checkpoint_name = checkpoint_name
        self.seed = seed
        self.timeout_ticks = runner.rules.timeout_ticks(
            task.geodesic_length_m, runner.body.limits)

        self._poses = []
        self._collided = []
        self._ticks_log = []
        self._last_step = None
        self._declared_arrival_tick = None
        self._success = False
        self._started = False
        self._finished = False

    def __iter__(self):
        """Run the episode, yielding each tick as it is decided and acted on.

        Ends on success — inside the goal radius — or on timeout, and on
        nothing else. Collisions are recorded every tick and never terminal.
        """
        import bridge

        if self._started:
            raise RuntimeError(
                "an episode runs once; build another with runner.episode(). "
                "Re-iterating would score a half-finished second run against "
                "the first one's tally.")
        self._started = True

        runner, body = self.runner, self.runner.body
        seed_episode(self.seed)

        # The floor's height belongs to the scene, not to the task, so it is
        # asked of the open scene here — exactly as P2 does when it places the
        # robot for the reference drive.
        floor_height = body.scene.floor_height

        driver = bridge.NomadBridge(runner.policy, body)
        driver.start_episode(self.task.topomap(), self.task.start_pose(floor_height))
        goal_xy = self.task.goal_xy

        for _ in range(self.timeout_ticks):
            record = driver.tick()

            self._poses.append(record.pose)
            self._collided.append(bool(record.collided))
            self._last_step = record.step
            self._ticks_log.append({
                "tick": record.index,
                "node": record.step.closest_node,
                "subgoal": record.step.subgoal_node,
                # What the distance head actually read, for the node it
                # localized onto and for the node it steered at. P5 logs them
                # because the two nodes alone cannot distinguish a trail that
                # is not advancing from one the head is confident about: the
                # subgoal only moves on when the closest reading falls under
                # `close_threshold`, and that number was previously visible
                # nowhere but a video frame.
                "dist_closest": _rounded(record.step.closest_distance()),
                "dist_subgoal": _rounded(record.step.subgoal_distance()),
                # The waypoint the PD controller actually steered to, in
                # metres. P4's overlay draws this; the old trace dropped it.
                "waypoint_m": [round(float(value), 4)
                               for value in record.waypoint_m[:2]],
                "v": round(record.v, 4),
                "w": round(record.w, 4),
                "collided": bool(record.collided),
                # Which side it was touched on — ahead, at a shoulder, behind.
                # P5's FOV experiment turns on whether a contact was inside
                # the camera's view or outside it.
                "contact_bearing_deg": [round(value, 1) for value
                                        in record.contact_bearings_deg],
            })
            if self._declared_arrival_tick is None and driver.reached_goal:
                self._declared_arrival_tick = record.index

            yield record

            # Measured after the tick, on the pose the tick produced — the
            # record holds the pose the tick *started* from.
            if runner._reached(goal_xy):
                self._success = True
                break

        # The odometer has to include the leg of the last tick, which no record
        # holds: each one stores where its tick began.
        self._poses.append(body.pose)
        self._finished = True

    def result(self):
        """Score the finished episode. Iterate it first."""
        if not self._finished:
            raise RuntimeError(
                "this episode has not finished; iterate it to completion "
                "before asking what it scored.")

        runner = self.runner
        goal_xy = self.task.goal_xy
        ticks = len(self._collided)

        episode_metrics = metrics.EpisodeMetrics(
            checkpoint=self.checkpoint_name,
            task=self.task,
            seed=self.seed,
            driver=runner.policy.driver.label(),
            success=self._success,
            collision_ticks=sum(self._collided),
            collision_events=metrics.count_collision_events(self._collided),
            path_length_m=metrics.path_length(self._poses),
            final_geodesic_distance_m=runner._geodesic_distance(goal_xy),
            final_euclidean_distance_m=runner._euclidean_distance(goal_xy),
            ticks=ticks,
            seconds=ticks * runner.body.limits.dt,
            timeout_ticks=self.timeout_ticks,
            success_radius_m=runner.rules.success_radius_m,
            success_metric=runner.rules.success_metric,
            declared_arrival_tick=self._declared_arrival_tick,
            final_node=self._last_step.closest_node if self._last_step else 0,
        )

        trace = episode_metrics.as_row()
        trace["poses"] = [[round(value, 4) for value in pose]
                          for pose in self._poses]
        trace["ticks_log"] = self._ticks_log
        return EpisodeResult(episode_metrics, trace)


class EpisodeRunner:
    """Runs episodes for one checkpoint, in one already-open simulator.

    Holds no per-episode state: `episode()` is handed the task and the seed and
    returns an `Episode` that owns everything about that run, so the same
    runner scores a whole scene's worth of tasks without anything leaking from
    one to the next except the simulator itself (which `reset` clears) and
    numpy's global RNG (which `seed_episode` overwrites).

    There is no `floor` here any more: the floor is bound into `body.scene`
    when the simulator is opened, so it stopped being a parameter every layer
    forwarded without reading.
    """

    def __init__(self, policy, body, rules=None, seed_offset=0):
        self.policy = policy
        self.body = body
        self.rules = rules or EpisodeRules()
        # Added to every task's seed. 0 is the task's own noise. A non-zero
        # offset replays the same task under different diffusion noise — how P5
        # measured run-to-run variation, and how P7 scores several seeds per
        # task. Fair either way as long as every arm runs the same offsets
        # (plan §7). The seed actually used is in every row's `seed` column.
        self.seed_offset = int(seed_offset)

    def _euclidean_distance(self, goal_xy):
        """How far the agent is from the goal right now, in a straight line."""
        position = np.asarray(self.body.pose[:2], dtype=float)
        return float(np.linalg.norm(position - np.asarray(goal_xy, dtype=float)))

    def _geodesic_distance(self, goal_xy):
        """How far the agent is from the goal around the furniture, or None."""
        return self.body.scene.geodesic_distance(self.body.pose, goal_xy)

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

    def seed_for(self, task):
        """The seed this runner gives a task — the one its row will record."""
        return episode_seed(task, self.seed_offset)

    def episode(self, task, checkpoint_name, seed=None):
        """Build the episode for one task. Iterate it to run it.

        `seed` defaults to the task's own, which is what the fairness protocol
        wants; it is an argument only so a rerun can deliberately vary it.
        """
        return Episode(self, task, checkpoint_name,
                       self.seed_for(task) if seed is None else int(seed))
