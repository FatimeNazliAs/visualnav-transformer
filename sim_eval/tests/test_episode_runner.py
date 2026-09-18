"""Pin when an episode ends, and how long it is allowed to take.

The stop conditions *are* the measurement. Ending an episode a tick early stops
the odometer early and inflates its SPL; ending it on the model's own claim of
arrival lets a lost agent score; ending it on a collision quietly turns the
collision metric into a second success metric. None of those crash, and none of
them are visible in the resulting table — so they are pinned here, against a
simulator and a policy faked well enough that the whole file runs in a second.

    ./sim_eval/run_tests.sh
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import episode_runner  # noqa: E402
import pd_control  # noqa: E402
import task_set  # noqa: E402
from driver import DriverConfig  # noqa: E402
from nomad_policy import PolicyStep  # noqa: E402
from sim_scene import SimScene  # noqa: E402

LIMITS = pd_control.RobotLimits.from_config()

# The waypoint the fake policy always returns, in metres: straight ahead, far
# enough that the PD controller saturates at max_v. One tick is then exactly
# max_v * dt = 0.05 m, which makes every distance below countable by hand.
STRAIGHT_AHEAD = [0.05, 0.0]
METRES_PER_TICK = LIMITS.max_v * LIMITS.dt

# Deliberately not a multiple of the 0.05 m step. A radius that the agent lands
# exactly on makes the arrival tick a question about floating-point accumulation
# rather than about the rule under test — 80 steps of 0.05 m is 4.000000000000006,
# which is outside a radius of exactly 1 m.
SUCCESS_RADIUS_M = 0.875
WIDE_RADIUS_M = 1.875

# Goal at 5 m, 0.05 m per tick: the agent crosses into a 0.875 m radius on its
# 83rd tick (83 * 0.05 = 4.15 m, leaving 0.85 m), and into a 1.875 m one on its
# 63rd (3.15 m, leaving 1.85 m).
ARRIVAL_TICK = 83
WIDE_ARRIVAL_TICK = 63


class NoPath(Exception):
    """Stands in for networkx's NetworkXNoPath, matched by name (not by class,
    because networkx is iGibson's dependency and not imported here)."""


NoPath.__name__ = "NetworkXNoPath"


class FakeScene:
    """Answers the one question an episode asks a scene: how far to the goal.

    `detour` is how much longer the way round the furniture is than the way
    through it. 1.0 is an empty room; anything above it is what makes the two
    success rules disagree.
    """

    def __init__(self, detour=1.0, raises=None):
        self.floor_heights = [0.0]
        self.detour = detour
        self.raises = raises
        self.queries = 0

    def get_shortest_path(self, floor, source, target, entire_path=False):
        self.queries += 1
        if self.raises is not None:
            raise self.raises
        straight = float(np.linalg.norm(
            np.asarray(target, dtype=float) - np.asarray(source, dtype=float)))
        return np.array([source, target]), straight * self.detour


class FakeBody:
    """A robot that drives straight east at whatever speed it is commanded.

    Yaw and w are ignored on purpose: the runner's job is to decide when to
    stop and what to count, and a body whose position is `ticks * 0.05 m` makes
    every number in the assertions checkable by hand.
    """

    def __init__(self, scene=None, collide_on=()):
        self.limits = LIMITS
        # The world half of the adapter, as `SimBody` exposes it. The raw fake
        # is wrapped in a real `SimScene`, so the geodesic rule that decides
        # success is tested through the code that actually runs it.
        self.scene = SimScene(scene or FakeScene(), floor=0)
        self.collide_on = set(collide_on)
        self.ticks = 0
        self.resets = 0
        self._x = 0.0

    @property
    def pose(self):
        return (self._x, 0.0, 0.0)

    def reset(self):
        self.resets += 1
        self.ticks = 0
        self._x = 0.0

    def place(self, position, orientation):
        self._x = float(position[0])

    def observe(self):
        return Image.new("RGB", (8, 6))

    def command(self, v, w):
        collided = self.ticks in self.collide_on
        self._x += float(v) * self.limits.dt
        self.ticks += 1
        return collided


class FakePolicy:
    """Always points straight ahead, and claims whatever node it is told to."""

    def __init__(self, context_size=3, claims_goal_from=None, distances=()):
        self.spec = types.SimpleNamespace(context_size=context_size)
        # What the distance head "said" this tick, by window offset. Empty by
        # default: most of these tests are about when an episode stops, and a
        # step with no scores is the blank case the trace has to survive.
        self.distances = list(distances)
        self.model_params = {"normalize": False}
        # A real one: the row's provenance column comes off it.
        self.driver = DriverConfig()
        self.claims_goal_from = claims_goal_from
        self.calls = 0

    def encode_topomap(self, topomap):
        return list(range(len(topomap)))

    def act(self, context_frames, encoded_topomap, closest_node, goal_node):
        node = closest_node
        if self.claims_goal_from is not None and self.calls >= self.claims_goal_from:
            node = goal_node
        self.calls += 1
        # A real `PolicyStep`, not a stand-in: the trace reads the distance
        # head back out of it (`closest_distance`, `subgoal_distance`), and a
        # fake that merely carries the same field names would not exercise the
        # window arithmetic those readings depend on.
        return PolicyStep(
            waypoint=list(STRAIGHT_AHEAD), closest_node=node,
            subgoal_node=min(node + 1, goal_node),
            distances=self.distances, samples=[])


@pytest.fixture
def task(tmp_path):
    """A real `task_set.Task` over a real (tiny) topomap directory on disk.

    Built rather than faked so the metadata contract P2 writes and P3 reads is
    exercised: goal pose, geodesic length, node list, start pose.
    """
    return make_task(tmp_path)


def make_task(directory, goal_x=5.0, geodesic_m=5.0, nodes=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(nodes):
        Image.new("RGB", (8, 6)).save(directory / "{}.png".format(index))
    metadata = {
        "start_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
        "goal_pose": {"x": goal_x, "y": 0.0, "yaw": 0.0},
        "planned_goal_xy": [goal_x + 0.3, 0.0],
        "geodesic_length_m": geodesic_m,
        "nodes": [{"node": index} for index in range(nodes)],
    }
    return task_set.Task("Rs_00", "Rs", directory, seed=1000, metadata=metadata)


def make_episode(task, body=None, policy=None, rules=None):
    rules = rules or episode_runner.EpisodeRules(success_radius_m=SUCCESS_RADIUS_M)
    runner = episode_runner.EpisodeRunner(
        policy or FakePolicy(), body or FakeBody(), rules=rules)
    return runner.episode(task, "fake_checkpoint")


def run_episode(task, body=None, policy=None, rules=None, on_tick=None):
    """Drive an episode to its end the way a consumer does, and score it."""
    body = body or FakeBody()
    episode = make_episode(task, body=body, policy=policy, rules=rules)
    for record in episode:
        if on_tick is not None:
            on_tick(record)
    return episode.result(), body


# --- the timeout formula -----------------------------------------------------

def test_timeout_is_the_shortest_path_at_top_speed_times_the_slack():
    """ceil(slack * geodesic / (max_v * dt) + turn allowance / dt)."""
    rules = episode_runner.EpisodeRules(timeout_slack=4.0, turn_allowance_s=0.0)
    # 5 m at 0.05 m/tick is 100 ticks; four times that is 400.
    assert rules.timeout_ticks(5.0, LIMITS) == 400


def test_the_turn_allowance_defaults_to_a_half_turn_at_max_w():
    """Turning on the spot buys no distance, so it cannot come out of a budget
    that is measured in distance."""
    rules = episode_runner.EpisodeRules(timeout_slack=4.0)
    assert rules.turn_allowance(LIMITS) == pytest.approx(np.pi / LIMITS.max_w)
    assert rules.timeout_ticks(5.0, LIMITS) == 400 + int(
        np.ceil(np.pi / LIMITS.max_w / LIMITS.dt))


def test_a_longer_task_gets_a_longer_budget():
    rules = episode_runner.EpisodeRules()
    assert rules.timeout_ticks(8.0, LIMITS) > rules.timeout_ticks(3.0, LIMITS)


def test_a_very_short_task_still_gets_the_floor():
    rules = episode_runner.EpisodeRules(min_timeout_ticks=40, turn_allowance_s=0.0)
    assert rules.timeout_ticks(0.1, LIMITS) == 40


def test_a_nonsense_rule_is_refused_rather_than_scored():
    with pytest.raises(ValueError):
        episode_runner.EpisodeRules(timeout_slack=0.0)
    with pytest.raises(ValueError):
        episode_runner.EpisodeRules(success_radius_m=0.0)


# --- success -----------------------------------------------------------------

def test_an_episode_ends_the_tick_the_agent_enters_the_goal_radius(task):
    """Goal at 5 m, radius 0.875 m, 0.05 m per tick: arrival is the 83rd tick."""
    result, _body = run_episode(task)
    assert result.metrics.success
    assert result.metrics.ticks == ARRIVAL_TICK
    assert result.metrics.outcome == "success"


def test_the_path_includes_the_leg_of_the_final_tick(task):
    """Each record holds the pose its tick *started* from, so the last leg is
    only counted if the finishing pose is appended. Without it every SPL in the
    table is computed against a path one tick too short."""
    result, _body = run_episode(task)
    assert result.metrics.path_length_m == pytest.approx(ARRIVAL_TICK * METRES_PER_TICK)
    assert result.trace()["poses"][-1][0] == pytest.approx(4.15)


def test_success_is_measured_after_the_tick_not_before_it(task):
    """The pose in a record is where the tick began; success is about where it
    ended. Testing the earlier one would end every episode a tick late."""
    result, _body = run_episode(task)
    assert result.metrics.final_euclidean_distance_m == pytest.approx(0.85)


def test_a_wider_radius_ends_the_episode_sooner(task):
    result, _body = run_episode(
        task, rules=episode_runner.EpisodeRules(success_radius_m=WIDE_RADIUS_M))
    assert result.metrics.ticks == WIDE_ARRIVAL_TICK


# --- timeout -----------------------------------------------------------------

def test_an_agent_that_never_arrives_spends_exactly_its_budget(tmp_path):
    task = make_task(tmp_path, goal_x=100.0, geodesic_m=0.5)
    rules = episode_runner.EpisodeRules(turn_allowance_s=0.0)
    result, _body = run_episode(task, rules=rules)
    assert not result.metrics.success
    assert result.metrics.outcome == "timeout"
    assert result.metrics.ticks == rules.timeout_ticks(0.5, LIMITS)


def test_a_failed_episode_scores_zero_spl(tmp_path):
    task = make_task(tmp_path, goal_x=100.0, geodesic_m=0.5)
    result, _body = run_episode(task)
    assert result.metrics.spl == 0.0


# --- collisions are counted, never terminal (plan §6) ------------------------

def test_collisions_do_not_end_an_episode(task):
    body = FakeBody(collide_on=range(0, 40))
    result, _body = run_episode(task, body=body)
    assert result.metrics.success
    assert result.metrics.ticks == ARRIVAL_TICK
    assert result.metrics.collision_ticks == 40


def test_a_run_of_contact_is_counted_as_ticks_and_as_one_event(task):
    body = FakeBody(collide_on=(3, 4, 5, 20))
    result, _body = run_episode(task, body=body)
    assert result.metrics.collision_ticks == 4
    assert result.metrics.collision_events == 2


# --- the model's own opinion is not an end condition -------------------------

def test_the_policy_claiming_arrival_does_not_end_the_episode(tmp_path):
    """The bridge stops when the distance head localizes onto the last node —
    navigate.py's rule, and a claim about where the agent *thinks* it is.
    Ending on it would let a lost agent score, and would stop the odometer
    early on one that was merely early."""
    task = make_task(tmp_path, goal_x=100.0, geodesic_m=0.5)
    policy = FakePolicy(claims_goal_from=5)
    rules = episode_runner.EpisodeRules(turn_allowance_s=0.0)
    result, _body = run_episode(task, policy=policy, rules=rules)

    assert result.metrics.declared_arrival_tick == 5
    assert not result.metrics.success
    assert result.metrics.ticks == rules.timeout_ticks(0.5, LIMITS)


def test_an_episode_that_never_claims_arrival_records_none(task):
    result, _body = run_episode(task)
    assert result.metrics.declared_arrival_tick is None


# --- which distance the radius is measured with ------------------------------

def test_the_two_success_rules_disagree_where_the_furniture_is(task):
    """The case that decided the default. With a 30% detour around the
    furniture, the agent is inside a 0.875 m radius in a straight line four
    ticks before it is inside it geodesically — and the straight-line reading
    would call that arrived."""
    body = lambda: FakeBody(scene=FakeScene(detour=1.3))  # noqa: E731

    strict = run_episode(task, body=body(), rules=episode_runner.EpisodeRules(
        success_radius_m=SUCCESS_RADIUS_M, success_metric="geodesic"))[0]
    loose = run_episode(task, body=body(), rules=episode_runner.EpisodeRules(
        success_radius_m=SUCCESS_RADIUS_M, success_metric="euclidean"))[0]

    assert loose.metrics.success and strict.metrics.success
    assert loose.metrics.ticks == ARRIVAL_TICK
    assert strict.metrics.ticks == 87


def test_the_straight_line_check_gates_the_geodesic_one(task):
    """Being outside the radius in a straight line proves being outside it
    geodesically, so only the ticks already inside pay for an A* — the scene's
    graph grows by a node per query, and one per tick would be hundreds."""
    scene = FakeScene()
    result, _body = run_episode(task, body=FakeBody(scene=scene))
    assert result.metrics.ticks == ARRIVAL_TICK
    assert scene.queries <= 3


def test_an_unspecified_success_rule_is_refused():
    with pytest.raises(ValueError):
        episode_runner.EpisodeRules(success_metric="manhattan")


# --- the final distance ------------------------------------------------------

def test_the_final_distance_to_goal_is_geodesic(task):
    """Plan §6: a metre from the goal with a wall in between is not a metre.

    With a 20% detour the agent stops 0.70 m from the goal in a straight line
    and 0.84 m around the furniture, and it is the second number the table
    reports."""
    result, _body = run_episode(task, body=FakeBody(scene=FakeScene(detour=1.2)))
    assert result.metrics.final_euclidean_distance_m == pytest.approx(0.70)
    assert result.metrics.final_geodesic_distance_m == pytest.approx(0.84)


def test_an_agent_off_the_traversable_component_has_no_geodesic(task):
    """It climbed onto the furniture. That is a fact about the episode, and
    reporting it as 0 m would read as "arrived"."""
    scene = FakeScene(raises=NoPath("no path"))
    result, _body = run_episode(task, body=FakeBody(scene=scene),
                                rules=episode_runner.EpisodeRules(
                                    success_radius_m=SUCCESS_RADIUS_M,
                                    success_metric="euclidean"))
    assert result.metrics.final_geodesic_distance_m is None
    assert result.metrics.as_row()["final_geodesic_distance_m"] == ""


def test_an_agent_with_no_path_to_the_goal_has_not_reached_it(tmp_path):
    """Under the geodesic rule, standing on the sofa a metre from the goal is
    not arriving — there is no path from where it is to where it was sent."""
    task = make_task(tmp_path, goal_x=5.0, geodesic_m=0.5)
    scene = FakeScene(raises=NoPath("no path"))
    rules = episode_runner.EpisodeRules(success_radius_m=SUCCESS_RADIUS_M,
                                        turn_allowance_s=0.0)
    result, _body = run_episode(task, body=FakeBody(scene=scene), rules=rules)
    assert not result.metrics.success
    assert result.metrics.ticks == rules.timeout_ticks(0.5, LIMITS)


def test_a_scene_that_cannot_plan_at_all_is_an_error_not_a_blank(task):
    """Swallowing every exception would turn a broken scene into a table full
    of silently empty distances."""
    scene = FakeScene(raises=RuntimeError("the graph was never built"))
    with pytest.raises(RuntimeError):
        run_episode(task, body=FakeBody(scene=scene))


# --- provenance --------------------------------------------------------------

def test_an_episode_is_seeded_from_its_task_so_every_arm_faces_the_same_one(task):
    result, _body = run_episode(task)
    assert result.metrics.seed == task.seed


def test_the_trace_carries_the_path_and_the_per_tick_decisions(task):
    trace = run_episode(task)[0].trace()
    assert trace["task_id"] == "Rs_00"
    assert len(trace["poses"]) == trace["ticks"] + 1
    assert len(trace["ticks_log"]) == trace["ticks"]
    assert set(trace["ticks_log"][0]) == {
        "tick", "node", "subgoal", "dist_closest", "dist_subgoal",
        "waypoint_m", "v", "w", "collided"}


def test_the_trace_carries_the_waypoint_the_robot_steered_to(task):
    """P4's overlay draws it, and the trace used to drop it."""
    trace = run_episode(task)[0].trace()
    assert trace["ticks_log"][0]["waypoint_m"] == [0.05, 0.0]


def test_the_trace_carries_what_the_distance_head_read(task):
    """P5's question is why an episode failed, and the two nodes alone cannot
    answer it: the subgoal only advances when the head's reading for the
    closest node falls under `close_threshold`, so a trail that never advances
    and a head that is confident look identical without these two columns."""
    # Window [n-1, n, n+1] with the middle scored lowest, so the closest node
    # reads 1.0 and the subgoal beside it reads 4.0.
    policy = FakePolicy(distances=[7.0, 1.0, 4.0])
    tick = run_episode(task, policy=policy)[0].trace()["ticks_log"][0]
    assert (tick["dist_closest"], tick["dist_subgoal"]) == (1.0, 4.0)


def test_a_head_that_was_not_asked_leaves_the_trace_blank_rather_than_zero(task):
    """A distance of 0 reads as "arrived". A reading that does not exist is
    None, the same rule the table already follows for an unreachable goal."""
    tick = run_episode(task)[0].trace()["ticks_log"][0]
    assert tick["dist_closest"] is None and tick["dist_subgoal"] is None


# --- the episode is a stream -------------------------------------------------

def test_iterating_the_episode_is_what_runs_it(task):
    """One record per tick, in order, as it happens."""
    seen = []
    episode = make_episode(task)
    for record in episode:
        seen.append(record.index)
    assert seen == list(range(ARRIVAL_TICK))
    assert episode.result().metrics.ticks == ARRIVAL_TICK


def test_the_episode_keeps_no_records_and_so_no_frames(task):
    """Each record pins a decoded camera frame. An episode's worth is hundreds
    of megabytes and a scene's worth is gigabytes, so nothing may hold them —
    this is the whole reason the loop was inverted."""
    episode = make_episode(task)
    for _record in episode:
        pass
    result = episode.result()

    held = [name for name, value in vars(episode).items()
            if isinstance(value, list) and any(
                hasattr(item, "frame") for item in value)]
    assert held == []
    assert not hasattr(result, "records")


def test_a_consumer_sees_the_frame_the_model_saw(task):
    """The frame is on the stream, so a recorder can encode it and drop it."""
    frames = [record.frame for record in make_episode(task)]
    assert len(frames) == ARRIVAL_TICK
    assert frames[0].size == (8, 6)


def test_the_record_carries_the_policy_step_rather_than_a_copy_of_it(task):
    """`distances` and `samples` are what P4's and P5's overlays draw, and a
    field-by-field copy silently dropped them."""
    record = next(iter(make_episode(task)))
    assert record.step.closest_node == 0
    assert hasattr(record.step, "distances")
    assert hasattr(record.step, "samples")


def test_an_episode_runs_once(task):
    """Re-iterating would score half a second run against the first's tally."""
    episode = make_episode(task)
    for _record in episode:
        pass
    with pytest.raises(RuntimeError):
        for _record in episode:
            pass


def test_an_unfinished_episode_refuses_to_be_scored(task):
    """Scoring a partial run would report a timeout that never happened."""
    episode = make_episode(task)
    with pytest.raises(RuntimeError):
        episode.result()


def test_the_budget_is_known_before_the_episode_runs(task):
    """A consumer sizing a video needs it up front."""
    episode = make_episode(task)
    assert episode.timeout_ticks == episode_runner.EpisodeRules(
        success_radius_m=SUCCESS_RADIUS_M).timeout_ticks(5.0, LIMITS)
