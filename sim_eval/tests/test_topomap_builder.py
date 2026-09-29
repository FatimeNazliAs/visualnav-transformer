"""Pin the topomap builder's format and its start/goal rules.

Two things here must not drift. The **format**, because `navigate.py` reads a
trail by sorting `0.png, 1.png, ...` numerically and treating the last node as
the goal image — so a gap in the numbering, or a last node that is not the
goal, is a broken task that still looks like a working one. And the **sampler's
rules**, because "connected by the nav mesh, bounded geodesic length" (plan §5)
is what separates a task from a coincidence.

The simulator is faked, deliberately: the parts under test are arithmetic and
file layout, and pinning them must not need a GPU, a scene or 30 seconds.

    ./sim_eval/run_tests.sh
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bridge  # noqa: E402
import pd_control  # noqa: E402
import topomap_builder  # noqa: E402
from sim_scene import SimScene  # noqa: E402

LIMITS = pd_control.RobotLimits.from_config()


def fake_scene(*args, **kwargs):
    """A `SimScene` over the raw fake below — what the builder is handed now.

    The builder talks to the adapter rather than to iGibson, so the tests wrap
    the fake exactly as `SimBody` wraps the real thing. That also means these
    23 tests exercise `SimScene` itself for free.
    """
    return SimScene(FakeScene(*args, **kwargs), floor=0)


class FakeScene:
    """Just enough of iGibson's IndoorScene to exercise the sampler.

    `points` are handed out in order by `get_random_point`, and the geodesic
    length is the straight-line distance — the sampler's job is to accept and
    reject pairs, not to plan.
    """

    def __init__(self, points, off_mesh=()):
        self.scene_id = "Fake"
        self.floor_heights = [0.0]
        self._points = [np.asarray(point, dtype=float) for point in points]
        self._off_mesh = {tuple(point) for point in off_mesh}
        self._draws = 0

    def get_random_point(self, floor=None):
        point = self._points[self._draws % len(self._points)]
        self._draws += 1
        return floor, np.array([point[0], point[1], 0.0])

    def has_node(self, floor, world_xy):
        return tuple(np.asarray(world_xy, dtype=float)[:2]) not in self._off_mesh

    def get_shortest_path(self, floor, source, target, entire_path=False):
        source, target = np.asarray(source, float), np.asarray(target, float)
        steps = max(2, int(np.ceil(np.linalg.norm(target - source) / 0.2)) + 1)
        path = np.linspace(source, target, steps)
        return path, float(np.linalg.norm(target - source))


class FakeBody:
    """A robot that obeys (v, w) exactly, so the trail's shape is predictable."""

    def __init__(self, pose=(0.0, 0.0, 0.0), blocked_at=None):
        self.limits = LIMITS
        self.blocked_at = blocked_at
        self._pose = np.asarray(pose, dtype=float)
        self._frame_index = 0

    @property
    def pose(self):
        return tuple(float(value) for value in self._pose)

    def observe(self):
        # A distinct colour per frame, so a mis-ordered trail is visible in the
        # round-trip test rather than merely counted.
        self._frame_index += 1
        return Image.new("RGB", (8, 6), (self._frame_index, 0, 0))

    def command(self, v, w):
        """Obey exactly, unless a wall has been put in the way at `blocked_at`."""
        x, y, yaw = self._pose
        moved = np.array([x + v * np.cos(yaw) * self.limits.dt,
                          y + v * np.sin(yaw) * self.limits.dt,
                          yaw + w * self.limits.dt])
        if self.blocked_at is not None and moved[0] >= self.blocked_at:
            self._pose = np.array([x, y, moved[2]])
            return True
        self._pose = moved
        return False


def make_config(**overrides):
    values = {"scene_config": "configs/locobot_rs_bridge.yaml", "spacing_ticks": 4,
              "max_ticks": 200}
    values.update(overrides)
    return topomap_builder.TopomapConfig(**values)


def straight_path(length_m=2.0):
    steps = int(length_m / 0.2) + 1
    return np.array([[step * 0.2, 0.0] for step in range(steps)])


# --- config ------------------------------------------------------------------

def test_the_shipped_config_parses_and_points_at_a_real_world():
    config = topomap_builder.TopomapConfig.from_yaml()
    assert config.scene_config.exists()
    assert config.spacing_ticks >= 1
    assert config.sampler.min_geodesic_m < config.sampler.max_geodesic_m


def test_scene_id_override_reaches_the_world_config():
    assert make_config(scene_id="Bolton").world_config()["scene_id"] == "Bolton"
    assert make_config().world_config()["scene_id"] == "Rs"


def test_half_a_start_goal_pair_is_refused():
    with pytest.raises(topomap_builder.TopomapError):
        make_config(start=[0.0, 0.0])


def test_zero_spacing_is_refused():
    with pytest.raises(topomap_builder.TopomapError):
        make_config(spacing_ticks=0)


def test_a_zero_tick_drive_is_refused():
    """It would capture no nodes, and everything downstream reads the last one."""
    with pytest.raises(topomap_builder.TopomapError):
        make_config(max_ticks=0)


# --- the sampler -------------------------------------------------------------

def test_sampler_rejects_pairs_that_are_too_short_or_too_long():
    scene = fake_scene([[0.0, 0.0], [0.5, 0.0],      # 0.5 m — trivial
                       [0.0, 0.0], [20.0, 0.0],     # 20 m  — beyond the house
                       [0.0, 0.0], [4.0, 0.0]])     # 4 m   — accepted
    sampler = topomap_builder.SamplerParams(3.0, 8.0, max_attempts=10)
    _start, goal, _path, geodesic = topomap_builder.sample_start_goal(scene, sampler)
    assert geodesic == pytest.approx(4.0)
    assert goal[0] == pytest.approx(4.0)


def test_sampler_skips_endpoints_that_are_not_on_the_nav_mesh():
    scene = fake_scene([[0.0, 0.0], [4.0, 0.0], [0.0, 0.0], [5.0, 0.0]],
                      off_mesh=[(4.0, 0.0)])
    sampler = topomap_builder.SamplerParams(3.0, 8.0, max_attempts=10)
    _start, goal, _path, _geodesic = topomap_builder.sample_start_goal(scene, sampler)
    assert goal[0] == pytest.approx(5.0)


def test_sampler_gives_up_with_a_message_rather_than_looping_forever():
    scene = fake_scene([[0.0, 0.0], [0.1, 0.0]])
    sampler = topomap_builder.SamplerParams(3.0, 8.0, max_attempts=5)
    with pytest.raises(topomap_builder.TopomapError):
        topomap_builder.sample_start_goal(scene, sampler)


def test_a_configured_pair_is_used_as_given_and_never_sampled():
    scene = fake_scene([[9.0, 9.0]])
    config = make_config(start=[0.0, 0.0], goal=[3.0, 0.0])
    start, goal, _path, geodesic = topomap_builder.resolve_start_goal(scene, config)
    assert list(start) == [0.0, 0.0] and list(goal) == [3.0, 0.0]
    assert geodesic == pytest.approx(3.0)


def test_a_configured_pair_off_the_nav_mesh_is_refused():
    scene = fake_scene([[0.0, 0.0]], off_mesh=[(3.0, 0.0)])
    config = make_config(start=[0.0, 0.0], goal=[3.0, 0.0])
    with pytest.raises(topomap_builder.TopomapError):
        topomap_builder.resolve_start_goal(scene, config)


# --- the trail ---------------------------------------------------------------

def test_nodes_are_spaced_by_the_configured_edge_length():
    trail = topomap_builder.drive_reference_path(
        FakeBody(), straight_path(2.0), make_config(spacing_ticks=4))
    assert trail.arrived
    # Every node but the appended goal sits on a spacing boundary.
    assert [node["tick"] for node in trail.nodes[:-1]] == \
        list(range(0, trail.nodes[-2]["tick"] + 1, 4))


def test_spacing_one_keeps_every_tick():
    trail = topomap_builder.drive_reference_path(
        FakeBody(), straight_path(1.0), make_config(spacing_ticks=1))
    assert len(trail.nodes) == trail.ticks


def test_the_last_node_is_always_the_goal_view():
    """navigate.py treats the last node as the goal, whatever the modulo said."""
    path = straight_path(2.0)
    for spacing in (1, 3, 4, 7):
        trail = topomap_builder.drive_reference_path(
            FakeBody(), path, make_config(spacing_ticks=spacing))
        assert trail.arrived
        assert trail.nodes[-1]["tick"] == trail.ticks - 1
        assert np.linalg.norm(
            np.asarray(trail.nodes[-1]["pose"][:2]) - path[-1]) <= 0.2


def test_node_zero_is_the_start_pose():
    trail = topomap_builder.drive_reference_path(
        FakeBody(pose=(0.0, 0.0, 0.0)), straight_path(1.0), make_config())
    assert trail.nodes[0]["tick"] == 0
    assert trail.nodes[0]["pose"] == [0.0, 0.0, 0.0]


def test_a_clean_arrival_is_accepted():
    trail = topomap_builder.drive_reference_path(
        FakeBody(), straight_path(2.0), make_config())
    assert trail.rejection(topomap_builder.AcceptanceParams()) is None


# --- rejecting a trail that is not a task ------------------------------------

def test_a_wedged_robot_is_caught_early_instead_of_burning_max_ticks():
    """Rs plans through furniture; a stuck drive must cost seconds, not minutes."""
    config = make_config(max_ticks=400, acceptance={"stuck_ticks": 20})
    trail = topomap_builder.drive_reference_path(
        FakeBody(blocked_at=1.0), straight_path(4.0), config)
    assert trail.stuck
    assert trail.ticks < 60
    assert "progress" in trail.rejection(config.acceptance)


def test_turning_in_place_is_progress_and_not_being_stuck():
    """Lining up with the path takes up to 7.9 s of pure rotation at max_w.

    Counting only translation rejected most of Rs: every start pose facing away
    from its path looked wedged before it had finished turning toward it.
    """
    config = make_config(max_ticks=60, acceptance={"stuck_ticks": 20})
    # The path runs back the way the robot is facing, so it must turn ~180 deg
    # before it can drive at all.
    path = np.array([[-step * 0.2, 0.0] for step in range(16)])
    trail = topomap_builder.drive_reference_path(FakeBody(pose=(0.0, 0.0, 0.0)),
                                                 path, config)
    assert not trail.stuck


def test_running_out_of_ticks_is_rejected_and_says_how_far_short():
    config = make_config(max_ticks=10, acceptance={"stuck_ticks": 999})
    trail = topomap_builder.drive_reference_path(
        FakeBody(), straight_path(50.0), config)
    assert not trail.arrived and not trail.stuck
    assert trail.ticks == 10
    assert "ran out of ticks" in trail.rejection(config.acceptance)


def test_a_drive_that_scrapes_the_furniture_is_rejected():
    """The reference path is meant to be the perfect path, not a survivable one."""
    trail = topomap_builder.Trail(
        nodes=[{"node": 0}], poses=[(0.0, 0.0, 0.0)], arrived=True, stuck=False,
        collision_ticks=3, goal_distance_m=0.1)
    assert "collided" in trail.rejection(topomap_builder.AcceptanceParams())
    assert trail.rejection(
        topomap_builder.AcceptanceParams(max_collision_ticks=3)) is None


# --- the format --------------------------------------------------------------

def test_the_written_trail_is_what_the_bridge_reads_back(tmp_path):
    """The acceptance test, in miniature: node 10 must not sort before node 2."""
    nodes = topomap_builder.drive_reference_path(
        FakeBody(), straight_path(4.0), make_config(spacing_ticks=1)).nodes
    assert len(nodes) > 10, "needs a two-digit node to test the sort order"

    topomap_builder.write_topomap(nodes, tmp_path)
    loaded = bridge.load_topomap(tmp_path)

    assert len(loaded) == len(nodes)
    assert [frame.getpixel((0, 0))[0] for frame in loaded] == \
        [node["frame"].getpixel((0, 0))[0] for node in nodes]


def test_writing_a_shorter_trail_removes_the_previous_one(tmp_path):
    """A leftover node from a longer run would silently extend the new trail."""
    long_nodes = topomap_builder.drive_reference_path(
        FakeBody(), straight_path(4.0), make_config(spacing_ticks=1)).nodes
    topomap_builder.write_topomap(long_nodes, tmp_path)

    short_nodes = topomap_builder.drive_reference_path(
        FakeBody(), straight_path(1.0), make_config(spacing_ticks=1)).nodes
    topomap_builder.write_topomap(short_nodes, tmp_path)

    assert len(bridge.load_topomap(tmp_path)) == len(short_nodes)


def test_metadata_carries_what_p3_needs_to_score_a_rollout(tmp_path):
    config = make_config(spacing_ticks=4)
    path = straight_path(3.0)
    trail = topomap_builder.drive_reference_path(FakeBody(), path, config)
    intrinsics = {"width": 640, "height": 480, "vertical_fov_deg": 45.0,
                  "intrinsic_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}

    metadata = topomap_builder.build_metadata(
        config, "Rs", trail, path, 3.0, intrinsics, LIMITS)
    (tmp_path / topomap_builder.METADATA_NAME).write_text(json.dumps(metadata))

    loaded = topomap_builder.load_metadata(tmp_path)
    assert loaded["scene"]["id"] == "Rs"
    assert loaded["geodesic_length_m"] == pytest.approx(3.0)
    assert loaded["spacing"]["ticks_per_node"] == 4
    assert loaded["spacing"]["seconds_per_node"] == pytest.approx(1.0)
    assert loaded["camera"]["width"] == 640
    assert set(loaded["start_pose"]) == {"x", "y", "yaw"}
    assert set(loaded["goal_pose"]) == {"x", "y", "yaw"}
    assert len(loaded["nodes"]) == len(trail.nodes)


def test_missing_metadata_says_how_to_fix_it(tmp_path):
    with pytest.raises(topomap_builder.TopomapError):
        topomap_builder.load_metadata(tmp_path)
