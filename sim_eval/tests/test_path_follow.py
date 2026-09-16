"""Pin the reference-path follower, so a "cleanup" cannot change what a trail is.

The follower decides where the goal trail physically goes. If it silently
starts cutting corners or spinning, every topomap built afterwards is a
different task from the ones built before — and nothing downstream would say
so, because the trail would still look like a trail.

Needs no GPU, no iGibson and no checkpoint, but run it in the container, which
is where pytest and numpy live:

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import path_follow  # noqa: E402
import pd_control  # noqa: E402

LIMITS = pd_control.RobotLimits.from_config()
PARAMS = path_follow.FollowerParams()

# A straight path east, 0.2 m apart — the spacing iGibson's planner returns.
STRAIGHT = np.array([[x / 5.0, 0.0] for x in range(11)])


def test_target_skips_ahead_to_the_lookahead_distance():
    index = path_follow.advance_target(STRAIGHT, [0.0, 0.0], 0, lookahead_m=0.5)
    assert np.linalg.norm(STRAIGHT[index]) >= 0.5


def test_target_catches_up_to_a_robot_that_has_driven_past_it():
    """Otherwise the follower turns around to chase a point already behind it."""
    index = path_follow.advance_target(STRAIGHT, [1.0, 0.0], 0, lookahead_m=0.5)
    assert STRAIGHT[index][0] > 1.0
    assert np.linalg.norm(STRAIGHT[index] - [1.0, 0.0]) >= 0.5


def test_target_never_moves_backwards():
    """A path that doubles back near itself must not re-target a passed point."""
    index = path_follow.advance_target(STRAIGHT, [1.0, 0.0], 0, lookahead_m=0.5)
    back = path_follow.advance_target(STRAIGHT, [0.0, 0.0], index, lookahead_m=0.5)
    assert back == index


def test_target_stops_at_the_last_point():
    """At the end of the path there is nothing further out to aim at."""
    index = path_follow.advance_target(STRAIGHT, STRAIGHT[-1], 0, lookahead_m=0.5)
    assert index == len(STRAIGHT) - 1


def test_facing_the_target_drives_straight_at_full_speed():
    v, w = path_follow.follow_step((0.0, 0.0, 0.0), [1.0, 0.0], LIMITS, PARAMS)
    assert v == pytest.approx(LIMITS.max_v)
    assert w == pytest.approx(0.0)


def test_a_target_to_the_left_turns_left():
    """+y is left and a left turn is +w, the convention the bridge drives in."""
    _v, w = path_follow.follow_step((0.0, 0.0, 0.0), [1.0, 1.0], LIMITS, PARAMS)
    assert w > 0


def test_a_target_behind_turns_in_place_rather_than_arcing_into_furniture():
    v, w = path_follow.follow_step((0.0, 0.0, 0.0), [-1.0, 0.0], LIMITS, PARAMS)
    assert v == 0.0
    assert abs(w) == pytest.approx(LIMITS.max_w)


def test_commands_never_exceed_the_robots_own_limits():
    """Decision E: the reference path is driven inside the policy's envelope."""
    for yaw in np.linspace(-np.pi, np.pi, 37):
        for target in ([5.0, 0.0], [-5.0, 2.0], [0.1, -3.0]):
            v, w = path_follow.follow_step((0.0, 0.0, yaw), target, LIMITS, PARAMS)
            assert 0.0 <= v <= LIMITS.max_v
            assert abs(w) <= LIMITS.max_w


def test_a_near_target_slows_down_instead_of_overshooting():
    """Half a centimetre away, one tick at max_v would sail past it."""
    v, _w = path_follow.follow_step((0.0, 0.0, 0.0), [0.005, 0.0], LIMITS, PARAMS)
    assert v == pytest.approx(0.005 / LIMITS.dt)


def test_arrival_is_the_configured_radius():
    assert path_follow.has_arrived((0.0, 0.0, 0.0), [0.19, 0.0], 0.2)
    assert not path_follow.has_arrived((0.0, 0.0, 0.0), [0.21, 0.0], 0.2)


def test_initial_yaw_looks_down_the_path_not_at_the_first_point():
    """A path that starts with a turn: aiming at path[1] faces the wrong way."""
    path = np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.2], [0.5, 0.6], [0.5, 1.0]])
    yaw = path_follow.initial_yaw(path, lookahead_m=0.5)
    bearing_to_first = np.arctan2(*np.flip(path[1] - path[0]))
    assert yaw > bearing_to_first  # turned toward where the path actually goes
    assert yaw == pytest.approx(np.arctan2(*np.flip(path[3] - path[0])))


def test_initial_yaw_refuses_a_path_with_no_direction():
    with pytest.raises(ValueError):
        path_follow.initial_yaw(np.array([[0.0, 0.0]]), lookahead_m=0.5)


def test_a_straight_path_is_actually_followed_to_the_end():
    """The unit tests above are per-tick; this one closes the loop on them.

    Integrates the follower against a perfect (instantly obeyed) robot, which
    is not the simulator — but if the controller cannot drive a straight line
    with ideal physics, it will not drive one with real physics either.
    """
    pose = np.array([0.0, 0.0, np.pi])  # starts facing the wrong way
    index = 0
    for _ in range(200):
        if path_follow.has_arrived(pose, STRAIGHT[-1], PARAMS.arrival_radius_m):
            break
        index = path_follow.advance_target(
            STRAIGHT, pose[:2], index, PARAMS.lookahead_m)
        v, w = path_follow.follow_step(pose, STRAIGHT[index], LIMITS, PARAMS)
        pose = pose + np.array([v * np.cos(pose[2]), v * np.sin(pose[2]), w]) * LIMITS.dt
    assert path_follow.has_arrived(pose, STRAIGHT[-1], PARAMS.arrival_radius_m)
