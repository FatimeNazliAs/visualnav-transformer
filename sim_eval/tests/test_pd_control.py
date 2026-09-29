"""Pin the PD controller to the real robot's behaviour.

The bridge's whole claim is that the steering is the deployment steering
(plan decision E). That claim is only worth anything if a later change cannot
quietly alter it, so the deployment mapping from waypoint to (v, w) is
pinned here — including the limits coming from the robot's own config file
rather than from numbers typed into sim_eval.

Needs no GPU, no iGibson and no checkpoint — but run it in the container,
which is where pytest and numpy live:

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pd_control  # noqa: E402

LIMITS = pd_control.RobotLimits.from_config()


def test_limits_come_from_the_deployment_robot_config():
    """These are the LoCoBot's numbers; decision E says mirror them exactly."""
    assert LIMITS.max_v == 0.2
    assert LIMITS.max_w == 0.4
    assert LIMITS.frame_rate == 4
    assert LIMITS.dt == 0.25


def test_straight_ahead_waypoint_gives_v_dx_over_dt_and_no_turn():
    v, w = pd_control.pd_controller(np.array([0.04, 0.0]), LIMITS)
    assert v == pytest.approx(0.04 / LIMITS.dt)
    assert w == pytest.approx(0.0)


def test_waypoint_to_the_left_turns_left():
    """+y is left, and a left turn is +w — the convention NoMaD was trained in."""
    _v, w = pd_control.pd_controller(np.array([0.04, 0.02]), LIMITS)
    assert w > 0


def test_waypoint_to_the_right_turns_right():
    _v, w = pd_control.pd_controller(np.array([0.04, -0.02]), LIMITS)
    assert w < 0


def test_velocity_is_clipped_to_the_robot_limits():
    v, w = pd_control.pd_controller(np.array([10.0, 10.0]), LIMITS)
    assert v == LIMITS.max_v
    assert w == LIMITS.max_w


def test_reverse_is_never_commanded():
    """`np.clip(v, 0, MAX_V)`: the deployment controller cannot back up."""
    v, _w = pd_control.pd_controller(np.array([-1.0, 0.0]), LIMITS)
    assert v == 0


def test_sideways_waypoint_spins_in_place():
    """The dx-near-zero branch: turn at pi/(2*dt), clipped to max_w."""
    v, w = pd_control.pd_controller(np.array([0.0, 0.02]), LIMITS)
    assert v == 0
    assert w == LIMITS.max_w


def test_four_dim_waypoint_uses_heading_only_when_the_step_is_zero():
    """learn_angle checkpoints emit (dx, dy, hx, hy); this branch is untouched."""
    v, w = pd_control.pd_controller(np.array([0.0, 0.0, 0.0, 1.0]), LIMITS)
    assert v == 0
    assert w == LIMITS.max_w


def test_waypoint_must_be_two_or_four_dimensional():
    with pytest.raises(AssertionError):
        pd_control.pd_controller(np.array([1.0, 2.0, 3.0]), LIMITS)


@pytest.mark.parametrize("theta,expected", [
    (0.0, 0.0),
    (np.pi / 2, np.pi / 2),
    (3 * np.pi, -np.pi),
    (-3 * np.pi / 2, np.pi / 2),
])
def test_clip_angle_wraps_into_plus_minus_pi(theta, expected):
    assert pd_control.clip_angle(theta) == pytest.approx(expected)
