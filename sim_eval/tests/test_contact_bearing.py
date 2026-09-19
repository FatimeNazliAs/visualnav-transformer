"""Pin which side of the robot a contact is on.

P5's field-of-view experiment asks one question of every collision: was it in
front of the robot, where the camera could see it, or at a shoulder, where a
58-degree camera could not? The answer is one `atan2` and one wrap, and both
are easy to get subtly wrong — a sign flip turns every left clip into a right
one, and a missing wrap reports a contact just behind the robot as +190
degrees, which no threshold will ever classify as "behind". Neither crashes.

No simulator: `contact_bearings_deg` is pure arithmetic on a pose and a point.

    ./sim_eval/run_tests.sh
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge import contact_bearings_deg  # noqa: E402

AT_ORIGIN_FACING_EAST = (0.0, 0.0, 0.0)


def bearing(point, pose=AT_ORIGIN_FACING_EAST):
    return contact_bearings_deg([point], pose)[0]


def test_dead_ahead_is_zero():
    assert bearing((1.0, 0.0)) == pytest.approx(0.0)


def test_left_is_positive_the_ros_convention():
    """+w turns left in ROS, and NoMaD inherits it — so a bearing that called
    the left negative would contradict every other angle in the harness."""
    assert bearing((0.0, 1.0)) == pytest.approx(90.0)
    assert bearing((0.0, -1.0)) == pytest.approx(-90.0)


def test_the_bearing_follows_the_heading_not_the_map():
    """Facing north, a point to the map's east is on the robot's right."""
    facing_north = (0.0, 0.0, math.pi / 2)
    assert bearing((1.0, 0.0), facing_north) == pytest.approx(-90.0)


def test_behind_wraps_into_minus_180_to_180():
    facing_slightly_left = (0.0, 0.0, math.radians(10))
    value = bearing((-1.0, -0.001), facing_slightly_left)
    assert -180.0 <= value <= 180.0
    assert abs(value) == pytest.approx(170.0, abs=0.1)


def test_the_robots_position_is_subtracted_first():
    assert bearing((5.0, 3.0), (4.0, 2.0, 0.0)) == pytest.approx(45.0)


def test_no_contacts_is_no_bearings():
    assert contact_bearings_deg([], AT_ORIGIN_FACING_EAST) == []

