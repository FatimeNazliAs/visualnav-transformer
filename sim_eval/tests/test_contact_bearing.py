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


# --- the camera tilt ------------------------------------------------------------
# P5's pitch experiment re-aims the camera on the render side. The sign is the
# whole risk: a tilt that went up when asked to go down would still render a
# perfectly plausible room, and the sweep would quietly measure the wrong thing.

import numpy as np  # noqa: E402

from bridge import URDF_CAMERA_TILT_DEG, tilted_camera_axes  # noqa: E402

LEVEL = np.eye(3)


def test_no_change_leaves_the_camera_where_it_was():
    view, up = tilted_camera_axes(LEVEL, 0.0)
    assert np.allclose(view, [1, 0, 0]) and np.allclose(up, [0, 0, 1])


def test_positive_is_down_the_urdfs_own_sign():
    view, up = tilted_camera_axes(LEVEL, 20.0)
    assert view[2] == pytest.approx(-math.sin(math.radians(20)))
    assert view[0] == pytest.approx(math.cos(math.radians(20)))
    assert up[0] > 0          # up leans forward when the camera looks down


def test_levelling_the_urdfs_camera_gives_a_level_view():
    """What the sweep's 0 means: the URDF's 20-degree-down camera, tilted back
    by 20, looks at the horizon."""
    tilt = math.radians(URDF_CAMERA_TILT_DEG)
    urdf_eyes = np.array([[math.cos(tilt), 0, math.sin(tilt)],
                          [0, 1, 0],
                          [-math.sin(tilt), 0, math.cos(tilt)]])
    view, up = tilted_camera_axes(urdf_eyes, 0.0 - URDF_CAMERA_TILT_DEG)
    assert np.allclose(view, [1, 0, 0]) and np.allclose(up, [0, 0, 1])


def test_the_tilt_follows_the_robots_heading():
    """Facing north, tilting down still points north, just lower."""
    facing_north = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    view, _up = tilted_camera_axes(facing_north, 30.0)
    assert view[0] == pytest.approx(0.0, abs=1e-12)
    assert view[1] > 0 and view[2] < 0
