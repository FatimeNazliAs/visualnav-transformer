"""The virtual LIMO: geometry helpers (no simulator) and, with the HM3D mount, camera height and motion."""

import math

import numpy as np
import pytest

from mapmad_sim import config
from mapmad_sim.robot import (RobotSpec, arc_step, clip_command, forward, horizontal_distance, wrap_angle,
                              yaw_towards)

HOME = "00800-TEEsavR23oF"
needs_hm3d = pytest.mark.skipif(not config.hm3d_scene("minival", HOME).exists(), reason="HM3D mount (/hm3d) missing")


def spec(**overrides) -> RobotSpec:
    return RobotSpec.from_config(config.robot(), **overrides)


def test_spec_reads_robot_config():
    s = spec()
    assert (s.hfov_deg, s.camera_height_m, s.camera_forward_m, s.pitch_deg) == (66.5, 0.18, 0.084, -2.0)
    assert dict(s.navmesh)["agent_radius_m"] == 0.195 and dict(s.navmesh)["agent_max_climb_m"] == 0.05
    assert (s.max_v, s.max_w, s.dt, s.width, s.height) == (0.2, 0.4, 0.25, 320, 240)
    assert s.camera_wall_margin_m == 0.02
    assert spec(hfov_deg=120.0).hfov_deg == 120.0


def test_heading_conventions():
    assert np.allclose(forward(0.0), [0, 0, -1])  # yaw 0 looks along -z
    assert np.allclose(forward(math.pi / 2), [-1, 0, 0])  # positive yaw turns left (towards -x)
    assert yaw_towards([0, 0, 0], [0, 5, -2]) == pytest.approx(0.0)
    assert yaw_towards([1, 0, 1], [0, 0, 1]) == pytest.approx(math.pi / 2)
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert horizontal_distance([0, 5, 0], [3, -1, 4]) == pytest.approx(5.0)


def test_arc_step_is_exact():
    """10 arc sub-steps of 25 ms land on the circle of radius v / w, whatever the split."""
    v, w, yaw = 0.2, 0.4, 0.3
    one = arc_step(yaw, v, w, 0.25)
    ten = sum(arc_step(yaw + w * 0.025 * i, v, w, 0.025) for i in range(10))
    assert np.allclose(one, ten, atol=1e-12)
    assert np.linalg.norm(one) == pytest.approx(2 * v / w * math.sin(w * 0.25 / 2))  # chord length
    assert np.allclose(arc_step(yaw, v, 0.0, 0.25), 0.05 * forward(yaw))


def test_commands_are_clipped_to_limits():
    assert clip_command(spec(), 1.0, -2.0) == (0.2, -0.4)
    assert clip_command(spec(), -0.1, 0.0) == (0.0, 0.0)  # no reversing
    assert clip_command(spec(), 0.1, 0.3) == (0.1, 0.3)


@pytest.fixture(scope="module")
def robot():
    from mapmad_sim.robot import LimoSim

    with LimoSim("minival", HOME, spec(), semantic=True) as r:
        yield r


def open_spot(robot, seed: int) -> np.ndarray:
    robot.pathfinder.seed(seed)
    while True:
        p = np.array(robot.pathfinder.get_random_navigable_point())
        if np.isfinite(p).all() and robot.pathfinder.distance_to_closest_obstacle(p.astype(np.float32), 1.0) > 0.4:
            return p


@needs_hm3d
def test_camera_sits_on_real_floor_height(robot):
    assert len(robot.navmesh_sha256) == 64
    assert robot.pathfinder.nav_mesh_settings.agent_radius == pytest.approx(0.195)
    robot.place(open_spot(robot, 3), 0.0)
    assert 0.0 < robot.floor_below_feet < 0.3  # the navmesh floats above the floor
    assert robot.camera_height_above_floor == pytest.approx(0.18, abs=1e-4)
    assert robot.observe()["rgb"].shape == (240, 320, 3)


@needs_hm3d
def test_straight_drive_moves_5cm_per_step(robot):
    robot.place(open_spot(robot, 4), 0.0)
    start = robot.position.copy()
    move = robot.drive(0.2, 0.0)
    if not move.collided:
        assert move.travelled_m == pytest.approx(0.05, abs=1e-3)
        assert horizontal_distance(start, robot.position) == pytest.approx(0.05, abs=1e-3)


@needs_hm3d
def test_turn_in_place_and_wall_collisions(robot):
    robot.place(open_spot(robot, 5), 0.0)
    move = robot.drive(0.0, 0.4)
    assert robot.yaw == pytest.approx(0.1, abs=1e-6) and move.travelled_m == 0.0 and not move.collided
    collided = [robot.drive(0.2, 0.0).collided for _ in range(400)]  # straight ahead hits a wall sooner or later
    assert any(collided)
    assert robot.camera_clear(robot.position, robot.yaw)  # the camera never went into the wall


@needs_hm3d
def test_robot_can_turn_on_the_spot_in_a_corner(robot):
    """Pressed into a wall, the robot can still turn away from it (turning into it is refused, like a body)."""
    turned = []
    for w in (0.4, -0.4):
        robot.place(open_spot(robot, 8), 0.0)
        for _ in range(400):
            robot.drive(0.2, 0.0)  # into the wall ahead
        before = robot.yaw
        for _ in range(8):
            robot.drive(0.0, w)
        turned.append(abs(wrap_angle(robot.yaw - before)))
    assert max(turned) == pytest.approx(0.8, abs=1e-6)  # 8 x 0.1 rad in the free direction


@needs_hm3d
def test_motion_is_deterministic(robot):
    def drive():
        robot.place(open_spot(robot, 6), 0.3)
        for i in range(30):
            robot.drive(0.2, 0.4 * math.sin(i / 3.0))
        return robot.position.copy(), robot.yaw

    (p1, y1), (p2, y2) = drive(), drive()
    assert np.array_equal(p1, p2) and y1 == y2
