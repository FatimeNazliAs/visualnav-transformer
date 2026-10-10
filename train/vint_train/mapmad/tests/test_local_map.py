"""local_map.py: centre, rotation sign and the first-seen cut (numpy only; runs in both containers)."""

import math

import numpy as np

from vint_train.mapmad.local_map import cell_centres, known_at, local_map

RES = 0.1


def drive_map(first_seen_obstacle: np.ndarray, first_seen_explored: np.ndarray = None) -> dict:
    """A 20 m x 20 m map with origin (-10, -10): cell [i, j] centre = (-9.95 + 0.1 i, -9.95 + 0.1 j)."""
    explored = first_seen_explored if first_seen_explored is not None else np.full_like(first_seen_obstacle, -1)
    return {"obstacle_first_seen": first_seen_obstacle, "explored_first_seen": explored,
            "origin": np.array([-10.0, -10.0]), "resolution": np.float64(RES)}


def one_cell(x: float, y: float, frame: int = 0) -> np.ndarray:
    grid = np.full((200, 200), -1, np.int32)
    grid[int(math.floor((x + 10.0) / RES)), int(math.floor((y + 10.0) / RES))] = frame
    return grid


def where(layer: np.ndarray):
    rows, cols = np.nonzero(layer)
    return list(zip(rows.tolist(), cols.tolist()))


def test_centre_is_between_the_middle_cells():
    forward, left = cell_centres(64, RES)
    assert np.isclose(forward[31, 0], 0.05) and np.isclose(forward[32, 0], -0.05)  # robot between rows 31 and 32
    assert np.isclose(left[0, 31], 0.05) and np.isclose(left[0, 32], -0.05)
    assert np.isclose(forward[0, 0], 3.15) and np.isclose(left[0, 0], 3.15)  # top-left = ahead-left


def test_obstacle_ahead_is_up_and_left_is_left():
    # robot at the origin facing +X (yaw 0): a cell 1 m ahead lands 10 rows above the centre
    m = drive_map(one_cell(1.05, 0.05))
    assert where(local_map(m, 0, (0.0, 0.0, 0.0))[0]) == [(21, 31)]
    # 1 m to the robot's left (+Y) lands 10 columns left of the centre
    m = drive_map(one_cell(0.05, 1.05))
    assert where(local_map(m, 0, (0.0, 0.0, 0.0))[0]) == [(31, 21)]


def test_rotation_sign():
    # facing +Y (yaw +90 deg, a left turn): the cell at +Y is now ahead, the cell at +X is on the right
    pose = (0.0, 0.0, math.pi / 2)
    assert where(local_map(drive_map(one_cell(0.05, 1.05)), 0, pose)[0]) == [(21, 32)]
    assert where(local_map(drive_map(one_cell(1.05, -0.05)), 0, pose)[0]) == [(32, 42)]


def test_robot_offset_and_outside_map():
    m = drive_map(one_cell(5.05, 3.05))
    assert where(local_map(m, 0, (4.0, 3.0, 0.0))[0]) == [(21, 31)]
    far = local_map(m, 0, (50.0, 50.0, 0.0))
    assert far.shape == (2, 64, 64) and far.dtype == np.float32 and far.sum() == 0.0


def test_first_seen_cut():
    first = np.full((200, 200), -1, np.int32)
    first[110, 100], first[115, 100], first[120, 100] = 0, 5, 9
    m = drive_map(first, first.copy())
    counts = [int(local_map(m, t, (0.0, 0.0, 0.0))[0].sum()) for t in (0, 4, 5, 8, 9, 100)]
    assert counts == [1, 1, 2, 2, 3, 3]
    assert np.array_equal(local_map(m, 5, (0.0, 0.0, 0.0))[1], local_map(m, 5, (0.0, 0.0, 0.0))[0])
    assert not known_at(np.array([-1]), 10)[0]
