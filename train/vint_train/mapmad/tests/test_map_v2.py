"""map_v2.py (Phase 8.1, G8 row 1): both windows' crop and rotation, far = summary of near, frontier on hand-made
maps, a false alarm fades, and the v1 crop is untouched (numpy only; runs in both containers)."""

import math

import numpy as np

from vint_train.mapmad.local_map import local_map
from vint_train.mapmad.map_v2 import (LAYERS_V2, TargetFusion, far_window, frontier, frontier_cells, label_clusters,
                                      layers_at, near_window, summarise, target_at, visited_mask)

RES = 0.1
ORIGIN = (-20.0, -20.0)  # 40 m x 40 m map: cell [i, j] centre = (-19.95 + 0.1 i, -19.95 + 0.1 j)
SHAPE = (400, 400)


def cell(x: float, y: float):
    return int(math.floor((x - ORIGIN[0]) / RES)), int(math.floor((y - ORIGIN[1]) / RES))


def layers_with(x: float, y: float, name: str = "obstacle", value: float = 1.0) -> dict:
    out = {n: np.zeros(SHAPE, np.float32) for n in LAYERS_V2}
    out[name][cell(x, y)] = value
    return out


def where(layer: np.ndarray):
    rows, cols = np.nonzero(layer)
    return list(zip(rows.tolist(), cols.tolist()))


# --- windows: crop and rotation ---------------------------------------------------------------------------------

def test_near_window_ahead_is_up_and_left_is_left():
    # robot at the origin facing +X: 1 m ahead = 10 rows above the centre (rows 31/32 straddle the robot)
    assert where(near_window(layers_with(1.05, 0.05), ORIGIN, RES, (0.0, 0.0, 0.0))[0]) == [(21, 31)]
    # 1 m to the left (+Y when facing +X) = 10 columns left of the centre
    assert where(near_window(layers_with(0.05, 1.05), ORIGIN, RES, (0.0, 0.0, 0.0))[0]) == [(31, 21)]


def test_near_window_rotates_with_the_robot():
    # the same world cell, robot turned left by 90 deg: what was ahead is now on the robot's right
    w = near_window(layers_with(1.05, 0.05), ORIGIN, RES, (0.0, 0.0, math.pi / 2))
    assert where(w[0]) == [(31, 42)]


def test_far_window_scale_and_rotation():
    # 10 m ahead -> 25 far cells (40 cm) above the centre; robot between far rows 31 and 32
    far = far_window(layers_with(10.05, 0.05), ORIGIN, RES, (0.0, 0.0, 0.0))
    assert where(far[0]) == [(6, 31)]
    far = far_window(layers_with(10.05, 0.05), ORIGIN, RES, (0.0, 0.0, -math.pi / 2))  # turned right: it is on the left
    assert where(far[0]) == [(32, 6)]  # 0.05 m behind the turning centre -> the row just below it
    assert far.shape == (4, 64, 64) and far.dtype == np.float32


def test_far_is_the_summary_of_near():
    rng = np.random.default_rng(0)
    layers = {"obstacle": rng.random(SHAPE) < 0.1, "explored": rng.random(SHAPE) < 0.6,
              "frontier": rng.random(SHAPE) < 0.05, "target": (rng.random(SHAPE) * (rng.random(SHAPE) < 0.02)).astype(np.float32)}
    pose = (1.234, -0.77, 0.6)
    near = near_window(layers, ORIGIN, RES, pose)
    far = far_window(layers, ORIGIN, RES, pose)
    centre = far[:, 24:40, 24:40]
    assert np.array_equal(centre, summarise(near))
    # the summary rules themselves on one block
    b = near[:, :4, :4]
    assert centre[0, 0, 0] == float(b[0].any()) and centre[2, 0, 0] == float(b[2].any())
    assert np.isclose(centre[1, 0, 0], b[1].mean()) and centre[3, 0, 0] == b[3].max()


def test_v1_crop_unchanged_with_new_features_off():
    rng = np.random.default_rng(1)
    first = {k: np.where(rng.random(SHAPE) < 0.3, rng.integers(0, 50, SHAPE), -1).astype(np.int32)
             for k in ("obstacle", "explored")}
    dm = {"obstacle_first_seen": first["obstacle"], "explored_first_seen": first["explored"],
          "origin": np.array(ORIGIN), "resolution": np.float64(RES)}
    pose = (0.3, -1.1, 2.2)
    v1 = local_map(dm, 25, pose)
    v2 = near_window(layers_at(dm, 25), ORIGIN, RES, pose)
    assert v1.tobytes() == v2[:2].tobytes()  # byte-identical layers 0-1


# --- frontier ---------------------------------------------------------------------------------------------------

def room(explored_cols: int) -> tuple:
    """A 10 x 20 room: walls on rows 0 / 9 and column 0; columns 1..explored_cols-1 explored, the rest unknown."""
    obstacle = np.zeros((10, 20), bool)
    obstacle[0, :] = obstacle[9, :] = obstacle[:, 0] = True
    explored = np.zeros((10, 20), bool)
    explored[:, :explored_cols] = True
    return obstacle, explored


def test_frontier_is_the_explored_free_edge_next_to_unknown():
    obstacle, explored = room(8)
    f = frontier(obstacle, explored)
    assert where(f) == [(i, 7) for i in range(1, 9)]  # column 7, between the walls (wall cells are never frontier)


def test_frontier_diagonal_contact_does_not_count_and_small_clusters_drop():
    obstacle = np.zeros((10, 10), bool)
    explored = np.ones((10, 10), bool)
    explored[5, 5] = False  # one unknown cell: 4 edge neighbours are frontier cells, one 8-connected cluster
    assert set(where(frontier_cells(obstacle, explored))) == {(4, 5), (6, 5), (5, 4), (5, 6)}
    assert frontier(obstacle, explored).sum() == 4  # 4 cells = kept (>= 4)
    explored[5, 5] = True
    explored[0, 0] = False  # corner hole: 2 frontier cells -> dropped
    assert frontier(obstacle, explored).sum() == 0


def test_cluster_labels_are_8_connected():
    m = np.zeros((5, 5), bool)
    m[0, 0] = m[1, 1] = m[2, 2] = True  # diagonal chain = one cluster
    m[4, 4] = True
    labels, n = label_clusters(m)
    assert n == 2 and labels[0, 0] == labels[2, 2] == 1 and labels[4, 4] == 2


def test_fully_known_room_has_no_frontier():
    obstacle, explored = room(20)
    explored[:, :] = True
    assert not frontier(obstacle, explored).any()



def test_trajectory_ring_drops_frontier_cells_near_the_past_path():
    # rules v2 1b: a 2 x 6 strip of frontier cells; positions next to its left end hide those cells, then the
    # remaining cells are clustered (a remainder < 4 cells drops out)
    obstacle = np.zeros((10, 10), bool)
    explored = np.ones((10, 10), bool)
    explored[4:6, 2:8] = False  # unknown block: frontier ring around it
    plain = frontier(obstacle, explored)
    near = visited_mask((10, 10), (0.0, 0.0), 0.1, np.array([[0.35, 0.15]]), 0.25)
    ring = frontier(obstacle, explored, visited=near)
    assert not (ring & near).any() and (plain & near).any()
    assert ring.sum() == (plain & ~near).sum()
    explored[:, :] = True
    explored[5, 5] = False  # one unknown cell: a 4-cell cluster; hiding cell (4, 5) leaves 3 -> the cluster drops
    one = visited_mask((10, 10), (0.0, 0.0), 0.1, np.array([[0.45, 0.55]]), 0.05)
    assert where(one) == [(4, 5)]
    assert frontier(obstacle, explored).sum() == 4 and frontier(obstacle, explored, visited=one).sum() == 0


# --- target fusion ----------------------------------------------------------------------------------------------

def test_false_alarm_fades():
    fus = TargetFusion((5, 5))
    seen = np.arange(25)
    fus.update(0, seen, {12: 0.6})  # false mark at cell 12
    assert np.isclose(fus.grid().flat[12], 0.6)
    fus.update(1, seen, {})  # seen again, no detection: mean (0.6 + 0) / 2
    assert np.isclose(fus.grid().flat[12], 0.3)
    for t in range(2, 6):
        fus.update(t, seen, {})
    assert np.isclose(fus.grid().flat[12], 0.1)  # 0.6 / 6 views


def test_fusion_only_counts_views_that_saw_the_cell():
    fus = TargetFusion((1, 4))
    fus.update(0, np.array([0, 1]), {1: 0.8})
    fus.update(1, np.array([2, 3]), {})  # cell 1 not in view: unchanged
    assert np.isclose(fus.grid()[0, 1], 0.8)
    fus.update(2, np.array([1]), {1: 0.4})
    assert np.isclose(fus.grid()[0, 1], 0.6)


def test_sparse_updates_rebuild_every_frame():
    rng = np.random.default_rng(3)
    fus = TargetFusion((6, 7))
    online = []
    for t in range(12):
        view = rng.choice(42, 20, replace=False)
        scores = {int(c): float(rng.uniform(0.3, 1.0)) for c in rng.choice(42, 3, replace=False)} if t % 3 else {}
        fus.update(t, view, scores)
        online.append(fus.grid().copy())
    u = fus.updates()
    for t in range(12):
        assert np.array_equal(target_at(u["cell"], u["frame"], u["value"], t, (6, 7)), online[t])
