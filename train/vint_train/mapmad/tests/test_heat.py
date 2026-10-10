"""heat.py + map_sample.py: where the heat lands (ahead / left / behind / diagonal / footprint), the shared pose,
the first-seen cut of the map layers, and perturbation off in eval. numpy only."""

import math

import numpy as np
import pytest

from vint_train.mapmad import heat as H
from vint_train.mapmad.local_map import cell_centres, local_to_world, world_to_local
from vint_train.mapmad.map_sample import MapSampleBuilder

RES, SIZE = 0.1, 64
ORIGIN = (0.0, 0.0, 0.0)  # robot at the origin facing +X


def peak(h: np.ndarray):
    r, c = np.unravel_index(int(np.argmax(h)), h.shape)
    return int(r), int(c)


def spot(x: float, y: float) -> H.Target:
    return H.Target("spot", (x, y))


def test_world_to_local_inverts_local_to_world():
    f, l = cell_centres(SIZE, RES)
    pose = (3.2, -1.7, 2.1)
    wx, wy = local_to_world(f, l, pose)
    f2, l2 = world_to_local(wx, wy, pose)
    assert np.allclose(f, f2) and np.allclose(l, l2)


def test_ahead_is_top_centre():
    h = H.draw_heat(spot(2.0, 0.0), ORIGIN)
    r, c = peak(h)
    assert r < 31 and c in (31, 32) and math.isclose(float(h.max()), 1.0, rel_tol=0.03)
    assert h.dtype == np.float32 and h.shape == (SIZE, SIZE) and 0.0 <= h.min() <= h.max() <= 1.0


def test_left_is_left_half():
    r, c = peak(H.draw_heat(spot(0.0, 2.0), ORIGIN))
    assert c < 31 and r in (31, 32)


def test_behind_and_outside_is_bottom_edge():
    t = spot(-8.0, 0.0)
    assert H.is_outside(t, ORIGIN, SIZE, RES)
    h = H.draw_heat(t, ORIGIN)
    r, c = peak(h)
    assert r >= SIZE - 4 and c in (31, 32)  # 0.3 m (3 cells) inside the bottom edge
    assert math.isclose(float(h.max()), 1.0, rel_tol=0.05)


def test_diagonal_outside_goes_to_the_correct_edge():
    # ahead-left, more left than ahead: the line leaves through the LEFT edge, in the upper half
    r, c = peak(H.draw_heat(spot(4.0, 8.0), ORIGIN))
    assert c <= 3 and r < 31
    # behind-right, more behind than right: leaves through the BOTTOM edge, in the right half
    r, c = peak(H.draw_heat(spot(-9.0, -3.0), ORIGIN))
    assert r >= SIZE - 4 and c > 32


def test_object_footprint_is_one_inside():
    # a 1 m x 0.6 m rectangle 1.5-2.5 m ahead, centred left-right
    t = H.Target("object", (2.0, 0.0), (1.5, 2.5, -0.3, 0.3))
    h = H.draw_heat(t, ORIGIN)
    f, l = cell_centres(SIZE, RES)
    inside = (f > 1.5) & (f < 2.5) & (np.abs(l) < 0.3)
    assert inside.sum() > 0 and np.all(h[inside] == 1.0)
    assert h[(f < 0)].max() < 0.01  # nothing behind the robot


def test_footprint_partly_inside_is_clipped_not_border():
    t = H.Target("object", (3.5, 0.0), (3.0, 4.5, -0.5, 0.5))  # crosses the top edge (3.2 m)
    assert not H.is_outside(t, ORIGIN, SIZE, RES)
    h = H.draw_heat(t, ORIGIN)
    assert h[0, 31] == 1.0 and h[0, 32] == 1.0


def test_target_from_meta_uses_habitat_frame():
    meta = {"target": {"kind": "object", "box_center": [1.0, 0.5, -2.0], "box_size": [0.4, 1.0, 0.2]}}
    t = H.target_from_meta(meta)
    assert t.centre == (2.0, -1.0)  # X = -z, Y = -x
    assert np.allclose(t.rect, (1.9, 2.1, -1.2, -0.8))
    meta = {"target": {"kind": "spot", "end_position": [-3.0, 0.1, 4.0]}}
    assert H.target_from_meta(meta).centre == (-4.0, 3.0)


class FakeStore:
    """A DriveStore stand-in: one drive, a target and a floor map."""

    def __init__(self, target: H.Target, floor: dict) -> None:
        self.targets = {"d": target}
        self._floor = floor

    def floor_map(self, drive: str) -> dict:
        return self._floor


def floor_with_cell(x: float, y: float, frame: int, radius: int = 0) -> dict:
    """Floor map with the cell under (x, y) (and `radius` cells around it) first seen at `frame`."""
    grid = np.full((200, 200), -1, np.int32)
    i, j = int(math.floor((x + 10.0) / RES)), int(math.floor((y + 10.0) / RES))
    grid[i - radius:i + radius + 1, j - radius:j + radius + 1] = frame
    return {"obstacle_first_seen": grid, "explored_first_seen": grid.copy(), "origin": np.array([-10.0, -10.0]),
            "resolution": np.float64(RES)}


def test_heat_uses_the_same_pose_as_the_map():
    # a 3 x 3-cell obstacle block centred on the spot target: for any pose the heat peak lies on the block
    x, y = 1.25, 0.85
    builder = MapSampleBuilder(FakeStore(spot(x, y), floor_with_cell(x, y, 0, radius=1)))
    for pose in [(0.0, 0.0, 0.0), (0.3, -0.2, 0.7), (-0.5, 0.4, -2.4), (0.1, 0.1, math.pi)]:
        m = builder.build("d", 0, pose, heat_on=True)
        assert 4 <= m[0].sum() <= 16 and m[0][peak(m[2])] == 1.0


def test_map_layers_respect_first_seen_cut_and_heat_off():
    builder = MapSampleBuilder(FakeStore(spot(1.0, 0.0), floor_with_cell(1.05, 0.05, 7)))
    assert builder.build("d", 6, ORIGIN, heat_on=False).sum() == 0.0
    m = builder.build("d", 7, ORIGIN, heat_on=False)
    assert m[0].sum() == 1.0 and m[1].sum() == 1.0 and m[2].sum() == 0.0
    assert m.dtype == np.float32 and m.shape == (3, SIZE, SIZE)


def test_perturbation_off_in_eval_and_each_kind_changes_the_heat():
    store = FakeStore(spot(1.5, 0.5), floor_with_cell(0.0, 0.0, 0))
    rng = np.random.default_rng(0)
    eval_builder = MapSampleBuilder(store, perturb_prob=0.0)
    assert all(eval_builder.draw_perturbation(rng) is None for _ in range(1000))
    train_builder = MapSampleBuilder(store, perturb_prob=0.3)
    draws = [train_builder.draw_perturbation(rng) for _ in range(6000)]
    share = sum(d is not None for d in draws) / len(draws)
    assert 0.27 < share < 0.33 and {d for d in draws if d} == {"shift", "blur", "false_blob"}
    clean = train_builder.heat("d", ORIGIN)
    for kind in ("shift", "blur", "false_blob"):
        h = train_builder.heat("d", ORIGIN, kind, np.random.default_rng(1))
        assert not np.array_equal(h, clean) and math.isclose(float(h.max()), 1.0, rel_tol=0.05)


def test_shift_distance_and_false_blob_distance():
    rng = np.random.default_rng(3)
    t = spot(1.0, 1.0)
    for _ in range(200):
        moved = H.perturb_target(t, rng, (0.3, 1.0))
        assert 0.3 - 1e-9 <= math.hypot(moved.centre[0] - 1.0, moved.centre[1] - 1.0) <= 1.0 + 1e-9
    base = H.draw_heat(t, ORIGIN)
    h = H.add_false_blob(base, (1.0, 1.0), rng, SIZE, RES, 0.3, 1.0)
    extra = (h > 0.95) & (base < 0.5)
    f, l = cell_centres(SIZE, RES)
    assert extra.any() and np.all(np.hypot(f[extra] - 1.0, l[extra] - 1.0) >= 0.85)


def test_rotate_180_moves_left_to_right():
    h = H.draw_heat(spot(0.0, 2.0), ORIGIN)
    r, c = peak(H.rotate_180(h))
    assert c > 32 and r in (31, 32)


@pytest.mark.parametrize("size", [64])
def test_border_point_is_inset_along_the_line(size):
    bf, bl = H.border_point(spot(-8.0, 0.0), ORIGIN, size, RES, 0.3)
    assert math.isclose(bf, -2.9, abs_tol=1e-9) and math.isclose(bl, 0.0, abs_tol=1e-9)
