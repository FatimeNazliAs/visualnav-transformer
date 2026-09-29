"""Pin the object-reach distance on hand-placed poses.

Object-reach success is "within 1 m of the nearest instance, around the
furniture", and every number in a word-goal table rests on it. Four things are
worth pinning, because each is a way to write a plausible distance that is
wrong:

  * **footprint, not centre** — a robot a step from a sofa's edge is near the
    sofa, however far its centre is.
  * **around, not through** — a wall between robot and object makes the
    distance the way round it.
  * **no leaking** — a free cell a straight half metre from a chair, but on the
    far side of the counter the chair stands at, is not next to the chair.
  * **nearest instance** — two chairs: the distance is to the nearer one, and
    `nearest_instance` names it.

Plus: a robot cut off from every instance gets no distance, not a number.

Hand-made maps at iGibson's 0.1 m — no GPU, no iGibson, no simulator.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import objects  # noqa: E402
from objects import DistanceField, GridMap, Instance, SceneObjects  # noqa: E402

RES = 0.1
SIZE = 60  # 6 m x 6 m, world -3..3
TOL = 0.08  # a cell's diagonal half-width, plus rounding


def open_map():
    return np.full((SIZE, SIZE), objects.FREE, dtype=np.uint8)


def block(trav, grid, xmin, xmax, ymin, ymax):
    """Mark every cell whose centre lies in the box as an obstacle."""
    for row in range(SIZE):
        for col in range(SIZE):
            x, y = grid.centre(row, col)
            if xmin <= x <= xmax and ymin <= y <= ymax:
                trav[row, col] = 0
    return GridMap(trav, RES)


def box_instance(instance_id, box, category="chair"):
    return Instance(instance_id, category, [box], view_from=[0.0, 0.0])


def with_obstacle_at_footprint(trav, instance):
    """Objects are obstacles on the map, as the Gibson houses' furniture is."""
    grid = GridMap(trav, RES)
    xmin, xmax, ymin, ymax = instance.boxes[0]
    return block(trav, grid, xmin - 0.2, xmax + 0.2, ymin - 0.2, ymax + 0.2)


def test_open_floor_is_straight_line_to_the_footprint():
    chair = box_instance("c", [0.5, 0.7, -0.1, 0.1])
    grid = with_obstacle_at_footprint(open_map(), chair)
    field = DistanceField(grid, [chair])
    assert field.distance([-1.0, 0.0]) == pytest.approx(1.5, abs=TOL)
    assert field.distance([0.6, -1.3]) == pytest.approx(1.2, abs=TOL)


def test_footprint_not_centre():
    sofa = box_instance("s", [-1.0, 1.0, 0.0, 0.8], category="sofa")
    grid = with_obstacle_at_footprint(open_map(), sofa)
    field = DistanceField(grid, [sofa])
    # 0.8 m from the front edge, 1.2 m from the centre (0, 0.4).
    robot = [0.0, -0.8]
    assert field.distance(robot) == pytest.approx(0.8, abs=TOL)
    assert np.linalg.norm(sofa.centre - robot) == pytest.approx(1.2)
    # A metre along the edge from the centre is still 0.8 m from the sofa.
    assert field.distance([0.9, -0.8]) == pytest.approx(0.8, abs=TOL)


def test_distance_goes_around_a_wall():
    chair = box_instance("c", [0.5, 0.7, -0.1, 0.1])
    trav = open_map()
    grid = with_obstacle_at_footprint(trav, chair)
    # A wall at x = 0 from the bottom of the map up to y = 1, gap above it.
    grid = block(trav, grid, -0.05, 0.05, -3.0, 1.0)
    field = DistanceField(grid, [chair])
    robot = [-0.5, 0.0]
    straight = chair.surface_distance(robot)
    assert straight == pytest.approx(1.0)
    # Round the wall's end at (0, 1.0): up ~1.1, across, down ~0.9.
    around = field.distance(robot)
    assert around > 2.0
    assert around == pytest.approx(
        np.hypot(0.5, 1.15) + np.hypot(0.5, 1.15 - 0.1), abs=0.25)


def test_no_leak_through_a_counter():
    # A chair at a counter: the counter is a 0.3 m slab 0.1 m past the chair.
    chair = box_instance("c", [-0.2, 0.2, 0.0, 0.4])
    trav = open_map()
    grid = with_obstacle_at_footprint(trav, chair)
    grid = block(trav, grid, -1.5, 1.5, 0.5, 0.8)
    field = DistanceField(grid, [chair])
    behind = [0.0, 1.1]  # 0.7 m straight from the chair, through the counter
    assert grid.is_free(behind)
    assert chair.surface_distance(behind) == pytest.approx(0.7)
    assert field.distance(behind) > 2.5  # round the counter's end, 1.5 m away


def test_nearest_of_two_instances():
    near = box_instance("near", [1.0, 1.2, -0.1, 0.1])
    far = box_instance("far", [-2.2, -2.0, -0.1, 0.1])
    trav = open_map()
    with_obstacle_at_footprint(trav, near)
    grid = with_obstacle_at_footprint(trav, far)
    objs = SceneObjects("Test", grid, [near, far])
    robot = [0.0, 0.0]
    assert objs.category_field("chair").distance(robot) == pytest.approx(1.0, abs=TOL)
    instance, distance = objs.nearest_instance("chair", robot)
    assert instance.id == "near"
    assert distance == pytest.approx(1.0, abs=TOL)
    # Past the midpoint the other one is nearer.
    instance, _ = objs.nearest_instance("chair", [-1.2, 0.0])
    assert instance.id == "far"


def test_cut_off_robot_has_no_distance():
    chair = box_instance("c", [0.5, 0.7, -0.1, 0.1])
    trav = open_map()
    grid = with_obstacle_at_footprint(trav, chair)
    # A closed room around (-2, -2).
    grid = block(trav, grid, -2.8, -1.2, -2.8, -2.7)
    grid = block(trav, grid, -2.8, -1.2, -1.3, -1.2)
    grid = block(trav, grid, -2.8, -2.7, -2.8, -1.2)
    grid = block(trav, grid, -1.3, -1.2, -2.8, -1.2)
    field = DistanceField(grid, [chair])
    assert field.distance([-2.0, -2.0]) is None
    assert field.distance([-0.5, 0.0]) is not None


def test_lookup_from_an_erosion_margin():
    chair = box_instance("c", [0.5, 0.7, -0.1, 0.1])
    grid = with_obstacle_at_footprint(open_map(), chair)
    field = DistanceField(grid, [chair])
    margin = [0.35, 0.0]  # inside the blocked ring round the chair
    assert not grid.is_free(margin)
    assert field.distance(margin) == pytest.approx(0.15, abs=TOL)


def test_grid_matches_igibson_truncation():
    grid = GridMap(open_map(), RES)
    # iGibson 2.2.2 IndoorScene.world_to_map: flip(xy / res + size / 2).astype(int)
    for xy in ([0.0, 0.0], [1.23, -0.47], [-2.96, 2.91], [0.3905, 1.7215]):
        expected = np.flip(np.asarray(xy) / RES + SIZE / 2.0).astype(int)
        assert grid.cell(xy) == tuple(expected)
        assert np.all(np.abs(grid.centre(*grid.cell(xy)) - xy) <= RES / 2 + 1e-9)


def test_bad_box_is_refused():
    with pytest.raises(objects.AnnotationError):
        Instance("x", "chair", [[1.0, 0.5, 0.0, 1.0]], view_from=[0, 0])


def test_annotation_file_loads_and_uses_goal_words():
    instances = objects.load_annotations("Rs")
    assert {i.category for i in instances} >= {"chair", "sofa", "table"}
    try:
        from vint_train.data.clip_goal_utils import GOAL_WORDS
    except ImportError:
        pytest.skip("train/ is not on the path")
    assert {i.category for i in instances} <= set(GOAL_WORDS)


def test_mesh_clearance_sees_what_the_map_misses():
    # A low table's vertices at 0.4 m: absent from a trav map, present here.
    table = np.array([[x, y, 0.4] for x in np.linspace(0.0, 0.6, 13)
                      for y in np.linspace(0.0, 0.4, 9)])
    floor = np.array([[x, y, 0.0] for x in np.linspace(-2, 2, 41)
                      for y in np.linspace(-2, 2, 41)])
    lamp_top = np.array([[-1.0, -1.0, 1.5]])  # above the body band: ignored
    clearance = objects.MeshClearance(np.vstack([table, floor, lamp_top]))
    assert clearance.clearance([0.3, 0.2]) == pytest.approx(0.0, abs=0.06)
    assert clearance.clearance([0.3, -0.5]) == pytest.approx(0.5, abs=0.06)
    assert clearance.clearance([-1.0, -1.0]) > 1.0
