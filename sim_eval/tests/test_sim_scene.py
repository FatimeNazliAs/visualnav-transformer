"""Pin the world half of the simulator adapter.

`SimScene` exists because twelve call sites were reaching through `body.env`
into iGibson to ask the questions it now answers. Moving them behind an
interface is only worth anything if the answers are right, and two of them
decide numbers that reach the results table:

  * **the geodesic rule.** "Not connected" is a real outcome — an agent that
    climbs onto furniture leaves the traversable component — and it must come
    back as *no distance*, not as an exception and not as a number. Every other
    failure must still raise: a scene that cannot plan at all would otherwise
    fill a table with silent blanks.
  * **the axis flip.** iGibson's `world_to_map` returns `(row, col)`, because
    the map image's first index runs along world y. A caller that forgets is
    not wrong by a little; it is transposed.

The floor is the third thing worth pinning. It used to be a parameter every
layer forwarded, and binding it in one place is only safe if it is actually
used for every lookup.

Runs against a fake scene — no GPU, no iGibson, no simulator.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_scene import SimScene  # noqa: E402


class NoPath(Exception):
    """Stands in for networkx's NetworkXNoPath, matched by name (not by class,
    because networkx is iGibson's dependency and not imported here)."""


NoPath.__name__ = "NetworkXNoPath"


class FakeIndoorScene:
    """Just enough of iGibson's IndoorScene, and it records what it was asked.

    Every floor answers differently on purpose, so a lookup that ignores the
    bound floor cannot pass.
    """

    def __init__(self, raises=None, floors=(0.0, 3.0), points=((1.0, 2.0, 0.0),)):
        self.scene_id = "Fake"
        self.floor_heights = list(floors)
        self.trav_map_resolution = 0.1
        self.floor_map = [np.full((4, 4), floor_index, dtype=np.uint8)
                          for floor_index in range(len(floors))]
        self.raises = raises
        self.asked_floors = []
        self._points = [np.asarray(point, dtype=float) for point in points]
        self._draws = 0

    def get_shortest_path(self, floor, source, target, entire_path=False):
        self.asked_floors.append(floor)
        if self.raises is not None:
            raise self.raises
        straight = float(np.linalg.norm(
            np.asarray(target, dtype=float) - np.asarray(source, dtype=float)))
        # `entire_path` is echoed back so the caller's choice is visible.
        path = (np.array([source, target]) if entire_path
                else np.array([target]))
        return path, straight

    def get_random_point(self, floor=None):
        self.asked_floors.append(floor)
        point = self._points[self._draws % len(self._points)]
        self._draws += 1
        return floor, point

    def has_node(self, floor, world_xy):
        self.asked_floors.append(floor)
        return bool(world_xy[0] >= 0)

    def world_to_map(self, xy):
        # The real one flips: (x, y) metres in, (row, col) pixels out.
        return np.array([xy[1] * 10, xy[0] * 10])


# --- the geodesic rule -------------------------------------------------------

def test_a_connected_pair_comes_back_as_a_distance():
    scene = SimScene(FakeIndoorScene(), floor=0)
    assert scene.geodesic_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_an_unconnected_pair_is_no_distance_rather_than_an_exception():
    scene = SimScene(FakeIndoorScene(raises=NoPath("no path")), floor=0)
    assert scene.geodesic_distance((0.0, 0.0), (1.0, 0.0)) is None


class NodeNotFound(Exception):
    """networkx's other way of saying "that point is not on the graph"."""


def test_a_missing_node_is_also_no_distance():
    scene = SimScene(FakeIndoorScene(raises=NodeNotFound("gone")), floor=0)
    assert scene.geodesic_distance((0.0, 0.0), (1.0, 0.0)) is None


def test_a_scene_that_cannot_plan_at_all_still_raises():
    """Swallowing every exception would turn a broken scene into a table full
    of blanks that each look like a legitimate 'no route'."""
    scene = SimScene(FakeIndoorScene(raises=RuntimeError("no graph")), floor=0)
    with pytest.raises(RuntimeError):
        scene.geodesic_distance((0.0, 0.0), (1.0, 0.0))


def test_a_pose_is_accepted_where_a_point_is_expected():
    """The runner asks with the robot's (x, y, yaw); only the first two count."""
    scene = SimScene(FakeIndoorScene(), floor=0)
    assert scene.geodesic_distance((0.0, 0.0, 1.57), (3.0, 4.0, -2.0)) \
        == pytest.approx(5.0)


# --- the whole path ----------------------------------------------------------

def test_the_shortest_path_comes_back_as_a_polyline_and_a_length():
    scene = SimScene(FakeIndoorScene(), floor=0)
    path, length = scene.shortest_path((0.0, 0.0), (0.0, 2.0))
    # entire_path=True: the builder needs the route, not just its end.
    assert path.shape == (2, 2)
    assert length == pytest.approx(2.0)


# --- the bound floor ---------------------------------------------------------

def test_every_lookup_uses_the_floor_the_scene_was_bound_to():
    raw = FakeIndoorScene(floors=(0.0, 3.0))
    scene = SimScene(raw, floor=1)

    assert scene.floor_height == 3.0
    scene.geodesic_distance((0.0, 0.0), (1.0, 0.0))
    scene.shortest_path((0.0, 0.0), (1.0, 0.0))
    scene.random_point()
    scene.has_node((1.0, 1.0))

    assert raw.asked_floors == [1, 1, 1, 1]
    # floor_map is indexed by the same floor, and each floor differs.
    assert scene.trav_map.max() == 1


def test_the_scene_reports_its_own_identity_and_scale():
    scene = SimScene(FakeIndoorScene(), floor=0)
    assert scene.scene_id == "Fake"
    assert scene.trav_map_resolution == pytest.approx(0.1)


# --- the sampler's two helpers ----------------------------------------------

def test_a_random_point_is_just_the_xy():
    scene = SimScene(FakeIndoorScene(points=((1.0, 2.0, 9.9),)), floor=0)
    assert list(scene.random_point()) == [1.0, 2.0]


def test_has_node_answers_for_a_point_without_being_handed_a_floor():
    scene = SimScene(FakeIndoorScene(), floor=0)
    assert scene.has_node((1.0, 1.0))
    assert not scene.has_node((-1.0, 1.0))


# --- the axis flip -----------------------------------------------------------

def test_world_to_map_undoes_igibsons_flip_and_takes_many_points():
    scene = SimScene(FakeIndoorScene(), floor=0)
    columns, rows = scene.world_to_map([(1.0, 2.0), (3.0, 4.0)])
    # The fake returns (row, col) = (y*10, x*10); columns must come back as x.
    assert list(columns) == [10.0, 30.0]
    assert list(rows) == [20.0, 40.0]


def test_world_to_map_ignores_a_third_coordinate():
    scene = SimScene(FakeIndoorScene(), floor=0)
    columns, rows = scene.world_to_map([(1.0, 2.0, 0.5)])
    assert (list(columns), list(rows)) == ([10.0], [20.0])
