"""The world's end of the simulator adapter: what the harness asks the house.

`SimBody` adapts the *robot* — observe, command, pose. This adapts the
**world**: where the floor is, what is traversable, how far two points are
around the furniture, where a random valid start is. The two are separate
because they are separate questions with separate lifetimes — one body drives
many episodes through one scene — and because conflating them is what made the
old seam leak.

**It leaked measurably.** Before this module existed, twelve call sites across
seven files reached through `body.env` into iGibson to ask a question the
adapter did not cover: `body.env.scene.floor_heights[floor]`,
`body.env.scene` handed to a geodesic helper, `body.env.simulator.renderer`
passed to the GPU check at five sites. The P1 architecture review predicted
exactly this ("P3 needs geodesic distance ...; P4 needs the traversability
map. All would leak the same way") and P3 and P4 each proved it. An adapter
whose callers route around it is not protecting anything, so the interface was
widened to cover what they actually ask.

**The floor is bound once, not threaded.** Every question below is about one
floor, and passing `floor` beside the scene at every call site is how
`episode_runner`, `topomap_builder` and the recorder each ended up carrying a
`floor` argument they only forwarded. It is a constructor argument here, and
the callers lost their copies.

**No iGibson import.** This wraps whatever it is handed, so the whole module —
including the geodesic rule that decides success — is exercised in
`tests/test_sim_scene.py` against a fake scene, with no GPU and no simulator.
"""

import numpy as np

# `scene.get_shortest_path` raises networkx's NoPath when two points are not
# connected, which happens for real: an agent that climbs onto furniture leaves
# the traversable component entirely. That is a fact about the episode, not an
# error, so it is reported as "no geodesic" rather than raised.
#
# networkx is not imported here (it is iGibson's dependency, not ours), so the
# exception is identified by name rather than by class.
NO_PATH_ERRORS = ("NetworkXNoPath", "NodeNotFound")


class SimScene:
    """One floor of one house, answering the questions the harness has of it.

    Wraps iGibson's `IndoorScene` (`scene` below) and binds the floor. The raw
    object is deliberately not exposed: everything callers were reaching for is
    a named method here, and a new need should become another one rather than
    an escape hatch.
    """

    def __init__(self, scene, floor=0):
        self._scene = scene
        self.floor = int(floor)

    @property
    def scene_id(self):
        return str(self._scene.scene_id)

    @property
    def floor_height(self):
        """The z the robot is dropped onto when it is placed on this floor."""
        return float(self._scene.floor_heights[self.floor])

    # --- the nav mesh -------------------------------------------------------

    def random_point(self):
        """A uniformly random traversable point on this floor, as (x, y).

        Draws through numpy's global RNG, because iGibson's `get_random_point`
        does — seed it before calling (`topomap_builder.build_topomap` does).
        """
        _floor, point = self._scene.get_random_point(floor=self.floor)
        return np.asarray(point, dtype=float)[:2]

    def has_node(self, xy):
        """Is this point on the nav mesh's largest connected component?"""
        return bool(self._scene.has_node(self.floor, np.asarray(xy, dtype=float)[:2]))

    def shortest_path(self, source_xy, goal_xy):
        """A* over the traversability graph: `(path, length)`, both in metres.

        `entire_path=True` returns the whole polyline rather than the task's
        fixed number of waypoints, so the length that comes back is measured on
        the unsubsampled path — the true shortest-path length SPL needs.
        """
        path, length = self._scene.get_shortest_path(
            self.floor, np.asarray(source_xy, dtype=float)[:2],
            np.asarray(goal_xy, dtype=float)[:2], entire_path=True)
        return np.asarray(path, dtype=float), float(length)

    def geodesic_distance(self, source_xy, goal_xy):
        """Distance around the furniture, or None if the two are not connected.

        The difference from a straight line is exactly where the metric is read
        (plan §6): an agent stopped a metre from the goal with a wall between
        them has not nearly arrived.

        `get_shortest_path` grafts an off-graph endpoint onto the graph as a
        new node with a single edge to its nearest neighbour, so asking this
        *mutates the scene*. That is safe, and it is worth saying why, because
        it is the reason this is not simply asked every tick: a node with one
        edge is a leaf, no shortest path ever routes *through* a leaf, and so
        no later query comes back shorter for having been called. The graph
        still grows by a node per call, which is why callers ask only at the
        end of an episode and on the handful of ticks already inside the goal
        radius.
        """
        try:
            _path, length = self._scene.get_shortest_path(
                self.floor, np.asarray(source_xy, dtype=float)[:2],
                np.asarray(goal_xy, dtype=float)[:2], entire_path=False)
        except Exception as error:                   # noqa: BLE001 - see below
            # Anything that is not "these points are not connected" is
            # re-raised: a scene that cannot plan at all is a defect, and
            # swallowing it would turn every episode's distance into a silent
            # blank.
            if type(error).__name__ not in NO_PATH_ERRORS:
                raise
            return None
        return float(length)

    # --- the traversability map ---------------------------------------------

    @property
    def trav_map(self):
        """This floor's traversability image: 255 where the robot may drive."""
        return self._scene.floor_map[self.floor]

    @property
    def trav_map_resolution(self):
        """Metres per pixel of `trav_map`."""
        return float(self._scene.trav_map_resolution)

    def world_to_map(self, points):
        """World metres -> (columns, rows) of `trav_map`, for many points at once.

        iGibson's own `world_to_map` returns `(row, col)` — it flips the axes,
        because the map image's first index runs along world y — and takes one
        point at a time. Plotting wants them the other way round and in bulk,
        so the flip is undone here, once, for everybody.
        """
        rows_columns = np.array([
            self._scene.world_to_map(np.asarray(point, dtype=float)[:2])
            for point in points])
        return rows_columns[:, 1], rows_columns[:, 0]
