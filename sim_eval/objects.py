"""Where the goal words are, and how far the robot is from the nearest one.

A word goal has no goal pose. "Go to the chair" is reached at *any* chair, so
the success rule of the language-goal sim (Phase 7) is object-reach: the
episode succeeds when the robot is within `success_radius_m` of the nearest
instance of the target category, measured around the furniture.

**Distance is to the footprint, not to a centre point.** A sofa's centre is
inside the sofa, more than a metre from any point the robot can stand on, so
"within 1 m of the centre" would ask a robot to climb it. Each instance is
annotated as one or more boxes (`object_annotations.yaml`), and the distance
read here is to the nearest point of the nearest box.

**A distance field, not a planner call per tick.** `SimScene.geodesic_distance`
grafts every query point onto the scene's graph, which is why the topomap
runner asks it only a handful of times per episode. Object-reach needs an
answer on every tick and for every instance, so this computes one field per
category, once per scene: Dijkstra over the scene's own (eroded) traversability
map, seeded from the free cells next to each footprint. A lookup is then a
window of array reads, it never touches the simulator, and it is deterministic.

The field's sources are the free cells within `source_reach_m` of a footprint,
each costing its straight-line distance to it — the last leg to an object is a
straight line, exactly as `get_shortest_path`'s graft makes it. Straight only
where it is clear: a source is dropped if the segment to the footprint runs
through an obstacle that is not the object's own erosion margin, which is what
stops a bar chair "leaking" through the counter it stands at into the kitchen
behind it.

Pure numpy, no simulator: `tests/test_objects.py` pins it on hand-made maps.
"""

import heapq
import math
from pathlib import Path

import numpy as np
import yaml

SIM_EVAL_DIR = Path(__file__).resolve().parent
ANNOTATIONS_PATH = SIM_EVAL_DIR / "object_annotations.yaml"

# How far from a footprint a free cell may be and still count as touching it.
# Above the scene's erosion (2 cells at 0.1 m) plus a cell, so every footprint
# with open floor beside it gets sources.
SOURCE_REACH_M = 0.6

# An obstacle on the straight segment from a cell to a footprint is tolerated
# only this close to the footprint: that is the object's own erosion margin.
# A wall or counter further out blocks the segment.
LOS_MARGIN_M = 0.32

# How far around the robot's cell a lookup searches for free cells, for a
# robot standing in an erosion margin (it drives closer to things than the
# eroded map allows).
LOOKUP_RADIUS_CELLS = 3

FREE = 255
_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class AnnotationError(ValueError):
    """Raised for an annotation file that cannot be what it claims."""


class Instance:
    """One annotated object: its category, its footprint and its photo pose."""

    def __init__(self, instance_id, category, boxes, view_from, note=""):
        self.id = str(instance_id)
        self.category = str(category)
        self.boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.view_from = np.asarray(view_from, dtype=float).reshape(2)
        self.note = str(note)
        bad = (self.boxes[:, 0] >= self.boxes[:, 1]) | (self.boxes[:, 2] >= self.boxes[:, 3])
        if bad.any():
            raise AnnotationError(
                "{}: a box must be [xmin, xmax, ymin, ymax] with min < max, got {}"
                .format(self.id, self.boxes[bad].tolist()))

    @classmethod
    def from_dict(cls, values):
        return cls(values["id"], values["category"], values["boxes"],
                   values["view_from"], values.get("note", ""))

    @property
    def centre(self):
        """The footprint's area-weighted centre — where a photo is aimed."""
        areas = ((self.boxes[:, 1] - self.boxes[:, 0])
                 * (self.boxes[:, 3] - self.boxes[:, 2]))
        centres = np.stack([(self.boxes[:, 0] + self.boxes[:, 1]) / 2,
                            (self.boxes[:, 2] + self.boxes[:, 3]) / 2], axis=1)
        return (centres * areas[:, None]).sum(0) / areas.sum()

    def surface_distance(self, xy):
        """Straight-line distance from `xy` to the footprint; 0 inside it."""
        return float(np.min(_box_distances(np.asarray(xy, dtype=float), self.boxes)))

    def nearest_point(self, xy):
        """The footprint point closest to `xy`."""
        xy = np.asarray(xy, dtype=float)
        points = np.stack([np.clip(xy[0], self.boxes[:, 0], self.boxes[:, 1]),
                           np.clip(xy[1], self.boxes[:, 2], self.boxes[:, 3])], axis=1)
        return points[int(np.argmin(np.linalg.norm(points - xy, axis=1)))]

    def view_yaw(self):
        """The heading that faces the footprint's centre from `view_from`."""
        dx, dy = self.centre - self.view_from
        return float(math.atan2(dy, dx))


def _box_distances(xy, boxes):
    dx = np.maximum(np.maximum(boxes[:, 0] - xy[0], 0.0), xy[0] - boxes[:, 1])
    dy = np.maximum(np.maximum(boxes[:, 2] - xy[1], 0.0), xy[1] - boxes[:, 3])
    return np.hypot(dx, dy)


def load_annotations(scene, path=ANNOTATIONS_PATH):
    """The instances annotated in one house, or AnnotationError."""
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    if scene not in data:
        raise AnnotationError("{} has no objects annotated in {}. Known houses: {}"
                              .format(scene, path, ", ".join(sorted(data))))
    instances = [Instance.from_dict(entry) for entry in data[scene]]
    ids = [instance.id for instance in instances]
    if len(set(ids)) != len(ids):
        raise AnnotationError("{}: instance ids repeat: {}".format(scene, ids))
    return instances


def categories(instances):
    """The categories present, in annotation order."""
    return list(dict.fromkeys(instance.category for instance in instances))


class GridMap:
    """A traversability map and the world <-> cell arithmetic iGibson uses.

    iGibson 2.2.2's `IndoorScene.world_to_map` is
    `flip(xy / res + size / 2).astype(int)` — truncation, not rounding: cell
    (row, col) covers world [(col - size/2) res, (col + 1 - size/2) res) along
    x, rows along y, the map's centre at the world origin. Restated here so a
    field can be computed without a scene; `lg7_1_annotations.py` checks it
    against the live scene's own conversion (it caught a rounding version).
    """

    def __init__(self, trav_map, resolution):
        self.free = np.asarray(trav_map) == FREE
        self.resolution = float(resolution)
        self.size = self.free.shape[0]
        if self.free.shape[0] != self.free.shape[1]:
            raise ValueError("iGibson trav maps are square, got {}".format(self.free.shape))

    def cell(self, xy):
        """(row, col) of a world point, truncated as iGibson truncates."""
        col = int(xy[0] / self.resolution + self.size / 2.0)
        row = int(xy[1] / self.resolution + self.size / 2.0)
        return row, col

    def centre(self, row, col):
        """The world point at the middle of a cell (iGibson's map_to_world
        gives its corner)."""
        return np.array([(col + 0.5 - self.size / 2.0) * self.resolution,
                         (row + 0.5 - self.size / 2.0) * self.resolution])

    def inside(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def is_free(self, xy):
        row, col = self.cell(xy)
        return self.inside(row, col) and bool(self.free[row, col])

    def first_blocked(self, start_xy, end_xy):
        """The first non-free point on the segment start -> end, or None."""
        start_xy, end_xy = np.asarray(start_xy, float), np.asarray(end_xy, float)
        length = float(np.linalg.norm(end_xy - start_xy))
        steps = max(int(math.ceil(length / (self.resolution / 4))), 1)
        for k in range(steps + 1):
            point = start_xy + (end_xy - start_xy) * (k / steps)
            if not self.is_free(point):
                return point
        return None


def clear_line(grid, xy, instance, margin_m=LOS_MARGIN_M):
    """Is the straight segment from `xy` to the instance's footprint clear,
    apart from the object's own margin?"""
    target = instance.nearest_point(xy)
    blocked = grid.first_blocked(xy, target)
    return blocked is None or instance.surface_distance(blocked) <= margin_m


class DistanceField:
    """Geodesic distance to the nearest footprint of a set of instances."""

    def __init__(self, grid, instances, source_reach_m=SOURCE_REACH_M,
                 los_margin_m=LOS_MARGIN_M):
        self.grid = grid
        self.instances = list(instances)
        if not self.instances:
            raise ValueError("a distance field needs at least one instance")
        self.source_reach_m = float(source_reach_m)
        self.los_margin_m = float(los_margin_m)
        self.values = self._dijkstra(self._sources(source_reach_m, los_margin_m))

    def _sources(self, reach_m, margin_m):
        grid = self.grid
        rows, cols = np.nonzero(grid.free)
        sources = {}
        for row, col in zip(rows, cols):
            centre = grid.centre(row, col)
            for instance in self.instances:
                distance = instance.surface_distance(centre)
                if distance > reach_m or not clear_line(grid, centre, instance, margin_m):
                    continue
                if distance < sources.get((row, col), math.inf):
                    sources[(row, col)] = distance
        if not sources:
            raise AnnotationError(
                "no free cell within {} m of {} — is the footprint on this map?"
                .format(reach_m, ", ".join(i.id for i in self.instances)))
        return sources

    def _dijkstra(self, sources):
        grid = self.grid
        values = np.full(grid.free.shape, np.inf)
        heap = [(cost, row, col) for (row, col), cost in sources.items()]
        for cost, row, col in heap:
            values[row, col] = cost
        heapq.heapify(heap)
        while heap:
            cost, row, col = heapq.heappop(heap)
            if cost > values[row, col]:
                continue
            for d_row, d_col in _NEIGHBOURS:
                n_row, n_col = row + d_row, col + d_col
                if not grid.inside(n_row, n_col) or not grid.free[n_row, n_col]:
                    continue
                step = grid.resolution * (math.sqrt(2.0) if d_row and d_col else 1.0)
                if cost + step < values[n_row, n_col]:
                    values[n_row, n_col] = cost + step
                    heapq.heappush(heap, (cost + step, n_row, n_col))
        return values

    def distance(self, xy):
        """Distance from a world point to the nearest footprint, or None.

        The minimum, over the free cells around the point, of the cell's field
        value plus the straight step to it: continuous in the robot's position
        rather than stepping by a cell, and defined for a robot standing in an
        erosion margin. None if no nearby cell connects to any footprint — the
        robot has left the traversable component.
        """
        xy = np.asarray(xy, dtype=float)[:2]
        row, col = self.grid.cell(xy)
        # Beside a footprint the straight line is the distance, exactly as it
        # is for a source cell; routing back out through a free cell would
        # overstate it for a robot inside the object's erosion margin.
        best = min((instance.surface_distance(xy) for instance in self.instances
                    if instance.surface_distance(xy) <= self.source_reach_m
                    and clear_line(self.grid, xy, instance, self.los_margin_m)),
                   default=math.inf)
        radius = LOOKUP_RADIUS_CELLS
        for n_row in range(row - radius, row + radius + 1):
            for n_col in range(col - radius, col + radius + 1):
                if not self.grid.inside(n_row, n_col):
                    continue
                value = self.values[n_row, n_col]
                if math.isfinite(value):
                    step = float(np.linalg.norm(self.grid.centre(n_row, n_col) - xy))
                    best = min(best, value + step)
        return None if not math.isfinite(best) else float(best)


class SceneObjects:
    """One house's annotated objects, with a field per category and per instance."""

    def __init__(self, scene_id, grid, instances):
        self.scene_id = scene_id
        self.grid = grid
        self.instances = {instance.id: instance for instance in instances}
        self._category_fields = {}
        self._instance_fields = {}

    @classmethod
    def for_scene(cls, sim_scene, path=ANNOTATIONS_PATH):
        """From an open `SimScene`: its own eroded map, its own resolution."""
        grid = GridMap(sim_scene.trav_map, sim_scene.trav_map_resolution)
        return cls(sim_scene.scene_id, grid, load_annotations(sim_scene.scene_id, path))

    def of_category(self, category):
        found = [i for i in self.instances.values() if i.category == category]
        if not found:
            raise AnnotationError("{} has no {!r} annotated".format(self.scene_id, category))
        return found

    def category_field(self, category):
        if category not in self._category_fields:
            self._category_fields[category] = DistanceField(
                self.grid, self.of_category(category))
        return self._category_fields[category]

    def instance_field(self, instance_id):
        if instance_id not in self._instance_fields:
            self._instance_fields[instance_id] = DistanceField(
                self.grid, [self.instances[instance_id]])
        return self._instance_fields[instance_id]

    def nearest_instance(self, category, xy):
        """(instance, distance) of the category's instance geodesically nearest `xy`."""
        best = None
        for instance in self.of_category(category):
            distance = self.instance_field(instance.id).distance(xy)
            if distance is not None and (best is None or distance < best[1]):
                best = (instance, distance)
        return best


# --- physical clearance, from the mesh itself ---------------------------------
#
# The Gibson traversability maps are not the furniture. Rs's marks the low
# coffee table as floor, and a notch under the TV console as free, so a start
# can be placed wedged against furniture the map does not show. A word task's
# start is therefore also checked against the house mesh: no vertex at
# robot-body height within `min_clearance_m`.

import os  # noqa: E402

MESH_NAME = "mesh_z_up.obj"
BODY_BAND_M = (0.05, 0.6)   # above the floor: the LoCoBot's base and camera mast
CLEARANCE_RES_M = 0.05


class MeshClearance:
    """Distance from a floor point to the nearest mesh geometry at body height."""

    def __init__(self, vertices, floor_height=0.0, band_m=BODY_BAND_M,
                 resolution=CLEARANCE_RES_M, extent_m=5.0):
        from scipy import ndimage

        vertices = np.asarray(vertices, dtype=float)
        low, high = floor_height + band_m[0], floor_height + band_m[1]
        body = vertices[(vertices[:, 2] > low) & (vertices[:, 2] < high), :2]
        self.resolution = float(resolution)
        self.extent_m = float(extent_m)
        size = int(round(2 * extent_m / resolution))
        occupied = np.zeros((size, size), dtype=bool)
        index = ((body + extent_m) / resolution).astype(int)
        keep = (index >= 0).all(1) & (index < size).all(1)
        occupied[index[keep, 1], index[keep, 0]] = True
        self.distance = ndimage.distance_transform_edt(~occupied) * resolution

    @classmethod
    def for_scene(cls, scene_id, floor_height=0.0):
        root = os.environ.get("GIBSON_DATASET_PATH", "/igibson_data/g_dataset")
        path = Path(root) / scene_id / MESH_NAME
        with open(path, "r") as handle:
            vertices = [line.split()[1:4] for line in handle if line.startswith("v ")]
        return cls(np.asarray(vertices, dtype=float), floor_height)

    def clearance(self, xy):
        col, row = ((np.asarray(xy, dtype=float)[:2] + self.extent_m)
                    / self.resolution).astype(int)
        if not (0 <= row < self.distance.shape[0] and 0 <= col < self.distance.shape[1]):
            return 0.0
        return float(self.distance[row, col])
