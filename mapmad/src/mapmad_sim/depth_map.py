"""The whole-floor obstacle / explored map of one drive, built from the simulated depth camera (plan D8, D13).

Every frame t, the depth picture (same pose, FOV and size as the colour camera) is turned into 3D points.
A point's height is measured from the REAL floor under the robot (floor probe, robot.py):
- 0 < depth <= max_depth_m (4 m), otherwise the pixel is unknown (0 = HM3D mesh hole, far = unreliable);
- height in [floor_min_m, obstacle_band_m[0]) (-0.10 .. 0.05 m): floor -> counts for `explored`;
- height in obstacle_band_m (0.05 .. 0.55 m): obstacle -> counts for `obstacle` and `explored`;
- higher than the band (tabletops above LIMO, ceiling, the floor above) or lower than floor_min_m (a stairwell
  going down, the floor below): ignored.
A cell becomes obstacle / explored at the first frame where its running point count reaches min_points (3, the
threshold of habitat-starter's examples/06_topdown_map.py). That frame index is stored per cell (-1 = never), so
the map "as known at frame t" never shows anything seen later (vint_train/mapmad/local_map.py).

Cells: `resolution` m (0.1) in the 2D frame of traj_data.pkl (frames.py): cell [i, j] spans
X in [origin[0] + i * res, ...), Y in [origin[1] + j * res, ...).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from mapmad_sim.camera import focal_px
from mapmad_sim.frames import to_2d


@dataclass(frozen=True)
class MapSpec:
    """Map-building numbers from the run config (`map` block of p2_datagen.yaml)."""

    resolution: float = 0.1
    max_depth_m: float = 4.0
    obstacle_band_m: Tuple[float, float] = (0.05, 0.55)
    floor_min_m: float = -0.10
    min_points: int = 3
    margin_m: float = 1.0  # map bounds = navmesh bounds + this

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "MapSpec":
        return cls(resolution=float(cfg["resolution_m"]), max_depth_m=float(cfg["max_depth_m"]),
                   obstacle_band_m=tuple(cfg["obstacle_band_m"]), floor_min_m=float(cfg["floor_min_m"]),
                   min_points=int(cfg["min_points"]), margin_m=float(cfg["margin_m"]))


def rotation(yaw: float, pitch_deg: float) -> np.ndarray:
    """Camera-to-world rotation of a Habitat camera: heading yaw about +y, then pitch (+ = up) about x."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return r_yaw @ r_pitch


def camera_points(depth: np.ndarray, hfov_deg: float, max_depth: float) -> Tuple[np.ndarray, np.ndarray]:
    """Planar depth (H, W) -> camera-frame points (N, 3) (OpenGL: x right, y up, looking along -z) and the
    flat pixel indices they came from. Pixels with depth 0 or > max_depth are dropped (unknown). Same maths as
    habitat_starter.mapping.backproject_depth (pixel centres, centred principal point)."""
    h, w = depth.shape
    f = focal_px(w, hfov_deg)
    rows, cols = np.mgrid[0:h, 0:w]
    d = depth.astype(np.float64).ravel()
    keep = np.flatnonzero((d > 0.0) & (d <= max_depth))
    d = d[keep]
    x = (cols.ravel()[keep] + 0.5 - w / 2.0) / f * d
    y = -(rows.ravel()[keep] + 0.5 - h / 2.0) / f * d
    return np.stack([x, y, -d], axis=1), keep


def world_points(depth: np.ndarray, hfov_deg: float, max_depth: float, cam_position: Sequence[float],
                 cam_rotation: np.ndarray) -> np.ndarray:
    """Depth picture -> Habitat world points (N, 3) of its valid pixels."""
    pts, _ = camera_points(depth, hfov_deg, max_depth)
    return pts @ np.asarray(cam_rotation).T + np.asarray(cam_position, dtype=np.float64)


class FirstSeenMap:
    """Running whole-floor map of one drive (see the module docstring)."""

    def __init__(self, spec: MapSpec, origin: Sequence[float], shape: Tuple[int, int]) -> None:
        self.spec = spec
        self.origin = np.asarray(origin, dtype=np.float64)
        self.shape = (int(shape[0]), int(shape[1]))
        n = self.shape[0] * self.shape[1]
        self.counts = {"obstacle": np.zeros(n, np.int32), "explored": np.zeros(n, np.int32)}
        self.first_seen = {k: np.full(n, -1, np.int32) for k in self.counts}
        self.points_outside = 0

    @classmethod
    def around(cls, spec: MapSpec, bounds_low: Sequence[float], bounds_high: Sequence[float]) -> "FirstSeenMap":
        """Map covering a Habitat box (navmesh bounds) plus spec.margin_m on every side."""
        corners = to_2d(np.array([bounds_low, bounds_high], dtype=np.float64))
        low, high = corners.min(axis=0) - spec.margin_m, corners.max(axis=0) + spec.margin_m
        shape = np.ceil((high - low) / spec.resolution).astype(int)
        return cls(spec, low, (shape[0], shape[1]))

    def cells(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flat cell index of each (X, Y) point and a mask of the points inside the map."""
        ij = np.floor((xy - self.origin) / self.spec.resolution).astype(np.int64)
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < self.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < self.shape[1])
        return ij[:, 0] * self.shape[1] + ij[:, 1], inside

    def add(self, points: np.ndarray, floor_y: float, t: int) -> Dict[str, int]:
        """Add frame t's world points (N, 3), heights measured from the real floor at Habitat height floor_y.
        Returns how many points went to each class."""
        height = points[:, 1] - floor_y
        lo, hi = self.spec.obstacle_band_m
        obstacle = (height >= lo) & (height <= hi)
        explored = obstacle | ((height >= self.spec.floor_min_m) & (height < lo))
        flat, inside = self.cells(to_2d(points))
        self.points_outside += int((explored & ~inside).sum())
        for layer, mask in (("obstacle", obstacle), ("explored", explored)):
            sel = flat[mask & inside]
            self.counts[layer] += np.bincount(sel, minlength=len(self.counts[layer])).astype(np.int32)
            new = (self.counts[layer] >= self.spec.min_points) & (self.first_seen[layer] < 0)
            self.first_seen[layer][new] = t
        return {"obstacle": int(obstacle.sum()), "floor": int((explored & ~obstacle).sum()), "points": len(points)}

    def arrays(self, floor_y: float) -> Dict[str, Any]:
        """Everything mapmad_map.npz stores (vint_train/mapmad/local_map.py reads it)."""
        return {"obstacle_first_seen": self.first_seen["obstacle"].reshape(self.shape),
                "explored_first_seen": self.first_seen["explored"].reshape(self.shape),
                "origin": self.origin, "resolution": np.float64(self.spec.resolution),
                "floor_height": np.float64(floor_y),
                "obstacle_band_m": np.asarray(self.spec.obstacle_band_m, dtype=np.float64),
                "max_depth_m": np.float64(self.spec.max_depth_m), "min_points": np.int32(self.spec.min_points)}
