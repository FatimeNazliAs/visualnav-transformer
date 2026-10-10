"""Cut the robot's local top-down map (plan D2-D3) out of a drive's whole-floor map. Imports numpy only and
runs on Python 3.8: used by the NoMaD loader (Phase 3) and by the Habitat-side checks (Phase 2).

A drive's map (`mapmad_map.npz`, written by mapmad_sim.depth_map) covers its whole floor in 10 cm cells in the
2D frame of traj_data.pkl (X, Y; mapmad_sim/frames.py). Cell [i, j] spans
X in [origin[0] + i * res, origin[0] + (i + 1) * res) and Y in [origin[1] + j * res, ...). Per cell it stores
the first frame index at which the cell was seen as obstacle / as explored (-1 = never), so the map "as known
at frame t" is `0 <= first_seen <= t` (frame t's own depth picture included, nothing later).

The local map: `size` x `size` cells of `resolution` metres, the robot's turning centre in the middle (between
cells size/2 - 1 and size/2), forward = up (row 0 is the farthest ahead), the robot's left = image left.
Example: a wall 1 m straight ahead lies 10 rows above the centre in the middle columns.
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

LAYERS = ("obstacle", "explored")
MAP_FILE = "mapmad_map.npz"


def load_drive_map(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """A drive's map file (or its folder) as a dict: obstacle_first_seen, explored_first_seen, origin, resolution, ..."""
    path = Path(path)
    if path.is_dir():
        path = path / MAP_FILE
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def cell_centres(size: int, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """(forward, left) in metres of every local-map pixel's centre, each (size, size)."""
    offsets = (size / 2.0 - 0.5 - np.arange(size)) * resolution  # row/col 0 = farthest ahead / farthest left
    forward, left = np.meshgrid(offsets, offsets, indexing="ij")
    return forward, left


def local_to_world(forward: np.ndarray, left: np.ndarray, pose: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Robot-frame points -> world (X, Y) for pose (x, y, yaw) in traj_data.pkl's frame."""
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return x + forward * c - left * s, y + forward * s + left * c


def world_to_local(wx: np.ndarray, wy: np.ndarray, pose: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """World (X, Y) -> robot frame (forward, left) for pose (x, y, yaw); the exact inverse of local_to_world."""
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = np.asarray(wx, dtype=np.float64) - x, np.asarray(wy, dtype=np.float64) - y
    return dx * c + dy * s, -dx * s + dy * c


def known_at(first_seen: np.ndarray, t: int) -> np.ndarray:
    """Cells already seen at frame t: 0 <= first_seen <= t (bool)."""
    return (first_seen >= 0) & (first_seen <= t)


def local_map(drive_map: Dict[str, np.ndarray], t: int, pose: Tuple[float, float, float], size: int = 64,
              resolution: float = 0.1) -> np.ndarray:
    """(2, size, size) float32 map at frame t around `pose`: layer 0 obstacle, layer 1 explored (1 = yes).

    Each local pixel takes the stored cell under its centre (nearest cell); outside the stored map = 0.
    """
    forward, left = cell_centres(size, resolution)
    wx, wy = local_to_world(forward, left, pose)
    origin, res = np.asarray(drive_map["origin"], dtype=np.float64), float(drive_map["resolution"])
    i = np.floor((wx - origin[0]) / res).astype(np.int64)
    j = np.floor((wy - origin[1]) / res).astype(np.int64)
    out = np.zeros((len(LAYERS), size, size), dtype=np.float32)
    for k, layer in enumerate(LAYERS):
        grid = known_at(drive_map[f"{layer}_first_seen"], t)
        inside = (i >= 0) & (i < grid.shape[0]) & (j >= 0) & (j < grid.shape[1])
        out[k][inside] = grid[i[inside], j[inside]]
    return out
