"""The expert driver's planning grid: LIMO's floor map (navmesh, robot.py) of one island at 5 cm per cell.

The navmesh is already shrunk by LIMO's radius (0.195 m), so a cell on the island is a place where the turning
centre fits; `clearance` = how far (m) a cell is from the island's edge = room to spare beyond the robot's
radius. The shortest path that hugs corners (what the navmesh's own find_path returns) leaves no room for a
smooth curve, so `plan` runs Dijkstra on the 8-connected grid with every step's cost raised where clearance is
below `preferred_clearance_m`: paths keep to the middle of corridors and doorways where there is room, and go
close to walls only where they must. All points in and out are in the 2D frame of traj_data.pkl (frames.py).

Habitat's top-down views have rows along +z and columns along +x, starting at the navmesh bounds' low corner.
"""

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra



class FloorGrid:
    """Island `mask`, `clearance` (m) and Dijkstra planning on it."""

    def __init__(self, mask: np.ndarray, low_xz: Sequence[float], cell_m: float, preferred_clearance_m: float,
                 clearance_weight: float) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self.low_xz = np.asarray(low_xz, dtype=np.float64)  # Habitat (x, z) of cell [0, 0]'s corner
        self.cell_m = float(cell_m)
        self.clearance = ndimage.distance_transform_edt(self.mask) * self.cell_m  # 1 cell = on the island's edge
        self.preferred = float(preferred_clearance_m)
        self.weight = float(clearance_weight)
        self._graph = None

    @classmethod
    def from_pathfinder(cls, pathfinder: Any, point: Sequence[float], cell_m: float, preferred_clearance_m: float,
                        clearance_weight: float) -> "FloorGrid":
        """The island of the navmesh `point` (Habitat (x, y, z)) at its height."""
        island = pathfinder.get_island(np.asarray(point, dtype=np.float32))
        view = np.asarray(pathfinder.get_topdown_island_view(cell_m, float(point[1]), 0.5))
        low = pathfinder.get_bounds()[0]
        return cls(view == island, (low[0], low[2]), cell_m, preferred_clearance_m, clearance_weight)

    # --- coordinates -------------------------------------------------------------------------------------------
    def cell_of(self, xy: Sequence[float]) -> Tuple[int, int]:
        """(row, col) of the cell under a 2D point (X, Y) = Habitat (-z, -x)."""
        xy = np.asarray(xy, dtype=np.float64)
        return int(math.floor((-xy[0] - self.low_xz[1]) / self.cell_m)), int(math.floor((-xy[1] - self.low_xz[0]) / self.cell_m))

    def centre_of(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """2D points (N, 2) of cell centres."""
        x = self.low_xz[0] + (np.asarray(cols) + 0.5) * self.cell_m
        z = self.low_xz[1] + (np.asarray(rows) + 0.5) * self.cell_m
        return np.stack([-z, -x], axis=-1)

    def inside(self, rc: Tuple[int, int]) -> bool:
        return 0 <= rc[0] < self.mask.shape[0] and 0 <= rc[1] < self.mask.shape[1]

    def clearance_at(self, xy: Sequence[float]) -> float:
        """Clearance (m) of the cell under a 2D point; 0 off the island."""
        rc = self.cell_of(xy)
        return float(self.clearance[rc]) if self.inside(rc) else 0.0

    def nearest_free_cell(self, xy: Sequence[float], max_m: float = 0.25) -> Optional[Tuple[int, int]]:
        """The island cell nearest to a 2D point, at most max_m away; None if there is none."""
        r0, c0 = self.cell_of(xy)
        k = int(math.ceil(max_m / self.cell_m))
        rows, cols = np.mgrid[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]
        ok = (rows >= 0) & (rows < self.mask.shape[0]) & (cols >= 0) & (cols < self.mask.shape[1])
        rows, cols = rows[ok], cols[ok]
        free = self.mask[rows, cols]
        if not free.any():
            return None
        rows, cols = rows[free], cols[free]
        d = np.hypot(rows - r0, cols - c0)
        i = int(d.argmin())
        return (int(rows[i]), int(cols[i])) if d[i] * self.cell_m <= max_m else None

    def segment_clear(self, a_xy: Sequence[float], b_xy: Sequence[float], min_clearance_m: float) -> bool:
        """True if every point of the straight segment a-b (2D) has at least min_clearance_m to spare."""
        a, b = np.asarray(a_xy, dtype=np.float64), np.asarray(b_xy, dtype=np.float64)
        n = max(2, int(math.ceil(np.hypot(*(b - a)) / (self.cell_m / 2))) + 1)
        return all(self.clearance_at(a + (b - a) * k / (n - 1)) >= min_clearance_m for k in range(n))

    # --- planning ----------------------------------------------------------------------------------------------
    def _step_costs(self) -> np.ndarray:
        """Per-cell cost factor >= 1: 1 + weight * (shortfall below the preferred clearance, 0..1)."""
        short = np.clip((self.preferred - self.clearance) / self.preferred, 0.0, 1.0)
        return 1.0 + self.weight * short

    def graph(self):
        """Sparse 8-connected graph of the island cells (a diagonal step needs both side cells free too)."""
        if self._graph is None:
            h, w = self.mask.shape
            cost = self._step_costs()
            r, c = np.nonzero(self.mask)
            rows, cols, vals = [], [], []
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):  # each undirected edge once
                r2, c2 = r + dr, c + dc
                ok = (r2 < h) & (c2 >= 0) & (c2 < w)
                ok[ok] = self.mask[r2[ok], c2[ok]]
                if dr and dc:  # no cutting a blocked corner
                    ok[ok] = self.mask[r[ok], c2[ok]] & self.mask[r2[ok], c[ok]]
                a, b = r[ok] * w + c[ok], r2[ok] * w + c2[ok]
                rows.append(a)
                cols.append(b)
                vals.append(math.hypot(dr, dc) * self.cell_m * 0.5 * (cost.flat[a] + cost.flat[b]))
            self._graph = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(h * w, h * w)).tocsr()
        return self._graph

    def plan(self, a_xy: Sequence[float], b_xy: Sequence[float]) -> Optional[np.ndarray]:
        """Cell-centre path (N, 2) from a to b (2D points), endpoints replaced by a and b; None if unreachable."""
        a, b = self.nearest_free_cell(a_xy), self.nearest_free_cell(b_xy)
        if a is None or b is None:
            return None
        w = self.mask.shape[1]
        src, dst = a[0] * w + a[1], b[0] * w + b[1]
        dist, pred = dijkstra(self.graph(), directed=False, indices=src, return_predecessors=True)
        if not np.isfinite(dist[dst]):
            return None
        chain = [dst]
        while chain[-1] != src:
            chain.append(int(pred[chain[-1]]))
        chain = np.array(chain[::-1])
        pts = self.centre_of(chain // w, chain % w)
        pts[0], pts[-1] = np.asarray(a_xy, dtype=np.float64), np.asarray(b_xy, dtype=np.float64)
        return pts

