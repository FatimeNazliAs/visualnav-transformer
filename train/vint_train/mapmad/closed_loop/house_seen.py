"""House seen (v2 plan 8.8 search phase; Phase 12 block item 7): how much of the start storey's floor the robot has had
in view, on the TRUE map. numpy only.

True map = the walkable area of the start storey (the server's top-down picture at reset: navmesh slice, rows = z,
cols = x of Habitat, 0.05 m per pixel), on a coarser grid of cell_m (0.1 m: the pixel at each cell's centre). The
house = the walkable cells 8-connected to the start's cell. Per step, a house cell counts as SEEN when:
- its centre is <= max_m (3.0 m) from the robot's turning centre;
- it lies within the camera's horizontal field of view (|bearing| <= hfov / 2);
- line of sight on the true map: every point sampled every half cell on the straight line from the robot to it is
  walkable (furniture and walls -- not walkable -- block the view).
`HouseSeen.update(position, yaw)` adds this step's cells and returns the running count; `share` = seen / house cells.
Habitat yaw: 0 looks along -z, positive turns left (robot.forward).
"""

import math
from typing import Sequence, Tuple

import numpy as np


def coarse_grid(grid: np.ndarray, m_per_px: float, cell_m: float) -> Tuple[np.ndarray, int]:
    """(walkable grid at cell_m, pixels per cell): the pixel at each coarse cell's centre."""
    k = max(1, int(round(cell_m / m_per_px)))
    return np.asarray(grid, dtype=bool)[k // 2::k, k // 2::k], k


def flood(free: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    """Cells 8-connected to `start` through free cells (bool); empty if start is not free."""
    out = np.zeros_like(free, dtype=bool)
    if not (0 <= start[0] < free.shape[0] and 0 <= start[1] < free.shape[1]) or not free[start]:
        return out
    out[start] = True
    while True:
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        g2 = grown.copy()
        g2[1:, 1:] |= out[:-1, :-1]
        g2[1:, :-1] |= out[:-1, 1:]
        g2[:-1, 1:] |= out[1:, :-1]
        g2[:-1, :-1] |= out[1:, 1:]
        g2 &= free
        if np.array_equal(g2, out):
            return out
        out = g2


class HouseSeen:
    """Running "house seen" counter of one episode (module docstring)."""

    def __init__(self, grid: np.ndarray, origin: Sequence[float], m_per_px: float, start_position: Sequence[float],
                 hfov_deg: float, max_m: float = 3.0, cell_m: float = 0.1) -> None:
        self.free, k = coarse_grid(grid, m_per_px, cell_m)
        self.res = k * m_per_px
        self.origin = np.asarray(origin, dtype=np.float64)  # coarse cell q spans [origin + q res, origin + (q + 1) res)
        self.house = flood(self.free, self.cell(start_position))
        self.seen = np.zeros_like(self.house)
        self.half_fov = math.radians(hfov_deg) / 2.0
        r = int(math.ceil(max_m / self.res))
        di, dj = np.mgrid[-r:r + 1, -r:r + 1]
        dist = np.hypot(di, dj) * self.res
        keep = (dist <= max_m) & (dist > 0)
        self.di, self.dj = di[keep], dj[keep]
        self.angle = np.arctan2(self.di, self.dj)  # rows = z, cols = x
        n = int(np.ceil(np.hypot(self.di, self.dj).max() * 2)) + 1
        t = np.linspace(0.0, 1.0, n)[1:]
        self.si = np.rint(self.di[:, None] * t[None, :]).astype(np.int64)  # sample offsets along each ray
        self.sj = np.rint(self.dj[:, None] * t[None, :]).astype(np.int64)

    @property
    def house_cells(self) -> int:
        return int(self.house.sum())

    def cell(self, position: Sequence[float]) -> Tuple[int, int]:
        """(row, col) of a Habitat position (x, y, z)."""
        return (int(math.floor((position[2] - self.origin[1]) / self.res)),
                int(math.floor((position[0] - self.origin[0]) / self.res)))

    def update(self, position: Sequence[float], yaw: float) -> int:
        """Add the cells seen from this pose; returns the number of house cells seen so far."""
        r0, c0 = self.cell(position)
        heading = math.atan2(-math.cos(yaw), -math.sin(yaw))  # (row, col) direction of forward = (-cos yaw, -sin yaw)
        rel = np.abs(np.arctan2(np.sin(self.angle - heading), np.cos(self.angle - heading)))
        sel = rel <= self.half_fov
        rows, cols = r0 + self.si[sel], c0 + self.sj[sel]
        h, w = self.free.shape
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        ok = np.where(inside, self.free[np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)], False).all(axis=1)
        tr, tc = r0 + self.di[sel][ok], c0 + self.dj[sel][ok]
        self.seen[tr, tc] = True
        self.seen &= self.house
        return int(self.seen.sum())

    def share(self) -> float:
        return float(self.seen.sum()) / max(self.house_cells, 1)
