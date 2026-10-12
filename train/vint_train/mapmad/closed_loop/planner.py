"""Grid path planning for the map-planner arm E (Phase 5 block item 9). numpy + heapq only.

Grid = the robot's own whole-storey online map (mapmad_sim/online_map.py `grid`): cell [i, j] spans
X in [origin[0] + i res, ...), Y in [origin[1] + j res, ...) of the 2D frame (traj_data.pkl's).
- blocked = within inflation_m + res / 2 of a known obstacle cell (centre to centre: the robot's radius plus the
  obstacle cell's half width); unknown and explored cells are free; cells within start_clear_m of the robot are
  free (its own position is never blocked);
- goal = the target's footprint rectangle: A* (8-connected, step costs res and res sqrt 2, heuristic = distance
  to the rectangle) to the first free cell whose centre is <= goal_radius_m (planner_policy: inflation + 1.5
  cells) from it. If none is reachable (or after MAX_EXPANSIONS cells), the cell nearest the rectangle among those
  reached (ties: shorter path) is the goal ("the nearest reachable free cell next to the box");
- no path at all (the robot boxed in by inflated cells): plan again without inflation; still none -> no plan.
"""

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np

SQRT2 = math.sqrt(2.0)
MAX_EXPANSIONS = 100_000  # a whole storey is ~100,000-200,000 cells; caps the time of a search that cannot succeed
NEIGHBOURS = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0), (0, 1, 1.0), (1, -1, SQRT2), (1, 0, 1.0),
              (1, 1, SQRT2)]
Rect = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


def disk_offsets(radius_cells: float) -> List[Tuple[int, int]]:
    r = int(math.floor(radius_cells))
    return [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1) if math.hypot(di, dj) <= radius_cells]


def dilate(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Cells within radius_cells (centre to centre) of a True cell."""
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for di, dj in disk_offsets(radius_cells):
        src = mask[max(0, -di):h - max(0, di), max(0, -dj):w - max(0, dj)]
        out[max(0, di):h - max(0, -di), max(0, dj):w - max(0, -dj)] |= src
    return out


def cell_of(xy: Tuple[float, float], origin: Tuple[float, float], res: float) -> Tuple[int, int]:
    return int(math.floor((xy[0] - origin[0]) / res)), int(math.floor((xy[1] - origin[1]) / res))


def centre_of(cell: Tuple[int, int], origin: Tuple[float, float], res: float) -> Tuple[float, float]:
    return origin[0] + (cell[0] + 0.5) * res, origin[1] + (cell[1] + 0.5) * res


def rect_distance(x: float, y: float, rect: Rect) -> float:
    """Distance from (x, y) to the rectangle (0 inside)."""
    xmin, xmax, ymin, ymax = rect
    return math.hypot(max(xmin - x, x - xmax, 0.0), max(ymin - y, y - ymax, 0.0))


def blocked_cells(obstacle: np.ndarray, res: float, inflation_m: float, robot_cell: Tuple[int, int],
                  start_clear_m: float) -> np.ndarray:
    """The planning obstacles: inflated known obstacles minus the robot's own surroundings."""
    blocked = dilate(obstacle.astype(bool), inflation_m / res + 0.5) if inflation_m > 0 else obstacle.astype(bool)
    i0, j0 = robot_cell
    for di, dj in disk_offsets(start_clear_m / res):
        i, j = i0 + di, j0 + dj
        if 0 <= i < blocked.shape[0] and 0 <= j < blocked.shape[1]:
            blocked[i, j] = False
    return blocked


def astar(blocked: np.ndarray, start: Tuple[int, int], rect: Rect, origin: Tuple[float, float], res: float,
          goal_radius_m: float) -> Optional[List[Tuple[int, int]]]:
    """Cells from start to the goal (module docstring), start included; None if the start is outside the grid."""
    h, w = blocked.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w):
        return None

    def to_rect(c: Tuple[int, int]) -> float:
        x, y = centre_of(c, origin, res)
        return rect_distance(x, y, rect)

    g = {start: 0.0}
    parent = {start: None}
    heap = [(to_rect(start), 0.0, start)]
    best, best_key = start, (to_rect(start), 0.0)
    closed = set()
    while heap:
        _, cost, cell = heapq.heappop(heap)
        if cell in closed:
            continue
        closed.add(cell)
        if len(closed) > MAX_EXPANSIONS:
            break
        d = to_rect(cell)
        if d <= goal_radius_m:
            best = cell
            break
        if (d, cost) < best_key:
            best, best_key = cell, (d, cost)
        i, j = cell
        for di, dj, step in NEIGHBOURS:
            n = (i + di, j + dj)
            if not (0 <= n[0] < h and 0 <= n[1] < w) or blocked[n] or n in closed:
                continue
            c = cost + step * res
            if c < g.get(n, math.inf):
                g[n], parent[n] = c, cell
                heapq.heappush(heap, (c + to_rect(n), c, n))
    path = [best]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def lookahead_point(path_xy: np.ndarray, lookahead_m: float) -> np.ndarray:
    """The first path point at least lookahead_m along the path from its start (the last point if shorter)."""
    if len(path_xy) == 1:
        return path_xy[0]
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path_xy, axis=0).T))])
    k = int(np.searchsorted(along, lookahead_m))
    return path_xy[min(k, len(path_xy) - 1)]
