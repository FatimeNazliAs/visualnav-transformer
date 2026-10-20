"""MapMaD v2 map (plan v2 §8.1, Phase 8 block item 3): four layers on the whole-storey map and two robot-centred windows.

Imports numpy only and runs on Python 3.8: used by the drive generator v2 and the explorer teacher (Habitat side, via
sys.path like online_map.py), the alignment sheet, and the later NoMaD loader (Phase 10). The v1 crop
(`local_map.local_map`) is not changed; layers 0-1 of the near window are exactly its output (test_map_v2.py).

Whole-storey layers at frame t (10 cm cells, cell [i, j] as in local_map.py):
0. obstacle: `local_map.known_at(obstacle_first_seen, t)` (as v1);
1. explored: `known_at(explored_first_seen, t)` (as v1);
2. frontier: an explored, free (not obstacle) cell that shares an edge with an unknown (not explored) cell; 8-connected
   clusters of fewer than `min_cluster` (4) cells are dropped; recomputed from layers 0-1 (`frontier`). Cells outside
   the stored map count as known, so the map's own border is never a frontier (it lies >= 1 m beyond the floor map);
3. target seen here, 0..1: the fused detections (`TargetFusion`), stored as sparse updates (`target_at`).
The category name is never on the map.

Windows (`near_window`, `far_window`): both centred on the robot's turning centre, forward = up, robot's left = image
left, exactly local_map.py's geometry. Each output pixel takes the stored cell under its centre (nearest cell); outside
the stored map = 0.
- near: 64 x 64 at 10 cm (6.4 m) = local_map.local_map's sampling, 4 layers;
- far: 64 x 64 at 40 cm (25.6 m). A 256 x 256 grid is sampled at 10 cm like the near window, then every 4 x 4 block
  is summarised: obstacle = any, explored = mean, frontier = any, target = max. The near window is the far grid's
  central 64 x 64 at 10 cm, so the central 16 x 16 far cells are exactly the summary of the near window.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from vint_train.mapmad.local_map import cell_centres, known_at, local_to_world

LAYERS_V2 = ("obstacle", "explored", "frontier", "target")
FAR_BLOCK = 4  # far cell = 4 x 4 near cells
SUMMARY = {"obstacle": "any", "explored": "mean", "frontier": "any", "target": "max"}
TARGET_FILE = "mapmad_target.npz"
Pose = Tuple[float, float, float]


# --- layer 2: frontier ------------------------------------------------------------------------------------------

def _shift(mask: np.ndarray, di: int, dj: int, fill: bool) -> np.ndarray:
    """mask moved by (di, dj): out[i, j] = mask[i - di, j - dj], `fill` where that is outside."""
    out = np.full_like(mask, fill)
    h, w = mask.shape
    out[max(di, 0):h + min(di, 0), max(dj, 0):w + min(dj, 0)] = mask[max(-di, 0):h - max(di, 0), max(-dj, 0):w - max(dj, 0)]
    return out


def frontier_cells(obstacle: np.ndarray, explored: np.ndarray, visited: Optional[np.ndarray] = None) -> np.ndarray:
    """Explored free cells with an unknown edge neighbour (before the cluster-size filter). `visited` (optional, the
    proposal P1 at S-pilot): cells near earlier robot positions never count as unknown (the camera's blind zone)."""
    free = explored & ~obstacle
    unknown = ~explored if visited is None else ~explored & ~visited
    touch = np.zeros_like(free)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        touch |= _shift(unknown, di, dj, False)
    return free & touch


def label_clusters(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected components of a bool grid: (labels with 0 = none, 1..n, number n). Numbered in the order of
    their first cell in row-major order (deterministic)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    h, w = mask.shape
    n = 0
    for start in zip(*np.nonzero(mask)):
        if labels[start]:
            continue
        n += 1
        labels[start] = n
        stack = [start]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < h and 0 <= b < w and mask[a, b] and not labels[a, b]:
                        labels[a, b] = n
                        stack.append((a, b))
    return labels, n


def frontier_clusters(obstacle: np.ndarray, explored: np.ndarray, min_cluster: int = 4,
                      visited: Optional[np.ndarray] = None) -> np.ndarray:
    """Cluster labels (0 = no frontier, 1..n) of the frontier clusters with >= min_cluster cells, renumbered 1..n."""
    labels, n = label_clusters(frontier_cells(obstacle, explored, visited))
    if n == 0:
        return labels
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_cluster
    keep[0] = False
    new = np.zeros(n + 1, dtype=np.int32)
    new[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return new[labels]


def frontier(obstacle: np.ndarray, explored: np.ndarray, min_cluster: int = 4,
             visited: Optional[np.ndarray] = None) -> np.ndarray:
    """Layer 2 (bool): frontier cells of the clusters kept (module docstring)."""
    return frontier_clusters(obstacle, explored, min_cluster, visited) > 0


def visited_mask(shape: Sequence[int], origin: Sequence[float], res: float, positions: np.ndarray,
                 radius_m: float) -> np.ndarray:
    """Cells whose centre lies within radius_m of any of the (X, Y) positions (proposal P1)."""
    out = np.zeros((int(shape[0]), int(shape[1])), dtype=bool)
    if radius_m <= 0 or len(positions) == 0:
        return out
    r = int(np.ceil(radius_m / res))
    di, dj = np.mgrid[-r:r + 1, -r:r + 1]
    for x, y in np.asarray(positions, dtype=np.float64).reshape(-1, 2):
        i0, j0 = int(np.floor((x - origin[0]) / res)), int(np.floor((y - origin[1]) / res))
        ci = origin[0] + (i0 + di + 0.5) * res - x
        cj = origin[1] + (j0 + dj + 0.5) * res - y
        ok = (ci ** 2 + cj ** 2 <= radius_m ** 2) & (i0 + di >= 0) & (i0 + di < out.shape[0]) & (j0 + dj >= 0) & (j0 + dj < out.shape[1])
        out[(i0 + di)[ok], (j0 + dj)[ok]] = True
    return out


# --- layer 3: target seen here ----------------------------------------------------------------------------------

class TargetFusion:
    """Fused "target seen here" values of one storey map (block item 2): for every cell in view, value = equal-weight
    mean of the scores of all views that saw it; a view without a detection scores 0. A false alarm seen once
    therefore fades as soon as the cell is seen again without it. Every change is kept as a sparse update
    (cell, frame, value), so the layer at any frame can be rebuilt from the saved drive (`target_at`)."""

    def __init__(self, shape: Tuple[int, int]) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        n = self.shape[0] * self.shape[1]
        self.total = np.zeros(n, dtype=np.float64)
        self.count = np.zeros(n, dtype=np.int32)
        self.value = np.zeros(n, dtype=np.float32)
        self._cells: List[np.ndarray] = []
        self._frames: List[np.ndarray] = []
        self._values: List[np.ndarray] = []

    def update(self, t: int, in_view: np.ndarray, scores: Dict[int, float]) -> None:
        """Frame t: `in_view` = flat indices of the cells this view saw; `scores` = flat cell -> detection score
        (cells with a score count as seen too). Records the cells whose value changed."""
        cells = np.union1d(np.asarray(in_view, dtype=np.int64), np.fromiter(scores.keys(), dtype=np.int64, count=len(scores)))
        if cells.size == 0:
            return
        s = np.zeros(cells.size, dtype=np.float64)
        if scores:
            pos = np.searchsorted(cells, np.fromiter(scores.keys(), dtype=np.int64, count=len(scores)))
            s[pos] = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
        self.total[cells] += s
        self.count[cells] += 1
        new = (self.total[cells] / self.count[cells]).astype(np.float32)
        changed = new != self.value[cells]
        self.value[cells] = new
        if changed.any():
            self._cells.append(cells[changed].astype(np.int32))
            self._frames.append(np.full(int(changed.sum()), t, dtype=np.int32))
            self._values.append(new[changed])

    def grid(self) -> np.ndarray:
        """The current layer (H, W) float32."""
        return self.value.reshape(self.shape)

    def updates(self) -> Dict[str, np.ndarray]:
        """All changes so far: cell (flat index), frame, value; frames non-decreasing."""
        cat = (lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt))
        return {"cell": cat(self._cells, np.int32), "frame": cat(self._frames, np.int32),
                "value": cat(self._values, np.float32)}


def target_at(cell: np.ndarray, frame: np.ndarray, value: np.ndarray, t: int, shape: Sequence[int]) -> np.ndarray:
    """Layer 3 (H, W) float32 at frame t from sparse updates (the last update of each cell with frame <= t)."""
    out = np.zeros(int(shape[0]) * int(shape[1]), dtype=np.float32)
    k = int(np.searchsorted(frame, t, side="right"))  # updates are stored in frame order
    cells, last = np.unique(cell[:k][::-1], return_index=True)  # first in reverse order = the last update
    out[cells] = value[:k][::-1][last]
    return out.reshape(int(shape[0]), int(shape[1]))


# --- the four layers at frame t ---------------------------------------------------------------------------------

def layers_at(drive_map: Dict[str, np.ndarray], t: int, target: Optional[Dict[str, np.ndarray]] = None,
              target_key: str = "noisy", min_cluster: int = 4, positions: Optional[np.ndarray] = None,
              visited_m: float = 0.0) -> Dict[str, np.ndarray]:
    """Whole-storey layers at frame t: obstacle, explored, frontier (bool) and target (float32; zeros without a target
    file). `drive_map` = mapmad_map.npz's dict (or OnlineMap's live arrays); `target` = mapmad_target.npz's dict;
    target_key = "noisy" (training input) or "clean" (simulator truth). positions + visited_m > 0: proposal P1 (the
    robot's positions at frames 0..t, traj_data.pkl; cells within visited_m of them never count as unknown)."""
    obstacle = known_at(drive_map["obstacle_first_seen"], t)
    explored = known_at(drive_map["explored_first_seen"], t)
    if target is None:
        tgt = np.zeros(obstacle.shape, dtype=np.float32)
    else:
        tgt = target_at(target[f"{target_key}_cell"], target[f"{target_key}_frame"], target[f"{target_key}_value"],
                        t, obstacle.shape)
    visited = None
    if positions is not None and visited_m > 0:
        visited = visited_mask(obstacle.shape, drive_map["origin"], float(drive_map["resolution"]), positions[: t + 1], visited_m)
    return {"obstacle": obstacle, "explored": explored, "frontier": frontier(obstacle, explored, min_cluster, visited),
            "target": tgt}


# --- windows ----------------------------------------------------------------------------------------------------

def sample_layers(layers: Dict[str, np.ndarray], origin: Sequence[float], res: float, pose: Pose, size: int,
                  resolution: float, names: Iterable[str] = LAYERS_V2) -> np.ndarray:
    """(len(names), size, size) float32: each pixel = the stored cell under its centre (local_map.local_map's
    sampling, the same pose and world->cell maths); outside the stored map = 0."""
    forward, left = cell_centres(size, resolution)
    wx, wy = local_to_world(forward, left, pose)
    origin = np.asarray(origin, dtype=np.float64)
    i = np.floor((wx - origin[0]) / res).astype(np.int64)
    j = np.floor((wy - origin[1]) / res).astype(np.int64)
    names = list(names)
    out = np.zeros((len(names), size, size), dtype=np.float32)
    for k, name in enumerate(names):
        grid = layers[name]
        inside = (i >= 0) & (i < grid.shape[0]) & (j >= 0) & (j < grid.shape[1])
        out[k][inside] = grid[i[inside], j[inside]]
    return out


def near_window(layers: Dict[str, np.ndarray], origin: Sequence[float], res: float, pose: Pose, size: int = 64,
                resolution: float = 0.1) -> np.ndarray:
    """(4, 64, 64) near window, 10 cm cells, layers LAYERS_V2."""
    return sample_layers(layers, origin, res, pose, size, resolution)


def summarise(fine: np.ndarray, block: int = FAR_BLOCK) -> np.ndarray:
    """(4, B*n, B*n) fine grid -> (4, n, n): every block x block summarised per layer (SUMMARY)."""
    c, h, w = fine.shape
    blocks = fine.reshape(c, h // block, block, w // block, block)
    out = np.empty((c, h // block, w // block), dtype=np.float32)
    for k, name in enumerate(LAYERS_V2):
        b = blocks[k]
        if SUMMARY[name] == "any":
            out[k] = (b > 0).any(axis=(1, 3))
        elif SUMMARY[name] == "mean":
            out[k] = b.mean(axis=(1, 3))
        else:
            out[k] = b.max(axis=(1, 3))
    return out


def far_window(layers: Dict[str, np.ndarray], origin: Sequence[float], res: float, pose: Pose, size: int = 64,
               resolution: float = 0.4, block: int = FAR_BLOCK) -> np.ndarray:
    """(4, 64, 64) far window, 40 cm cells: a (64 * 4)^2 grid at 10 cm summarised in 4 x 4 blocks."""
    fine = sample_layers(layers, origin, res, pose, size * block, resolution / block)
    return summarise(fine, block)
