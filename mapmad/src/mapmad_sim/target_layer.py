"""The "target seen here" layer in training (Phase 8 block item 2): simulator truth with deliberate errors.

Per frame (one depth + semantic picture at the robot's pose):
- view cells: the 10 cm storey cells hit by a valid depth pixel (0 < depth <= 4 m) no lower than floor_min_m below
  the real floor (the same points as the obstacle / explored map, plus everything above the obstacle band: a TV on
  a wall). These are the cells "in view" for the fusion (in the field of view, <= 4 m, passing the depth test,
  because they are where the depth rays actually ended);
- detections: an instance of the category counts as detected only with >= 200 pixels at depth <= 4 m (the same
  rule as "seen"); its mask pixels go through depth onto cells;
- clean layer: detected instances' cells score 1.0 (truth, no noise);
- noisy layer: per instance and frame a confidence ~ U(0.4, 1.0), missed with p = 0.2 (score 0); per frame with
  p = 0.05 one false mark: a 3 x 3-cell blob of value ~ U(0.3, 0.7) centred on a random obstacle cell in view;
- a cell's score in one frame = the maximum over the detections / blob covering it; both layers are then fused over
  views by map_v2.TargetFusion (equal-weight mean; no detection = 0).
One noise realisation per drive from its own random stream (seed derived from the drive id, datagen_v2.py).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.depth_map import FirstSeenMap, MapSpec, camera_points
from mapmad_sim.frames import to_2d


@dataclass(frozen=True)
class NoiseSpec:
    """Block item 2 numbers (p8_datagen.yaml `target_layer`)."""

    min_pixels: int = 200
    max_depth_m: float = 4.0
    confidence: Tuple[float, float] = (0.4, 1.0)
    miss_p: float = 0.2
    false_p: float = 0.05
    false_value: Tuple[float, float] = (0.3, 0.7)
    false_blob_cells: int = 3

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "NoiseSpec":
        return cls(min_pixels=int(cfg["min_pixels"]), max_depth_m=float(cfg["max_depth_m"]),
                   confidence=tuple(cfg["confidence"]), miss_p=float(cfg["miss_p"]), false_p=float(cfg["false_p"]),
                   false_value=tuple(cfg["false_value"]), false_blob_cells=int(cfg["false_blob_cells"]))


@dataclass
class FrameCells:
    """Where one frame's depth pixels land on the storey map."""

    pixel_cell: np.ndarray  # (H * W,) flat cell index per pixel, -1 = no valid point / outside the map
    view: np.ndarray  # flat cell indices in view (sorted, unique)
    obstacle_view: np.ndarray  # cells in view that hold an obstacle-band point this frame (sorted, unique)


def frame_cells(depth: np.ndarray, hfov_deg: float, cam_tf: np.ndarray, floor_y: float, fmap: FirstSeenMap,
                spec: MapSpec, max_depth_m: float) -> FrameCells:
    """Project every valid depth pixel onto the storey cells (module docstring)."""
    pts, keep = camera_points(depth, hfov_deg, max_depth_m)
    world = pts @ cam_tf[:3, :3].T + cam_tf[:3, 3]
    height = world[:, 1] - floor_y
    ok = height >= spec.floor_min_m
    flat, inside = fmap.cells(to_2d(world))
    ok &= inside
    pixel_cell = np.full(depth.size, -1, dtype=np.int64)
    pixel_cell[keep[ok]] = flat[ok]
    lo, hi = spec.obstacle_band_m
    obst = ok & (height >= lo) & (height <= hi)
    return FrameCells(pixel_cell, np.unique(flat[ok]), np.unique(flat[obst]))


@dataclass
class Detection:
    """One instance detected in one frame (raw record saved per drive)."""

    frame: int
    instance_id: int
    pixels: int
    confidence: float
    missed: bool
    cells: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.int64))


def detect(semantic: np.ndarray, depth: np.ndarray, instance_ids: Sequence[int], spec: NoiseSpec) -> Dict[int, np.ndarray]:
    """Instance id -> its flat pixel indices with 0 < depth <= max_depth, for instances with >= min_pixels of them."""
    near = ((depth > 0) & (depth <= spec.max_depth_m)).ravel()
    sem = semantic.ravel()
    out = {}
    for i in instance_ids:
        px = np.flatnonzero((sem == i) & near)
        if px.size >= spec.min_pixels:
            out[int(i)] = px
    return out


def blob(centre: int, shape: Tuple[int, int], size: int) -> np.ndarray:
    """Flat indices of the size x size cells centred on a flat cell (clipped to the map)."""
    i, j = divmod(int(centre), shape[1])
    r = size // 2
    ii, jj = np.mgrid[max(0, i - r):min(shape[0], i + r + 1), max(0, j - r):min(shape[1], j + r + 1)]
    return (ii * shape[1] + jj).ravel()


@dataclass
class FrameScores:
    """One frame's input to the two fusions, plus its raw record."""

    clean: Dict[int, float]
    noisy: Dict[int, float]
    detections: List[Detection]
    false_mark: Optional[Tuple[int, float]]  # (centre cell, value)


def max_into(scores: Dict[int, float], cells: np.ndarray, value: float) -> None:
    for c in cells.tolist():
        if value > scores.get(c, -1.0):
            scores[c] = value


def frame_scores(t: int, found: Dict[int, np.ndarray], cells: FrameCells, shape: Tuple[int, int], spec: NoiseSpec,
                 rng: np.random.Generator) -> FrameScores:
    """Clean and noisy per-cell scores of frame t (module docstring). Random draws, in this order: per detected
    instance (sorted by id) confidence then miss; then the false-mark draw (and its cell and value if it fires)."""
    clean: Dict[int, float] = {}
    noisy: Dict[int, float] = {}
    dets = []
    for inst in sorted(found):
        c = cells.pixel_cell[found[inst]]
        c = np.unique(c[c >= 0])
        conf = float(rng.uniform(*spec.confidence))
        missed = bool(rng.random() < spec.miss_p)
        dets.append(Detection(t, inst, int(found[inst].size), round(conf, 4), missed, c))
        max_into(clean, c, 1.0)
        max_into(noisy, c, 0.0 if missed else conf)
    false_mark = None
    if rng.random() < spec.false_p and cells.obstacle_view.size:
        centre = int(cells.obstacle_view[int(rng.integers(cells.obstacle_view.size))])
        value = float(rng.uniform(*spec.false_value))
        max_into(noisy, blob(centre, shape, spec.false_blob_cells), value)
        false_mark = (centre, round(value, 4))
    return FrameScores(clean, noisy, dets, false_mark)
