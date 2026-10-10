"""Goal heat layer (plan D4-D6, Phase 3 confirmation item 2-3): a warm hill (0-1) where the target is, in the same
64 x 64 robot-centred window as `local_map.py`. numpy only, Python 3.8.

Rules (exact):
- spot target: heat = exp(-d^2 / (2 sigma^2)), d = distance from the cell centre to the spot (sigma 0.3 m, peak 1);
- object target: d = distance from the cell centre to the footprint rectangle (0 inside -> heat 1 inside); a
  footprint partly inside the window is drawn clipped;
- "outside": the spot / the whole footprint lies outside the window square. Then one blob (sigma, peak 1) is drawn
  where the line robot -> target (spot / footprint centre) crosses the square's edge, moved `border_inset`
  (0.3 m = 3 cells) back toward the robot (D5).
Pose: cell centres go to the world with `local_map.cell_centres` + `local_map.local_to_world` -- the very functions
`local_map` uses -- and the target comes into the robot frame with `local_map.world_to_local` (their inverse).

Footprint: the drive meta stores the world-axis-aligned box around the object's oriented box (Habitat x, y, z;
`mapmad_sim.objects.object_box`); its x-z extent becomes an axis-aligned rectangle in the 2D frame
(X = -z, Y = -x; `mapmad_sim/frames.py`).

Perturbation (training only, item 3): `perturb_target` + `draw_heat(..., sigma=...)` + `add_false_blob`; the
choice of which one (and whether) is made by `map_sample.MapSampleBuilder`.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from vint_train.mapmad.local_map import cell_centres, local_to_world, world_to_local


@dataclass(frozen=True)
class Target:
    """A drive's target in the 2D frame: a spot (rect None) or an object footprint rect (xmin, xmax, ymin, ymax)."""

    kind: str  # "spot" | "object"
    centre: Tuple[float, float]  # spot, or footprint centre (X, Y)
    rect: Optional[Tuple[float, float, float, float]] = None


def habitat_to_2d(position) -> Tuple[float, float]:
    """Habitat (x, y, z) -> (X, Y) = (-z, -x), as mapmad_sim.frames.to_2d."""
    return -float(position[2]), -float(position[0])


def target_from_meta(meta: Dict[str, Any]) -> Target:
    """The heat target of a drive from mapmad_meta.json: spot -> end position; object -> box footprint."""
    t = meta["target"]
    if t["kind"] == "spot":
        return Target("spot", habitat_to_2d(t["end_position"]))
    if t["kind"] == "object":
        cx, _, cz = t["box_center"]
        sx, _, sz = t["box_size"]
        rect = (-(cz + sz / 2.0), -(cz - sz / 2.0), -(cx + sx / 2.0), -(cx - sx / 2.0))
        return Target("object", habitat_to_2d(t["box_center"]), rect)
    raise ValueError(f"unknown target kind {t['kind']!r}")


def _rect_corners(rect: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = rect
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float64)


def _polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons given as (N, 2) corner lists in order."""
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for ex, ey in edges:
            axis = np.array([-ey, ex])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def is_outside(target: Target, pose: Tuple[float, float, float], size: int, resolution: float) -> bool:
    """True if the spot / the whole footprint lies outside the window square (robot frame, half-width size*res/2)."""
    half = size * resolution / 2.0
    if target.rect is None:
        f, l = world_to_local(target.centre[0], target.centre[1], pose)
        return bool(abs(f) >= half or abs(l) >= half)
    corners = _rect_corners(target.rect)
    f, l = world_to_local(corners[:, 0], corners[:, 1], pose)
    square = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return not _polygons_overlap(np.stack([f, l], axis=1), square)


def border_point(target: Target, pose: Tuple[float, float, float], size: int, resolution: float,
                 inset: float) -> Tuple[float, float]:
    """(forward, left) of the border blob: line robot -> target centre meets the square's edge, moved `inset` back."""
    half = size * resolution / 2.0
    f, l = world_to_local(target.centre[0], target.centre[1], pose)
    f, l = float(f), float(l)
    scale = half / max(abs(f), abs(l))
    bf, bl = f * scale, l * scale
    length = float(np.hypot(bf, bl))
    keep = max(length - inset, 0.0) / length
    return bf * keep, bl * keep


def gaussian(d: np.ndarray, sigma: float) -> np.ndarray:
    """Peak-1 Gaussian of a distance array."""
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def draw_heat(target: Target, pose: Tuple[float, float, float], size: int = 64, resolution: float = 0.1,
              sigma: float = 0.3, inset: float = 0.3) -> np.ndarray:
    """(size, size) float32 heat for `target` seen from `pose` (rules in the module docstring)."""
    forward, left = cell_centres(size, resolution)
    if is_outside(target, pose, size, resolution):
        bf, bl = border_point(target, pose, size, resolution, inset)
        return gaussian(np.hypot(forward - bf, left - bl), sigma).astype(np.float32)
    wx, wy = local_to_world(forward, left, pose)
    if target.rect is None:
        d = np.hypot(wx - target.centre[0], wy - target.centre[1])
    else:
        xmin, xmax, ymin, ymax = target.rect
        dx = np.maximum(np.maximum(xmin - wx, wx - xmax), 0.0)
        dy = np.maximum(np.maximum(ymin - wy, wy - ymax), 0.0)
        d = np.hypot(dx, dy)
    return gaussian(d, sigma).astype(np.float32)


def blob_centre(target: Target, pose: Tuple[float, float, float], size: int, resolution: float,
                inset: float) -> Tuple[float, float]:
    """(forward, left) of the true blob's centre: the border point if outside, else the target centre."""
    if is_outside(target, pose, size, resolution):
        return border_point(target, pose, size, resolution, inset)
    f, l = world_to_local(target.centre[0], target.centre[1], pose)
    return float(f), float(l)


def perturb_target(target: Target, rng: np.random.Generator, shift_range: Tuple[float, float]) -> Target:
    """The target moved by U(shift_range) metres in a uniformly random direction (rect moves with it)."""
    dist = rng.uniform(*shift_range)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    dx, dy = dist * np.cos(angle), dist * np.sin(angle)
    rect = None if target.rect is None else (target.rect[0] + dx, target.rect[1] + dx, target.rect[2] + dy,
                                             target.rect[3] + dy)
    return replace(target, centre=(target.centre[0] + dx, target.centre[1] + dy), rect=rect)


def add_false_blob(heat: np.ndarray, true_centre: Tuple[float, float], rng: np.random.Generator, size: int,
                   resolution: float, sigma: float, min_dist: float, tries: int = 100) -> np.ndarray:
    """heat with one extra peak-1 blob at a uniform point in the window >= min_dist from the true blob (max-merged).

    Gives up (returns heat unchanged) after `tries` rejected draws, which only happens if the window is tiny.
    """
    half = size * resolution / 2.0
    forward, left = cell_centres(size, resolution)
    for _ in range(tries):
        f, l = rng.uniform(-half, half, size=2)
        if np.hypot(f - true_centre[0], l - true_centre[1]) >= min_dist:
            blob = gaussian(np.hypot(forward - f, left - l), sigma).astype(np.float32)
            return np.maximum(heat, blob)
    return heat


def rotate_180(heat: np.ndarray) -> np.ndarray:
    """Heat rotated 180 degrees about the robot (the window centre): the offline 'wrong heat' arm."""
    return np.ascontiguousarray(heat[::-1, ::-1])
