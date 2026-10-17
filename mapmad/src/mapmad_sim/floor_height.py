"""Phase 7.4 Method B ("floor motion"): camera height and pitch from two pictures of a flat floor and the robot's
odometry between them.

Idea: a floor pixel of picture 1 is a ray; for a guessed camera height h and pitch, the ray hits the floor at one
point. Move that point by the robot's known motion (odometry) and project it into picture 2. Only the true (h, pitch)
makes the floor of picture 1 land on the same floor texture in picture 2 (the metric step from odometry fixes the
scale). Score = normalised cross-correlation (NCC) of the grey values; grid search over h and pitch.

Frames: robot = x forward, y left, z up (ROS / NoMaD odometry); camera = OpenCV (x right, y down, z forward), at
`forward_m` in front of the turning centre and h above the floor, pitched by `pitch` (+ = up), no roll.
Camera models: anything with project / unproject of mapmad_sim.lens.

    surface = score_grid(img1, img2, lens, rel_pose, heights, pitches)   # one pair: NCC per (pitch, height)
    est = pool(np.stack(surfaces), heights, pitches)                     # many pairs -> PooledEstimate
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import numpy as np


def camera_axes(pitch_rad: float) -> np.ndarray:
    """3x3 matrix whose columns are the camera's x (right), y (down), z (forward) axes in the robot frame."""
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    return np.array([[0.0, s, c],
                     [-1.0, 0.0, 0.0],
                     [0.0, -c, s]])


def relative_pose(xy1: Sequence[float], yaw1: float, xy2: Sequence[float], yaw2: float) -> Tuple[float, float, float]:
    """Robot pose 2 in robot frame 1: (dx forward, dy left, dyaw) from two world poses (odometry)."""
    d = np.asarray(xy2, dtype=np.float64) - np.asarray(xy1, dtype=np.float64)
    c, s = math.cos(yaw1), math.sin(yaw1)
    return float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1]), float(math.atan2(math.sin(yaw2 - yaw1), math.cos(yaw2 - yaw1)))


def inverse_pose(rel: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Robot pose 1 in robot frame 2, from pose 2 in frame 1."""
    dx, dy, dyaw = rel
    c, s = math.cos(dyaw), math.sin(dyaw)
    return -(c * dx + s * dy), -(-s * dx + c * dy), -dyaw


def floor_pixels(lens: Any, rows: Tuple[float, float], cols: Tuple[float, float], step: int) -> np.ndarray:
    """(N, 2) pixel centres on a grid in the given row / column fractions of the picture (the floor region)."""
    v = np.arange(int(rows[0] * lens.height), int(rows[1] * lens.height), step) + 0.5
    u = np.arange(int(cols[0] * lens.width), int(cols[1] * lens.width), step) + 0.5
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


def bilinear(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Grey values of a float image at sub-pixel positions (caller keeps uv inside the picture)."""
    h, w = img.shape
    x = np.clip(uv[:, 0] - 0.5, 0, w - 1.001)
    y = np.clip(uv[:, 1] - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else -1.0


def warp_scores(img1: np.ndarray, img2: np.ndarray, lens: Any, pixels: np.ndarray, rel: Tuple[float, float, float],
                heights: Sequence[float], pitch_deg: float, forward_m: float, min_valid: float,
                max_range_m: float) -> np.ndarray:
    """NCC for every height at one pitch (-1 where too few floor points project validly)."""
    rays, ok_ray = lens.unproject(pixels)
    axes = camera_axes(math.radians(pitch_deg))
    d = rays @ axes.T  # rays in robot frame 1
    down = (d[:, 2] < -1e-3) & ok_ray
    g1 = bilinear(img1, pixels)
    dx, dy, dyaw = rel
    c, s = math.cos(dyaw), math.sin(dyaw)
    out = np.full(len(heights), -1.0)
    for i, h in enumerate(heights):
        t = np.where(down, -h / np.where(down, d[:, 2], -1.0), 0.0)
        p = np.array([forward_m, 0.0, h]) + t[:, None] * d  # floor points, robot frame 1
        keep = down & (np.hypot(p[:, 0], p[:, 1]) < max_range_m)
        q = p - np.array([dx, dy, 0.0])
        q2 = np.stack([c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1], q[:, 2]], axis=1)  # robot frame 2
        cam = (q2 - np.array([forward_m, 0.0, h])) @ axes  # camera frame 2 (OpenCV)
        uv, ok = lens.project(cam)
        ok &= keep & (cam[:, 2] > 0.05) & (uv[:, 0] > 1) & (uv[:, 0] < lens.width - 1) & (uv[:, 1] > 1) & (uv[:, 1] < lens.height - 1)
        if ok.sum() < min_valid * len(pixels):
            continue
        out[i] = ncc(g1[ok], bilinear(img2, uv[ok]))
    return out


def score_grid(img1: np.ndarray, img2: np.ndarray, lens: Any, rel: Tuple[float, float, float], heights: Sequence[float],
               pitches: Sequence[float], forward_m: float = 0.0, rows: Tuple[float, float] = (0.62, 0.95),
               cols: Tuple[float, float] = (0.2, 0.8), step: int = 2, min_valid: float = 0.5,
               max_range_m: float = 4.0) -> np.ndarray:
    """(n_pitches, n_heights) NCC surface of one pair (-1 = not enough valid floor points). The floor pixels are taken
    from the LATER picture (img2) and looked up in the earlier one: when the robot drives forward, everything it sees
    later was already in view earlier (farther away), while near floor points of the earlier picture pass under it."""
    pixels = floor_pixels(lens, rows, cols, step)
    return np.stack([warp_scores(img2, img1, lens, pixels, inverse_pose(rel), np.asarray(heights, dtype=np.float64), p,
                                 forward_m, min_valid, max_range_m) for p in pitches])


@dataclass(frozen=True)
class PooledEstimate:
    """One (height, pitch) for a whole set of pairs: the maximum of the summed NCC surfaces; range = percentiles of
    the same maximum over bootstrap resamples of the pairs."""

    height_m: float
    pitch_deg: float
    height_range_m: Tuple[float, float]  # 5-95 % over bootstrap resamples
    pitch_range_deg: Tuple[float, float]
    n_pairs: int


def pool(surfaces: np.ndarray, heights: Sequence[float], pitches: Sequence[float], n_boot: int = 1000,
         seed: int = 0, percentiles: Tuple[float, float] = (5.0, 95.0)) -> PooledEstimate:
    """Best (height, pitch) of the mean surface over pairs (surfaces: (n_pairs, n_pitches, n_heights); a pair's
    invalid cells count as 0), plus bootstrap percentiles."""
    s = np.where(surfaces < -0.999, 0.0, surfaces)
    heights, pitches = np.asarray(heights), np.asarray(pitches)

    def best(sub: np.ndarray) -> Tuple[float, float]:
        k, i = np.unravel_index(int(np.argmax(sub.mean(axis=0))), sub.shape[1:])
        return float(heights[i]), float(pitches[k])

    h, p = best(s)
    rng = np.random.default_rng(seed)
    boot = np.array([best(s[rng.integers(0, len(s), len(s))]) for _ in range(n_boot)])
    return PooledEstimate(height_m=h, pitch_deg=p,
                          height_range_m=tuple(float(v) for v in np.percentile(boot[:, 0], percentiles)),
                          pitch_range_deg=tuple(float(v) for v in np.percentile(boot[:, 1], percentiles)),
                          n_pairs=len(s))

