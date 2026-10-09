"""Pinhole-camera geometry for matching Habitat's camera to LIMO's: where the horizon falls in a picture.

Conventions: image rows grow downwards; pitch is in degrees, positive = looking up. For a camera above a
flat floor, every floor-parallel line meets the horizon, which lies at row cy + f * tan(pitch) for a camera
without roll. Habitat's camera is a centred pinhole (cy = height / 2), LIMO's colour camera has cy = 213 of
480, so the sim needs a small pitch to put its horizon on the same row as the real one.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np


def focal_px(width: int, hfov_deg: float) -> float:
    """Focal length in pixels of a pinhole camera `width` pixels wide with horizontal FOV hfov_deg."""
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def fov_deg(size_px: float, f: float) -> float:
    """Field of view (degrees) across size_px pixels for focal length f, centred principal point."""
    return math.degrees(2.0 * math.atan(size_px / 2.0 / f))


def horizon_row(pitch_deg: float, f: float, cy: float) -> float:
    """Image row of the horizon for a camera with this pitch (no roll)."""
    return cy + f * math.tan(math.radians(pitch_deg))


def pitch_for_horizon(row: float, f: float, cy: float) -> float:
    """The pitch (degrees, + = up) that puts the horizon on `row`; inverse of horizon_row."""
    return math.degrees(math.atan((row - cy) / f))


def vanishing_point(segments: np.ndarray, max_dist_px: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Meeting point (x, y) of line segments [[x1, y1, x2, y2], ...] that ignores unrelated lines.

    Every pair of lines proposes its crossing; the one that the most segment length passes within
    max_dist_px of wins, and the point is refitted by least squares (weighted by length) on those lines.
    Returns the point and a boolean mask of the segments used.
    """
    p1, p2 = segments[:, :2], segments[:, 2:]
    d = p2 - p1
    length = np.hypot(d[:, 0], d[:, 1])
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1) / length[:, None]  # unit normal of each line
    offset = -(normal * p1).sum(axis=1)  # line: normal . x + offset = 0
    best_support, keep = -1.0, np.zeros(len(segments), dtype=bool)
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            a = normal[[i, j]]
            if abs(np.linalg.det(a)) < 1e-3:  # (nearly) parallel in the picture
                continue
            near = np.abs(normal @ np.linalg.solve(a, -offset[[i, j]]) + offset) <= max_dist_px
            if length[near].sum() > best_support:
                best_support, keep = length[near].sum(), near
    w = length[keep]
    point = np.linalg.solve((normal[keep] * w[:, None]).T @ normal[keep],
                            -(normal[keep] * (w * offset[keep])[:, None]).sum(axis=0))
    return point, keep


def floor_plane(depth: np.ndarray, mask: np.ndarray, f: float, cx: float, cy: float,
                tolerance_m: float = 0.02, iterations: int = 200, seed: int = 0) -> Optional[Dict[str, float]]:
    """Fit the floor plane to the masked pixels of a planar depth image (metres; 0 = no reading).

    RANSAC on 3D points, then a least-squares refit on the inliers. Returns the camera's height above
    that floor, its pitch and roll (degrees) and the horizon row at the principal point's column, or None
    if fewer than 100 pixels are usable.
    """
    rows, cols = np.nonzero(mask & (depth > 0))
    if len(rows) < 100:
        return None
    z = depth[rows, cols].astype(np.float64)
    points = np.stack([(cols - cx) / f * z, (rows - cy) / f * z, z], axis=1)  # x right, y down, z forward
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs(points @ n - n @ p[0]) <= tolerance_m
        if inliers.sum() > best.sum():
            best = inliers
    centre = points[best].mean(axis=0)
    n = np.linalg.svd(points[best] - centre, full_matrices=False)[2][2]
    n = n if n[1] > 0 else -n  # point "down" (+y in camera coordinates)
    height = float(n @ centre)
    return {
        "height_m": height,
        "pitch_deg": math.degrees(math.atan2(-n[2], n[1])),
        "roll_deg": math.degrees(math.atan2(n[0], n[1])),
        "horizon_row": cy - f * n[2] / n[1],
        "inlier_share": float(best.mean()),
        "pixels": int(len(points)),
    }
