"""Phase 7.4 Method B ("floor motion") without the simulator: pose algebra, and a synthetic textured floor seen by a
pinhole camera at a known height, from which score_grid + pool must recover that height and pitch."""

import math

import numpy as np
import pytest

from mapmad_sim.floor_height import camera_axes, inverse_pose, pool, relative_pose, score_grid
from mapmad_sim.lens import OpenCVPinhole


def test_relative_and_inverse_pose():
    rel = relative_pose([1.0, 2.0], math.pi / 2, [1.0, 2.5], math.pi / 2 + 0.3)
    assert rel == pytest.approx((0.5, 0.0, 0.3), abs=1e-9)  # 0.5 m along the robot's heading, turned 0.3 rad left
    back = inverse_pose(inverse_pose(rel))
    assert back == pytest.approx(rel, abs=1e-12)
    dx, dy, dyaw = inverse_pose((0.4, 0.0, 0.0))
    assert (dx, dy, dyaw) == pytest.approx((-0.4, 0.0, 0.0))


def test_camera_axes_level_camera_looks_forward():
    axes = camera_axes(0.0)
    assert np.allclose(axes[:, 2], [1, 0, 0]) and np.allclose(axes[:, 0], [0, -1, 0]) and np.allclose(axes[:, 1], [0, 0, -1])
    assert np.allclose(axes.T @ axes, np.eye(3))


def floor_picture(lens: OpenCVPinhole, height_m: float, pitch_deg: float, x_m: float, forward_m: float) -> np.ndarray:
    """Grey picture of a textured flat floor from a camera `height_m` up, `x_m` along the floor (robot x)."""
    v, u = np.mgrid[0:lens.height, 0:lens.width]
    rays, _ = lens.unproject(np.stack([u.ravel() + 0.5, v.ravel() + 0.5], axis=1).astype(np.float64))
    d = rays @ camera_axes(math.radians(pitch_deg)).T  # robot frame
    t = np.where(d[:, 2] < -1e-6, height_m / -np.minimum(d[:, 2], -1e-6), np.nan)
    x, y = x_m + forward_m + t * d[:, 0], t * d[:, 1]
    g = np.sin(9.0 * x) + np.cos(7.0 * y) + 0.5 * np.sin(5.0 * x + 6.0 * y)
    return np.nan_to_num(g, nan=0.0).reshape(lens.height, lens.width)


def test_score_grid_recovers_known_height_and_pitch():
    lens = OpenCVPinhole.from_hfov(90.0, 160, 120)
    heights, pitches = np.round(np.arange(0.20, 0.61, 0.02), 3), np.arange(-6.0, 6.01, 2.0)
    surfaces = []
    for step in (0.06, 0.08, 0.10):
        a = floor_picture(lens, 0.40, -2.0, 0.0, 0.084)
        b = floor_picture(lens, 0.40, -2.0, step, 0.084)
        surfaces.append(score_grid(a, b, lens, (step, 0.0, 0.0), heights, pitches, forward_m=0.084))
    est = pool(np.stack(surfaces), heights, pitches, n_boot=50)
    assert est.height_m == pytest.approx(0.40, abs=0.021) and est.pitch_deg == pytest.approx(-2.0, abs=1e-9)
    assert est.n_pairs == 3
