"""Checks of the horizon geometry in mapmad_sim.camera on synthetic pictures (no simulator needed)."""

import math

import numpy as np
import pytest

from mapmad_sim import camera

F, CX, CY, W, H = 488.0918, 320.8124, 213.0621, 640, 480  # LIMO colour camera


def synthetic_floor_depth(height_m: float, pitch_deg: float) -> np.ndarray:
    """Planar depth of a flat floor seen by the camera above; 0 where the ray misses the floor."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    t = math.radians(pitch_deg)
    down = np.array([0.0, math.cos(t), -math.sin(t)])  # world "down" in camera coordinates
    ray_down = (u - CX) / F * down[0] + (v - CY) / F * down[1] + down[2]
    return np.where(ray_down > 1e-6, height_m / np.maximum(ray_down, 1e-6), 0.0)


def test_limo_fov_from_intrinsics():
    assert camera.fov_deg(W, F) == pytest.approx(66.5, abs=0.1)
    assert camera.fov_deg(H, F) == pytest.approx(52.4, abs=0.1)
    assert camera.focal_px(W, camera.fov_deg(W, F)) == pytest.approx(F)


def test_horizon_and_pitch_are_inverse():
    assert camera.horizon_row(0.0, F, 240.0) == 240.0
    assert camera.horizon_row(-2.0, F, 240.0) < 240.0  # looking down moves the horizon up the picture
    assert camera.pitch_for_horizon(camera.horizon_row(-2.0, F, 240.0), F, 240.0) == pytest.approx(-2.0)


def test_vanishing_point_of_converging_lines_with_one_outlier():
    vp = np.array([315.0, 222.0])
    starts = np.array([[0.0, 480.0], [640.0, 470.0], [640.0, 300.0], [290.0, 480.0]])
    segs = np.hstack([starts, starts + 0.6 * (vp - starts)])
    segs = np.vstack([segs, [[0.0, 100.0, 600.0, 120.0]]])  # unrelated line
    point, keep = camera.vanishing_point(segs)
    assert point == pytest.approx(vp, abs=0.5) and keep.tolist() == [True] * 4 + [False]


@pytest.mark.parametrize("pitch", [0.0, -2.0, 1.5])
def test_floor_plane_recovers_height_pitch_and_horizon(pitch):
    depth = synthetic_floor_depth(0.18, pitch)
    depth[depth > 8.0] = 0.0  # like a real sensor: no far readings
    fit = camera.floor_plane(depth, np.ones_like(depth, dtype=bool), F, CX, CY)
    assert fit["height_m"] == pytest.approx(0.18, abs=1e-3)
    assert fit["pitch_deg"] == pytest.approx(pitch, abs=0.05)
    assert fit["horizon_row"] == pytest.approx(camera.horizon_row(pitch, F, CY), abs=0.5)
