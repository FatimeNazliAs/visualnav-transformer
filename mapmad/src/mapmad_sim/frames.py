"""Habitat's 3D world frame -> the 2D frame NoMaD's datasets use (traj_data.pkl `position`, `yaw`).

NoMaD (checked on 200 GoStanford trajectories, Phase 2 log): position (x, y) in metres in a right-handed plane,
yaw = heading of motion, counter-clockwise positive; `to_local_coords` then puts forward on +x and left on +y.

Habitat: y is up, yaw is a rotation about +y, yaw 0 looks along -z and positive yaw turns left. So
    X = -z,  Y = -x,  yaw unchanged
(yaw 0 looks along -z = +X; a left turn from there faces -x = +Y). Every MapMaD 2D quantity (traj_data.pkl,
the per-drive maps, the expert driver) uses this frame.
"""

import math
from typing import Sequence, Tuple

import numpy as np


def to_2d(position: Sequence[float]) -> np.ndarray:
    """Habitat (x, y, z) -> NoMaD (X, Y) = (-z, -x). Works on (3,) or (N, 3)."""
    p = np.asarray(position, dtype=np.float64)
    return np.stack([-p[..., 2], -p[..., 0]], axis=-1)


def to_3d(xy: Sequence[float], height: float) -> np.ndarray:
    """NoMaD (X, Y) + Habitat height y -> Habitat (x, y, z) = (-Y, height, -X)."""
    xy = np.asarray(xy, dtype=np.float64)
    return np.array([-xy[1], height, -xy[0]])


def heading(yaw: float) -> np.ndarray:
    """Unit vector (X, Y) the robot faces in the 2D frame."""
    return np.array([math.cos(yaw), math.sin(yaw)])


def yaw_of(src: Sequence[float], dst: Sequence[float]) -> float:
    """Heading (2D frame) that points from src to dst, both (X, Y)."""
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def to_robot(points: np.ndarray, pose: Tuple[float, float, float]) -> np.ndarray:
    """(N, 2) world points -> robot frame (forward, left); same maths as NoMaD's to_local_coords."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(points, dtype=np.float64) - [x, y]
    return np.stack([d[..., 0] * c + d[..., 1] * s, -d[..., 0] * s + d[..., 1] * c], axis=-1)
