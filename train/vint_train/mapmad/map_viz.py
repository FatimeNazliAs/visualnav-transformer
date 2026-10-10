"""Pictures of a 64 x 64 local map (local_map.py) for videos and checks. numpy + OpenCV only (both containers).

Colours: unknown = dark grey, explored = light grey, obstacle = red; the robot = yellow triangle in the centre
pointing up (forward); optional: camera field of view (two lines), a target marker (blue dot, or a blue arrow on
the border when the target lies outside the window, the straight-line direction as in plan D5).
"""

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

UNKNOWN, EXPLORED, OBSTACLE = (70, 70, 70), (225, 225, 225), (215, 30, 30)
ROBOT, FOV, TARGET = (255, 210, 0), (255, 140, 0), (40, 110, 255)


def render(layers: np.ndarray, px: int = 480) -> np.ndarray:
    """(2, S, S) obstacle/explored layers -> RGB picture px x px (nearest-neighbour upscale)."""
    obstacle, explored = layers[0] > 0.5, layers[1] > 0.5
    img = np.empty(obstacle.shape + (3,), np.uint8)
    img[:] = UNKNOWN
    img[explored] = EXPLORED
    img[obstacle] = OBSTACLE
    return cv2.resize(img, (px, px), interpolation=cv2.INTER_NEAREST)


def local_px(forward: float, left: float, size: int, resolution: float, px: int) -> Tuple[float, float]:
    """Picture (x, y) pixel of a robot-frame point (forward, left) in a rendered local map."""
    scale = px / size
    return (size / 2.0 - left / resolution) * scale, (size / 2.0 - forward / resolution) * scale


def draw_robot(img: np.ndarray, size: int, hfov_deg: Optional[float] = None, fov_m: float = 3.0,
               resolution: float = 0.1) -> None:
    px = img.shape[0]
    cx, cy = local_px(0.0, 0.0, size, resolution, px)
    if hfov_deg:
        for side in (-1, 1):
            a = side * math.radians(hfov_deg) / 2.0
            x, y = local_px(fov_m * math.cos(a), fov_m * math.sin(a), size, resolution, px)
            cv2.line(img, (int(cx), int(cy)), (int(x), int(y)), FOV, 1, cv2.LINE_AA)
    s = px / size * 1.5
    tri = np.array([[cx, cy - 1.6 * s], [cx - s, cy + s], [cx + s, cy + s]], np.int32)
    cv2.fillPoly(img, [tri], ROBOT)


def draw_target(img: np.ndarray, target_robot: Sequence[float], size: int, resolution: float = 0.1) -> None:
    """Target at robot-frame (forward, left): a dot inside the window, else an arrow tip on its border."""
    px = img.shape[0]
    f, l = float(target_robot[0]), float(target_robot[1])
    half = size / 2.0 * resolution
    if max(abs(f), abs(l)) < half:
        x, y = local_px(f, l, size, resolution, px)
        cv2.circle(img, (int(x), int(y)), max(4, px // 60), TARGET, -1, cv2.LINE_AA)
        return
    k = (half - resolution / 2) / max(abs(f), abs(l))
    x, y = local_px(f * k, l * k, size, resolution, px)
    cx, cy = local_px(0.0, 0.0, size, resolution, px)
    d = np.array([x - cx, y - cy]) / max(math.hypot(x - cx, y - cy), 1e-9)
    tail = (int(x - d[0] * px / 12), int(y - d[1] * px / 12))
    cv2.arrowedLine(img, tail, (int(x), int(y)), TARGET, 3, cv2.LINE_AA, tipLength=0.4)
