"""Episode videos (mp4, H.264, headless): LIMO's camera on the left, a top-down map on the right, a caption bar.

Top-down map: walkable area of the start's floor (grey), target object box (red outline), target point (red dot)
with the 1 m success circle, start (green), driven path (blue), robot (yellow arrow). Photo arms show the goal
photo in the camera's top-left corner.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CAM_SCALE = 2  # 320x240 -> 640x480
MAP_PX = 480
BAR_PX = 40
GOAL_INSET = (160, 120)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, BLUE, YELLOW, GREY = (60, 200, 60), (230, 50, 50), (60, 120, 255), (255, 210, 0), (200, 200, 200)


class TopdownMap:
    """World (x, z) <-> picture pixels of the cropped, scaled walkable-area map."""

    def __init__(self, grid: np.ndarray, origin: Sequence[float], m_per_px: float,
                 keep_points: Sequence[Sequence[float]], margin_m: float = 1.0) -> None:
        rows, cols = np.nonzero(grid)
        if len(rows) == 0:
            rows, cols = np.array([0, grid.shape[0] - 1]), np.array([0, grid.shape[1] - 1])
        px = [self._grid_px(p, origin, m_per_px) for p in keep_points]
        m = int(margin_m / m_per_px)
        r0 = max(min(rows.min(), *(p[1] for p in px)) - m, 0)
        r1 = min(max(rows.max(), *(p[1] for p in px)) + m, grid.shape[0] - 1)
        c0 = max(min(cols.min(), *(p[0] for p in px)) - m, 0)
        c1 = min(max(cols.max(), *(p[0] for p in px)) + m, grid.shape[1] - 1)
        crop = grid[r0:r1 + 1, c0:c1 + 1]
        self.scale = MAP_PX / max(crop.shape)
        self.origin, self.m_per_px, self.r0, self.c0 = np.asarray(origin, float), m_per_px, r0, c0
        small = cv2.resize(crop.astype(np.uint8), (int(crop.shape[1] * self.scale), int(crop.shape[0] * self.scale)),
                           interpolation=cv2.INTER_NEAREST)
        self.base = np.full((MAP_PX, MAP_PX, 3), 40, np.uint8)
        self.base[:small.shape[0], :small.shape[1]][small > 0] = GREY

    @staticmethod
    def _grid_px(p: Sequence[float], origin: Sequence[float], m_per_px: float) -> Tuple[int, int]:
        return int((p[0] - origin[0]) / m_per_px), int((p[2] - origin[1]) / m_per_px)

    def px(self, p: Sequence[float]) -> Tuple[int, int]:
        """(col, row) of world point p = (x, y, z)."""
        c, r = self._grid_px(p, self.origin, self.m_per_px)
        return int((c - self.c0) * self.scale), int((r - self.r0) * self.scale)

    def metres(self, d: float) -> int:
        return max(1, int(d / self.m_per_px * self.scale))


def draw_map(tmap: TopdownMap, episode: Dict[str, Any], path: List[Sequence[float]], yaw: float,
             success_m: float) -> np.ndarray:
    img = tmap.base.copy()
    t = episode["target"]
    c, s = t["box_center"], t["box_size"]
    lo = tmap.px([c[0] - s[0] / 2, 0, c[2] - s[2] / 2])
    hi = tmap.px([c[0] + s[0] / 2, 0, c[2] + s[2] / 2])
    cv2.rectangle(img, lo, hi, RED, 2)
    cv2.circle(img, tmap.px(t["point"]), tmap.metres(success_m), RED, 1, cv2.LINE_AA)
    cv2.circle(img, tmap.px(t["point"]), 5, RED, -1, cv2.LINE_AA)
    cv2.circle(img, tmap.px(episode["start_position"]), 6, GREEN, -1, cv2.LINE_AA)
    if len(path) > 1:
        cv2.polylines(img, [np.array([tmap.px(p) for p in path], np.int32)], False, BLUE, 2, cv2.LINE_AA)
    here = np.array(tmap.px(path[-1]))
    tip = here + (np.array([-math.sin(yaw), -math.cos(yaw)]) * 14).astype(int)  # forward = -z
    cv2.arrowedLine(img, tuple(int(v) for v in here), tuple(int(v) for v in tip), YELLOW, 3, cv2.LINE_AA, tipLength=0.5)
    return img


def text(img: np.ndarray, s: str, org: Tuple[int, int], scale: float = 0.6, colour=WHITE) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def compose(rgb: np.ndarray, map_img: np.ndarray, caption: str, status: str, status_colour,
            goal_rgb: Optional[np.ndarray]) -> np.ndarray:
    """One video frame (RGB): caption bar over [camera | map]."""
    cam = cv2.resize(rgb, (rgb.shape[1] * CAM_SCALE, rgb.shape[0] * CAM_SCALE), interpolation=cv2.INTER_NEAREST)
    if goal_rgb is not None:
        inset = cv2.resize(goal_rgb, GOAL_INSET, interpolation=cv2.INTER_AREA)
        cam[4:4 + GOAL_INSET[1], 4:4 + GOAL_INSET[0]] = inset
        cv2.rectangle(cam, (4, 4), (4 + GOAL_INSET[0], 4 + GOAL_INSET[1]), WHITE, 1)
        text(cam, "goal photo", (8, GOAL_INSET[1] + 20), 0.5)
    height = max(cam.shape[0], MAP_PX)
    body = np.zeros((height, cam.shape[1] + MAP_PX, 3), np.uint8)
    body[:cam.shape[0], :cam.shape[1]] = cam
    body[:MAP_PX, cam.shape[1]:] = map_img
    bar = np.zeros((BAR_PX, body.shape[1], 3), np.uint8)
    text(bar, caption, (8, 26))
    text(bar, status, (body.shape[1] - 150, 26), 0.7, status_colour)
    return np.vstack([bar, body])


class VideoWriter:
    """Streams RGB frames into an H.264 mp4 (imageio-ffmpeg's bundled ffmpeg)."""

    def __init__(self, path: Path, size: Tuple[int, int], fps: float) -> None:
        import imageio_ffmpeg

        path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.writer = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                                  macro_block_size=8, quality=None, output_params=["-crf", "23"])
        self.writer.send(None)

    def write(self, frame: np.ndarray) -> None:
        assert frame.shape[1::-1] == self.size, f"frame {frame.shape} != {self.size}"
        self.writer.send(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.writer.close()
