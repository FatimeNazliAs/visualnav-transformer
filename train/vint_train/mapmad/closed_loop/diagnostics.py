"""Stuck examples: what NoMaD was given and what it answered (inside naz_mapmad, after p1_replay.py frames).

    python -m vint_train.mapmad.closed_loop.diagnostics          # every <outputs>/p1_baseline/diagnostics/*/

Per example folder (<arm>__<episode>__<step>): runs the re-rendered frame_0..3.png (+ goal.png) through the
deployment `transform_images` exactly as NoMaD got them, saves the tensor (input_tensor.npy, (1, 12, 96, 96),
normalised) and panel.png: the 4 pictures at 96x96 (normalisation undone for display, i.e. the resized pictures),
the goal photo, sample 0's 8 waypoints top-down (robot at the origin facing up, waypoint #2 marked) and the
chosen (v, w) from the step log.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml
from PIL import Image as PILImage

from mapmad_bridge import steplog
from mapmad_sim.run_layout import P1_CONFIG, RunLayout
from vint_train.mapmad.closed_loop.deployment import transform_images
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
SCALE = 3  # 96 -> 288 px in the panel
PLOT = 288


def as_fed(tensor: np.ndarray) -> List[np.ndarray]:
    """(1, 3k, 96, 96) normalised tensor -> k RGB uint8 pictures (normalisation undone)."""
    out = []
    for k in range(tensor.shape[1] // 3):
        chw = tensor[0, 3 * k:3 * k + 3] * STD[:, None, None] + MEAN[:, None, None]
        out.append(np.clip(np.round(chw.transpose(1, 2, 0) * 255.0), 0, 255).astype(np.uint8))
    return out


def waypoint_plot(path_m: List[List[float]], chosen: List[float], v: float, w: float) -> np.ndarray:
    """Top-down: robot at the bottom centre facing up; x forward = up, y left = left; 1 m = PLOT/1.2 px."""
    img = np.full((PLOT, PLOT, 3), 255, np.uint8)
    px_per_m = PLOT / 1.2
    origin = np.array([PLOT // 2, PLOT - 30])

    def px(p):
        return tuple(int(v) for v in origin + np.array([-p[1], -p[0]]) * px_per_m)

    for r in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(img, tuple(int(v) for v in origin), int(r * px_per_m), (225, 225, 225), 1)
    cv2.arrowedLine(img, px([0, 0]), px([0.12, 0]), (0, 0, 0), 2, tipLength=0.4)
    pts = [px([0, 0])] + [px(p) for p in path_m]
    cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 120, 0), 2, cv2.LINE_AA)
    for i, p in enumerate(path_m):
        cv2.circle(img, px(p), 6 if i == 2 else 3, (0, 0, 255) if i == 2 else (255, 120, 0), -1, cv2.LINE_AA)
    cv2.putText(img, f"v {v:.2f} m/s  w {w:+.2f} rad/s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"wp#2 ({chosen[0]:.3f}, {chosen[1]:+.3f}) m", (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return img


def step_line(log: Path, step: int) -> Dict[str, Any]:
    return next(s for s in steplog.steps(steplog.read(log)) if s["step"] == step)


def panel(folder: Path, layout: RunLayout, image_size: List[int]) -> None:
    arm, episode, step = folder.name.split("__")
    pictures = [PILImage.open(folder / f"frame_{k}.png").convert("RGB") for k in range(4)]
    tensor = transform_images(pictures, image_size, center_crop=False).numpy()
    np.save(folder / "input_tensor.npy", tensor)
    fed = as_fed(tensor)
    tiles = [cv2.resize(f, (96 * SCALE, 96 * SCALE), interpolation=cv2.INTER_NEAREST) for f in fed]
    goal_path = folder / "goal.png"
    if goal_path.exists():
        goal = as_fed(transform_images(PILImage.open(goal_path).convert("RGB"), image_size).numpy())[0]
        tiles.append(cv2.resize(goal, (96 * SCALE, 96 * SCALE), interpolation=cv2.INTER_NEAREST))
    line = step_line(layout.log(arm, episode), int(step))
    tiles.append(waypoint_plot(line["policy"]["path_m"], line["policy"]["waypoint_m"], line["v"], line["w"]))
    labels = ["t-3", "t-2", "t-1", "t (newest)"] + (["goal photo"] if goal_path.exists() else []) + ["sample 0"]
    for tile, label in zip(tiles, labels):
        cv2.putText(tile, label, (6, PLOT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (6, PLOT - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    sheet = np.hstack(tiles)
    title = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"{arm} | {episode} | command at step {step} (inside a stuck streak) | NoMaD input 4 x 96x96",
                (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(folder / "panel.png"), cv2.cvtColor(np.vstack([title, sheet]), cv2.COLOR_RGB2BGR))
    print(f"{folder.name}: v {line['v']:.3f} w {line['w']:+.3f} waypoint#2 {line['policy']['waypoint_m']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=P1_CONFIG)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    image_size = yaml.safe_load(layout.nomad_config.read_text())["image_size"]
    for folder in sorted((layout.root / "diagnostics").glob("*__*__*")):
        panel(folder, layout, image_size)


if __name__ == "__main__":
    main()
