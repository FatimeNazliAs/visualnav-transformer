"""Stop-T picture check (Phase 3 confirmation item 13): 16 samples, each as one row of four tiles:
camera frame (frame t) | obstacle | explored | heat -- with the expert's next 8 waypoints (at the config's
waypoint spacing) drawn on the three map tiles and a yellow forward arrow at the robot.

What to check: the path runs through explored free space (left/right and frame index right) and bends toward
the heat; for targets outside the window the heat sits on the border in the target's direction.

Mix (deterministic from the frozen offline list): all 5 Habitat modes (photo / explore: the model does not see the
map, the tiles show what it would be), spot and object targets, 4 targets outside the window, 2 perturbed heats
(shift, false blob; these exist only in training).

    python -m vint_train.mapmad.sample_sheet --config config/mapmad.yaml \
        --list /outputs/mapmad/p3_model/offline/offline_samples.json --out /outputs/mapmad/p3_model/sample_sheet.png
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vint_train.mapmad import modes as M
from vint_train.mapmad.config import load_config
from vint_train.mapmad.dataset import build_mapmad_dataset
from vint_train.mapmad.heat import is_outside
from vint_train.mapmad.map_viz import local_px

TILE = 192
SEED = 13
# (mode, perturbation) per sheet row, and which rows must have the target outside the window
PLAN: List[Tuple[str, Optional[str]]] = [
    ("map_goal", None), ("map_goal", None), ("map_goal", None), ("map_goal", None),
    ("photo_map", None), ("photo_map", None), ("photo_map", None), ("explore_map", None),
    ("explore_map", None), ("photo", None), ("photo", None), ("explore", None),
    ("map_goal", "shift"), ("map_goal", "false_blob"), ("map_goal", None), ("photo_map", None),
]
OUTSIDE_ROWS = {0, 4, 8, 14}


def gray_tile(layer: np.ndarray, colour: Tuple[int, int, int]) -> np.ndarray:
    """0-1 layer -> RGB tile: black background, `colour` scaled by the value."""
    img = (layer[..., None] * np.array(colour, dtype=np.float32)[None, None]).astype(np.uint8)
    return cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_NEAREST)


def draw_path(img: np.ndarray, waypoints_m: np.ndarray, size: int, res: float) -> None:
    """Expert path (forward, left) in metres as a cyan polyline with dots, plus the robot's forward arrow."""
    pts = [local_px(0.0, 0.0, size, res, TILE)] + [local_px(f, l, size, res, TILE) for f, l in waypoints_m]
    pts = np.round(np.array(pts)).astype(np.int32)
    cv2.polylines(img, [pts], False, (0, 255, 255), 1, cv2.LINE_AA)
    for p in pts[1:]:
        cv2.circle(img, tuple(int(v) for v in p), 2, (0, 255, 255), -1)
    c = tuple(int(v) for v in pts[0])
    cv2.arrowedLine(img, c, (c[0], c[1] - 14), (255, 220, 0), 2, tipLength=0.4)


def label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def pick_samples(ds, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """16 listed samples matching PLAN's outside/inside rows, alternating spot/object (seeded order)."""
    order = np.random.default_rng(SEED).permutation(len(samples))
    chosen, used = [], set()
    for row in range(len(PLAN)):
        want_outside, want_type = row in OUTSIDE_ROWS, ("spot", "object")[row % 2]
        for j in order:
            s = samples[int(j)]
            if s["drive"] in used or s["type"] != want_type:
                continue
            traj = ds._get_trajectory(s["drive"])
            pose = (float(traj["position"][s["t"]][0]), float(traj["position"][s["t"]][1]), float(traj["yaw"][s["t"]]))
            if is_outside(ds.builder.store.targets[s["drive"]], pose, ds.map_size, ds.builder.resolution) == want_outside:
                chosen.append(dict(s, outside=want_outside))
                used.add(s["drive"])
                break
    return chosen


def sheet(cfg: Dict[str, Any], samples: List[Dict[str, Any]]) -> np.ndarray:
    ds = build_mapmad_dataset(cfg, "habitat_mapmad", "test", train=False)
    spacing_m = ds.data_config["metric_waypoint_spacing"] * ds.waypoint_spacing
    size, res = ds.map_size, ds.builder.resolution
    rows = []
    for row, s in enumerate(pick_samples(ds, samples)):
        mode_name, pert = PLAN[row]
        mode = M.BY_NAME[mode_name]
        shown = M.Mode(mode.id, mode.name, mode.goal_mask, 0, True)  # draw map + heat even where the model lacks them
        out = ds.habitat_sample(s["drive"], s["t"], shown, np.random.default_rng([SEED, row]), goal_offset=8
                                if mode.goal_mask == 0 else None, perturbation=pert)
        cam = cv2.cvtColor(cv2.imread(os.path.join(ds.data_folder, s["drive"], f"{s['t']}.jpg")), cv2.COLOR_BGR2RGB)
        cam = cv2.resize(cam, (TILE, TILE), interpolation=cv2.INTER_AREA)  # 320 x 240 squeezed to the tile
        layers = out[7].numpy()
        waypoints = out[2].numpy()[:, :2] * spacing_m  # normalised -> metres (forward, left)
        tiles = [cam, gray_tile(layers[0], (230, 60, 60)), gray_tile(layers[1], (220, 220, 220)),
                 gray_tile(layers[2], (255, 150, 30))]
        for t in tiles[1:]:
            draw_path(t, waypoints, size, res)
        hidden = " MAP HIDDEN" if mode.map_mask else ""
        label(tiles[0], f"{row}: {mode_name}{hidden}")
        label(tiles[1], f"{s['type']} {'OUTSIDE' if s['outside'] else 'inside'}")
        label(tiles[2], f"{s['drive']} t={s['t']}")
        label(tiles[3], f"heat {'(' + pert + ')' if pert else ''}{'' if mode.heat_on else ' off for model'}")
        rows.append(np.concatenate(tiles, axis=1))
    half = len(rows) // 2
    left, right = np.concatenate(rows[:half], axis=0), np.concatenate(rows[half:], axis=0)
    gap = np.full((left.shape[0], 8, 3), 255, np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    with open(args.list) as f:
        samples = json.load(f)["samples"]
    img = sheet(cfg, samples)
    cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("saved", args.out, img.shape)


if __name__ == "__main__":
    main()
