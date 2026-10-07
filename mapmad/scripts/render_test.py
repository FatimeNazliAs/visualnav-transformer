"""Phase 0 test render (gate G0 row 2): colour, depth and label pictures from inside one HM3D home.

    python mapmad/scripts/render_test.py                       # minival home 00800-TEEsavR23oF
    python mapmad/scripts/render_test.py --home 00808-y9hTuugGdiq --seed 3

Four views: one looking at the kitchen (refrigerator or stove, seen from 1.5-3.5 m) and three random
walkable spots with a random heading. The camera is LIMO's (configs/robot_limo.yaml): height above the
floor, horizontal FOV. Each view is rendered at every --resolutions size and saved as
  rgb.png, depth.png (colour scale 0-10 m) + depth.npy (metres), semantic.png (one colour per object)
  + semantic.npy (object ids), overlay.png (labels on top of the colour picture, largest objects named),
plus one sheet_<WxH>.png with all views side by side, objects.csv (every labelled object of the home)
and views.csv with alignment hints per view (hints, not a gate; Naz checks the pictures):
  - unlabelled_share: pixels that belong to no object (id 0) or to category "unknown";
  - floor_flat_share: of the pixels labelled "floor", the share whose 3D point (from depth) lies within
    5 cm of their median height. Labels shifted against the mesh spill "floor" onto walls and furniture
    and push this down (habitat-starter gotcha 1: 12% vs 97% for MP3D without its config);
  - floor_below_feet_m: how far that floor plane lies below the agent's feet. The feet stand on the
    navmesh, which can float up to one navmesh cell (0.2 m) above the floor mesh, so the camera is
    this much higher above the visible floor than sensor_height says.
"""

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_starter import SimSettings, make_sim
from habitat_starter.semantics import box_center_size
from matplotlib import colormaps

from mapmad_sim import config
from mapmad_sim.run_info import save_run_info

KITCHEN_TARGETS = ("refrigerator", "oven and stove", "stove", "oven", "stovetop")
FLOOR_TOLERANCE_M = 0.05
DEPTH_MAX_M = 10.0
TARGET_SHARE = (0.03, 0.35)  # the kitchen target covers 3-35% of the kitchen view: visible, not filling it
TARGET_DIST_M = (1.5, 3.5)  # floor distance from the camera to the target's box centre
NEAR_M, MAX_NEAR_SHARE = 1.0, 0.5  # random views: at most half of the picture closer than 1 m


def yaw_towards(src: np.ndarray, dst: np.ndarray) -> float:
    """Heading (rotation about y) that points the camera from src at dst; cameras look along -z."""
    d = dst - src
    return math.atan2(-d[0], -d[2])


def kitchen_view(sim: habitat_sim.Simulator) -> Tuple[np.ndarray, float, str]:
    """A walkable spot TARGET_DIST_M from a refrigerator/stove on the same floor, facing it, with the
    target covering TARGET_SHARE of the picture (nothing hides it, and it doesn't fill the view)."""
    objects = [o for o in sim.semantic_scene.objects if o is not None and o.category is not None]
    for name in KITCHEN_TARGETS:
        for obj in (o for o in objects if o.category.name() == name):
            center, _ = box_center_size(obj.aabb)
            for _ in range(300):
                p = np.array(sim.pathfinder.get_random_navigable_point_near(center, TARGET_DIST_M[1], 100))
                if np.isnan(p).any():
                    continue
                flat = math.hypot(*(center - p)[[0, 2]])
                if not (TARGET_DIST_M[0] <= flat <= TARGET_DIST_M[1] and 0.0 <= center[1] - p[1] <= 1.5):
                    continue
                yaw = yaw_towards(p, center)
                share = (place_agent(sim, p, yaw)["semantic_sensor"] == obj.semantic_id).mean()
                if TARGET_SHARE[0] <= share <= TARGET_SHARE[1]:
                    return p, yaw, f"kitchen ({name} {obj.id})"
    raise RuntimeError(f"no refrigerator/stove view matching {TARGET_SHARE=} {TARGET_DIST_M=}; try another --home")


def random_views(sim: habitat_sim.Simulator, rng: np.random.Generator, n: int) -> List[Tuple[np.ndarray, float, str]]:
    """n walkable spots with a random heading, skipping views that mostly face a wall up close."""
    views = []
    while len(views) < n:
        p = np.array(sim.pathfinder.get_random_navigable_point())
        if np.isnan(p).any():
            continue
        yaw = float(rng.uniform(-math.pi, math.pi))
        if (place_agent(sim, p, yaw)["depth_sensor"] < NEAR_M).mean() <= MAX_NEAR_SHARE:
            views.append((p, yaw, f"random {len(views) + 1}"))
    return views


def place_agent(sim: habitat_sim.Simulator, position: np.ndarray, yaw: float) -> Dict[str, np.ndarray]:
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = position
    state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)
    return sim.get_sensor_observations()


def instance_colours(n: int) -> np.ndarray:
    """A fixed random colour per object id (id 0 = no object = black)."""
    colours = np.random.default_rng(0).integers(40, 256, size=(n + 1, 3), dtype=np.uint8)
    colours[0] = 0
    return colours


def floor_check(depth: np.ndarray, floor_mask: np.ndarray, settings: SimSettings) -> Tuple[Optional[float], Optional[float]]:
    """(floor_flat_share, floor_below_feet_m) for the floor-labelled pixels; (None, None) if there are none.

    Each pixel's height above the agent's feet comes from its depth (camera level, no pitch):
    height = sensor_height - (row - cy) * depth / f, since image rows go down and the camera's y goes up.
    """
    if floor_mask.sum() == 0:
        return None, None
    f = settings.intrinsics()[0, 0]
    rows = np.arange(depth.shape[0], dtype=np.float32)[:, None] - settings.height / 2.0
    heights = (settings.sensor_height - rows * depth / f)[floor_mask]
    median = float(np.median(heights))
    return float((np.abs(heights - median) <= FLOOR_TOLERANCE_M).mean()), -median


def overlay(rgb: np.ndarray, sem_rgb: np.ndarray, sem: np.ndarray, names: Dict[int, str], top: int = 6) -> np.ndarray:
    """Colour picture with the label colours blended on top and the largest objects named."""
    out = cv2.addWeighted(rgb, 0.5, sem_rgb, 0.5, 0)
    ids, counts = np.unique(sem, return_counts=True)
    scale = rgb.shape[1] / 640.0
    for obj_id in ids[np.argsort(-counts)][:top]:
        if obj_id == 0:
            continue
        ys, xs = np.nonzero(sem == obj_id)
        x, y = int(np.median(xs)), int(np.median(ys))
        label = names.get(int(obj_id), "?")
        for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale + 0.1, colour, width, cv2.LINE_AA)
    return out


def alignment_hints(sem: np.ndarray, depth: np.ndarray, unknown: np.ndarray, floor: np.ndarray,
                    settings: SimSettings) -> Dict[str, object]:
    """The views.csv hint columns for one view (see the module docstring); ids in `unknown`/`floor` are object ids."""
    floor_mask = np.isin(sem, floor)
    flat_share, below_feet = floor_check(depth, floor_mask, settings)
    return {
        "no_object_share": round(float((sem == 0).mean()), 4),
        "unknown_share": round(float(np.isin(sem, unknown).mean()), 4),
        "unlabelled_share": round(float(((sem == 0) | np.isin(sem, unknown)).mean()), 4),
        "floor_pixels": int(floor_mask.sum()),
        "floor_flat_share": None if flat_share is None else round(flat_share, 4),
        "floor_below_feet_m": None if below_feet is None else round(below_feet, 3),
    }


def save_view(view_dir: Path, rgb: np.ndarray, depth: np.ndarray, sem: np.ndarray, colours: np.ndarray,
              names: Dict[int, str]) -> np.ndarray:
    """Write rgb/depth/semantic/overlay pictures + depth/semantic arrays of one view; returns its sheet row."""
    sem_rgb = colours[np.clip(sem, 0, len(colours) - 1)]
    depth_rgb = (colormaps["turbo"](np.clip(depth / DEPTH_MAX_M, 0, 1))[..., :3] * 255).astype(np.uint8)
    over = overlay(rgb, sem_rgb, sem, names)
    view_dir.mkdir(parents=True, exist_ok=True)
    for name, image in (("rgb", rgb), ("depth", depth_rgb), ("semantic", sem_rgb), ("overlay", over)):
        save_png(view_dir / f"{name}.png", image)
    np.save(view_dir / "depth.npy", depth.astype(np.float32))
    np.save(view_dir / "semantic.npy", sem.astype(np.int32))
    return np.concatenate([rgb, depth_rgb, sem_rgb, over], axis=1)


def save_png(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def write_objects_csv(sim: habitat_sim.Simulator, path: Path) -> Dict[int, str]:
    """Every labelled object: id, category, box centre and size (metres, Habitat world frame)."""
    names = {}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "object", "category", "center_x", "center_y", "center_z",
                    "size_x", "size_y", "size_z", "region"])
        for obj in sim.semantic_scene.objects:
            if obj is None or obj.category is None:
                continue
            c, s = box_center_size(obj.aabb)
            names[obj.semantic_id] = obj.category.name()
            w.writerow([obj.semantic_id, obj.id, obj.category.name(), *np.round(c, 3), *np.round(s, 3),
                        obj.region.id if obj.region is not None else ""])
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", default="00800-TEEsavR23oF", help="labelled minival home")
    p.add_argument("--split", default="minival")
    p.add_argument("--resolutions", nargs="+", default=["320x240", "640x480"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", type=Path, help="default: <outputs>/p0_setup/render/<home>")
    args = p.parse_args()

    robot = config.robot()["camera"]
    out = args.out or config.paths()["outputs"] / "p0_setup" / "render" / args.home
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    views: Sequence[Tuple[np.ndarray, float, str]] = ()
    rows = []
    for res in args.resolutions:
        width, height = map(int, res.split("x"))
        settings = config.hm3d_sim_settings(
            args.split, args.home, width=width, height=height, hfov=float(robot["hfov_deg"]),
            sensor_height=float(robot["height_m"]), semantic_sensor=True, gpu_device_id=args.gpu, seed=args.seed,
        )
        with make_sim(settings) as sim:
            if not views:  # pick the views once, reuse them for every resolution
                names = write_objects_csv(sim, out / "objects.csv")
                sim.pathfinder.seed(args.seed)
                kitchen = kitchen_view(sim)
                # reseed: the random views must not depend on how many points the kitchen search drew
                sim.pathfinder.seed(args.seed + 1)
                views = [kitchen, *random_views(sim, rng, 3)]
            colours = instance_colours(max(names) + 1)
            unknown = np.array([i for i, n in names.items() if n == "unknown"], dtype=np.int64)
            floor = np.array([i for i, n in names.items() if n == "floor"], dtype=np.int64)
            sheet = []
            for k, (position, yaw, label) in enumerate(views):
                obs = place_agent(sim, position, yaw)
                rgb = obs["color_sensor"][..., :3]
                depth = obs["depth_sensor"]
                sem = obs["semantic_sensor"].astype(np.int64)
                sheet.append(save_view(out / res / f"view{k}", rgb, depth, sem, colours, names))
                rows.append({
                    "resolution": res, "view": k, "label": label,
                    "x": round(float(position[0]), 3), "y": round(float(position[1]), 3), "z": round(float(position[2]), 3),
                    "yaw_deg": round(math.degrees(yaw), 1),
                    **alignment_hints(sem, depth, unknown, floor, settings),
                })
                print(rows[-1], flush=True)
            save_png(out / f"sheet_{res}.png", np.concatenate(sheet, axis=0))
    with open(out / "views.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_run_info(out, {**vars(args), "camera": robot}, args.seed)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
