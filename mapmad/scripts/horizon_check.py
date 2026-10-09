"""G0 row 4: match the simulated camera's horizon to the real LIMO camera's.

    python mapmad/scripts/horizon_check.py                 # frames in /outputs/mapmad/p0_setup/robot/limo_frames

1. Real horizon (colour): in each color_*.png, find straight lines on the floor and along the wall's
   bottom edge (LSD, lower part of the picture) and where they meet; that row is the horizon.
   Cross-check: fit the floor plane in depth_raw_16bit.png -> camera height, pitch -> horizon row.
2. Sim pitch: Habitat's camera is a centred pinhole, so pitch = atan((real_row - 240) / f).
3. Sim check: in a labelled HM3D home, pick a view with much floor in sight, put the camera
   camera.height_m above the REAL floor (the navmesh floats above it), pitch it, render 640x480 and
   fit the floor plane in its depth -> the rendered horizon row (should equal the real one).
Writes horizon_check.json, horizon_check.png (real | sim, horizon drawn) and run_info.json.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_starter import make_sim

from mapmad_sim import camera, config
from mapmad_sim.run_info import save_run_info

MIN_SEGMENT_PX = 40.0
FLOOR_TOP_ROW = 235  # real colour: only lines below this row (floor, wall bottom edge)
MIN_SLOPE = 0.08  # |sin| of the line angle; skips the horizontal lines of furniture
DEPTH_FLOOR_TOP_ROW = 300  # real depth (400 rows): rows below this see only floor in these frames
CANDIDATE_VIEWS = 60
FAR_M = 3.0


def real_horizon_from_lines(path: Path) -> Tuple[float, np.ndarray]:
    """Horizon row of one real colour frame (where its floor lines meet) and the line segments used."""
    grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    segs = cv2.createLineSegmentDetector().detect(grey)[0][:, 0].astype(np.float64)
    d = segs[:, 2:] - segs[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    pick = (length >= MIN_SEGMENT_PX) & (np.minimum(segs[:, 1], segs[:, 3]) >= FLOOR_TOP_ROW) \
        & (np.abs(d[:, 1]) / length >= MIN_SLOPE)
    point, keep = camera.vanishing_point(segs[pick])
    return float(point[1]), segs[pick][keep]


def real_horizon_from_depth(path: Path, depth_cam: Dict, colour_cam: Dict) -> Dict[str, float]:
    """Floor-plane fit of the real depth frame; its pitch gives the horizon row in the colour picture."""
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
    mask = np.zeros(depth.shape, dtype=bool)
    mask[DEPTH_FLOOR_TOP_ROW:] = True
    fit = camera.floor_plane(depth, mask, depth_cam["fx"], depth_cam["cx"], depth_cam["cy"])
    fit["colour_horizon_row"] = camera.horizon_row(fit["pitch_deg"], colour_cam["fy"], colour_cam["cy"])
    return fit


def set_camera_pose(sim: habitat_sim.Simulator, height_above_feet: float, pitch_deg: float) -> None:
    """Move every sensor to this height above the agent's feet and pitch it (+ = up)."""
    for sensor in sim.get_agent(0)._sensors.values():
        sensor.node.translation = mn.Vector3(0.0, height_above_feet, 0.0)
        sensor.node.rotation = mn.Quaternion.rotation(mn.Deg(pitch_deg), mn.Vector3.x_axis())


def observe(sim: habitat_sim.Simulator, position: np.ndarray, yaw: float) -> Dict[str, np.ndarray]:
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position, state.rotation = position, quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    agent.set_state(state, reset_sensors=False)
    return sim.get_sensor_observations()


def sim_floor_fit(obs: Dict[str, np.ndarray], floor_ids: np.ndarray, f: float, w: int, h: int) -> Dict[str, float]:
    return camera.floor_plane(obs["depth_sensor"], np.isin(obs["semantic_sensor"], floor_ids), f, w / 2.0, h / 2.0)


def sim_check(args: argparse.Namespace, cam: Dict, pitch_deg: float) -> Tuple[Dict[str, object], np.ndarray]:
    """Render one HM3D view with LIMO's camera height above the real floor and the sim pitch."""
    w, h = cam["width"], cam["height"]
    settings = config.hm3d_sim_settings(args.split, args.home, width=w, height=h, hfov=float(cam["hfov_deg"]),
                                        sensor_height=float(cam["height_m"]), semantic_sensor=True,
                                        gpu_device_id=args.gpu, seed=args.seed)
    f = camera.focal_px(w, cam["hfov_deg"])
    with make_sim(settings) as sim:
        floor_ids = np.array([o.semantic_id for o in sim.semantic_scene.objects
                              if o is not None and o.category is not None and o.category.name() == "floor"])
        sim.pathfinder.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        best: Tuple[int, np.ndarray, float] = (-1, np.zeros(3), 0.0)
        for _ in range(CANDIDATE_VIEWS):  # level camera, cam height above the feet
            p, yaw = np.array(sim.pathfinder.get_random_navigable_point()), float(rng.uniform(-math.pi, math.pi))
            obs = observe(sim, p, yaw)
            far_floor = int((np.isin(obs["semantic_sensor"], floor_ids) & (obs["depth_sensor"] > FAR_M)).sum())
            if far_floor > best[0]:
                best = (far_floor, p, yaw)
        _, p, yaw = best
        level = sim_floor_fit(observe(sim, p, yaw), floor_ids, f, w, h)
        below_feet = level["height_m"] - cam["height_m"]  # navmesh offset above the real floor
        set_camera_pose(sim, cam["height_m"] - below_feet, pitch_deg)
        obs = observe(sim, p, yaw)
        fit = sim_floor_fit(obs, floor_ids, f, w, h)
    info = {"home": args.home, "position": [round(float(x), 3) for x in p], "yaw_deg": round(math.degrees(yaw), 1),
            "floor_below_feet_m": round(below_feet, 3), "sensor_above_feet_m": round(cam["height_m"] - below_feet, 3),
            "pitch_deg": pitch_deg, "expected_horizon_row": camera.horizon_row(pitch_deg, f, h / 2.0),
            "fit": fit}
    return info, obs["color_sensor"][..., :3]


def draw(rgb: np.ndarray, row: float, title: str, segments: List[np.ndarray] = ()) -> np.ndarray:
    """BGR picture with the horizon as a red line, line segments used in yellow, and a caption."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.shape[2] == 3 else rgb.copy()
    for x1, y1, x2, y2 in segments:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2, cv2.LINE_AA)
    y = int(round(row))
    cv2.line(out, (0, y), (out.shape[1], y), (0, 0, 255), 1, cv2.LINE_AA)
    for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(out, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, width, cv2.LINE_AA)
        cv2.putText(out, f"horizon row {row:.1f} / {out.shape[0]}", (8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, width, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=Path, default=config.paths()["outputs"] / "p0_setup/robot/limo_frames")
    parser.add_argument("--out", type=Path, default=config.paths()["outputs"] / "p0_setup/robot")
    parser.add_argument("--split", default="minival")
    parser.add_argument("--home", default="00800-TEEsavR23oF")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    robot = config.robot()
    cam, depth_cam = robot["camera"], robot["depth"]
    rows, segments = {}, {}
    for path in sorted(args.frames.glob("color_[0-9].png")):
        rows[path.name], segments[path.name] = real_horizon_from_lines(path)
    real_row = float(np.mean(list(rows.values())))
    depth_fit = real_horizon_from_depth(args.frames / "depth_raw_16bit.png", depth_cam, cam)

    f_sim = camera.focal_px(cam["width"], cam["hfov_deg"])
    pitch = round(camera.pitch_for_horizon(real_row, f_sim, cam["height"] / 2.0), 1)
    sim, sim_rgb = sim_check(args, cam, pitch)

    result = {
        "real": {"horizon_row_lines": rows, "horizon_row": real_row, "segments_used": {k: len(v) for k, v in segments.items()},
                 "depth_fit": depth_fit, "physical_pitch_deg_from_lines": camera.pitch_for_horizon(real_row, cam["fy"], cam["cy"])},
        "sim_pitch_deg": pitch,
        "sim": sim,
        "difference_rows": sim["fit"]["horizon_row"] - real_row,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "horizon_check.json").write_text(json.dumps(result, indent=2) + "\n")
    first = sorted(rows)[0]
    real_rgb = cv2.cvtColor(cv2.imread(str(args.frames / first)), cv2.COLOR_BGR2RGB)
    sheet = np.hstack([draw(real_rgb, real_row, f"real LIMO ({first})", segments[first]),
                       draw(sim_rgb, sim["fit"]["horizon_row"], f"sim {args.home}, pitch {pitch:+.1f} deg, 0.18 m")])
    cv2.imwrite(str(args.out / "horizon_check.png"), sheet)
    save_run_info(args.out, {**vars(args), "frames": str(args.frames), "out": str(args.out)}, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
