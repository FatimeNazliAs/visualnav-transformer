"""Phase 8.0: camera_fair.yaml (pin025) loads through the simulator code; the map's depth camera moves with it.

    python mapmad/scripts/p8_camera_check.py --gpu 1 --home 00746     # inside naz_mapmad_habitat, a dev home

At 20 random real-floor poses: colour and depth camera have the same pose; camera height above the REAL floor = 0.25 m;
pictures 320 x 240; HFOV 66.5 deg; pitch -2 deg; LIMO floor map (radius 0.195 m, height 0.55 m) unchanged.
Writes <outputs>/p8_search_data/checks/camera_check.json. Never an exam home.
"""

import argparse
import json
import math

import numpy as np

from mapmad_sim import config, home_splits
from mapmad_sim.datagen_v2 import robot_spec
from mapmad_sim.drive_sampler import DriveSampler
from mapmad_sim.robot import LimoSim
from mapmad_sim.run_layout import read_config

CONFIG = config.CONFIG_DIR / "p8_datagen.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--home", default="00746")
    args = p.parse_args()
    cfg = read_config(CONFIG)
    home = home_splits.by_prefix(home_splits.train_homes("train"), [args.home])[0]
    spec = robot_spec(cfg)
    nav = dict(spec.navmesh)
    rows = []
    with LimoSim("train", home, spec, semantic=False, depth=True, gpu=args.gpu, seed=0) as robot:
        sampler = DriveSampler(robot, cfg, np.random.default_rng(0), None)
        for _ in range(20):
            s = sampler.start()
            robot.place(np.array(s.position), s.yaw)
            obs = robot.observe()
            tf = {u: np.array(robot.sim._sensors[u]._sensor_object.node.absolute_transformation()) for u in robot.camera_uuids}
            rows.append({"height_m": robot.camera_height_above_floor,
                         "depth_equals_colour_pose": bool(np.allclose(tf["depth_sensor"], tf["color_sensor"])),
                         "rgb_shape": list(obs["rgb"].shape), "depth_shape": list(obs["depth"].shape)})
    heights = [r["height_m"] for r in rows]
    out = {"home": home, "camera": cfg["camera"], "spec": {"hfov_deg": spec.hfov_deg, "width": spec.width, "height": spec.height,
           "camera_height_m": spec.camera_height_m, "pitch_deg": spec.pitch_deg, "forward_m": spec.camera_forward_m,
           "navmesh_radius_m": nav["agent_radius_m"], "navmesh_height_m": nav["agent_height_m"]},
           "height_min_max": [min(heights), max(heights)], "poses": rows,
           "pass": all(r["depth_equals_colour_pose"] for r in rows) and max(abs(h - 0.25) for h in heights) < 1e-3
           and all(r["rgb_shape"] == [240, 320, 3] and r["depth_shape"] == [240, 320] for r in rows)
           and math.isclose(spec.hfov_deg, 66.5) and spec.pitch_deg == -2.0
           and nav["agent_radius_m"] == 0.195 and nav["agent_height_m"] == 0.55}
    path = config.paths()["outputs"] / cfg["out_dir"] / "checks" / "camera_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("spec", "height_min_max", "pass")}))


if __name__ == "__main__":
    main()
