"""Phase 7.4: Method B on simulator pictures with a known camera height (helper for test_floor_height.py and the
height report). Dev home 00746."""

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

HOME, SPLIT = "00746-RTV2n6fXB2w", "train"


def synthetic_pairs(height_m: float, pitch_deg: float, fisheye=None, n_pairs: int = 6, steps: int = 6,
                    seed: int = 11) -> List[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]]:
    """(grey 1, grey 2, odometry robot-2-in-robot-1) for n_pairs drives of `steps` x 0.05 m straight ahead."""
    import cv2

    from mapmad_sim import config
    from mapmad_sim.floor_height import relative_pose
    from mapmad_sim.frames import to_2d
    from mapmad_sim.robot import LimoSim, RobotSpec

    spec = dataclasses.replace(RobotSpec.from_config(config.robot()), camera_height_m=height_m, pitch_deg=pitch_deg,
                               fisheye=fisheye, hfov_deg=90.0)
    out = []
    rng = np.random.default_rng(seed)
    with LimoSim(SPLIT, HOME, spec) as robot:
        robot.pathfinder.seed(seed)
        while len(out) < n_pairs:
            p = robot.pathfinder.get_random_navigable_point()
            try:
                robot.place(p, float(rng.uniform(-np.pi, np.pi)))
            except ValueError:
                continue
            g1 = cv2.cvtColor(robot.observe()["rgb"], cv2.COLOR_RGB2GRAY).astype(np.float32)
            xy1, yaw1 = to_2d(robot.position), robot.yaw
            moved = sum(robot.drive(0.2, 0.0).travelled_m for _ in range(steps))
            if moved < 0.8 * 0.05 * steps:  # blocked by a wall: skip
                continue
            g2 = cv2.cvtColor(robot.observe()["rgb"], cv2.COLOR_RGB2GRAY).astype(np.float32)
            out.append((g1, g2, relative_pose(xy1, yaw1, to_2d(robot.position), robot.yaw)))
    return out
