"""Phase 7.1: sha256 of pinhole renders at fixed poses, so test_lens.py can prove the lens option leaves the default
camera byte-identical.

    python mapmad/tests/p7_render_reference.py            # writes mapmad/tests/data/p7_pinhole_render_sha256.json

Run once with the code from before the lens option (`mapmad` @ e97b875); the test re-renders the same poses with the
current code and compares. Dev home 00746 (Phase 1 held-out train home, never an exam home).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

HOME = "00746-RTV2n6fXB2w"
SPLIT = "train"
N_POSES = 6
POSE_SEED = 7
OUT = Path(__file__).resolve().parent / "data" / "p7_pinhole_render_sha256.json"


def reference_poses(robot, n: int = N_POSES, seed: int = POSE_SEED) -> List[Dict[str, object]]:
    """n navigable floor points with headings, drawn from the robot's pathfinder with a fixed seed."""
    import math

    import numpy as np

    robot.pathfinder.seed(seed)
    rng = np.random.default_rng(seed)
    poses = []
    while len(poses) < n:
        p = np.asarray(robot.pathfinder.get_random_navigable_point(), dtype=np.float64)
        try:
            robot.place(p, 0.0)
        except ValueError:
            continue
        poses.append({"position": [round(float(v), 6) for v in p], "yaw": round(float(rng.uniform(-math.pi, math.pi)), 6)})
    return poses


def render_hashes(robot, poses: List[Dict[str, object]]) -> List[str]:
    """sha256 of the RGB picture at every pose."""
    out = []
    for pose in poses:
        robot.place(pose["position"], pose["yaw"])
        out.append(hashlib.sha256(robot.observe()["rgb"].tobytes()).hexdigest())
    return out


def main() -> None:
    from mapmad_sim import config
    from mapmad_sim.robot import LimoSim, RobotSpec

    with LimoSim(SPLIT, HOME, RobotSpec.from_config(config.robot()), gpu=0) as robot:
        poses = reference_poses(robot)
        hashes = render_hashes(robot, poses)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"home": HOME, "split": SPLIT, "poses": poses, "rgb_sha256": hashes}, indent=1) + "\n")
    print(OUT, hashes)


if __name__ == "__main__":
    main()
