"""The exam-time map (Phase 5 block item 2): the training map, built online while the robot drives.

Same code as the practice drives, imported -- not copied:
- every control step, the depth picture at the robot's pose goes into a whole-storey `depth_map.FirstSeenMap` with
  `depth_map.world_points` exactly as Phase 2's drive loop does (`datagen.DriveRunner.record_frame`): band
  0.05-0.55 m above the REAL floor under the robot (floor probe), max 4 m, invalid depth and points > 10 cm below
  the floor = unknown, first-seen frame index per cell; numbers from p2_datagen.yaml `map`;
- the policy's 3 x 64 x 64 tensor is cut and drawn by the TRAINING code: `vint_train.mapmad.map_sample.
  MapSampleBuilder.build` (-> `local_map.local_map` + `heat.draw_heat`) on a store that serves this live map instead
  of a drive's mapmad_map.npz, with the train config's `map` block (64 cells, 0.1 m, heat sigma 0.3 m, border inset
  0.3 m) and no perturbation;
- the heat target is the episode's object box read with `heat.target_from_meta` (world-aligned box around the
  oriented box, as in mapmad_meta.json);
- pose = the true simulator pose (as in training; a limitation, logged): (X, Y) = frames.to_2d(turning centre),
  yaw = the robot's yaw.
The planner policy gets the whole-storey grids instead (`grid`): obstacle / explored as known at this step.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from mapmad_sim.depth_map import FirstSeenMap, MapSpec, world_points
from mapmad_sim.frames import to_2d

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "train") not in sys.path:
    sys.path.insert(0, str(REPO / "train"))  # numpy-only NoMaD-side modules (local_map, heat, map_sample)

from vint_train.mapmad import heat as heat_lib  # noqa: E402
from vint_train.mapmad.map_sample import MapSampleBuilder  # noqa: E402

LIVE = "live"  # the one "drive" the live store holds


class LiveStore:
    """Stands in for map_sample.DriveStore: one target and the live first-seen map instead of files on disk."""

    def __init__(self, target: heat_lib.Target, fmap: FirstSeenMap) -> None:
        self.targets = {LIVE: target}
        self.fmap = fmap

    def floor_map(self, drive: str) -> Dict[str, np.ndarray]:
        """The live map in mapmad_map.npz's layout (local_map.local_map reads the first-seen grids)."""
        return self.fmap.arrays(float("nan"))


def heat_target(episode: Dict[str, Any]) -> heat_lib.Target:
    """The episode's object footprint as the heat code reads a drive's mapmad_meta.json target."""
    t = episode["target"]
    return heat_lib.target_from_meta({"target": {"kind": "object", "box_center": t["box_center"],
                                                 "box_size": t["box_size"]}})


class OnlineMap:
    """The map of one episode, grown one depth picture per control step."""

    def __init__(self, robot: Any, episode: Dict[str, Any], map_spec: MapSpec, crop_cfg: Dict[str, Any]) -> None:
        low, high = robot.pathfinder.get_bounds()
        self.fmap = FirstSeenMap.around(map_spec, low, high)
        self.spec = map_spec
        self.store = LiveStore(heat_target(episode), self.fmap)
        self.builder = MapSampleBuilder(self.store, **dict(crop_cfg, perturb_prob=0.0))
        self.t = -1  # index of the last frame added

    def add_frame(self, robot: Any, depth: np.ndarray) -> None:
        """Frame t + 1: the depth picture at the robot's current pose (datagen.DriveRunner.record_frame's map part)."""
        self.t += 1
        cam = robot.camera_transform()
        self.fmap.add(world_points(depth, robot.spec.hfov_deg, self.spec.max_depth_m, cam[:3, 3], cam[:3, :3]),
                      robot.floor_y, self.t)

    @staticmethod
    def pose2d(robot: Any) -> Tuple[float, float, float]:
        """(X, Y, yaw) of the turning centre in the map's 2D frame (traj_data.pkl's frame)."""
        x, y = to_2d(robot.position)
        return float(x), float(y), float(robot.yaw)

    def tensor(self, robot: Any) -> np.ndarray:
        """float32 [3, 64, 64] for the policy: obstacle, explored (first seen <= t), heat (training code)."""
        return self.builder.build(LIVE, self.t, self.pose2d(robot), heat_on=True)

    def grid(self, robot: Any) -> Dict[str, Any]:
        """Whole-storey map for the planner: obstacle / explored (uint8, 1 = known), origin, resolution, pose."""
        known = {k: ((v >= 0) & (v <= self.t)).reshape(self.fmap.shape).astype(np.uint8)
                 for k, v in self.fmap.first_seen.items()}
        return {"obstacle": known["obstacle"], "explored": known["explored"],
                "origin": [float(v) for v in self.fmap.origin], "resolution": float(self.spec.resolution),
                "pose": list(self.pose2d(robot))}

    def goal_rect(self) -> Tuple[float, float, float, float]:
        """The target footprint (xmin, xmax, ymin, ymax) in the 2D frame: the planner's goal (block item 9 E)."""
        return tuple(float(v) for v in self.store.targets[LIVE].rect)
