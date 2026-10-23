"""Arm F, map + frontier (Phase 12 block item 3): the explorer teacher decides on the simulator side
(mapmad_sim.search_exam.FrontierAgent: teacher_step on the robot's own online map, every control step); this policy
only steers, exactly as the v1 planner arm (planner_policy.PlannerPolicy):
- path (search / approach): the decision's next point (the path point lookahead_m = 0.5 m ahead) in the robot frame
  -> NoMaD's pd_controller (v 0-0.2 m/s, |w| <= 0.4 rad/s, no reversing); a point behind the robot (forward <= 0) ->
  a turn on the spot toward it;
- arrived: turn on the spot toward face_yaw until within face_tolerance_deg, then stand (the agent then marks the mark
  as visited and searches on);
- teacher failure (no_frontier, blacklist_exhausted, no_path): turn on the spot (left), as the planner arm without a
  plan -- it may uncover new frontier.
Deterministic: no random numbers. Inputs: obs.extras["teacher"] only (decision + the true pose, as the planner arm).
"""

import math
from typing import Any, Dict

import numpy as np

from vint_train.mapmad.closed_loop.deployment import pd_controller
from vint_train.mapmad.closed_loop.policy import Command, Observation, Policy
from vint_train.mapmad.local_map import world_to_local

FAILS = ("no_frontier", "blacklist_exhausted", "no_path")


class FrontierPolicy(Policy):
    name = "frontier"
    context_frames = 1

    def __init__(self, face_tolerance_deg: float = 15.0, max_v: float = 0.2, max_w: float = 0.4,
                 rate_hz: float = 4.0) -> None:
        self.face_tol = math.radians(face_tolerance_deg)
        self.max_v, self.max_w, self.dt = max_v, max_w, 1.0 / rate_hz

    def reset(self, meta: Dict[str, Any]) -> None:
        pass

    def act(self, obs: Observation) -> Command:
        d = obs.extras["teacher"]
        pose, mode = tuple(d["pose"]), d["mode"]
        info = {"mode": mode, "next_point": d.get("next_point"), "mark": d.get("mark")}
        if mode in FAILS or d.get("next_point") is None and mode != "arrived":
            return Command(0.0, self.max_w, info)
        if mode == "arrived":
            err = math.atan2(math.sin(d["face_yaw"] - pose[2]), math.cos(d["face_yaw"] - pose[2])) \
                if d.get("face_yaw") is not None else 0.0
            if abs(err) <= self.face_tol:
                return Command(0.0, 0.0, info)
            return Command(0.0, math.copysign(min(self.max_w, abs(err) / self.dt), err), info)
        fwd, left = world_to_local(d["next_point"][0], d["next_point"][1], pose)
        fwd, left = float(fwd), float(left)
        if fwd <= 0.0:
            v, w = 0.0, math.copysign(self.max_w, left if left != 0.0 else 1.0)
        else:
            v, w = pd_controller(np.array([fwd, left]), self.max_v, self.max_w, self.dt)
        info["aim_m"] = [round(fwd, 4), round(left, 4)]
        return Command(v, w, info)
