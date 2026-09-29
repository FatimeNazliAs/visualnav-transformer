"""The real robot's PD controller, with ROS taken out and nothing else changed.

Ported from `deployment/src/pd_controller.py` (plan decision E: mirror the real
LoCoBot exactly, tune nothing). `pd_controller` and `clip_angle` below are the
deployment functions line for line; the limits come from the robot's own
`deployment/config/robot.yaml` rather than being re-typed here, so the sim
cannot silently drift away from the hardware.

What ROS contributed and the sim does not need:

  * `ROSData` + `WAYPOINT_TIMEOUT` — a staleness check on waypoints arriving
    over a topic. In the bridge the waypoint is computed inside the same tick
    that consumes it, so it is never stale.
  * `RATE = 9` — the deployment node republished the last waypoint's velocity
    at 9 Hz to keep `/cmd_vel` alive between navigate.py's 4 Hz waypoints.
    A synchronous loop has no gap to fill: one tick produces one command.
  * `reverse_mode` — flipped by the joystick node on the real robot. There is
    no joystick, and it is False for every autonomous run.

None of those touch the mapping from waypoint to (v, w), which is the part
being mirrored.
"""

from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOT_CONFIG_PATH = REPO_ROOT / "deployment" / "config" / "robot.yaml"

EPS = 1e-8


def load_robot_config(config_path=ROBOT_CONFIG_PATH):
    """Read the deployment robot config — the real LoCoBot's own limits."""
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


class RobotLimits:
    """max_v, max_w and the control period, exactly as the real robot has them."""

    def __init__(self, max_v, max_w, frame_rate):
        self.max_v = float(max_v)
        self.max_w = float(max_w)
        self.frame_rate = float(frame_rate)

    @classmethod
    def from_config(cls, config_path=ROBOT_CONFIG_PATH):
        config = load_robot_config(config_path)
        return cls(config["max_v"], config["max_w"], config["frame_rate"])

    @property
    def dt(self):
        """Control period in seconds — `DT` in pd_controller.py."""
        return 1.0 / self.frame_rate

    def __repr__(self):
        return "RobotLimits(max_v={}, max_w={}, frame_rate={})".format(
            self.max_v, self.max_w, self.frame_rate)


def clip_angle(theta) -> float:
    """Clip angle to [-pi, pi]"""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi


def pd_controller(waypoint: np.ndarray, limits: RobotLimits):
    """PD controller for the robot"""
    dt = limits.dt
    assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    # this controller only uses the predicted heading if dx and dy near zero
    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0
        w = clip_angle(np.arctan2(hy, hx)) / dt
    elif np.abs(dx) < EPS:
        v = 0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / dx) / dt
    v = np.clip(v, 0, limits.max_v)
    w = np.clip(w, -limits.max_w, limits.max_w)
    return v, w
