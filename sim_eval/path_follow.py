"""Drive the robot along a planned path — the sim's stand-in for the human teleop.

`deployment/src/create_topomap.sh` builds a real topomap by having a person
joystick the robot down the route while frames are saved. There is nobody to
joystick a simulator, so P2 replaces the person with the scene's own shortest
path (plan §5): iGibson's A* over the traversability graph says where the
perfect route goes, and this module is the hand that follows it.

It is deliberately **not** NoMaD's controller and not a tuned one. The
reference path only has to be driven *correctly* — the model's job is to repeat
it later, and how well it does that is the measurement. So this is the smallest
follower that tracks a polyline:

    pure pursuit for *where* to aim, `heading_error / dt` for *how hard to turn*

The second half is the shape `pd_control.pd_controller` already uses — "turn
this much in one control period, clipped to the robot's own max_w" — so the
reference path is driven inside the same envelope the policy will be held to
(`max_v`, `max_w`, 4 Hz), not faster or sharper than the policy could manage.

Pure numpy: no torch, no iGibson, no GPU, so `tests/test_path_follow.py` pins
the behaviour without a simulator.
"""

import numpy as np

from pd_control import clip_angle


class FollowerParams:
    """The three numbers that decide how the reference path is tracked.

    `lookahead_m`        how far down the path to aim. Too small and the robot
                         hunts across the line; too large and it cuts corners.
    `arrival_radius_m`   how close to the last path point counts as arrived.
    `turn_in_place_rad`  above this heading error the robot stops and turns
                         rather than driving an arc. A differential drive can
                         spin on the spot, and a start pose facing away from
                         the path is normal, so the alternative is a long
                         curved detour through furniture before the route even
                         begins.
    """

    def __init__(self, lookahead_m=0.5, arrival_radius_m=0.2, turn_in_place_rad=0.5):
        self.lookahead_m = float(lookahead_m)
        self.arrival_radius_m = float(arrival_radius_m)
        self.turn_in_place_rad = float(turn_in_place_rad)

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))

    def as_dict(self):
        return {
            "lookahead_m": self.lookahead_m,
            "arrival_radius_m": self.arrival_radius_m,
            "turn_in_place_rad": self.turn_in_place_rad,
        }

    def __repr__(self):
        return "FollowerParams({})".format(
            ", ".join("{}={}".format(key, value)
                      for key, value in self.as_dict().items()))


def advance_target(path, position, index, lookahead_m):
    """Pure pursuit: where on the path to aim, given where the robot is now.

    Two forward walks, and both are needed:

      1. **Catch up to the robot.** Slide forward while the next point is
         closer than the current one, which lands on the path point the robot
         is beside. Without this the target stays behind a robot that has
         driven past it, and the follower turns around to chase it.
      2. **Push out to the lookahead.** Aim at the first point further away
         than `lookahead_m`. Aiming at the nearest point instead makes the
         robot hunt across the line rather than converge onto it.

    The index only ever moves forward. Letting it move back would let the robot
    re-target a point it has already passed whenever the path doubles back near
    itself — which in a small house it does, at every doorway.
    """
    path = np.asarray(path, dtype=float)
    position = np.asarray(position, dtype=float)[:2]
    last = len(path) - 1

    def distance(at):
        return float(np.linalg.norm(path[at] - position))

    while index < last and distance(index + 1) <= distance(index):
        index += 1
    while index < last and distance(index) < lookahead_m:
        index += 1
    return index


def follow_step(pose, target_xy, limits, params):
    """One control tick of tracking: pose + aim point -> (v, w) in SI units.

    Clipped to the robot's own limits, so the reference path is never driven
    faster or turned harder than the policy following it will be allowed to.
    """
    x, y, yaw = pose
    dx, dy = np.asarray(target_xy, dtype=float)[:2] - np.array([x, y])

    heading_error = clip_angle(np.arctan2(dy, dx) - yaw)
    w = float(np.clip(heading_error / limits.dt, -limits.max_w, limits.max_w))

    if abs(heading_error) > params.turn_in_place_rad:
        v = 0.0
    else:
        distance = float(np.hypot(dx, dy))
        v = float(np.clip(distance / limits.dt, 0.0, limits.max_v))
    return v, w


def has_arrived(pose, goal_xy, radius_m):
    """True once the robot is within `radius_m` of the path's last point."""
    offset = np.asarray(goal_xy, dtype=float)[:2] - np.asarray(pose[:2], dtype=float)
    return bool(np.linalg.norm(offset) <= radius_m)


def initial_yaw(path, lookahead_m):
    """Which way to face at the start: down the path, not at the first point.

    Aiming at `path[1]` points the robot at something 0.2 m away, and on a path
    that starts with a turn that heading is off by most of the turn. Taking the
    bearing to the first point a lookahead away instead makes node 0 of the
    topomap show where the route goes — which is the whole content of node 0.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        raise ValueError("a path needs at least two points to have a direction")
    target_index = advance_target(path, path[0], 0, lookahead_m)
    dx, dy = path[target_index] - path[0]
    return float(np.arctan2(dy, dx))
