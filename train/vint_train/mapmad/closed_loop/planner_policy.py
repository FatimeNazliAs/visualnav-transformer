"""Arm E, the map planner (Phase 5 block item 9): A* on the robot's own whole-storey online obstacle grid toward the
target box (planner.py), replanned every control step, steered at the path point 0.5 m ahead.

Inputs per step (from the simulator server, `outputs.grid`): the whole-storey obstacle / explored grids as known
at this step (the same online map MapMaD's tensor is cut from, never the navmesh), and the robot's pose in that
map (the true pose the map is built with -- perfect localisation, as for the map itself); at reset the target's
footprint rectangle (`goal_rect`, the box the heat is drawn from). Never the geodesic distance.

Steering: the look-ahead point in the robot frame (forward, left) goes through NoMaD's pd_controller with the same
limits (v 0-0.2 m/s, |w| <= 0.4 rad/s, no reversing). A point behind the robot (forward <= 0) -- which
pd_controller cannot follow -- gets a turn on the spot toward it. Already next to the box (path of one cell): aim at
the box's nearest point. No plan: turn on the spot (left). Deterministic: no random numbers.
"""

import math
from typing import Any, Dict, Optional

import numpy as np

from vint_train.mapmad.closed_loop import planner as P
from vint_train.mapmad.closed_loop.deployment import pd_controller
from vint_train.mapmad.closed_loop.policy import Command, Observation, Policy
from vint_train.mapmad.local_map import world_to_local


class PlannerPolicy(Policy):
    name = "planner"
    context_frames = 1

    def __init__(self, inflation_m: float, lookahead_m: float, start_clear_m: float, max_v: float = 0.2,
                 max_w: float = 0.4, rate_hz: float = 4.0) -> None:
        self.inflation_m, self.lookahead_m, self.start_clear_m = inflation_m, lookahead_m, start_clear_m
        self.max_v, self.max_w, self.dt = max_v, max_w, 1.0 / rate_hz
        self.goal_rect: Optional[P.Rect] = None

    def reset(self, meta: Dict[str, Any]) -> None:
        self.goal_rect = tuple(meta["goal_rect"])

    def plan(self, grid: Dict[str, Any]):
        """(path cells, inflated: bool) or (None, False)."""
        origin, res = tuple(grid["origin"]), float(grid["resolution"])
        x, y, _ = grid["pose"]
        start = P.cell_of((x, y), origin, res)
        goal_radius = self.inflation_m + 1.5 * res
        for inflation in (self.inflation_m, 0.0):
            blocked = P.blocked_cells(np.asarray(grid["obstacle"]), res, inflation, start, self.start_clear_m)
            path = P.astar(blocked, start, self.goal_rect, origin, res, goal_radius)
            if path is not None and (len(path) > 1 or self._at_goal(path[-1], origin, res, goal_radius)):
                return path, inflation > 0
        return None, False

    def _at_goal(self, cell, origin, res, goal_radius) -> bool:
        x, y = P.centre_of(cell, origin, res)
        return P.rect_distance(x, y, self.goal_rect) <= goal_radius

    def act(self, obs: Observation) -> Command:
        grid = obs.extras["grid"]
        pose = tuple(grid["pose"])
        origin, res = tuple(grid["origin"]), float(grid["resolution"])
        path, inflated = self.plan(grid)
        if path is None:
            return Command(0.0, self.max_w, {"plan": "none"})
        if len(path) == 1:  # next to the box already: head for its nearest point
            xmin, xmax, ymin, ymax = self.goal_rect
            target = np.array([min(max(pose[0], xmin), xmax), min(max(pose[1], ymin), ymax)])
            mode = "at_goal"
        else:
            pts = np.array([P.centre_of(c, origin, res) for c in path])
            pts[0] = pose[:2]  # the path starts at the robot itself, not its cell centre
            target = P.lookahead_point(pts, self.lookahead_m)
            mode = "path" if inflated else "path_uninflated"
        fwd, left = world_to_local(target[0], target[1], pose)
        fwd, left = float(fwd), float(left)
        if fwd <= 0.0:
            v, w = 0.0, math.copysign(self.max_w, left if left != 0.0 else 1.0)
        else:
            v, w = pd_controller(np.array([fwd, left]), self.max_v, self.max_w, self.dt)
        return Command(v, w, {"plan": mode, "path_cells": len(path), "aim_m": [round(fwd, 4), round(left, 4)]})
