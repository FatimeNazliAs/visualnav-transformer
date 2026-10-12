"""Phase 5 policies: MapMaD context frames, de-normalisation and masks; the planner on toy grids."""

import math

import numpy as np
import torch

from vint_train.mapmad.closed_loop import planner as P
from vint_train.mapmad.closed_loop.mapmad_policy import MapMaDPolicy, context_indices
from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy
from vint_train.mapmad.closed_loop.planner_policy import PlannerPolicy
from vint_train.mapmad.closed_loop.policy import Observation


# --- context frames ------------------------------------------------------------------------------------------------
def test_context_frames_are_t_minus_15_10_5_0():
    assert context_indices(16, 3, 5) == [0, 5, 10, 15]  # buffer full: steps t-15, t-10, t-5, t
    assert context_indices(40, 3, 5) == [24, 29, 34, 39]


def test_missing_frames_repeat_the_first():
    assert context_indices(1, 3, 5) == [0, 0, 0, 0]  # step 0: only the start picture
    assert context_indices(7, 3, 5) == [0, 0, 1, 6]  # step 6: t-15, t-10 do not exist yet


# --- MapMaD inputs (model-free: the policy object is built without loading weights) ----------------------------------
def bare_policy(goal: str) -> MapMaDPolicy:
    p = MapMaDPolicy.__new__(MapMaDPolicy)
    p.goal, p.spacing, p.step_m, p.map_size = goal, 5, 0.25, 64
    p.cfg, p.device, p.image_size = {"context_size": 3}, torch.device("cpu"), [96, 96]
    p.max_v, p.rate_hz = 0.2, 4.0
    return p


def frames(n: int):
    return [np.full((240, 320, 3), k, np.uint8) for k in range(n)]


def obs_for(n: int) -> Observation:
    heat = np.zeros((3, 64, 64), np.float32)
    heat[2, 10, 20] = 1.0
    return Observation(frames=frames(n), goal_image=np.full((240, 320, 3), 200, np.uint8), step=n - 1,
                       extras={"map": heat})


def test_mask_switches():
    want = {"map": (1, 0, True), "masked": (1, 1, False), "photo": (0, 1, False)}  # goal mask, map mask, map given
    for goal, (gm, mm, shown) in want.items():
        inp = bare_policy(goal).encoder_inputs(obs_for(20))
        assert int(inp["input_goal_mask"]) == gm and int(inp["input_map_mask"]) == mm
        assert bool(inp["map_img"].abs().sum() > 0) == shown
        assert inp["obs_img"].shape == (1, 12, 96, 96) and inp["map_img"].shape == (1, 3, 64, 64)


def test_context_pictures_follow_the_indices():
    inp = bare_policy("map").encoder_inputs(obs_for(7))
    # picture k is filled with value k; frames 0, 0, 1, 6 -> their means after ImageNet normalisation differ
    means = inp["obs_img"][0].reshape(4, 3, -1).mean(dim=(1, 2))
    assert torch.allclose(means[0], means[1]) and means[1] < means[2] < means[3]


def test_denormalisation_scale():
    assert bare_policy("map").metres_per_unit == 0.25  # MapMaD: 0.05 m x spacing 5
    official = NomadPolicy.__new__(NomadPolicy)
    official.max_v, official.rate_hz = 0.2, 4.0
    assert math.isclose(official.metres_per_unit, 0.05)  # Phase 1: MAX_V / RATE


def test_mapmad_never_waits_official_waits_for_four():
    assert bare_policy("map").ready(obs_for(1))
    official = NomadPolicy.__new__(NomadPolicy)
    official.context_frames = 4
    assert not official.ready(obs_for(3)) and official.ready(obs_for(4))


# --- planner -------------------------------------------------------------------------------------------------------
def test_dilate_disk():
    m = np.zeros((7, 7), bool)
    m[3, 3] = True
    d = P.dilate(m, 1.5)
    assert d.sum() == 9 and d[2, 2] and not d[1, 3]


def test_astar_goes_around_a_wall():
    res, origin = 0.1, (0.0, 0.0)
    obstacle = np.zeros((40, 40), np.uint8)
    obstacle[20, 0:30] = 1  # a wall at X = 2.0..2.1 m from Y = 0 to 3 m
    blocked = P.blocked_cells(obstacle, res, 0.195, (5, 5), 0.2)
    goal = (3.0, 3.2, 0.4, 0.6)  # box behind the wall
    path = P.astar(blocked, (5, 5), goal, origin, res, 0.345)
    end = P.centre_of(path[-1], origin, res)
    assert P.rect_distance(*end, goal) <= 0.345
    assert max(c[1] for c in path) >= 30  # had to pass the wall's open end (Y >= 3 m)
    assert not any(blocked[c] for c in path)


def test_unreachable_box_goes_to_nearest_reached_cell():
    res, origin = 0.1, (0.0, 0.0)
    obstacle = np.zeros((30, 30), np.uint8)
    obstacle[15, :] = 1  # a wall across the whole grid
    blocked = P.blocked_cells(obstacle, res, 0.0, (5, 5), 0.0)
    path = P.astar(blocked, (5, 5), (2.0, 2.2, 1.0, 1.2), origin, res, 0.15)
    assert path[-1][0] == 14  # the free row right in front of the wall


def test_planner_policy_steers_and_turns():
    pol = PlannerPolicy(inflation_m=0.195, lookahead_m=0.5, start_clear_m=0.2)
    pol.reset({"goal_rect": [3.0, 3.2, 1.0, 1.2]})
    grid = {"obstacle": np.zeros((50, 30), np.uint8), "origin": [0.0, 0.0], "resolution": 0.1, "pose": [0.55, 1.15, 0.0]}
    cmd = pol.act(Observation(frames=[], goal_image=None, step=0, extras={"grid": grid}))
    assert cmd.v > 0 and abs(cmd.w) < 1e-6  # straight ahead along +X
    grid["pose"] = [0.55, 1.15, math.pi]  # facing away: turn on the spot
    cmd = pol.act(Observation(frames=[], goal_image=None, step=0, extras={"grid": grid}))
    assert cmd.v == 0.0 and abs(cmd.w) == 0.4
