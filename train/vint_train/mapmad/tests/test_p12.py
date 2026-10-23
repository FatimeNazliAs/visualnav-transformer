"""Phase 12 (NoMaD side): house-seen counter on a hand-made map, scoring from logs (oracle + rule stop, wrong object,
near-miss, live rule stop = log rule stop), the headline and tau rules, arm F's steering, suite comparison."""

import json
import math

import numpy as np
import pytest

from vint_train.mapmad.closed_loop.frontier_policy import FrontierPolicy
from vint_train.mapmad.closed_loop.house_seen import HouseSeen, flood
from vint_train.mapmad.closed_loop.p12_score import (box_distance_2d, headline_rule, redundancy, score_log,
                                                    tau_rule)
from vint_train.mapmad.closed_loop.policy import Observation


# --- house seen ---------------------------------------------------------------------------------------------------
def test_house_seen_line_of_sight_and_fov():
    grid = np.ones((80, 80), bool)  # 4 m x 4 m at 0.05 m per pixel, all walkable
    grid[:, 40:42] = False  # a wall along x = 2.0 m (cols 40-41), full height of the map
    hs = HouseSeen(grid, (0.0, 0.0), 0.05, [1.0, 0.0, 2.0], hfov_deg=66.5, max_m=3.0, cell_m=0.1)
    assert hs.house_cells == 20 * 40  # the left half only (the wall cuts the house), 0.1 m cells
    n = hs.update([1.0, 0.0, 2.0], yaw=math.pi / 2)  # yaw +90 deg looks along -x (towards x = 0)
    assert 0 < n < hs.house_cells
    seen_x = np.nonzero(hs.seen.any(axis=0))[0]
    assert seen_x.max() <= 10  # nothing seen behind the robot (x > 1.0 m)
    m = hs.update([1.0, 0.0, 2.0], yaw=-math.pi / 2)  # facing +x: up to the wall, never beyond it
    assert m > n and not hs.seen[:, 20:].any()


def test_flood_8_connected():
    free = np.zeros((5, 5), bool)
    free[0, 0] = free[1, 1] = free[4, 4] = True
    assert flood(free, (0, 0)).sum() == 2


# --- scoring ------------------------------------------------------------------------------------------------------
INST = [{"centre": [-3.0, 0.5, -2.0], "size": [0.6, 1.0, 0.6], "view_points": []}]  # 2D frame: X = 2.0, Y = 3.0


def make_log(vp, rule, cells=None, in_view=None, collided=None):
    head = {"kind": "episode", "episode_id": "e1", "arm": "M2", "seed_index": 0, "home": "00800", "category": "chair",
            "distance_bin": 1, "start": {"position": [0.0, 0.0, 0.0], "yaw": 0.0, "in_view": False},
            "start_vp_geodesic_m": 4.0, "instances": INST, "stuck_steps": 3}
    steps = []
    for i, v in enumerate(vp):
        steps.append({"kind": "step", "step": i + 1, "vp_geodesic_m": v, "travelled_m": 0.5, "position": [0.5 * (i + 1), 0.0, 0.0],
                      "collided": bool(collided[i]) if collided else False, "in_view": bool(in_view[i]) if in_view else False,
                      "rule_value": rule[i], "rule_cell": (cells[i] if cells else [2.0, 3.0])})
    return [head] + steps + [{"kind": "result"}]


def test_score_oracle_and_rule_stop():
    vp = [3.5, 3.0, 2.5, 0.15, 0.1, 0.9]
    rule = [0.0, 0.2, 0.55, 0.62, 0.8, 0.9]
    r = score_log(make_log(vp, rule, in_view=[0, 1, 1, 1, 1, 1]), success_m=0.2)
    assert r["oracle_success"] and r["oracle_step"] == 4 and r["first_sight"] == 2
    assert r["oracle_spl"] == pytest.approx(4.0 / max(4.0, 2.0)) and r["oracle_approach_steps"] == 2
    s50, s70, s95 = r["rule"]["0.50"], r["rule"]["0.70"], r["rule"]["0.95"]
    assert s50["step"] == 3 and not s50["success"] and s50["near_miss"] is False  # 2.5 m: not a near-miss
    assert s70["step"] == 5 and s70["success"] and s70["spl"] == pytest.approx(1.0)
    assert not s95["stopped"] and not s95["success"]


def test_wrong_object_and_near_miss():
    vp = [3.0, 1.1, 1.0]
    rule = [0.1, 0.7, 0.9]
    cells = [None, [9.0, 9.0], [2.0, 3.0]]  # far from the instance box / on it
    r = score_log(make_log(vp, rule, cells=cells), success_m=0.2)
    assert r["rule"]["0.60"]["wrong_object"] and r["rule"]["0.60"]["near_miss"]  # 1.1 m <= 0.2 + 1.0
    assert not r["rule"]["0.85"]["wrong_object"] and r["rule"]["0.85"]["near_miss"]
    assert box_distance_2d([2.0, 3.0], INST[0]) == 0.0


def test_live_rule_stop_equals_log_rule_stop():
    """A run that stops live at the first rule value >= tau has the same trajectory up to the stop (the stop changes
    nothing before it), so its stop step and success equal the log scoring of the full run."""
    vp = [3.0, 2.0, 0.1, 0.1, 2.0]
    rule = [0.1, 0.3, 0.65, 0.7, 0.2]
    full = score_log(make_log(vp, rule), success_m=0.2)
    for tau in (0.3, 0.6, 0.7):
        k = next(i for i, v in enumerate(rule) if v >= tau)
        live = score_log(make_log(vp[:k + 1], rule[:k + 1]), taus=(tau,), success_m=0.2)["rule"][f"{tau:.2f}"]
        assert live["step"] == full["rule"][f"{tau:.2f}"]["step"] and live["success"] == full["rule"][f"{tau:.2f}"]["success"]


def test_redundancy_and_stuck():
    assert redundancy([[0, 0, 0], [0.6, 0, 0], [0.1, 0, 0], [0.6, 0, 0]], 0.5) == pytest.approx(0.5)
    r = score_log(make_log([3, 3, 3, 3], [0, 0, 0, 0], collided=[1, 1, 1, 0]), success_m=0.2)
    assert r["stuck_streaks"] == 1 and r["collisions"] == 3


def test_headline_and_tau_rules():
    assert headline_rule({"v2": 0.05, "v21": 0.04}) == "v21"  # within the 0.02 margin
    assert headline_rule({"v2": 0.10, "v21": 0.07}) == "v2"
    assert headline_rule({"v2": -0.01, "v21": -0.03}) == "v21"  # both <= 0: the rule anyway (exactly -0.02 apart)
    assert tau_rule({0.3: 0.2, 0.5: 0.4, 0.6: 0.4, 0.7: 0.1}) == 0.6  # tie -> higher tau


# --- arm F steering -----------------------------------------------------------------------------------------------
def obs(d):
    return Observation(frames=[], goal_image=None, step=0, extras={"teacher": d})


def test_frontier_policy_steering():
    p = FrontierPolicy()
    c = p.act(obs({"mode": "search", "pose": [0, 0, 0], "next_point": [0.5, 0.0]}))
    assert c.v == pytest.approx(0.2) and abs(c.w) < 1e-9
    c = p.act(obs({"mode": "approach", "pose": [0, 0, 0], "next_point": [-0.5, 0.2]}))
    assert c.v == 0.0 and c.w == pytest.approx(0.4)  # behind and to the left: turn left on the spot
    c = p.act(obs({"mode": "arrived", "pose": [0, 0, 0], "next_point": [0, 0], "face_yaw": 0.1}))
    assert (c.v, c.w) == (0.0, 0.0)  # within 15 deg: stand
    c = p.act(obs({"mode": "arrived", "pose": [0, 0, 0], "next_point": [0, 0], "face_yaw": -1.0}))
    assert c.v == 0.0 and c.w == pytest.approx(-0.4)
    c = p.act(obs({"mode": "no_frontier", "pose": [0, 0, 0], "next_point": None}))
    assert (c.v, c.w) == (0.0, 0.4)


# --- suite comparison ---------------------------------------------------------------------------------------------
def test_suite_compare(tmp_path):
    from vint_train.mapmad.offline_suite import compare_dirs

    for d in ("a", "b"):
        (tmp_path / d / "search").mkdir(parents=True)
        np.savez(tmp_path / d / "search" / "scores.npz", x=np.array([1.0, np.nan]), ids=np.array(["s1", "s2"]))
        (tmp_path / d / "g10").mkdir()
        (tmp_path / d / "g10" / "g10_rows_1_2_cpu.json").write_text(json.dumps(
            {"row1": {"pass": True, "outputs": {"c": {"equal": True}}}, "row2": {"pass": True}}))
    res = compare_dirs(tmp_path / "a", tmp_path / "b")
    assert all(r["pass"] for r in res.values()) and len(res) == 2
    np.savez(tmp_path / "b" / "search" / "scores.npz", x=np.array([1.0, 2.0]), ids=np.array(["s1", "s2"]))
    assert not compare_dirs(tmp_path / "a", tmp_path / "b")["search/scores.npz"]["pass"]
