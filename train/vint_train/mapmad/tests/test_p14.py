"""Phase 14a NoMaD-side tests: the driving switches (A1 picker, W4 rule), their wiring into the policy and the runner
(switch off = Phase 13 behaviour), the stop-bench judge and pick rules."""

import math
from pathlib import Path

import numpy as np
import pytest

from mapmad_bridge.client import Frame
from vint_train.mapmad.closed_loop.drive_switches import (StickPicker, Unstuck, UnstuckSpec, coherence_pick,
                                                           open_direction, shift_plan)


# --- A1 -----------------------------------------------------------------------------------------------------------
def test_shift_plan_straight_and_turn():
    plan = np.array([[1.0, 0.0], [2.0, 0.0]])
    out = shift_plan(plan, 0.2, 0.0, 0.25)  # 0.05 m forward
    assert np.allclose(out, [[0.95, 0.0], [1.95, 0.0]])
    turned = shift_plan(np.array([[1.0, 0.0]]), 0.0, 0.4, 0.25)  # turn left 0.1 rad on the spot: the point moves right
    assert turned[0, 1] < 0 and math.isclose(math.hypot(*turned[0]), 1.0)


def test_coherence_pick_weights_near_waypoints_more():
    prev = np.zeros((8, 2))
    a = np.zeros((8, 2)); a[0] = [0.5, 0]  # far at waypoint 0 (weight 1)
    b = np.zeros((8, 2)); b[7] = [0.5, 0]  # far at waypoint 7 (weight 0.5^7)
    assert coherence_pick(np.stack([a, b]), prev, 0.5) == 1
    assert coherence_pick(np.stack([b, b]), prev, 0.5) == 0  # ties -> the lower index


def test_stick_picker_resets_every_10_steps_and_after_reset():
    p = StickPicker(0.5, 10, 0.25)
    s = np.zeros((8, 8, 2))
    s[3] = 0.001  # sample 3 is the closest to sample 0's plan after a zero move
    s[0] = 0.01
    assert p.pick(s, 0) == 0  # first step: sample 0
    p.executed(0.0, 0.0)
    assert p.pick(s, 1) != 0 or np.allclose(s[0], s[p.pick(s, 1)])
    assert p.pick(s, 10) == 0  # periodic reset
    p.reset()
    assert p.pick(s, 11) == 0  # after a reset (episode start / W4 manoeuvre)


# --- W4 -----------------------------------------------------------------------------------------------------------
def test_open_direction_finds_the_free_side():
    obst = np.zeros((64, 64))
    obst[:32, :] = 1  # everything ahead (rows 0..31) blocked
    obst[32:, :32] = 1  # left half behind blocked too
    ang, free = open_direction(obst)
    assert -180 <= math.degrees(ang) <= -90  # the free quadrant: behind-right
    assert free > 2.0


def test_unstuck_triggers_on_collisions_and_runs_the_manoeuvre():
    u = Unstuck(UnstuckSpec(), 0.4, 0.25)
    maps = np.zeros((2, 4, 64, 64), np.float32)
    maps[0, 0, :32, :] = 1  # wall ahead -> the turn goes behind
    for t in range(19):
        assert u.command(t, {"maps": maps}) is None
        u.observe(t, True, (0.0, 0.0))
    u.observe(19, True, (0.0, 0.0))
    cmds = []
    t = 20
    while True:
        c = u.command(t, {"maps": maps})
        if c is None:
            break
        cmds.append(c)
        u.observe(t, False, (0.0, 0.0))
        t += 1
    rev = [c for c in cmds if c[3]["unstuck"] == "reverse"]
    turn = [c for c in cmds if c[3]["unstuck"] == "turn"]
    assert len(rev) == 10 and all(c[0] == -0.1 and c[2] for c in rev)
    assert turn and all(abs(c[1]) <= 0.4 + 1e-9 and not c[2] for c in turn)
    turned = abs(sum(c[1] for c in turn) * 0.25)
    assert math.radians(90) < turned <= math.pi + 1e-9  # away from the wall ahead (ties: the smallest such turn)
    assert turned == pytest.approx(math.radians(u.events[0]["turn_deg"]) * np.sign(u.events[0]["turn_deg"]), abs=1e-6)
    assert u.events and u.events[0]["why"] == "collisions"


def test_unstuck_still_rule_and_cap_per_100_steps():
    u = Unstuck(UnstuckSpec(), 0.4, 0.25)
    extras = {"maps": np.zeros((2, 4, 64, 64), np.float32)}
    starts = 0
    for t in range(100):
        c = u.command(t, extras)
        if c is not None and c[3]["unstuck_n"] > starts:
            starts = c[3]["unstuck_n"]
        u.observe(t, False, (0.0, 0.0))  # never moves, never collides
    assert u.events[0]["why"] == "still" and len(u.starts) <= 3


# --- wiring (switch off = Phase 13) ---------------------------------------------------------------------------------
class _Net:
    """A NomadPolicy stand-in: fixed samples, no network."""


def _policy(picker=None):
    from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy

    pol = NomadPolicy.__new__(NomadPolicy)
    pol.cfg = {"normalize": True, "context_size": 3}
    pol.sample_index, pol.waypoint_index = 0, 1
    pol.max_v, pol.max_w, pol.rate_hz = 0.2, 0.4, 4.0
    pol.context_frames = 1
    pol.picker = picker
    rng = np.random.default_rng(0)
    paths = rng.normal(size=(8, 8, 2)).cumsum(axis=1)
    pol.sample_paths = lambda obs: paths
    pol.ready = lambda obs: True
    return pol, paths


def test_policy_without_picker_is_phase13_sample0():
    from vint_train.mapmad.closed_loop.policy import Observation

    pol, paths = _policy()
    cmd = pol.act(Observation(frames=[], goal_image=None, step=3))
    assert "sample" not in cmd.info
    assert cmd.info["path_m"] == (paths[0] * 0.05).round(4).tolist()


def test_policy_with_picker_logs_its_sample():
    from vint_train.mapmad.closed_loop.policy import Observation

    pol, _ = _policy(StickPicker(0.5, 10, 0.25))
    first = pol.act(Observation(frames=[], goal_image=None, step=0))
    second = pol.act(Observation(frames=[], goal_image=None, step=1))
    assert first.info["sample"] == 0 and 0 <= second.info["sample"] < 8


class _SearchSim:
    """A search server stand-in: the robot never moves and always collides (W4 must fire)."""

    def __init__(self) -> None:
        self.calls = []

    def _frame(self, move=None):
        st = {"position": [0.0, 0.0, 0.0], "yaw": 0.0, "vp_geodesic_m": 5.0, "in_view": False,
              "first_in_view_step": None, "seen_instances": []}
        return Frame(rgb=np.zeros((4, 4, 3), np.uint8), state=st, move=move,
                     extras={"maps": np.zeros((2, 4, 64, 64), np.float32)}, navmesh_sha256="x")

    def reset(self, *a, **k):
        return self._frame()

    def step(self, v, w, reverse=False):
        self.calls.append((v, w, reverse))
        return self._frame({"v": v, "w": w, "travelled_m": 0.0, "short_m": 0.05, "collided": True})


class _Fixed:
    name, context_frames = "fixed", 1

    def reset(self, meta):
        pass

    def act(self, obs):
        from vint_train.mapmad.closed_loop.policy import Command

        return Command(0.2, 0.0, {"waypoint_m": [0.05, 0.0]})


def _run(tmp_path, unstuck):
    from vint_train.mapmad.closed_loop.search_runner import SearchSettings, run_search_episode

    sim = _SearchSim()
    ep = {"episode_id": "e1", "home": "h", "distance_bin": 0, "navmesh_sha256": "x", "target": {"category": "chair"}}
    run = SearchSettings(0.2, 60, 100, 0.05, stop_at_success=False)
    res, _ = run_search_episode(sim, _Fixed(), ep, "A", run, 0, 1, tmp_path / f"{unstuck is not None}.jsonl", "fp",
                                {"noise_seed": 1}, {"hfov_deg": 66.5, "name": "pin025"}, unstuck=unstuck)
    return sim, res


def test_runner_without_unstuck_never_reverses(tmp_path):
    sim, res = _run(tmp_path, None)
    assert not any(c[2] for c in sim.calls) and "unstuck_events" not in res


def test_runner_with_w4_reverses_only_inside_the_manoeuvre(tmp_path):
    sim, res = _run(tmp_path, Unstuck(UnstuckSpec(), 0.4, 0.25))
    assert res["unstuck_events"] >= 1 and res["unstuck_steps"] >= 10
    assert all(v < 0 for v, w, r in sim.calls if r) and all(v >= 0 for v, w, r in sim.calls if not r)


# --- stop bench -------------------------------------------------------------------------------------------------------
def test_judge_and_pick_rules():
    from vint_train.mapmad.closed_loop.p14_stopbench import judge, pick_rule, r2_order

    vp = [5.0, 1.0, 0.1]
    path = np.array([0.0, 1.0, 2.0])
    inst = [{"centre": [0.0, 0.5, 0.0], "size": [0.4, 1.0, 0.4]}]
    ok = judge((2, (0.0, 0.0)), vp, path, 5.0, inst)
    assert ok["success"] and not ok["wrong_object"] and ok["spl"] == pytest.approx(1.0)
    bad = judge((1, (5.0, 5.0)), vp, path, 5.0, inst)
    assert not bad["success"] and bad["wrong_object"] and bad["near_miss"]
    hit = {"step": 3, "success": True, "spl": 1.0, "wrong_object": False, "near_miss": False}
    miss = {"step": None, "success": False, "spl": 0.0, "wrong_object": False, "near_miss": False}
    rows = {"X": [{"episode_id": "e", "rules": {"r2_v0.40_g1": hit, "r2_v0.50_g1": hit, "r2_v0.50_g2": miss},
                   "oracle_success": True}]}
    key, _ = pick_rule(rows, ["r2_v0.40_g1", "r2_v0.50_g1", "r2_v0.50_g2"], {"e": "A"}, ["X"], r2_order)
    assert key == "r2_v0.50_g1"  # tie on success -> the higher v
