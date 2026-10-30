"""Phase 13 arm VF (Habitat side, no simulator): VLFM's cone confidence and fusion against hand-computed numbers, the
visible part of the cone, the frontier value, VLFM's frontier choice (hysteresis, acyclic enforcer), VF on a toy map
(best-value frontier; exactly F once a mark exists) and the VF arm's wiring.

    docker exec -w /app/visualnav-transformer naz_mapmad_habitat python -m pytest mapmad/tests/test_p13_vf.py -q
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from mapmad_sim import config
from mapmad_sim.arm_table import ArmTable
from mapmad_sim.run_layout import read_config
from mapmad_sim.search_exam import FrontierAgent, FrontierSpec
from mapmad_sim.search_vf import (CameraView, FrontierChooser, ValueMap, ValueSpec, VFAgent, VFSpec, confidence_cone,
                                  frame_cone)

from test_p12 import GOLDEN, REPO, load_train_config

SPEC = ValueSpec()


# --- cone confidence (VLFM value_map.py:337-355) --------------------------------------------------------------------
def test_confidence_cone_hand_numbers():
    cone = confidence_cone(math.radians(90.0), 1.0, 20, 0.25)  # 41 x 41, camera at (20, 20), looking along +row
    c = 20
    assert cone.shape == (41, 41)
    assert cone[c, c] == pytest.approx(1.0)  # on the axis: cos^2(0) = 1
    assert cone[c + 10, c] == pytest.approx(1.0)
    # 26.565 deg off the axis -> remapped to 53.13 deg (x 90/45) -> cos^2 = 0.36 -> 0.25 + 0.75 x 0.36 = 0.52
    assert cone[c + 10, c + 5] == pytest.approx(0.52, abs=1e-5)
    assert cone[c + 10, c - 5] == pytest.approx(0.52, abs=1e-5)
    assert cone[c - 10, c] == 0.0  # behind the camera
    assert cone[c + 19, c + 19] == 0.0  # beyond max depth (radius 20 px)


def straight_view(depth_m: float, hfov_deg: float = 90.0, max_depth: float = 4.0) -> CameraView:
    """A level camera at the 2D origin looking along +X (Habitat -z, identity rotation), uniform planar depth."""
    tf = np.eye(4)
    tf[1, 3] = 0.25
    return CameraView(np.full((60, 80), depth_m), tf, hfov_deg, max_depth)


def test_visible_cone_stops_at_the_depth_frame():
    shape, origin, res = (80, 80), (-4.0, -4.0), 0.1
    cells, conf = frame_cone(straight_view(1.0), shape, origin, res, SPEC, {})
    got = dict(zip(cells.tolist(), conf.tolist()))

    def cell(x, y):
        return int(math.floor((x - origin[0]) / res)) * shape[1] + int(math.floor((y - origin[1]) / res))

    # in front, before the wall at 1 m: 5.2 deg off the axis -> 10.4 deg -> 0.25 + 0.75 cos^2 = 0.976 (5 cm pixel)
    assert got[cell(0.55, 0.05)] == pytest.approx(0.976, abs=0.01)
    assert cell(1.55, 0.05) not in got  # behind the wall (not shown by the depth frame)
    assert cell(-0.55, 0.05) not in got  # behind the camera
    assert cell(0.55, 0.95) not in got  # outside the 90 deg FOV (60 deg off the axis)
    far = dict(zip(*frame_cone(straight_view(9.0), shape, origin, res, SPEC, {})))  # nothing within 4 m: clipped
    assert cell(3.55, 0.05) in far and cell(4.45, 0.05) not in far


# --- fusion (VLFM value_map.py:357-429, use_max_confidence False) ---------------------------------------------------
def test_fusion_hand_numbers():
    vm = ValueMap((1, 4), SPEC)
    cells = np.arange(4)
    vm.fuse(cells, np.array([0.5, 0.5, 0.2, 0.0]), 0.2)  # first view: value = cosine, confidence = new
    assert vm.value.tolist() == pytest.approx([0.2, 0.2, 0.2, 0.0])
    assert vm.conf.tolist() == pytest.approx([0.5, 0.5, 0.2, 0.0])
    vm.fuse(cells, np.array([0.8, 0.3, 0.3, 0.0]), 0.3)
    # cell 0: value (0.2 x 0.5 + 0.3 x 0.8) / 1.3, confidence (0.25 + 0.64) / 1.3
    assert vm.value[0] == pytest.approx(0.34 / 1.3) and vm.conf[0] == pytest.approx(0.89 / 1.3)
    # cell 1: new 0.3 < 0.35 and < old 0.5 -> ignored
    assert vm.value[1] == pytest.approx(0.2) and vm.conf[1] == pytest.approx(0.5)
    # cell 2: new 0.3 < 0.35 but > old 0.2 -> fused: (0.2 x 0.2 + 0.3 x 0.3) / 0.5, (0.04 + 0.09) / 0.5
    assert vm.value[2] == pytest.approx(0.13 / 0.5) and vm.conf[2] == pytest.approx(0.13 / 0.5)
    assert vm.value[3] == 0.0 and vm.conf[3] == 0.0  # never seen: 0 / 0 -> 0


def test_fusion_max_confidence_variant():
    vm = ValueMap((1, 2), ValueSpec(use_max_confidence=True))
    vm.fuse(np.arange(2), np.array([0.5, 0.5]), 0.2)
    vm.fuse(np.arange(2), np.array([0.8, 0.4]), 0.3)
    assert vm.value.tolist() == pytest.approx([0.3, 0.2]) and vm.conf.tolist() == pytest.approx([0.8, 0.5])


def test_point_value_median_of_seen_cells():
    vm = ValueMap((20, 20), SPEC)
    origin, res = (0.0, 0.0), 0.1
    assert vm.point_value((1.05, 1.05), origin, res) == -1.0  # nothing seen within 0.5 m
    vm.value[10 * 20 + 10], vm.value[10 * 20 + 12], vm.value[12 * 20 + 10] = 0.1, 0.3, 0.5
    vm.value[10 * 20 + 16] = 0.9  # 0.6 m away: outside the radius
    assert vm.point_value((1.05, 1.05), origin, res) == pytest.approx(0.3)


# --- VLFM's frontier choice (itm_policy.py:76-152) ------------------------------------------------------------------
def test_chooser_best_then_hysteresis():
    ch = FrontierChooser(SPEC)
    pts = np.array([[1.0, 0.0], [5.0, 0.0], [0.0, 3.0]])
    robot = np.array([0.0, 0.0])
    p, v = ch.choose(pts, [0.2, 0.3, 0.25], robot)
    assert p.tolist() == [5.0, 0.0] and v == 0.3
    p, v = ch.choose(pts, [0.2, 0.295, 0.30], robot)  # 0.295 + 0.01 > 0.3: stick to the last frontier
    assert p.tolist() == [5.0, 0.0]
    moved = np.array([[1.0, 0.0], [5.3, 0.1], [0.0, 3.0]])  # the same frontier moved < 0.5 m: still "the last one"
    p, _ = ch.choose(moved, [0.2, 0.29, 0.30], robot)  # 0.29 + 0.01 > 0.295
    assert p.tolist() == [5.3, 0.1]
    p, v = ch.choose(moved, [0.2, 0.25, 0.30], robot)  # 0.25 + 0.01 <= 0.29: switch to the best
    assert p.tolist() == [0.0, 3.0] and v == 0.30


def test_chooser_acyclic_enforcer():
    ch = FrontierChooser(SPEC)
    pts = np.array([[1.0, 0.0], [5.0, 0.0]])
    robot = np.array([0.0, 0.0])
    assert ch.choose(pts, [0.5, 0.4], robot)[0].tolist() == [1.0, 0.0]
    ch.last_frontier, ch.last_value = None, float("-inf")  # no last frontier: the cyclic rule alone
    assert ch.choose(pts, [0.5, 0.4], robot)[0].tolist() == [5.0, 0.0]  # same state + action seen: suppressed
    ch.last_frontier = None
    # both suppressed now -> VLFM's fallback as written: the farthest point by unsorted index, looked up sorted
    assert ch.choose(pts, [0.5, 0.4], robot)[0].tolist() == [5.0, 0.0]
    ch2 = FrontierChooser(SPEC)
    ch2.history = {ch2._key(robot, np.array([1.0, 0.0]), (0.5, 0.4)), ch2._key(robot, np.array([5.0, 0.0]), (0.5, 0.4))}
    assert ch2.choose(pts, [0.4, 0.5], robot)[0].tolist() == [1.0, 0.0]  # index 1 (farthest) of the SORTED list


# --- VF on a toy map ------------------------------------------------------------------------------------------------
class FakeITM:
    def __init__(self, value: float = 0.2) -> None:
        self.value, self.calls = value, []

    def cosine(self, rgb, text):
        self.calls.append(text)
        return self.value


def corridor(mark: bool = False) -> SimpleNamespace:
    """60 x 60 cells of 10 cm: a corridor along X (rows 10-50, cols 25-35) between two walls, open (unknown) at both
    ends -> two frontier clusters (rows 10 and 50); the robot at row 25."""
    shape = (60, 60)
    obstacle = np.full(shape, -1, np.int32)
    explored = np.full(shape, -1, np.int32)
    explored[10:51, 24:37] = 0
    obstacle[10:51, 24] = obstacle[10:51, 36] = 0
    value = np.zeros(shape[0] * shape[1])
    count = np.zeros(shape[0] * shape[1], np.int32)
    if mark:
        value[40 * 60 + 30], count[40 * 60 + 30] = 0.9, 2
    fmap = SimpleNamespace(first_seen={"obstacle": obstacle.ravel(), "explored": explored.ravel()}, shape=shape,
                           origin=np.array([0.0, 0.0]))
    return SimpleNamespace(fmap=fmap, bumped=np.zeros(shape, bool),
                           spec=SimpleNamespace(min_cluster=4, map_spec=SimpleNamespace(resolution=0.1)),
                           fusions={"noisy": SimpleNamespace(value=value, count=count)})


def vf_spec() -> VFSpec:
    return VFSpec.from_wire({"teacher": {"spec": {}, "tau": 0.45, "seen": "layer", "mode": "search",
                                         "mark_radius_m": 1.0, "min_views": 2},
                             "value_map": read_config(config.CONFIG_DIR / "p13_vf.yaml")["vf"],
                             "itm": {"host": "h", "port": 1}})


POSE = (2.55, 3.05, 0.0)


def test_vf_explores_the_best_value_frontier():
    search = corridor()
    agent = VFAgent(vf_spec(), FakeITM(), "Seems like there is a couch ahead.", search.fmap.shape)
    agent.values.value[45 * 60 + 28: 45 * 60 + 33] = 0.3  # near the far end (row 50)
    agent.values.value[14 * 60 + 28: 14 * 60 + 33] = 0.1  # near the near end (row 10)
    out, info = agent.explore(search, agent.f.own_map(search), POSE)
    assert out["mode"] == "search" and out["new_choice"] and out["mark"] is None
    assert out["goal"] == pytest.approx([5.05, 3.05]) and info["value"] == pytest.approx(0.3)
    assert info["n_frontiers"] == 2 and out["next_point"][0] > POSE[0]  # heading to +X, away from F's nearest end
    f_out = FrontierAgent(vf_spec().frontier).step(search, POSE)
    assert f_out["goal"][0] < POSE[0]  # F alone takes the nearer end
    out2, _ = agent.explore(search, agent.f.own_map(search), POSE)
    assert out2["goal"] == out["goal"] and not out2["new_choice"]  # hysteresis: same frontier
    blank = VFAgent(vf_spec(), FakeITM(), "x", search.fmap.shape)  # nothing seen: both -1, stable order = cluster 1
    assert blank.explore(search, blank.f.own_map(search), POSE)[0]["goal"] == pytest.approx([1.05, 3.05])


def test_vf_with_a_mark_is_exactly_f():
    search = corridor(mark=True)
    itm = FakeITM(0.25)
    agent = VFAgent(vf_spec(), itm, "Seems like there is a couch ahead.", search.fmap.shape)
    view = straight_view(1.0, 66.5)
    f = FrontierAgent(vf_spec().frontier)
    for pose in (POSE, (3.0, 3.05, 0.1), (3.95, 3.05, 0.0)):
        out = agent.step(search, pose, np.zeros((60, 80, 3), np.uint8), view)
        vf = out.pop("vf")
        assert vf["mark_active"] and vf["cosine"] == 0.25
        assert json.dumps(out, sort_keys=True) == json.dumps(f.step(search, pose), sort_keys=True)
    assert itm.calls == ["Seems like there is a couch ahead."] * 3  # one ITM call per control step


# --- the VF arm's wiring --------------------------------------------------------------------------------------------
def vf_table(tmp_path):
    cfg = read_config(config.CONFIG_DIR / "p12_exam.yaml")
    cfg["arms"]["VF"] = {"policy": "vf", "target": "recogniser", "recogniser": "yoloe", "label": "VLFM-style"}
    t = ArmTable.from_config(cfg, tmp_path / cfg["out_dir"], tmp_path, REPO)
    t.base.mkdir(parents=True)
    (t.base / "P12_DEV_CHOICES.json").write_text(json.dumps({"headline": GOLDEN["headline"], "tau": GOLDEN["tau"]}))
    return t


def test_vf_arm_wire_golden(tmp_path):
    from mapmad_sim.search_wire import arm_wire

    t = vf_table(tmp_path)
    assert t.arm("VF").deterministic and t.seeds("VF", [0, 1, 2]) == [0]
    recog = tuple(GOLDEN["arms"]["F"]["recog"])
    w = arm_wire(t, "VF", load_train_config, recog, itm=("naz_mapmad_recog", 5941))
    f = arm_wire(t, "F", load_train_config, recog)
    sv = json.loads(json.dumps(w.search_v2))
    golden_f = GOLDEN["arms"]["F"]["search_v2"]
    assert {k: v for k, v in sv.items() if k != "vf"} == {k: v for k, v in golden_f.items() if k != "teacher"}
    assert sv["vf"] == {"itm": {"host": "naz_mapmad_recog", "port": 5941}, "teacher": golden_f["teacher"],
                        "value_map": {"prompt": "Seems like there is a {target} ahead.", "min_confidence": 0.25,
                                      "decision_threshold": 0.35, "use_max_confidence": False, "value_radius_m": 0.5,
                                      "stick_margin": 0.01, "stick_radius_m": 0.5, "local_px_per_m": 20}}
    assert sv["vf"]["teacher"]["tau"] == GOLDEN["tau"] == 0.45
    assert w.camera == f.camera and w.extra is None
    spec = VFSpec.from_wire(sv["vf"])
    assert spec.frontier == FrontierSpec.from_wire(golden_f["teacher"]) and spec.itm == ("naz_mapmad_recog", 5941)
    with pytest.raises(SystemExit):
        arm_wire(t, "VF", load_train_config, recog)  # no ITM server


def test_vf_arm_rules(tmp_path):
    from mapmad_sim.arm_table import Arm

    with pytest.raises(ValueError):
        Arm("x", policy="vf", checkpoint="v2", target="recogniser", recogniser="yoloe")
    with pytest.raises(ValueError):
        Arm("x", policy="vf", target="truth_noisy")
    assert Arm("x", policy="vf", target="recogniser", recogniser="yoloe").deterministic
