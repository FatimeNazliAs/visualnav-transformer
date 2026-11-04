"""Phase 14b-2: training-frame labels and selection (R1), the A-mark switch (R4), arm-table and run-plan entries."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mapmad_sim import amark as am  # noqa: E402
from mapmad_sim import photo_labels as pl  # noqa: E402
from mapmad_sim.arm_table import Arm  # noqa: E402
from mapmad_sim.explorer import APPROACH, ARRIVED, OwnMap, TeacherSpec  # noqa: E402

ts = pytest.importorskip("p14b2_trainset")

LOOK = ["stool", "bench", "ottoman", "footstool", "pouf", "pouffe"]


def tables():
    mapping = {"chair": "chair", "armchair": "chair", "sofa": "sofa", "bed": "bed", "toilet": "toilet", "tv": "tv_monitor",
               "potted plant": "plant"}
    return pl.six_tables(["chair", "armchair", "sofa", "bed", "bar stool", "daybed", "toilet", "tv", "potted plant",
                          "table", "flower vase"], mapping)


def test_instance_role_labels_ignores_and_lookalikes():
    t = tables()
    assert ts.instance_role("chair", 60, t, 50, LOOK) == ("label", "chair")
    assert ts.instance_role("chair", 49, t, 50, LOOK) == ("ignore", None)  # small target: not background
    assert ts.instance_role("daybed", 500, t, 50, LOOK) == ("ignore", None)  # near-synonym of bed / sofa
    assert ts.instance_role("bar stool", 500, t, 50, LOOK) == ("background", None)  # 14a look-alike: a negative
    assert ts.instance_role("table", 500, t, 50, LOOK) == ("background", None)


def test_label_maps_paint_classes_and_ignore():
    sem = np.array([[1, 1, 2], [3, 0, 2]])
    inst, cls = ts.label_maps(sem, {1: ("label", "bed"), 2: ("ignore", None), 3: ("background", None)})
    assert inst.tolist() == [[1, 1, 0], [0, 0, 0]]
    assert cls.tolist() == [[ts.CLASS_ID["bed"]] * 2 + [255], [0, 0, 255]]


def test_select_frames_share_weights_and_home_split():
    frames = [{"frame_id": f"f{k:03d}", "home": f"h{k % 10}", "n_labelled": int(k < 60), "free": k >= 60,
               "lookalike": k >= 90, "labelled": {}} for k in range(120)]
    chosen, st = ts.select_frames(frames, 50, 0.4, 3.0, 0.2, 7)
    assert st["positive"] == 30 and st["free"] == 20 and st["chosen"] == 50
    assert st["free_lookalike"] >= 6  # 30 of 60 free frames are look-alikes, drawn 3x as often
    val = {f["home"] for f in chosen if f["split"] == "val"}
    assert val and not val & {f["home"] for f in chosen if f["split"] == "train"}
    again, _ = ts.select_frames(frames, 50, 0.4, 3.0, 0.2, 7)
    assert again == chosen


def test_select_frames_fewer_positives_keeps_the_share():
    frames = [{"frame_id": f"f{k}", "home": "h", "n_labelled": int(k < 6), "free": k >= 6, "lookalike": False,
               "labelled": {}} for k in range(100)]
    _, st = ts.select_frames(frames, 60, 0.4, 3.0, 0.1, 1)
    assert st["positive"] == 6 and st["free"] == 4


def test_overlap_by_scene_id():
    pool = ["00006-HkseAnWCgqk", "00007-UQuchpekHRJ"]
    ex = {"photo_test": ["00081-5biL7VEkByM"], "hm3d_val": ["00800-HkseAnWCgqk"], "mp3d": ["17DRP5sb8fy"]}
    assert ts.overlap(pool, ex) == {"photo_test": [], "hm3d_val": ["00800-HkseAnWCgqk"], "mp3d": []}


# --- A-mark ---------------------------------------------------------------------------------------------------------

SHAPE, ORIGIN, RES = (40, 40), (0.0, 0.0), 0.1


def flat(i, j):
    return i * SHAPE[1] + j


def test_accepted_cell_needs_value_views_view_distance():
    spec = am.AMarkSpec(TeacherSpec(), tau=0.45, min_views=2, trigger_m=3.0)
    value, count = np.zeros(1600), np.zeros(1600, np.int32)
    near, far, weak, once = flat(10, 12), flat(38, 38), flat(10, 14), flat(11, 12)
    value[[near, far, weak, once]] = [0.6, 0.9, 0.3, 0.8]
    count[[near, far, weak, once]] = [2, 5, 4, 1]
    view = np.array([near, far, weak, once])
    pose = (1.0, 1.0, 0.0)
    assert am.accepted_cell(value, count, view, SHAPE, ORIGIN, RES, pose, spec) == near
    assert am.accepted_cell(value, count, view[1:], SHAPE, ORIGIN, RES, pose, spec) == -1  # far: > 3 m
    ex = np.zeros(1600, bool)
    ex[near] = True
    assert am.accepted_cell(value, count, view, SHAPE, ORIGIN, RES, pose, spec, ex) == -1


class FakeSearch:
    def __init__(self, value, count, view):
        self.fusions = {"noisy": SimpleNamespace(value=value, count=count)}
        self.fmap = SimpleNamespace(shape=SHAPE, origin=ORIGIN)
        self.last_cells = SimpleNamespace(view=view)


def own():
    free = np.ones(SHAPE, bool)
    return OwnMap(np.zeros(SHAPE, bool), free, ORIGIN, RES, None, None)


def test_agent_drives_confirms_and_never_retriggers(monkeypatch):
    k = flat(15, 10)
    value, count = np.zeros(1600), np.zeros(1600, np.int32)
    value[k], count[k] = 0.7, 3
    search = FakeSearch(value, count, np.array([k]))
    modes = iter([APPROACH, ARRIVED])

    def fake_step(own_map, pose, state, seen, spec, searching):
        mode = next(modes)
        return SimpleNamespace(mode=mode, next_point=(1.4, 1.0), face_yaw=0.0 if mode == ARRIVED else None), state
    monkeypatch.setattr(am, "teacher_step", fake_step)
    agent = am.AMarkAgent(am.AMarkSpec(TeacherSpec(), tau=0.45))
    out = agent.step(search, own(), (1.0, 1.0, 0.0), 0.1)
    assert out["steer"] and out["event"] == "trigger" and out["mode"] == APPROACH
    out = agent.step(search, own(), (1.45, 1.05, 0.0), 0.5)  # arrived, facing, rule value >= tau
    assert not out["steer"] and out["event"] == "confirmed"
    out = agent.step(search, own(), (1.45, 1.05, 0.0), 0.5)  # the handled cluster is excluded
    assert out["event"] is None and agent.counts["confirmed"] == 1 and agent.counts["trigger"] == 1


def test_agent_rejects_unconfirmed_no_path_and_timeout(monkeypatch):
    k = flat(15, 10)
    value, count = np.zeros(1600), np.zeros(1600, np.int32)
    value[k], count[k] = 0.7, 3
    search = FakeSearch(value, count, np.array([k]))
    monkeypatch.setattr(am, "teacher_step", lambda *a: (SimpleNamespace(mode=ARRIVED, next_point=None, face_yaw=0.0), a[2]))
    agent = am.AMarkAgent(am.AMarkSpec(TeacherSpec(), tau=0.45))
    assert agent.step(search, own(), (1.45, 1.05, 0.0), 0.2)["event"] == "unconfirmed"
    monkeypatch.setattr(am, "teacher_step", lambda *a: (SimpleNamespace(mode="no_path", next_point=None, face_yaw=None), a[2]))
    agent = am.AMarkAgent(am.AMarkSpec(TeacherSpec(), tau=0.45))
    assert agent.step(search, own(), (1.0, 1.0, 0.0), 0.0)["event"] == "no_path"
    monkeypatch.setattr(am, "teacher_step", lambda *a: (SimpleNamespace(mode=APPROACH, next_point=(1.2, 1.0), face_yaw=None), a[2]))
    agent = am.AMarkAgent(am.AMarkSpec(TeacherSpec(), tau=0.45, max_steps=2))
    events = [agent.step(search, own(), (1.0, 1.0, 0.0), 0.0)["event"] for _ in range(3)]
    assert events == ["trigger", None, "timeout"]


def test_agent_no_free_cell_rejects():
    k = flat(15, 10)
    value, count = np.zeros(1600), np.zeros(1600, np.int32)
    value[k], count[k] = 0.7, 3
    blocked = OwnMap(np.ones(SHAPE, bool), np.ones(SHAPE, bool), ORIGIN, RES, None, None)
    agent = am.AMarkAgent(am.AMarkSpec(TeacherSpec(), tau=0.45))
    assert agent.step(FakeSearch(value, count, np.array([k])), blocked, (1.0, 1.0, 0.0), 0.0)["event"] == "no_free"


def test_arm_mark_drive_field():
    a = Arm("x", checkpoint="v21b", target="recogniser", recogniser="yoloe", unstuck="w4", mark_drive="amark")
    assert a.switches == {"unstuck": "w4", "mark_drive": "amark"}
    assert Arm("y", checkpoint="v21b").switches == {}
    with pytest.raises(ValueError):
        Arm("z", checkpoint="v21b", mark_drive="go")
    with pytest.raises(ValueError):
        Arm("f", policy="frontier", target="recogniser", recogniser="yoloe", mark_drive="amark")
