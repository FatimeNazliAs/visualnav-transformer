"""Phase 14a Habitat-side tests: stop evidence + stop rules (parity with TargetFusion, Rule 1, Rule 2, rejected
spots), diagnosis classes, mark counts, run plan, arm switches, the reverse clamp, the practice split, Qwen helpers."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mapmad_sim.explorer  # noqa: F401  (puts train/ on sys.path: vint_train.mapmad.map_v2)
from mapmad_sim.arm_table import Arm
from mapmad_sim.diagnosis import TeacherView, Window, classify, windows_of
from mapmad_sim.mark_check import mark_sightings
from mapmad_sim.recogniser import Detection
from mapmad_sim.stop_evidence import EvidenceRecorder, load
from mapmad_sim.stop_rules import EvidenceReplay, StopSpec, first_stops, near_weight, r1_grid, r2_grid, rule_fires
from mapmad_sim.target_layer import FrameCells

SHAPE, RES = (40, 40), 0.1


def cells_of(view, pixel_cell):
    return FrameCells(np.asarray(pixel_cell, np.int64), np.unique(np.asarray(view, np.int64)), np.zeros(0, np.int64))


def det(score, raw, px_idx, hw=(4, 4), box=(0, 0, 2, 2)):
    m = np.zeros(hw[0] * hw[1], bool)
    m[list(px_idx)] = True
    return Detection(score, m.reshape(hw), None, raw, box)


def synthetic(frames, path):
    """frames: [(pose, view cells, [(value, raw, pixel idx)], pixel_cell table)] -> saved evidence + TargetFusion."""
    from vint_train.mapmad.map_v2 import TargetFusion
    from mapmad_sim.recogniser import frame_scores

    rec = EvidenceRecorder(SHAPE, (0.0, 0.0), RES, [7])
    fus = TargetFusion(SHAPE)
    for t, (pose, view, dets, table) in enumerate(frames):
        fc = cells_of(view, table)
        ds = [det(v, r, px) for v, r, px in dets]
        rec.add(t, pose, fc, ds, np.zeros((4, 4), np.int32))
        fus.update(t, fc.view, frame_scores(ds, fc, (4, 4)))
    first = {"obstacle": np.full(SHAPE[0] * SHAPE[1], -1, np.int32), "explored": np.zeros(SHAPE[0] * SHAPE[1], np.int32)}
    rec.save(path, first, [(2, 1.0, 1.0)])
    return load(path), fus


def cell(i, j):
    return i * SHAPE[1] + j


def test_replay_matches_target_fusion_and_round_trip(tmp_path):
    k = cell(12, 10)  # 0.25 m ahead of a robot at (1.0, 1.05) facing +x
    table = [k] * 8 + [-1] * 8
    frames = [((1.0, 1.05, 0.0), [k, cell(5, 5)], [], table),
              ((1.0, 1.05, 0.0), [k], [(0.8, 0.5, range(8))], table),
              ((1.0, 1.05, 0.0), [k], [(0.6, 0.2, range(8)), (0.9, 0.3, range(4))], table),
              ((1.0, 1.05, 0.0), [k], [], table)]
    ev, fus = synthetic(frames, tmp_path / "e.npz")
    assert ev["bumps"].tolist() == [[2.0, 1.0, 1.0]] and len(ev["pose"]) == 4
    rep = EvidenceReplay(ev)
    for t in range(4):
        rep.advance()
    assert rep.value(np.array([k]))[0] == fus.value[k]  # (0 + 0.8 + 0.9 + 0) / 4, float32 like TargetFusion
    assert rep.today_value() == pytest.approx(float(fus.value[k]))


def test_rule2_needs_close_confident_looks_and_misses_lower_it(tmp_path):
    k = cell(12, 10)
    table = [k] * 16
    big = range(16)
    frames = [((1.0, 1.05, 0.0), [k], [], table)] + [((1.0, 1.05, 0.0), [k], [(0.9, 0.5, big)], table)] * 3
    ev, _ = synthetic(frames, tmp_path / "e.npz")
    ev["det_px"][:] = 2000  # size factor 1
    stops = first_stops(ev, StopSpec())
    # w = 0.9 x 1 x near(0.25 m) = 0.9 per hit; one miss of weight 1: value = 3 x 0.81 / (2.7 + 1) = 0.657
    assert stops["r2_v0.65_g3"] is not None and stops["r2_v0.65_g3"][0] == 3
    assert stops["r2_v0.70_g1"] is None
    assert near_weight(np.array([1.0, 2.75, 4.0, 9.0]), StopSpec()).tolist() == pytest.approx([1.0, 0.625, 0.25, 0.25])


def test_rule1_votes_on_raw_scores_ahead_only(tmp_path):
    k = cell(12, 10)
    table = [k] * 16
    frames = [((1.0, 1.05, 0.0), [k], [], table)] + [((1.0, 1.05, 0.0), [k], [(0.5, 0.3, range(16))], table)] * 3
    ev, _ = synthetic(frames, tmp_path / "e.npz")
    stops = first_stops(ev, StopSpec())
    assert stops["r1_s0.25_n3_h0.5"][0] == 2  # looks 3, hits 2 at step 2 (frame 0 = miss)
    assert stops["r1_s0.40_n2_h0.3"] is None  # raw 0.3 < 0.4: no hit
    behind = [((1.0, 1.05, math.pi), [k], [], table)] + [((1.0, 1.05, math.pi), [k], [(0.5, 0.3, range(16))], table)] * 3
    ev2, _ = synthetic(behind, tmp_path / "f.npz")
    assert first_stops(ev2, StopSpec())["r1_s0.25_n3_h0.5"] is None  # the cell is behind the robot


def test_rejected_spots_are_skipped(tmp_path):
    k = cell(12, 10)
    table = [k] * 16
    frames = [((1.0, 1.05, 0.0), [k], [(0.9, 0.5, range(16))], table)] * 4
    ev, _ = synthetic(frames, tmp_path / "e.npz")
    ev["det_px"][:] = 2000
    fires = [(t, f.cells.tolist()) for t, rep, f in rule_fires(ev, "r2_v0.50_g1")]
    assert fires and fires[0][1] == [k]
    rep = EvidenceReplay(ev)
    assert rep.excluded(np.array([k]), [rep.cell_xy(k)]).tolist() == [True]
    assert len(r1_grid()) == 60 and len(r2_grid()) == 27


def test_diagnosis_classes():
    tv = TeacherView((5.0, 0.0), 5.0, (1.0, 0.0))
    agree = Window(0, 20, (0.8, 0.0), 0, tv, TeacherView((5.0, 0.0), 4.0, (1.0, 0.0)), (0, 0), (0.8, 0), 1, 1)
    assert classify(agree) == "agree"
    stalled = Window(0, 20, (0.8, 0.0), 0, tv, TeacherView((5.0, 0.0), 5.0, (1.0, 0.0)), (0, 0), (0.8, 0), 1, 1)
    assert classify(stalled) == "bad_driving"
    wrong = Window(0, 20, (-0.8, 0.0), 0, tv, tv, (0, 0), (-0.8, 0), 2, 1)
    assert classify(wrong) == "wrong_place"
    bumpy = Window(0, 20, (0.8, 0.0), 12, tv, tv, (0, 0), (0.8, 0), 1, 1)
    assert classify(bumpy) == "bad_driving"
    still = Window(0, 20, (0.01, 0.0), 0, tv, tv, (0, 0), (0.01, 0), None, 1)
    assert classify(still) == "other_still"
    assert windows_of(45) == [(0, 20), (20, 40)] and windows_of(55) == [(0, 20), (20, 40), (40, 55)]


def test_mark_sightings_true_and_wrong(tmp_path):
    k, w = cell(12, 10), cell(30, 30)
    table = [k] * 8 + [w] * 8
    rec = EvidenceRecorder(SHAPE, (0.0, 0.0), RES, [7])
    sem = np.zeros((4, 4), np.int32)
    sem.reshape(-1)[:8] = 7  # the true target: top half (box [0, 0, 4, 2])
    for t in range(3):
        fc = cells_of([k, w], table)
        rec.add(t, (1.0, 1.05, 0.0), fc, [det(0.9, 0.5, range(8), box=(0, 0, 4, 2)),
                                           det(0.5, 0.1, range(8, 16), box=(0, 2, 4, 4))], sem)
    rec.save(tmp_path / "m.npz", {"obstacle": np.zeros(1600, np.int32), "explored": np.zeros(1600, np.int32)}, [])
    r = mark_sightings(load(tmp_path / "m.npz"))
    assert len(r["true"]) == 3 and all(x["marked"] for x in r["true"])
    assert len(r["wrong"]) == 3 and all(x["marked"] for x in r["wrong"])  # 0.5 >= 0.45: a wrong object marked


def test_run_plan_p14(monkeypatch):
    from mapmad_sim.run_plan_p14 import P14Plan

    cfg = {"sets": {"p13d": {"folder": "a"}, "p14d": {"folder": "b"}}, "arms": {}}
    plan = P14Plan(cfg, SimpleNamespace(base=Path("/nonexistent")))
    assert [c.label for c in plan.calls("baseline")] == ["base_p13d", "base_p13d_VF", "base_p14d", "base_p14d_VF"]
    drive = plan.calls("drive")
    assert len(drive) == 6 and drive[0].arms == ("M21b_a1", "M21bh_a1", "N_a1")
    monkeypatch.setenv("P14_EPISODES", "x y")
    plan = P14Plan(cfg, SimpleNamespace(base=Path("/nonexistent")))
    plan.episode_ids = lambda s: ["x"] if s == "p13d" else []
    assert plan.calls("baseline")[0].episodes == ("x",)


def test_arm_switches():
    a = Arm("M21b_a1w4", checkpoint="v21b", target="recogniser", recogniser="yoloe", pick="stick", unstuck="w4")
    assert a.switches == {"pick": "stick", "unstuck": "w4"}
    assert Arm("M21b", checkpoint="v21b").switches == {}
    with pytest.raises(ValueError):
        Arm("F_a1", policy="frontier", target="recogniser", recogniser="yoloe", pick="stick")
    with pytest.raises(ValueError):
        Arm("X", checkpoint="v21b", pick="best")


def test_reverse_only_when_allowed():
    from mapmad_sim.robot import clip_command

    spec = SimpleNamespace(max_v=0.2, max_w=0.4)
    assert clip_command(spec, -0.1, 0.0) == (0.0, 0.0)
    assert clip_command(spec, -0.1, 0.0, allow_reverse=True) == (-0.1, 0.0)
    assert clip_command(spec, -0.5, 0.9, allow_reverse=True) == (-0.2, 0.4)


def test_practice_split_every_category_in_both_halves():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from make_p14_sets import practice_split

    eps = [{"home": f"h{i % 6}", "target": {"category": c}} for i, c in enumerate(["chair", "bed", "sofa"] * 6)]
    sp = practice_split(eps, 14003)
    assert set(sp["A"]) | set(sp["B"]) == {f"h{i}" for i in range(6)} and not set(sp["A"]) & set(sp["B"])


def test_qwen_helpers():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from p14_qwen_check import answer_of, cell_box, crop

    assert answer_of("Yes.") == "yes" and answer_of(" no") == "no" and answer_of("Maybe") == "unsure"
    cam = np.eye(4)  # camera at the origin looking along -z (Habitat): 2D (X, Y) = (-z, -x)
    assert cell_box((-1.0, 0.0), 0.1, -0.5, cam, 90.0, (240, 320)) is None  # X = -1 -> z = +1: behind the camera
    box = cell_box((2.0, 0.0), 0.1, -0.5, cam, 90.0, (240, 320))
    assert box is not None and box[0] < 160 < box[2]
    assert crop(np.zeros((240, 320, 3), np.uint8), [100, 100, 110, 110]).shape[:2] == (96, 96)


def test_strength_colours_handles_empty_frames():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from p14_videos import strength_colours

    assert strength_colours(np.array([])).shape == (0, 3)
    assert strength_colours(np.array([0.0, 1.0])).tolist() == [[13, 8, 135], [240, 249, 33]]


def test_zoom_window_is_the_networks_near_window():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from p14_videos import window_cells, zoom_px
    from vint_train.mapmad.map_v2 import near_window

    rng = np.random.default_rng(0)
    shape, origin, res, pose = (120, 100), (-3.0, -2.0), 0.1, (2.13, 1.71, 0.7)
    grid = rng.random(shape).astype(np.float32)
    flat, inside = window_cells(pose, origin, res, shape)
    ours = np.where(inside, grid.reshape(-1)[flat], 0.0)
    net = near_window({n: grid for n in ("obstacle", "explored", "frontier", "target")}, origin, res, pose)[3]
    assert np.array_equal(ours, net)
    ahead = (pose[0] + math.cos(pose[2]), pose[1] + math.sin(pose[2]))  # 1 m straight ahead: 10 cells up
    assert np.allclose(zoom_px(ahead, pose), (256.0, 256.0 - 80.0))


def test_truth_marks_turn_true_pixels_into_full_score_looks(tmp_path):
    from mapmad_sim.stop_rules import truth_marks

    k = cell(12, 10)  # 0.25 m ahead of the robot
    table = [k] * 8 + [-1] * 8
    sem = np.zeros((4, 4), np.int32)
    sem.reshape(-1)[:8] = 7  # the target instance (id 7) on the pixels that land in cell k
    rec = EvidenceRecorder(SHAPE, (0.0, 0.0), RES, [7])
    for t in range(3):
        rec.add(t, (1.0, 1.05, 0.0), cells_of([k], table), [det(0.1, 0.1, range(8))] if t else [], sem)
    first = {"obstacle": np.full(SHAPE[0] * SHAPE[1], -1, np.int32), "explored": np.zeros(SHAPE[0] * SHAPE[1], np.int32)}
    rec.save(tmp_path / "e.npz", first, [])
    ev = load(tmp_path / "e.npz")
    assert first_stops(ev)["today"] is None  # the recogniser's weak marks (0.1) never reach tau
    tm = truth_marks(ev)
    assert tm["det_value"].tolist() == [1.0, 1.0, 1.0] and tm["det_cells"].tolist() == [k, k, k]
    assert tm["view"] is ev["view"] and tm["pose"] is ev["pose"]
    stop = first_stops(tm)["today"]
    assert stop is not None and stop[0] == 1  # frame 1: two looks at value 1.0
