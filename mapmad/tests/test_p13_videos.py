"""Phase 13 checkable videos (step 13.1): the pure parts of mapmad_sim/video_events.py (IoU rule, true boxes, first
YOLOE hit, timeline events from a synthetic log through p12_score, box-equality check, same-run comparison)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mapmad_sim import video_events as ve

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))


# --- IoU rule -----------------------------------------------------------------------------------------------------
def test_iou_basic():
    assert ve.iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert ve.iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0  # touching edges: no overlap
    assert ve.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert ve.iou([0, 0, 0, 10], [0, 0, 0, 10]) == 0.0  # no area


def test_hit_threshold_is_inclusive():
    true = [{"id": 1, "box": [0, 0, 10, 10]}]
    # IoU exactly 0.3: box [0, 0, 3, 10] inside -> 30 / 100
    assert ve.hit([{"box": [0, 0, 3, 10], "score": 0.1, "value": 0.1}], true)
    assert not ve.hit([{"box": [0, 0, 2.9, 10], "score": 0.1, "value": 0.1}], true)
    assert not ve.hit([{"box": None, "point": [1, 1], "score": 0.9, "value": 0.9}], true)  # point detections never hit


def test_true_boxes_pixel_edges():
    sem = np.zeros((6, 8), np.uint32)
    sem[1:3, 2:5] = 7  # rows 1-2, cols 2-4
    sem[5, 7] = 9
    sem[0, 0] = 4  # not a target id
    out = ve.true_boxes(sem, [9, 7, 11])
    assert out == [{"id": 7, "box": [2, 1, 5, 3], "px": 6}, {"id": 9, "box": [7, 5, 8, 6], "px": 1}]


def test_first_hit_needs_the_same_frame():
    dets = {0: [{"box": [0, 0, 10, 10], "score": 0.5, "value": 0.5}], 1: [], 2: [{"box": [1, 1, 10, 10], "score": 0.2,
                                                                                 "value": 0.3}]}
    trues = {0: [], 1: [{"id": 3, "box": [0, 0, 10, 10]}], 2: [{"id": 3, "box": [0, 0, 10, 10]}]}
    assert ve.first_hit(dets, trues) == 2
    assert ve.first_hit(dets, {0: [], 1: [], 2: []}) is None


# --- events from a synthetic log --------------------------------------------------------------------------------------
def synthetic_log(rule_values, vps, in_view, rule_cells=None):
    """A Phase 12 style log: header + steps 1..n (vps[0] / in_view[0] = the start)."""
    head = {"kind": "episode", "episode_id": "e", "arm": "A", "seed_index": 0, "home": "h", "category": "bed",
            "distance_bin": 0, "start_vp_geodesic_m": vps[0], "noise_seed": 5,
            "start": {"position": [0.0, 0.0, 0.0], "yaw": 0.0, "in_view": in_view[0]},
            "instances": [{"centre": [0.0, 0.5, -3.0], "size": [1.0, 1.0, 1.0]}]}
    steps = []
    for t in range(1, len(vps)):
        steps.append({"kind": "step", "step": t, "position": [0.0, 0.0, -0.05 * t], "yaw": 0.0, "travelled_m": 0.05,
                      "collided": False, "vp_geodesic_m": vps[t], "in_view": in_view[t], "rule_value": rule_values[t - 1],
                      "rule_cell": (rule_cells or {}).get(t, [3.0, 0.0])})
    return [head] + steps + [{"kind": "result"}]


def score(log, tau=0.45):
    from vint_train.mapmad.closed_loop import p12_score

    return p12_score.score_log(log, taus=[tau])


def test_events_oracle_and_right_object_rule_stop():
    vps = [3.0, 2.0, 1.0, 0.5, 0.15, 0.1]
    log = synthetic_log([0.1, 0.2, 0.3, 0.4, 0.5], vps, [False, False, True, True, True, True])
    ev = ve.timeline_events(score(log), 0.45, hit_step=2)
    assert [(e.step, e.kind) for e in ev] == [(2, "sight"), (2, "hit"), (4, "oracle"), (5, "rule")]
    assert ev[-1].how == "right object" and ev[-1].text == "rule stop 5: right object"


def test_events_wrong_object_near_miss_outside():
    vps = [3.0, 2.0, 1.0]
    far = {2: [-3.0, 3.0]}  # 2D frame (X, Y) -> Habitat x = -3, z = 3: > 1 m from the box at (0, -3)
    s = score(synthetic_log([0.1, 0.9], vps, [False] * 3, far))
    assert ve.timeline_events(s, 0.45, None)[-1].how == "wrong object"
    s = score(synthetic_log([0.1, 0.9], vps, [False] * 3))  # cell at the box, 1.0 m walk -> near miss
    assert ve.timeline_events(s, 0.45, None)[-1].how == "near miss"
    s = score(synthetic_log([0.1, 0.9], [3.0, 2.0, 2.0], [False] * 3))
    assert ve.timeline_events(s, 0.45, None)[-1].how == "outside"
    s = score(synthetic_log([0.1, 0.2], vps, [False] * 3))  # never >= tau: no rule stop, no sight, no oracle
    assert ve.timeline_events(s, 0.45, None) == []


def test_rule_stop_ignores_the_start():
    log = synthetic_log([0.1, 0.9], [3.0, 2.0, 1.0], [False] * 3)
    log[0]["start_rule"] = {"value": 0.99, "cell": None}  # step 0 never stops (step numbers start at 1)
    assert ve.timeline_events(score(log), 0.45, None)[-1].step == 2


# --- box check --------------------------------------------------------------------------------------------------------
LOGGED = [{"box": [0.0, 1.5, 100.25, 80.0], "point": None, "score": 0.6234, "value": 0.51},
          {"box": None, "point": [3.0, 4.0], "score": 0.9, "value": 0.9},
          {"box": [5.0, 6.0, 7.0, 8.0], "point": None, "score": 0.012, "value": 0.40}]


def test_compare_boxes_equal():
    drawn = [{"box": [0.0, 1.5, 100.25, 80.0], "label": "couch 0.62", "thick": True},
             {"box": [5.0, 6.0, 7.0, 8.0], "label": "couch 0.01", "thick": False}]
    assert ve.compare_boxes(drawn, LOGGED, "couch", 0.45) is None


@pytest.mark.parametrize("change", ["order", "value", "label", "thick", "missing"])
def test_compare_boxes_catches_differences(change):
    drawn = [{"box": [0.0, 1.5, 100.25, 80.0], "label": "couch 0.62", "thick": True},
             {"box": [5.0, 6.0, 7.0, 8.0], "label": "couch 0.01", "thick": False}]
    if change == "order":
        drawn.reverse()
    elif change == "value":
        drawn[0]["box"][2] = 100.3
    elif change == "label":
        drawn[0]["label"] = "couch 0.63"
    elif change == "thick":
        drawn[1]["thick"] = True
    else:
        drawn.pop()
    assert ve.compare_boxes(drawn, LOGGED, "couch", 0.45) is not None


def test_read_dets_needs_every_step(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text("".join(json.dumps({"step": t, "dets": []}) + "\n" for t in (0, 1, 2)))
    assert sorted(ve.read_dets(f)) == [0, 1, 2]
    f.write_text("".join(json.dumps({"step": t, "dets": []}) + "\n" for t in (0, 2)))
    with pytest.raises(ValueError):
        ve.read_dets(f)


# --- same-run ---------------------------------------------------------------------------------------------------------
def test_compare_runs_prefix_and_difference():
    long = synthetic_log([0.1] * 5, [3.0] * 6, [False] * 6)
    short = synthetic_log([0.1] * 2, [3.0] * 3, [False] * 3)
    r = ve.compare_runs(short, long)
    assert r["verdict"] == "reproducible" and r["prefix_steps"] == 2 and not r["full_length"]
    bad = synthetic_log([0.1] * 2, [3.0] * 3, [False] * 3)
    bad[2]["yaw"] = 0.001
    r = ve.compare_runs(bad, long)
    assert r["verdict"] == "not reproducible" and r["first_differing_step"] == 2
    moved = synthetic_log([0.1] * 2, [3.0] * 3, [False] * 3)
    moved[0]["start"]["position"] = [0.0, 0.0, 0.1]
    assert ve.compare_runs(moved, long)["first_differing_step"] == 0
