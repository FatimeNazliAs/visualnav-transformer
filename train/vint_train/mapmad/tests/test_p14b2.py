"""Phase 14b-2 tables (NoMaD side): the detector pick, the keep rule, detection share on a target, cell effects."""

import numpy as np
import pytest

pytest.importorskip("mapmad_sim")

from vint_train.mapmad.closed_loop import p14b2_tables as t  # noqa: E402


def test_choose_prefers_half_a_f1_then_base_then_lp():
    rows = {"yoloe": {"f1_half_a": 0.67}, "yoloe_lp": {"f1_half_a": 0.72}, "yoloe_ft": {"f1_half_a": 0.72}}
    assert t.choose(rows) == "yoloe_lp"  # tie LP / FT -> LP (the milder change)
    rows["yoloe_ft"]["f1_half_a"] = 0.75
    assert t.choose(rows) == "yoloe_ft"
    assert t.choose({"yoloe": {"f1_half_a": 0.7}, "yoloe_lp": {"f1_half_a": 0.7}}) == "yoloe"  # tie -> base


def test_keep_rule():
    assert t.keep(0.02, 0.30, 0.30, 0.02)
    assert not t.keep(0.019, 0.40, 0.30, 0.02)
    assert not t.keep(0.05, 0.29, 0.30, 0.02)


def test_on_target_share_counts_iou_with_true_boxes_of_the_same_frame():
    ev = {"det_t": np.array([0, 0, 1]), "det_box": np.array([[0, 0, 10, 10], [50, 50, 60, 60], [0, 0, 10, 10.0]]),
          "true_t": np.array([0]), "true_box": np.array([[1, 1, 10, 10.0]])}
    assert t.on_target_share(ev) == (3, 1)  # frame 1 has no true box


def test_cell_effect_paired_success_and_pooled_precision():
    half_of = {f"e{k}": "B" for k in range(6)}
    home_of = {f"e{k}": f"h{k % 3}" for k in range(6)}

    def run(s, stopped):
        return {"success": s, "stopped": stopped, "oracle": 1.0, "wrong": 0.0, "collisions": 0.0, "stuck": 0.0}
    data = {"new": {a: {f"e{k}": run(float(k < 3), 1.0) for k in range(6)} for a in t.NET_ARMS},
            "old": {a: {f"e{k}": run(float(k < 1), 1.0) for k in range(6)} for a in t.NET_ARMS}}
    e = t.cell_effect(data, "new", "old", half_of, home_of, 500, 1)
    assert e["success"]["diff"] == pytest.approx(2 / 6)
    assert e["precision"]["new"] == pytest.approx(0.5) and e["precision"]["old"] == pytest.approx(1 / 6)
