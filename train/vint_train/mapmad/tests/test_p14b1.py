"""Phase 14b-1 bench logic (NoMaD side): picks and ties, the keep-rule chain, the vectorised box distance."""

import numpy as np
import pytest

pytest.importorskip("mapmad_sim")

from vint_train.mapmad.closed_loop import p14b_bench as b  # noqa: E402
from vint_train.mapmad.closed_loop.p12_score import box_distance_2d  # noqa: E402


def test_pick_prefers_score_then_higher_threshold_then_milder_variant():
    sc = {("v1", "t_tau0.40"): 0.10, ("v1", "t_tau0.50"): 0.10, ("v2", "t_tau0.50"): 0.10, ("v1", "t_tau0.60"): 0.09}
    v, k, s = b.pick(lambda a, c: sc[(a, c)], list(sc))
    assert (v, k, s) == ("v1", "t_tau0.50", 0.10)
    assert b.key_order("r1_s0.10_n2_h0.5") > b.key_order("r1_s0.10_n3_h0.5")  # fewer looks preferred
    assert b.key_order("r2_v0.50_g1|f3") > b.key_order("r2_v0.45_g1|f3")


def test_family_keys_split_f3():
    off, on = b.family_keys("r2", False), b.family_keys("r2", True)
    assert len(off) == len(on) == 14 * 3 and all(k.endswith("|f3") for k in on)


def summ(**rows):
    return {r: {"mean": {"B": {"success": v}}} for r, v in rows.items()}


def test_recommend_needs_every_step_of_a_combination():
    s = summ(base=0.10, base14a=0.10, f3=0.13, f2=0.11, f1med=0.13, f1clu=0.12, f1best=0.13, f1f2=0.16, f1f2f3=0.19,
             f1f2f3c=0.20)
    helps = {"f3<base": True, "f2<base": False, "f1med<base": True, "f1clu<base": False, "f1f2<f2": True,
             "f1f2<f1best": True, "f1f2f3<f1f2": True, "f1f2f3c<f1f2f3": False}
    keep = {f: {k: {"helps": v} for k, v in helps.items()} for f in b.FAMILIES}
    summs = {f: s for f in b.FAMILIES}
    rec = b.recommend(keep, summs)
    assert rec["r2"]["row"] == "f1f2f3"  # c_cat did not help; F2 alone did not, but on top of F1 it did
    helps["f1f2<f1best"] = False
    keep = {f: {k: {"helps": v} for k, v in helps.items()} for f in b.FAMILIES}
    assert b.recommend(keep, summs)["r2"]["row"] in ("f3", "f1med")


def test_box_dist_cells_matches_box_distance_2d():
    inst = {"centre": [1.0, 0.5, -2.0], "size": [0.8, 1.0, 0.4]}
    pts = np.array([[2.0, -1.0], [0.0, 0.0], [2.1, -0.9]])
    got = b.box_dist_cells(pts, [inst])
    assert got == pytest.approx([box_distance_2d(p, inst) for p in pts])


def test_ece_zero_for_calibrated_bins():
    p = np.array([0.15] * 20 + [0.85] * 20)
    y = np.array([1] * 3 + [0] * 17 + [1] * 17 + [0] * 3)
    assert b.ece(p, y) == pytest.approx(0.0)


def _square_or_die(x: int) -> int:
    import os

    if x == 3:
        os._exit(9)  # a worker dying mid-task (as in the 14b-1 chain)
    return x * x


def test_imap_safe_yields_all_and_fails_fast_on_a_dead_worker():
    from concurrent.futures.process import BrokenProcessPool

    assert sorted(b.imap_safe(_square_or_die, [1, 2, 4], 2)) == [1, 4, 16]
    with pytest.raises(BrokenProcessPool):
        list(b.imap_safe(_square_or_die, [1, 2, 3, 4], 2))
