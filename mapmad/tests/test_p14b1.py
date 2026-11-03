"""Phase 14b-1: mark projection (F1), stop variants (F2 / F3 / tentative / c_cat), the bench helpers."""

import numpy as np
import pytest

import mapmad_sim.explorer  # noqa: F401  (puts train/ on sys.path)
from mapmad_sim.mark_projection import (FrameGeometry, MarkTable, ProjSpec, Voxels, box_fill_mask, erode,
                                        largest_cluster, median_filter, neighbour_graph, outer_third, project_mask,
                                        voxelize)
from mapmad_sim.stop_variants import (F3_TAG, RunGeometry, Variant, all_keys, calibrated, first_stops, rule_keys,
                                      variant_evidence)

H, W = 24, 32


def geometry(depth, cell_of_pixel=None):
    """A frame whose pixel k lands on cell k (or the given table) and whose world points are (col, row, depth)."""
    d = np.asarray(depth, np.float64)
    pc = np.arange(d.size) if cell_of_pixel is None else np.asarray(cell_of_pixel)
    rows, cols = np.mgrid[0:H, 0:W]
    world = np.stack([cols.ravel() * 0.01, rows.ravel() * 0.01, d.ravel()], axis=1)
    world[(d.ravel() <= 0) | (d.ravel() > 4.0)] = np.nan
    return FrameGeometry(d, pc.astype(np.int64), world)


def test_median_filter_drops_the_wall_behind_the_object():
    depth = np.full((H, W), 1.0)
    depth[:, 20:] = 3.5  # rim pixels hit the wall 2.5 m behind
    mask = np.zeros((H, W), bool)
    mask[4:20, 4:24] = True
    px = np.flatnonzero(mask)
    kept = median_filter(px, depth.ravel(), 0.3)
    assert set(kept.tolist()) == set(np.flatnonzero(mask & (depth == 1.0)).tolist())


def test_largest_cluster_is_weighted_and_min_points_counts_pixels():
    pts = np.concatenate([np.zeros((30, 3)), np.ones((10, 3)) * 5.0])  # two blobs, 30 and 10 pixels
    vox = voxelize(pts, 0.05)
    assert vox.weight.tolist() == [30, 10]
    g = neighbour_graph(vox.centre, 0.2)
    keep = largest_cluster(vox, g, 20)
    assert keep.tolist() == [True, False]
    assert not largest_cluster(vox, g, 31).any()  # no voxel reaches 31 pixels -> no mark
    single = voxelize(pts, 0.0)  # size 0 = plain DBSCAN, one point per voxel
    assert largest_cluster(single, neighbour_graph(single.centre, 0.2), 20).sum() == 30


def test_border_voxel_joins_its_core_neighbour():
    vox = Voxels(np.array([[0.0, 0, 0], [0.15, 0, 0], [0.3, 0, 0]]), np.array([50, 50, 1]), np.arange(3))
    keep = largest_cluster(vox, neighbour_graph(vox.centre, 0.2), 60)
    assert keep.tolist() == [True, True, True]  # voxel 2 has 51 px in reach (< 60) but borders core voxel 1


def test_project_mask_variants_and_tentative():
    depth = np.full((H, W), 1.0)
    depth[:, 28:] = 3.9
    mask = np.zeros((H, W), bool)
    mask[2:22, 2:30] = True
    spec = ProjSpec(min_px=(20,), erode_k=(1,), med_d=(0.3,))
    rm = project_mask(mask, (2, 2, 30, 22), geometry(depth), spec)
    assert rm.cells["base"].size == mask.sum()
    assert rm.cells["med_d0.3"].size == (mask & (depth == 1.0)).sum()
    clu = rm.cells["clu_k1_m20"]
    assert clu.size and set(clu.tolist()) <= set(np.flatnonzero(erode(mask, 1)).tolist())
    assert not rm.tentative["clu_k1_m20"]  # centred box, cluster at ~1 m
    rm2 = project_mask(mask & (np.arange(W) >= 24)[None, :], (24, 2, 32, 22), geometry(depth), spec)
    assert rm2.tentative["clu_k1_m20"]  # box in the right third
    assert outer_third((0, 0, 10, 5), W) and not outer_third((5, 0, 20, 5), W)


def test_box_fill_and_mark_table():
    depth = np.full((H, W), 2.0)
    depth[5:10, 5:10] = 0.0
    m = box_fill_mask((0, 0, 12, 12), depth)
    assert m.sum() == 144 - 25
    tab = MarkTable("det", ProjSpec(min_px=(20,), erode_k=(1,), med_d=(0.3,)))
    tab.add(project_mask(m, (0, 0, 12, 12), geometry(depth), tab.spec), fallback=1.0)
    a = tab.arrays()
    assert a["det_ptr_base"].tolist() == [0, 119] and a["det_fallback"].tolist() == [1.0]


# --- stop variants ----------------------------------------------------------------------------------------------------
SHAPE, RES = (40, 40), 0.1


def cell(i, j):
    return i * SHAPE[1] + j


def evidence(frames, dets):
    """frames: [(pose, view cells)]; dets: [(t, raw, value, px, cells, box)] -> an evidence dict."""
    view = [np.asarray(v, np.int32) for _, v in frames]
    ptr = np.concatenate([[0], np.cumsum([v.size for v in view])])
    dc = [np.asarray(d[4], np.int32) for d in dets]
    dptr = np.concatenate([[0], np.cumsum([c.size for c in dc])])
    return {"shape": np.array(SHAPE), "origin": np.zeros(2), "res": np.array(RES),
            "pose": np.array([p for p, _ in frames], np.float64), "view": np.concatenate(view), "view_ptr": ptr,
            "det_t": np.array([d[0] for d in dets], np.int32), "det_raw": np.array([d[1] for d in dets]),
            "det_value": np.array([d[2] for d in dets]), "det_px": np.array([d[3] for d in dets], np.int32),
            "det_box": np.array([d[5] for d in dets], np.float64).reshape(-1, 4),
            "det_cells": np.concatenate(dc) if dc else np.zeros(0, np.int32), "det_cells_ptr": dptr,
            "true_t": np.zeros(0, np.int32), "true_px": np.zeros(0, np.int32), "true_box": np.zeros((0, 4)),
            "true_id": np.zeros(0, np.int32)}


def proj_from(ev, tent=None):
    p = {"det_cells_base": ev["det_cells"], "det_ptr_base": ev["det_cells_ptr"]}
    p["det_cells_clu_k1_m20"], p["det_ptr_clu_k1_m20"] = ev["det_cells"], ev["det_cells_ptr"]
    p["det_tent_clu_k1_m20"] = np.asarray(tent if tent is not None else np.zeros(len(ev["det_t"]), bool))
    return p


ROBOT = (1.0, 1.05, 0.0)
AHEAD = cell(15, 10)  # 0.55 m ahead
BEHIND = cell(5, 10)  # 0.45 m behind


def test_f3_needs_the_cell_in_view_now():
    frames = [(ROBOT, [AHEAD, BEHIND])] * 3 + [(ROBOT, [cell(30, 30)])] * 2
    dets = [(1, 0.5, 0.9, 2000, [AHEAD, BEHIND], (0, 0, 9, 9)), (2, 0.5, 0.9, 2000, [AHEAD, BEHIND], (0, 0, 9, 9))]
    ev = evidence(frames, dets)
    st = first_stops(variant_evidence(ev, proj_from(ev), Variant()), RunGeometry(ev))
    assert st["t_tau0.45"] == (2, AHEAD) and st["t_tau0.45" + F3_TAG] == (2, AHEAD)
    ev2 = evidence([(ROBOT, [AHEAD, BEHIND])] * 3 + [(ROBOT, [cell(30, 30)])], dets)
    st2 = first_stops(variant_evidence(ev2, proj_from(ev2), Variant()), RunGeometry(ev2))
    # Rule 2 counts cells in any direction: behind fires as well as ahead; F3 only ahead and in view
    assert st2["r2_v0.50_g1"] is not None and st2["r2_v0.50_g1" + F3_TAG][1] == AHEAD
    frames3 = [(ROBOT, [BEHIND])] * 2 + [(ROBOT, [cell(30, 30)])] * 3  # the ahead cell is never in view
    dets3 = [(1, 0.5, 0.9, 2000, [AHEAD], (0, 0, 9, 9)), (2, 0.5, 0.9, 2000, [AHEAD], (0, 0, 9, 9))]
    ev3 = evidence(frames3, dets3)
    st3 = first_stops(variant_evidence(ev3, proj_from(ev3), Variant()), RunGeometry(ev3))
    assert st3["r2_v0.50_g1"] is not None and st3["r2_v0.50_g1" + F3_TAG] is None


def test_f2_calibrated_scores_and_c_cat():
    assert calibrated(np.array([0.5]), 2.6)[0] == pytest.approx(0.5)
    assert calibrated(np.array([0.08]), 1.0)[0] == pytest.approx(0.08)
    low = calibrated(np.array([0.08]), 2.6)[0]
    frames = [(ROBOT, [AHEAD])] * 4
    dets = [(1, 0.08, 0.45, 2000, [AHEAD], (0, 0, 9, 9)), (2, 0.08, 0.45, 2000, [AHEAD], (0, 0, 9, 9))]
    ev = evidence(frames, dets)
    geo = RunGeometry(ev)
    st_map = first_stops(variant_evidence(ev, proj_from(ev), Variant()), geo)
    st_cal = first_stops(variant_evidence(ev, proj_from(ev), Variant(score="cal"), 2.6), geo)
    assert st_map["t_tau0.30"] is not None  # map value 0.45 >= 0.30 (the floor makes it easy)
    assert (st_cal["t_tau0.30"] is not None) == (low * 2 / 2 >= 0.30)
    cc = variant_evidence(ev, proj_from(ev), Variant(score="cal", ccat=0.5), 2.6)
    assert cc["det_cells"].size == 0 and cc["det_cells_ptr"].tolist() == [0, 0, 0]


def test_tentative_look_is_deleted_when_seen_again_without_detection():
    frames2 = [(ROBOT, [cell(30, 30)]), (ROBOT, [cell(30, 30)]), (ROBOT, [AHEAD]), (ROBOT, [AHEAD]), (ROBOT, [AHEAD])]
    dets2 = [(1, 0.5, 0.9, 2000, [AHEAD], (0, 0, 9, 9)), (4, 0.5, 0.9, 100, [AHEAD], (0, 0, 9, 9))]
    ev2 = evidence(frames2, dets2)
    geo2 = RunGeometry(ev2)
    k2 = first_stops(variant_evidence(ev2, proj_from(ev2), Variant(proj="clu_k1_m20")), geo2)
    g2 = first_stops(variant_evidence(ev2, proj_from(ev2, [True, False]), Variant(proj="clu_k1_m20")), geo2)
    # kept: frame 2 = (0.9 + 0) / 2 looks = 0.45 >= 0.40; tentative: deleted at frame 2 (in view, no detection),
    # frame 4 = (0 + 0 + 0 + 0.9) / 4 = 0.225
    assert k2["t_tau0.40"] == (2, AHEAD) and g2["t_tau0.40"] is None and g2["t_tau0.20"] == (4, AHEAD)


def test_keys_cover_the_14a_points():
    keys = set(rule_keys())
    assert {"t_tau0.45", "r1_s0.10_n2_h0.5", "r2_v0.50_g1"} <= keys
    assert len(all_keys()) == 2 * len(keys)


def test_variant_names_round_trip():
    for v in (Variant(), Variant("true", "med_d0.3", "map"), Variant("det", "clu_k2_m50", "cal", 0.3)):
        assert Variant.parse(v.name) == v


def test_bench_ece_and_temperature():
    from vint_train.mapmad.closed_loop.p14b_bench import ece, fit_temperature

    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.05, 0.95, 4000)
    y = rng.random(4000) < p_true
    raw = calibrated(p_true, 1 / 2.0)  # over-confident scores: sigmoid(2 logit)
    t, _ = fit_temperature(raw, y, [round(0.4 + 0.05 * k, 2) for k in range(53)], 10)
    assert t == pytest.approx(2.0, abs=0.2)
    assert ece(calibrated(raw, t), y) < ece(raw, y)
