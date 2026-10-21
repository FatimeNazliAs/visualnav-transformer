"""Phase 9 photo test: labels, frame set, scoring and the choice rule (no simulator, no model)."""

import math

import numpy as np
import pytest

from mapmad_sim import photo_frames as pf
from mapmad_sim import photo_labels as pl
from mapmad_sim import photo_scoring as ps

MAPPING = {"chair": "chair", "armchair": "chair", "tv": "tv_monitor", "monitor": "tv_monitor", "bed": "bed",
           "couch": "sofa", "sofa": "sofa", "toilet": "toilet", "plant": "plant", "flower vase": "plant"}


def inst(i, name, px=500, depth=2.0, box=(0, 0, 10, 10), centre=(0.0, 0.0, -2.0), size=(0.5, 1.0, 0.5), edge=False):
    return pf.Instance(i, name, px, depth, px, list(box), edge, list(centre), list(size))


# --- labels --------------------------------------------------------------------------------------------------------

def test_near_synonyms_use_words_and_exclusions():
    names = ["chair", "stool", "kitchen chair", "tv", "led tv", "tv stand", "laptop", "bedside table", "crib",
             "toilet paper", "urinal", "vase", "cat tree", "pot"]
    near = pl.near_synonyms(names, MAPPING)
    assert near["chair"] == ["kitchen chair", "stool"]
    assert near["tv_monitor"] == ["laptop", "led tv"]  # "tv stand" is furniture (NOT_NEAR)
    assert near["bed"] == ["crib"]  # "bedside" is a different word
    assert near["toilet"] == ["urinal"]
    assert near["plant"] == ["pot", "vase"]


def test_six_tables_roles_and_prompts():
    t = pl.six_tables(["chair", "stool", "table"], MAPPING)
    assert t["chair"].role("Chair") == "counted" and t["chair"].role("stool") == "ignore"
    assert t["chair"].role("table") == "background"
    assert t["plant"].prompt == "potted plant" and t["sofa"].prompt == "couch" and t["tv_monitor"].prompt == "tv"


def test_ovon_table_by_name():
    t = pl.ovon_table("office chair", ["office chair", "chair", "office chair cushion", "table"])
    assert t.counted == {"office chair"}
    assert t.ignore == {"chair", "office chair cushion"}
    assert pl.slug("Potted plant") == "potted_plant"


# --- frames --------------------------------------------------------------------------------------------------------

def test_frame_instances_counts_and_boxes():
    sem = np.zeros((4, 6), np.uint32)
    sem[1:3, 0:2] = 5
    depth = np.full((4, 6), 2.0, np.float32)
    depth[1, 0] = 0.0
    out = pf.frame_instances(sem, depth, {5: "chair"}, {5: (np.array([1.0, 0.5, 2.0]), np.array([0.5, 1.0, 0.5]))})
    assert len(out) == 1
    i = out[0]
    assert (i.id, i.name, i.pixels, i.near_pixels, i.box2d, i.touches_edge) == (5, "chair", 4, 3, [0, 1, 2, 3], True)
    assert i.median_depth_m == 2.0 and i.world_centre == [1.0, 0.5, 2.0]


def test_counted_rule_and_status():
    table = pl.six_tables(["chair", "stool"], MAPPING)["chair"]
    assert pf.item_status([inst(1, "chair", px=200, depth=4.0)], table) == "positive"
    assert pf.item_status([inst(1, "chair", px=199)], table) == "ignore_only"
    assert pf.item_status([inst(1, "chair", depth=4.01)], table) == "ignore_only"
    assert pf.item_status([inst(1, "stool")], table) == "free"
    assert pf.item_status([], table) == "free"


def test_thin_poses():
    xy = np.array([[0, 0], [0.5, 0], [1.0, 0], [1.2, 0], [1.2, 0]], float)
    yaw = np.array([0, 0, 0, 0, math.radians(31)])
    assert pf.thin_poses(xy, yaw) == [0, 2, 4]


def frames_pool(n_home=10, per=60):
    rng = np.random.default_rng(0)
    out = []
    for h in range(n_home):
        for k in range(per):
            st = {c: ("positive" if rng.random() < 0.3 else ("ignore_only" if rng.random() < 0.05 else "free"))
                  for c in pl.CATEGORIES}
            out.append({"frame_id": pf.frame_id(f"{h:05d}-x", "rnd", k), "home": f"{h:05d}-x", "src": "rnd", "status": st})
    return out


def test_select_items_shares_caps_and_ids():
    tables = pl.six_tables(["chair"], MAPPING)
    log = []
    items = pf.select_items(frames_pool(), tables, pf.SelectSpec(seed=1, max_pos=100), log=log)
    for c in pl.CATEGORIES:
        sel = [i for i in items if i["key"] == c]
        n = {s: sum(i["status"] == s for i in sel) for s in pf.STATUSES}
        assert n["positive"] == 100
        assert 0.50 <= n["free"] / len(sel) <= 0.60
    counts = {}
    for i in items:
        counts[i["home"]] = counts.get(i["home"], 0) + 1
    assert max(counts.values()) <= 0.2 * len(items)
    assert len({i["item_id"] for i in items}) == len(items)
    assert items == pf.select_items(frames_pool(), tables, pf.SelectSpec(seed=1, max_pos=100))  # seeded


def test_home_cap_drops_free_items():
    items = [{"item_id": f"a{k}", "home": "A", "status": "free"} for k in range(5)] + \
            [{"item_id": f"b{k}", "home": f"B{k}", "status": "free"} for k in range(5)]
    log = []
    out = pf.cap_homes(items, 0.2, log)
    assert sum(i["home"] == "A" for i in out) <= 0.2 * len(out) + 1e-9 and log


def test_duplicates_refused():
    with pytest.raises(ValueError):
        pf.check_unique(["p9q-a", "p9q-a"], "item")


def test_split_halves_balanced_and_seeded():
    homes = [f"{k:05d}" for k in range(10)]
    pos = {h: {"chair": 10, "tv": (5 if k < 3 else 0)} for k, h in enumerate(homes)}
    a, b, why = pf.split_halves(homes, pos, seed=3)
    assert len(a) == len(b) == 5 and set(a) | set(b) == set(homes)
    tv_a = sum(pos[h]["tv"] for h in a)
    assert 0 < tv_a < 15 and why["worst_share"] >= 0.3
    assert (a, b) == pf.split_halves(homes, pos, seed=3)[:2]


# --- scoring -------------------------------------------------------------------------------------------------------

def make_frame():
    """8 x 8 frame: a chair (id 1, counted) top-left, a stool (id 2, near-synonym) bottom-right, a table (3)."""
    sem = np.zeros((8, 8), np.int64)
    sem[0:4, 0:4] = 1
    sem[4:8, 4:8] = 2
    sem[0:2, 6:8] = 3
    world = np.full((64, 3), np.nan)
    for y in range(8):
        for x in range(8):
            world[y * 8 + x] = {1: [0.0, 0.5, -2.0], 2: [3.0, 0.5, -2.0], 3: [-3.0, 0.5, -2.0]}.get(sem[y, x], [9.0, 0.5, 9.0])
    instances = [inst(1, "chair", box=(0, 0, 4, 4), centre=(0, 0.5, -2)), inst(2, "stool", box=(4, 4, 8, 8), centre=(3, 0.5, -2)),
                 inst(3, "table", box=(6, 0, 8, 2), centre=(-3, 0.5, -2))]
    return ps.FrameData(instances, sem, world)


def mask(y0, y1, x0, x1):
    m = np.zeros((8, 8), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_label_item_pixel_and_map_labels():
    fd = make_frame()
    table = pl.six_tables(["chair", "stool", "table"], MAPPING)["chair"]
    truth = ps.item_truth(fd.instances, table)
    boxes = np.array([[0, 0, 4, 4], [0, 0, 4, 4], [4, 4, 8, 8], [6, 0, 8, 2]], float)
    masks = [mask(0, 4, 0, 4), mask(0, 4, 0, 4), mask(4, 8, 4, 8), mask(0, 2, 6, 8)]
    dets = ps.label_item("q", fd, truth, np.array([0.9, 0.8, 0.7, 0.6]), boxes, np.full((4, 2), np.nan), masks,
                         np.zeros(3))
    by = {d.score: d for d in dets}
    assert (by[0.9].pix, by[0.9].map, by[0.9].hits) == ("TP", "hit", (1,))
    assert (by[0.8].pix, by[0.8].map) == ("FP", "hit")  # duplicate: pixel FP, still a map hit
    assert (by[0.7].pix, by[0.7].map) == ("IGN", "IGN")  # the stool
    assert (by[0.6].pix, by[0.6].map, by[0.6].found) == ("FP", "miss", "table")


def test_point_detections_and_off_map():
    fd = make_frame()
    fd.world[0] = np.nan  # pixel (0, 0) has no drawable point
    truth = ps.item_truth(fd.instances, pl.six_tables(["chair", "stool", "table"], MAPPING)["chair"])
    pts = np.array([[1, 1], [0, 0], [7, 0]], float)
    dets = ps.label_item("q", fd, truth, np.ones(3), np.full((3, 4), np.nan), pts, [None] * 3, np.zeros(3))
    assert [(d.pix, d.map) for d in dets] == [("TP", "hit"), ("FP", "off"), ("FP", "miss")]


def test_metrics_ap_and_tuning():
    items = [ps.ItemInfo("a", "chair", "positive", "A", "rnd", [inst(1, "chair")]),
             ps.ItemInfo("b", "chair", "free", "A", "rnd", [])]
    dets = [ps.Det("a", 0.9, "TP", "hit", (1,), 1), ps.Det("b", 0.5, "FP", "miss"), ps.Det("b", 0.2, "FP", "miss")]
    m = ps.metrics(dets, items, 0.3)
    assert (m["precision"], m["recall"], m["fa_per_free"], m["H"], m["R"]) == (0.5, 1.0, 1.0, 0.5, 1.0)
    assert m["ap50"] == pytest.approx(1.0)
    t, f = ps.tune(dets, items)
    assert t == 0.51 and f == pytest.approx(1.0)  # lowest threshold that drops both false alarms
    assert ps.box_distance_xz(np.array([1.6, 0, 0]), [0, 0, 0], [2, 2, 2]) == pytest.approx(0.6)
    assert ps.iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)


def row(f1, s, kind="open", family="dino", ran=True):
    return {"F1": f1, "median_s": s, "kind": kind, "family": family, "ran": ran}


def test_choice_rule_tie_family_and_vlm():
    models = {"gd": row(0.60, 0.10), "ll": row(0.595, 0.05), "ow": row(0.50, 0.08, family="clip_vit"),
              "ye": row(0.40, 0.02, family="fast_yolo"), "rf": row(0.90, 0.01, kind="fixed", family="fixed"),
              "qw": row(0.495, 0.03, kind="vlm", family="vlm"), "slow": row(0.99, 0.30, family="sam_text")}
    ch = ps.choose(models)
    assert ch["R1"] == "ll"  # tie within 0.01 -> the faster; "slow" does not qualify, RF-DETR is fixed
    assert ch["R2"] == "ow"  # the VLM ties and is faster, but is not the best of R2's pool -> not eligible


def test_choice_vlm_only_if_best():
    models = {"ow": row(0.50, 0.08, family="clip_vit"), "qw": row(0.55, 0.20, kind="vlm", family="vlm"),
              "gd": row(0.60, 0.10)}
    ch = ps.choose(models)
    assert ch["R1"] == "gd" and ch["R2"] == "qw"  # the VLM has the best F1 of R2's pool -> eligible


def test_choice_fallback():
    models = {"a": row(0.6, 0.30), "b": row(0.5, 0.40, family="clip_vit"), "c": row(0.7, 0.20, family="fast_yolo")}
    ch = ps.choose(models)
    assert ch["R1"] == "c" and ch["R2"] == "a" and ch["fallback_flags"]
