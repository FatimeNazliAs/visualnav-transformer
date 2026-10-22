"""Model output -> RawDets, per wrapper, with fake model outputs (no weights): the model-specific mapping that the
photo test depends on (the RF-DETR COCO-id bug of 2026-10-22 lived here)."""

import numpy as np
import pytest

from mapmad_recog.models import (COCO_NAME, box_dets, coco_ids, owlv2_dets, owlv2_square, rfdetr_dets,
                                 ultralytics_dets, vlm_dets)

HW = (240, 320)
COCO = {1: "person", 56: "broccoli", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 70: "toilet", 72: "tv"}


def test_rfdetr_keeps_the_asked_coco_id_only():
    class_id = np.array([62, 56, 63, 62])  # chair, broccoli, couch, chair (predict() ids)
    scores = np.float32([0.9, 0.8, 0.7, 0.2])
    boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 2, 2], [300, 200, 400, 300]], np.float32)
    dets = rfdetr_dets(class_id, scores, boxes, "chair", coco_ids(COCO), HW)
    assert [round(d.score, 3) for d in dets] == [0.9, 0.2]
    assert dets[1].box.tolist() == [300, 200, 320, 240]  # clipped to the picture
    assert [round(d.score, 3) for d in rfdetr_dets(class_id, scores, boxes, "couch", coco_ids(COCO), HW)] == [0.7]
    assert rfdetr_dets(class_id, scores, boxes, "tv", coco_ids(COCO), HW) == []


def test_owlv2_boxes_live_in_the_padded_square():
    assert owlv2_square(HW) == (320, 320)
    boxes = np.array([[10, 10, 50, 60], [0, 250, 40, 300]], np.float32)  # the second lies wholly in the padding
    dets = owlv2_dets(np.float32([0.5, 0.4]), boxes, HW)
    assert len(dets) == 1 and dets[0].box.tolist() == [10, 10, 50, 60]


def test_ultralytics_drops_masks_of_another_size():
    masks = np.zeros((2, 240, 320), bool)
    dets = ultralytics_dets(np.float32([0.3, 0.6]), np.zeros((2, 4)), masks, HW)
    assert all(d.mask is not None for d in dets)
    small = ultralytics_dets(np.float32([0.3]), np.zeros((1, 4)), np.zeros((1, 120, 160), bool), HW)
    assert small[0].mask is None
    assert ultralytics_dets(np.float32([0.3]), np.zeros((1, 4)), None, HW)[0].mask is None  # YOLO-World: boxes only


def test_box_dets_keep_masks_and_clip_boxes():
    m = np.ones(HW, bool)
    d = box_dets(np.float32([0.7]), np.array([[-5, -5, 400, 300]]), HW, [m])[0]
    assert d.box.tolist() == [0, 0, 320, 240] and d.mask is m


def test_vlm_answers_become_boxes_or_points():
    q = vlm_dets("qwen3vl", '```json\n[{"bbox_2d": [0, 0, 500, 500], "label": "chair"}]\n```', HW)
    assert len(q) == 1 and q[0].score == 1.0 and q[0].box.tolist() == [0, 0, 160, 120] and q[0].point is None
    assert vlm_dets("qwen3vl", "[]", HW) == []
    m = vlm_dets("molmo2", '<points coords="1 500 500;2 250 750">chair</points>', HW)
    assert [p.point.tolist() for p in m] == [[160, 120], [80, 180]] and m[0].box is None
    assert vlm_dets("molmo2", "There is no chair in the image.", HW) == []


def test_rfdetr_table_matches_predict_ids():
    coco = pytest.importorskip("rfdetr.assets.coco_classes")  # only in naz_mapmad_recog
    ids = coco_ids(coco.COCO_CLASSES)
    assert {p: ids[n] for p, n in COCO_NAME.items()} == {"chair": 62, "couch": 63, "bed": 65, "toilet": 70, "tv": 72,
                                                        "potted plant": 64}
