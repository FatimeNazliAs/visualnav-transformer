"""Phase 9 plug-in: interface, score mapping, truth parity with Phase 8's clean layer, a dummy recogniser drawing the
layer, the sidecar wire format, the VLM answer parsers and unique ids (no simulator, no model)."""

import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from mapmad_bridge import protocol
from mapmad_recog.models import RawDet, molmo_points, parse_json_list, qwen_boxes
from mapmad_sim import photo_frames as pf
from mapmad_sim.recogniser import (Detection, SidecarRecogniser, TruthRecogniser, detection_cells, frame_scores,
                                   map_score, unpack)
from mapmad_sim.target_layer import FrameCells, NoiseSpec, frame_scores as p8_frame_scores, detect
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
from vint_train.mapmad.map_v2 import TargetFusion  # noqa: E402


def scene(h=6, w=8):
    """Semantic + depth + a pixel -> cell table: instance 7 (20 px), instance 9 (4 px), one far pixel."""
    sem = np.zeros((h, w), np.uint32)
    sem[0:4, 0:5] = 7
    sem[4:6, 6:8] = 9
    depth = np.full((h, w), 2.0, np.float32)
    depth[0, 0] = 5.0  # beyond 4 m: no cell, not counted
    pixel_cell = np.arange(h * w, dtype=np.int64) // 2
    pixel_cell[0] = -1
    return sem, depth, FrameCells(pixel_cell, np.unique(pixel_cell[pixel_cell >= 0]), np.zeros(0, np.int64))


def test_map_score():
    assert map_score(0.3, 0.3, 0.8) == pytest.approx(0.4)
    assert map_score(0.55, 0.3, 0.8) == pytest.approx(0.7)
    assert map_score(0.95, 0.3, 0.8) == pytest.approx(1.0)
    assert map_score(0.5, 0.5, 0.5) == 1.0


def test_truth_through_plugin_equals_phase8_clean():
    sem, depth, cells = scene()
    spec = NoiseSpec(min_pixels=4)
    rec = TruthRecogniser({"chair": [7, 9]}, spec)
    dets = rec.detect(None, depth, "chair", semantic=sem)
    assert [d.instance for d in dets] == [7, 9] and all(d.score == 1.0 for d in dets)
    ours = frame_scores(dets, cells, depth.shape)
    p8 = p8_frame_scores(0, detect(sem, depth, [7, 9], spec), cells, (10, 10), spec, np.random.default_rng(0)).clean
    assert ours == p8
    fa, fb = TargetFusion((10, 10)), TargetFusion((10, 10))
    fa.update(0, cells.view, ours)
    fb.update(0, cells.view, p8)
    for k in ("cell", "frame", "value"):
        assert np.array_equal(fa.updates()[k], fb.updates()[k])


class Dummy:
    """A recogniser that always finds a fixed box-shaped mask (or a point) with one score."""

    name = "dummy"

    def __init__(self, score, point=None):
        self.score, self.point = score, point

    def detect(self, rgb, depth, category, semantic=None):
        if self.point is not None:
            return [Detection(self.score, point=self.point)]
        m = np.zeros(depth.shape, bool)
        m[4:6, 6:8] = True
        return [Detection(self.score, m)]


def test_dummy_recogniser_draws_and_fades():
    sem, depth, cells = scene()
    fus = TargetFusion((10, 10))
    fus.update(0, cells.view, frame_scores(Dummy(0.7).detect(None, depth, "tv"), cells, depth.shape))
    cells_hit = detection_cells(Dummy(0.7).detect(None, depth, "tv")[0], cells, depth.shape)
    assert np.allclose(fus.grid().reshape(-1)[cells_hit], 0.7)
    fus.update(1, cells.view, {})  # seen again without a detection: the mark fades to the mean
    assert np.allclose(fus.grid().reshape(-1)[cells_hit], 0.35)
    pt = Dummy(1.0, point=(1.0, 0.0)).detect(None, depth, "tv")[0]
    assert detection_cells(pt, cells, depth.shape).tolist() == [0]  # pixel 1 -> cell 0


def test_unpack_reply():
    h, w = 4, 4
    m = np.zeros((h, w), bool)
    m[1:3, 1:3] = True
    reply = {"scores": np.array([0.9, 0.2], np.float32), "boxes": np.array([[1, 1, 3, 3], [0, 0, 1, 1]], np.float32),
             "points": np.full((2, 2), np.nan, np.float32), "masks": np.stack([np.packbits(m.reshape(-1))] * 2)}
    dets = unpack(reply, (h, w), lambda s: map_score(s, 0.3, 0.9), min_score=0.3)
    assert len(dets) == 1 and dets[0].raw_score == pytest.approx(0.9) and dets[0].score == pytest.approx(1.0)
    assert np.array_equal(dets[0].mask, m) and dets[0].point is None


def test_sidecar_round_trip_with_key():
    """A fake recogniser server speaking the wire format; the client sends the key, the threshold, the prompt."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    seen = {}

    def serve():
        conn, _ = srv.accept()
        with conn:
            hello = protocol.recv(conn)
            seen["key"] = hello["key"]
            protocol.send(conn, {"model": "x", "name": "x", "family": "dino", "kind": "open", "output": "box"})
            msg = protocol.recv(conn)
            seen["prompt"], seen["min"] = msg["prompt"], msg["min_score"]
            protocol.send(conn, {"scores": np.array([0.5], np.float32), "boxes": np.zeros((1, 4), np.float32),
                                 "points": np.full((1, 2), np.nan, np.float32), "model_ms": 1.0, "sam_ms": 0.0, "text": ""})
            protocol.recv(conn)
            protocol.send(conn, {})

    th = threading.Thread(target=serve)
    th.start()
    with SidecarRecogniser("x", "127.0.0.1", port, 0.4, 0.9, False, "k3y") as rec:
        dets = rec.detect(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), "plant")
    th.join(5)
    srv.close()
    assert seen == {"key": "k3y", "prompt": "potted plant", "min": 0.4}
    assert len(dets) == 1 and dets[0].score == pytest.approx(0.4 + 0.6 * 0.2) and dets[0].mask is None


def test_vlm_parsers():
    text = '```json\n[{"bbox_2d": [100, 200, 500, 1000], "label": "chair"}, {"bbox_2d": "bad"}]\n```'
    boxes = qwen_boxes(text, 320, 240)
    assert len(boxes) == 1 and np.allclose(boxes[0], [32, 48, 160, 240])
    assert qwen_boxes("[]", 320, 240) == [] and parse_json_list("There is no chair.") == []
    pts = molmo_points('<points coords="1 500 500;2 100 900">chair</points>', 320, 240)
    assert [p.tolist() for p in pts] == [[160.0, 120.0], [32.0, 216.0]]
    old = molmo_points('<point x="50.0" y="25.0" alt="chair">chair</point>', 320, 240)
    assert [p.tolist() for p in old] == [[160.0, 60.0]]
    assert molmo_points("There is no chair in the image.", 320, 240) == []
    assert RawDet(1.0).box is None


def test_unique_ids():
    assert pf.frame_id("00746-RTV2n6fXB2w", "drv", 12) == "p9f-00746-drv-0012"
    assert pf.item_id("p9f-00746-drv-0012", "plant") == "p9q-p9f-00746-drv-0012-plant"
    assert pf.item_id("p9f-00746-drv-0012", "office chair", ovon=True) == "p9o-p9f-00746-drv-0012-office_chair"
