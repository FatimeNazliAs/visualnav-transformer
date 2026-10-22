"""Detection format round trip: pack (server) -> save_part (photo test) -> load_parts (scorer), and pack -> unpack
(plug-in) give back the same scores, boxes, points and masks."""

import numpy as np

from mapmad_bridge.detections import load_parts, pack, present, save_part, unpack_mask

HW = (6, 5)  # 30 pixels: not a multiple of 8, so the packed rows carry padding bits


def masks(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.random(HW) > 0.5 for _ in range(n)]


def row(item, reply, text=""):
    return {**reply, "item_id": item, "rtt_ms": 1.0, "model_ms": 0.5, "sam_ms": 0.0, "text": text}


def test_pack_marks_absent_boxes_and_points_with_nan_rows():
    r = pack([0.9, 0.4], [[1, 2, 3, 4], None], [None, [2.5, 3.5]], [None, None])
    assert present(r["boxes"][0]).tolist() == [1, 2, 3, 4] and present(r["boxes"][1]) is None
    assert present(r["points"][0]) is None and present(r["points"][1]).tolist() == [2.5, 3.5]
    assert "masks" not in r  # masks only when every detection has one


def test_unpack_mask_drops_padding_bits():
    m = masks(1)[0]
    assert np.array_equal(unpack_mask(pack([0.5], [[0, 0, 1, 1]], [None], [m])["masks"][0], HW), m)


def test_part_file_round_trip(tmp_path):
    ma = masks(3)
    rows = [row("a", pack([0.9, 0.3], [[0, 0, 2, 2], [1, 1, 4, 5]], [None, None], ma[:2])),
            row("b", pack([], [], [], [])),  # no detections
            row("c", pack([1.0], [None], [[2.0, 3.0]], [None]), text="<points coords=\"1 400 500\">"),
            row("d", pack([0.7], [[0, 1, 3, 4]], [None], [ma[2]]))]
    save_part(tmp_path, 0, rows, HW)
    got = load_parts(tmp_path, HW)
    assert sorted(got) == ["a", "b", "c", "d"]
    assert got["a"]["scores"].tolist() == np.float32([0.9, 0.3]).astype(np.float64).tolist()
    assert all(np.array_equal(x, y) for x, y in zip(got["a"]["masks"], ma[:2]))
    assert len(got["b"]["scores"]) == 0
    assert got["c"]["masks"] == [None] and present(got["c"]["boxes"][0]) is None
    assert got["c"]["points"][0].tolist() == [2.0, 3.0]
    assert np.array_equal(got["d"]["masks"][0], ma[2])
    assert (tmp_path / "part_000.json").read_text().count("points coords") == 1
