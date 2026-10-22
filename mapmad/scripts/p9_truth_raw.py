"""Simulator truth through the photo-test pipeline (the upper limit, block item 1), in naz_mapmad_habitat_p9, CPU only:

    python mapmad/scripts/p9_truth_raw.py

Writes photo_test/truth/raw/part_*.npz in the same format as a model (p9_photo_test.py), score 1.0 per detection:
- 6 categories: Phase 8's truth rule exactly as the generator and the plug-in's TruthRecogniser use it: the home's
  ObjectNav goal ids of the category, an instance with >= 200 pixels at depth (0, 4] m; mask = those pixels;
- OVON names: the same pixel rule on the instances whose raw label is counted for the name (by name).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapmad_bridge import detections  # noqa: E402
from mapmad_sim import config, objectnav, photo_labels as pl  # noqa: E402
from mapmad_sim.photo_frames import instances_from_json, write_json_atomic  # noqa: E402
from p9_photo_test import CHUNK  # noqa: E402

MIN_PX, MAX_D = 200, 4.0


def main() -> None:
    root = config.paths()["outputs"] / "p9_recognisers"
    fr = root / "frames"
    frames = {f["frame_id"]: f for f in json.loads((fr / "frames.json").read_text())}
    items = sorted(json.loads((fr / "items_six.json").read_text()) + json.loads((fr / "items_ovon.json").read_text()),
                   key=lambda i: i["item_id"])
    ovon = json.loads((fr / "ovon_mapping.json").read_text())["mapped"]
    goal_ids: Dict[str, Dict[str, List[int]]] = {}
    out = root / "photo_test" / "truth" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, (len(items) + CHUNK - 1) // CHUNK):
        rows = []
        for it in items[k * CHUNK:(k + 1) * CHUNK]:
            f = frames[it["frame_id"]]
            home = f["home"]
            if home not in goal_ids:
                goal_ids[home] = {}
                for g in objectnav.home_goals("train", home):
                    goal_ids[home].setdefault(g.category, []).append(g.object_id)
            inst = instances_from_json(f["instances"])
            if it["item_id"].startswith("p9q-"):
                ids = set(goal_ids[home].get(it["key"], []))
            else:
                counted = set(ovon[it["key"]]["counted"])
                ids = {i.id for i in inst if pl.normalise(i.name) in counted}
            sem = cv2.imread(str(fr / "img" / f"{it['frame_id']}_sem.png"), cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(str(fr / "img" / f"{it['frame_id']}_depth.png"), cv2.IMREAD_UNCHANGED) / 1000.0
            near = (depth > 0) & (depth <= MAX_D)
            masks, boxes = [], []
            for i in sorted(ids):
                m = (sem == i) & near
                if m.sum() >= MIN_PX:
                    ys, xs = np.nonzero(m)
                    masks.append(m)
                    boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            n = len(masks)
            rows.append({**detections.pack(np.ones(n), boxes, [None] * n, masks), "item_id": it["item_id"],
                         "rtt_ms": 0.0, "model_ms": 0.0, "sam_ms": 0.0, "text": ""})
        detections.save_part(out, k, rows, detections.FRAME_HW)
    write_json_atomic(out / "stats.json", {"items": len(items), "errors": 0, "peak_vram_mb": 0.0,
                                           "info": {"model": "truth", "kind": "truth"}})
    print(f"truth: {len(items)} items")


if __name__ == "__main__":
    main()
