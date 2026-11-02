"""Phase 14a step 14.5 counts (report only): when YOLOE puts a box on the true target, does the map get a strong mark
on that object? From one run's stop evidence (stop_evidence.py) and the fused layer rebuilt by stop_rules.EvidenceReplay
(today's value = TargetFusion's mean).

- object sighting = a frame with a recogniser box on a true target (IoU >= 0.3 with that instance's true box in the
  same frame, video_events.iou); the object's cells = every cell any frame of the run put under that instance's pixels;
  its strength = the largest layer value over the object's cells in frames t .. t + 20; marked = strength >= 0.45;
- wrong sighting = a recogniser box whose IoU with every true box of its frame is < 0.3; its cells = the cells under
  its mask; strength = the same maximum over those cells in frames t .. t + 20.
Returns per run: the list of sightings (true and wrong) with their strengths.
"""

from typing import Any, Dict, List

import numpy as np

from mapmad_sim.stop_rules import EvidenceReplay, StopSpec
from mapmad_sim.video_events import iou

IOU_HIT = 0.3
HORIZON = 20


def mark_sightings(ev: Dict[str, np.ndarray], tau: float = 0.45, horizon: int = HORIZON) -> Dict[str, Any]:
    """{"true": [{t, instance, strength, marked}], "wrong": [{t, strength, marked}]} of one run."""
    det_t, true_t = ev["det_t"], ev["true_t"]
    obj_cells: Dict[int, set] = {}
    for k in range(len(true_t)):
        cells = ev["true_cells"][ev["true_cells_ptr"][k]:ev["true_cells_ptr"][k + 1]]
        obj_cells.setdefault(int(ev["true_id"][k]), set()).update(cells.tolist())
    true_hits: List[Dict[str, Any]] = []
    wrong: List[Dict[str, Any]] = []
    for k in range(len(det_t)):
        t = int(det_t[k])
        box = ev["det_box"][k]
        if not np.isfinite(box).all():
            continue
        trues = np.flatnonzero(true_t == t)
        best_iou, best_id = 0.0, None
        for j in trues.tolist():
            v = iou(box.tolist(), ev["true_box"][j].tolist())
            if v > best_iou:
                best_iou, best_id = v, int(ev["true_id"][j])
        if best_iou >= IOU_HIT:
            true_hits.append({"t": t, "instance": best_id, "det": k})
        else:
            cells = ev["det_cells"][ev["det_cells_ptr"][k]:ev["det_cells_ptr"][k + 1]]
            wrong.append({"t": t, "det": k, "cells": cells.astype(np.int64)})
    obj_all = set().union(*obj_cells.values()) if obj_cells else set()
    wrong_all = set(np.concatenate([w["cells"] for w in wrong]).tolist()) if wrong else set()
    interest = np.asarray(sorted(obj_all | wrong_all), np.int64)
    pos = {c: i for i, c in enumerate(interest.tolist())}
    rep = EvidenceReplay(ev, StopSpec())
    hist = np.zeros((rep.frames, interest.size), np.float32)
    for t in range(rep.frames):
        rep.advance()
        if interest.size:
            hist[t] = rep.value(interest)

    def strength(cells, t: int) -> float:
        idx = [pos[c] for c in cells if c in pos]
        if not idx:
            return 0.0
        return float(hist[t:min(t + horizon + 1, rep.frames)][:, idx].max())

    out_true = []
    for h in true_hits:
        s = strength(sorted(obj_cells.get(h["instance"], ())), h["t"])
        out_true.append({"t": h["t"], "instance": h["instance"], "strength": round(s, 4), "marked": s >= tau - 1e-12})
    out_wrong = []
    for w in wrong:
        s = strength(w["cells"].tolist(), w["t"])
        out_wrong.append({"t": w["t"], "strength": round(s, 4), "marked": s >= tau - 1e-12})
    return {"true": out_true, "wrong": out_wrong}
