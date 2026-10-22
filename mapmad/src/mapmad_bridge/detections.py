"""Phase 9 detection format, shared by the recogniser sidecar (naz_mapmad_recog, Python 3.11) and the Habitat side
(naz_mapmad_habitat, Python 3.9). numpy only.

Reply arrays (one detect call, N detections):
- "scores" (N,) f32
- "boxes"  (N, 4) f32 x0, y0, x1, y1; a NaN row = no box (point detections)
- "points" (N, 2) f32 x, y; a NaN row = no point (box / mask detections)
- "masks"  (N, ceil(H*W/8)) uint8, each (H, W) bool mask flattened and bit-packed; present only when every
  detection has a mask

Part file (raw outputs of up to 250 items): part_<k>.npz with the reply arrays of all items concatenated, plus
"item_id" (items,), "det_item" (N,) the item row of each detection, "has_mask" (N,) (rows without a mask are zero),
"rtt_ms" / "model_ms" / "sam_ms" (items,); part_<k>.json = item id -> VLM answer (only non-empty answers).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FRAME_HW = (240, 320)  # Phase 9 frame set (pin025 camera): height, width


def pack(scores: Sequence[float], boxes: Sequence[Optional[Any]], points: Sequence[Optional[Any]],
         masks: Sequence[Optional[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Per-detection score, box (or None), point (or None) and (H, W) bool mask (or None) -> the reply arrays."""
    n = len(scores)
    out = {"scores": np.asarray(scores, np.float32).reshape(n), "boxes": np.full((n, 4), np.nan, np.float32),
           "points": np.full((n, 2), np.nan, np.float32)}
    for k in range(n):
        if boxes[k] is not None:
            out["boxes"][k] = boxes[k]
        if points[k] is not None:
            out["points"][k] = points[k]
    if n and all(m is not None for m in masks):
        out["masks"] = np.stack([np.packbits(np.asarray(m).reshape(-1)) for m in masks])
    return out


def unpack_mask(bits: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """One bit-packed mask row -> (H, W) bool."""
    h, w = hw
    return np.unpackbits(bits)[: h * w].astype(bool).reshape(h, w)


def present(row: np.ndarray) -> Optional[np.ndarray]:
    """A box / point row, or None when it is a NaN row (absent)."""
    return None if np.isnan(row).any() else row


def save_part(folder: Path, k: int, rows: List[Dict[str, Any]], hw: Tuple[int, int]) -> None:
    """Rows (reply arrays + "item_id", "rtt_ms", "model_ms", "sam_ms", "text") -> part_<k>.npz / .json, atomically."""
    width = (hw[0] * hw[1] + 7) // 8
    det_item, scores, boxes, points, masks, has_mask = [], [], [], [], [], []
    for j, r in enumerate(rows):
        n = len(r["scores"])
        det_item += [j] * n
        scores.append(np.asarray(r["scores"], np.float32))
        boxes.append(np.asarray(r["boxes"], np.float32).reshape(-1, 4))
        points.append(np.asarray(r["points"], np.float32).reshape(-1, 2))
        m = r.get("masks")
        masks.append(np.asarray(m, np.uint8) if m is not None else np.zeros((n, width), np.uint8))
        has_mask += [m is not None] * n
    arrays = {"item_id": np.asarray([r["item_id"] for r in rows]), "det_item": np.asarray(det_item, np.int32),
              "scores": np.concatenate(scores), "boxes": np.concatenate(boxes), "points": np.concatenate(points),
              "masks": np.concatenate(masks), "has_mask": np.asarray(has_mask, bool),
              "rtt_ms": np.asarray([r["rtt_ms"] for r in rows], np.float32),
              "model_ms": np.asarray([r["model_ms"] for r in rows], np.float32),
              "sam_ms": np.asarray([r["sam_ms"] for r in rows], np.float32)}
    tmp = folder / f"part_{k:03d}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, folder / f"part_{k:03d}.npz")
    texts = folder / f"part_{k:03d}.json"
    tmp_json = texts.with_suffix(".json.tmp")
    answers = {r["item_id"]: r["text"] for r in rows if r["text"]}
    tmp_json.write_text(json.dumps(answers, indent=1, default=float) + "\n")
    tmp_json.replace(texts)


def load_parts(folder: Path, hw: Tuple[int, int]) -> Dict[str, Dict[str, Any]]:
    """Item id -> {"scores", "boxes", "points" (float64), "masks" (list of (H, W) bool or None)} from part_*.npz."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(folder.glob("part_*.npz")):
        if ".tmp" in path.name:
            continue
        z = np.load(path)
        for j, item in enumerate(z["item_id"].tolist()):
            sel = np.nonzero(z["det_item"] == j)[0]
            masks = [unpack_mask(z["masks"][k], hw) if z["has_mask"][k] else None for k in sel]
            out[item] = {"scores": z["scores"][sel].astype(np.float64), "boxes": z["boxes"][sel].astype(np.float64),
                         "points": z["points"][sel].astype(np.float64), "masks": masks}
    return out
