"""Phase 9 photo-test scoring (block items 1, 4, 6): pure numpy, from the saved raw outputs; the rules are in
p9_rules.md. Every detection gets two labels once (they do not depend on the threshold):

Pixel level (AP50, precision, recall, false alarms):
- truth of an item = its counted instances (photo_frames.Instance.counted, raw label counted for the asked table);
  ignore instances = instances of the asked category below the counted rule + near-synonym instances;
- detections in score order (best first) are matched to a not yet matched counted instance: box models (and mask
  models) by box IoU >= 0.5 (the best IoU wins), point models by the point lying on the instance's pixels -> TP;
- otherwise IGN if the box has IoU >= 0.5 with an ignore instance's box, or >= 50% of the detection's mask (box
  region without a mask) lies on unlabelled pixels (id 0) or ignore instances (a point: on such a pixel);
- otherwise FP. Greedy matching in score order means the labels of the kept detections at any threshold are the
  labels computed here.
Map level (the 1 m rule):
- a detection's point = the median (per axis) of the world points of its mask pixels (a point model: its pixel)
  with depth (0, 4] m and height >= floor - 0.10 m (the pixels the map would draw); none -> "off" (not drawn;
  neither hit nor miss);
- hit if the point is within 1 m (horizontal, the map plane) of a counted instance's world box; otherwise IGN if
  within 1 m of an ignore instance's box; otherwise miss;
- H = hits / (hits + misses) over kept detections; R = share of (item, counted instance) pairs with >= 1 kept hit;
  F1 = 2 H R / (H + R).
Thresholds: one per model, the grid value 0.01 ... 0.99 with the highest map F1 on half A (6 categories; ties -> the
lowest threshold); per-category thresholds are reported only. VLMs: score 1, one operating point, AP50 n/a.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mapmad_sim.depth_map import camera_points
from mapmad_sim.photo_frames import Instance, instances_from_json
from mapmad_sim.photo_labels import LabelTable

GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
IOU_MIN = 0.5
IGNORE_SHARE = 0.5
HIT_M = 1.0
MAX_DEPTH_M = 4.0
FLOOR_MIN_M = -0.10
DIST_BINS = ((0.0, 2.0), (2.0, 4.0))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def box_distance_xz(p: np.ndarray, centre: Sequence[float], size: Sequence[float]) -> float:
    """Horizontal distance (Habitat x, z) from a world point to an axis-aligned world box (0 inside)."""
    dx = max(abs(p[0] - centre[0]) - size[0] / 2.0, 0.0)
    dz = max(abs(p[2] - centre[2]) - size[2] / 2.0, 0.0)
    return math.hypot(dx, dz)


@dataclass
class FrameData:
    """What scoring needs of one frame: truth instances, semantic ids, and the world point of every drawable pixel."""

    instances: List[Instance]
    semantic: np.ndarray  # (H, W) int
    world: np.ndarray  # (H * W, 3) world point per pixel (NaN = not drawable)

    @classmethod
    def load(cls, frame: Dict[str, Any], img_dir: Path) -> "FrameData":
        fid = frame["frame_id"]
        sem = cv2.imread(str(img_dir / f"{fid}_sem.png"), cv2.IMREAD_UNCHANGED).astype(np.int64)
        depth = cv2.imread(str(img_dir / f"{fid}_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        pts, keep = camera_points(depth, frame["hfov_deg"], MAX_DEPTH_M)
        tf = np.asarray(frame["camera_tf"], np.float64)
        w = pts @ tf[:3, :3].T + tf[:3, 3]
        ok = w[:, 1] - frame["floor_y"] >= FLOOR_MIN_M
        world = np.full((depth.size, 3), np.nan)
        world[keep[ok]] = w[ok]
        return cls(instances_from_json(frame["instances"]), sem, world)


@dataclass
class ItemTruth:
    counted: List[Instance]
    ignore: List[Instance]


def item_truth(instances: Sequence[Instance], table: LabelTable) -> ItemTruth:
    counted, ignore = [], []
    for i in instances:
        role = table.role(i.name)
        if role == "counted" and i.counted():
            counted.append(i)
        elif role in ("counted", "ignore"):
            ignore.append(i)
    return ItemTruth(counted, ignore)


@dataclass
class Det:
    """One raw detection with its two labels."""

    item: str
    score: float
    pix: str  # TP / FP / IGN
    map: str  # hit / miss / IGN / off
    hits: Tuple[int, ...] = ()  # counted instance ids within 1 m
    matched: int = -1  # pixel-level matched instance id
    found: str = ""  # raw label under the detection (confusion pairs)
    dist_m: float = float("nan")  # horizontal distance of its map point from the camera


def detection_pixels(mask: Optional[np.ndarray], box: Optional[np.ndarray], point: Optional[np.ndarray],
                     shape: Tuple[int, int]) -> np.ndarray:
    """Flat pixel indices a detection covers: its mask, else its box region, else its point's pixel."""
    h, w = shape
    if mask is not None:
        return np.flatnonzero(mask.reshape(-1))
    if box is not None and not np.isnan(box).any():
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        ys, xs = np.mgrid[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)]
        return (ys * w + xs).ravel()
    if point is not None and not np.isnan(point).any():
        x, y = int(min(max(round(point[0]), 0), w - 1)), int(min(max(round(point[1]), 0), h - 1))
        return np.array([y * w + x])
    return np.zeros(0, np.int64)


def label_item(item_id: str, frame: FrameData, truth: ItemTruth, scores: np.ndarray, boxes: np.ndarray,
               points: np.ndarray, masks: List[Optional[np.ndarray]], cam: np.ndarray) -> List[Det]:
    """Both labels of every detection of one item (module docstring)."""
    h, w = frame.semantic.shape
    sem = frame.semantic.reshape(-1)
    ign_ids = {i.id for i in truth.ignore}
    by_id = {i.id: i for i in truth.counted}
    matched: set = set()
    out = []
    for k in np.argsort(-scores, kind="stable"):
        box = boxes[k] if not np.isnan(boxes[k]).any() else None
        pt = points[k] if not np.isnan(points[k]).any() else None
        px = detection_pixels(masks[k], box, pt, (h, w))
        if box is None and masks[k] is not None and px.size:
            ys, xs = np.divmod(px, w)
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)
        d = Det(item_id, float(scores[k]), "FP", "miss")
        # pixel level
        if pt is not None and box is None:
            sid = int(sem[px[0]]) if px.size else 0
            if sid in by_id and sid not in matched:
                d.pix, d.matched = "TP", sid
            elif sid == 0 or sid in ign_ids:
                d.pix = "IGN"
        else:
            cands = [(iou(box, i.box2d), i.id) for i in truth.counted if i.id not in matched] if box is not None else []
            best = max(cands) if cands else (0.0, -1)
            if best[0] >= IOU_MIN:
                d.pix, d.matched = "TP", best[1]
            elif box is not None and any(iou(box, i.box2d) >= IOU_MIN for i in truth.ignore):
                d.pix = "IGN"
            elif px.size and np.isin(sem[px], list(ign_ids | {0})).mean() >= IGNORE_SHARE:
                d.pix = "IGN"
        if d.matched >= 0:
            matched.add(d.matched)
        # map level
        pts = frame.world[px] if px.size else np.zeros((0, 3))
        pts = pts[~np.isnan(pts[:, 0])]
        if not len(pts):
            d.map = "off"
        else:
            p = np.median(pts, axis=0)
            d.dist_m = float(math.hypot(p[0] - cam[0], p[2] - cam[2]))
            d.hits = tuple(i.id for i in truth.counted
                           if i.world_centre is not None and box_distance_xz(p, i.world_centre, i.world_size) <= HIT_M)
            if d.hits:
                d.map = "hit"
            elif any(i.world_centre is not None and box_distance_xz(p, i.world_centre, i.world_size) <= HIT_M
                     for i in truth.ignore):
                d.map = "IGN"
        ids = sem[px][sem[px] > 0] if px.size else np.zeros(0, np.int64)
        if ids.size:
            top = int(np.bincount(ids).argmax())
            d.found = next((i.name for i in frame.instances if i.id == top), "")
        out.append(d)
    return out


# --- metrics at a threshold ---------------------------------------------------------------------------------------

@dataclass
class ItemInfo:
    item_id: str
    key: str
    status: str
    half: str
    src: str
    counted: List[Instance] = field(default_factory=list)


def ap50(dets: Sequence[Det], npos: int) -> float:
    """VOC all-point AP from the pixel labels (IGN dropped)."""
    ds = sorted((d for d in dets if d.pix != "IGN"), key=lambda d: -d.score)
    if npos == 0:
        return float("nan")
    if not ds:
        return 0.0
    tp = np.cumsum([d.pix == "TP" for d in ds])
    fp = np.cumsum([d.pix == "FP" for d in ds])
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def f1(h: float, r: float) -> float:
    return 2 * h * r / (h + r) if (h + r) > 0 else 0.0


def metrics(dets: Sequence[Det], items: Sequence[ItemInfo], t: float, with_ap: bool = True) -> Dict[str, Any]:
    """All block item 4 numbers for a set of items at threshold t."""
    ids = {i.item_id for i in items}
    ds = [d for d in dets if d.item in ids]
    kept = [d for d in ds if d.score >= t]
    npos = sum(len(i.counted) for i in items)
    tp = sum(d.pix == "TP" for d in kept)
    fp = sum(d.pix == "FP" for d in kept)
    free = {i.item_id for i in items if i.status == "free"}
    hits = sum(d.map == "hit" for d in kept)
    miss = sum(d.map == "miss" for d in kept)
    covered = {(d.item, h) for d in kept if d.map == "hit" for h in d.hits}
    pairs = {(i.item_id, c.id) for i in items for c in i.counted}
    h = hits / (hits + miss) if hits + miss else float("nan")
    r = len(covered & pairs) / len(pairs) if pairs else float("nan")
    return {"items": len(items), "counted": npos, "kept": len(kept),
            "ap50": ap50(ds, npos) if with_ap else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / npos if npos else float("nan"),
            "fa_per_free": sum(d.pix == "FP" and d.item in free for d in kept) / len(free) if free else float("nan"),
            "free_items": len(free), "map_hits": hits, "map_misses": miss, "off_map": sum(d.map == "off" for d in kept),
            "H": h, "R": r, "F1": f1(0.0 if math.isnan(h) else h, 0.0 if math.isnan(r) else r)}


def tune(dets: Sequence[Det], items: Sequence[ItemInfo], vlm: bool = False) -> Tuple[float, float]:
    """(threshold, half-A F1) by the grid rule (ties -> lowest threshold). VLMs: (0.0, F1 at their one point)."""
    if vlm:
        return 0.0, metrics(dets, items, 0.0, with_ap=False)["F1"]
    best_t, best = float(GRID[0]), -1.0
    for t in GRID:
        v = metrics(dets, items, float(t), with_ap=False)["F1"]
        if v > best + 1e-12:
            best_t, best = float(t), v
    return best_t, best


def recall_breakdown(dets: Sequence[Det], items: Sequence[ItemInfo], t: float) -> Dict[str, Any]:
    """Pixel and map recall by distance bin (median depth) and by whether the instance touches the image edge."""
    kept = [d for d in dets if d.score >= t]
    pix = {(d.item, d.matched) for d in kept if d.pix == "TP"}
    mp = {(d.item, h) for d in kept if d.map == "hit" for h in d.hits}
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for i in items:
        for c in i.counted:
            for lo, hi in DIST_BINS:
                if lo <= c.median_depth_m < hi or (hi == DIST_BINS[-1][1] and c.median_depth_m == hi):
                    groups.setdefault(f"depth_{lo:g}-{hi:g}m", []).append((i.item_id, c.id))
            groups.setdefault("cut_by_edge" if c.touches_edge else "whole_in_view", []).append((i.item_id, c.id))
    return {g: {"n": len(v), "recall": sum(p in pix for p in v) / len(v), "map_recall": sum(p in mp for p in v) / len(v)}
            for g, v in sorted(groups.items())}


def confusions(dets: Sequence[Det], items: Sequence[ItemInfo], t: float, category_of: Dict[str, str],
               top: int = 10) -> List[Dict[str, Any]]:
    """Most common (asked -> found) pairs among kept pixel false alarms; found = the mapped category of the raw label
    under the detection, else the raw label itself."""
    key = {i.item_id: i.key for i in items}
    counts: Dict[Tuple[str, str], int] = {}
    for d in dets:
        if d.item in key and d.score >= t and d.pix == "FP":
            found = category_of.get(d.found, d.found or "unlabelled")
            counts[(key[d.item], found)] = counts.get((key[d.item], found), 0) + 1
    return [{"asked": a, "found": f, "n": n} for (a, f), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]]


# --- the choice rule (block item 1) -------------------------------------------------------------------------------

def choose(models: Dict[str, Dict[str, Any]], max_latency_s: float = 0.25, tie: float = 0.01) -> Dict[str, Any]:
    """R1, R2 from {model: {"F1", "median_s", "kind", "family", "ran"}} (block item 1, mechanically):
    open-vocabulary pool (kind open or vlm; RF-DETR = fixed, never); qualifies if median latency <= 0.25 s;
    R1 = best qualifying by F1, a tie (F1 within 0.01 of the best) -> the fastest of the tied; a VLM may take a slot
    only if it has the best F1 of that slot's pool; R2 = the same over the qualifying models of another family.
    Fallback: fewer than 2 -> the fastest remaining open-vocabulary models (another family if possible), flagged."""
    pool = {m: v for m, v in models.items() if v.get("ran") and v["kind"] in ("open", "vlm")}
    log: List[str] = []

    def pick(cands: Dict[str, Dict[str, Any]], slot: str) -> Optional[str]:
        if not cands:
            return None
        best = max(v["F1"] for v in cands.values())
        tied = [m for m, v in cands.items() if v["F1"] >= best - tie]
        m = min(sorted(tied), key=lambda k: cands[k]["median_s"])
        if cands[m]["kind"] == "vlm" and cands[m]["F1"] < best:
            log.append(f"{slot}: VLM {m} tied but not best on F1 -> not eligible")
            rest = {k: v for k, v in cands.items() if k != m}
            return pick(rest, slot)
        log.append(f"{slot}: best F1 {best:.4f}; tied within {tie}: {sorted(tied)}; fastest = {m}")
        return m

    qual = {m: v for m, v in pool.items() if v["median_s"] <= max_latency_s}
    r1 = pick(qual, "R1")
    r2 = pick({m: v for m, v in qual.items() if r1 is not None and v["family"] != models[r1]["family"]}, "R2")
    flags = []
    chosen = [m for m in (r1, r2) if m is not None]
    if len(chosen) < 2:
        rest = sorted((m for m in pool if m not in chosen), key=lambda m: pool[m]["median_s"])
        other = [m for m in rest if not chosen or pool[m]["family"] != pool[chosen[0]]["family"]]
        for m in (other or rest):
            if len(chosen) == 2:
                break
            chosen.append(m)
            flags.append(f"{m}: fallback (fastest remaining open-vocabulary model{'' if other else ', same family'})")
    return {"R1": chosen[0] if chosen else None, "R2": chosen[1] if len(chosen) > 1 else None,
            "qualifying": sorted(qual), "fallback_flags": flags, "log": log}


def load_raw(folder: Path, hw: Tuple[int, int]) -> Dict[str, Dict[str, Any]]:
    """Item id -> {"scores", "boxes", "points", "masks" (list of (H, W) bool or None)} from part_*.npz."""
    h, w = hw
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(folder.glob("part_*.npz")):
        if ".tmp" in path.name:
            continue
        z = np.load(path)
        for j, item in enumerate(z["item_id"].tolist()):
            sel = np.nonzero(z["det_item"] == j)[0]
            masks = [np.unpackbits(z["masks"][k])[: h * w].astype(bool).reshape(h, w) if z["has_mask"][k] else None
                     for k in sel]
            out[item] = {"scores": z["scores"][sel].astype(np.float64), "boxes": z["boxes"][sel].astype(np.float64),
                         "points": z["points"][sel].astype(np.float64), "masks": masks}
    return out


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    tmp.replace(path)
