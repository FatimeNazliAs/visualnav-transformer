"""Phase 9 photo-test frame set (block items 2, 3, 6): truth per frame, query items, halves, fingerprint.

A frame = one pin025 picture (RGB + depth + semantic, 320 x 240) at a pose in one of the 10 labelled Phase 1 homes:
- src "rnd": a random real-floor pose (seeded); src "drv": a pose along a Phase 8 test-split drive of that home,
  re-rendered, thinned so kept poses of a drive are >= 1 m or >= 30 deg apart.
Frame id p9f-<home 5 digits>-<src>-<n>; a query item = one frame + one asked category:
p9q-<frame id>-<category key> (6 categories) or p9o-<frame id>-<OVON name slug> (OVON). Ids must be unique.

Truth per frame (`frame_instances`): every semantic id with >= 1 pixel (0 = unlabelled, never an instance): raw
label, pixels, median depth of its pixels with depth > 0, pixels at depth (0, 4] m, 2-D box, whether the mask touches
the image edge, and the object's world box (objects.object_box). For an asked category (photo_labels.LabelTable):
- counted instance: raw label counted, >= 200 pixels and median depth <= 4 m;
- item status: "positive" (>= 1 counted instance), "free" (target-free: no pixel of any instance of the category),
  else "ignore_only" (the category is in view only below the counted rule).
Selection (`select_items`, seeded): per category, positives up to `max_pos`, then target-free items so that they
make up `free_share` of (positives + free), plus ignore-only items at their natural rate among non-positive frames
(at most `max_ignore_ratio` x free). Each class is drawn round-robin over the homes in a seeded order (so no home
dominates); afterwards no home may hold more than `home_cap` of all items (the largest home loses target-free items
first, logged).
"""

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.photo_labels import LabelTable, slug

MIN_PIXELS = 200
MAX_MEDIAN_DEPTH_M = 4.0
STATUSES = ("positive", "free", "ignore_only")


def frame_id(home: str, src: str, n: int) -> str:
    """p9f-00746-rnd-0012."""
    return f"p9f-{home[:5]}-{src}-{n:04d}"


def item_id(frame: str, key: str, ovon: bool = False) -> str:
    """p9q-<frame>-<category> or p9o-<frame>-<ovon slug>."""
    return f"{'p9o' if ovon else 'p9q'}-{frame}-{slug(key)}"


def check_unique(ids: Sequence[str], what: str) -> None:
    """Refuse a list with repeated ids (plan §0.4)."""
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(f"duplicate {what} ids, e.g. {dup}")


@dataclass
class Instance:
    """One semantic instance in view in one frame."""

    id: int
    name: str
    pixels: int
    median_depth_m: float
    near_pixels: int  # pixels with 0 < depth <= 4 m (Phase 8's seen rule counts these)
    box2d: List[int]  # x0, y0, x1, y1 (x1, y1 exclusive)
    touches_edge: bool
    world_centre: Optional[List[float]] = None  # Habitat x, y, z of the object's world box
    world_size: Optional[List[float]] = None

    def counted(self) -> bool:
        return self.pixels >= MIN_PIXELS and self.median_depth_m <= MAX_MEDIAN_DEPTH_M


def frame_instances(semantic: np.ndarray, depth: np.ndarray, names: Dict[int, str],
                    boxes: Dict[int, Tuple[np.ndarray, np.ndarray]], max_depth_m: float = 4.0) -> List[Instance]:
    """Every instance with >= 1 pixel in the frame (module docstring), sorted by id."""
    sem = semantic.astype(np.int64)
    h, w = sem.shape
    out = []
    for i in np.unique(sem):
        if i == 0:
            continue
        ys, xs = np.nonzero(sem == i)
        d = depth[ys, xs]
        valid = d > 0
        med = float(np.median(d[valid])) if valid.any() else math.inf
        box = boxes.get(int(i))
        out.append(Instance(int(i), names.get(int(i), ""), int(ys.size), round(med, 4),
                            int(((d > 0) & (d <= max_depth_m)).sum()),
                            [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                            bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1),
                            None if box is None else [round(float(v), 4) for v in box[0]],
                            None if box is None else [round(float(v), 4) for v in box[1]]))
    return out


def item_status(instances: Iterable[Instance], table: LabelTable) -> str:
    """positive / free / ignore_only for one asked category (module docstring)."""
    asked = [i for i in instances if table.role(i.name) == "counted"]
    if any(i.counted() for i in asked):
        return "positive"
    return "ignore_only" if asked else "free"


def thin_poses(xy: np.ndarray, yaw: np.ndarray, min_move_m: float = 1.0, min_turn_deg: float = 30.0) -> List[int]:
    """Frame indices of a drive kept so that each kept pose is >= min_move_m or >= min_turn_deg from the last kept."""
    keep = [0] if len(xy) else []
    for t in range(1, len(xy)):
        k = keep[-1]
        turn = abs(math.degrees(math.atan2(math.sin(yaw[t] - yaw[k]), math.cos(yaw[t] - yaw[k]))))
        if float(np.hypot(*(xy[t] - xy[k]))) >= min_move_m or turn >= min_turn_deg:
            keep.append(t)
    return keep


@dataclass
class SelectSpec:
    seed: int
    max_pos: int = 200
    free_share: float = 0.55
    max_ignore_ratio: float = 0.10
    home_cap: float = 0.20


def _round_robin(frames: List[Dict[str, Any]], n: int, rng: np.random.Generator, homes: Sequence[str]) -> List[Dict[str, Any]]:
    """n frames drawn one home at a time (seeded home order, seeded order inside each home)."""
    by_home: Dict[str, List[Dict[str, Any]]] = {h: [] for h in homes}
    for f in frames:
        by_home[f["home"]].append(f)
    queues = []
    for h in homes:
        fs = sorted(by_home[h], key=lambda f: f["frame_id"])
        order = rng.permutation(len(fs))
        queues.append([fs[k] for k in order])
    out: List[Dict[str, Any]] = []
    for row in itertools.zip_longest(*queues):
        for f in row:
            if f is not None and len(out) < n:
                out.append(f)
    return out


def select_items(frames: List[Dict[str, Any]], tables: Dict[str, LabelTable], spec: SelectSpec, ovon: bool = False,
                 log: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Query items for every table (module docstring). `frames`: dicts with frame_id, home, src and `status`
    {table key: status}. Returns items {item_id, frame_id, home, src, key, prompt, status}."""
    log = log if log is not None else []
    homes = sorted({f["home"] for f in frames})
    items: List[Dict[str, Any]] = []
    for k, key in enumerate(sorted(tables)):
        rng = np.random.default_rng([spec.seed, k])
        home_order = [homes[j] for j in rng.permutation(len(homes))]
        pools = {s: [f for f in frames if f["status"].get(key) == s] for s in STATUSES}
        pos = _round_robin(pools["positive"], min(spec.max_pos, len(pools["positive"])), rng, home_order)
        n_free = int(round(len(pos) * spec.free_share / (1.0 - spec.free_share)))
        free = _round_robin(pools["free"], min(n_free, len(pools["free"])), rng, home_order)
        nonpos = len(pools["free"]) + len(pools["ignore_only"])
        q = len(pools["ignore_only"]) / nonpos if nonpos else 0.0
        n_ign = min(int(round(len(free) * q / (1.0 - q))) if q < 1.0 else 0, int(spec.max_ignore_ratio * len(free)))
        ign = _round_robin(pools["ignore_only"], min(n_ign, len(pools["ignore_only"])), rng, home_order)
        if len(free) < n_free:
            log.append(f"{key}: only {len(free)} target-free frames for {n_free} wanted")
        for status, chosen in (("positive", pos), ("free", free), ("ignore_only", ign)):
            for f in chosen:
                items.append({"item_id": item_id(f["frame_id"], key, ovon), "frame_id": f["frame_id"], "home": f["home"],
                              "src": f["src"], "key": key, "prompt": tables[key].prompt, "status": status})
    items = cap_homes(items, spec.home_cap, log)
    check_unique([i["item_id"] for i in items], "item")
    return sorted(items, key=lambda i: i["item_id"])


def cap_homes(items: List[Dict[str, Any]], cap: float, log: List[str]) -> List[Dict[str, Any]]:
    """Drop target-free items (last drawn first) of the largest home until no home holds > cap of all items."""
    items = list(items)
    while items:
        counts: Dict[str, int] = {}
        for i in items:
            counts[i["home"]] = counts.get(i["home"], 0) + 1
        home, n = max(sorted(counts.items()), key=lambda kv: kv[1])
        if n <= cap * len(items):
            return items
        drop = [k for k, i in enumerate(items) if i["home"] == home and i["status"] == "free"]
        if not drop:
            log.append(f"home cap: {home} holds {n}/{len(items)} items and has no target-free item left to drop")
            return items
        log.append(f"home cap: dropped {items[drop[-1]]['item_id']} ({home} {n}/{len(items)})")
        items.pop(drop[-1])
    return items


def split_halves(homes: Sequence[str], pos_by_home: Dict[str, Dict[str, int]], seed: int,
                 min_share: float = 0.3) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Half A / half B, 5 + 5 homes (block item 3): the first split in a seeded order of all C(10, 5) splits (A always
    holds the first home, so each split appears once) in which every category has >= min_share of its positives in
    each half; if none, the split with the best worst-category share (logged)."""
    homes = sorted(homes)
    cats = sorted({c for v in pos_by_home.values() for c in v})
    splits = [list(a) for a in itertools.combinations(homes, len(homes) // 2) if homes[0] in a]
    order = np.random.default_rng(seed).permutation(len(splits))

    def worst(a: Sequence[str]) -> float:
        shares = []
        for c in cats:
            tot = sum(pos_by_home[h].get(c, 0) for h in homes)
            ina = sum(pos_by_home[h].get(c, 0) for h in a)
            shares.append(min(ina, tot - ina) / tot if tot else 0.0)
        return min(shares) if shares else 0.0

    for k in order:
        if worst(splits[k]) >= min_share:
            a = splits[k]
            return a, [h for h in homes if h not in a], {"rule": f"first seeded split with every category >= {min_share}",
                                                         "worst_share": round(worst(a), 4)}
    best = max(range(len(splits)), key=lambda k: (worst(splits[k]), -int(np.nonzero(order == k)[0][0])))
    a = splits[best]
    return a, [h for h in homes if h not in a], {"rule": "closest balance (no split met the minimum)",
                                                 "worst_share": round(worst(a), 4)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(folder: Path, files: Sequence[str]) -> Dict[str, Any]:
    """sha256 of each file (relative to folder) and one sha256 over the sorted "<sha256>  <name>" lines."""
    lines = sorted(f"{sha256_file(folder / f)}  {f}" for f in files)
    return {"files": len(lines), "sha256": hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()}


def write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float) + "\n")
    tmp.replace(path)


def instances_to_json(instances: Sequence[Instance]) -> List[Dict[str, Any]]:
    return [asdict(i) for i in instances]


def instances_from_json(rows: Sequence[Dict[str, Any]]) -> List[Instance]:
    return [Instance(**r) for r in rows]
