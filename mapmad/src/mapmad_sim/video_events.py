"""Phase 13 checkable videos (mapmad/scripts/p13_videos.py): the pure parts, numpy only (no simulator).

- True target boxes of one camera frame (`true_boxes`): for every instance id of the episode's category that has at
  least one pixel in the frame's semantic picture, the bounding box of its pixels in pixel-edge coordinates
  [x0, y0, x1 + 1, y1 + 1] (the box covers the pixels fully, like the recogniser's boxes [x0, y0, x1, y1] in px).
- First YOLOE hit on the true object (`first_hit`): the first step whose logged recogniser boxes (the dets file) include
  one with intersection-over-union >= IOU_HIT (0.3) with any true target box of the same frame (`iou`: area of the
  intersection / area of the union; boxes as [x0, y0, x1, y1]).
- Timeline events (`timeline_events`) from p12_score.score_log's numbers of the run: first true sight (`in_view` first
  True, simulator truth: an instance >= 200 px at <= 4 m), first YOLOE hit, oracle success (first step in the zone =
  within 0.2 m walking distance of a view point), rule stop at tau (first step >= 1 with rule_value >= tau; "right
  object" in the zone, else "wrong object" (triggering cell > 1.0 m from every target box), "near miss" (walking
  distance <= 1.2 m) or "outside").
- Box check (`compare_boxes`): the boxes a video drew on a frame = the logged boxes of that step (exact list, order,
  values; label = "<prompt> <raw score, 2 digits>"; thick = map value >= tau).
- Same-run check (`compare_runs`): two logs of one (arm, episode, seed) drove the same poses (start pose of the
  header, then position + yaw of every step, exact), over their common prefix of steps.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

IOU_HIT = 0.3
Box = Sequence[float]


@dataclass(frozen=True)
class Event:
    """One marker of the timeline: step, kind (sight | hit | oracle | rule), short text, and for the rule stop how it
    ended (right object | wrong object | near miss | outside)."""

    step: int
    kind: str
    text: str
    how: Optional[str] = None


# --- boxes --------------------------------------------------------------------------------------------------------
def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes [x0, y0, x1, y1] (0 when either has no area)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def true_boxes(semantic: np.ndarray, instance_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """[{"id", "box": [x0, y0, x1 + 1, y1 + 1], "px"}] of every listed instance seen in the semantic picture, by id."""
    out = []
    present = set(np.unique(semantic).tolist())
    for i in sorted(int(x) for x in instance_ids):
        if i not in present:
            continue
        ys, xs = np.nonzero(semantic == i)
        out.append({"id": i, "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    "px": int(xs.size)})
    return out


def hit(dets: Sequence[Dict[str, Any]], trues: Sequence[Dict[str, Any]], thr: float = IOU_HIT) -> bool:
    """A logged recogniser box overlaps a true target box with IoU >= thr."""
    return any(d.get("box") is not None and iou(d["box"], t["box"]) >= thr for d in dets for t in trues)


def first_hit(dets_by_step: Dict[int, List[Dict[str, Any]]], trues_by_step: Dict[int, List[Dict[str, Any]]],
              thr: float = IOU_HIT) -> Optional[int]:
    """The first step with a hit (module docstring), or None."""
    for t in sorted(dets_by_step):
        if hit(dets_by_step[t], trues_by_step.get(t, []), thr):
            return t
    return None


# --- the dets file ------------------------------------------------------------------------------------------------
def read_dets(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """step -> logged detections of that frame; the steps must be 0, 1, 2, ... without gaps."""
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    out = {int(r["step"]): r["dets"] for r in rows}
    if sorted(out) != list(range(len(rows))):
        raise ValueError(f"{path}: steps are not 0..{len(rows) - 1}")
    return out


def box_label(prompt: str, score: float) -> str:
    """The text drawn on a recogniser box: prompt word + raw score (2 digits), e.g. "couch 0.62"."""
    return f"{prompt} {score:.2f}"


def expected_boxes(dets: Sequence[Dict[str, Any]], prompt: str, tau: float) -> List[Dict[str, Any]]:
    """What a frame must show for these logged detections: one drawn box per detection with a box, in log order."""
    return [{"box": list(d["box"]), "label": box_label(prompt, d["score"]), "thick": d["value"] >= tau - 1e-12}
            for d in dets if d.get("box") is not None]


def compare_boxes(drawn: Sequence[Dict[str, Any]], logged: Sequence[Dict[str, Any]], prompt: str,
                  tau: float) -> Optional[str]:
    """None when the drawn boxes equal the logged ones exactly (module docstring), else what differs."""
    want = expected_boxes(logged, prompt, tau)
    got = [{"box": list(d["box"]), "label": d["label"], "thick": bool(d["thick"])} for d in drawn]
    if len(got) != len(want):
        return f"{len(got)} boxes drawn, {len(want)} logged"
    for k, (g, w) in enumerate(zip(got, want)):
        if g != w:
            return f"box {k}: drawn {g} != logged {w}"
    return None


# --- events -------------------------------------------------------------------------------------------------------
def rule_how(r: Dict[str, Any]) -> str:
    """How a rule stop ended (p12_score order: success first)."""
    if r["success"]:
        return "right object"
    if r["wrong_object"]:
        return "wrong object"
    return "near miss" if r["near_miss"] else "outside"


def timeline_events(score: Dict[str, Any], tau: float, hit_step: Optional[int]) -> List[Event]:
    """The run's timeline markers in step order (module docstring); score = p12_score.score_log(log, taus=[tau])."""
    ev = []
    if score.get("first_sight") is not None:
        ev.append(Event(int(score["first_sight"]), "sight", f"1st true sight {score['first_sight']}"))
    if hit_step is not None:
        ev.append(Event(int(hit_step), "hit", f"1st YOLOE hit {hit_step}"))
    if score.get("oracle_step") is not None:
        ev.append(Event(int(score["oracle_step"]), "oracle", f"at view spot {score['oracle_step']}"))
    r = (score.get("rule") or {}).get(f"{tau:.2f}") or {}
    if r.get("step") is not None:
        how = rule_how(r)
        ev.append(Event(int(r["step"]), "rule", f"rule stop {r['step']}: {how}", how))
    order = {"sight": 0, "hit": 1, "oracle": 2, "rule": 3}
    return sorted(ev, key=lambda e: (e.step, order[e.kind]))


# --- poses --------------------------------------------------------------------------------------------------------
def poses(log: List[Dict[str, Any]]) -> List[Tuple[List[float], float]]:
    """(position, yaw) of the start (header) and of every step, in order."""
    head = log[0]
    return [(list(head["start"]["position"]), float(head["start"]["yaw"]))] + \
        [(list(s["position"]), float(s["yaw"])) for s in log if s.get("kind") == "step"]


def compare_runs(new: List[Dict[str, Any]], old: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Same poses over the common prefix (start + steps 1..n, n = the shorter run's steps)? first differing step
    (0 = the start) or None; the lengths are reported, a shorter run alone is not a difference."""
    a, b = poses(new), poses(old)
    n = min(len(a), len(b))
    first = next((t for t in range(n) if a[t] != b[t]), None)
    same = new[0]["episode_id"] == old[0]["episode_id"]
    return {"verdict": "reproducible" if (first is None and same) else "not reproducible", "first_differing_step": first,
            "prefix_steps": n - 1, "new_steps": len(a) - 1, "old_steps": len(b) - 1,
            "full_length": len(a) == len(b), "same_episode": same,
            "same_noise_seed": new[0].get("noise_seed") == old[0].get("noise_seed")}
