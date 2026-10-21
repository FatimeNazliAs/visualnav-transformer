"""MapMaD v2 offline harness (Phase 10 block items 5-6): frozen sample lists and the comparison of one `score` output.

Lists (offline_eval make-lists-v2; JSON with the sha256 of the samples, never overwritten):
- `search`: 5,000 samples spread evenly over the held-out SEARCH drives (test split, collision drives excluded),
  ~25 per drive (offline_list.make_list rule: frames t - 15 >= 0 and t + 40 <= last frame, seed LIST_SEED);
- `explore`: 2,000 samples over the held-out EXPLORATION drives (report-only arm "exploration drives maps vs hidden").
Each sample: drive, t, id, type (search / explore), category (bed ... tv_monitor / explore), yaw_change_deg = the
teacher's heading change over the 8 waypoints (frames t -> t + 40 at spacing 5) -> forward (|change| <= 30 deg) /
turn class of the forward/turn report.

`compare_v2(scores, samples)`:
- row3 (G10, primary; p10_rules.md v2): maps (noisy layer) - zeroed (both map tokens shown, all four layers zeroed),
  drive-level paired bootstrap 95% CI (10,000), pass = CI high < 0; R = (zeroed - maps) / zeroed. Without a zeroed
  arm (exploration list, in-training subset) row3 is primary - hidden and never passes on the search list;
- maps_vs_hidden (report only, the rules v1 row 3): primary - hidden, shown next to its step-0 baseline;
- report only: every other arm vs maps and vs hidden; per category and forward/turn class (and the forward /
  sideways parts of the loss) of primary - hidden; attention share per map token (near, far) per arm.
"""

import math
from typing import Any, Dict, List

import numpy as np

from vint_train.mapmad.offline_scoring import ROW3_CONTROL
from vint_train.mapmad.offline_stats import paired_stats

TURN_DEG = 30.0
N_SEARCH, N_EXPLORE = 5000, 2000
AHEAD = 40  # 8 waypoints x spacing 5


def v2_fields(meta: Dict[str, Any], traj: Dict[str, np.ndarray], t: int) -> Dict[str, Any]:
    """The v2 list fields of a sample (module docstring)."""
    yaw = np.asarray(traj["yaw"], dtype=np.float64).reshape(-1)
    d = math.degrees(math.atan2(math.sin(yaw[t + AHEAD] - yaw[t]), math.cos(yaw[t + AHEAD] - yaw[t])))
    return {"type": meta["kind"], "category": meta["category"] if meta["kind"] == "search" else "explore",
            "yaw_change_deg": round(d, 3)}


def motion_class(yaw_change_deg: float) -> str:
    return "turn" if abs(yaw_change_deg) > TURN_DEG else "forward"


def _paired(a: np.ndarray, b: np.ndarray, drives: np.ndarray, sel: np.ndarray) -> Dict[str, Any]:
    return paired_stats(a[sel], b[sel], drives[sel]) if sel.any() else {"n_samples": 0}


def compare_v2(scores: Dict[str, np.ndarray], samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The comparison of one v2 score file (module docstring); scores hold "<arm>.<field>" arrays in list order."""
    drives = np.asarray(scores["drive"])
    arms = sorted({k.split(".")[0] for k in scores if k.endswith(".gc_action_loss")})
    loss = {a: scores[f"{a}.gc_action_loss"] for a in arms}
    out: Dict[str, Any] = {"means": {a: float(v.mean()) for a, v in loss.items()}, "arms": arms}
    primary = "maps" if "maps" in loss else "explore_maps"
    control = ROW3_CONTROL if primary == "maps" and ROW3_CONTROL in loss else "hidden"
    a, b = loss[primary], loss["hidden"]
    out["row3"] = dict(paired_stats(a, loss[control], drives), arm=primary, control=control)
    out["row3"]["pass"] = bool(control == ROW3_CONTROL and out["row3"]["diff"] < 0
                               and out["row3"]["diff_ci_drive"][1] < 0)
    out["maps_vs_hidden"] = dict(paired_stats(a, b, drives), arm=primary, control="hidden")
    out["vs_maps"] = {x: paired_stats(loss[x], a, drives) for x in arms if x not in (primary, "hidden")}
    out["vs_hidden"] = {x: paired_stats(loss[x], b, drives) for x in arms if x not in (primary, "hidden")}
    cats = np.array([s["category"] for s in samples])
    out["per_category"] = {str(c): _paired(a, b, drives, cats == c) for c in np.unique(cats)}
    cls = np.array([motion_class(s["yaw_change_deg"]) for s in samples])
    out["forward_turn"] = {}
    for c in ("forward", "turn"):
        sel = cls == c
        row = _paired(a, b, drives, sel)
        row["share"] = float(sel.mean())
        for arm in (primary, "hidden"):
            for part in ("forward", "sideways"):
                key = f"{arm}.loss_{part}"
                if key in scores:
                    row[f"{arm}_{part}_part"] = float(scores[key][sel].mean()) if sel.any() else None
        out["forward_turn"][c] = row
    out["attention_share"] = {}
    for arm in arms:
        sh = np.asarray(scores[f"{arm}.attn_share"], dtype=np.float64)
        if sh.ndim == 2 and not np.all(np.isnan(sh)):
            out["attention_share"][arm] = {"near": float(np.nanmean(sh[:, 0])), "far": float(np.nanmean(sh[:, 1]))}
    return out
