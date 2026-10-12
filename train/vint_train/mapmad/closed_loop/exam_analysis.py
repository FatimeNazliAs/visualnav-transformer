"""Phase 5 analysis from the logs (block items 10-12): per run facts, failure types, per episode means, tables.

Per run (one log): primary success, success@500, SPL, collisions, stuck streaks (>= 100 consecutive collision steps),
final distance, secondary "reached at any step" (target point <= 1.0 m, any side = nearest view point <= 1.0 m),
and the failure type of a failed run, first match in this order (labelled automatically, block item 12):
1. stuck            -- the run has a stuck streak;
2. wrong_instance   -- at some step the turning centre came within WRONG_INSTANCE_M (1.0 m, horizontal distance to
                       the box) of ANOTHER instance of the target's category on the same storey (navmesh height
                       between 2.0 m below and 0.5 m above the box bottom, Phase 1's object-storey rule);
3. circling         -- more than CIRCLING_SHARE (50%) of the path's cell entries go into a cell visited before
                       (CIRCLING_CELL_M = 0.25 m cells of the turning centre; consecutive steps in one cell = one
                       entry; at least 10 entries);
4. timeout_moving   -- everything else (drove until the step limit without one of the above).
Per episode: the mean over its seed runs. Crashed runs (`*.crash.json`, block item 13b) drop the episode from every arm.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_bridge import steplog
from mapmad_sim.episodes import box_distance_xz
from mapmad_sim.run_layout import RunLayout, parse_run_id

WRONG_INSTANCE_M = 1.0
CIRCLING_CELL_M = 0.25
CIRCLING_SHARE = 0.5
CIRCLING_MIN_ENTRIES = 10
FAILURE_TYPES = ("success", "stuck", "wrong_instance", "circling", "timeout_moving")
RUN_METRICS = ("success", "success@500", "spl", "collisions", "stuck_streaks", "final_vp_geodesic_m",
               "target_point_reached", "any_side_reached", "path_length_m")


def revisit_share(positions: Sequence[Sequence[float]], cell_m: float = CIRCLING_CELL_M) -> Tuple[float, int]:
    """(share of cell entries into an already visited cell, number of entries) of a path (Habitat x, y, z)."""
    cells = [(math.floor(p[0] / cell_m), math.floor(p[2] / cell_m)) for p in positions]
    entries = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
    seen, again = set(), 0
    for c in entries:
        again += c in seen
        seen.add(c)
    return (again / len(entries) if entries else 0.0), len(entries)


def wrong_instance_step(positions: Sequence[Sequence[float]], others: Sequence[Dict[str, Any]]) -> Optional[int]:
    """First index of a position within WRONG_INSTANCE_M of another instance's box on the same storey."""
    for i, p in enumerate(positions):
        for o in others:
            c, s = o["box_center"], o["box_size"]
            bottom = c[1] - s[1] / 2.0
            if -2.0 <= p[1] - bottom <= 0.5 and box_distance_xz(p, c, s) <= WRONG_INSTANCE_M:
                return i
    return None


def run_facts(log_path: Path, episode: Dict[str, Any], stuck_steps: int, success_at: Sequence[int]) -> Dict[str, Any]:
    """One row per run log (see the module docstring)."""
    log = steplog.read(log_path)
    head, res = steplog.header(log), steplog.result(log)
    positions = [p for p, _ in steplog.poses(log)]
    streaks = steplog.stuck_streaks(log, stuck_steps)
    share, entries = revisit_share(positions)
    wrong = wrong_instance_step(positions, episode.get("other_instances", []))
    row = {"arm": res["arm"], "episode_id": res["episode_id"], "seed_index": res["seed_index"], "home": res["home"],
           "category": res["category"], "distance_bin": res["distance_bin"], "success": float(res["success"]),
           "spl": res["spl"], "collisions": res["collisions"], "stuck_streaks": len(streaks),
           "final_vp_geodesic_m": res["final_vp_geodesic_m"], "path_length_m": res["path_length_m"],
           "target_point_reached": float(res["target_point_step"] is not None),
           "any_side_reached": float(res["any_side_step"] is not None), "steps": res["steps"],
           "revisit_share": share, "wrong_instance_step": wrong, "weights_sha256": head.get("weights_sha256"),
           "episodes_fingerprint": head["episodes_fingerprint"]}
    for n in success_at:
        row[f"success@{n}"] = float(res["success_step"] is not None and res["success_step"] <= n)
    if res["success"]:
        kind = "success"
    elif streaks:
        kind = "stuck"
    elif wrong is not None:
        kind = "wrong_instance"
    elif share > CIRCLING_SHARE and entries >= CIRCLING_MIN_ENTRIES:
        kind = "circling"
    else:
        kind = "timeout_moving"
    row["failure_type"] = kind
    return row


def load_runs(layout: RunLayout, arms: Sequence[str], episodes: Dict[str, Dict[str, Any]], stuck_steps: int,
              success_at: Sequence[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(run rows of the given arms, episode ids with a crashed run in any arm)."""
    crashed = layout.crashed_episodes(arms)
    rows = [run_facts(p, episodes[parse_run_id(p.name)[0]], stuck_steps, success_at)
            for arm in arms for p in layout.logs(arm)]
    return [r for r in rows if r["episode_id"] not in crashed], crashed


def episode_means(rows: List[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """arm -> episode id -> metric -> mean over the episode's seed runs."""
    acc: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        acc[(r["arm"], r["episode_id"])].append(r)
    out: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (arm, ep), rs in acc.items():
        out[arm][ep] = {m: float(np.mean([float(r[m]) for r in rs if r[m] is not None])) if any(
            r[m] is not None for r in rs) else float("nan") for m in metrics}
        out[arm][ep]["runs"] = len(rs)
    return out


def failure_counts(rows: List[Dict[str, Any]], arm: str) -> Dict[str, int]:
    c = Counter(r["failure_type"] for r in rows if r["arm"] == arm)
    return {k: c.get(k, 0) for k in FAILURE_TYPES}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float) + "\n")
