"""Read Phase 1 step logs (no new runs): stuck streaks, failure types, per-category / per-home success.

A stuck streak is a run of >= STUCK_STEPS consecutive steps that each counted as a collision (the robot pushes
against a wall or furniture). Failure types of an episode:
- success;
- fail_stuck: failed and had a stuck streak;
- fail_near: failed, no stuck streak, came within NEAR_M (geodesic) of the target point at some step;
- fail_far: everything else (wandered without getting close).
"""

from pathlib import Path
from typing import Any, Dict, List

from mapmad_bridge import steplog

STUCK_STEPS = 100
NEAR_M = 2.0
FAILURE_TYPES = ("success", "fail_stuck", "fail_near", "fail_far")


def first_stuck(log: List[Dict[str, Any]], min_steps: int = STUCK_STEPS):
    """Step number where the first streak of >= min_steps consecutive collision steps starts, or None."""
    return steplog.first_stuck(log, min_steps)


def failure_type(log: List[Dict[str, Any]]) -> str:
    if steplog.result(log)["success"]:
        return "success"
    if first_stuck(log) is not None:
        return "fail_stuck"
    closest = min((s["geodesic_m"] for s in steplog.steps(log) if s["geodesic_m"] is not None), default=None)
    return "fail_near" if closest is not None and closest <= NEAR_M else "fail_far"


def episode_facts(path: Path) -> Dict[str, Any]:
    """One row per episode log: ids, success, stuck start, failure type, closest geodesic distance."""
    log = steplog.read(path)
    result = steplog.result(log)
    geos = [s["geodesic_m"] for s in steplog.steps(log) if s["geodesic_m"] is not None]
    return {"episode_id": result["episode_id"], "arm": result["arm"], "home": result["home"],
            "category": result["category"], "success": float(result["success"]), "stuck_start": first_stuck(log),
            "failure_type": failure_type(log), "closest_geodesic_m": min(geos) if geos else None}
