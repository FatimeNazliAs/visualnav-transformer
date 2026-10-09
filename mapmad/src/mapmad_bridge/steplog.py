"""Reading closed-loop step logs (written by vint_train.mapmad.closed_loop.runner). Stdlib only, Python 3.8+:
used in both containers.

A log is JSONL: line 1 = episode header ("kind": "episode", with the start state), then one "step" line per
control step (step 1, 2, ...; v, w, position, yaw, geodesic_m, collided, ...), last line = "result".
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

Log = List[Dict[str, Any]]


def read(path: Path) -> Log:
    with open(path) as f:
        return [json.loads(line) for line in f]


def header(log: Log) -> Dict[str, Any]:
    return log[0]


def steps(log: Log) -> List[Dict[str, Any]]:
    return [line for line in log if line["kind"] == "step"]


def result(log: Log) -> Dict[str, Any]:
    return log[-1]


def poses(log: Log) -> List[Tuple[List[float], float]]:
    """(position, yaw) after every step; index 0 = the start pose."""
    start = header(log)["start"]
    return [(start["position"], start["yaw"])] + [(s["position"], s["yaw"]) for s in steps(log)]


def runs(flags: List[bool]) -> Iterator[Tuple[int, int]]:
    """(start index, length) of every run of True values."""
    start = None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            yield start, i - start
            start = None


def stuck_streaks(log: Log, min_steps: int) -> List[Tuple[int, int]]:
    """(first step number, length) of every run of >= min_steps consecutive collision steps."""
    st = steps(log)
    return [(st[i]["step"], n) for i, n in runs([s["collided"] for s in st]) if n >= min_steps]


def first_stuck(log: Log, min_steps: int) -> Optional[int]:
    found = stuck_streaks(log, min_steps)
    return found[0][0] if found else None
