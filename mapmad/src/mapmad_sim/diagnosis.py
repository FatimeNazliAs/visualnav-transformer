"""Phase 14a step 14.3 (report only): does a learned searcher fail by going to the WRONG PLACE or by BAD DRIVING?
The explorer teacher (explorer.teacher_step, frozen blacklist as the DAgger labeller) is asked at the robot's own
states of the search phase (before the first true sight) and the robot's movement is compared with its answer in
20-step windows (kickoff 14.3, p14a_rules.md §6).

Per window [t0, t1] (t1 = t0 + 20, or the search phase's end; windows shorter than 10 steps are dropped):
- teacher at t0: committed goal g0, path length L0, path direction d (the teacher's path point 1.0 m along its path,
  or for DAgger rollouts the direction of its look-ahead label); at t1: goal g1, path length L1;
- robot: net movement m = position(t1) - position(t0); collision steps c;
- progress = (g1 within 0.3 m of g0 and L1 < L0 - 0.1 m) or (g1 elsewhere and the straight distance robot -> g0
  fell by >= 0.1 m);
- frontier clusters at t0 (map_v2.frontier_clusters on the own map): the teacher's cluster = the one nearest g0; the
  robot's = the cluster whose distance to the robot fell most over the window (>= 0.1 m), else none.
Classes, first match: bad_driving if c >= 10 (scaled by the window length / 20); other_teacher if the teacher had no
goal; other_still if |m| < 0.1 m; agree if angle(m, d) <= 30 deg and progress; bad_driving if angle <= 30 deg (no
progress); wrong_place if angle > 60 deg and the robot's cluster exists and differs from the teacher's; other.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CLASSES = ("agree", "wrong_place", "bad_driving", "other", "other_still", "other_teacher")
WINDOW = 20
MIN_WINDOW = 10
ALONG_M = 1.0


@dataclass
class TeacherView:
    """What the teacher said at one step: goal, path length, direction (unit 2D vector or None)."""

    goal: Optional[Tuple[float, float]]
    length: float
    direction: Optional[Tuple[float, float]]


@dataclass
class Window:
    t0: int
    t1: int
    move: Tuple[float, float]
    collisions: int
    teacher0: TeacherView
    teacher1: TeacherView
    robot0: Tuple[float, float]
    robot1: Tuple[float, float]
    robot_cluster: Optional[int]
    teacher_cluster: Optional[int]


def angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = math.hypot(*a), math.hypot(*b)
    if na == 0 or nb == 0:
        return 180.0
    c = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def progress(w: Window) -> bool:
    g0, g1 = w.teacher0.goal, w.teacher1.goal
    if g0 is None:
        return False
    if g1 is not None and math.hypot(g1[0] - g0[0], g1[1] - g0[1]) <= 0.3:
        return math.isfinite(w.teacher1.length) and w.teacher1.length < w.teacher0.length - 0.1
    d0 = math.hypot(g0[0] - w.robot0[0], g0[1] - w.robot0[1])
    d1 = math.hypot(g0[0] - w.robot1[0], g0[1] - w.robot1[1])
    return d1 <= d0 - 0.1


def classify(w: Window) -> str:
    """The window's class (module docstring)."""
    n = w.t1 - w.t0
    if w.collisions >= 10 * n / WINDOW:
        return "bad_driving"
    if w.teacher0.goal is None or w.teacher0.direction is None:
        return "other_teacher"
    if math.hypot(*w.move) < 0.1:
        return "other_still"
    a = angle_deg(w.move, w.teacher0.direction)
    if a <= 30.0:
        return "agree" if progress(w) else "bad_driving"
    if a > 60.0 and w.robot_cluster is not None and w.robot_cluster != w.teacher_cluster:
        return "wrong_place"
    return "other"


def cluster_of_point(labels: np.ndarray, origin: Sequence[float], res: float, p: Sequence[float]) -> Optional[int]:
    """The frontier cluster with a cell nearest to p (None without clusters)."""
    ii, jj = np.nonzero(labels)
    if ii.size == 0:
        return None
    cx, cy = origin[0] + (ii + 0.5) * res, origin[1] + (jj + 0.5) * res
    k = int(np.argmin(np.hypot(cx - p[0], cy - p[1])))
    return int(labels[ii[k], jj[k]])


def robot_cluster(labels: np.ndarray, origin: Sequence[float], res: float, p0: Sequence[float],
                  p1: Sequence[float]) -> Optional[int]:
    """The cluster whose distance to the robot fell most from p0 to p1 (>= 0.1 m), else None."""
    ii, jj = np.nonzero(labels)
    if ii.size == 0:
        return None
    cx, cy = origin[0] + (ii + 0.5) * res, origin[1] + (jj + 0.5) * res
    lab = labels[ii, jj]
    d0 = np.hypot(cx - p0[0], cy - p0[1])
    d1 = np.hypot(cx - p1[0], cy - p1[1])
    best, gain = None, 0.1
    for c in np.unique(lab).tolist():
        m = lab == c
        g = float(d0[m].min() - d1[m].min())
        if g >= gain:
            best, gain = int(c), g
    return best


def path_direction(path: np.ndarray, pose: Sequence[float], along_m: float = ALONG_M) -> Optional[Tuple[float, float]]:
    """Unit vector from the pose to the point along_m along the path (explorer.path_point), None for a path of < 2
    points or zero length."""
    from mapmad_sim.explorer import path_point

    if path is None or len(path) < 2:
        return None
    p = path_point(np.asarray(path, np.float64), along_m)
    d = (p[0] - pose[0], p[1] - pose[1])
    n = math.hypot(*d)
    return None if n < 1e-6 else (d[0] / n, d[1] / n)


def windows_of(end: int, window: int = WINDOW, min_window: int = MIN_WINDOW) -> List[Tuple[int, int]]:
    """[(t0, t1)] covering frames 0..end in steps of `window` (the last one kept if >= min_window steps)."""
    out = []
    t0 = 0
    while t0 < end:
        t1 = min(t0 + window, end)
        if t1 - t0 >= min_window:
            out.append((t0, t1))
        t0 += window
    return out


def summarise(classes: Sequence[Tuple[str, int]]) -> Dict[str, Any]:
    """Shares of windows and of search steps per class from (class, window length) pairs."""
    n = len(classes)
    steps = sum(k for _, k in classes)
    return {"windows": n, "steps": steps,
            "window_share": {c: (sum(1 for x, _ in classes if x == c) / n if n else None) for c in CLASSES},
            "step_share": {c: (sum(k for x, k in classes if x == c) / steps if steps else None) for c in CLASSES}}
