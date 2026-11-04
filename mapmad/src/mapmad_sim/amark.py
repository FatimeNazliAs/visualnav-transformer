"""Phase 14b-2 R4 switch A-mark, "drive to the mark" (simulator side; search_v2 setting `amark`; rules p14b2_rules.md
§R4; VLFM: navigate to the closest point of the accepted object cluster, then stop facing it).

Per control step, after the rule value (search_exam.rule_value) of the frame just added:
- **Idle:** the target layer holds an *accepted cluster* when a cell has noisy-layer value >= tau (the stop rule's
  threshold), was seen from >= min_views views, is in view now (one of this frame's view cells) and lies <= trigger_m
  (3 m) from the robot, and is not within reject_m of a cluster already handled. The strongest such cell (ties: the
  first in row-major order) is the cluster; its goal = the nearest free cell (explored, outside the inflated obstacles
  and bumps; search_exam.nearest_free) within goal_m (0.9 m). No such cell -> the cluster is rejected ("no_free").
- **Driving:** explorer.teacher_step toward the goal on the robot's own map (arm F's planner: the same TeacherSpec,
  inflation, arrival 0.1 m, face_yaw toward the cluster), the NoMaD side steers like arm F (FrontierPolicy). The
  network is not asked meanwhile.
- **On arrival, facing the cluster** (ARRIVED within face_tolerance_deg): the stop rule is evaluated there -- the rule
  value of this step >= tau -> "confirmed" (that step is a rule stop, scored offline as always); else "unconfirmed",
  and the cluster is rejected. Planner failure ("no_path") or more than max_steps steps -> rejected too.
- Every handled cluster (confirmed or rejected) is remembered (a disk of reject_m around it is excluded) and the
  network drives on.
The step's output (`extras["amark"]`, only while the switch is on): {"steer": bool, "mode", "next_point", "face_yaw",
"pose", "event" (trigger | confirmed | unconfirmed | no_free | no_path | timeout | None), "cluster": [x, y] | None}.
Deterministic (no random numbers). Off (no `amark` key) -> nothing is created and no reply changes.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.explorer import APPROACH, ARRIVED, OwnMap, SeenInstance, TeacherSpec, TeacherState, teacher_step
from mapmad_sim.search_exam import cell_centre, disk_mask, nearest_free

EVENTS = ("trigger", "confirmed", "unconfirmed", "no_free", "no_path", "timeout")


@dataclass(frozen=True)
class AMarkSpec:
    teacher: TeacherSpec
    tau: float = 0.45
    min_views: int = 2
    trigger_m: float = 3.0
    goal_m: float = 0.9
    reject_m: float = 0.5
    max_steps: int = 120

    @classmethod
    def from_wire(cls, d: Dict[str, Any]) -> "AMarkSpec":
        return cls(TeacherSpec.from_config(d["spec"]), float(d["tau"]), int(d["min_views"]), float(d["trigger_m"]),
                   float(d["goal_m"]), float(d["reject_m"]), int(d["max_steps"]))


def accepted_cell(value: np.ndarray, count: np.ndarray, view: np.ndarray, shape: Tuple[int, int],
                  origin: Sequence[float], res: float, pose: Sequence[float], spec: AMarkSpec,
                  exclude: Optional[np.ndarray] = None) -> int:
    """Flat index of the strongest accepted cell (module docstring, Idle), -1 if none."""
    view = np.asarray(view, np.int64)
    if view.size == 0:
        return -1
    ok = (value[view] >= spec.tau - 1e-12) & (count[view] >= spec.min_views)
    if exclude is not None:
        ok &= ~exclude[view]
    cand = view[ok]
    if cand.size == 0:
        return -1
    i, j = np.divmod(cand, shape[1])
    cx, cy = origin[0] + (i + 0.5) * res, origin[1] + (j + 0.5) * res
    cand = cand[np.hypot(cx - pose[0], cy - pose[1]) <= spec.trigger_m]
    if cand.size == 0:
        return -1
    best = value[cand].max()
    return int(np.sort(cand[value[cand] >= best])[0])  # ties: first in row-major order


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class AMarkAgent:
    """The A-mark switch on the live map of one episode (module docstring)."""

    spec: AMarkSpec
    state: TeacherState = field(default_factory=TeacherState)
    handled: List[Tuple[float, float]] = field(default_factory=list)
    active: Optional[Tuple[int, Tuple[float, float], Tuple[float, float]]] = None  # (cell, centre, goal)
    steps: int = 0
    counts: Dict[str, int] = field(default_factory=lambda: {e: 0 for e in EVENTS})

    def _exclude(self, shape: Tuple[int, int], origin: Sequence[float], res: float) -> Optional[np.ndarray]:
        if not self.handled:
            return None
        ex = np.zeros(shape[0] * shape[1], dtype=bool)
        for c in self.handled:
            ex |= disk_mask(shape, origin, res, c, self.spec.reject_m).ravel()
        return ex

    def _close(self, event: str, pose: Sequence[float]) -> Dict[str, Any]:
        centre = self.active[1]
        self.handled.append(centre)
        self.active, self.steps = None, 0
        self.state = replace(self.state, approach=None, known_instances=())
        self.counts[event] += 1
        return {"steer": False, "event": event, "cluster": list(centre), "pose": [float(v) for v in pose]}

    def step(self, search: Any, own: OwnMap, pose: Tuple[float, float, float], rule_value: float) -> Dict[str, Any]:
        """This step's A-mark output (module docstring)."""
        fus = search.fusions["noisy"]
        shape, origin, res = search.fmap.shape, search.fmap.origin, own.res
        event = None
        if self.active is None:
            k = accepted_cell(fus.value, fus.count, search.last_cells.view, shape, origin, res, pose, self.spec,
                              self._exclude(shape, origin, res))
            if k < 0:
                return {"steer": False, "event": None, "cluster": None, "pose": [float(v) for v in pose]}
            centre = cell_centre(k, shape, origin, res)
            goal = nearest_free(own, centre, self.spec.goal_m, self.spec.teacher.inflation_m)
            self.active, self.steps = (k, centre, goal if goal is not None else centre), 0
            if goal is None:
                return self._close("no_free", pose)
            self.counts["trigger"] += 1
            event = "trigger"
        k, centre, goal = self.active
        self.steps += 1
        if self.steps > self.spec.max_steps:
            return self._close("timeout", pose)
        dec, self.state = teacher_step(own, pose, self.state, [SeenInstance(int(k), centre, (goal,))],
                                       self.spec.teacher, True)
        if dec.mode not in (APPROACH, ARRIVED):
            return self._close("no_path", pose)
        if (dec.mode == ARRIVED and dec.face_yaw is not None
                and abs(wrap(dec.face_yaw - pose[2])) <= math.radians(self.spec.teacher.face_tolerance_deg)):
            return self._close("confirmed" if rule_value >= self.spec.tau - 1e-12 else "unconfirmed", pose)
        return {"steer": True, "event": event, "mode": dec.mode, "pose": [float(v) for v in pose],
                "next_point": None if dec.next_point is None else [float(v) for v in dec.next_point],
                "face_yaw": dec.face_yaw, "cluster": list(centre), "goal": list(goal)}
