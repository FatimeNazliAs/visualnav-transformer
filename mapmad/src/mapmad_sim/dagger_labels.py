"""DAgger labels (Phase 11 block item 3): at every visited state of a saved drive, what the explorer teacher would do
from that exact state, on the robot's own map, as the network's 8 output waypoints.

Per frame t of a drive (Phase 8 drive format; a student rollout or, for the label check, a teacher drive):
1. own map at t: obstacle / explored as known at t (mapmad_map.npz first-seen <= t), cells bumped before t
   (mapmad_frames.npz bump_events, map_v2.bumped_cells: a planning obstacle, as the generator gave the teacher);
   seen = simulator-truth "seen" instances (mapmad_frames.npz seen_events at frames <= t; >= 200 px at <= 4 m, sticky);
2. explorer.teacher_step(own map, pose_t, teacher state, seen) -> the teacher's path. The teacher state is carried
   from frame to frame along the drive: the committed frontier stays committed until reached / gone (then re-chosen
   as usual). STUDENT rollouts: the blacklist counter is FROZEN (spec.blacklist_window_steps -> never: a frontier is
   never dropped because the student did not drive toward it); the drive's own spec otherwise;
3. label = kinematic look-ahead on the frozen map (own map at t, nothing seen later): the Phase 8 motion model
   (expert.PurePursuit, 0.2 m/s, 0.4 rad/s, 4 Hz, never reversing; a new choice > 30 deg off first turns on the spot as
   the generator's retarget; arrived / at the end of an approach path: turn on the spot to face the instance centre,
   then stand) for `steps` (40) control steps from pose_t -> the poses after every `every`-th (5th) step -> 8 waypoints
   in the robot frame at t (metres; / 0.25 m = the network's normalised format, vint_dataset._compute_actions).
   Two variants (LabelSpec.replan): False = follow the path of frame t to its end ("frozen path"); True = re-ask the
   teacher at every look-ahead step from the simulated pose on the same frozen map, with its state carried, and follow
   that path for one step ("replanning", as the generator's drive loop does every step);
4. no label (`reason`): collision (the move after frame t is inside a run of >= collision_skip consecutive collision
   steps; such states are excluded like Phase 8's collision drives) or teacher_fail (no_frontier, blacklist_exhausted,
   no_path). A state that is both counts as collision when LabelSpec.collision_first (else as teacher_fail).
`label_drive` returns per-frame arrays (labels.npz); `label_check` compares look-ahead labels on teacher drives with
the teacher's real future poses at the same frames.
"""

import math
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.explorer import (APPROACH, ARRIVED, BLACKLIST_EXHAUSTED, NO_FRONTIER, NO_PATH, Decision, OwnMap,
                                 SeenInstance, TeacherSpec, TeacherState, teacher_step)
from mapmad_sim.expert import PursuitSpec, PurePursuit, TrackedPath, spot_turn, wrap
from mapmad_sim.frames import to_2d, to_robot
from vint_train.mapmad.local_map import known_at
from vint_train.mapmad.map_v2 import bumped_cells

REASONS = ("ok", NO_FRONTIER, BLACKLIST_EXHAUSTED, NO_PATH, "collision")
TEACHER_FAILS = (NO_FRONTIER, BLACKLIST_EXHAUSTED, NO_PATH)
LABELS_FILE = "labels.npz"
FROZEN_WINDOW = 10 ** 9  # blacklist_window_steps that never fills: the progress counter is frozen
Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class LabelSpec:
    """configs/p11_dagger.yaml `rollouts` + the generator's motion numbers."""

    steps: int = 40
    every: int = 5
    collision_skip: int = 20
    retarget_turn_deg: float = 30.0
    face_tolerance_deg: float = 15.0
    replan: bool = True  # look-ahead variant (module docstring, step 3)
    collision_first: bool = True  # a state that is both: skipped as collision (step 4)


def frozen_blacklist(spec: TeacherSpec) -> TeacherSpec:
    """The teacher spec for student rollouts: identical, but the blacklist's progress counter never fires."""
    return replace(spec, blacklist_window_steps=FROZEN_WINDOW)


# --- the look-ahead ---------------------------------------------------------------------------------------------

def arc(pose: Pose, v: float, w: float, dt: float) -> Pose:
    """Exact unicycle step in the 2D frame (forward = (cos yaw, sin yaw); robot.arc_step mapped by frames.to_2d)."""
    x, y, yaw = pose
    if abs(w) < 1e-9:
        return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw
    y1 = yaw + w * dt
    return x + (v / w) * (math.sin(y1) - math.sin(yaw)), y - (v / w) * (math.cos(y1) - math.cos(yaw)), y1


def lookahead(dec: Decision, pose: Pose, face_centre: Optional[Tuple[float, float]], pursuit: PursuitSpec,
              spec: LabelSpec) -> np.ndarray:
    """(steps + 1, 3) poses of the Phase 8 motion model following the teacher's decision (module docstring, step 3)."""
    poses = [tuple(pose)]
    pp = PurePursuit(pursuit, TrackedPath(dec.path)) if len(dec.path) >= 2 else None
    retarget = dec.new_choice and dec.next_point is not None
    for _ in range(spec.steps):
        p = poses[-1]
        v = w = 0.0
        facing = dec.mode == ARRIVED or pp is None or pp.distance_to_end(p) <= pursuit.goal_tolerance_m
        if facing:
            yaw = dec.face_yaw if dec.face_yaw is not None else (
                math.atan2(face_centre[1] - p[1], face_centre[0] - p[0]) if (dec.mode == APPROACH and face_centre) else None)
            if yaw is not None and abs(wrap(yaw - p[2])) > math.radians(spec.face_tolerance_deg):
                w = spot_turn(wrap(yaw - p[2]), pursuit.max_w, pursuit.dt)
        else:
            aim = math.atan2(*to_robot(np.asarray([dec.next_point]), p)[0][::-1]) if retarget else 0.0
            if retarget and abs(aim) >= math.radians(spec.retarget_turn_deg):
                w = spot_turn(aim, pursuit.max_w, pursuit.dt)
            else:
                retarget = False
                c = pp.command(p)
                v, w = (0.0, 0.0) if c.done else (c.v, c.w)
        poses.append(arc(p, v, w, pursuit.dt))
    return np.asarray(poses, dtype=np.float64)


def motion_step(dec: Decision, pose: Pose, retarget: bool, pursuit: PursuitSpec, spec: LabelSpec) -> Tuple[float, float, bool]:
    """One control step of the generator's drive loop for the teacher's decision of this step: (v, w, still retargeting)
    -- the retarget spot turn while a new choice is > 30 deg off, pure pursuit on the decision's path (a fresh
    TrackedPath, as datagen_v2._m_drive), the face turn when arrived, standing on a teacher failure."""
    if dec.mode in TEACHER_FAILS:
        return 0.0, 0.0, False
    if dec.mode == ARRIVED or len(dec.path) < 2:
        if dec.face_yaw is not None and abs(wrap(dec.face_yaw - pose[2])) > math.radians(spec.face_tolerance_deg):
            return 0.0, spot_turn(wrap(dec.face_yaw - pose[2]), pursuit.max_w, pursuit.dt), False
        return 0.0, 0.0, False
    if retarget and dec.next_point is not None:
        aim = math.atan2(*to_robot(np.asarray([dec.next_point]), pose)[0][::-1])
        if abs(aim) >= math.radians(spec.retarget_turn_deg):
            return 0.0, spot_turn(aim, pursuit.max_w, pursuit.dt), True
    c = PurePursuit(pursuit, TrackedPath(dec.path)).command(pose)
    return (0.0, 0.0, False) if c.done else (c.v, c.w, False)


def lookahead_replan(own: OwnMap, dec: Decision, state: TeacherState, pose: Pose, seen: Sequence[SeenInstance],
                     teacher: TeacherSpec, pursuit: PursuitSpec, spec: LabelSpec) -> np.ndarray:
    """(steps + 1, 3) poses of the replanning look-ahead (module docstring, step 3): `dec` / `state` = the teacher's
    decision at frame t and its memory after it; every later step asks teacher_step again on the same own map."""
    poses = [tuple(pose)]
    retarget = dec.new_choice
    for k in range(spec.steps):
        p = poses[-1]
        if k:
            dec, state = teacher_step(own, p, state, seen, teacher, searching=True)
            retarget = retarget or dec.new_choice
        v, w, retarget = motion_step(dec, p, retarget, pursuit, spec)
        poses.append(arc(p, v, w, pursuit.dt))
    return np.asarray(poses, dtype=np.float64)


def label_poses(own: OwnMap, dec: Decision, state: TeacherState, pose: Pose, seen: Sequence[SeenInstance],
                teacher: TeacherSpec, pursuit: PursuitSpec, spec: LabelSpec) -> np.ndarray:
    """The look-ahead poses of the variant spec.replan selects."""
    if spec.replan:
        return lookahead_replan(own, dec, state, pose, seen, teacher, pursuit, spec)
    centre = next((s.centre for s in seen if dec.info.get("instance") == s.instance_id), None)
    return lookahead(dec, pose, centre, pursuit, spec)


def waypoints_of(poses: np.ndarray, every: int) -> np.ndarray:
    """(8, 2) robot-frame metres of poses[every], poses[2 every], ... relative to poses[0]."""
    return to_robot(poses[every::every, :2], tuple(poses[0])).astype(np.float64)


# --- one drive --------------------------------------------------------------------------------------------------

def load_drive(folder: Path) -> Dict[str, Any]:
    """What labelling needs from a saved drive."""
    import json

    with np.load(folder / "mapmad_map.npz") as f:
        dm = {k: f[k] for k in f.files}
    with np.load(folder / "mapmad_frames.npz") as f:
        fr = {k: f[k] for k in f.files}
    with open(folder / "traj_data.pkl", "rb") as f:
        traj = pickle.load(f)
    meta = json.loads((folder / "mapmad_meta.json").read_text())
    return {"map": dm, "frames": fr, "traj": traj, "meta": meta}


def seen_instances(meta: Dict[str, Any]) -> Dict[int, SeenInstance]:
    """The drive's usable instances as the teacher sees them once seen (datagen_v2.Instance.seen)."""
    out = {}
    for i in meta["instances"]:
        c = to_2d(i["centre"])
        out[int(i["object_id"])] = SeenInstance(int(i["object_id"]), (float(c[0]), float(c[1])),
                                                tuple((float(p[0]), float(p[1])) for p in to_2d(np.asarray(i["view_points"]))))
    return out


def collision_frames(collided: np.ndarray, min_run: int) -> np.ndarray:
    """Bool per frame: the move after the frame is inside a run of >= min_run consecutive collision steps."""
    out = np.zeros(len(collided), dtype=bool)
    start = None
    for i, c in enumerate(list(collided) + [False]):
        if c and start is None:
            start = i
        elif not c and start is not None:
            if i - start >= min_run:
                out[start:i] = True
            start = None
    return out


def teacher_states(drive: Dict[str, Any], spec: TeacherSpec):
    """(t, pose, decision, seen, own map, teacher state after the decision) for every frame, the state carried along
    (module docstring, steps 1-2)."""
    dm, fr, traj = drive["map"], drive["frames"], drive["traj"]
    insts = seen_instances(drive["meta"])
    events = np.asarray(fr["seen_events"], dtype=np.int64).reshape(-1, 2)
    origin, res = tuple(float(v) for v in dm["origin"]), float(dm["resolution"])
    state = TeacherState()
    for t in range(len(traj["position"])):
        own = OwnMap(known_at(dm["obstacle_first_seen"], t), known_at(dm["explored_first_seen"], t), origin, res,
                     bumped_cells(dm["obstacle_first_seen"].shape, origin, res, fr["bump_events"], t))
        pose = (float(traj["position"][t, 0]), float(traj["position"][t, 1]), float(traj["yaw"][t]))
        seen = [insts[int(i)] for i in sorted(set(events[events[:, 0] <= t, 1].tolist())) if int(i) in insts]
        dec, state = teacher_step(own, pose, state, seen, spec, searching=True)
        yield t, pose, dec, seen, own, state


def label_drive(folder: Path, teacher: TeacherSpec, pursuit: PursuitSpec, spec: LabelSpec,
                freeze_blacklist: bool = True) -> Dict[str, np.ndarray]:
    """labels.npz arrays of one drive (module docstring): waypoints_m (N, 8, 2) float32 (NaN = no label), valid (N,),
    reason (N,) index into REASONS, teacher_mode (N,) index into explorer modes of datagen_v2.MODES, goal (N, 2),
    path_length_m (N,), new_choice (N,)."""
    from mapmad_sim.datagen_v2 import MODES

    drive = load_drive(Path(folder))
    tspec = frozen_blacklist(teacher) if freeze_blacklist else teacher
    n = len(drive["traj"]["position"])
    k = spec.steps // spec.every
    wp = np.full((n, k, 2), np.nan, dtype=np.float32)
    reason = np.zeros(n, dtype=np.uint8)
    mode = np.zeros(n, dtype=np.uint8)
    goal = np.full((n, 2), np.nan, dtype=np.float64)
    plen = np.full(n, np.nan, dtype=np.float32)
    new_choice = np.zeros(n, dtype=bool)
    coll = collision_frames(np.asarray(drive["frames"]["collided"], bool), spec.collision_skip)
    for t, pose, dec, seen, own, state in teacher_states(drive, tspec):
        mode[t] = MODES.index(dec.mode)
        if dec.goal is not None:
            goal[t] = dec.goal
        plen[t] = dec.path_length_m
        new_choice[t] = dec.new_choice
        if coll[t] and spec.collision_first:
            reason[t] = REASONS.index("collision")
            continue
        if dec.mode in TEACHER_FAILS:
            reason[t] = REASONS.index(dec.mode)
            continue
        if coll[t]:
            reason[t] = REASONS.index("collision")
            continue
        wp[t] = waypoints_of(label_poses(own, dec, state, pose, seen, tspec, pursuit, spec), spec.every)
    teacher_fail = np.isin(mode, [MODES.index(m) for m in TEACHER_FAILS])
    return {"waypoints_m": wp, "valid": reason == 0, "reason": reason, "reason_names": np.asarray(REASONS),
            "teacher_fail": teacher_fail, "collision_streak": coll,
            "teacher_mode": mode, "teacher_mode_names": np.asarray(MODES), "goal_xy": goal, "path_length_m": plen,
            "new_choice": new_choice, "frozen_blacklist": np.asarray(freeze_blacklist),
            "label_steps": np.asarray(spec.steps), "label_every": np.asarray(spec.every),
            "replan": np.asarray(spec.replan), "collision_first": np.asarray(spec.collision_first)}


def label_check_drive(folder: Path, teacher: TeacherSpec, pursuit: PursuitSpec, spec: LabelSpec) -> Dict[str, Any]:
    """Label check on one TEACHER drive: at every frame with `steps` future frames, the look-ahead label (the drive's
    own teacher spec, blacklist not frozen) vs the real future waypoints (traj_data.pkl at t + 5, ..., t + 40):
    per-frame mean waypoint error (m) and the 8 per-waypoint errors, phase (search / approach), and whether the
    relabelled goal equals the logged one (a relabelling sanity check)."""
    from mapmad_sim.datagen_v2 import PHASES_V2

    drive = load_drive(Path(folder))
    traj, fr = drive["traj"], drive["frames"]
    n = len(traj["position"])
    logged_goal = np.asarray(fr["goal_xy"], dtype=np.float64)
    errors, per_wp, phases, same_goal = [], [], [], []
    for t, pose, dec, seen, own, state in teacher_states(drive, teacher):
        if t + spec.steps >= n or dec.mode in TEACHER_FAILS:
            continue
        label = waypoints_of(label_poses(own, dec, state, pose, seen, teacher, pursuit, spec), spec.every)
        future = np.stack([traj["position"][t:t + spec.steps + 1:spec.every, 0],
                           traj["position"][t:t + spec.steps + 1:spec.every, 1]], axis=1)
        real = to_robot(future[1:], pose)
        d = np.hypot(*(label - real).T)
        errors.append(float(d.mean()))
        per_wp.append(d.tolist())
        phases.append(str(PHASES_V2[int(fr["phase"][t])]))
        g = dec.goal
        same_goal.append(bool(g is not None and np.allclose(g, logged_goal[t], atol=1e-6)))
    return {"drive": Path(folder).name, "errors": errors, "per_waypoint": per_wp, "phases": phases, "same_goal": same_goal}
