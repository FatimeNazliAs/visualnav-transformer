"""Phase 8 drive generator v2: search drives and exploration drives driven by the explorer teacher (v2 plan §8.2-8.3,
block items 1-6; numbers in configs/p8_datagen.yaml, rules in configs/p8_rules.md).

A drive, step by step (one control step = 0.25 s, one saved frame per step), camera pin025 (camera_fair.yaml):
1. frame t: colour picture -> t.jpg, pose -> traj_data.pkl, depth picture -> the drive's first-seen storey map
   (depth_map.py, as Phase 2); search drives: the semantic picture -> detections of the category -> the clean and the
   noisy "target seen here" layers (target_layer.py, map_v2.TargetFusion) and the sticky "seen" set;
2. teacher: explorer.teacher_step(own map at t, pose, memory, seen instances) -> path, next path point, mode;
3. command for the next 0.25 s, by motion phase (Phase 2 motion, pure pursuit on the teacher's path):
   wall_recovery (wall starts) -> start_turn -> drive; retarget (a new frontier / target choice with the aim > 30 deg
   off: spot turn until < 30 deg); kick (kick drives, once: an arc 20-40 deg off the path); detour (detour drives, 1-2
   times: a sideways bulge of 0.5-1.5 m on the floor map, rejoining the teacher path 2.5 m ahead); face (arrived:
   spot turn until the instance centre is within 15 deg); end;
4. the LIMO model moves (robot.py). Never reverses.
Ends: search -> faced the object (success if the walking distance to the nearest view point of any instance of the
category is <= 0.2 m), or no frontier / blacklist exhausted / no path (failure), > 2,000 steps (timeout) or >= 20
consecutive collision steps (collision): both discarded. Exploration -> 1,000 steps or no frontier left (complete);
>= 20 collision steps discarded. Failures and discards are counted, never redrawn (one attempt per drive id).

Every drive has its own seed (drive_seed: sha256 of the config seed and the drive id) and two random streams
(sampling + motion variations; target-layer noise), so a drive comes out the same whichever worker runs it and in
whatever order. Files per drive folder <drive id>: Phase 2's (0.jpg ..., traj_data.pkl, mapmad_frames.npz,
mapmad_map.npz, mapmad_meta.json) plus mapmad_target.npz (map_v2.TARGET_FILE: sparse clean / noisy target-layer
updates and the raw per-frame detections). mapmad_frames.npz: `phase` = search / approach / explore per frame (the
teacher's mode; Phase 2's motion phase is now `motion`), the teacher's choice per step (goal, next point, path length,
new choice, why the last goal was dropped, clusters), its memory events (blacklist, seen) and its time per step.
"""

import hashlib
import json
import math
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mapmad_sim import config, objectnav
from mapmad_sim.camera import centre_depth
from mapmad_sim.camera_setting import CameraSetting
from mapmad_sim.depth_map import FirstSeenMap, MapSpec, world_points
from mapmad_sim.drive_sampler import DriveSampler, Start
from mapmad_sim.expert import PursuitSpec, PurePursuit, TrackedPath, is_spot_turn, resample, smooth_path, spot_turn, wrap
from mapmad_sim.explorer import (APPROACH, ARRIVED, BLACKLIST_EXHAUSTED, EXPLORE, NO_FRONTIER, NO_PATH, SEARCH,
                                 Decision, OwnMap, SeenInstance, TeacherSpec, TeacherState, teacher_step)
from mapmad_sim.floor_grid import FloorGrid
from mapmad_sim.frames import to_2d
from mapmad_sim.route import Variations, deviation, detour_point, draw_variations, max_turn, path_length
from mapmad_sim.robot import LimoSim, RobotSpec
from mapmad_sim.run_layout import write_text_atomic
from mapmad_sim.target_layer import NoiseSpec, detect, frame_cells, frame_scores
from vint_train.mapmad.map_v2 import TARGET_FILE, TargetFusion, visited_mask

PHASES_V2 = (SEARCH, APPROACH, EXPLORE)  # `phase` per frame (block item 4)
MOTIONS = ("start_turn", "wall_recovery", "drive", "retarget", "kick", "detour", "face", "end", "look_start", "look")
# G8 row 3: the four allowed spot-turn types; "look" (a look-around after the first frontier choice) is reported on its
# own and counts as outside them
SPOT_TURN_TYPES = {"start_turn": "start", "look_start": "start", "wall_recovery": "wall_recovery", "face": "face",
                   "retarget": "retarget", "look": "look_around"}
STUCK_MODES = (NO_FRONTIER, BLACKLIST_EXHAUSTED)
MODES = (SEARCH, APPROACH, ARRIVED, EXPLORE, NO_FRONTIER, BLACKLIST_EXHAUSTED, NO_PATH)
DROPPED = ("", "reached", "gone", "unreachable", "blacklisted")
FAILURE_MODES = (NO_FRONTIER, BLACKLIST_EXHAUSTED, NO_PATH)


def drive_seed(cfg_seed: int, drive_id: str) -> int:
    """The drive's seed: first 8 bytes of sha256("<config seed>:<drive id>")."""
    return int.from_bytes(hashlib.sha256(f"{cfg_seed}:{drive_id}".encode()).digest()[:8], "little")


def drive_id(prefix: str, home: str, kind: str, n: int) -> str:
    """p8d-<home>-<search|explore>-<n:03d> (block item 10)."""
    return f"{prefix}-{home}-{kind}-{n:03d}"


def home_drive_ids(prefix: str, home: str, counts: Dict[str, int]) -> List[Tuple[str, str]]:
    """(drive id, kind) of every drive of a home, searches first; refuses duplicates."""
    out = [(drive_id(prefix, home, kind, n), kind) for kind in ("search", "explore") for n in range(counts.get(kind, 0))]
    ids = [d for d, _ in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{home}: duplicate drive ids")
    return out


def robot_spec(cfg: Dict[str, Any]) -> RobotSpec:
    """LIMO (robot_limo.yaml) with the Phase 7 fair camera (pin025); the depth camera moves with it."""
    cam_cfg = config.load_yaml(config.CONFIG_DIR / cfg["camera"])
    spec = CameraSetting.from_config(cam_cfg["name"], cam_cfg).apply(RobotSpec.from_config(config.robot()))
    if [spec.width, spec.height] != list(cam_cfg["resolution"]) or spec.camera_forward_m != cam_cfg["forward_of_base_m"]:
        raise ValueError("camera_fair.yaml resolution / forward offset differ from robot_limo.yaml")
    return spec


# --- the search task ----------------------------------------------------------------------------------------------

@dataclass
class Instance:
    """One ObjectNav goal instance of the home, as the teacher and the success check need it."""

    object_id: int
    centre: List[float]  # Habitat box centre
    size: List[float]
    view_points: List[List[float]]  # snapped onto LIMO's floor map (Habitat), usable (Phase 2 rule)

    def seen(self) -> SeenInstance:
        c = to_2d(self.centre)
        return SeenInstance(self.object_id, (float(c[0]), float(c[1])),
                            tuple((float(p[0]), float(p[1])) for p in to_2d(np.asarray(self.view_points))))


@dataclass
class SearchTask:
    category: str
    instances: List[Instance]  # every instance of the category in the home with a usable view point
    geodesic_m: float  # start -> nearest view point (walking)
    all_ids: List[int]  # every goal instance of the category (detections count all of them; "seen" only usable ones)


class HomeTasks:
    """Per-home lookups for search drives (labelled homes): instances per category with usable view points.
    Start filter (p8_rules_v2.md item 1c, `sampling.seeable_filter`): a category is searchable in this home only if
    from >= 1 usable view point of >= 1 of its instances, one of 8 headings (pin025) shows a usable instance of the
    category with >= min_pixels at depth (0, max_depth] (the "seen" rule). Checked once per home, before any drive,
    in a fixed order (no random draws: results do not depend on the drive order)."""

    def __init__(self, sampler: DriveSampler, goals: Sequence[objectnav.Goal], seeable: Optional[NoiseSpec] = None,
                 headings: int = 8) -> None:
        self.sampler = sampler
        self.by_category: Dict[str, List[Instance]] = {}
        for g in goals:
            box = sampler.labels.box(g.object_id)
            if box is None:
                continue
            vps = [p for p, _, _ in sampler.usable_view_points(g)]
            if not vps:
                continue
            self.by_category.setdefault(g.category, []).append(Instance(
                g.object_id, [float(v) for v in box[0]], [float(v) for v in box[1]], [[float(v) for v in p] for p in vps]))
        self.categories = sorted(self.by_category)
        self.renders: Dict[str, int] = {}
        self.seeable: Dict[str, bool] = {c: True for c in self.categories}
        if seeable is not None:
            self.seeable = {c: self._seeable(c, seeable, headings) for c in self.categories}

    def _seeable(self, category: str, spec: NoiseSpec, n: int) -> bool:
        """Item 1c for one category (stops at the first view point and heading that passes)."""
        robot = self.sampler.robot
        insts = self.by_category[category]
        ids = [inst.object_id for inst in insts]
        self.renders[category] = 0
        for inst in insts:
            for vp in inst.view_points:
                p = np.asarray(vp, dtype=np.float64)
                for k in range(n):
                    robot.place(p, 2.0 * math.pi * k / n)
                    obs = robot.observe()
                    self.renders[category] += 1
                    if detect(obs["semantic"], obs["depth"], ids, spec):
                        return True
        return False

    def summary(self) -> Dict[str, Any]:
        """Item 1c per category, for the home summary (drop rates per category)."""
        return {c: {"instances": len(self.by_category[c]), "seeable": self.seeable[c],
                    "renders": self.renders.get(c, 0)} for c in self.categories}

    def all_ids(self, category: str) -> List[int]:
        """Every semantic id of the category's goal instances (also those without a usable view point)."""
        return sorted({g.object_id for g in self.sampler.goals if g.category == category})


# --- one drive ----------------------------------------------------------------------------------------------------

@dataclass
class Record:
    """Everything one drive produced (kept in memory; written only if the drive is kept)."""

    jpgs: List[bytes] = field(default_factory=list)
    xy: List[np.ndarray] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    position3: List[np.ndarray] = field(default_factory=list)
    floor_y: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)
    collided: List[bool] = field(default_factory=list)
    motion: List[int] = field(default_factory=list)
    phase: List[int] = field(default_factory=list)
    forced: List[bool] = field(default_factory=list)
    travelled: List[float] = field(default_factory=list)
    mode: List[int] = field(default_factory=list)
    goal: List[Tuple[float, float]] = field(default_factory=list)
    next_point: List[Tuple[float, float]] = field(default_factory=list)
    path_len: List[float] = field(default_factory=list)
    new_choice: List[bool] = field(default_factory=list)
    dropped: List[int] = field(default_factory=list)
    clusters: List[int] = field(default_factory=list)
    teacher_ms: List[float] = field(default_factory=list)
    blacklist_events: List[Tuple[int, float, float, float]] = field(default_factory=list)  # frame, x, y, radius
    seen_events: List[Tuple[int, int]] = field(default_factory=list)
    bump_events: List[Tuple[int, float, float]] = field(default_factory=list)  # frame, x, y of a bumped cell
    safety: List[bool] = field(default_factory=list)  # the step followed the floor-map safety path
    safety_failed: int = 0  # steps where the safety step found no floor-map path (the teacher's path was kept)
    detections: List[Tuple[int, int, int, float, bool]] = field(default_factory=list)  # frame, instance, px, conf, missed
    false_marks: List[Tuple[int, int, float]] = field(default_factory=list)  # frame, cell, value
    outcome: Optional[str] = None  # success / complete / timeout / collision / no_frontier / ... / not_at_viewpoint
    final_geodesic_m: Optional[float] = None  # search drives that ended facing an object: walking distance to a view point
    retarget_turns: int = 0
    kick_done: bool = False
    detours: List[Dict[str, Any]] = field(default_factory=list)
    wall_recovery_steps: int = 0


@dataclass
class Motion:
    """The drive loop's motion state."""

    phase: str
    var: Variations
    kick_left: float
    kick_after: int  # a kick is due from this step on
    detour_after: int  # the next detour is due from this step on
    detours_left: int
    look_turned: float = 0.0  # radians turned in the current look-around (one full turn at most)
    look_at: Optional[Tuple[float, float]] = None  # where the last look-around was
    searching: bool = True  # a search drive (exploration drives look around only at the start)
    chose: bool = False  # the teacher has chosen a goal at least once
    detour: Optional[Dict[str, Any]] = None  # active detour: tracked path, rejoin point, start step, record


class DriveRunnerV2:
    """Drives the explorer teacher in a LimoSim and records frames, map, target layers and the teacher's choices."""

    def __init__(self, robot: LimoSim, cfg: Dict[str, Any]) -> None:
        self.robot, self.cfg = robot, cfg
        s = robot.spec
        self.pursuit_spec = PursuitSpec.from_config(cfg["expert"], s.max_v, s.max_w, s.dt)
        self.map_spec = MapSpec.from_config(cfg["map"])
        self.teacher_spec = TeacherSpec.from_config(cfg["teacher"])
        self.noise_spec = NoiseSpec.from_config(cfg["target_layer"])
        self.jpg_params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg["record"]["jpg_quality"])]
        self.grid: Optional[FloorGrid] = None  # the start island's floor map (kicks / detours need room; not the teacher)

    def pose2d(self) -> Tuple[float, float, float]:
        x, y = to_2d(self.robot.position)
        return float(x), float(y), float(self.robot.yaw)

    def run(self, start: Start, task: Optional[SearchTask], var: Variations, motion_rng: np.random.Generator,
            noise_rng: np.random.Generator, on_step: Optional[Callable[..., None]] = None) -> Tuple[Record, FirstSeenMap, Dict[str, TargetFusion]]:
        """One drive from `start` (see the module docstring). `on_step(t, fmap, fusions, pose)` sees the live state
        after frame t was added (map parity check)."""
        r, lim, mo = self.robot, self.cfg["limits"], self.cfg["motion"]
        low, high = r.pathfinder.get_bounds()
        fmap = FirstSeenMap.around(self.map_spec, low, high)
        fusions = {"clean": TargetFusion(fmap.shape), "noisy": TargetFusion(fmap.shape)}
        rec = Record()
        searching = task is not None
        ids = sorted(task.all_ids) if searching else []
        usable = {inst.object_id: inst for inst in task.instances} if searching else {}
        seen: Dict[int, SeenInstance] = {}
        mstate = Motion(phase="wall_recovery" if start.wall_recovery else "start_turn", var=var,
                        kick_left=abs(math.radians(var.kick_deg)),
                        kick_after=int(motion_rng.integers(mo["kick_after_steps"][0], mo["kick_after_steps"][1] + 1)),
                        detour_after=int(motion_rng.integers(mo["detour_after_steps"][0], mo["detour_after_steps"][1] + 1)),
                        detours_left=var.detours, searching=task is not None)
        tstate = TeacherState()
        bumped = np.zeros(fmap.shape, dtype=bool)
        visited_m = self.teacher_spec.frontier_ignore_trajectory_m  # rules v2 1b (0 = p8_rules.md: no ring)
        visited = np.zeros(fmap.shape, dtype=bool) if visited_m > 0 else None
        self._fmap, self._bumped = fmap, bumped  # the current drive's map and bump cells (_bump_at)
        r.place(np.array(start.position), start.yaw)
        streak = 0
        max_steps = lim["max_steps"] if searching else lim["explore_steps"]
        approached = False
        while True:
            t = len(rec.xy)
            obs = r.observe()
            depth = self._record_frame(rec, fmap, obs)
            if searching:
                cam = r.camera_transform()
                cells = frame_cells(depth, r.spec.hfov_deg, cam, r.floor_y, fmap, self.map_spec, self.noise_spec.max_depth_m)
                found = detect(obs["semantic"], depth, ids, self.noise_spec)
                sc = frame_scores(t, found, cells, fmap.shape, self.noise_spec, noise_rng)
                fusions["clean"].update(t, cells.view, sc.clean)
                fusions["noisy"].update(t, cells.view, sc.noisy)
                for d in sc.detections:
                    rec.detections.append((t, d.instance_id, d.pixels, d.confidence, d.missed))
                if sc.false_mark is not None:
                    rec.false_marks.append((t, sc.false_mark[0], sc.false_mark[1]))
                for inst in sorted(found):
                    if inst in usable and inst not in seen:
                        seen[inst] = usable[inst].seen()
                        rec.seen_events.append((t, inst))
            if on_step is not None:
                on_step(t, fmap, fusions, self.pose2d())
            pose = self.pose2d()
            if visited is not None:
                visited |= visited_mask(fmap.shape, fmap.origin, self.map_spec.resolution, np.array([pose[:2]]), visited_m)
            own = OwnMap(fmap.first_seen["obstacle"].reshape(fmap.shape) >= 0,
                         fmap.first_seen["explored"].reshape(fmap.shape) >= 0, tuple(fmap.origin), self.map_spec.resolution,
                         bumped, visited)
            t0 = time.perf_counter()
            dec, tstate_new = teacher_step(own, pose, tstate, [seen[k] for k in sorted(seen)], self.teacher_spec, searching)
            rec.teacher_ms.append((time.perf_counter() - t0) * 1000.0)
            if len(tstate_new.blacklist) > len(tstate.blacklist):
                for b in tstate_new.blacklist[len(tstate.blacklist):]:
                    rec.blacklist_events.append((t, b[0], b[1], b[2]))
            tstate = tstate_new
            approached = approached or dec.mode in (APPROACH, ARRIVED)
            self._log_teacher(rec, dec, searching, approached)
            mstate.chose = mstate.chose or dec.new_choice
            rec.safety.append(False)
            v, w, forced = self._command(mstate, rec, dec, pose, depth, t, motion_rng)
            if mstate.phase == "end":
                self._log_move(rec, 0.0, 0.0, False, "end", False, 0.0)
                rec.outcome = self._outcome(rec, dec, task)
                return rec, fmap, fusions
            move = r.drive(v, w)
            self._log_move(rec, move.v, move.w, move.collided, mstate.phase, forced, move.travelled_m)
            if mstate.phase == "kick" and rec.kick_done:
                mstate.phase = "drive"
            if move.collided and move.v > 0.0 and move.travelled_m < 0.5 * move.v * r.spec.dt:
                self._bump(rec, fmap, bumped, len(rec.v) - 1)
            streak = streak + 1 if move.collided else 0
            if streak >= lim["max_collision_streak"]:
                rec.outcome = "collision"
                return rec, fmap, fusions
            if len(rec.v) >= max_steps:
                rec.outcome = "timeout" if searching else "complete"
                return rec, fmap, fusions

    def _bump(self, rec: Record, fmap: FirstSeenMap, bumped: np.ndarray, t: int) -> None:
        """A forward move stopped short (collision, < half its length): mark the cell bump_ahead_m ahead as blocked for
        the teacher (OwnMap.bumped)."""
        x, y, yaw = self.pose2d()
        d = self.cfg["motion"]["bump_ahead_m"]
        self._bump_at(rec, (x + d * math.cos(yaw), y + d * math.sin(yaw)))

    def _bump_at(self, rec: Record, xy: Sequence[float]) -> None:
        """Mark the storey cell under xy as bumped (blocked for the teacher) and log it."""
        flat, inside = self._fmap.cells(np.asarray([xy], dtype=np.float64))
        if inside[0] and not self._bumped.flat[flat[0]]:
            self._bumped.flat[flat[0]] = True
            rec.bump_events.append((len(rec.xy) - 1, float(xy[0]), float(xy[1])))

    # --- recording ---------------------------------------------------------------------------------------------
    def _record_frame(self, rec: Record, fmap: FirstSeenMap, obs: Dict[str, np.ndarray]) -> np.ndarray:
        r, t = self.robot, len(rec.xy)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR), self.jpg_params)
        assert ok
        rec.jpgs.append(buf.tobytes())
        x, y, yaw = self.pose2d()
        rec.xy.append(np.array([x, y]))
        rec.yaw.append(yaw)
        rec.position3.append(r.position.copy())
        rec.floor_y.append(r.floor_y)
        cam = r.camera_transform()
        fmap.add(world_points(obs["depth"], r.spec.hfov_deg, self.map_spec.max_depth_m, cam[:3, 3], cam[:3, :3]), r.floor_y, t)
        return obs["depth"]

    @staticmethod
    def _log_teacher(rec: Record, dec: Decision, searching: bool, approached: bool) -> None:
        nan2 = (float("nan"), float("nan"))
        rec.mode.append(MODES.index(dec.mode))
        rec.phase.append(PHASES_V2.index(APPROACH if approached else (SEARCH if searching else EXPLORE)))
        rec.goal.append(tuple(dec.goal) if dec.goal is not None else nan2)
        rec.next_point.append(tuple(dec.next_point) if dec.next_point is not None else nan2)
        rec.path_len.append(float(dec.path_length_m))
        rec.new_choice.append(bool(dec.new_choice))
        rec.dropped.append(DROPPED.index(dec.dropped or ""))
        rec.clusters.append(int(dec.n_clusters))

    @staticmethod
    def _log_move(rec: Record, v: float, w: float, collided: bool, motion: str, forced: bool, travelled: float) -> None:
        rec.v.append(v)
        rec.w.append(w)
        rec.collided.append(bool(collided))
        rec.motion.append(MOTIONS.index(motion))
        rec.forced.append(bool(forced))
        rec.travelled.append(travelled)

    def _outcome(self, rec: Record, dec: Decision, task: Optional[SearchTask]) -> str:
        if task is None:
            return "complete"
        if dec.mode in FAILURE_MODES:
            return dec.mode
        vps = [p for inst in task.instances for p in inst.view_points]
        geo = self.robot.geodesic_to_any(self.robot.position, vps)
        rec.final_geodesic_m = round(float(geo), 3)
        return "success" if geo <= self.cfg["limits"]["success_m"] else "not_at_viewpoint"

    # --- the motion phase machine (one handler per phase, as datagen.DriveRunner) ----------------------------------
    def _command(self, ms: Motion, rec: Record, dec: Decision, pose, depth, t: int,
                 rng: np.random.Generator) -> Tuple[float, float, bool]:
        while ms.phase != "end":
            cmd = getattr(self, f"_m_{ms.phase}")(ms, rec, dec, pose, depth, t, rng)
            if cmd is not None:
                return cmd
        return 0.0, 0.0, False

    def _aim_angle(self, dec: Decision, pose) -> float:
        if dec.next_point is None:
            return 0.0
        a = math.atan2(dec.next_point[1] - pose[1], dec.next_point[0] - pose[0]) - pose[2]
        return wrap(a)

    def _m_wall_recovery(self, ms, rec, dec, pose, depth, t, rng):
        """Turn on the spot toward the path side (left if there is no path yet: a wall start often shows no
        frontier) until the centre depth > clear_depth_m or the path is within path_within_deg."""
        rule, alpha = self.cfg["variations"]["wall_recovery"], self._aim_angle(dec, pose)
        has_path = dec.next_point is not None
        if dec.mode == ARRIVED or centre_depth(depth, rule["centre_px"]) > rule["clear_depth_m"] \
                or (has_path and abs(alpha) <= math.radians(rule["path_within_deg"])):
            ms.phase = "start_turn"
            return None
        rec.wall_recovery_steps += 1
        return 0.0, math.copysign(self.pursuit_spec.max_w, alpha if alpha else 1.0), False

    def _look_due(self, ms, dec, pose) -> bool:
        """Stuck (no frontier / all dropped): look around once per place (>= look_move_m from the last look-around)
        before the drive ends -- "no frontier left" counts only after the robot has looked (p8_rules.md §8)."""
        if dec.mode not in STUCK_MODES or (ms.chose and not ms.searching):  # exploration: mid-drive "stuck" = done
            return False
        if ms.look_at is not None and math.hypot(pose[0] - ms.look_at[0], pose[1] - ms.look_at[1]) < self.cfg["motion"]["look_move_m"]:
            return False
        ms.look_at, ms.look_turned = (pose[0], pose[1]), 0.0
        ms.phase = "look" if ms.chose else "look_start"
        return True

    def _m_look_start(self, ms, rec, dec, pose, depth, t, rng):
        return self._m_look(ms, rec, dec, pose, depth, t, rng)

    def _m_look(self, ms, rec, dec, pose, depth, t, rng):
        """Turn on the spot (left) until the teacher has a frontier >= look_stop_m away by path (a real new option, not
        the scrap next to the robot that left it stuck), the target is seen, or one full turn is done."""
        sp = self.pursuit_spec
        found = dec.mode in (SEARCH, EXPLORE) and dec.path_length_m >= self.cfg["motion"]["look_stop_m"]
        if found or dec.mode in (APPROACH, ARRIVED) or ms.look_turned >= 2.0 * math.pi - 1e-9:
            ms.phase = "start_turn" if ms.phase == "look_start" else "drive"
            return None
        ms.look_turned += sp.max_w * sp.dt
        return 0.0, sp.max_w, False

    def _m_start_turn(self, ms, rec, dec, pose, depth, t, rng):
        """Turn on the spot while the aim point is > start_turn_deg off."""
        sp, alpha = self.pursuit_spec, self._aim_angle(dec, pose)
        if self._look_due(ms, dec, pose):
            return None
        if dec.mode in FAILURE_MODES or dec.mode == ARRIVED or abs(alpha) <= math.radians(sp.start_turn_deg):
            ms.phase = "drive"
            return None
        return 0.0, spot_turn(alpha, sp.max_w, sp.dt), False

    def _m_drive(self, ms, rec, dec, pose, depth, t, rng):
        if self._look_due(ms, dec, pose):
            return None
        if dec.mode in FAILURE_MODES:
            ms.phase = "end"
            return None
        if dec.mode == ARRIVED:
            ms.phase = "face"
            return None
        alpha = self._aim_angle(dec, pose)
        if dec.new_choice and abs(alpha) > math.radians(self.cfg["motion"]["retarget_turn_deg"]):
            ms.phase = "retarget"
            rec.retarget_turns += 1
            return None
        if self._detour_due(ms, dec, pose, t, rng):
            ms.phase = "detour"
            return None
        if self._kick_due(ms, rec, dec, pose, t):
            ms.phase = "kick"
            return None
        c = PurePursuit(self.pursuit_spec, TrackedPath(self._safe_path(rec, dec.path, pose))).command(pose)
        if c.done:
            return 0.0, 0.0, False
        return c.v, c.w, c.forced_spot_turn

    def _safe_path(self, rec: Record, path: np.ndarray, pose) -> np.ndarray:
        """Safety step (p8_rules.md §8): the teacher's path if its next safety_ahead_m keeps safety_clearance_m from
        the floor map's edge; otherwise a short smoothed path on the floor map (navmesh, as Phase 2's motion) to the
        farthest safe point of that stretch. The teacher's choice is unchanged; only how the robot follows it."""
        mo, ex = self.cfg["motion"], self.cfg["expert"]
        if self.grid is None or len(path) < 2:
            return path
        pts = resample(path, 0.05)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        ahead = pts[: int(np.searchsorted(s, mo["safety_ahead_m"])) + 1]
        clear = np.array([self.grid.clearance_at(p) for p in ahead])
        if (clear[1:] >= mo["safety_clearance_m"]).all() and self.grid.clearance_at(pose[:2]) >= mo["safety_robot_clearance_m"]:
            return path
        if self.grid.clearance_at(pose[:2]) < mo["safety_robot_clearance_m"]:  # touching: a bumper would tell
            near = ahead[s[: len(ahead)] <= mo["bump_ahead_m"]]
            blocked = [p for p in near if self.grid.clearance_at(p) <= 0.0]
            if blocked:
                self._bump_at(rec, blocked[0])
        ok = np.nonzero(clear >= mo["safety_clearance_m"])[0]
        k = int(ok[-1]) if ok.size and ok[-1] > 0 else len(ahead) - 1
        local = self.grid.plan_local(pose[:2], ahead[k], mo["safety_limit_m"])
        rest = pts[k + 1:]
        if local is None or len(local) < 2:  # the safe point lies behind something only the floor map shows
            local = self.grid.plan_local(pose[:2], pts[-1], mo["safety_far_limit_m"])
            rest = pts[:0]
        if local is None or len(local) < 2:
            rec.safety_failed += 1
            return path
        rec.safety[-1] = True
        joined = np.concatenate([local, rest]) if len(rest) else local
        return smooth_path(joined, self.grid.clearance_at, ex["smooth_step_m"], ex["smooth_window_m"], ex["smooth_passes"],
                           ex["smooth_min_clearance_m"])

    def _m_retarget(self, ms, rec, dec, pose, depth, t, rng):
        sp, alpha = self.pursuit_spec, self._aim_angle(dec, pose)
        if dec.mode in FAILURE_MODES or dec.mode == ARRIVED or abs(alpha) < math.radians(self.cfg["motion"]["retarget_turn_deg"]):
            ms.phase = "drive"
            return None
        return 0.0, spot_turn(alpha, sp.max_w, sp.dt), False

    def _m_face(self, ms, rec, dec, pose, depth, t, rng):
        if dec.face_yaw is None:  # the teacher no longer says "arrived" (e.g. a new instance): drive on
            ms.phase = "drive"
            return None
        err = wrap(dec.face_yaw - pose[2])
        if abs(err) <= math.radians(self.teacher_spec.face_tolerance_deg):
            ms.phase = "end"
            return None
        sp = self.pursuit_spec
        return 0.0, spot_turn(err, sp.max_w, sp.dt), False

    # kick: as Phase 2, but due after a drawn number of steps (the online path has no fixed length)
    def _kick_due(self, ms, rec, dec, pose, t) -> bool:
        if ms.kick_left <= 0 or rec.kick_done or t < ms.kick_after or len(dec.path) < 2:
            return False
        k = self.cfg["variations"]["kick"]
        path = TrackedPath(resample(dec.path, 0.05))
        return path.length >= k["min_remaining_m"] and self._clearance(pose) >= k["min_clearance_m"] and \
            path.turn_ahead(0.0, k["straight_ahead_m"]) <= math.radians(k["max_turn_ahead_deg"]) and \
            abs(self._aim_angle(dec, pose)) <= math.radians(k["max_heading_error_deg"])

    def _m_kick(self, ms, rec, dec, pose, depth, t, rng):
        sp = self.pursuit_spec
        w = math.copysign(min(sp.max_w, ms.kick_left / sp.dt), ms.var.kick_deg)
        ms.kick_left -= abs(w) * sp.dt
        if ms.kick_left <= 1e-6:
            rec.kick_done = True
        return self.cfg["variations"]["kick"]["v"], w, False

    # detour: a sideways bulge on the floor map (navmesh truth, like Phase 2's), then back to the teacher's path
    def _detour_due(self, ms, dec, pose, t, rng) -> bool:
        if ms.detours_left <= 0 or t < ms.detour_after or dec.mode not in (SEARCH, EXPLORE) or self.grid is None:
            return False
        mo, dc = self.cfg["motion"], self.cfg["variations"]["detour"]
        path = resample(dec.path, 0.05)
        if path_length(path) < mo["detour_ahead_m"] + 0.5:
            return False
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
        ahead = path[: int(np.searchsorted(s, mo["detour_ahead_m"])) + 1]
        via = detour_point(self.grid, ahead, 0.5, dc, rng)
        if via is None:
            ms.detour_after = t + 10  # try again a little later
            return False
        rejoin = ahead[-1]
        pieces = [self.grid.plan(pose[:2], via[0]), self.grid.plan(via[0], rejoin)]
        if any(p is None for p in pieces):
            ms.detour_after = t + 10
            return False
        ex = self.cfg["expert"]
        full = np.concatenate([pieces[0], pieces[1][1:]])
        smooth = smooth_path(full, self.grid.clearance_at, ex["smooth_step_m"], ex["smooth_window_m"], ex["smooth_passes"],
                             ex["smooth_min_clearance_m"])
        if max_turn(smooth, dc["turn_window_m"]) > math.radians(dc["max_turn_deg"]):
            ms.detour_after = t + 10
            return False
        ms.detour = {"path": TrackedPath(smooth), "rejoin": rejoin, "start": t,
                     "record": {"start": t, "offset_m": round(via[1], 3), "via": [round(float(v), 3) for v in via[0]],
                                "deviation_m": round(deviation(smooth, ahead), 3)}}
        ms.detours_left -= 1
        ms.detour_after = t + mo["detour_gap_steps"]
        return True

    def _m_detour(self, ms, rec, dec, pose, depth, t, rng):
        d, mo = ms.detour, self.cfg["motion"]
        near = math.hypot(d["rejoin"][0] - pose[0], d["rejoin"][1] - pose[1]) <= mo["detour_rejoin_m"]
        if near or t - d["start"] >= mo["detour_max_steps"] or dec.mode not in (SEARCH, EXPLORE):
            d["record"].update({"end": t, "rejoined": bool(near)})
            rec.detours.append(d["record"])
            ms.detour = None
            ms.phase = "drive"
            return None
        c = PurePursuit(self.pursuit_spec, d["path"]).command(pose)
        if c.done:
            d["record"].update({"end": t, "rejoined": True})
            rec.detours.append(d["record"])
            ms.detour = None
            ms.phase = "drive"
            return None
        return c.v, c.w, c.forced_spot_turn

    def _clearance(self, pose) -> float:
        return self.grid.clearance_at(pose[:2]) if self.grid is not None else 0.0


# --- writing ------------------------------------------------------------------------------------------------------

def frame_arrays(rec: Record) -> Dict[str, np.ndarray]:
    """mapmad_frames.npz (module docstring)."""
    f32 = (lambda xs: np.asarray(xs, np.float32))
    return {"v": f32(rec.v), "w": f32(rec.w), "collided": np.asarray(rec.collided, bool),
            "motion": np.asarray(rec.motion, np.uint8), "motion_names": np.asarray(MOTIONS),
            "phase": np.asarray(rec.phase, np.uint8), "phase_names": np.asarray(PHASES_V2),
            "teacher_mode": np.asarray(rec.mode, np.uint8), "teacher_mode_names": np.asarray(MODES),
            "forced_spot_turn": np.asarray(rec.forced, bool), "travelled_m": f32(rec.travelled),
            "safety_path": np.asarray(rec.safety, bool),
            "goal_xy": np.asarray(rec.goal, np.float64).reshape(-1, 2),
            "next_point_xy": np.asarray(rec.next_point, np.float64).reshape(-1, 2),
            "path_length_m": f32(rec.path_len), "new_choice": np.asarray(rec.new_choice, bool),
            "dropped": np.asarray(rec.dropped, np.uint8), "dropped_names": np.asarray(DROPPED),
            "frontier_clusters": np.asarray(rec.clusters, np.int32), "teacher_ms": f32(rec.teacher_ms),
            "blacklist_events": np.asarray(rec.blacklist_events, np.float64).reshape(-1, 4),
            "seen_events": np.asarray(rec.seen_events, np.int64).reshape(-1, 2),
            "bump_events": np.asarray(rec.bump_events, np.float64).reshape(-1, 3),
            "position_habitat": np.asarray(rec.position3, np.float64), "floor_y": np.asarray(rec.floor_y, np.float64)}


def target_arrays(rec: Record, fusions: Dict[str, TargetFusion], category: Optional[str]) -> Dict[str, np.ndarray]:
    """mapmad_target.npz: sparse clean / noisy updates (map_v2.target_at) and the raw detections."""
    out: Dict[str, np.ndarray] = {"shape": np.asarray(fusions["clean"].shape, np.int32),
                                  "category": np.asarray(category or "")}
    for name, fus in fusions.items():
        for k, v in fus.updates().items():
            out[f"{name}_{k}"] = v
    det = np.asarray([(d[0], d[1], d[2]) for d in rec.detections], np.int64).reshape(-1, 3)
    out.update({"det_frame": det[:, 0].astype(np.int32), "det_instance": det[:, 1], "det_pixels": det[:, 2].astype(np.int32),
                "det_confidence": np.asarray([d[3] for d in rec.detections], np.float32),
                "det_missed": np.asarray([d[4] for d in rec.detections], bool)})
    fm = np.asarray(rec.false_marks, np.float64).reshape(-1, 3)
    out.update({"false_frame": fm[:, 0].astype(np.int32), "false_cell": fm[:, 1].astype(np.int64),
                "false_value": fm[:, 2].astype(np.float32)})
    return out


def drive_stats(rec: Record) -> Dict[str, Any]:
    """Per-drive numbers for G8 row 3 and the pilot sheet."""
    motions = [MOTIONS[m] for m in rec.motion]
    spot = [is_spot_turn(v, w) for v, w in zip(rec.v, rec.w)]
    by_type: Dict[str, int] = {}
    for s, m in zip(spot, motions):
        if s:
            k = SPOT_TURN_TYPES.get(m, "outside")
            by_type[k] = by_type.get(k, 0) + 1
    step = np.hypot(*np.diff(np.asarray(rec.xy), axis=0).T) if len(rec.xy) > 1 else np.zeros(0)
    phases = [PHASES_V2[p] for p in rec.phase]
    ms = np.asarray(rec.teacher_ms) if rec.teacher_ms else np.zeros(1)
    return {"frames": len(rec.xy), "collision_steps": int(sum(rec.collided)),
            "spot_turn_frames": int(sum(spot)), "spot_turn_frames_by_type": by_type,
            "spot_turn_frames_outside": by_type.get("outside", 0),
            "forced_spot_turn_frames": int(sum(rec.forced)),
            "frames_by_motion": {m: motions.count(m) for m in MOTIONS if motions.count(m)},
            "frames_by_phase": {p: phases.count(p) for p in PHASES_V2 if phases.count(p)},
            "path_m": round(float(step.sum()), 3), "retarget_turns": rec.retarget_turns,
            "new_choices": int(sum(rec.new_choice)), "blacklisted": len(rec.blacklist_events),
            "kick_done": rec.kick_done, "detours": rec.detours,
            "detour_realised": any(d["deviation_m"] >= 0.4 for d in rec.detours),
            "wall_recovery_steps": rec.wall_recovery_steps,
            "teacher_ms_median": round(float(np.median(ms)), 3), "teacher_ms_p95": round(float(np.percentile(ms, 95)), 3),
            "detections": len(rec.detections), "false_marks": len(rec.false_marks), "bumps": len(rec.bump_events),
            "safety_steps": int(sum(rec.safety)), "safety_failed": rec.safety_failed}


def write_drive(folder: Path, rec: Record, fmap: FirstSeenMap, fusions: Dict[str, TargetFusion], meta: Dict[str, Any],
                category: Optional[str]) -> int:
    """Write one drive (module docstring); returns its size in bytes."""
    folder.mkdir(parents=True, exist_ok=False)
    for t, jpg in enumerate(rec.jpgs):
        (folder / f"{t}.jpg").write_bytes(jpg)
    with open(folder / "traj_data.pkl", "wb") as f:
        pickle.dump({"position": np.asarray(rec.xy, np.float64), "yaw": np.asarray(rec.yaw, np.float64)}, f)
    np.savez_compressed(folder / "mapmad_frames.npz", **frame_arrays(rec))
    np.savez_compressed(folder / "mapmad_map.npz", **fmap.arrays(float(np.median(rec.floor_y))))
    np.savez_compressed(folder / TARGET_FILE, **target_arrays(rec, fusions, category))
    (folder / "mapmad_meta.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    return sum(p.stat().st_size for p in folder.iterdir())


# --- one home -----------------------------------------------------------------------------------------------------

class HomeGeneratorV2:
    """All drives of one home: one LimoSim, drives seeded by id (module docstring)."""

    def __init__(self, cfg: Dict[str, Any], home: str, labelled: bool, gpu: int) -> None:
        self.cfg, self.home = cfg, home
        self.spec = robot_spec(cfg)
        goals = objectnav.home_goals(cfg["objectnav_split"], home) if labelled else None
        self.robot = LimoSim(cfg["split"], home, self.spec, semantic=labelled, depth=True, gpu=gpu,
                             seed=drive_seed(cfg["seed"], home) % (2**31))
        self.sampler = DriveSampler(self.robot, cfg, np.random.default_rng(0), goals)
        seeable = NoiseSpec.from_config(cfg["target_layer"]) if cfg["sampling"].get("seeable_filter") else None
        self.tasks = HomeTasks(self.sampler, goals, seeable, cfg["sampling"]["visibility_headings"]) if labelled else None
        self.runner = DriveRunnerV2(self.robot, cfg)
        self.grids: Dict[Tuple[int, float], FloorGrid] = {}

    def close(self) -> None:
        self.robot.close()

    def grid_for(self, point: Sequence[float]) -> FloorGrid:
        island = int(self.robot.pathfinder.get_island(np.asarray(point, np.float32)))
        key = (island, round(float(point[1]), 1))
        if key not in self.grids:
            ex = self.cfg["expert"]
            self.grids[key] = FloorGrid.from_pathfinder(self.robot.pathfinder, point, ex["grid_cell_m"],
                                                        ex["preferred_clearance_m"], ex["clearance_weight"])
        return self.grids[key]

    # --- starts ----------------------------------------------------------------------------------------------------
    def _start(self, var: Variations, notes: Dict[str, Any]) -> Start:
        start = self.sampler.wall_start() if var.wall_recovery else None
        if var.wall_recovery and start is None:
            notes["wall_start_unavailable"] = True
        return start or self.sampler.start()

    def _hidden(self, start: Start, ids: Sequence[int]) -> bool:
        """No instance of the category shows in any of 8 headings at the start (pin025, any pixel)."""
        n = self.cfg["sampling"]["visibility_headings"]
        p = np.array(start.position)
        for k in range(n):
            self.robot.place(p, start.yaw + 2.0 * math.pi * k / n)
            if np.isin(self.robot.observe()["semantic"], ids).any():
                return False
        return True

    def sample_search(self, var: Variations, rng: np.random.Generator, notes: Dict[str, Any]) -> Optional[Tuple[Start, SearchTask]]:
        """Block item 4: category uniform among those with a usable instance; start on the real floor, 3-15 m walking
        to the nearest view point on the start storey, no instance visible in 8 headings. Rules v2 item 1c: a drawn
        category that fails the start filter is dropped (notes `dropped_categories`) and drawn again."""
        s = self.cfg["sampling"]
        if not any(self.tasks.seeable.values()):
            notes["no_seeable_category"] = True
            return None
        cat = self._draw_category(rng, notes)
        lo, hi = s["search_geodesic_m"]
        for k in range(s["start_tries"]):
            if k and k % s["category_redraw_every"] == 0:
                cat = self._draw_category(rng, notes)
            start = self._start(var, notes)
            insts = self.tasks.by_category[cat]
            vps = [vp for inst in insts for vp in inst.view_points if self.sampler.same_floor(start.position, vp)]
            if not vps:
                continue
            geo = self.robot.geodesic_to_any(start.position, vps)
            if not (math.isfinite(geo) and lo <= geo <= hi):
                continue
            if not self._hidden(start, self.tasks.all_ids(cat)):
                notes["visible_starts_rejected"] = notes.get("visible_starts_rejected", 0) + 1
                continue
            return start, SearchTask(cat, insts, round(float(geo), 3), self.tasks.all_ids(cat))
        return None

    def _draw_category(self, rng: np.random.Generator, notes: Dict[str, Any]) -> str:
        """Uniform over the home's categories; one failing the start filter is logged and drawn again."""
        cats = self.tasks.categories
        while True:
            cat = cats[int(rng.integers(len(cats)))]
            if self.tasks.seeable[cat]:
                return cat
            notes.setdefault("dropped_categories", []).append(cat)

    # --- one drive -------------------------------------------------------------------------------------------------
    def drive(self, did: str, kind: str, out: Path, failed_dir: Optional[Path] = None,
              on_step: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
        """Sample and run one drive; returns its summary (written to `out / did` only if kept)."""
        t0, c0 = time.time(), time.process_time()
        seed = drive_seed(self.cfg["seed"], did)
        rng = np.random.default_rng([seed, 0])
        noise_rng = np.random.default_rng([seed, 1])
        self.sampler.rng = rng  # starts and view-point checks draw from this drive's stream
        var = draw_variations(self.cfg, rng)
        notes: Dict[str, Any] = {}
        task = None
        if kind == "search":
            sampled = self.sample_search(var, rng, notes)
            if sampled is None:
                return {"drive_id": did, "home": self.home, "kind": kind, "outcome": "no_sample", "kept": False,
                        "variations": asdict(var), "notes": notes}
            start, task = sampled
        else:
            start = self._start(var, notes)
        self.runner.grid = self.grid_for(start.position)
        rec, fmap, fusions = self.runner.run(start, task, var, rng, noise_rng, on_step)
        stats = drive_stats(rec)
        kept = rec.outcome in ("success", "complete")
        meta = {"drive_id": did, "home": self.home, "split": self.cfg["split"], "kind": kind, "seed": seed,
                "outcome": rec.outcome, "kept": kept, "category": task.category if task else None,
                "instances": [asdict(i) for i in task.instances] if task else [],
                "start_geodesic_m": task.geodesic_m if task else None,
                "final_geodesic_m": rec.final_geodesic_m,
                "navmesh_sha256": self.robot.navmesh_sha256, "floor_y": round(float(np.median(rec.floor_y)), 4),
                "island": int(self.robot.pathfinder.get_island(np.asarray(start.position, np.float32))),
                "start": asdict(start), "variations": asdict(var), "notes": notes,
                "wall_recovery_realised": bool(start.wall_recovery), "camera": self.cfg["camera"],
                "metric_waypoint_spacing": self.cfg["record"]["metric_waypoint_spacing"],
                "frontier_ignore_trajectory_m": self.runner.teacher_spec.frontier_ignore_trajectory_m,
                "frame_convention": "traj_data.pkl: X = -z, Y = -x of Habitat, yaw unchanged (mapmad_sim/frames.py)",
                **stats, "jpg_bytes": sum(len(j) for j in rec.jpgs)}
        folder = out / did if kept else (failed_dir / did if failed_dir is not None else None)
        if folder is not None:
            meta["bytes"] = write_drive(folder, rec, fmap, fusions, meta, task.category if task else None)
        meta["seconds"] = round(time.time() - t0, 3)
        meta["cpu_seconds"] = round(time.process_time() - c0, 3)
        if folder is not None:
            write_text_atomic(folder / "mapmad_meta.json", json.dumps(meta, indent=1, default=float) + "\n")
        return meta


def generate_home(cfg: Dict[str, Any], dataset_dir: Path, home: str, labelled: bool, gpu: int, prefix: str,
                  counts: Dict[str, int], keep_failed: bool, only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """All drives of one home into dataset_dir (resumable as Phase 2: _tmp/<home> -> moved when the home is done;
    _homes/<home>.json marks it finished). `only`: debug, just these drive ids."""
    done = dataset_dir / "_homes" / f"{home}.json"
    if done.exists():
        return json.loads(done.read_text())
    tmp = dataset_dir / "_tmp" / home
    if tmp.exists():  # an interrupted earlier run: our own unfinished output, redone from the same seeds
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    failed_tmp = tmp / "_failed" if keep_failed else None
    t0 = time.time()
    gen = HomeGeneratorV2(cfg, home, labelled, gpu)
    try:
        ids = [(d, k) for d, k in home_drive_ids(prefix, home, counts) if only is None or d in only]
        summaries = [gen.drive(d, k, tmp, failed_tmp) for d, k in ids]
    finally:
        gen.close()
    for d in sorted(p for p in tmp.iterdir() if p.name != "_failed"):
        target = dataset_dir / d.name
        if target.exists():
            raise FileExistsError(f"{target} exists: duplicate drive id")
        os.replace(d, target)
    if failed_tmp is not None and failed_tmp.exists():
        (dataset_dir / "_failed").mkdir(exist_ok=True)
        for d in sorted(failed_tmp.iterdir()):
            os.replace(d, dataset_dir / "_failed" / d.name)
        failed_tmp.rmdir()
    tmp.rmdir()
    summary = {"home": home, "labelled": labelled, "seconds": round(time.time() - t0, 2),
               "navmesh_sha256": gen.robot.navmesh_sha256, "floor_checks": dict(gen.sampler.floor_checks),
               "categories": gen.tasks.summary() if gen.tasks is not None else None,
               "drives": sum(s.get("kept", False) for s in summaries), "summaries": summaries}
    write_text_atomic(done, json.dumps(summary, indent=1, default=float) + "\n")
    return summary
