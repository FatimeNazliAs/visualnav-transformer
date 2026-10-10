"""One practice drive, end to end, and all drives of one home (Phase 2 tasks 2.3-2.6, 2.9).

A drive, step by step (one control step = 0.25 s, one saved frame per step):
1. route: Dijkstra on the start's island (floor_grid.py), through 1-2 detour points for detour drives, then
   smoothed (expert.smooth_path);
2. frame t: colour picture -> t.jpg, pose -> traj_data.pkl, depth picture -> the drive's first-seen map
   (depth_map.py), target pixel share (object drives);
3. command for the next 0.25 s, by phase:
   wall_recovery (wall starts): turn on the spot toward the path side until the centre depth > 1.0 m or the path
     is within 30 deg;  start_turn: turn on the spot while the aim point is > 30 deg off;
   drive: pure pursuit (expert.py);  kick (kick drives, once): an arc at 0.15 m/s turning 20-40 deg off, then
     pure pursuit steers back;  final_turn (object drives): turn on the spot to face the object box (+-10 deg);
4. the LIMO model moves (robot.py, collisions = move > 1 cm short); repeat until the end.
Discard: more steps than (3 x geodesic / max_v + 30 s) / 0.25 s, or >= 20 consecutive collision steps.

Files per drive folder (<home>_<k>): 0.jpg ... N-1.jpg, traj_data.pkl (position (N, 2), yaw (N,), NoMaD's frame),
mapmad_frames.npz (per frame: command after it, phase, collision, ...), mapmad_map.npz (first-seen grids, read by
vint_train/mapmad/local_map.py), mapmad_meta.json (target, start, variations, sha256 of the floor map, ...).
A home's drives are written to <dataset>/_tmp/<home>/ and moved into <dataset>/ when the whole home is done;
<dataset>/_homes/<home>.json marks it finished (a restarted run skips it, an unfinished _tmp is redone).
"""

import json
import math
import os
import pickle
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from mapmad_sim import config, objectnav
from mapmad_sim.camera import centre_depth
from mapmad_sim.depth_map import FirstSeenMap, MapSpec, world_points
from mapmad_sim.drive_sampler import DriveSampler, Start, Target
from mapmad_sim.expert import PursuitSpec, PurePursuit, TrackedPath, is_spot_turn, spot_turn, wrap
from mapmad_sim.floor_grid import FloorGrid
from mapmad_sim.frames import to_2d
from mapmad_sim.route import Route, Variations, draw_variations, plan_route
from mapmad_sim.robot import LimoSim, RobotSpec

PHASES = ("start_turn", "wall_recovery", "drive", "kick", "final_turn", "end")
ALLOWED_SPOT_TURN_PHASES = ("start_turn", "wall_recovery", "final_turn")  # gate G2 row 2
FINAL_TURN_DONE_RAD = math.radians(0.5)


# --- one drive ------------------------------------------------------------------------------------------------

@dataclass
class DriveRecord:
    """Everything one drive produced (kept in memory; written only if the drive is kept)."""

    jpgs: List[bytes] = field(default_factory=list)
    xy: List[np.ndarray] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)
    position3: List[np.ndarray] = field(default_factory=list)
    floor_y: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    w: List[float] = field(default_factory=list)
    collided: List[bool] = field(default_factory=list)
    phase: List[int] = field(default_factory=list)
    forced_spot_turn: List[bool] = field(default_factory=list)
    target_share: List[float] = field(default_factory=list)
    travelled: List[float] = field(default_factory=list)
    discard: Optional[str] = None
    kick_done: bool = False
    wall_recovery_steps: int = 0


class DriveRunner:
    """Drives the expert along a route in a LimoSim and records frames, map and log."""

    def __init__(self, robot: LimoSim, cfg: Dict[str, Any]) -> None:
        self.robot, self.cfg = robot, cfg
        s = robot.spec
        self.pursuit_spec = PursuitSpec.from_config(cfg["expert"], s.max_v, s.max_w, s.dt)
        self.map_spec = MapSpec.from_config(cfg["map"])
        self.jpg_params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg["record"]["jpg_quality"])]
        self.grid: Optional[FloorGrid] = None  # the current drive's planning grid (kick needs its clearance)

    def pose2d(self) -> Tuple[float, float, float]:
        x, y = to_2d(self.robot.position)
        return float(x), float(y), float(self.robot.yaw)

    def record_frame(self, rec: DriveRecord, fmap: FirstSeenMap, object_id: Optional[int]) -> np.ndarray:
        """Save frame len(rec.xy): picture, pose, depth into the map; returns the depth picture."""
        r, t = self.robot, len(rec.xy)
        obs = r.observe()
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR), self.jpg_params)
        assert ok
        rec.jpgs.append(buf.tobytes())
        x, y, yaw = self.pose2d()
        rec.xy.append(np.array([x, y]))
        rec.yaw.append(yaw)
        rec.position3.append(r.position.copy())
        rec.floor_y.append(r.floor_y)
        cam = r.camera_transform()
        fmap.add(world_points(obs["depth"], r.spec.hfov_deg, self.map_spec.max_depth_m, cam[:3, 3], cam[:3, :3]),
                 r.floor_y, t)
        rec.target_share.append(float((obs["semantic"] == object_id).mean()) if object_id is not None else float("nan"))
        return obs["depth"]

    def run(self, start: Start, target: Target, route: Route, var: Variations, geodesic: float) -> Tuple[DriveRecord, FirstSeenMap]:
        r, cfg, sp = self.robot, self.cfg, self.pursuit_spec
        wall_cfg, kick_cfg, disc = cfg["variations"]["wall_recovery"], cfg["variations"]["kick"], cfg["discard"]
        low, high = r.pathfinder.get_bounds()
        fmap = FirstSeenMap.around(self.map_spec, low, high)
        pursuit = PurePursuit(sp, TrackedPath(route.path))
        rec = DriveRecord()
        limit = int(math.ceil((disc["time_factor"] * geodesic / sp.max_v + disc["extra_turn_s"]) / sp.dt))
        grid_clearance = self._grid_clearance
        r.place(np.array(start.position), start.yaw)
        phase = "wall_recovery" if start.wall_recovery else "start_turn"
        kick_left = abs(math.radians(var.kick_deg))
        streak = 0
        while True:
            depth = self.record_frame(rec, fmap, target.object_id)
            pose = self.pose2d()
            v = w = 0.0
            forced = False
            if phase == "wall_recovery":
                alpha = pursuit.angle_to_aim(pose)
                if centre_depth(depth, wall_cfg["centre_px"]) > wall_cfg["clear_depth_m"] or \
                        abs(alpha) <= math.radians(wall_cfg["path_within_deg"]):
                    phase = "start_turn"
                else:
                    w = math.copysign(sp.max_w, alpha)
                    rec.wall_recovery_steps += 1
            if phase == "start_turn":
                alpha = pursuit.angle_to_aim(pose)
                if abs(alpha) <= math.radians(sp.start_turn_deg):
                    phase = "drive"
                else:
                    w = spot_turn(alpha, sp.max_w, sp.dt)
            if phase == "drive" and kick_left > 0 and not rec.kick_done:
                progress = float(pursuit.path.s[pursuit.path.index]) / max(pursuit.path.length, 1e-9)
                remaining = pursuit.path.length - float(pursuit.path.s[pursuit.path.index])
                # only on a straight stretch, lined up with the path, with room around (pilot 4: a kick out of a
                # bend pushed the robot 63 deg off and its way back touched the wall)
                straight = pursuit.path.turn_ahead(float(pursuit.path.s[pursuit.path.index]), kick_cfg["straight_ahead_m"])
                if progress >= var.kick_at and remaining >= kick_cfg["min_remaining_m"] and \
                        grid_clearance(pose) >= kick_cfg["min_clearance_m"] and \
                        straight <= math.radians(kick_cfg["max_turn_ahead_deg"]) and \
                        abs(pursuit.angle_to_aim(pose)) <= math.radians(kick_cfg["max_heading_error_deg"]):
                    phase = "kick"
            if phase == "kick":
                w = math.copysign(min(sp.max_w, kick_left / sp.dt), var.kick_deg)
                v = kick_cfg["v"]
                kick_left -= abs(w) * sp.dt
                if kick_left <= 1e-6:
                    rec.kick_done = True
            elif phase == "drive":
                c = pursuit.command(pose)
                if c.done:
                    phase = "final_turn" if target.final_yaw is not None else "end"
                else:
                    v, w, forced = c.v, c.w, c.forced_spot_turn
            if phase == "final_turn":
                err = wrap(target.final_yaw - pose[2])
                if abs(err) <= FINAL_TURN_DONE_RAD:  # spot_turn lands on the heading; the gate allows +-10 deg
                    phase = "end"
                else:
                    w = spot_turn(err, sp.max_w, sp.dt)
            if phase == "end":
                self._log(rec, 0.0, 0.0, False, "end", False, 0.0)
                return rec, fmap
            move = r.drive(v, w)
            self._log(rec, move.v, move.w, move.collided, phase, forced, move.travelled_m)
            if phase == "kick" and rec.kick_done:
                phase = "drive"
            streak = streak + 1 if move.collided else 0
            if streak >= disc["max_collision_streak"]:
                rec.discard = "collision_streak"
                return rec, fmap
            if len(rec.v) >= limit:
                rec.discard = "time_limit"
                return rec, fmap

    @staticmethod
    def _log(rec: DriveRecord, v: float, w: float, collided: bool, phase: str, forced: bool, travelled: float) -> None:
        rec.v.append(v)
        rec.w.append(w)
        rec.collided.append(bool(collided))
        rec.phase.append(PHASES.index(phase))
        rec.forced_spot_turn.append(bool(forced))
        rec.travelled.append(travelled)

    def _grid_clearance(self, pose: Tuple[float, float, float]) -> float:
        return self.grid.clearance_at(pose[:2]) if self.grid is not None else 0.0


# --- writing ----------------------------------------------------------------------------------------------------

def frame_arrays(rec: DriveRecord, route: Route) -> Dict[str, np.ndarray]:
    """mapmad_frames.npz: per frame t, the command executed after it (the last frame: 0, phase 'end'); plus the
    route the expert followed (route_xy) and the shortest path (shortest_xy), 2D frame, every 5 cm."""
    return {"route_xy": route.path.astype(np.float32), "shortest_xy": route.shortest.astype(np.float32),"v": np.asarray(rec.v, np.float32), "w": np.asarray(rec.w, np.float32),
            "collided": np.asarray(rec.collided, bool), "phase": np.asarray(rec.phase, np.uint8),
            "phase_names": np.asarray(PHASES), "forced_spot_turn": np.asarray(rec.forced_spot_turn, bool),
            "target_share": np.asarray(rec.target_share, np.float32), "travelled_m": np.asarray(rec.travelled, np.float32),
            "position_habitat": np.asarray(rec.position3, np.float64), "floor_y": np.asarray(rec.floor_y, np.float64)}


def drive_stats(rec: DriveRecord) -> Dict[str, Any]:
    """Per-drive numbers for gate G2 row 2 and the pilot report."""
    phases = [PHASES[p] for p in rec.phase]
    spot = [is_spot_turn(v, w) for v, w in zip(rec.v, rec.w)]
    outside = sum(s and p not in ALLOWED_SPOT_TURN_PHASES for s, p in zip(spot, phases))
    step = np.hypot(*np.diff(np.asarray(rec.xy), axis=0).T) if len(rec.xy) > 1 else np.zeros(0)
    return {"frames": len(rec.xy), "collision_steps": int(sum(rec.collided)),
            "spot_turn_frames": int(sum(spot)), "spot_turn_frames_outside": int(outside),
            "forced_spot_turn_frames": int(sum(rec.forced_spot_turn)),
            "frames_by_phase": {p: phases.count(p) for p in PHASES if phases.count(p)},
            "path_m": round(float(step.sum()), 3), "mean_m_per_frame": round(float(step.mean()), 4) if step.size else 0.0,
            "kick_done": rec.kick_done, "wall_recovery_steps": rec.wall_recovery_steps}


def write_drive(folder: Path, rec: DriveRecord, fmap: FirstSeenMap, route: Route, meta: Dict[str, Any]) -> int:
    """Write one kept drive (see the module docstring); returns its size in bytes."""
    folder.mkdir(parents=True, exist_ok=False)
    for t, jpg in enumerate(rec.jpgs):
        (folder / f"{t}.jpg").write_bytes(jpg)
    with open(folder / "traj_data.pkl", "wb") as f:
        pickle.dump({"position": np.asarray(rec.xy, np.float64), "yaw": np.asarray(rec.yaw, np.float64)}, f)
    np.savez_compressed(folder / "mapmad_frames.npz", **frame_arrays(rec, route))
    np.savez_compressed(folder / "mapmad_map.npz", **fmap.arrays(float(np.median(rec.floor_y))))
    (folder / "mapmad_meta.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    return sum(p.stat().st_size for p in folder.iterdir())


# --- one home ---------------------------------------------------------------------------------------------------

def wanted_drives(labelled: bool, per_home: Dict[str, Dict[str, int]], rng: np.random.Generator) -> List[str]:
    counts = per_home["labelled" if labelled else "unlabelled"]
    kinds = [k for k, n in sorted(counts.items()) for _ in range(n)]
    return [kinds[i] for i in rng.permutation(len(kinds))]


class HomeGenerator:
    """All drives of one home: sample, plan, drive, keep or discard (see the module docstring)."""

    def __init__(self, cfg: Dict[str, Any], home: str, home_index: int, labelled: bool, gpu: int) -> None:
        self.cfg, self.home, self.labelled = cfg, home, labelled
        self.seed = int(cfg["seed"]) + home_index
        self.rng = np.random.default_rng(self.seed)
        self.spec = RobotSpec.from_config(config.robot())
        with_goals = labelled and objectnav.has_goals(cfg["objectnav_split"], home)
        goals = objectnav.home_goals(cfg["objectnav_split"], home) if with_goals else None
        self.robot = LimoSim(cfg["split"], home, self.spec, semantic=labelled, depth=True, gpu=gpu, seed=self.seed)
        self.sampler = DriveSampler(self.robot, cfg, self.rng, goals)
        self.runner = DriveRunner(self.robot, cfg)
        self.grids: Dict[Tuple[int, float], FloorGrid] = {}
        self.goal_cache: Dict[int, Any] = {}

    def close(self) -> None:
        self.robot.close()

    def grid_for(self, point: List[float]) -> FloorGrid:
        island = int(self.robot.pathfinder.get_island(np.asarray(point, np.float32)))
        key = (island, round(float(point[1]), 1))
        if key not in self.grids:
            ex = self.cfg["expert"]
            self.grids[key] = FloorGrid.from_pathfinder(self.robot.pathfinder, point, ex["grid_cell_m"],
                                                        ex["preferred_clearance_m"], ex["clearance_weight"])
        return self.grids[key]

    def sample(self, kind: str, var: Variations) -> Optional[Tuple[Start, Target, float, Dict[str, Any]]]:
        """(start, target, geodesic, notes) for one drive, or None if nothing fits after the configured tries."""
        notes: Dict[str, Any] = {}
        goal = self.sampler.pick_goal() if kind == "object" else None
        for k in range(self.cfg["sampling"]["start_tries"]):
            start = self.sampler.wall_start() if var.wall_recovery else None
            if var.wall_recovery and start is None:
                notes["wall_start_unavailable"] = True
            start = start or self.sampler.start()
            if kind == "spot":
                found = self.sampler.spot_target(start)
            else:
                if k and k % 20 == 0:  # this instance is hard to reach from random starts: another one
                    goal = self.sampler.pick_goal()
                found = self.sampler.object_target(goal, start, self.goal_cache)
            if found is not None:
                target, geo = found
                return start, target, geo, notes
        return None

    def drive(self, drive_id: str, kind: str, var: Variations, out: Path) -> Dict[str, Any]:
        """Sample and run one drive (redrawn after a discard, up to attempts_per_drive); returns its summary."""
        attempts: List[Dict[str, Any]] = []
        for attempt in range(self.cfg["discard"]["attempts_per_drive"]):
            t0, c0 = time.time(), time.process_time()
            sampled = self.sample(kind, var)
            if sampled is None:
                attempts.append({"discard": "no_sample"})
                continue
            start, target, geo, notes = sampled
            start.visibility = (self.sampler.object_visibility(start, target.object_id) if kind == "object"
                                else self.sampler.spot_visibility(start, target))
            grid = self.grid_for(start.position)
            route = plan_route(grid, to_2d(start.position), to_2d(target.end_position), var.detours, self.cfg, self.rng)
            if route is None:
                attempts.append({"discard": "no_route"})
                continue
            self.runner.grid = grid
            rec, fmap = self.runner.run(start, target, route, var, geo)
            stats = drive_stats(rec)
            if rec.discard:
                attempts.append({"discard": rec.discard, **stats})
                continue
            meta = {"drive_id": drive_id, "home": self.home, "split": self.cfg["split"], "kind": kind,
                    "home_seed": self.seed, "attempt": attempt, "discarded_attempts": attempts,
                    "navmesh_sha256": self.robot.navmesh_sha256, "floor_y": round(float(np.median(rec.floor_y)), 4),
                    "island": int(self.robot.pathfinder.get_island(np.asarray(start.position, np.float32))),
                    "geodesic_m": round(geo, 3), "start": asdict(start), "target": asdict(target),
                    "variations": asdict(var), "detour_points_2d": route.detour_points,
                    "detour_offsets_m": route.detour_offsets_m, "max_deviation_m": round(route.max_deviation_m, 3),
                    "detour_realised": route.max_deviation_m >= self.cfg["variations"]["detour"]["realised_m"],
                    "wall_recovery_realised": bool(start.wall_recovery), "notes": notes,
                    "end_error_m": round(float(np.hypot(*(rec.xy[-1] - to_2d(target.end_position)))), 3),
                    "final_heading_error_deg": None if target.final_yaw is None else
                    round(math.degrees(abs(wrap(target.final_yaw - rec.yaw[-1]))), 2),
                    "metric_waypoint_spacing": self.cfg["record"]["metric_waypoint_spacing"],
                    "frame_convention": "traj_data.pkl: X = -z, Y = -x of Habitat, yaw unchanged (mapmad_sim/frames.py)",
                    **stats}
            meta["jpg_bytes"] = sum(len(j) for j in rec.jpgs)
            meta["bytes"] = write_drive(out / drive_id, rec, fmap, route, meta)
            meta["seconds"] = round(time.time() - t0, 3)
            meta["cpu_seconds"] = round(time.process_time() - c0, 3)
            (out / drive_id / "mapmad_meta.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
            return meta
        return {"drive_id": drive_id, "home": self.home, "kind": kind, "missing": True, "discarded_attempts": attempts,
                "variations": asdict(var)}

    def run(self, out: Path, drives_per_home: Dict[str, Dict[str, int]], max_drives: Optional[int] = None) -> Dict[str, Any]:
        t0 = time.time()
        kinds = wanted_drives(bool(self.sampler.goals), drives_per_home, self.rng)[:max_drives]
        summaries = []
        for k, kind in enumerate(kinds):
            var = draw_variations(self.cfg, self.rng)
            summaries.append(self.drive(f"{self.home}_{k:03d}", kind, var, out))
        discards = sum(len(s.get("discarded_attempts", [])) for s in summaries)
        tries = discards + sum(not s.get("missing") for s in summaries)
        return {"home": self.home, "labelled": self.labelled, "seed": self.seed,
                "navmesh_sha256": self.robot.navmesh_sha256, "seconds": round(time.time() - t0, 2),
                "drives": len([s for s in summaries if not s.get("missing")]), "missing": [s["drive_id"] for s in summaries if s.get("missing")],
                "discards": discards, "discard_share": round(discards / max(tries, 1), 3),
                "flag_many_discards": discards / max(tries, 1) > self.cfg["discard"]["flag_home_share"],
                "floor_checks": dict(self.sampler.floor_checks),
                "summaries": summaries}


def generate_home(cfg: Dict[str, Any], dataset_dir: Path, home: str, home_index: int, labelled: bool, gpu: int,
                  drives_per_home: Dict[str, Dict[str, int]], max_drives: Optional[int] = None) -> Dict[str, Any]:
    """Generate one home into dataset_dir (resumable; see the module docstring). Returns the home summary."""
    done = dataset_dir / "_homes" / f"{home}.json"
    if done.exists():
        return json.loads(done.read_text())
    tmp = dataset_dir / "_tmp" / home
    if tmp.exists():  # an interrupted earlier run: our own unfinished output, redone from the same seed
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    gen = HomeGenerator(cfg, home, home_index, labelled, gpu)
    try:
        summary = gen.run(tmp, drives_per_home, max_drives)
    finally:
        gen.close()
    for d in sorted(tmp.iterdir()):
        target = dataset_dir / d.name
        if target.exists():  # left over from a run interrupted between the moves below
            shutil.rmtree(target)
        os.replace(d, target)
    tmp.rmdir()
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return summary
