"""The v2 search map built online while a POLICY drives (Phase 11 item 11.1; Phase 12 reuses it): exactly the map the
Phase 8 generator built while the explorer teacher drove, so the network sees in closed loop what it saw in training.

Every control step, at frame t (`SearchMap.add_frame`, the generator's order, datagen_v2.DriveRunnerV2.run):
1. the depth picture -> the whole-storey first-seen map (depth_map.FirstSeenMap, Phase 2/8 builder: band
   0.05-0.55 m above the real floor, max 4 m, invalid = unknown);
2. the semantic picture -> simulator-truth detections of the episode's category (target_layer.detect: >= 200 px at
   <= 4 m) -> clean and noisy scores (target_layer.frame_scores: U(0.4, 1.0) per instance and frame, miss 0.2, false
   mark 0.05) -> both fusions (map_v2.TargetFusion); the noise comes from the episode's own random stream (seeded by
   episode id + seed index, so every arm sees the same noise); usable instances detected join the sticky `seen` set;
3. `windows(pose)` = the network's input [2, 4, 64, 64] (near 10 cm, far 40 cm; obstacle, explored, frontier, noisy
   target) with map_v2.near_window / far_window; frontier = the teacher's plain frontier INCLUDING the cells bumped
   before t (map_v2.frontier_clusters(obstacle | bumped, explored)), = map_v2.teacher_frontier_at on the saved drive.
After the move (`after_move`): a forward move stopped short (collision, < half its length) marks the cell
bump_ahead_m (0.3 m) ahead as bumped (datagen_v2.DriveRunnerV2._bump), first used at frame t + 1. The generator's
second bump source (its floor-map safety step) belongs to the teacher's motion and has no student counterpart.

With `record=True` the whole episode is kept in a datagen_v2.Record and `write` saves it in the Phase 8 drive format
(datagen_v2.write_drive: frames, traj_data.pkl, mapmad_frames.npz, mapmad_map.npz, mapmad_target.npz,
mapmad_meta.json), so the NoMaD loader, the frontier cache and the labeller read a student drive like a teacher drive.
Phase 12 (search_exam.py): `source` says what draws the NOISY layer -- truth_noisy (the above, Phase 11 exactly),
recogniser (a Phase 9 model through the sidecar, every frame, its detections into the same fusion) or none (an empty
layer, no truth detection either: exploration and the fisheye arm); the clean layer stays simulator truth.

Student drives: `teacher_mode` = search on every frame (no teacher drove; the frontier cache then checks every frame),
`frontier_clusters` = the live cluster count, `phase` = search until a usable instance is first seen, approach after.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from mapmad_sim.datagen_v2 import MODES, MOTIONS, PHASES_V2, Instance, Record, write_drive
from mapmad_sim.depth_map import FirstSeenMap, MapSpec, world_points
from mapmad_sim.explorer import APPROACH, SEARCH, SeenInstance
from mapmad_sim.frames import to_2d
from mapmad_sim.target_layer import NoiseSpec, detect, frame_cells, frame_scores
from vint_train.mapmad.map_v2 import TargetFusion, far_window, frontier_clusters, near_window

Pose = Sequence[float]


@dataclass(frozen=True)
class SearchSpec:
    """The numbers of the online map (Phase 8 config blocks): builder, target noise, bump rule, windows."""

    map_spec: MapSpec
    noise: NoiseSpec
    bump_ahead_m: float = 0.3
    size: int = 64
    near_m: float = 0.1
    far_m: float = 0.4
    min_cluster: int = 4
    jpg_quality: int = 90

    @classmethod
    def from_wire(cls, d: Dict[str, Any]) -> "SearchSpec":
        """From the `search_v2` settings sent at reset (sidecar_server; built by search_wire.search_settings)."""
        return cls(MapSpec.from_config(d["builder"]), NoiseSpec.from_config(d["target_layer"]),
                   float(d["bump_ahead_m"]), int(d["size"]), float(d["near_m"]), float(d["far_m"]),
                   int(d["min_cluster"]), int(d["jpg_quality"]))


def instances_of(episode: Dict[str, Any]) -> List[Instance]:
    """The episode's usable target instances (target.instances: object id, Habitat box centre/size, snapped view
    points), as datagen_v2.Instance."""
    return [Instance(int(i["object_id"]), list(i["centre"]), list(i["size"]), [list(p) for p in i["view_points"]])
            for i in episode["target"]["instances"]]


class SearchMap:
    """The live map v2 of one search episode (module docstring)."""

    def __init__(self, robot: Any, spec: SearchSpec, category_ids: Sequence[int], instances: Sequence[Instance],
                 noise_rng: np.random.Generator, record: bool = False, source: str = "truth_noisy",
                 recogniser: Optional[Any] = None, prompt: Optional[str] = None) -> None:
        low, high = robot.pathfinder.get_bounds()
        self.spec = spec
        self.fmap = FirstSeenMap.around(spec.map_spec, low, high)
        self.fusions = {"clean": TargetFusion(self.fmap.shape), "noisy": TargetFusion(self.fmap.shape)}
        self.bumped = np.zeros(self.fmap.shape, dtype=bool)
        self.ids = sorted(int(i) for i in category_ids)
        self.usable = {inst.object_id: inst for inst in instances}
        self.seen: Dict[int, SeenInstance] = {}
        self.rng = noise_rng
        self.t = -1
        self.in_view = False  # a category instance passes the "seen" rule in the current frame (any instance)
        self.first_in_view: Optional[int] = None  # first such frame (search / approach split)
        self.n_clusters = 0
        self.rec: Optional[Record] = Record() if record else None
        self.jpg_params = [cv2.IMWRITE_JPEG_QUALITY, spec.jpg_quality]
        # Phase 12 (search_exam.py): what draws the noisy layer; truth_noisy = Phase 11 exactly
        self.source, self.recogniser, self.prompt = source, recogniser, prompt
        self.n_dets = 0  # detections drawn into the noisy layer this frame (recogniser: kept detections)

    # --- frame t --------------------------------------------------------------------------------------------------
    def add_frame(self, robot: Any, obs: Dict[str, np.ndarray]) -> int:
        """Frame t + 1 at the robot's current pose (module docstring, steps 1-2); returns its index."""
        self.t += 1
        t, depth = self.t, obs["depth"]
        cam = robot.camera_transform()
        self.fmap.add(world_points(depth, robot.spec.hfov_deg, self.spec.map_spec.max_depth_m, cam[:3, 3], cam[:3, :3]),
                      robot.floor_y, t)
        cells = frame_cells(depth, robot.spec.hfov_deg, cam, robot.floor_y, self.fmap, self.spec.map_spec,
                            self.spec.noise.max_depth_m)
        found = detect(obs["semantic"], depth, self.ids, self.spec.noise) if self.source != "none" else {}
        sc = frame_scores(t, found, cells, self.fmap.shape, self.spec.noise, self.rng)
        noisy = sc.noisy
        self.n_dets = sum(not d.missed for d in sc.detections) + (sc.false_mark is not None)
        if self.source == "recogniser":  # Phase 12: the recogniser's detections, through the same fusion
            from mapmad_sim.recogniser import frame_scores as recogniser_scores

            dets = self.recogniser.detect(obs["rgb"], depth, self.prompt)
            noisy = recogniser_scores(dets, cells, depth.shape)
            self.n_dets = len(dets)
        elif self.source == "none":  # no target layer at all (no false marks either)
            noisy, self.n_dets = {}, 0
        self.fusions["clean"].update(t, cells.view, sc.clean)
        self.fusions["noisy"].update(t, cells.view, noisy)
        self.in_view = bool(found)
        if found and self.first_in_view is None:
            self.first_in_view = t
        new_seen = [inst for inst in sorted(found) if inst in self.usable and inst not in self.seen]
        for inst in new_seen:
            self.seen[inst] = self.usable[inst].seen()
        if self.rec is not None:
            self._record_frame(robot, obs, sc, new_seen)
        return t

    def layers(self) -> Dict[str, np.ndarray]:
        """Whole-storey layers now: obstacle, explored, frontier (bool), noisy target (float32)."""
        obstacle = (self.fmap.first_seen["obstacle"] >= 0).reshape(self.fmap.shape)
        explored = (self.fmap.first_seen["explored"] >= 0).reshape(self.fmap.shape)
        labels = frontier_clusters(obstacle | self.bumped, explored, self.spec.min_cluster)
        self.n_clusters = int(labels.max())
        return {"obstacle": obstacle, "explored": explored, "frontier": labels > 0,
                "target": self.fusions["noisy"].grid()}

    def windows(self, pose: Pose) -> np.ndarray:
        """float32 [2, 4, size, size]: near and far windows of the four layers at the robot's pose."""
        lay = self.layers()
        res = self.spec.map_spec.resolution
        near = near_window(lay, self.fmap.origin, res, tuple(pose), self.spec.size, self.spec.near_m)
        far = far_window(lay, self.fmap.origin, res, tuple(pose), self.spec.size, self.spec.far_m)
        if self.rec is not None:
            self.rec.clusters.append(self.n_clusters)
        return np.stack([near, far]).astype(np.float32)

    # --- after the move -------------------------------------------------------------------------------------------
    def after_move(self, move: Any, pose: Pose, dt: float) -> None:
        """The move that followed frame t: bump rule (module docstring) and, when recording, the move itself."""
        if move.collided and move.v > 0.0 and move.travelled_m < 0.5 * move.v * dt:
            d = self.spec.bump_ahead_m
            self.bump_at((pose[0] + d * math.cos(pose[2]), pose[1] + d * math.sin(pose[2])))
        if self.rec is not None:
            self._record_move(move.v, move.w, move.collided, move.travelled_m)

    def bump_at(self, xy: Sequence[float]) -> None:
        """Mark the storey cell under xy as bumped (a planning obstacle: frontier only, never the obstacle layer)."""
        flat, inside = self.fmap.cells(np.asarray([xy], dtype=np.float64))
        if inside[0] and not self.bumped.flat[flat[0]]:
            self.bumped.flat[flat[0]] = True
            if self.rec is not None:
                self.rec.bump_events.append((self.t, float(xy[0]), float(xy[1])))

    # --- recording ------------------------------------------------------------------------------------------------
    def _record_frame(self, robot: Any, obs: Dict[str, np.ndarray], sc: Any, new_seen: List[int]) -> None:
        rec, t = self.rec, self.t
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR), self.jpg_params)
        assert ok
        rec.jpgs.append(buf.tobytes())
        x, y = to_2d(robot.position)
        rec.xy.append(np.array([float(x), float(y)]))
        rec.yaw.append(float(robot.yaw))
        rec.position3.append(robot.position.copy())
        rec.floor_y.append(robot.floor_y)
        for d in sc.detections:
            rec.detections.append((t, d.instance_id, d.pixels, d.confidence, d.missed))
        if sc.false_mark is not None:
            rec.false_marks.append((t, sc.false_mark[0], sc.false_mark[1]))
        for inst in new_seen:
            rec.seen_events.append((t, inst))
        nan2 = (float("nan"), float("nan"))
        rec.mode.append(MODES.index(SEARCH))
        rec.phase.append(PHASES_V2.index(APPROACH if self.seen else SEARCH))
        rec.goal.append(nan2)
        rec.next_point.append(nan2)
        rec.path_len.append(float("nan"))
        rec.new_choice.append(False)
        rec.dropped.append(0)
        rec.teacher_ms.append(0.0)
        rec.ring_fallback.append(False)
        rec.safety.append(False)

    def _record_move(self, v: float, w: float, collided: bool, travelled: float) -> None:
        rec = self.rec
        rec.v.append(float(v))
        rec.w.append(float(w))
        rec.collided.append(bool(collided))
        rec.motion.append(MOTIONS.index("drive"))
        rec.forced.append(False)
        rec.travelled.append(float(travelled))

    def close_record(self) -> None:
        """The last frame gets the generator's "end" move (v = w = 0), so every per-frame array has one entry per
        frame; a frame whose windows were never asked for (none in the runner) repeats the last cluster count."""
        rec = self.rec
        while len(rec.v) < len(rec.xy):
            rec.v.append(0.0)
            rec.w.append(0.0)
            rec.collided.append(False)
            rec.motion.append(MOTIONS.index("end"))
            rec.forced.append(False)
            rec.travelled.append(0.0)
        while len(rec.clusters) < len(rec.xy):
            rec.clusters.append(rec.clusters[-1] if rec.clusters else 0)

    def write(self, folder: Path, meta: Dict[str, Any], category: str) -> int:
        """Save the recorded episode in the Phase 8 drive format; returns its size in bytes."""
        if self.rec is None:
            raise RuntimeError("this episode was not recorded (record=False)")
        self.close_record()
        return write_drive(Path(folder), self.rec, self.fmap, self.fusions, meta, category)
