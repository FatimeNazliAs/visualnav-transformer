"""The virtual LIMO in Habitat: LIMO's camera on a differential-drive base that moves on the navmesh.

Frames: Habitat world, y up. The robot's pose is its turning centre (LIMO base_link) on the navmesh plus a
heading `yaw` (rotation about +y; yaw 0 looks along -z, positive yaw turns left). The camera sits
`forward_of_base_m` in front of the turning centre, `height_m` above the REAL floor (the navmesh floats
above it, so a 4x4 depth camera under the robot measures the floor at every pose).

Floor map: HM3D's baked navmesh is for a 1.5 m tall agent (space under tables counts as blocked). It is rebuilt
at load with LIMO's size (robot_limo.yaml `navmesh`: radius, height, max step, cell height); its sha256 is kept.

One command (v, w), v clipped to [0, max_v] and |w| to max_w, is held for 1 / control_hz seconds and integrated
exactly along its arc in `substeps` moves. Each move goes through the navmesh (`try_step`: slides along walls).
The command is a collision if its moves together came up more than collision_tol_m short of the intended
(chord) lengths. Only if the camera sits ahead of the body radius, moves that would push it into a wall are
refused too (a refused move still turns on the spot if the turn alone keeps the camera free).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

FLOOR_PROBE = "floor_probe"
FLOOR_LABEL = "floor_label"
UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class RobotSpec:
    """Camera and motion numbers of the virtual LIMO (from configs/robot_limo.yaml)."""

    hfov_deg: float
    width: int
    height: int
    camera_height_m: float  # above the real floor
    camera_forward_m: float  # in front of the turning centre
    pitch_deg: float  # + = up
    max_v: float
    max_w: float
    control_hz: float
    substeps: int
    collision_tol_m: float
    camera_wall_margin_m: float
    probe_above_feet_m: float
    probe_hfov_deg: float
    probe_max_m: float
    navmesh: Tuple[Tuple[str, Any], ...] = ()  # robot_limo.yaml `navmesh` entries (hashable)

    @classmethod
    def from_config(cls, robot: Dict[str, Any], hfov_deg: Optional[float] = None) -> "RobotSpec":
        """Spec from the robot_limo.yaml dict; hfov_deg overrides the measured FOV (e.g. 120 for the FOV check)."""
        cam, drive, sim = robot["camera"], robot["drive"], robot["sim"]
        return cls(hfov_deg=float(hfov_deg or cam["hfov_deg"]), width=int(sim["render_width"]),
                   height=int(sim["render_height"]), camera_height_m=float(cam["height_m"]),
                   camera_forward_m=float(cam["forward_of_base_m"]), pitch_deg=float(cam["sim_pitch_deg"]),
                   max_v=float(drive["max_v_mps"]), max_w=float(drive["max_w_radps"]),
                   control_hz=float(sim["control_hz"]), substeps=int(sim["substeps"]),
                   collision_tol_m=float(sim["collision_tol_m"]),
                   camera_wall_margin_m=float(sim["camera_wall_margin_m"]), probe_above_feet_m=float(sim["floor_probe_above_feet_m"]),
                   probe_hfov_deg=float(sim["floor_probe_hfov_deg"]), probe_max_m=float(sim["floor_probe_max_m"]),
                   navmesh=tuple(sorted(robot["navmesh"].items())))

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz


@dataclass(frozen=True)
class Move:
    """What one (v, w) command did."""

    v: float  # command after clipping to the speed limits
    w: float
    travelled_m: float  # distance the turning centre moved
    short_m: float  # commanded distance that was not driven (wall, sliding)
    refused_substeps: int  # sub-moves refused because the camera would hit a wall
    collided: bool


def forward(yaw: float) -> np.ndarray:
    """Unit vector the robot faces for heading yaw (Habitat: yaw 0 looks along -z)."""
    return np.array([-math.sin(yaw), 0.0, -math.cos(yaw)])


def wrap_angle(a: float) -> float:
    """Angle in (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_towards(src: Sequence[float], dst: Sequence[float]) -> float:
    """Heading that points the robot from src to dst (horizontal)."""
    d = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    return math.atan2(-d[0], -d[2])


def horizontal_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(a[0] - b[0], a[2] - b[2]))


def clip_command(spec: RobotSpec, v: float, w: float) -> Tuple[float, float]:
    """Speed limits: v in [0, max_v] (no reversing, as NoMaD's controller), |w| <= max_w."""
    return float(np.clip(v, 0.0, spec.max_v)), float(np.clip(w, -spec.max_w, spec.max_w))


def arc_step(yaw: float, v: float, w: float, dt: float) -> np.ndarray:
    """Exact displacement of a unicycle holding (v, w) for dt from heading yaw (Habitat frame)."""
    if abs(w) < 1e-9:
        return v * dt * forward(yaw)
    y1 = yaw + w * dt
    return (v / w) * np.array([math.cos(y1) - math.cos(yaw), 0.0, -(math.sin(y1) - math.sin(yaw))])


def navmesh_settings(navmesh: Dict[str, Any]) -> Any:
    """habitat_sim.NavMeshSettings from robot_limo.yaml `navmesh` (other fields: habitat's defaults)."""
    import habitat_sim

    s = habitat_sim.NavMeshSettings()
    s.set_defaults()
    s.agent_radius, s.agent_height = navmesh["agent_radius_m"], navmesh["agent_height_m"]
    s.agent_max_climb, s.cell_height = navmesh["agent_max_climb_m"], navmesh["cell_height_m"]
    s.cell_size, s.include_static_objects = navmesh["cell_size_m"], bool(navmesh["include_static_objects"])
    return s


def navmesh_sha256(pathfinder: Any) -> str:
    """Fingerprint of a navmesh: sha256 of its triangle vertices (float32, in order)."""
    import hashlib

    return hashlib.sha256(np.asarray(pathfinder.build_navmesh_vertices(), dtype=np.float32).tobytes()).hexdigest()


class LimoSim:
    """One HM3D home with the virtual LIMO in it. Use as a context manager or call close()."""

    def __init__(self, split: str, home: str, spec: RobotSpec, semantic: bool = False, depth: bool = False,
                 gpu: int = 0, seed: int = 0) -> None:
        import habitat_sim
        from habitat_starter.sim import make_cfg

        from mapmad_sim import config

        self.split, self.home, self.spec = split, home, spec
        settings = config.hm3d_sim_settings(split, home, width=spec.width, height=spec.height, hfov=spec.hfov_deg,
                                            sensor_height=spec.camera_height_m, depth_sensor=depth,
                                            semantic_sensor=semantic, gpu_device_id=gpu, seed=seed)
        cfg = make_cfg(settings)
        probe = habitat_sim.CameraSensorSpec()
        probe.uuid, probe.sensor_type = FLOOR_PROBE, habitat_sim.SensorType.DEPTH
        probe.resolution, probe.hfov = [4, 4], spec.probe_hfov_deg
        probe.position = [0.0, spec.probe_above_feet_m, 0.0]
        probe.orientation = [-math.pi / 2, 0.0, 0.0]  # straight down
        cfg.agents[0].sensor_specifications.append(probe)
        if semantic:  # what surface the robot stands on (floor, rug, bed, stairs ...)
            label = habitat_sim.CameraSensorSpec()
            label.uuid, label.sensor_type = FLOOR_LABEL, habitat_sim.SensorType.SEMANTIC
            label.resolution, label.hfov = probe.resolution, probe.hfov
            label.position, label.orientation = probe.position, probe.orientation
            cfg.agents[0].sensor_specifications.append(label)
        self.sim = habitat_sim.Simulator(cfg)
        self.sim.initialize_agent(0)
        self.sim.seed(seed)
        self.pathfinder = self.sim.pathfinder
        if spec.navmesh:  # LIMO-sized floor map instead of HM3D's baked one
            if not self.sim.recompute_navmesh(self.pathfinder, navmesh_settings(dict(spec.navmesh))):
                raise RuntimeError(f"{home}: navmesh recompute failed")
        self.navmesh_sha256 = navmesh_sha256(self.pathfinder)
        self.free_radius = float(self.pathfinder.nav_mesh_settings.agent_radius) - spec.camera_wall_margin_m
        self.check_camera = spec.camera_forward_m > self.free_radius  # camera ahead of the body radius
        self.camera_uuids = [s.uuid for s in cfg.agents[0].sensor_specifications if s.uuid not in (FLOOR_PROBE, FLOOR_LABEL)]
        self.position = np.zeros(3)
        self.yaw = 0.0
        self.floor_below_feet = float("nan")  # real floor this far below the navmesh point (m)
        self.floor_probe_failures = 0

    # --- pose -------------------------------------------------------------------------------------------------

    def place(self, position: Sequence[float], yaw: float) -> None:
        """Put the turning centre on `position` (a navmesh point) facing `yaw`; raises if no floor is found below."""
        self.floor_below_feet = float("nan")
        self._set_pose(np.asarray(position, dtype=np.float64), yaw)
        if math.isnan(self.floor_below_feet):
            raise ValueError(f"{self.home}: no floor found under {np.round(position, 3).tolist()}")

    def _set_pose(self, position: np.ndarray, yaw: float) -> None:
        from habitat_sim.utils.common import quat_from_angle_axis

        self.position, self.yaw = position, wrap_angle(yaw)
        agent = self.sim.get_agent(0)
        state = agent.get_state()
        state.position = self.position.astype(np.float32)
        state.rotation = quat_from_angle_axis(self.yaw, UP)
        agent.set_state(state, reset_sensors=False)
        self._measure_floor()
        self._place_cameras()

    def _measure_floor(self) -> None:
        """Real floor height under the turning centre from the downward 4x4 depth camera."""
        sensor = self.sim._sensors[FLOOR_PROBE]
        sensor.draw_observation()
        depth = float(np.median(np.asarray(sensor.get_observation())))
        if 0.0 < depth <= self.spec.probe_max_m:
            self.floor_below_feet = depth - self.spec.probe_above_feet_m
        else:  # hole in the mesh: keep the last good height
            self.floor_probe_failures += 1

    def _place_cameras(self) -> None:
        import magnum as mn

        if math.isnan(self.floor_below_feet):  # no floor measured yet: place() raises, cameras stay where they were
            return
        above_feet = self.spec.camera_height_m - self.floor_below_feet
        for uuid in self.camera_uuids:
            node = self.sim._sensors[uuid]._sensor_object.node
            node.translation = mn.Vector3(0.0, above_feet, -self.spec.camera_forward_m)
            node.rotation = mn.Quaternion.rotation(mn.Deg(self.spec.pitch_deg), mn.Vector3.x_axis())

    @property
    def camera_height_above_floor(self) -> float:
        """Check value: should equal spec.camera_height_m."""
        node = self.sim._sensors[self.camera_uuids[0]]._sensor_object.node
        return float(node.translation[1]) + self.floor_below_feet

    # --- motion -----------------------------------------------------------------------------------------------

    def camera_clear(self, position: np.ndarray, yaw: float) -> bool:
        """True if the camera point (in front of the turning centre, at feet level) is in free space: at most
        (navmesh agent radius - camera_wall_margin_m) from the walkable area, whose edge lies one agent radius
        inside the walls. So the camera may come within the margin (2 cm) of a wall, never into it. Always true
        when the camera sits inside the body radius (LIMO: 0.084 m < 0.195 m)."""
        if not self.check_camera:
            return True
        cam = position + self.spec.camera_forward_m * forward(yaw)
        snapped = np.array(self.pathfinder.snap_point(cam.astype(np.float32)), dtype=np.float64)
        if not np.isfinite(snapped).all() or abs(snapped[1] - position[1]) > 0.5:
            return False
        return horizontal_distance(cam, snapped) <= self.free_radius

    def drive(self, v: float, w: float) -> Move:
        """Hold (v, w) for one control period, in sub-steps checked against the navmesh; see the module docstring."""
        v, w = clip_command(self.spec, v, w)
        dt = self.spec.dt / self.spec.substeps
        position, yaw = self.position.copy(), self.yaw
        travelled = short = 0.0
        refused = 0
        for _ in range(self.spec.substeps):
            delta = arc_step(yaw, v, w, dt)
            step = float(math.hypot(delta[0], delta[2]))  # intended (chord) length
            new_yaw = yaw + w * dt
            new_position = position
            if step > 0.0:
                target = position + delta
                new_position = np.array(self.pathfinder.try_step(position.astype(np.float32), target.astype(np.float32)),
                                        dtype=np.float64)
            # refuse moves that push the camera off the walkable area; a camera already off it (only possible
            # at a hand-placed pose) may move freely, so the robot is never trapped
            if not self.camera_clear(new_position, new_yaw) and self.camera_clear(position, yaw):
                refused += 1
                short += step
                if self.camera_clear(position, new_yaw):  # the turn alone is fine: turn on the spot
                    yaw = new_yaw
                continue
            moved = horizontal_distance(new_position, position)
            travelled += moved
            short += max(0.0, step - moved)
            position, yaw = new_position, new_yaw
        self._set_pose(position, yaw)
        return Move(v=v, w=w, travelled_m=travelled, short_m=short, refused_substeps=refused,
                    collided=refused > 0 or short > self.spec.collision_tol_m)

    # --- observations and queries -----------------------------------------------------------------------------

    def observe(self) -> Dict[str, np.ndarray]:
        """Camera pictures at the current pose: rgb (H, W, 3) uint8, plus depth / semantic if switched on."""
        out = {}
        for uuid in self.camera_uuids:
            sensor = self.sim._sensors[uuid]
            sensor.draw_observation()
            out[uuid] = np.asarray(sensor.get_observation())
        obs = {"rgb": np.ascontiguousarray(out.pop("color_sensor")[..., :3])}
        if "depth_sensor" in out:
            obs["depth"] = out["depth_sensor"]
        if "semantic_sensor" in out:
            obs["semantic"] = out["semantic_sensor"]
        return obs

    def surface_id(self) -> int:
        """Object id of the surface under the turning centre (needs semantic=True)."""
        sensor = self.sim._sensors[FLOOR_LABEL]
        sensor.draw_observation()
        return int(np.bincount(np.asarray(sensor.get_observation()).ravel()).argmax())

    def geodesic(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Walking distance on the navmesh (inf if b cannot be reached from a)."""
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(a, dtype=np.float32)
        path.requested_end = np.asarray(b, dtype=np.float32)
        return float(path.geodesic_distance) if self.pathfinder.find_path(path) else float("inf")

    def topdown(self, meters_per_pixel: float, height: float) -> Tuple[np.ndarray, np.ndarray]:
        """Walkable-area picture of the navmesh slice at `height` (rows = z, cols = x) and its (x, z) origin."""
        grid = np.asarray(self.pathfinder.get_topdown_view(meters_per_pixel, height), dtype=bool)
        low = self.pathfinder.get_bounds()[0]
        return grid, np.array([low[0], low[2]], dtype=np.float64)

    def close(self) -> None:
        self.sim.close()

    def __enter__(self) -> "LimoSim":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
