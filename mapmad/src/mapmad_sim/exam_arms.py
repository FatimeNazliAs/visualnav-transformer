"""Phase 5 arms (configs/p5_exam.yaml `arms`): what each arm's policy is given and which runs it has. Pure Python, so
both containers use the same rules (runner, gate, videos, S-freeze estimate).

    arm = ExamArm.from_config("C", cfg["arms"]["C"])
    arm.outputs            # what the simulator server sends with every frame
    arm.seed_indices([0, 1, 2])   # the planner (deterministic) runs index 0 only
    arm.episode_for(episode)      # ablation 1 (perturbed heat): the episode with its heat box moved by heat_shift_m

Phase 7 (configs/p7_sweep.yaml): an arm may name a camera setting (`camera: <name>` of the config's `cameras:`;
the server then renders with that lens / height on that floor map) and an episode type it runs on
(`episode_type: in_view | out_of_view`). Arms without them behave exactly as in Phase 5.
"""

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

MAP_LAYERS = ("obstacle", "explored", "heat")  # the map tensor's channels, in order
HEAT_SHIFT_SALT = "mapmad-p5-heat-shift"  # ablation 1: direction = sha256("<salt>/<episode_id>"), every distance


def heat_shift_angle(episode_id: str) -> float:
    """Ablation 1: the seeded random direction (radians in Habitat's x-z plane, uniform in [0, 2 pi)) of an episode's
    heat shift; it depends on the episode only, so the 0.5 m and 1.0 m shifts point the same way."""
    u = int(hashlib.sha256(f"{HEAT_SHIFT_SALT}/{episode_id}".encode()).hexdigest()[:8], 16) / 16 ** 8
    return 2.0 * math.pi * u


def shifted_heat_episode(episode: Dict[str, Any], shift_m: float) -> Dict[str, Any]:
    """A copy of the episode whose object box (the heat target only: `online_map.heat_target`) is moved shift_m
    metres along heat_shift_angle; the box size, view points, target point and success test are unchanged."""
    out = copy.deepcopy(episode)
    a = heat_shift_angle(episode["episode_id"])
    x, y, z = out["target"]["box_center"]
    out["target"]["box_center"] = [x + shift_m * math.cos(a), y, z + shift_m * math.sin(a)]
    return out


@dataclass
class ExamArm:
    """One arm: which policy, which goal inputs, which server outputs."""

    name: str
    policy: str  # nomad | mapmad | planner
    photo: bool = False
    map: bool = False
    waypoint_index: Optional[int] = None  # MapMaD arms: overrides `mapmad.waypoint_index` (dev waypoint test)
    hfov_deg: float = 66.5
    heat_shift_m: float = 0.0  # ablation 1: heat target moved this far (map arms only); 0 = the exam's heat
    map_zero: Sequence[str] = ()  # layer ablation: these MAP_LAYERS are set to 0 in the policy's input
    camera: Optional[Dict[str, Any]] = None  # Phase 7: camera setting config dict with "name" (None = LIMO's camera)
    episode_type: Optional[str] = None  # Phase 7: run only on episodes of this type (None = all)
    context_spacing: Optional[int] = None  # Phase 7 extra: official NoMaD's context frame every k-th step (None = 1)

    def __post_init__(self) -> None:
        self.map_zero = tuple(self.map_zero)
        if (self.heat_shift_m or self.map_zero) and not self.map_given:
            raise ValueError(f"arm {self.name}: heat_shift_m / map_zero need an arm that is given the map")
        bad = set(self.map_zero) - set(MAP_LAYERS)
        if bad:
            raise ValueError(f"arm {self.name}: unknown map layers {sorted(bad)} (known: {MAP_LAYERS})")
        if self.camera is not None:
            self.hfov_deg = float(self.camera.get("hfov_deg", self.hfov_deg))

    @property
    def map_zero_indices(self) -> List[int]:
        """Channel indices of the map layers this arm's policy gets as 0."""
        return [MAP_LAYERS.index(n) for n in self.map_zero]

    @classmethod
    def from_config(cls, name: str, spec: Dict[str, Any]) -> "ExamArm":
        return cls(name=name, **spec)

    @property
    def outputs(self) -> Dict[str, Any]:
        """What the server sends with every frame (all arms get the depth camera, so a home is never reloaded
        between arms)."""
        return {"depth": True, "map": self.map_given, "grid": self.policy == "planner"}

    @property
    def map_given(self) -> bool:
        """The policy receives the map tensor (MapMaD with the map goal)."""
        return self.policy == "mapmad" and self.map

    @property
    def deterministic(self) -> bool:
        """The planner has no noise: its seed-index-0 run counts for every seed index."""
        return self.policy == "planner"

    def seed_indices(self, seeds: Sequence[int]) -> List[int]:
        """The seed indices this arm actually runs."""
        return [k for k in seeds if not self.deterministic or k == 0]

    def run_seed_index(self, seed_index: int) -> int:
        """The seed index of the run that stands for `seed_index` (the planner: always 0)."""
        return 0 if self.deterministic else seed_index

    def runs_on(self, episode: Dict[str, Any]) -> bool:
        """Phase 7: the arm runs on this episode (its type matches episode_type, or no type is set)."""
        return self.episode_type is None or episode.get("type") == self.episode_type

    def episode_for(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """The episode as this arm's simulator server gets it (heat box shifted for ablation 1; with a Phase 7 camera,
        the floor-map hash of that camera's floor map; else unchanged)."""
        if self.camera is not None:
            out = dict(episode)
            out["navmesh_sha256"] = episode["navmesh_sha256_by_floor_map"][floor_map_key(self.camera)]
            return out
        return shifted_heat_episode(episode, self.heat_shift_m) if self.heat_shift_m else episode

    def log_header(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Extra log-header fields of this arm (heat shift, zeroed map layers; none for the exam arms, so their logs
        keep the exam's bytes)."""
        out: Dict[str, Any] = {}
        if self.heat_shift_m:
            out.update({"heat_shift_m": self.heat_shift_m,
                        "heat_shift_angle_rad": round(heat_shift_angle(episode["episode_id"]), 6),
                        "heat_box_center": [round(v, 4) for v in self.episode_for(episode)["target"]["box_center"]]})
        if self.map_zero:
            out["map_zero"] = list(self.map_zero)
        if self.camera is not None:
            out["camera"] = self.camera["name"]
        return out

    @property
    def reset_goal(self) -> str:
        """The goal named to the policy at episode start: the photo for photo arms, otherwise masked."""
        return "photo" if self.photo else "masked"

    @property
    def mapmad_goal(self) -> str:
        """MapMaD's goal input: the map, the photo (map masked), or both masked."""
        return "map" if self.map else ("photo" if self.photo else "masked")

    @property
    def weights_key(self) -> Optional[str]:
        """Which frozen weights hash this arm's logs carry (`nomad` / `mapmad`; the planner has none)."""
        return self.policy if self.policy in ("nomad", "mapmad") else None


def floor_map_key(camera: Dict[str, Any]) -> str:
    """Phase 7: the episode-file key of a camera's floor map = its agent height in metres, "0.55" or e.g. "1.00"."""
    from mapmad_sim import camera_setting, config

    limo = float(config.robot()["navmesh"]["agent_height_m"])
    return camera_setting.floor_map_key(camera_setting.CameraSetting.from_wire(camera).floor_map_height(limo))


def arms_of(cfg: Dict[str, Any], names: Optional[Sequence[str]] = None) -> List[ExamArm]:
    """The config's arms (all, or these names in this order); a `camera: <name>` is replaced by that entry of
    cfg["cameras"] (with its name)."""
    out = []
    for n in names or list(cfg["arms"]):
        spec = dict(cfg["arms"][n])
        if isinstance(spec.get("camera"), str):
            spec["camera"] = {"name": spec["camera"], **cfg["cameras"][spec["camera"]]}
        out.append(ExamArm.from_config(n, spec))
    return out
