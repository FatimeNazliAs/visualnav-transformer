"""Phase 7 camera settings: lens x height for the camera sweep, and how a setting changes the virtual LIMO.

A setting (config `cameras:` entry, e.g. p7_sweep.yaml) is plain data, so it travels over the bridge as JSON:

    pinhole_018: {lens: pinhole, height_m: 0.18, pitch_deg: -2.0, hfov_deg: 66.5}
    fisheye_095: {lens: fisheye, height_m: 0.95, pitch_deg: 0.0, double_sphere: {fx: ..., circle_px: ...}}

Applied to robot_limo.yaml's RobotSpec: camera height above the real floor, pitch, lens (pinhole HFOV or the
double-sphere fisheye of the colour camera; the depth camera stays LIMO's pinhole), and the floor map: a camera
higher than the floor map's agent height (0.55 m) raises that height to camera + 0.05 m (block item 5), so the robot
never drives its camera under a table. Body radius, motion, forward offset (0.084 m): unchanged.
"""

import dataclasses
from typing import Any, Dict, Optional

from mapmad_sim.lens import DoubleSphere

LENSES = ("pinhole", "fisheye")
CAMERA_CLEARANCE_M = 0.05  # floor-map height = camera height + this, when the camera is above the LIMO height
DEFAULT_HFOV_DEG = 66.5  # LIMO's colour camera (robot_limo.yaml)


def floor_map_key(agent_height_m: float) -> str:
    """Episode-file key of the floor map built for this agent height: "0.55" (LIMO) or e.g. "1.00"."""
    return f"{agent_height_m:.2f}"


def with_floor_map(spec: Any, agent_height_m: float) -> Any:
    """The RobotSpec with its floor map (navmesh) built for this agent height; everything else unchanged."""
    nav = dict(spec.navmesh)
    nav["agent_height_m"] = agent_height_m
    return dataclasses.replace(spec, navmesh=tuple(sorted(nav.items())))


@dataclasses.dataclass(frozen=True)
class CameraSetting:
    """One camera of the sweep."""

    name: str
    lens: str
    height_m: float
    pitch_deg: float
    hfov_deg: float = DEFAULT_HFOV_DEG  # pinhole only (and the depth camera)
    double_sphere: Optional[DoubleSphere] = None  # fisheye only

    def __post_init__(self) -> None:
        if self.lens not in LENSES:
            raise ValueError(f"camera {self.name}: lens must be one of {LENSES}")
        if (self.lens == "fisheye") != (self.double_sphere is not None):
            raise ValueError(f"camera {self.name}: a fisheye needs double_sphere parameters (and a pinhole none)")

    @classmethod
    def from_config(cls, name: str, cfg: Dict[str, Any]) -> "CameraSetting":
        ds = cfg.get("double_sphere")
        return cls(name=name, lens=cfg["lens"], height_m=float(cfg["height_m"]), pitch_deg=float(cfg["pitch_deg"]),
                   hfov_deg=float(cfg.get("hfov_deg", DEFAULT_HFOV_DEG)),
                   double_sphere=DoubleSphere.from_config(ds) if ds else None)

    @classmethod
    def from_wire(cls, d: Dict[str, Any]) -> "CameraSetting":
        """From the dict that travels over the bridge and into the logs: the config entry plus its "name"."""
        return cls.from_config(d["name"], d)

    def to_config(self) -> Dict[str, Any]:
        """Plain JSON-able dict (round trip with from_config, apart from the name)."""
        out: Dict[str, Any] = {"lens": self.lens, "height_m": self.height_m, "pitch_deg": self.pitch_deg,
                               "hfov_deg": self.hfov_deg}
        if self.double_sphere is not None:
            out["double_sphere"] = self.double_sphere.to_config()
        return out

    def floor_map_height(self, limo_height_m: float) -> float:
        """Agent height of the floor map: LIMO's, or camera + 5 cm for a camera above it."""
        return round(self.height_m + CAMERA_CLEARANCE_M, 4) if self.height_m > limo_height_m else limo_height_m

    def apply(self, spec: Any) -> Any:
        """The RobotSpec with this camera (height, pitch, lens) and its floor map."""
        spec = with_floor_map(spec, self.floor_map_height(float(dict(spec.navmesh)["agent_height_m"])))
        return dataclasses.replace(spec, camera_height_m=self.height_m, pitch_deg=self.pitch_deg,
                                   hfov_deg=self.hfov_deg, fisheye=self.double_sphere)


def cameras_of(cfg: Dict[str, Any]) -> Dict[str, CameraSetting]:
    """All settings of a config's `cameras:` block, by name."""
    return {n: CameraSetting.from_config(n, c) for n, c in (cfg.get("cameras") or {}).items()}
