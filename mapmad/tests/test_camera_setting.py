"""Phase 7 camera settings and sweep arms (no simulator): what a setting changes on the robot, arm rules, config."""

import dataclasses

import pytest

from mapmad_sim import config
from mapmad_sim import camera_setting
from mapmad_sim.camera_setting import CameraSetting, cameras_of, with_floor_map
from mapmad_sim.exam_arms import ExamArm, arms_of, floor_map_key
from mapmad_sim.lens import DoubleSphere
from mapmad_sim.robot import RobotSpec
from mapmad_sim.run_layout import read_config

LENS = DoubleSphere.fit_equidistant(170.0, 96.9, 160, 120, cx=82.2, cy=54.0)
P7 = config.CONFIG_DIR / "p7_sweep.yaml"


def limo() -> RobotSpec:
    return RobotSpec.from_config(config.robot())


def test_pin018_is_the_phase5_camera():
    """The LIMO-like setting changes nothing on the robot (same spec, so same pictures and floor map)."""
    cam = CameraSetting("pin018", "pinhole", 0.18, -2.0, 66.5)
    assert cam.apply(limo()) == limo()


def test_setting_changes_height_pitch_lens_only():
    spec = CameraSetting("fish025", "fisheye", 0.25, 0.0, double_sphere=LENS).apply(limo())
    assert (spec.camera_height_m, spec.pitch_deg, spec.fisheye) == (0.25, 0.0, LENS)
    assert spec.navmesh == limo().navmesh and spec.camera_forward_m == 0.084 and spec.max_v == 0.2


def test_floor_map_rises_only_above_limo_height():
    assert CameraSetting("a", "pinhole", 0.55, -2.0).floor_map_height(0.55) == 0.55
    assert CameraSetting("b", "pinhole", 0.95, -2.0).floor_map_height(0.55) == 1.0
    nav = dict(CameraSetting("b", "pinhole", 0.95, -2.0).apply(limo()).navmesh)
    assert nav["agent_height_m"] == 1.0 and nav["agent_radius_m"] == 0.195


def test_floor_map_helpers_and_wire_form():
    """One floor-map key format ("0.55") for the episode file and the arms; the wire dict (config entry + name)
    round-trips; with_floor_map changes only the navmesh agent height."""
    assert camera_setting.floor_map_key(0.55) == "0.55" and camera_setting.floor_map_key(1.0) == "1.00"
    fish = CameraSetting("fish055", "fisheye", 0.55, 0.0, double_sphere=LENS)
    again = CameraSetting.from_wire({"name": "fish055", **fish.to_config()})
    assert again.name == "fish055" and again.to_config() == fish.to_config()  # (to_config rounds the lens numbers)
    raised = with_floor_map(limo(), 1.0)
    assert dict(raised.navmesh)["agent_height_m"] == 1.0
    assert dataclasses.replace(raised, navmesh=limo().navmesh) == limo()
    tall = CameraSetting("pin095", "pinhole", 0.95, -2.0)
    assert floor_map_key({"name": "pin095", **tall.to_config()}) == "1.00"


def test_lens_and_parameters_must_agree():
    with pytest.raises(ValueError):
        CameraSetting("x", "fisheye", 0.25, 0.0)
    with pytest.raises(ValueError):
        CameraSetting("x", "pinhole", 0.25, 0.0, double_sphere=LENS)
    with pytest.raises(ValueError):
        CameraSetting("x", "wide", 0.25, 0.0)


def test_config_round_trip():
    cam = CameraSetting("fish025", "fisheye", 0.25, 0.0, double_sphere=LENS)
    assert CameraSetting.from_config("fish025", cam.to_config()) == dataclasses.replace(
        cam, double_sphere=DoubleSphere.from_config(LENS.to_config()))


def test_sweep_config_arms_and_cameras():
    cfg = read_config(P7)
    cams = cameras_of(cfg)
    assert sorted(c.height_m for c in cams.values()) == [0.18, 0.18, 0.25, 0.25, 0.45, 0.45, 0.55, 0.55]
    assert all(c.floor_map_height(0.55) == 0.55 for c in cams.values())  # Naz 3b: no raised floor map
    arms = [a for a in arms_of(cfg) if a.name.startswith(("A_", "B_")) and a.camera is not None]
    assert len(arms) == 16 and all(a.camera["name"] in cams for a in arms)
    extra = [a for a in arms_of(cfg) if a.name.startswith("X_")]
    assert len(extra) == 8 and all(a.context_spacing == 3 and a.episode_type == "in_view" for a in extra)
    assert arms_of(cfg, ["B_limo"])[0].camera is None  # compatibility arm: the Phase 5 camera path
    b = next(a for a in arms if a.name == "B_fish025")
    assert b.photo and b.episode_type == "in_view" and b.log_header({})["camera"] == "fish025"


def test_arm_episode_rules():
    arm = ExamArm("B_x", "nomad", photo=True, camera={"name": "x", "lens": "pinhole", "height_m": 0.25,
                                                      "pitch_deg": -2.0, "hfov_deg": 66.5}, episode_type="in_view")
    ep = {"type": "in_view", "navmesh_sha256": "old", "navmesh_sha256_by_floor_map": {"0.55": "limo"}}
    assert arm.runs_on(ep) and not arm.runs_on({**ep, "type": "out_of_view"})
    assert arm.episode_for(ep)["navmesh_sha256"] == "limo" and ep["navmesh_sha256"] == "old"
    assert floor_map_key(arm.camera) == "0.55"
    plain = ExamArm("B", "nomad", photo=True)  # Phase 5 arm: unchanged behaviour
    assert plain.runs_on({"type": "out_of_view"}) and plain.episode_for(ep) is ep and plain.log_header(ep) == {}
