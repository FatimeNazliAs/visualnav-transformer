"""Phase 7.1 lens option: the double-sphere model (no simulator) and, with the HM3D mount, (1) the default pinhole camera
renders byte-identical pictures to the code before the lens option, (2) Habitat's fisheye puts a known 3D point on the
pixel our model predicts. Dev home 00746 (never an exam home)."""

import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mapmad_sim import config
from mapmad_sim.lens import DoubleSphere, from_habitat

REF = Path(__file__).resolve().parent / "data" / "p7_pinhole_render_sha256.json"
HOME, SPLIT = "00746-RTV2n6fXB2w", "train"
TEST_GPU = int(os.environ.get("MAPMAD_TEST_GPU", "0"))  # GPU that renders (the sweep launcher picks a free one)
needs_hm3d = pytest.mark.skipif(not config.hm3d_scene(SPLIT, HOME).exists(), reason="HM3D mount (/hm3d) missing")


def lens170() -> DoubleSphere:
    return DoubleSphere.fit_equidistant(fov_deg=170.0, circle_px=95.0, width=160, height=120)


def test_axis_lands_on_principal_point():
    lens = dataclasses.replace(lens170(), cx=81.5, cy=58.25)
    uv, ok = lens.project(np.array([[0.0, 0.0, 2.0]]))
    assert ok[0] and np.allclose(uv[0], [81.5, 58.25])


def test_fit_matches_equidistant_and_fov():
    lens = lens170()
    theta = np.radians(np.linspace(1, 85, 50))
    assert np.max(np.abs(lens.radius_px(theta) - 95.0 * theta / math.radians(85))) < 1.5  # pixels
    assert lens.fov_deg() == pytest.approx(170.0, abs=0.5)
    assert np.all(np.diff(lens.radius_px(theta)) > 0)


def test_directions_and_circle():
    lens = lens170()
    uv, ok = lens.project(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, -0.5]]))
    assert uv[0, 0] > lens.cx and uv[1, 1] > lens.cy  # right -> right, down (OpenCV y) -> down
    assert ok[:2].all() and not ok[2]  # 117 deg off-axis is outside the 170 deg circle
    assert np.allclose(from_habitat([[1.0, 2.0, -3.0]]), [[1.0, -2.0, 3.0]])
    rgb = np.full((120, 160, 3), 200, np.uint8)
    out = lens.apply_circle(rgb)
    assert out[0, 0].sum() == 0 and out[60, 80].sum() == 600  # corner black, centre kept


def test_config_round_trip():
    lens = lens170()
    again = DoubleSphere.from_config(lens.to_config())
    assert again.width == 160 and again.circle_px == 95.0 and again.xi == pytest.approx(lens.xi, abs=1e-6)


@needs_hm3d
def test_lens_off_renders_byte_identical_to_before():
    """Default RobotSpec (pinhole) at the reference poses -> the sha256s recorded with mapmad @ e97b875."""
    from mapmad_sim.robot import LimoSim, RobotSpec

    import p7_render_reference as ref_mod

    ref = json.loads(REF.read_text())
    with LimoSim(ref["split"], ref["home"], RobotSpec.from_config(config.robot()), gpu=TEST_GPU) as robot:
        assert ref_mod.render_hashes(robot, ref["poses"]) == ref["rgb_sha256"]


def _render_pair(lens: DoubleSphere, height_m: float = 0.5):
    """Co-located pinhole (semantic + depth, 90 deg, 160x160) and fisheye (semantic) pictures at 3 fixed poses."""
    import habitat_sim
    import magnum as mn
    from habitat_starter.sim import make_cfg

    settings = config.hm3d_sim_settings(SPLIT, HOME, width=160, height=160, hfov=90.0, sensor_height=height_m,
                                        depth_sensor=True, semantic_sensor=True, gpu_device_id=TEST_GPU)
    cfg = make_cfg(settings)
    fish = lens.habitat_spec("fish_semantic", mn.Vector3(0.0, height_m, 0.0))
    fish.sensor_type = habitat_sim.SensorType.SEMANTIC
    cfg.agents[0].sensor_specifications.append(fish)
    out = []
    with habitat_sim.Simulator(cfg) as sim:
        sim.pathfinder.seed(3)
        for k in range(3):
            agent = sim.initialize_agent(0)
            state = agent.get_state()
            state.position = sim.pathfinder.get_random_navigable_point()
            state.rotation = habitat_sim.utils.common.quat_from_angle_axis(1.1 * k, np.array([0.0, 1.0, 0.0]))
            agent.set_state(state)
            obs = sim.get_sensor_observations()
            out.append((obs["semantic_sensor"], obs["depth_sensor"], obs["fish_semantic"]))
    return out


def _agreement(lens: DoubleSphere, pairs) -> float:
    """Share of pinhole pixels (inside a uniform 5x5 label patch, depth 0.3-6 m) whose 3D point, projected with
    `lens`, lands on a fisheye pixel with the same object label."""
    f = 80.0 / math.tan(math.radians(45.0))  # 160 px, 90 deg
    hits = total = 0
    for sem, depth, fish in pairs:
        v, u = np.mgrid[2:158:3, 2:158:3]
        v, u = v.ravel(), u.ravel()
        patch = np.stack([sem[v + dv, u + du] for dv in range(-2, 3) for du in range(-2, 3)])
        d = depth[v, u]
        keep = (patch == patch[0]).all(axis=0) & (d > 0.3) & (d < 6.0)
        v, u, d = v[keep], u[keep], d[keep]
        pts = np.stack([(u + 0.5 - 80.0) / f * d, (v + 0.5 - 80.0) / f * d, d], axis=1)  # OpenCV frame
        uv, ok = lens.project(pts)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < lens.width) & (uv[:, 1] >= 0) & (uv[:, 1] < lens.height)
        q = np.floor(uv[ok]).astype(int)
        hits += int((fish[q[:, 1], q[:, 0]] == sem[v[ok], u[ok]]).sum())
        total += int(ok.sum())
    assert total > 500
    return hits / total


@needs_hm3d
def test_fisheye_puts_known_points_on_predicted_pixels():
    """Habitat's double-sphere render agrees with our projection (>= 95% same label); a wrong model (focal length
    10% off, principal point flipped) agrees clearly less, so the check can fail."""
    lens = lens170()
    pairs = _render_pair(lens)
    good = _agreement(lens, pairs)
    wrong = _agreement(dataclasses.replace(lens, fx=lens.fx * 1.1, fy=lens.fy * 1.1, cx=lens.width - lens.cx - 8.0), pairs)
    assert good >= 0.95, good
    assert wrong < good - 0.05, (good, wrong)
