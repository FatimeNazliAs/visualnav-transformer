"""Phase 5 ablation 1 (perturbed heat): the shifted heat target and the ablation arms (exam_arms.py)."""

import math
from pathlib import Path

import numpy as np
import pytest

from mapmad_sim.exam_arms import ExamArm, arms_of, heat_shift_angle, shifted_heat_episode
from mapmad_sim.run_layout import read_config
from vint_train.mapmad import heat as heat_lib

CONFIGS = Path(__file__).resolve().parents[4] / "mapmad" / "configs"
EPISODE = {"episode_id": "p5e-00800-sofa-2", "home": "00800-abc",
           "target": {"category": "sofa", "box_center": [1.0, 0.4, -2.0], "box_size": [2.0, 0.8, 1.0],
                      "point": [1.0, 0.0, -2.0], "view_points": [[0.0, 0.0, -1.0]]}}


def heat_of(episode):
    t = episode["target"]
    return heat_lib.target_from_meta({"target": {"kind": "object", "box_center": t["box_center"],
                                                 "box_size": t["box_size"]}})


@pytest.mark.parametrize("d", [0.5, 1.0])
def test_shift_moves_only_the_heat_box_by_d(d):
    out = shifted_heat_episode(EPISODE, d)
    (x0, y0, z0), (x1, y1, z1) = EPISODE["target"]["box_center"], out["target"]["box_center"]
    assert math.hypot(x1 - x0, z1 - z0) == pytest.approx(d) and y1 == y0
    assert EPISODE["target"]["box_center"] == [1.0, 0.4, -2.0]  # the frozen episode is not changed
    for k in ("box_size", "point", "view_points", "category"):
        assert out["target"][k] == EPISODE["target"][k]
    a, b = heat_of(EPISODE), heat_of(out)
    assert np.hypot(*np.subtract(b.centre, a.centre)) == pytest.approx(d)
    assert np.allclose(np.subtract(b.rect, a.rect), [b.centre[0] - a.centre[0]] * 2 + [b.centre[1] - a.centre[1]] * 2)


def test_direction_is_seeded_per_episode_and_shared_by_distances():
    a = heat_shift_angle(EPISODE["episode_id"])
    assert a == heat_shift_angle(EPISODE["episode_id"]) and 0.0 <= a < 2 * math.pi
    assert a != heat_shift_angle("p5e-00800-sofa-3")
    c0 = np.array(EPISODE["target"]["box_center"])
    u = np.array(shifted_heat_episode(EPISODE, 0.5)["target"]["box_center"]) - c0
    v = np.array(shifted_heat_episode(EPISODE, 1.0)["target"]["box_center"]) - c0
    assert np.allclose(2 * u, v)
    angles = [heat_shift_angle(f"e{i}") for i in range(2000)]
    assert np.histogram(angles, bins=4, range=(0, 2 * math.pi))[0].min() > 400  # roughly uniform


def test_exam_arms_unchanged_and_ablation_arms_from_config():
    cfg = read_config(CONFIGS / "p5_ablation_heat.yaml")
    exam = {a.name: a for a in arms_of(read_config(CONFIGS / "p5_exam.yaml"))}
    for a in exam.values():
        assert a.heat_shift_m == 0.0 and a.episode_for(EPISODE) is EPISODE and a.log_header(EPISODE) == {}
    h05, h10 = arms_of(cfg, ["C_h05", "C_h10"])
    for arm, d in ((h05, 0.5), (h10, 1.0)):
        assert (arm.outputs, arm.mapmad_goal, arm.reset_goal, arm.weights_key) == \
            (exam["C"].outputs, exam["C"].mapmad_goal, exam["C"].reset_goal, exam["C"].weights_key)
        assert arm.episode_for(EPISODE) == shifted_heat_episode(EPISODE, d)
        assert arm.log_header(EPISODE)["heat_shift_m"] == d
    assert cfg["mapmad"] == read_config(CONFIGS / "p5_exam.yaml")["mapmad"]


def test_shift_refused_on_arms_without_the_map():
    with pytest.raises(ValueError):
        ExamArm(name="D_h05", policy="mapmad", map=False, heat_shift_m=0.5)


# --- layer ablation (map_zero) and all-samples logging ---------------------------------------------------------------
def test_layer_ablation_arms_from_config():
    from mapmad_sim.exam_arms import MAP_LAYERS

    assert MAP_LAYERS == ("obstacle", "explored", "heat")
    heat_only, no_heat = arms_of(read_config(CONFIGS / "p5_layer_ablation.yaml"), ["C_heat_only", "C_no_heat"])
    assert heat_only.map_zero_indices == [0, 1] and no_heat.map_zero_indices == [2]
    assert heat_only.log_header(EPISODE) == {"map_zero": ["obstacle", "explored"]}
    assert heat_only.outputs == {"depth": True, "map": True, "grid": False} and heat_only.mapmad_goal == "map"
    for a in arms_of(read_config(CONFIGS / "p5_exam.yaml")):
        assert a.map_zero == () and a.map_zero_indices == []
    with pytest.raises(ValueError):
        ExamArm(name="D_x", policy="mapmad", map=False, map_zero=["heat"])
    with pytest.raises(ValueError):
        ExamArm(name="C_x", policy="mapmad", map=True, map_zero=["walls"])


def test_policy_zeroes_only_the_listed_layers():
    import torch

    from vint_train.mapmad.tests.test_p5_policies import bare_policy, obs_for

    obs = obs_for(5)
    obs.extras["map"][0, 1, 1] = obs.extras["map"][1, 2, 2] = 1.0  # one obstacle and one explored cell; heat at [2, 10, 20]
    full = bare_policy("map").encoder_inputs(obs)["map_img"][0]
    for layers, kept in (([0, 1], [2]), ([2], [0, 1])):
        p = bare_policy("map")
        p.zero_layers = layers
        got = p.encoder_inputs(obs)["map_img"][0]
        assert float(got[layers].abs().sum()) == 0.0 and torch.equal(got[kept], full[kept])
    assert np.array_equal(obs.extras["map"][2], full[2].numpy())  # the observation itself is not changed


def test_all_samples_logging_adds_one_field():
    from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy

    p = NomadPolicy.__new__(NomadPolicy)
    p.cfg, p.max_v, p.max_w, p.rate_hz = {"normalize": True}, 0.2, 0.4, 4.0
    p.sample_index, p.waypoint_index, p.context_frames = 0, 2, 1
    paths = np.cumsum(np.ones((8, 8, 2)) * np.arange(1, 9)[:, None, None], axis=1)  # sample k: steps of size k + 1
    p.sample_paths = lambda obs: paths
    from vint_train.mapmad.tests.test_p5_policies import obs_for

    plain = p.act(obs_for(1))
    p.log_all_samples = True
    both = p.act(obs_for(1))
    assert (plain.v, plain.w) == (both.v, both.w) and set(both.info) - set(plain.info) == {"paths_all_m"}
    assert np.allclose(both.info["paths_all_m"], paths * 0.05) and both.info["path_m"] == both.info["paths_all_m"][0]
