"""Phase 11 (NoMaD side): the maps-hidden arm is NoMaD's own computation, the step-0 pick rule, the search runner's
determinism and logging, the DAgger union draw, the v2 loader without DAgger unchanged, and DAgger sample alignment
(label / frames / map) on the DRY drives when they exist."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

OFFICIAL = "/outputs/mapmad/weights/official/nomad.pth"
TRAIN = Path(__file__).resolve().parents[3]


# --- maps hidden = NoMaD ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(OFFICIAL), reason="needs the official nomad.pth")
def test_maps_hidden_arm_is_nomad_own_computation():
    from vint_train.mapmad.closed_loop.nomad_policy import load_nomad
    from vint_train.mapmad.closed_loop.policy import Observation
    from vint_train.mapmad.closed_loop.search_policy import MapMaDV2Policy
    from vint_train.mapmad.config import load_config
    from vint_train.mapmad.inference_v2 import MapMaDV2Inference
    from vint_train.mapmad.weights import build_with_official

    cfg = load_config(str(TRAIN / "config" / "mapmad_v2.yaml"))
    dev = torch.device("cpu")
    model, _ = build_with_official(cfg, OFFICIAL, dev)
    net = MapMaDV2Inference("", cfg, dev, model=model)
    pol = MapMaDV2Policy("", "x", cfg, dev, maps=False, net=net)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(16)]
    maps = rng.random((2, 4, 64, 64)).astype(np.float32)  # non-zero maps: the masks must hide them
    obs = Observation(frames=frames, goal_image=None, step=15, extras={"maps": maps})
    with torch.no_grad():
        mine = model("vision_encoder", **pol.encoder_inputs(obs))
        nomad = load_nomad(OFFICIAL, dict(cfg, map_input=False), dev)
        inp = pol.encoder_inputs(obs)
        ref = nomad("vision_encoder", obs_img=inp["obs_img"], goal_img=inp["goal_img"],
                    input_goal_mask=inp["input_goal_mask"])
    assert torch.allclose(mine, ref, atol=1e-6, rtol=0)
    pol_maps = MapMaDV2Policy("", "x", cfg, dev, maps=True, net=net)
    assert pol_maps.encoder_inputs(obs)["input_map_mask"].tolist() == [[0, 0]]
    assert inp["input_map_mask"].tolist() == [[1, 1]] and inp["input_goal_mask"].tolist() == [1]


# --- step-0 pick rule ------------------------------------------------------------------------------------------------

def test_pick_rule_margin_and_tie():
    from vint_train.mapmad.closed_loop.search_tables import pick_rule

    s1 = np.array([0, 1 / 3, 2 / 3, 1.0] * 30)
    assert pick_rule(s1, s1) == (0.0, "run1")  # tie -> run 1
    up = s1 + np.r_[np.full(11, 1 / 3), np.zeros(109)]  # Delta = 11 / 360 = +0.0306
    d, w = pick_rule(s1, up)
    assert d > 0.03 and w == "runC"
    up = s1 + np.r_[np.full(10, 1 / 3), np.zeros(110)]  # Delta = 10 / 360 = +0.0278
    assert pick_rule(s1, up)[1] == "run1"
    assert pick_rule(s1, s1 - 0.1)[1] == "run1"


# --- the search runner ------------------------------------------------------------------------------------------------

class FakeSim:
    """A 1-D corridor: the robot moves 0.05 m per forward step; success after 2.0 m; target in view from 1.0 m."""

    def __init__(self):
        self.x = 0.0

    def _frame(self, move=None):
        vp = max(0.0, 2.0 - self.x)
        seen = self.x >= 1.0
        state = {"position": [self.x, 0.0, 0.0], "yaw": 0.0, "vp_geodesic_m": vp, "geodesic_m": vp,
                 "floor_below_feet_m": 0.0, "in_view": seen, "first_in_view_step": 20 if seen else None,
                 "seen_instances": [7] if seen else []}
        return SimpleNamespace(rgb=np.full((4, 4, 3), int(self.x * 10), np.uint8), state=state, move=move,
                               extras={"maps": np.zeros((2, 4, 64, 64), np.float32)}, navmesh_sha256="abc")

    def reset(self, *a, **k):
        self.x = 0.0
        return self._frame()

    def step(self, v, w):
        self.x += v * 0.25
        return self._frame({"v": v, "w": w, "travelled_m": v * 0.25, "short_m": 0.0, "collided": False})


class FakePolicy:
    name, context_frames = "fake", 16

    def reset(self, meta):
        self.rng = np.random.default_rng(meta["seed"])

    def act(self, obs):
        from vint_train.mapmad.closed_loop.policy import Command
        return Command(0.2, float(self.rng.normal()) * 0.01, {"waypoint_m": [0.05, 0.0]})


def test_search_runner_log_is_deterministic_and_splits_search_and_approach(tmp_path):
    from vint_train.mapmad.closed_loop.search_runner import SearchSettings, run_search_episode

    ep = {"episode_id": "p11d-00000-chair-1", "home": "00000-x", "distance_bin": 0, "navmesh_sha256": "abc",
          "target": {"category": "chair"}}
    run = SearchSettings(0.2, 100, 5, 0.05)
    logs = []
    for k in range(2):
        path = tmp_path / f"{k}.jsonl"
        res, _ = run_search_episode(FakeSim(), FakePolicy(), ep, "run1", run, 0, 123, path, "fp",
                                    {"noise_seed": 5}, {"name": "pin025", "hfov_deg": 66.5})
        logs.append(path.read_bytes())
    assert logs[0] == logs[1]
    assert res["success"] and res["success_step"] == 36  # 0.05 m per step from 2.0 m to <= 0.2 m
    assert res["first_in_view_step"] == 20 and res["search_steps"] == 20 and res["approach_steps"] == 16
    assert b"time" not in logs[0]


# --- DAgger union ------------------------------------------------------------------------------------------------------

def _fake_v2(n_search, n_seen, seen_p):
    from vint_train.mapmad.dataset_v2 import MapMaDV2Dataset

    ds = MapMaDV2Dataset.__new__(MapMaDV2Dataset)
    ds.pools = {"search": np.arange(n_search), "seen": np.arange(n_seen)}
    ds.index_to_data = [(f"p8-{i}", i, 0) for i in range(n_search)]
    ds.seen_p = seen_p
    return ds


def test_union_draw_is_uniform_over_both_sources():
    main, dag = _fake_v2(3000, 0, 0.0), _fake_v2(1000, 0, 0.0)
    dag.index_to_data = [(f"dag-{i}", i, 0) for i in range(1000)]
    main.attach_dagger(dag)
    rng = np.random.default_rng(1)
    src = [main._union_draw(rng)[0] is dag for _ in range(20000)]
    assert abs(np.mean(src) - 0.25) < 0.01  # 1000 / (3000 + 1000)


def test_union_seen_pool_covers_both_sources():
    main, dag = _fake_v2(3000, 300, 0.25), _fake_v2(1000, 100, 0.25)
    dag.index_to_data = [(f"dag-{i}", i, 0) for i in range(1000)]
    main.attach_dagger(dag)
    rng = np.random.default_rng(2)
    draws = [main._union_draw(rng) for _ in range(40000)]
    from_seen = [t for s, d, t in draws if (s is main and t < 300) or (s is dag and t < 100)]
    assert 0.31 < len(from_seen) / len(draws) < 0.34  # p 0.25 + the seen frames the plain draws hit (0.1 x 0.75)
    dag_share = np.mean([s is dag for s, _, _ in draws])
    assert abs(dag_share - 0.25) < 0.01


def test_v2_dataset_without_dagger_draws_as_before():
    from vint_train.mapmad import modes as M
    from vint_train.mapmad.dataset_v2 import MapMaDV2Dataset

    ds = _fake_v2(10, 0, 0.0)
    ds.has_maps, ds.mode_p = True, M.habitat_mode_table(
        {"search_map": 0.45, "explore_map": 0.2, "explore": 0.2, "photo": 0.15}, M.HABITAT_MODES_V2)
    ds.in_pool = {"search": np.r_[np.ones(5, bool), np.zeros(5, bool)], "explore": np.r_[np.zeros(5, bool), np.ones(5, bool)]}
    ds.pools = {"search": np.arange(5), "explore": np.arange(5, 10)}
    ds.seed = 3
    calls = []
    ds.habitat_sample = lambda d, t, mode, rng: calls.append((d, t, mode.name, float(rng.random())))
    assert MapMaDV2Dataset.dagger is None
    for key in range(50):
        ds.get(key % 10, key)
    expected = []
    for key in range(50):  # the Phase 10 code path, written out
        rng = np.random.default_rng([3, key])
        mode = M.draw_habitat_mode(ds.mode_p, rng, M.HABITAT_MODES_V2)
        i = ds.pool_index(key % 10, mode, rng)
        expected.append((f"p8-{i}", i, mode.name, float(rng.random())))
    assert calls == expected


# --- DAgger samples (DRY drives) -----------------------------------------------------------------------------------------

DRY_CFG = "/outputs/mapmad/p11_dagger/DRY/config/v21_dry.yaml"


@pytest.mark.skipif(not os.path.exists(DRY_CFG), reason="needs the Phase 11 DRY rollouts (p11_dryrun.yaml)")
def test_dagger_sample_carries_its_label_frames_and_map():
    from vint_train.mapmad.config import load_config
    from vint_train.mapmad.dataset_dagger import build_dagger_dataset

    cfg = load_config(DRY_CFG)
    ds = build_dagger_dataset(cfg, "train", True)
    i = len(ds) // 2
    drive, t, _ = ds.index_to_data[i]
    x = ds.get(i, 7)
    labels = np.load(os.path.join(ds.data_folder, drive, "labels.npz"))
    assert labels["valid"][t] and t >= 15
    assert np.allclose(x[2].numpy(), labels["waypoints_m"][t] / 0.25)
    ctx = torch.cat([ds._load_image(drive, ct) for ct in (t - 15, t - 10, t - 5, t)])
    assert torch.equal(x[0], ctx)
    assert int(x[-1]) == 7 and int(x[8]) == 1 and x[9].tolist() == [0, 0]  # search_map: photo hidden, maps shown
    from vint_train.mapmad.dataset import pose_at
    built = ds.builder.build(drive, t, pose_at(ds._get_trajectory(drive), t), True)
    assert np.array_equal(x[7].numpy(), built)
