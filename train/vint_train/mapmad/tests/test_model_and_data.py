"""MapMaD model hooks, weight loading, epoch sampler, layered configs and (if the tiny split is mounted) the dataset.

CPU only; the checkpoint-level G3 checks are in checks_g3.py (GPU, official weights)."""

import os

import numpy as np
import pytest
import torch
import yaml

from vint_train.mapmad import modes as M
from vint_train.mapmad.config import deep_merge, resolve_base
from vint_train.mapmad.map_encoder import MapEncoder
from vint_train.mapmad.sampler import EpochSampler, KeyedConcatDataset
from vint_train.mapmad.weights import load_official_into

CFG = {"model_type": "nomad", "vision_encoder": "nomad_vint", "encoding_size": 64, "context_size": 3,
       "mha_num_attention_heads": 4, "mha_num_attention_layers": 2, "mha_ff_dim_factor": 2, "down_dims": [16, 32],
       "cond_predict_scale": False}


def small_models():
    from vint_train.mapmad.closed_loop.nomad_policy import build_nomad
    torch.manual_seed(0)
    old = build_nomad(dict(CFG, map_input=False)).eval()
    new = build_nomad(dict(CFG, map_input=True)).eval()
    return old, new


def test_map_encoder_shape_params_and_zero_init():
    enc = MapEncoder(out_dim=256)
    out = enc(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 256) and torch.all(out == 0)  # last Linear starts at zero
    n_conv = sum(p.numel() for p in enc.conv.parameters())
    n_lin = sum(p.numel() for p in enc.proj.parameters())
    assert 0.23e6 < n_conv < 0.25e6 and n_lin == 2048 * 256 + 256
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in enc.modules())


def test_load_and_map_hidden_equals_old(tmp_path):
    old, new = small_models()
    path = tmp_path / "old.pth"
    torch.save({"module." + k: v for k, v in old.state_dict().items()}, path)  # DataParallel-style keys
    report = load_official_into(new, str(path))
    assert report["new_keys"] and all(k.startswith("vision_encoder.map_encoder.") for k in report["new_keys"])
    assert report["positional_encoding"]["grown"] and report["positional_encoding"]["model_rows"] == 6
    torch.nn.init.normal_(new.vision_encoder.map_encoder.proj.weight)  # a live encoder: the mask must hide it
    b = 4
    obs, goal = torch.randn(b, 12, 96, 96), torch.randn(b, 3, 96, 96)
    maps = torch.rand(b, 3, 64, 64)
    with torch.no_grad():
        for g in (0, 1):
            gm = torch.full((b,), g, dtype=torch.long)
            ref = old("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm)
            got = new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps,
                      input_map_mask=torch.ones(b, dtype=torch.long))
            assert torch.allclose(got, ref, atol=1e-6, rtol=0)
            shown = new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps,
                        input_map_mask=torch.zeros(b, dtype=torch.long))
            assert (shown - ref).abs().max() > 1e-4


def test_load_refuses_changed_positional_rows(tmp_path):
    old, new = small_models()
    state = old.state_dict()
    state["vision_encoder.positional_encoding.pos_enc"] = state["vision_encoder.positional_encoding.pos_enc"] + 1e-3
    torch.save(state, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="rows differ"):
        load_official_into(new, str(tmp_path / "bad.pth"))


def test_epoch_sampler_weights_determinism_and_codes():
    s = EpochSampler([1000, 300], [0.7, 0.3], 20000, seed=0)
    d0, d0b, d1 = s.draws(0), s.draws(0), s.draws(1)
    assert np.array_equal(d0, d0b) and not np.array_equal(d0, d1)
    share = float((d0 < 1000).mean())
    assert 0.68 < share < 0.72 and d0.max() < 1300 and len(s) == 20000
    s.set_epoch(3)
    codes = list(s)
    keys = [c // 1300 for c in codes]
    assert len(set(keys)) == len(keys) and min(keys) > 0 and [c % 1300 for c in codes] == s.draws(3).tolist()


class _Echo(torch.utils.data.Dataset):
    def __init__(self, n, tag):
        self.n, self.tag = n, tag

    def __len__(self):
        return self.n

    def get(self, i, key):
        return self.tag, i, key


def test_keyed_concat_decodes_dataset_index_and_key():
    ds = KeyedConcatDataset([_Echo(5, "a"), _Echo(3, "b")])
    assert ds[(7 * 8) + 6] == ("b", 1, 7)
    assert ds[2] == ("a", 2, 2)  # plain index: key = index


def test_layered_config(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"a": 1, "datasets": {"x": {"p": 1, "q": 2}}}))
    top = {"base_config": "base.yaml", "a": 2, "datasets": {"x": {"q": 3}}}
    (tmp_path / "top.yaml").write_text(yaml.safe_dump(top))
    assert resolve_base(str(tmp_path / "top.yaml"), top) == {"a": 2, "datasets": {"x": {"p": 1, "q": 3}}}
    assert deep_merge({"k": {"a": 1}}, {"k": 5}) == {"k": 5}


def test_mode_table_and_draws():
    p = M.habitat_mode_table({"photo": .2, "map_goal": .3, "photo_map": .15, "explore_map": .15, "explore": .2})
    rng = np.random.default_rng(0)
    names = [M.draw_habitat_mode(p, rng).name for _ in range(20000)]
    assert abs(names.count("map_goal") / 20000 - 0.30) < 0.015
    with pytest.raises(ValueError):
        M.habitat_mode_table({"photo": 1.0})
    assert M.BY_NAME["photo_map"].goal_mask == 0 and M.BY_NAME["photo_map"].map_mask == 0


# --- dataset on the tiny split (skipped where the data disk is not mounted) -----------------------------------
TINY_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "mapmad_smoke.yaml")


def tiny_config():
    from vint_train.mapmad.config import load_config
    cfg = load_config(TINY_CONFIG)
    if not os.path.isdir(cfg["datasets"]["habitat_mapmad"]["data_folder"]):
        pytest.skip("MapMaD data not mounted")
    return cfg


def test_dataset_modes_maps_and_masks():
    from vint_train.mapmad.dataset import build_mapmad_dataset
    cfg = tiny_config()
    ds = build_mapmad_dataset(cfg, "habitat_mapmad", "train", train=True)
    assert not any(n in ds.traj_names for n in open(cfg["datasets"]["habitat_mapmad"]["exclude_drives"]).read().split())
    drive, t, _ = ds.index_to_data[len(ds) // 2]
    for name in M.HABITAT_MODES:
        mode = M.BY_NAME[name]
        s = ds.habitat_sample(drive, t, mode, np.random.default_rng(0), perturbation=None)
        assert len(s) == 11 and s[7].shape == (3, 64, 64) and s[7].dtype == torch.float32
        assert int(s[8]) == mode.goal_mask and int(s[9]) == mode.map_mask and int(s[10]) == mode.id
        if mode.map_mask:
            assert s[7].abs().sum() == 0
        else:
            assert s[7][1].sum() > 0  # something explored around the robot
            if mode.heat_on:
                assert s[7][2].max() > 0.9
            else:  # explore_map: the model's heat input is exactly zero (the sample sheet draws it for display only)
                assert torch.count_nonzero(s[7][2]) == 0
        if mode.goal_mask:
            assert float(s[6]) == 1.0 and torch.equal(s[1], s[0][-3:])
    for k in range(50):  # photo_map never draws a negative photo
        s = ds.habitat_sample(drive, t, M.BY_NAME["photo_map"], np.random.default_rng(k), perturbation=None)
        assert 1 <= int(s[3]) <= 20
    assert torch.equal(ds.get(3, 99)[7], ds.get(3, 99)[7])  # same key, same sample


def test_eval_dataset_never_perturbs_and_gostanford_map_hidden():
    from vint_train.mapmad.dataset import build_mapmad_dataset
    cfg = tiny_config()
    test = build_mapmad_dataset(cfg, "habitat_mapmad", "test", train=False)
    assert test.builder.perturb_prob == 0.0
    gs = build_mapmad_dataset(cfg, "go_stanford", "test", train=False)
    modes = [int(gs[i][10]) for i in range(40)]
    assert set(modes) <= {5, 6} and all(int(gs[i][9]) == 1 and gs[i][7].abs().sum() == 0 for i in range(5))
