"""Phase 14b-3 (MapMaD v3): past-move token + head (model, growth from a v2.1b checkpoint, maps hidden + token hidden =
NoMaD), past displacements, the reverse controller, the DAgger union, the past fields of a sample, the training loss.
CPU only, tiny models."""

import math

import numpy as np
import pytest
import torch

from vint_train.mapmad.past_moves import past_displacements, past_loss

CFG = {"model_type": "nomad", "vision_encoder": "nomad_vint", "encoding_size": 64, "context_size": 3,
       "mha_num_attention_heads": 4, "mha_num_attention_layers": 2, "mha_ff_dim_factor": 2, "down_dims": [16, 32],
       "cond_predict_scale": False, "num_diffusion_iters": 10, "len_traj_pred": 8, "image_size": [96, 96],
       "map": {"tokens": 2, "layers": 4}}
PAST = {"n": 8, "spacing": 5, "scale_m": 0.25, "drop_p": 0.3, "lambda": 0.1}


def build(past: bool, map_input: bool = True):
    from vint_train.mapmad.closed_loop.nomad_policy import build_nomad

    cfg = dict(CFG, map_input=map_input, map=dict(CFG["map"], **({"past_moves": PAST} if past else {})))
    torch.manual_seed(0)
    return build_nomad(cfg).eval()


def inputs(b: int = 4, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, 12, 96, 96, generator=g), torch.randn(b, 3, 96, 96, generator=g),
            torch.rand(b, 2, 4, 64, 64, generator=g), torch.randn(b, 8, 3, generator=g))


# --- displacements --------------------------------------------------------------------------------------------------
def test_past_displacements_straight_turn_and_start():
    t = np.arange(60)
    xy = np.stack([0.05 * t, np.zeros(60)], axis=1)  # 0.05 m per step forward
    d = past_displacements(xy, np.zeros(60), 50)
    assert d.shape == (8, 3) and np.allclose(d[:, 0], 1.0) and np.allclose(d[:, 1:], 0.0)  # 0.25 m per move
    early = past_displacements(xy, np.zeros(60), 12)
    assert np.allclose(early[:5], 0.0) and early[-1, 0] == pytest.approx(1.0)  # before step 0: standing
    yaw = np.full(60, math.pi / 2)  # facing +Y, moving +Y: forward in its own frame
    xy2 = np.stack([np.zeros(60), 0.05 * t], axis=1)
    assert np.allclose(past_displacements(xy2, yaw, 50)[:, 0], 1.0)
    turn = past_displacements(np.zeros((60, 2)), 0.02 * t, 50)
    assert np.allclose(turn[:, 2], 0.1) and np.allclose(turn[:, :2], 0.0)


def test_past_loss_counts_only_valid_samples():
    pred, target = torch.zeros(3, 8, 3), torch.ones(3, 8, 3)
    assert float(past_loss(pred, target, torch.tensor([1.0, 0.0, 0.0]))) == pytest.approx(1.0)
    assert float(past_loss(pred, target, torch.zeros(3))) == 0.0


# --- model ----------------------------------------------------------------------------------------------------------
def test_v3_grows_from_v21b_and_without_the_token_is_v21b(tmp_path):
    from vint_train.mapmad.weights import PAST_PREFIX, load_own_into
    import hashlib

    old, new = build(False), build(True)
    for p in old.parameters():  # a "trained" v2.1b: every weight non-trivial
        torch.nn.init.normal_(p, std=0.05)
    torch.save(old.state_dict(), tmp_path / "v21b.pth")
    rep = load_own_into(new, str(tmp_path / "v21b.pth"), hashlib.sha256((tmp_path / "v21b.pth").read_bytes()).hexdigest())
    assert rep["positional_encoding"]["model_rows"] == 8 and all(k.startswith(PAST_PREFIX) for k in rep["new_keys"])
    obs, goal, maps, past = inputs()
    gm = torch.ones(4, dtype=torch.long)
    with torch.no_grad():
        for mm in ([0, 0], [1, 1], [0, 1]):
            m = torch.tensor([mm] * 4)
            ref = old("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps, input_map_mask=m)
            absent = new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps, input_map_mask=m)
            assert torch.equal(absent, ref)  # no token given = v2.1b's computation exactly
            shown = new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps,
                        input_map_mask=m, past_move=past, input_past_mask=torch.zeros(4, dtype=torch.long))
            assert torch.isfinite(shown).all()
    assert new.vision_encoder.predict_past(ref).shape == (4, 8, 3)


def test_maps_hidden_and_token_hidden_is_official_nomad(tmp_path):
    from vint_train.mapmad.weights import load_official_into

    official = build(False, map_input=False)
    torch.save(official.state_dict(), tmp_path / "nomad.pth")
    new = build(True)
    rep = load_official_into(new, str(tmp_path / "nomad.pth"))
    assert {k.split(".")[2] for k in rep["new_keys"]} == {"near", "far", "past", "past_head"}
    for name, enc in new.vision_encoder.map_encoder.items():  # live new encoders: hiding must not depend on them
        for p in enc.parameters():
            torch.nn.init.normal_(p, std=0.05)
    obs, goal, maps, past = inputs()
    with torch.no_grad():
        for g in (0, 1):
            gm = torch.full((4,), g, dtype=torch.long)
            ref = official("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm)
            hid = new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps,
                      input_map_mask=torch.ones((4, 2), dtype=torch.long), past_move=past,
                      input_past_mask=torch.ones(4, dtype=torch.long))
            assert torch.equal(hid, ref)
            shown = new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps,
                        input_map_mask=torch.ones((4, 2), dtype=torch.long), past_move=past,
                        input_past_mask=torch.zeros(4, dtype=torch.long))
            assert (shown - ref).abs().max() > 1e-5  # maps hidden, token shown: the token is used


def test_past_token_needs_v2_maps_and_a_mask():
    from vint_train.models.nomad.nomad_vint import NoMaD_ViNT

    with pytest.raises(ValueError):
        NoMaD_ViNT(context_size=3, obs_encoding_size=64, map_input=True, map_tokens=1, past_moves=8)
    new = build(True)
    obs, goal, maps, past = inputs()
    with pytest.raises(ValueError):
        new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=torch.ones(4, dtype=torch.long),
            map_img=maps, input_map_mask=torch.zeros((4, 2), dtype=torch.long), past_move=past)


def test_new_parameters_train_in_the_map_encoder_group():
    from vint_train.mapmad.lr_schedule import group_param_names

    names = group_param_names(build(True))["map_encoder"]
    assert any(".past." in n for n in names) and any(".past_head." in n for n in names)


# --- reverse controller ---------------------------------------------------------------------------------------------
def test_reverse_controller_only_for_waypoints_behind():
    from vint_train.mapmad.closed_loop.deployment import ReverseSpec, pd_controller

    r = ReverseSpec(150.0, 0.1)
    assert not r.behind(np.array([0.2, 0.1])) and not r.behind(np.array([-0.1, 0.2]))  # 117 deg: not behind
    assert r.behind(np.array([-0.25, 0.0])) and r.behind(np.array([-0.25, 0.05]))
    v, w = r.command(np.array([-0.25, 0.0]), 0.4, 0.25)
    assert v == pytest.approx(-0.1) and w == 0.0
    v, w = r.command(np.array([-0.25, 0.05]), 0.4, 0.25)  # point back-left: the rear turns left -> clockwise (w < 0)
    assert v < 0 and w < 0
    assert pd_controller(np.array([-0.25, 0.0]), 0.2, 0.4, 0.25)[0] == 0.0  # NoMaD's controller never reverses


def test_policy_reverse_flag_and_off_switch():
    from vint_train.mapmad.closed_loop.deployment import ReverseSpec
    from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy
    from vint_train.mapmad.closed_loop.policy import Observation

    class Fixed(NomadPolicy):
        def __init__(self, wp):
            self.wp = wp
            self.cfg = {"normalize": True}
            self.sample_index, self.waypoint_index, self.max_v, self.max_w, self.rate_hz = 0, 1, 0.2, 0.4, 4.0
            self.picker, self.log_all_samples, self.context_frames = None, False, 1

        @property
        def metres_per_unit(self):
            return 0.25

        def sample_paths(self, obs):
            return np.tile(np.asarray(self.wp, np.float64), (1, 8, 1))

    obs = Observation(frames=[np.zeros((4, 4, 3), np.uint8)], goal_image=None, step=0, extras={})
    p = Fixed([-1.0, 0.0])
    off = p.act(obs)
    assert off.v == 0.0 and "reverse" not in off.info
    p.reverse = ReverseSpec()
    on = p.act(obs)
    assert on.v == pytest.approx(-0.1) and on.info["reverse"] is True
    p.wp = [1.0, 0.2]
    fwd_on = p.act(obs)
    p.reverse = None
    assert (fwd_on.v, fwd_on.w) == (p.act(obs).v, p.act(obs).w) and "reverse" not in fwd_on.info


# --- data -----------------------------------------------------------------------------------------------------------
class _Part:
    def __init__(self, n, tag):
        self.index_to_data = [(f"d{tag}{i}", i, 0) for i in range(n)]
        self.traj_names = sorted({d for d, _, _ in self.index_to_data})
        self.pools = {"search": np.arange(n)}
        self.tag = tag

    def habitat_sample(self, drive, t, *a, **k):
        return (self.tag, drive, t)

    def _get_trajectory(self, drive):
        return {"position": np.zeros((5, 2)), "yaw": np.zeros(5), "drive": drive}


def test_dagger_union_concatenates_and_routes():
    from vint_train.mapmad.dataset_dagger import DaggerUnion

    u = DaggerUnion([_Part(3, "a"), _Part(2, "b")])
    assert len(u) == 5 and u.part_sizes() == [3, 2] and list(u.pools["search"]) == [0, 1, 2, 3, 4]
    key, t, _ = u.index_to_data[4]
    assert key == "1|db1" and u.habitat_sample(key, t) == ("b", "db1", 1)
    assert u._get_trajectory("0|da2")["drive"] == "da2"


def test_past_part_gostanford_hidden_and_drop_only_in_training():
    from vint_train.mapmad.dataset_v2 import MapMaDV2Dataset

    ds = MapMaDV2Dataset.__new__(MapMaDV2Dataset)
    ds.past_cfg, ds.past_train = dict(PAST), False
    gs = ds.past_part(None, np.random.default_rng(0))
    assert int(gs[1]) == 1 and float(gs[2]) == 0.0 and not gs[0].any()
    src = (_Part(1, "a"), "d", 4)
    shown = ds.past_part(src, np.random.default_rng(0))
    assert int(shown[1]) == 0 and float(shown[2]) == 1.0 and tuple(shown[0].shape) == (8, 3)
    ds.past_train, ds.past_cfg["drop_p"] = True, 1.0
    assert int(ds.past_part(src, np.random.default_rng(0))[1]) == 1


def test_to_batch_and_aux_loss_with_past_fields():
    from vint_train.mapmad.train_loop import encode, past_aux_loss, to_batch

    b = 2
    data = (torch.rand(b, 12, 96, 96), torch.rand(b, 3, 96, 96), torch.zeros(b, 8, 2), torch.zeros(b),
            torch.zeros(b, 2), torch.zeros(b), torch.ones(b), torch.rand(b, 2, 4, 64, 64), torch.ones(b, dtype=torch.long),
            torch.zeros(b, 2, dtype=torch.long), torch.tensor([7, 4]), torch.randn(b, 8, 3), torch.tensor([0, 1]),
            torch.tensor([1.0, 1.0]))
    batch = to_batch(data, lambda x: x, torch.device("cpu"))
    assert set(batch) >= {"past", "past_mask", "past_valid"}
    model = build(True)
    cond = encode(model, batch)
    loss = past_aux_loss(model, cond, batch)
    assert torch.isfinite(loss) and loss.requires_grad
    assert "past" not in to_batch(data[:11], lambda x: x, torch.device("cpu"))


def test_hidden_q_cap():
    """Rules amendment v1.1: a q above 1 is an error without a cap and becomes the cap with one."""
    from vint_train.mapmad import modes as M
    from vint_train.mapmad.dataset_v2 import MapMaDV2Dataset

    class _Fake:
        def __init__(self, n):
            self.pools = {"search": np.arange(n), "explore": np.arange(n)}

    ds = MapMaDV2Dataset.__new__(MapMaDV2Dataset)
    ds.mode_p = M.habitat_mode_table({"search_map": 0.45, "explore_map": 0.20, "explore": 0.20, "photo": 0.15},
                                     M.HABITAT_MODES_V2)
    ds.pools = {"search": np.arange(1000)}
    ds.dagger = _Fake(900)  # s = 0.4737 -> q = 1.066
    with pytest.raises(ValueError):
        ds.attach_dagger_hidden(_Fake(10))
    ds.attach_dagger_hidden(_Fake(10), q_cap=1.0)
    assert ds.hidden_q == 1.0 and ds.hidden_q_raw == pytest.approx(0.45 * 900 / 1900 / 0.20)
    ds.dagger = _Fake(100)  # q below the cap: unchanged
    ds.attach_dagger_hidden(_Fake(10), q_cap=1.0)
    assert ds.hidden_q == ds.hidden_q_raw == pytest.approx(0.45 * 100 / 1100 / 0.20)
