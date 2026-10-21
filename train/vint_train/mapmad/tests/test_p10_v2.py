"""Phase 10 (MapMaD v2): teacher frontier + cache, v2 modes, layered-config replace, the two-token model, attention
per map token, v2 dataset (tiny split, skipped if not mounted), offline arms, inference entry point, v1 unchanged.
CPU only."""

import math
import os

import numpy as np
import pytest
import torch

from vint_train.mapmad import modes as M
from vint_train.mapmad.attention import attention_share, attention_shares
from vint_train.mapmad.config import deep_merge, map_model_kwargs
from vint_train.mapmad.frontier_cache import build_drive, frontier_from_cache, mismatched_frames
from vint_train.mapmad.local_map import known_at
from vint_train.mapmad.map_v2 import bumped_cells, frontier, teacher_frontier_at, teacher_frontier_clusters

CFG = {"model_type": "nomad", "vision_encoder": "nomad_vint", "encoding_size": 64, "context_size": 3,
       "mha_num_attention_heads": 4, "mha_num_attention_layers": 2, "mha_ff_dim_factor": 2, "down_dims": [16, 32],
       "cond_predict_scale": False, "num_diffusion_iters": 10, "len_traj_pred": 8, "image_size": [96, 96],
       "map": {"tokens": 2, "layers": 4}}


# --- teacher frontier ----------------------------------------------------------------------------------------------
def corridor_map(n_frames: int = 6):
    """A 12 x 30 storey: walls on rows 0 / 11; columns 1..(5 + 3 t) explored at frame t (a growing corridor)."""
    shape = (12, 30)
    obstacle = np.full(shape, -1, np.int32)
    explored = np.full(shape, -1, np.int32)
    for t in range(n_frames):
        cols = slice(1, 5 + 3 * t)
        explored[:, cols] = np.where(explored[:, cols] < 0, t, explored[:, cols])
    obstacle[[0, 11], :] = np.where(explored[[0, 11], :] >= 0, explored[[0, 11], :], -1)
    return {"obstacle_first_seen": obstacle, "explored_first_seen": explored, "origin": np.array([0.0, 0.0]),
            "resolution": np.float64(0.1)}


def test_bumps_count_only_from_the_next_frame():
    ev = np.array([[3.0, 0.55, 0.25]])  # logged at frame 3 -> cell (5, 2) from frame 4 on
    assert not bumped_cells((12, 30), (0.0, 0.0), 0.1, ev, 3).any()
    b = bumped_cells((12, 30), (0.0, 0.0), 0.1, ev, 4)
    assert list(zip(*np.nonzero(b))) == [(5, 2)]
    assert not bumped_cells((12, 30), (0.0, 0.0), 0.1, np.array([[0.0, 9.0, 9.0]]), 5).any()  # outside: dropped


def test_teacher_frontier_is_plain_frontier_with_bumped_cells_as_obstacles():
    m = corridor_map()
    t = 2
    plain = frontier(known_at(m["obstacle_first_seen"], t), known_at(m["explored_first_seen"], t))
    assert np.array_equal(teacher_frontier_at(m, np.zeros((0, 3)), t), plain)
    edge = np.nonzero(plain)
    i, j = int(edge[0][4]), int(edge[1][4])  # a frontier cell in the middle of the corridor's front edge
    ev = np.array([[0.0, (i + 0.5) * 0.1, (j + 0.5) * 0.1]])
    with_bump = teacher_frontier_at(m, ev, t)
    assert not with_bump[i, j] and with_bump.sum() < plain.sum()
    labels = teacher_frontier_clusters(m, ev, t)
    assert labels.max() >= 1


def write_drive(folder, m, n_frames, bump_events, teacher_counts=None, modes=None):
    os.makedirs(folder, exist_ok=True)
    np.savez(os.path.join(folder, "mapmad_map.npz"), **m)
    counts = teacher_counts if teacher_counts is not None else [
        int(teacher_frontier_clusters(m, bump_events, t).max()) for t in range(n_frames)]
    names = np.array(["search", "approach", "arrived", "explore", "no_frontier", "blacklist_exhausted", "no_path"])
    np.savez(os.path.join(folder, "mapmad_frames.npz"), bump_events=bump_events,
             frontier_clusters=np.asarray(counts, np.int32), teacher_mode=np.asarray(modes or [3] * n_frames, np.uint8),
             teacher_mode_names=names)


def test_cache_rebuilds_every_frame_and_checks_the_teacher_count(tmp_path):
    m = corridor_map(8)
    ev = np.array([[2.0, 0.55, 0.85], [5.0, 0.35, 1.45]])
    write_drive(str(tmp_path / "d"), m, 8, ev)
    cache = build_drive(str(tmp_path / "d"))
    for t in range(8):
        assert np.array_equal(frontier_from_cache(cache, t), teacher_frontier_at(m, ev, t))
    assert len(mismatched_frames(cache)) == 0
    bad = [int(teacher_frontier_clusters(m, ev, t).max()) for t in range(8)]
    bad[6] += 1  # the generator logged another count at frame 6
    write_drive(str(tmp_path / "e"), m, 8, ev, teacher_counts=bad, modes=[3] * 5 + [1] + [3, 3])
    assert mismatched_frames(build_drive(str(tmp_path / "e"))).tolist() == [6]
    write_drive(str(tmp_path / "f"), m, 8, ev, teacher_counts=bad, modes=[3] * 6 + [1, 3])  # approach: not checked
    assert len(mismatched_frames(build_drive(str(tmp_path / "f")))) == 0


# --- modes, config -------------------------------------------------------------------------------------------------
def test_v2_mode_shares_within_one_point_on_100k_draws():
    shares = {"search_map": 0.45, "explore_map": 0.20, "explore": 0.20, "photo": 0.15}
    p = M.habitat_mode_table(shares, M.HABITAT_MODES_V2)
    names = [M.draw_habitat_mode(p, np.random.default_rng([1, k]), M.HABITAT_MODES_V2).name for k in range(100_000)]
    for m, s in shares.items():
        assert abs(names.count(m) / 100_000 - s) < 0.01
    sm = M.BY_NAME["search_map"]
    assert (sm.goal_mask, sm.map_mask, sm.heat_on) == (1, 0, True)
    with pytest.raises(ValueError):
        M.habitat_mode_table(shares)  # the v1 names


def test_v1_mode_draws_unchanged():
    p = M.habitat_mode_table({"photo": .2, "map_goal": .3, "photo_map": .15, "explore_map": .15, "explore": .2})
    a = [M.draw_habitat_mode(p, np.random.default_rng([0, k])).name for k in range(200)]
    b = [M.HABITAT_MODES[int(np.random.default_rng([0, k]).choice(5, p=p))] for k in range(200)]
    assert a == b


def test_replace_block_and_model_kwargs():
    base = {"datasets": {"habitat_mapmad": {"a": 1}, "go_stanford": {"b": 2}}, "k": 1}
    top = {"datasets": {"_replace": True, "habitat_mapmad_v2": {"c": 3}}}
    assert deep_merge(base, top)["datasets"] == {"habitat_mapmad_v2": {"c": 3}}
    assert deep_merge(base, {"datasets": {"go_stanford": {"b": 5}}})["datasets"]["habitat_mapmad"] == {"a": 1}
    assert map_model_kwargs({"map_input": False, "map": {"tokens": 2}}) == {"map_input": False}
    assert map_model_kwargs({"map_input": True, "map": {}}) == {"map_input": True, "map_tokens": 1, "map_channels": 3}
    assert map_model_kwargs(dict(CFG, map_input=True))["map_tokens"] == 2


# --- the model -----------------------------------------------------------------------------------------------------
def models():
    from vint_train.mapmad.closed_loop.nomad_policy import build_nomad
    torch.manual_seed(0)
    old = build_nomad(dict(CFG, map_input=False)).eval()
    new = build_nomad(dict(CFG, map_input=True)).eval()
    return old, new


def test_v2_loads_nomad_and_maps_hidden_is_nomad(tmp_path):
    from vint_train.mapmad.lr_schedule import group_param_names
    from vint_train.mapmad.weights import load_official_into
    old, new = models()
    torch.save(old.state_dict(), tmp_path / "old.pth")
    report = load_official_into(new, str(tmp_path / "old.pth"))
    assert {k.split(".")[2] for k in report["new_keys"]} == {"near", "far"} and len(report["new_keys"]) == 36
    assert report["positional_encoding"]["model_rows"] == 7
    assert len(group_param_names(new)["map_encoder"]) == 36
    b = 4
    obs, goal = torch.randn(b, 12, 96, 96), torch.randn(b, 3, 96, 96)
    maps = torch.rand(b, 2, 4, 64, 64)
    with torch.no_grad():
        zero = new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=torch.ones(b, dtype=torch.long),
                   map_img=maps, input_map_mask=torch.zeros((b, 2), dtype=torch.long))
        for enc in new.vision_encoder.map_encoder.values():
            torch.nn.init.normal_(enc.proj.weight)
        for g in (0, 1):
            gm = torch.full((b,), g, dtype=torch.long)
            ref = old("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm)
            hid = new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps,
                      input_map_mask=torch.ones((b, 2), dtype=torch.long))
            assert torch.equal(hid, ref)  # both maps hidden = NoMaD's own computation
            outs = [new("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps,
                        input_map_mask=torch.tensor([m] * b)) for m in ([0, 0], [0, 1], [1, 0])]
            assert all(torch.isfinite(o).all() and (o - ref).abs().max() > 1e-4 for o in outs)
            assert (outs[1] - outs[2]).abs().max() > 1e-4  # near only != far only
    tokens, padding = new.vision_encoder.map_tokens(new.vision_encoder._encode_images(obs, goal),
                                                    torch.ones(b, dtype=torch.long), maps,
                                                    torch.tensor([[0, 1]] * b))
    assert tokens.shape[1] == 7 and padding[:, -2:].tolist() == [[False, True]] * b
    assert torch.isfinite(zero).all()


def test_v1_attention_share_is_the_old_formula():
    torch.manual_seed(3)
    layer = torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=True, norm_first=True)
    enc = torch.nn.TransformerEncoder(layer, 2).eval()
    tokens = torch.randn(3, 6, 16)
    padding = torch.zeros(3, 6, dtype=torch.bool)
    padding[1, 4] = True
    _, share = attention_share(enc, tokens, padding)
    x, ws = tokens, []
    for lay in enc.layers:  # the Phase 4 definition, written out
        h = lay.norm1(x)
        a, w = lay.self_attn(h, h, h, key_padding_mask=padding, need_weights=True, average_attn_weights=False)
        x = x + lay.dropout1(a)
        x = x + lay.dropout2(lay.linear2(lay.dropout(lay.activation(lay.linear1(lay.norm2(x))))))
        ws.append(w[..., -1])
    q = (~padding).float()
    q[:, -1] = 0
    old = ((torch.stack(ws, 1) * q[:, None, None, :]).sum(-1) / q.sum(-1)[:, None, None]).mean(dim=(1, 2))
    assert torch.allclose(share, old, atol=1e-7)
    _, two = attention_shares(enc, tokens, padding, 2)
    assert two.shape == (3, 2) and torch.all(two >= 0) and torch.all(two.sum(1) <= 1)


# --- offline arms ----------------------------------------------------------------------------------------------------
def test_target_shift_is_seeded_and_one_metre():
    from vint_train.mapmad.offline_scoring import arm_options, shift_of
    a, b = shift_of(17), shift_of(17)
    assert a == b and math.isclose(math.hypot(*a), 1.0) and shift_of(18) != a
    assert arm_options("near_only", 1).show == (True, False) and arm_options("maps", 1).show == (True, True)
    assert arm_options("frontier_zeroed", 1).zero_layers == ("frontier",)
    assert arm_options("zeroed", 1).zero_layers == ("obstacle", "explored", "frontier", "target")
    assert arm_options("zeroed", 1).show == (True, True)


def test_row3_control_arm_is_always_scored_with_maps():
    from vint_train.mapmad.offline_scoring import ARMS_V2, with_row3_control
    assert with_row3_control(["maps", "hidden"]) == ["maps", "zeroed", "hidden"]
    assert with_row3_control(["explore_maps", "hidden"]) == ["explore_maps", "hidden"]
    assert with_row3_control(list(ARMS_V2)) == list(ARMS_V2)


def test_compare_v2_row3_and_reports():
    from vint_train.mapmad.offline_v2 import compare_v2
    rng = np.random.default_rng(0)
    n = 400
    drives = np.repeat([f"d{i}" for i in range(40)], 10)
    hidden = 1.0 + rng.random(n)
    scores = {"drive": drives, "hidden.gc_action_loss": hidden, "maps.gc_action_loss": hidden - 0.5,
              "zeroed.gc_action_loss": hidden - 0.4, "near_only.gc_action_loss": hidden - 0.2}
    for arm in ("hidden", "maps", "zeroed", "near_only"):
        scores[f"{arm}.attn_share"] = np.full((n, 2), np.nan) if arm == "hidden" else np.full((n, 2), 0.2)
        scores[f"{arm}.loss_forward"] = scores[f"{arm}.gc_action_loss"]
        scores[f"{arm}.loss_sideways"] = scores[f"{arm}.gc_action_loss"]
    samples = [{"category": "bed" if i % 2 else "chair", "yaw_change_deg": 50.0 if i % 4 == 0 else 0.0} for i in range(n)]
    res = compare_v2(scores, samples)
    assert res["row3"]["pass"] and res["row3"]["control"] == "zeroed" and res["row3"]["diff"] == pytest.approx(-0.1)
    assert res["maps_vs_hidden"]["diff"] == pytest.approx(-0.5) and res["maps_vs_hidden"]["control"] == "hidden"
    no_zeroed = {k: v for k, v in scores.items() if not k.startswith("zeroed.")}
    assert not compare_v2(no_zeroed, samples)["row3"]["pass"]  # maps - hidden alone never passes row 3 (rules v2)
    assert set(res["per_category"]) == {"bed", "chair"} and res["forward_turn"]["turn"]["share"] == pytest.approx(0.25)
    assert res["attention_share"]["maps"] == {"near": 0.2, "far": 0.2}
    assert res["vs_maps"]["near_only"]["diff"] == pytest.approx(0.3)


# --- inference entry point -------------------------------------------------------------------------------------------
def test_inference_entry_point_loads_a_checkpoint_and_is_reproducible(tmp_path):
    from vint_train.mapmad.closed_loop.mapmad_policy import file_sha256
    from vint_train.mapmad.inference_v2 import MapMaDV2Inference
    _, new = models()
    for enc in new.vision_encoder.map_encoder.values():
        torch.nn.init.normal_(enc.proj.weight, std=0.05)
    path = tmp_path / "v2.pth"
    torch.save(new.state_dict(), path)
    cfg = dict(CFG, map_input=True)
    net = MapMaDV2Inference(str(path), cfg, torch.device("cpu"), checkpoint_sha256=file_sha256(str(path)))
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(4)]
    near, far = rng.random((4, 64, 64)).astype(np.float32), rng.random((4, 64, 64)).astype(np.float32)
    a = net.predict(frames, near, far, seed=5)
    b = net.predict(frames, near, far, seed=5)
    assert a["waypoints"].shape == (8, 8, 2) and a["distance"].shape == (8,)
    assert np.array_equal(a["waypoints"], b["waypoints"]) and np.allclose(a["waypoints_m"], a["waypoints"] * 0.25)
    hidden = net.predict(frames, near, far, near_shown=False, far_shown=False, seed=5)
    assert not np.allclose(hidden["waypoints"], a["waypoints"])
    with pytest.raises(RuntimeError):
        MapMaDV2Inference(str(path), cfg, torch.device("cpu"), checkpoint_sha256="0" * 64)


# --- data on the tiny split (skipped where the data disk is not mounted) -------------------------------------------
SMOKE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "mapmad_v2_smoke.yaml")


def tiny():
    from vint_train.mapmad.config import load_config
    from vint_train.mapmad.dataset import build_mapmad_dataset
    cfg = load_config(SMOKE)
    if not os.path.isdir(cfg["map"]["frontier_cache"]):
        pytest.skip("Phase 10 data not mounted")
    return cfg, build_mapmad_dataset(cfg, "habitat_mapmad_v2", "train", train=True)


def test_v2_samples_modes_pools_and_options():
    from vint_train.mapmad.map_sample_v2 import MapOptions
    cfg, ds = tiny()
    assert len(ds.pools["search"]) and len(ds.pools["explore"])
    search_i = int(ds.pools["search"][len(ds.pools["search"]) // 2])
    drive, t, _ = ds.index_to_data[search_i]
    rng = lambda: np.random.default_rng(0)  # noqa: E731
    s = ds.habitat_sample(drive, t, M.BY_NAME["search_map"], rng())
    assert tuple(s[7].shape) == (2, 4, 64, 64) and s[9].tolist() == [0, 0] and int(s[10]) == 7
    near = ds.habitat_sample(drive, t, M.BY_NAME["search_map"], rng(), options=MapOptions(show=(True, False)))
    assert near[9].tolist() == [0, 1] and float(near[7][1].abs().sum()) == 0 and torch.equal(near[7][0], s[7][0])
    z = ds.habitat_sample(drive, t, M.BY_NAME["search_map"], rng(), options=MapOptions(zero_layers=("frontier",)))
    assert float(z[7][:, 2].abs().sum()) == 0 and torch.equal(z[7][:, 3], s[7][:, 3])
    hid = ds.habitat_sample(drive, t, M.BY_NAME["explore"], rng())
    assert hid[9].tolist() == [1, 1] and float(hid[7].abs().sum()) == 0
    for k in range(30):  # explore_map samples always come from exploration drives, with an empty target layer
        mode = M.BY_NAME["explore_map"]
        i = ds.pool_index(search_i, mode, np.random.default_rng(k))
        assert ds.kind(ds.index_to_data[i][0]) == "explore"
    assert torch.equal(ds.get(5, 99)[7], ds.get(5, 99)[7])  # same key, same sample


def test_frozen_lists_have_unique_ids_and_samples():
    path = "/outputs/mapmad/p10_v2_train/offline/lists/search_list.json"
    if not os.path.exists(path):
        pytest.skip("Phase 10 lists not mounted")
    from vint_train.mapmad.offline_list import load_list
    for name in ("search_list.json", "explore_list.json"):
        samples = load_list(os.path.join(os.path.dirname(path), name))["samples"]
        assert len({s["id"] for s in samples}) == len(samples)
        assert len({(s["drive"], s["t"]) for s in samples}) == len(samples)


def test_v1_checkpoint_still_loads_and_runs():
    from vint_train.mapmad.config import load_config
    from vint_train.mapmad.weights import load_checkpoint
    ckpt = "/outputs/mapmad/p4_train/C_2026_10_11_05_21_43/ema_19.pth"
    if not os.path.exists(ckpt):
        pytest.skip("v1 checkpoint not mounted")
    cfg = load_config(os.path.join(os.path.dirname(SMOKE), "mapmad_p4_C.yaml"), None)
    model = load_checkpoint(cfg, ckpt, torch.device("cpu"), map_input=True)
    assert model.vision_encoder.n_map_tokens == 1
    with torch.no_grad():
        out = model("vision_encoder", obs_img=torch.randn(2, 12, 96, 96), goal_img=torch.randn(2, 3, 96, 96),
                    input_goal_mask=torch.ones(2, dtype=torch.long), map_img=torch.rand(2, 3, 64, 64),
                    input_map_mask=torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 256) and torch.isfinite(out).all()


def test_target_groups_split_and_learned_effect():
    from vint_train.mapmad.target_groups import group_report, markdown, target_group
    assert [target_group(False, True), target_group(True, True), target_group(True, False)] == \
        ["empty", "in_view", "not_in_view"]
    n = 120
    drives = np.repeat([f"d{i}" for i in range(12)], 10)
    groups = np.array(["empty", "in_view", "not_in_view"] * 40)
    maps = np.full(n, 1.0)
    scores = {"drive": drives, "maps.gc_action_loss": maps, "hidden.gc_action_loss": maps + 2.0,
              "target_zeroed.gc_action_loss": maps + np.where(groups == "in_view", 0.5, 0.0),
              "target_shifted.gc_action_loss": maps + 0.1, "clean.gc_action_loss": maps - 0.1}
    pre = {"maps.gc_action_loss": maps + 1.0, "hidden.gc_action_loss": maps + 2.0}
    rep = group_report(scores, pre, groups)
    assert rep["in_view"]["n_samples"] == 40 and rep["in_view"]["n_drives"] == 12
    assert rep["in_view"]["target_zeroed"]["diff"] == pytest.approx(0.5)
    assert rep["empty"]["target_zeroed"]["diff"] == pytest.approx(0.0)
    assert rep["all"]["learned_map_effect"]["diff"] == pytest.approx(-1.0)  # (1 - 3) - (2 - 3)
    assert "| in_view | 40 / 12 |" in markdown(rep)


def test_seen_before_out_of_view_rule_and_decision():
    from vint_train.mapmad.target_groups import decide, seen_frames, seen_out_of_view, seen_report
    det = np.array([30, 31, 50])
    assert not seen_out_of_view(det, 30) and not seen_out_of_view(det, 31) and seen_out_of_view(det, 32)
    assert not seen_out_of_view(det, 50) and not seen_out_of_view(np.array([], int), 40)
    assert seen_frames(det, 100) == [32, 33] + list(range(34, 50)) + list(range(51, 60))  # t <= 100 - 41
    assert decide({"diff_ci_drive": [0.001, 0.02]}) == "A" and decide({"diff_ci_drive": [-0.001, 0.02]}) == "C"
    drives = np.repeat([f"d{i}" for i in range(10)], 5)
    maps = np.ones(50)
    rep = seen_report({"drive": drives, "maps.gc_action_loss": maps, "target_zeroed.gc_action_loss": maps + 0.2,
                       "target_shifted.gc_action_loss": maps})
    assert rep["decision"] == "A" and rep["n_drives"] == 10


def test_guard_margin():
    from vint_train.mapmad.target_groups import guard
    drives = np.repeat([f"d{i}" for i in range(10)], 5)
    ref = {"ids": np.arange(50), "drive": drives, "maps.gc_action_loss": np.ones(50)}
    ok = {"ids": np.arange(50), "drive": drives, "maps.gc_action_loss": np.full(50, 1.03)}
    bad = {"ids": np.arange(50), "drive": drives, "maps.gc_action_loss": np.full(50, 1.06)}
    assert guard(ok, ref)["pass"] and not guard(bad, ref)["pass"]


def test_seen_oversampling_draws_and_off_switch():
    from vint_train.mapmad.dataset_v2 import MapMaDV2Dataset
    ds = object.__new__(MapMaDV2Dataset)
    search = np.zeros(1000, bool)
    search[:500] = True
    ds.pools = {"search": np.flatnonzero(search), "explore": np.flatnonzero(~search), "seen": np.arange(10)}
    ds.in_pool = {"search": search, "explore": ~search}
    mode = M.BY_NAME["search_map"]
    ds.seen_p = 0.25
    rng = np.random.default_rng(0)
    draws = np.array([ds.pool_index(int(i), mode, rng) for i in rng.integers(0, 1000, 20000)])
    assert abs((draws < 10).mean() - (0.25 + 0.75 * 10 / 500)) < 0.01 and search[draws].all()
    ds.seen_p = 0.0  # off: the old rule, no extra random draw
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    got = [ds.pool_index(i, mode, a) for i in range(600)]
    kept = [i if search[i] else int(ds.pools["search"][int(b.integers(500))]) for i in range(600)]
    assert got == kept
