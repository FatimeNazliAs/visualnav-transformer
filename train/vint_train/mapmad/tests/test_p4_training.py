"""Phase 4: lr schedule + parameter groups, frozen subsets, trace, selection rule, selection-bias report, resume
state. CPU only."""

import math

import numpy as np
import pytest
import torch

from vint_train.mapmad.lr_schedule import WarmupCosine, build_optimizer, group_param_names, lr_factor
from vint_train.mapmad.select_checkpoint import select
from vint_train.mapmad.check_trace import TRACE_COLUMNS, append_trace, read_trace
from vint_train.mapmad.training_checks import (draw_gostanford_subset, draw_offline_subset, gostanford_report,
                                               subset_body_sha)

S, E = 782, 20  # steps per epoch (200,000 / 256), epochs


@pytest.fixture(scope="module")
def p4_model():
    """The real Phase 4 MapMaD model (config/mapmad_p4.yaml), random weights."""
    from vint_train.mapmad.closed_loop.nomad_policy import build_nomad
    from vint_train.mapmad.config import load_config
    cfg = load_config("config/mapmad_p4.yaml", "config/defaults.yaml")
    torch.manual_seed(0)
    return build_nomad(dict(cfg, map_input=True)), cfg


def cosine(n: int) -> float:
    """Confirmation item 3: 1-based epoch n >= 2."""
    return 0.5 * (1 + math.cos(math.pi * (n - 2) / 19))


@pytest.mark.parametrize("pretrained_lr", [1e-4, 1e-5])  # run A, run B
def test_lr_at_confirmed_points_for_both_groups(p4_model, pretrained_lr):
    model, cfg = p4_model
    opt = build_optimizer(model, dict(cfg, pretrained_lr=pretrained_lr))
    sched = WarmupCosine(opt, S, E, 0.01)
    expect = {  # (epoch 0-based, step) -> factor
        (0, 0): 0.01, (0, 391): 0.01 + 0.99 * 391 / 782, (0, 781): 0.01 + 0.99 * 781 / 782,
        (1, 0): 1.0,                      # global step 782 = epoch 2 = peak
        (1, 500): cosine(2), (10, 0): cosine(11), (19, 0): cosine(20), (19, 781): cosine(20)}
    for (e, s), f in expect.items():
        lrs = sched.apply(e, s)
        assert lrs["map_encoder"] == pytest.approx(1e-4 * f, rel=1e-12)
        assert lrs["pretrained"] == pytest.approx(pretrained_lr * f, rel=1e-12)
    # the numbers written in the confirmation
    assert lr_factor(0, 0, S, E, 0.01) * 1e-4 == pytest.approx(1e-6)
    assert lr_factor(0, 391, S, E, 0.01) * 1e-4 == pytest.approx(5.05e-5)
    assert lr_factor(10, 0, S, E, 0.01) * 1e-4 == pytest.approx(0.54e-4, abs=0.005e-4)
    assert lr_factor(19, 0, S, E, 0.01) * 1e-4 == pytest.approx(7e-7, abs=0.3e-7)
    if pretrained_lr == 1e-5:
        assert sched.apply(0, 0)["pretrained"] == pytest.approx(1e-7)


def test_param_groups_are_the_18_map_encoder_keys_vs_the_rest(p4_model):
    model, cfg = p4_model
    names = group_param_names(model)
    assert len(names["map_encoder"]) == 18
    assert all(n.startswith("vision_encoder.map_encoder.") for n in names["map_encoder"])
    assert not any("map_encoder" in n for n in names["pretrained"])
    all_names = [n for n, _ in model.named_parameters()]
    assert sorted(names["map_encoder"] + names["pretrained"]) == sorted(all_names)
    opt = build_optimizer(model, dict(cfg, pretrained_lr=1e-5))
    ids = [{id(p) for p in g["params"]} for g in opt.param_groups]
    params = dict(model.named_parameters())
    assert ids[0] == {id(params[n]) for n in names["map_encoder"]}
    assert ids[1] == {id(params[n]) for n in names["pretrained"]}
    assert [g["name"] for g in opt.param_groups] == ["map_encoder", "pretrained"]
    assert sum(p.numel() for p in opt.param_groups[0]["params"]) == pytest.approx(766e3, rel=0.01)


def test_gostanford_subset_is_seeded_sorted_and_unique():
    a, b = draw_gostanford_subset(5000), draw_gostanford_subset(5000)
    assert a == b and a == sorted(set(a)) and len(a) == 1000 and 0 <= a[0] and a[-1] < 5000
    assert draw_gostanford_subset(5000, seed=1) != a


def test_offline_subset_keeps_drives_whole_and_reaches_n():
    rng = np.random.default_rng(0)
    samples, i = [], 0
    for d in range(711):
        for _ in range(int(rng.integers(7, 9))):
            samples.append({"drive": f"d{d:03d}", "id": i})
            i += 1
    ids = draw_offline_subset(samples)
    assert 1000 <= len(ids) < 1008 and ids == draw_offline_subset(samples)
    chosen = set(ids)
    for d in {s["drive"] for s in samples if s["id"] in chosen}:
        assert all(s["id"] in chosen for s in samples if s["drive"] == d)  # whole drive


def test_subset_sha_ignores_only_its_own_field():
    body = {"ids": [1, 2], "seed": 4004}
    sha = subset_body_sha(body)
    assert subset_body_sha(dict(body, sha256=sha)) == sha and subset_body_sha(dict(body, seed=1)) != sha


def trace_rows(deltas):
    return [{"epoch": e, "ema_file": f"ema_{e - 1}.pth", "no_harm_delta": d} for e, d in deltas.items()]


def test_selection_rule_lines():
    a = trace_rows({10: 0.0, 12: 0.05, 14: 0.11, 16: 0.07, 18: 0.09, 20: 0.10})
    b = trace_rows({12: 0.01, 20: 0.02})
    assert select(a, b) == {"line": 1, "run": "A", "epoch": 16, "ema_file": "ema_15.pth", "no_harm_delta": 0.07}
    a_bad = trace_rows({10: 0.0, 12: 0.081, 20: 0.2})  # epoch 10 is below the floor
    assert select(a_bad, b)["line"] == 2 and select(a_bad, b)["epoch"] == 20
    assert select(a_bad, trace_rows({8: 0.0, 14: 0.5}))["line"] == 3
    assert select(trace_rows({20: 0.08}), b)["epoch"] == 20  # <= is inclusive, unrounded
    assert select(trace_rows({20: 0.0800001}), [])["line"] == 3


def test_trace_roundtrip(tmp_path):
    path = str(tmp_path / "t.csv")
    row = {k: 0.5 for k in TRACE_COLUMNS}
    row.update(run="A", epoch=2, ema_file="ema_1.pth", n_gs=1000, n_map=1000)
    append_trace(path, row)
    append_trace(path, dict(row, epoch=4))
    back = read_trace(path)
    assert [r["epoch"] for r in back] == [2, 4] and back[0]["no_harm_delta"] == 0.5 and back[0]["run"] == "A"


def test_gostanford_report_splits_inside_and_outside():
    n = 50
    items = [[f"traj{i // 5}", i, 5] for i in range(n)]
    official = np.ones(n)
    mapmad = np.ones(n)
    ids = list(range(10))
    mapmad[10:] += 0.2  # worse only outside the subset
    sub = {"harness_n": n, "harness_items": items, "ids": ids, "sha256": "x"}
    rep = gostanford_report(official, mapmad, sub)
    assert rep["inside_subset"]["diff"] == pytest.approx(0.0) and rep["outside_subset"]["diff"] == pytest.approx(0.2)
    assert rep["outside_flag"] and rep["full"]["n"] == n and rep["outside_subset"]["n"] == 40
    lo, hi = rep["full"]["diff_ci_trajectory"]
    assert lo <= rep["full"]["diff"] <= hi


def test_resume_state_roundtrip(tmp_path):
    from diffusers.training_utils import EMAModel

    from vint_train.mapmad.run_state import load_resume_state, save_ema_and_resume

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    ema = EMAModel(model=model, power=0.75)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()
        ema.step(model)
    save_ema_and_resume(str(tmp_path), 4, model, ema, opt)
    after_save = torch.rand(3)  # RNG continues from the saved state
    model2 = torch.nn.Linear(4, 2)
    ema2 = EMAModel(model=model2, power=0.75)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    assert load_resume_state(str(tmp_path / "latest.pth"), model2, ema2, opt2) == 5
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), model2.state_dict().values()))
    assert all(torch.equal(a, b) for a, b in zip(ema.averaged_model.state_dict().values(),
                                                 ema2.averaged_model.state_dict().values()))
    assert ema2.optimization_step == 3 and opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
    assert torch.equal(torch.rand(3), after_save)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ema_4.pth", "latest.pth"]
