"""Phase 5 statistics and failure-type helpers (exam_stats.py, exam_analysis.py)."""

import numpy as np

from vint_train.mapmad.closed_loop import exam_analysis as A
from vint_train.mapmad.closed_loop import exam_stats as S
from vint_train.mapmad.closed_loop.exam_runner import noise_seed


def test_cluster_bootstrap_keeps_homes_together():
    # home h1: diffs all +1, home h2: all -1, unequal sizes -> the mean is weighted by episodes
    v = np.array([1.0, 1.0, 1.0, -1.0])
    mean, lo, hi, p = S.cluster_bootstrap(v, ["h1", "h1", "h1", "h2"], n_boot=2000, seed=0)
    assert mean == 0.5
    # a resample is +1 (only h1), -1 (only h2) or 0.5 (both, 3:1): never anything else
    assert lo in (-1.0, 0.5) and hi in (0.5, 1.0) and 0.2 < p < 0.3


def test_one_sided_p_for_a_clear_effect():
    rng = np.random.default_rng(1)
    v = rng.normal(0.3, 0.1, 200)
    homes = [f"h{i % 20}" for i in range(200)]
    mean, lo, hi, p = S.cluster_bootstrap(v, homes)
    assert p == 0.0 and lo > 0.2


def test_paired_uses_common_episodes():
    a = {"e1": 1.0, "e2": 0.0, "e3": 1.0}
    b = {"e1": 0.0, "e2": 0.0, "e4": 1.0}
    r = S.paired(a, b, {"e1": "h1", "e2": "h2", "e3": "h1", "e4": "h2"}, n_boot=500)
    assert r["n"] == 2 and r["diff"] == 0.5 and r["homes"] == 2


def test_revisit_share():
    straight = [[0.1 * i, 0.0, 0.0] for i in range(40)]
    share, entries = A.revisit_share(straight)
    assert share == 0.0 and entries == 16
    loop = straight + straight[::-1] + straight
    share, _ = A.revisit_share(loop)
    assert share > 0.5


def test_wrong_instance_needs_same_storey_and_1m():
    other = [{"box_center": [2.0, 0.5, 0.0], "box_size": [0.4, 1.0, 0.4]}]  # bottom at y = 0
    assert A.wrong_instance_step([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], other) == 1  # 0.8 m from the box
    assert A.wrong_instance_step([[1.0, 3.0, 0.0]], other) is None  # the floor above


def test_noise_seed_same_for_every_arm_differs_per_seed_index():
    assert noise_seed("s", "ep1", 0) == noise_seed("s", "ep1", 0)
    assert len({noise_seed("s", "ep1", k) for k in range(3)}) == 3


def test_heat_bearing_sign_and_sectors():
    from vint_train.mapmad.closed_loop.exam_stuck import bearing_deg, sector

    # yaw 0 looks along -z; -x is to the left
    assert abs(bearing_deg([0, 0, 0], 0.0, (0.0, -2.0))) < 1e-9 and sector(0.0) == "ahead"
    assert abs(bearing_deg([0, 0, 0], 0.0, (-2.0, 0.0)) - 90.0) < 1e-9 and sector(90.0) == "side-left"
    assert sector(bearing_deg([0, 0, 0], 0.0, (1.0, 2.0))) == "behind-right"
