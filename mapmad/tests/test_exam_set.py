"""Phase 5 exam-set helpers that need no simulator (mapmad_sim/exam_set.py)."""

import pytest

from mapmad_sim.exam_set import bin_quotas, distance_bin, draw, episode_id, freeze, photo_order

BOX_C, BOX_S = [0.0, 0.5, 0.0], [1.0, 1.0, 1.0]  # box edges at x, z = +-0.5


def test_photo_order_nearest_vs_one_metre():
    vps = [[0.6, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]]  # 0.1 m, 1.0 m, 2.0 m from the box
    assert photo_order(vps, BOX_C, BOX_S, None) == [0, 1, 2]  # block item 8 as written: nearest first
    assert photo_order(vps, BOX_C, BOX_S, 1.0) == [1, 0, 2]  # 1a: closest to 1.0 m first (0.9 m off beats 1.0 m)


def test_bins_and_quotas():
    assert distance_bin(3.0, [3, 6, 10, 15]) == 0 and distance_bin(15.0, [3, 6, 10, 15]) == 2
    assert bin_quotas(300, 3) == [100, 100, 100] and bin_quotas(100, 3) == [34, 33, 33]


def test_draw_respects_home_cap_and_moves_shortfall():
    kept = [{"episode_id": f"h{h}-{b}-{i}", "home": f"h{h}", "distance_bin": b}
            for h in range(4) for b in range(3) for i in range(5 if b else 1)]
    chosen, rep = draw(kept, 12, 3, 3, seed=0)
    assert len(chosen) == 12 and max(rep["chosen_per_home"].values()) <= 3
    assert rep["chosen_per_bin"][0] == 4 and rep["moved"] == []
    chosen, rep = draw(kept, 30, 10, 3, seed=0)  # bin 0 has only 4 episodes -> its shortfall goes to bin 1
    assert rep["chosen_per_bin"][0] == 4 and rep["moved"][0]["to_bin"] == 1
    assert len(chosen) + rep["missing"] == 30 and max(rep["chosen_per_home"].values()) <= 10  # caps may leave a gap



def test_episode_ids_unique_across_categories():
    # ObjectNav val reuses episode ids within a home, once per category (bug found 2026-10-12)
    ids = {episode_id("exam", "00800-TEEsavR23oF", c, "2") for c in ("sofa", "toilet", "chair")}
    assert ids == {"p5e-00800-sofa-2", "p5e-00800-toilet-2", "p5e-00800-chair-2"}


def test_freeze_refuses_duplicate_ids(tmp_path):
    eps = [{"episode_id": "p5e-00800-2", "x": 1}, {"episode_id": "p5e-00800-2", "x": 2}]
    with pytest.raises(ValueError, match="duplicate episode ids"):
        freeze(tmp_path / "episodes.json", eps, {})
    assert not (tmp_path / "episodes.json").exists()
