"""Phase 13 pure parts (Habitat side): MP3D names and ids, generated view-point candidates, the Phase 13 run plan.

    docker exec -w /app/visualnav-transformer naz_mapmad_habitat python -m pytest mapmad/tests/test_p13.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

from mapmad_sim import config, objectnav
from mapmad_sim.mp3d_goals import VpRule, box_distance_xz, grid_candidates
from mapmad_sim.search_set import set_episode_id

REPO = Path(__file__).resolve().parents[2]


def test_scene_id_hm3d_and_mp3d():
    assert config.scene_id("00800-TEEsavR23oF") == "TEEsavR23oF"
    assert config.scene_id("2azQ1b91cZZ") == "2azQ1b91cZZ"
    assert config.is_mp3d("mp3d") and not config.is_mp3d("val")


def test_category_of_keeps_hm3d_rule_and_reads_mp3d_keys():
    assert objectnav.category_of("TEEsavR23oF.basis.glb_tv_monitor") == "tv_monitor"
    assert objectnav.category_of("TEEsavR23oF.basis.glb_chair") == "chair"
    assert objectnav.category_of("2azQ1b91cZZ.glb_chest_of_drawers") == "chest_of_drawers"
    assert objectnav.category_of("2azQ1b91cZZ.glb_tv_monitor") == "tv_monitor"


def test_episode_ids_hm3d_unchanged_mp3d_full_scene():
    assert set_episode_id("p12e", "00839-zt1RVoi7PcG", "bed", "0") == "p12e-00839-bed-0"
    assert set_episode_id("p13b", "2azQ1b91cZZ", "tv_monitor", "g007") == "p13b-2azQ1b91cZZ-tv_monitor-g007"


def test_box_distance_and_grid_candidates():
    c, s = np.array([1.0, 0.5, 2.0]), np.array([0.4, 1.0, 0.6])
    assert box_distance_xz([1.1, 0, 2.1], c, s) == 0.0
    assert box_distance_xz([1.2 + 0.5, 0, 2.0], c, s) == pytest.approx(0.5)
    pts = grid_candidates(c, s, VpRule(radius_m=0.5, grid_m=0.1))
    assert len(pts) > 0
    assert all(box_distance_xz(p, c, s) <= 0.5 + 1e-9 for p in pts)
    assert np.allclose(pts[:, 1], 0.0)  # asked at the box bottom
    assert np.allclose(np.round(pts[:, [0, 2]] / 0.1), pts[:, [0, 2]] / 0.1)  # on the world lattice


def test_p13_plan_order_and_runs(tmp_path):
    from mapmad_sim.run_plan_p13 import P13Plan

    out = tmp_path / "outputs"
    base = out / "p13_claim"
    for name, n in (("examb_set", 5), ("dev_set", 4)):
        (base / name).mkdir(parents=True)
        (base / name / "episodes.json").write_text(json.dumps({"fingerprint": "x", "episodes": [
            {"episode_id": f"{name}-{k}", "home": "h"} for k in range(n)]}))
    (out / "p12_search_exam" / "exam_set").mkdir(parents=True)
    (out / "p12_search_exam" / "exam_set" / "episodes.json").write_text(json.dumps({"fingerprint": "y", "episodes": [
        {"episode_id": f"e{k}", "home": "h"} for k in range(3)]}))
    (base / "P12_DEV_CHOICES.json").write_text(json.dumps({"headline": "v21", "tau": 0.45}))
    plan = P13Plan.load(REPO / "mapmad" / "configs" / "p13_claim.yaml", out)
    labels = [c.label for c in plan.calls("exams")]
    assert labels == ["hm3d_pair", "examb_v21", "examb_v21b", "examb_N", "examb_F", "examb_VF", "hm3d_VF"]
    runs = {c.label: plan.runs(c) for c in plan.calls("exams")}
    assert runs["hm3d_pair"] == 3 * 2 * 3 and runs["examb_F"] == 5 and runs["examb_VF"] == 5 and runs["hm3d_VF"] == 3
    assert {c.label: plan.runs(c) for c in plan.calls("dev")} == {"dev": 4 * 3, "dev_VF": 4}
    c, eps = plan.timing_call()
    assert c.label == "hm3d_pair" and eps == ["e0", "e1", "e2"]
