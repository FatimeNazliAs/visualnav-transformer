"""Episode helpers, the frozen-file fingerprint, object boxes and the category name map."""

import json
from collections import Counter

import numpy as np
import pytest

from mapmad_sim import categories, config, episodes, objectnav
from mapmad_sim.objects import name_matches


def test_box_distance_xz():
    center, size = [0.0, 1.0, 0.0], [2.0, 2.0, 1.0]
    assert episodes.box_distance_xz([0.5, 9.0, 0.2], center, size) == 0.0  # inside (height ignored)
    assert episodes.box_distance_xz([4.0, 0.0, 0.0], center, size) == pytest.approx(3.0)
    assert episodes.box_distance_xz([4.0, 0.0, 4.5], center, size) == pytest.approx(5.0)


def test_fingerprint_is_order_and_spacing_independent_but_value_sensitive():
    a = [{"x": 1, "y": [1.5, 2]}, {"x": 2}]
    b = json.loads(json.dumps(a, indent=4))
    assert episodes.fingerprint(a) == episodes.fingerprint(b)
    assert episodes.fingerprint(a) != episodes.fingerprint([{"x": 1, "y": [1.5, 2.0001]}, {"x": 2}])


def test_load_episodes_refuses_a_changed_file(tmp_path):
    eps = [{"episode_id": "e0", "start_yaw": 0.1}]
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"fingerprint": episodes.fingerprint(eps), "episodes": eps}))
    assert episodes.load_episodes(path)[1] == episodes.fingerprint(eps)
    eps[0]["start_yaw"] = 0.2
    path.write_text(json.dumps({"fingerprint": "0" * 64, "episodes": eps}))
    with pytest.raises(ValueError):
        episodes.load_episodes(path)


def test_floor_names_match_whole_words():
    words = ["floor", "carpet", "rug", "mat", "doormat"]
    assert name_matches("bath mat", words) and name_matches("floor", words) and name_matches("doormat", words)
    assert not name_matches("information", words) and not name_matches("mattress", words)


def test_name_map_needs_count_and_purity():
    counts = {"chair": Counter(chair=50), "tv": Counter(tv_monitor=30, chair=1), "box": Counter(plant=10, bed=10),
              "bidet": Counter(toilet=3)}
    assert categories.name_map(counts, min_count=20, min_purity=0.95) == {"chair": "chair", "tv": "tv_monitor"}


def test_category_map_file_is_clean():
    mapping = categories.load_map()
    assert set(mapping.values()) <= set(categories.CATEGORIES)
    assert mapping["chair"] == "chair" and mapping["tv"] == "tv_monitor" and "bidet" not in mapping


TRAIN_HOME = "00006-HkseAnWCgqk"
TRAIN_CONTENT = objectnav.content_file("train", TRAIN_HOME)
needs_hm3d = pytest.mark.skipif(not TRAIN_CONTENT.exists(), reason="HM3D / ObjectNav mounts missing")


@needs_hm3d
def test_objectnav_goals_have_categories_and_view_points():
    goals = objectnav.home_goals("train", TRAIN_HOME)
    assert goals and {g.category for g in goals} <= set(categories.CATEGORIES)
    assert all(len(g.view_points) > 0 and len(g.view_points[0]) == 3 and len(g.position) == 3 for g in goals)
    assert [g.object_id for g in goals] == sorted(g.object_id for g in goals)


def test_objectnav_category_keys():
    assert objectnav.category_of("abc.basis.glb_tv_monitor") == "tv_monitor"
    assert objectnav.category_of("abc.basis.glb_chair") == "chair"


@needs_hm3d
def test_object_boxes_come_from_the_obb_and_match_objectnav():
    """habitat-sim 0.3.3's aabb includes the world origin; our box must match ObjectNav's object positions."""
    from mapmad_sim.objects import object_box
    from mapmad_sim.robot import LimoSim, RobotSpec

    positions = {g.object_id: g.position for g in objectnav.home_goals("train", TRAIN_HOME)}
    with LimoSim("train", TRAIN_HOME, RobotSpec.from_config(config.robot()), semantic=True) as robot:
        objects = {o.semantic_id: o for o in robot.sim.semantic_scene.objects if o is not None}
        for obj_id in sorted(positions)[:10]:
            center, size = object_box(objects[obj_id])
            assert np.allclose(center, positions[obj_id], atol=0.01)
            assert np.all(size < 4.0)
