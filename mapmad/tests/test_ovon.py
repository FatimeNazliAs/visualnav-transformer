"""HM3D-OVON converter (Phase 8.0): the converted file is read by the unchanged objectnav.py loader."""

import gzip
import json
from pathlib import Path

from mapmad_sim import objectnav, ovon

SCENE = "4ok3usBNeis"
HOME = f"00877-{SCENE}"


def ovon_file(root: Path, split: str) -> Path:
    data = {
        "goals_by_category": {
            f"{SCENE}.basis.glb_kitchen cabinet": [
                {"object_category": "kitchen cabinet", "object_id": "kitchen cabinet_2", "position": [1.0, 0.0, 2.0],
                 "view_points": [{"agent_state": {"position": [1.5, -0.5, 2.0], "rotation": [0, 0, 0, 1]}, "iou": 0.3}]}],
            f"{SCENE}.basis.glb_bench": [
                {"object_category": "bench", "object_id": "bench_1", "position": [0.0, 0.0, 0.0],
                 "view_points": [{"agent_state": {"position": [0.5, -0.5, 0.0], "rotation": [0, 0, 0, 1]}, "iou": 0.2}]}],
        },
        "episodes": [
            {"episode_id": "7", "scene_id": f"hm3d/val//{HOME}/{SCENE}.basis.glb", "start_position": [0.1, -0.5, 0.2],
             "start_rotation": [0, 0.7071068, 0, 0.7071068], "object_category": "bench",
             "children_object_categories": ["towel"]}],
        "category_to_task_category_id": {}, "category_to_scene_annotation_category_id": {},
    }
    path = root / "hm3d" / split / "content" / f"{SCENE}.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt") as f:
        json.dump(data, f)
    return path


def test_converted_file_reads_with_objectnav_loader(tmp_path: Path) -> None:
    ovon_file(tmp_path, "val_seen")
    out = tmp_path / "converted"
    files = ovon.write_converted(tmp_path, out, "val_seen")
    assert [f.name for f in files] == [f"{SCENE}.json.gz"]
    goals = objectnav.home_goals("val_seen", HOME, out)
    assert [(g.object_id, g.category) for g in goals] == [(0, "bench"), (1, "kitchen cabinet")]
    assert goals[1].view_points == [[1.5, -0.5, 2.0]]
    eps = objectnav.home_episodes("val_seen", HOME, out)
    assert eps[0].home == HOME and eps[0].category == "bench" and abs(eps[0].start_yaw - 1.5707963) < 1e-6
    raw = ovon.read_content(files[0])
    assert raw["goals_by_category"][f"{SCENE}.basis.glb_bench"][0]["ovon_object_id"] == "bench_1"


def test_split_summary_ids_and_children(tmp_path: Path) -> None:
    ovon_file(tmp_path, "val_unseen")
    s = ovon.split_summary(tmp_path, "val_unseen")
    assert s["homes"] == [HOME] and s["episodes"] == 1 and s["unique_ids"]
    assert s["episodes_with_children"] == 1 and s["categories"] == {"bench": 1}
    assert ovon.episode_uid("val_unseen", HOME, "7") == "ovon-val_unseen-00877-7"
