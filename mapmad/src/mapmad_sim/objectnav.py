"""ObjectNav HM3D v2 goal files: which objects of a home are targets, of which category, and from where they are seen.

One file per home: <objectnav>/<split>/content/<scene>.json.gz, whose `goals_by_category` maps
"<scene>.basis.glb_<category>" to goal objects (object id, position, view points = agent poses that see the object),
and whose `episodes` hold the official starts (position, rotation quaternion [x, y, z, w], object category).
The only reader of these files (categories.py, episodes.py, exam_set.py and scripts/p1_replay.py use it).
"""

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from mapmad_sim import config


@dataclass
class Goal:
    """One ObjectNav goal object of a home."""

    object_id: int
    category: str
    position: List[float]  # object centre (= its OBB centre, see objects.py)
    view_points: List[List[float]]  # agent positions that see the object


def content_file(split: str, home: str, root: Optional[Path] = None) -> Path:
    """The goal file of a home, e.g. content_file("train", "00081-5biL7VEkByM") -> .../train/content/5biL7VEkByM.json.gz."""
    root = root or config.paths()["objectnav"]
    return root / split / "content" / f"{home.split('-', 1)[1]}.json.gz"


def category_of(key: str) -> str:
    """'<scene>.basis.glb_tv_monitor' -> 'tv_monitor', '<scene>.basis.glb_chair' -> 'chair'."""
    return "tv_monitor" if key.endswith("tv_monitor") else key.rsplit("_", 1)[-1]


def read_goals(path: Path) -> List[Goal]:
    """Every goal object of one goal file, ordered by object id (ties keep the file's category order, sorted)."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    goals = []
    for key, items in sorted(data["goals_by_category"].items()):
        for g in items:
            goals.append(Goal(int(g["object_id"]), category_of(key), list(g["position"]),
                              [v["agent_state"]["position"] for v in g["view_points"]]))
    return sorted(goals, key=lambda g: g.object_id)


def home_goals(split: str, home: str, root: Optional[Path] = None) -> List[Goal]:
    """Goal objects of one home (see read_goals)."""
    return read_goals(content_file(split, home, root))


def has_goals(split: str, home: str, root: Optional[Path] = None) -> bool:
    return content_file(split, home, root).exists()


@dataclass
class OfficialEpisode:
    """One official ObjectNav episode: where it starts and which category it asks for."""

    episode_id: str
    home: str  # HM3D folder name, e.g. "00877-4ok3usBNeis"
    start_position: List[float]
    start_yaw: float  # Habitat yaw (rotation about +y) of start_rotation
    category: str


def yaw_of_quaternion(q: List[float]) -> float:
    """Habitat yaw of a rotation about +y given as [x, y, z, w] (ObjectNav's start_rotation), in (-pi, pi]."""
    yaw = 2.0 * math.atan2(q[1], q[3])
    return math.atan2(math.sin(yaw), math.cos(yaw))


def home_episodes(split: str, home: str, root: Optional[Path] = None) -> List[OfficialEpisode]:
    """The official episodes of one home, in file order."""
    with gzip.open(content_file(split, home, root), "rt") as f:
        data = json.load(f)
    return [OfficialEpisode(str(e["episode_id"]), e["scene_id"].split("/")[-2], [float(v) for v in e["start_position"]],
                            yaw_of_quaternion(e["start_rotation"]), e["object_category"]) for e in data["episodes"]]
