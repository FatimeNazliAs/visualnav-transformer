"""ObjectNav HM3D v2 goal files: which objects of a home are targets, of which category, and from where they are seen.

One file per home: <objectnav>/<split>/content/<scene>.json.gz, whose `goals_by_category` maps
"<scene>.basis.glb_<category>" to goal objects (object id, position, view points = agent poses that see the object).
The only reader of these files (categories.py, episodes.py and scripts/p1_replay.py use it).
"""

import gzip
import json
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
