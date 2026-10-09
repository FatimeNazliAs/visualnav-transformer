"""HM3D free-text object names -> the 6 ObjectNav v2 target categories (chair, bed, plant, toilet, tv_monitor, sofa).

HM3D Semantics names objects in free text ("armchair", "flowerpot", "tv"). ObjectNav HM3D v2 lists, per home,
which object ids are goals of which category (`goals_by_category`). Reading both for the 145 train homes
counts how often each name is a goal of each category; a name maps to a category when it is that category's
goal often enough and (almost) never another's. Built once by scripts/build_category_map.py into
configs/objectnav_categories.yaml; homes without ObjectNav episodes (e.g. minival 00803, 00808) use it.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml

from mapmad_sim import config, objectnav

CATEGORIES = ("chair", "bed", "plant", "toilet", "tv_monitor", "sofa")
MAP_FILE = config.CONFIG_DIR / "objectnav_categories.yaml"


def read_semantic_txt(path: Path) -> Dict[int, str]:
    """Object id -> free-text name from an HM3D <id>.semantic.txt (lines: id,colour,"name",region)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return {int(r[0]): r[2].strip().lower() for r in rows[1:] if len(r) >= 3}


def home_dirs(scenes: Path, split: str) -> Dict[str, Path]:
    """Scene id (e.g. 'TEEsavR23oF') -> its home folder (e.g. /hm3d/minival/00800-TEEsavR23oF)."""
    return {p.name.split("-", 1)[1]: p for p in sorted((scenes / split).glob("*-*")) if p.is_dir()}


def count_goal_names(objectnav_split: Path, scenes: Path, hm3d_split: str) -> Dict[str, Counter]:
    """Name -> Counter(category -> number of goal objects with that name), over every home of a split."""
    dirs = home_dirs(scenes, hm3d_split)
    counts: Dict[str, Counter] = defaultdict(Counter)
    for content in sorted((objectnav_split / "content").glob("*.json.gz")):
        scene = content.name.split(".")[0]
        names = read_semantic_txt(dirs[scene] / f"{scene}.semantic.txt")
        for category, obj_id in sorted({(g.category, g.object_id) for g in objectnav.read_goals(content)}):
            counts[names[obj_id]][category] += 1
    return dict(counts)


def name_map(counts: Dict[str, Counter], min_count: int, min_purity: float) -> Dict[str, str]:
    """Names that are goals of one category at least min_count times and make up >= min_purity of their goal uses."""
    out = {}
    for name, by_cat in sorted(counts.items()):
        category, n = by_cat.most_common(1)[0]
        if n >= min_count and n / sum(by_cat.values()) >= min_purity:
            out[name] = category
    return out


def load_map(path: Optional[Path] = None) -> Dict[str, str]:
    """The name -> category map written by scripts/build_category_map.py."""
    with open(path or MAP_FILE) as f:
        return yaml.safe_load(f)["names"]


def instances_by_name(names: Dict[int, str], mapping: Dict[str, str]) -> Iterable[Tuple[int, str]]:
    """(object id, category) for every object of a home whose name maps to a target category."""
    for obj_id, name in sorted(names.items()):
        if name in mapping:
            yield obj_id, mapping[name]
