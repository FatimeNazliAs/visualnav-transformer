"""HM3D-OVON val episodes (Phase 8.0; exam in Phase 12): read, check and convert to the ObjectNav v2 layout.

HM3D-OVON (Yokoyama et al., huggingface.co/datasets/nyokoyama/hm3d_ovon, MIT) has three val splits:
val_seen (3,000 episodes, 79 categories), val_seen_synonyms (3,000, 50), val_unseen (3,000, 49). One file per home,
<ovon>/hm3d/<split>/content/<scene>.json.gz, shaped like ObjectNav v2 (habitat-lab 0.2.3) with two differences:
- a goal's `object_id` is a string ("bench_1"), not an HM3D semantic id, so objectnav.read_goals (int ids) fails;
- an episode may list `children_object_categories` (any of them also counts as the target).
`convert_content` writes the ObjectNav v2 layout that objectnav.py reads unchanged: integer object ids = the goal's
index in the file's sorted goal list (NOT an HM3D semantic id; never use it to pick semantic pixels), the OVON id kept
as `ovon_object_id`, and categories unchanged (they may contain spaces, e.g. "kitchen cabinet").
Only reads episode files: nothing here loads a scene (the 36 val homes are exam homes, never rendered in Phase 8).
"""

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

SPLITS = ("val_seen", "val_seen_synonyms", "val_unseen")


def content_dir(root: Path, split: str) -> Path:
    """<root>/hm3d/<split>/content (the extracted archive) or <root>/<split>/content (a converted copy)."""
    for d in (root / "hm3d" / split / "content", root / split / "content"):
        if d.is_dir():
            return d
    raise FileNotFoundError(f"no OVON content for {split} under {root}")


def read_content(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def iter_content(root: Path, split: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """(scene id, file contents) of every home of a split, sorted by scene id."""
    for path in sorted(content_dir(root, split).glob("*.json.gz")):
        yield path.name[: -len(".json.gz")], read_content(path)


def home_of_scene_id(scene_id: str) -> str:
    """'hm3d/val//00877-4ok3usBNeis/4ok3usBNeis.basis.glb' -> '00877-4ok3usBNeis' (as objectnav.home_episodes)."""
    return scene_id.split("/")[-2]


def convert_content(data: Dict[str, Any]) -> Dict[str, Any]:
    """One OVON content file -> the ObjectNav v2 layout objectnav.read_goals / home_episodes read (module docstring)."""
    goals = [(key, g) for key, items in sorted(data["goals_by_category"].items()) for g in items]
    goals.sort(key=lambda kg: (kg[0], str(kg[1]["object_id"])))
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for index, (key, g) in enumerate(goals):
        by_cat.setdefault(key, []).append({**g, "object_id": index, "ovon_object_id": str(g["object_id"])})
    return {"goals_by_category": by_cat, "episodes": data["episodes"],
            "category_to_task_category_id": data.get("category_to_task_category_id", {}),
            "category_to_scene_annotation_category_id": data.get("category_to_scene_annotation_category_id", {})}


def write_converted(root: Path, out: Path, split: str) -> List[Path]:
    """Convert every home of a split to <out>/<split>/content/<scene>.json.gz; returns the files written."""
    written = []
    for scene, data in iter_content(root, split):
        path = out / split / "content" / f"{scene}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            json.dump(convert_content(data), f)
        written.append(path)
    return written


def episode_uid(split: str, home: str, episode_id: str) -> str:
    """Unique id of one OVON episode across all splits and homes (plan §0.4): ovon-<split>-<home prefix>-<id>."""
    return f"ovon-{split}-{home[:5]}-{episode_id}"


def split_summary(root: Path, split: str) -> Dict[str, Any]:
    """Homes, episode count, categories and the unique-id check of one split."""
    homes, uids, categories, children = set(), [], {}, 0
    for _, data in iter_content(root, split):
        for e in data["episodes"]:
            home = home_of_scene_id(e["scene_id"])
            homes.add(home)
            uids.append(episode_uid(split, home, str(e["episode_id"])))
            categories[e["object_category"]] = categories.get(e["object_category"], 0) + 1
            children += bool(e.get("children_object_categories"))
    return {"split": split, "homes": sorted(homes), "episodes": len(uids), "unique_ids": len(set(uids)) == len(uids),
            "categories": dict(sorted(categories.items())), "episodes_with_children": children}
