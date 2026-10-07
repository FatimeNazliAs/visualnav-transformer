"""Check the ObjectNav HM3D v2 episodes against our HM3D homes (gate G0 row 3).

    python mapmad/scripts/objectnav_stats.py            # writes <outputs>/p0_setup/objectnav/

For train, val and val_mini: episode counts, target categories, homes. Then:
  - every episode's scene path resolves under the HM3D mount (/hm3d);
  - every val home is one of the 36 labelled val homes (and train: the 145 labelled train homes);
  - on --check-episodes random val episodes: each goal object id exists in that home's labels, its
    free-text label name is listed per target category, and the goal position lies at that object.
Writes objectnav_stats.md (readable), objectnav_stats.json (all numbers), example_episode.json, run_info.json.
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from habitat_starter import make_sim
from habitat_starter.semantics import box_center_size

from mapmad_sim import config
from mapmad_sim.run_info import save_run_info

SPLITS = ("train", "val", "val_mini")
SCENE_PREFIX = "hm3d_v0.2/"  # episode scene_id = hm3d_v0.2/<split>/<home>/<id>.basis.glb
GOAL_TOLERANCE_M = 0.25      # goal position may sit this far outside the object's box


def read_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def scene_path(scene_id: str, scenes: Path) -> Path:
    """Episode scene_id -> file under the HM3D mount; the episodes' own prefix is dropped."""
    return scenes / (scene_id[len(SCENE_PREFIX):] if scene_id.startswith(SCENE_PREFIX) else scene_id)


def load_split(root: Path, split: str) -> Dict[str, Any]:
    """Header file + every per-home content file of one split."""
    header = read_gz(root / split / f"{split}.json.gz")
    contents = {p.name[: -len(".json.gz")]: read_gz(p) for p in sorted((root / split / "content").glob("*.json.gz"))}
    return {"header": header, "contents": contents}


def split_stats(data: Dict[str, Any], split: str, scenes: Path) -> Dict[str, Any]:
    episodes = [e for c in data["contents"].values() for e in c["episodes"]]
    per_home = Counter(Path(e["scene_id"]).parent.name for e in episodes)
    homes = sorted(per_home)
    missing = sorted({e["scene_id"] for e in episodes if not scene_path(e["scene_id"], scenes).exists()})
    hm3d_split = "val" if split == "val_mini" else split
    labelled = set(config.labelled_homes(hm3d_split, scenes))
    return {
        "episodes": len(episodes),
        "homes": len(homes),
        "content_files": len(data["contents"]),
        "episodes_per_home": {"min": min(per_home.values()), "max": max(per_home.values())},
        "categories": dict(sorted(Counter(e["object_category"] for e in episodes).items())),
        "category_to_task_category_id": data["header"]["category_to_task_category_id"],
        "category_to_scene_annotation_category_id": data["header"]["category_to_scene_annotation_category_id"],
        "scene_paths_missing": missing,
        f"labelled_{hm3d_split}_homes": len(labelled),
        "homes_not_labelled": sorted(set(homes) - labelled),
        "labelled_homes_without_episodes": sorted(labelled - set(homes)) if split != "val_mini" else None,
        "home_list": homes if split == "val_mini" else None,
    }


def check_goals(data: Dict[str, Any], scenes: Path, n: int, seed: int) -> List[Dict[str, Any]]:
    """For n random episodes: do the goal objects exist in the home's labels, and where are they?"""
    episodes = [e for c in data["contents"].values() for e in c["episodes"]]
    rng = np.random.default_rng(seed)
    picked = [episodes[i] for i in sorted(rng.choice(len(episodes), size=n, replace=False))]
    by_home = defaultdict(list)
    for e in picked:
        by_home[e["scene_id"]].append(e)
    rows = []
    for scene_id, eps in by_home.items():
        home = Path(scene_id).parent.name
        # the colour camera must stay on: without it habitat-sim 0.3.3 loads no mesh
        # textures and leaves every HM3D object box at size 0
        settings = config.hm3d_sim_settings(Path(scene_id).parent.parent.name, home, scenes,
                                            depth_sensor=False, semantic_sensor=True, width=64, height=64)
        content = data["contents"][Path(scene_id).name.split(".")[0]]
        with make_sim(settings) as sim:
            objects = {o.semantic_id: o for o in sim.semantic_scene.objects if o is not None}
            if not any(np.any(box_center_size(o.aabb)[1] > 0) for o in objects.values()):
                raise RuntimeError(f"{home}: all semantic object boxes are empty; positions can't be checked")
            for e in eps:
                goals = content["goals_by_category"][f"{Path(scene_id).name}_{e['object_category']}"]
                for g in goals:
                    obj = objects.get(g["object_id"])
                    row = {"episode": e["episode_id"], "home": home, "category": e["object_category"],
                           "object_id": g["object_id"], "found": obj is not None}
                    if obj is not None:
                        center, size = box_center_size(obj.aabb)
                        outside = np.maximum(np.abs(np.array(g["position"]) - center) - size / 2, 0)
                        row.update(label_name=obj.category.name(),
                                   goal_at_object=bool(np.linalg.norm(outside) <= GOAL_TOLERANCE_M))
                    rows.append(row)
    return rows


def write_markdown(path: Path, stats: Dict[str, Any], goals: List[Dict[str, Any]]) -> None:
    lines = ["# ObjectNav HM3D v2 — Phase 0 check", ""]
    lines += ["| split | episodes | homes | labelled homes (HM3D) | scene paths missing | homes not labelled |",
              "|---|---|---|---|---|---|"]
    for split, s in stats["splits"].items():
        labelled = next(v for k, v in s.items() if k.startswith("labelled_") and k.endswith("_homes"))
        lines.append(f"| {split} | {s['episodes']} | {s['homes']} | {labelled} | {len(s['scene_paths_missing'])} "
                     f"| {len(s['homes_not_labelled'])} |")
    lines += ["", "## Target categories (episodes per category)", ""]
    for split, s in stats["splits"].items():
        lines.append(f"- **{split}:** " + ", ".join(f"{k} {v}" for k, v in s["categories"].items()))
    lines += ["", f"- `category_to_scene_annotation_category_id`: `{stats['splits']['val']['category_to_scene_annotation_category_id']}`"]
    lines += [f"- val_mini homes: {', '.join(stats['splits']['val_mini']['home_list'])}"]
    found = sum(r["found"] for r in goals)
    at = sum(r.get("goal_at_object", False) for r in goals)
    lines += ["", f"## Goal objects of {stats['checked_episodes']} random val episodes (seed {stats['seed']})", "",
              f"- goal objects: {len(goals)}; found in the home's labels: {found}; goal position at the object "
              f"(within {GOAL_TOLERANCE_M} m of its box): {at}",
              "- label names per target category (free text in HM3D Semantics → ObjectNav category):"]
    names = defaultdict(Counter)
    for r in goals:
        names[r["category"]][r.get("label_name", "<missing>")] += 1
    for cat, c in sorted(names.items()):
        lines.append(f"  - {cat}: " + ", ".join(f"\"{k}\" ×{v}" for k, v in c.most_common()))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check-episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="default: <outputs>/p0_setup/objectnav")
    args = p.parse_args()

    paths = config.paths()
    out = args.out or paths["outputs"] / "p0_setup" / "objectnav"
    out.mkdir(parents=True, exist_ok=True)
    data = {split: load_split(paths["objectnav"], split) for split in SPLITS}
    stats = {"splits": {split: split_stats(d, split, paths["hm3d_scenes"]) for split, d in data.items()},
             "checked_episodes": args.check_episodes, "seed": args.seed}
    goals = check_goals(data["val"], paths["hm3d_scenes"], args.check_episodes, args.seed)
    stats["goal_checks"] = goals
    example = next(iter(data["val"]["contents"].values()))["episodes"][0]
    (out / "example_episode.json").write_text(json.dumps(example, indent=2) + "\n")
    (out / "objectnav_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    write_markdown(out / "objectnav_stats.md", stats, goals)
    save_run_info(out, vars(args), args.seed)
    print((out / "objectnav_stats.md").read_text())


if __name__ == "__main__":
    main()
