"""Phase 8.0: check the HM3D-OVON val episodes and convert them for our loader (reads episode files only).

    python mapmad/scripts/p8_ovon_check.py      # inside naz_mapmad_habitat (or on the host with MAPMAD_* paths)

1. per split: homes, episodes, categories, unique episode ids (ovon.episode_uid);
2. every OVON home must be one of the 36 ObjectNav v2 val homes (exam homes) and exist in HM3D val; a home that is
   not is logged and dropped from the converted copy (block item 8, no stop);
3. convert to the ObjectNav v2 layout (<mapmad_data>/ovon/converted/<split>/content/) and read every converted
   file back with the unchanged objectnav.home_goals / home_episodes.
Writes <outputs>/p8_search_data/ovon_check/ovon_check.{json,md}. Nothing is rendered.
"""

import argparse
import json
import shutil
from pathlib import Path

from mapmad_sim import config, objectnav, ovon


def objectnav_val_homes() -> list:
    """The 36 ObjectNav v2 val homes (exam homes) as HM3D folder names."""
    eps_root = config.paths()["objectnav"] / "val" / "content"
    scenes = {p.name[: -len(".json.gz")] for p in eps_root.glob("*.json.gz")}
    val = config.paths()["hm3d_scenes"] / "val"
    return sorted(p.name for p in val.glob("*-*") if p.name.split("-", 1)[1] in scenes)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ovon", type=Path, default=None, help="default <mapmad_data>/ovon")
    p.add_argument("--out", type=Path, default=None, help="default <outputs>/p8_search_data/ovon_check")
    args = p.parse_args()
    paths = config.paths()
    root = args.ovon or paths["mapmad_data"] / "ovon"
    out = args.out or paths["outputs"] / "p8_search_data" / "ovon_check"
    out.mkdir(parents=True, exist_ok=True)
    exam_homes = objectnav_val_homes()
    converted = root / "converted"
    if converted.exists():  # our own derived copy, rebuilt from the archive every time
        shutil.rmtree(converted)

    report = {"source": "https://huggingface.co/datasets/nyokoyama/hm3d_ovon (MIT, revision f04c7581)",
              "exam_homes": len(exam_homes), "splits": {}}
    for split in ovon.SPLITS:
        s = ovon.split_summary(root, split)
        outside = [h for h in s["homes"] if h not in exam_homes]
        files = ovon.write_converted(root, converted, split)
        dropped = []
        for f in files:  # drop homes outside the 36 (logged)
            data = ovon.read_content(f)
            homes = {ovon.home_of_scene_id(e["scene_id"]) for e in data["episodes"]}
            if homes & set(outside):
                f.unlink()
                dropped.append(f.name)
        goals = episodes = 0
        for f in sorted((converted / split / "content").glob("*.json.gz")):
            scene = f.name[: -len(".json.gz")]
            home = next(h for h in exam_homes if h.endswith(scene))
            goals += len(objectnav.home_goals(split, home, converted))
            episodes += len(objectnav.home_episodes(split, home, converted))
        s.update({"homes_outside_exam_set": outside, "dropped_files": dropped, "all_in_exam_set": not outside,
                  "loader_goals": goals, "loader_episodes": episodes, "n_homes": len(s["homes"]),
                  "n_categories": len(s["categories"])})
        report["splits"][split] = s
        print(f"{split}: {s['episodes']} episodes, {s['n_homes']} homes, {s['n_categories']} categories, "
              f"unique ids {s['unique_ids']}, outside exam set {outside}, loader read {episodes} episodes / {goals} goals")

    (out / "ovon_check.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["# HM3D-OVON val check (Phase 8.0)", "", f"Source: {report['source']}", "",
             f"Exam homes (ObjectNav v2 val ∩ HM3D val): {len(exam_homes)}", "",
             "| split | episodes | homes | categories | unique ids | all homes in exam set | loader episodes | loader goals |",
             "|---|---|---|---|---|---|---|---|"]
    for split, s in report["splits"].items():
        lines.append(f"| {split} | {s['episodes']} | {s['n_homes']} | {s['n_categories']} | {s['unique_ids']} | "
                     f"{s['all_in_exam_set']} | {s['loader_episodes']} | {s['loader_goals']} |")
    (out / "ovon_check.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
