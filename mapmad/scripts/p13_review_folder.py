"""Phase 13 review folder for a reader who does not know the project: copies (never moves or deletes) the checked videos
into <p13 out>/videos_review/ under plain-word names, writes README.md and index.csv. Standard library only (host):

    python3 mapmad/scripts/p13_review_folder.py --p13 /mnt/shared_disk/nazli/nomad_outputs/mapmad/p13_claim

Sources: videos/ (13.1: M21, M21h, N, F on HM3D) and videos_rule/{hm3d,examb}/ (M21b, M21bh, VF); one copy per
(arm, episode) even when two picks chose the same episode. Folders: 0_start_here/ (START_HERE, chosen for clarity, not
representative) and one folder per kind of robot (ARM_FOLDER). File name: <HM3D|MP3D>_<target>_<reach>_<stop>[_<model
version>].mp4 from the video's own events (manifest): reach = reaches-<target> | sees-<target>-never-reaches |
never-sees-<target>; stop = what the stop rule did (STOP_WORDS). An existing file with another size is an error.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

ARM_FOLDER = {"M21": "1_MapMaD_with_map", "M21b": "1_MapMaD_with_map", "M21h": "2_MapMaD_map_hidden",
              "M21bh": "2_MapMaD_map_hidden", "N": "3_NoMaD_original", "F": "4_Frontier_planner_F",
              "VF": "5_VLFM_style_VF"}
ARM_WORDS = {"M21": "MapMaD-with-map", "M21b": "MapMaD-with-map", "M21h": "MapMaD-map-hidden",
             "M21bh": "MapMaD-map-hidden", "N": "NoMaD", "F": "Frontier-planner", "VF": "VLFM-style"}
VERSION = {"M21": "v2.1", "M21h": "v2.1", "M21b": "v2.1b", "M21bh": "v2.1b"}
STOP_WORDS = {"right object": "stops-at-it", "wrong object": "stops-at-a-wrong-object",
              "near miss": "stops-near-it-not-close-enough", "outside": "stops-in-a-wrong-place", None: "never-stops"}
SOURCES = ("videos", "videos_rule/hm3d", "videos_rule/examb")
# (arm, episode): chosen for clarity, NOT representative
START_HERE = [("M21b", "p12e-00820-toilet-2"), ("M21bh", "p12e-00820-toilet-2"), ("M21", "p12e-00871-bed-2"),
              ("F", "p12e-00820-toilet-2"), ("M21b", "p13b-QUCTc6BB5sX-chair-7"), ("M21b", "p13b-oLBMNvg9in8-bed-2")]
START_WHY = ["the same drive with the map ...", "... and with the map hidden (v2.1b): the map makes the difference",
             "a stop at a wrong object (a white sofa the detector boxed as 'bed')", "the frontier planner F succeeds",
             "a success in an MP3D home (never seen in training)", "a failure in an MP3D home"]


def dataset(episode_id: str) -> str:
    return "MP3D" if episode_id.startswith("p13b-") else "HM3D"


def what_happens(man: Dict[str, Any]) -> Tuple[str, str]:
    """(reach part, stop part) of the file name, from the video's events."""
    target = man["category"].replace("_", "-")
    kinds = {e["kind"]: e for e in man["events"]}
    if "oracle" in kinds:
        reach = f"reaches-{target}"
    elif "sight" in kinds:
        reach = f"sees-{target}-never-reaches"
    else:
        reach = f"never-sees-{target}"
    return reach, STOP_WORDS[kinds["rule"]["how"] if "rule" in kinds else None]


def manifests(p13: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """Every checked video once per (arm, episode), in source order."""
    seen, out = set(), []
    for src in SOURCES:
        for mf in sorted((p13 / src).glob("*.manifest.json")):
            man = json.loads(mf.read_text())
            key = (man["arm"], man["episode_id"])
            if key not in seen:
                seen.add(key)
                out.append((p13 / src / man["video"], man))
    return out


def copy(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.stat().st_size != src.stat().st_size:
            raise SystemExit(f"{dst} exists with another size: not overwritten")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def unique(name: str, used: set) -> str:
    """name, or name_2, name_3, ... when taken in the same folder."""
    k, out = 1, name
    while out in used:
        k += 1
        out = f"{name}_{k}"
    used.add(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p13", type=Path, required=True, help="the Phase 13 output folder (p13_claim)")
    args = p.parse_args()
    out = args.p13 / "videos_review"
    vids = manifests(args.p13)
    by_key = {(m["arm"], m["episode_id"]): (src, m) for src, m in vids}
    rows, used = [], {}
    for k, key in enumerate(START_HERE, 1):
        src, m = by_key[key]
        reach, stop = what_happens(m)
        name = f"{k}_{dataset(m['episode_id'])}_{m['category'].replace('_', '-')}_{ARM_WORDS[m['arm']]}_{reach}_{stop}"
        name = unique(name, used.setdefault("0_start_here", set()))
        copy(src, out / "0_start_here" / f"{name}.mp4")
        rows.append(("0_start_here", f"{name}.mp4", m, src))
    for src, m in vids:
        folder = ARM_FOLDER[m["arm"]]
        reach, stop = what_happens(m)
        name = f"{dataset(m['episode_id'])}_{m['category'].replace('_', '-')}_{reach}_{stop}"
        if m["arm"] in VERSION:
            name += f"_{VERSION[m['arm']]}"
        name = unique(name, used.setdefault(folder, set()))
        copy(src, out / folder / f"{name}.mp4")
        rows.append((folder, f"{name}.mp4", m, src))
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "file", "arm", "arm_name", "dataset", "episode", "target", "events", "source"])
        for folder, name, m, src in rows:
            w.writerow([folder, name, m["arm"], m["arm_name"], dataset(m["episode_id"]), m["episode_id"], m["category"],
                        "; ".join(e["text"] for e in m["events"]), str(src.relative_to(args.p13))])
    lines = ["# Start here: chosen for clarity, NOT representative", "",
             "These 6 videos were picked because they are easy to follow. They do not show how often each robot "
             "succeeds; see ../README.md for how to read the screen.", ""]
    lines += [f"{k}. `{name}`: {why}" for k, ((_, name, _, _), why) in enumerate(zip(rows[:len(START_HERE)], START_WHY), 1)]
    (out / "0_start_here" / "README.md").write_text("\n".join(lines) + "\n")
    readme = Path(__file__).with_name("p13_review_README.md")
    shutil.copy2(readme, out / "README.md")
    print(json.dumps({"videos": len(rows), "folders": sorted({r[0] for r in rows}), "out": str(out)}))


if __name__ == "__main__":
    main()
