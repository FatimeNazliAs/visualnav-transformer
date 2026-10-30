"""Phase 13 step 13.0: MP3D integrity against the official ObjectNav MP3D episodes, and the frozen home lists.

Inside naz_mapmad_habitat, from /app/visualnav-transformer (CPU + one GPU for rendering):

    python mapmad/scripts/p13_mp3d_check.py integrity --gpu 0 --worker 0 --workers 3   # per worker, resumable
    python mapmad/scripts/p13_mp3d_check.py report                                   # once, after all workers

`integrity`, per MP3D home with official episodes (train + val lists of the archive), with LIMO's floor map rebuilt
(the shipped .navmesh is never used) and our mp3d.scene_dataset_config.json:
- labels: every listed goal object exists in the home's semantic labels with the same mpcat40 category and its box
  centre (objects.object_box) within 0.1 m of the listed position;
- view points: every listed view point lies on the floor of the original mesh: LimoSim's downward floor probe finds a
  surface -0.05 to 0.35 m below it (official view points sit 0.124 m above the floor, median of 2azQ1b91cZZ);
- report only: a goal of our 6 categories shows (semantic pixels) from >= 1 of its first 5 view points, facing its
  box centre, with the pin025 camera.
Drop rule (p13_rules.md §2): a home with a label mismatch, or with more than 10 % of its view points off the floor
(a systematic offset = another mesh version), is dropped from exam B / the dev set and reported. Isolated off-floor
view points (mesh holes, ramps, thick rugs: 0-8.5 % per home, 0 label mismatches in 8,825 goals) are mesh quality,
not a mismatch; they never enter a success zone (exam_set's usable rule: snapped <= 0.2 m onto LIMO's real floor).
The first version (0.5 %) would have kept 1 of 11 val homes: replaced before the lists were used (phase13_log.md).
Test homes (no official episodes): the release-date check only (checks/mp3d_copy_check.txt).

`report` -> <p13>/checks/mp3d_integrity.md and <p13>/mp3d_homes.json: the frozen lists train / val / test (the
archive's train / val lists + the standard 18 test homes), minus dropped homes and homes shared with HM3D
(checks/mp3d_hm3d_overlap.json).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

from mapmad_sim import config, objectnav  # noqa: E402
from mapmad_sim.categories import CATEGORIES  # noqa: E402
from mapmad_sim.datagen_v2 import robot_spec  # noqa: E402
from mapmad_sim.objects import object_box  # noqa: E402
from mapmad_sim.robot import LimoSim, yaw_towards  # noqa: E402
from mapmad_sim.run_layout import read_config, write_text_atomic  # noqa: E402
from mapmad_sim.scene_labels import category_name  # noqa: E402

TEST_HOMES = ("2t7WUuJeko7", "5ZKStnWn8Zo", "ARNzJeq3xxb", "RPmz2sHmrrY", "UwV83HsGsw3", "Vt2qJdWjCF2", "WYY7iVyf5p8",
              "YFuZgdQ5vWj", "YVUC4YcDtcY", "fzynW3qQPVF", "gxdoqLR6rwA", "gYvKGZ5eRqb", "jtcxE69GiFV", "pa4otMbVnkk",
              "q9vSo1VnCiC", "rqfALeAoiTq", "wc2JMjhGNzB", "yqstnuAEVhm")  # the standard 18 MP3D test homes
POSITION_MAX_M = 0.1
FLOOR_BELOW_M = (-0.05, 0.35)
OFF_FLOOR_MAX_SHARE = 0.10
P13 = Path("/outputs/mapmad/p13_claim")


def official_homes() -> Dict[str, List[str]]:
    """Archive split -> its homes (train / val content files)."""
    root = config.paths()["mp3d_objectnav"]
    return {sp: sorted(p.name[: -len(".json.gz")] for p in (root / sp / "content").glob("*.json.gz"))
            for sp in ("train", "val")}


def check_home(home: str, split: str, gpu: int, dg: Dict[str, Any]) -> Dict[str, Any]:
    """Label, view-point and visibility checks of one home (module docstring)."""
    goals = objectnav.home_goals(split, home, config.paths()["mp3d_objectnav"])
    with LimoSim(config.MP3D_SPLIT, home, robot_spec(dg), semantic=True, depth=True, gpu=gpu) as robot:
        objs = {o.semantic_id: o for o in robot.sim.semantic_scene.objects if o is not None}
        label_bad, n_vp, off_floor, vis = [], 0, [], {"checked": 0, "visible": 0, "not_visible": []}
        for g in goals:
            o = objs.get(g.object_id)
            if o is None:
                label_bad.append({"object_id": g.object_id, "category": g.category, "why": "no such object"})
                continue
            centre, _ = object_box(o)
            cat, d = category_name(o, True), float(np.linalg.norm(centre - np.asarray(g.position)))
            if cat != g.category or d > POSITION_MAX_M:
                label_bad.append({"object_id": g.object_id, "category": g.category, "label": cat, "centre_m": round(d, 4)})
            for vp in g.view_points:
                n_vp += 1
                try:
                    robot.place(np.asarray(vp, dtype=np.float64), 0.0)
                    fb = robot.floor_below_feet
                except ValueError:
                    fb = math.nan
                if not FLOOR_BELOW_M[0] <= fb <= FLOOR_BELOW_M[1]:
                    off_floor.append({"object_id": g.object_id, "vp": [round(v, 3) for v in vp],
                                      "floor_below_m": None if math.isnan(fb) else round(fb, 3)})
            if g.category in CATEGORIES:
                vis["checked"] += 1
                seen = False
                for vp in g.view_points[:5]:
                    try:
                        robot.place(np.asarray(vp, dtype=np.float64), yaw_towards(vp, centre))
                    except ValueError:
                        continue
                    if (robot.observe()["semantic"] == g.object_id).any():
                        seen = True
                        break
                vis["visible"] += int(seen)
                if not seen:
                    vis["not_visible"].append({"object_id": g.object_id, "category": g.category})
        navmesh = robot.navmesh_sha256
        ours = {c: sorted(i for i, o in objs.items() if category_name(o, True) == c) for c in CATEGORIES}
    share = len(off_floor) / max(n_vp, 1)
    drop = bool(label_bad) or share > OFF_FLOOR_MAX_SHARE
    return {"home": home, "archive_split": split, "navmesh_sha256": navmesh, "goals": len(goals),
            "goals_ours": sum(g.category in CATEGORIES for g in goals), "label_mismatches": label_bad,
            "view_points": n_vp, "off_floor": len(off_floor), "off_floor_share": round(share, 6),
            "off_floor_examples": off_floor[:10], "visible_report_only": vis,
            "label_objects_ours": {c: len(v) for c, v in ours.items()}, "drop": drop,
            "drop_why": ("label mismatch" if label_bad else f"{share:.2%} view points off the floor") if drop else None}


def cmd_integrity(args: argparse.Namespace) -> None:
    dg = read_config(config.CONFIG_DIR / "p8_datagen_v3a.yaml")
    homes = official_homes()
    todo = [(h, sp) for sp in ("val", "train") for h in homes[sp]][args.worker::args.workers]
    out = Path(args.out)
    for home, sp in todo:
        path = out / f"{home}.json"
        if path.exists():
            continue
        t0 = time.time()
        rec = check_home(home, sp, args.gpu, dg)
        rec["seconds"] = round(time.time() - t0, 1)
        write_text_atomic(path, json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: rec[k] for k in ("home", "archive_split", "goals", "off_floor", "drop", "seconds")}), flush=True)


def cmd_report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    homes = official_homes()
    recs = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json"))}
    missing = [h for sp in homes.values() for h in sp if h not in recs]
    if missing:
        raise SystemExit(f"integrity missing for {len(missing)} homes: {missing[:5]}")
    overlap = json.loads((P13 / "checks" / "mp3d_hm3d_overlap.json").read_text())
    shared = set(overlap["shared_ids"]) | {s["mp3d"] for s in overlap["shared_mesh_sha256"]}
    all_ids = sorted(p.name for p in config.paths()["mp3d_scenes"].iterdir() if p.is_dir())
    test = [h for h in TEST_HOMES if h in all_ids]
    listed = set(homes["train"]) | set(homes["val"]) | set(test)
    dropped = {h: r["drop_why"] for h, r in recs.items() if r["drop"]}
    dropped.update({h: "shared with HM3D" for h in listed if h in shared})
    lists = {"train": [h for h in homes["train"] if h not in dropped], "val": [h for h in homes["val"] if h not in dropped],
             "test": [h for h in test if h not in dropped]}
    unlisted = sorted(set(all_ids) - listed)
    doc = {"lists": lists, "dropped": dropped, "unlisted_homes": unlisted,
           "rules": {"position_max_m": POSITION_MAX_M, "floor_below_m": FLOOR_BELOW_M,
                     "off_floor_max_share": OFF_FLOOR_MAX_SHARE}, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_text_atomic(P13 / "mp3d_homes.json", json.dumps(doc, indent=1) + "\n")
    lines = ["# MP3D integrity (Phase 13 step 13.0)", "",
             f"- Homes in the copy: {len(all_ids)}; archive train {len(homes['train'])}, val {len(homes['val'])}, "
             f"test (standard list) {len(test)}; not in any list: {len(unlisted)} {unlisted}.",
             f"- Shared with HM3D (ids or mesh sha256): {sorted(shared) or 'none'}.",
             f"- Dropped: {len(dropped)} {dropped or ''}.",
             f"- Kept: train {len(lists['train'])}, val {len(lists['val'])}, test {len(lists['test'])}.", "",
             "| home | split | goals | ours | label mismatches | view points | off floor | share | visible (ours, report) | drop |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for h, r in sorted(recs.items(), key=lambda kv: (kv[1]["archive_split"], kv[0])):
        v = r["visible_report_only"]
        lines.append(f"| {h} | {r['archive_split']} | {r['goals']} | {r['goals_ours']} | {len(r['label_mismatches'])} | "
                     f"{r['view_points']} | {r['off_floor']} | {r['off_floor_share']:.4%} | {v['visible']}/{v['checked']} | "
                     f"{'DROP' if r['drop'] else ''} |")
    write_text_atomic(P13 / "checks" / "mp3d_integrity.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:7]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["integrity", "report"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=str(P13 / "checks" / "mp3d_integrity"))
    args = p.parse_args()
    {"integrity": cmd_integrity, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
