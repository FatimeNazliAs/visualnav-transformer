"""Phase 14b-2 R1: detector training frames from the labelled HM3D train homes (no contamination), Habitat side
(naz_mapmad_habitat). Rules: p14b2_rules.md §R1; config p14b2_recog.yaml (`trainset`). Three steps:

    python mapmad/scripts/p14b2_trainset.py homes                         # pool + exclusion lists, sha256 (CPU)
    python mapmad/scripts/p14b2_trainset.py render --gpu 0 --shard 0/6    # render candidate poses per home (GPU)
    python mapmad/scripts/p14b2_trainset.py select                        # ~60k frames, 40 % target-free, 90/10 (CPU)

Outputs below <outputs>/p14b2_recog/trainset/:
- homes.json: pool homes (labelled HM3D train homes minus the photo-test homes) and the exclusion lists (photo-test
  homes, HM3D val homes, MP3D scenes) with sha256 of each list; the pool ∩ exclusions check (must be empty);
- cand/<home>/<frame>.jpg (RGB), <frame>_inst.png (uint16: semantic id of each labelled instance, 0 elsewhere),
  <frame>_cls.png (uint8: 0 background, 1..6 class, 255 ignore), cand/<home>.json (per frame: pose, instances, flags);
- selection.json (+ .sha256): the chosen frames with their train / val split.
Labels (one picture serves all 6 classes): an instance counted for a category (Phase 8 / 9 mapping) with >= 50 px is
that class; a counted instance below 50 px and a near-synonym (Phase 9 ignore list) are ignore, except near-synonyms
with a look-alike word (stool, bench, ...), which stay background; every other pixel is background.
Resumable: a finished home is not redone.
"""

import argparse
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from mapmad_sim import config, home_splits, photo_frames as pf, photo_labels as pl  # noqa: E402
from mapmad_sim.categories import load_map  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import read_config  # noqa: E402

CONFIG = config.CONFIG_DIR / "p14b2_recog.yaml"
CLASS_ID = {c: k + 1 for k, c in enumerate(pl.CATEGORIES)}  # 0 = background, 255 = ignore
IGNORE = 255


# --- labels (pure; tested) ------------------------------------------------------------------------------------------

def instance_role(name: str, pixels: int, tables: Dict[str, pl.LabelTable], min_pixels: int,
                  lookalike: Sequence[str]) -> Tuple[str, Optional[str]]:
    """('label', category) | ('ignore', None) | ('background', None) for one instance (module docstring)."""
    counted = [c for c, t in tables.items() if t.role(name) == "counted"]
    if counted:
        return ("label", counted[0]) if pixels >= min_pixels else ("ignore", None)
    if any(t.role(name) == "ignore" for t in tables.values()):
        if set(pl.words(pl.normalise(name))) & set(lookalike):
            return "background", None
        return "ignore", None
    return "background", None


def label_maps(semantic: np.ndarray, roles: Dict[int, Tuple[str, Optional[str]]]) -> Tuple[np.ndarray, np.ndarray]:
    """(instance map uint16, class map uint8) from the semantic picture and each instance's role."""
    sem = semantic.astype(np.int64)
    inst = np.zeros(sem.shape, np.uint16)
    cls = np.zeros(sem.shape, np.uint8)
    for i, (role, cat) in roles.items():
        m = sem == i
        if role == "label":
            inst[m] = i
            cls[m] = CLASS_ID[cat]
        elif role == "ignore":
            cls[m] = IGNORE
    return inst, cls


def has_lookalike(names: Sequence[str], lookalike: Sequence[str]) -> bool:
    return any(set(pl.words(pl.normalise(n))) & set(lookalike) for n in names)


def select_frames(frames: List[Dict[str, Any]], n_total: int, free_share: float, weight: float, val_share: float,
                  seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Positives + target-free frames (free drawn with look-alike frames `weight` x as likely), then a 90/10 split by
    home. Fewer positives than (1 - free_share) * n_total -> all positives and free scaled to keep the share."""
    rng = np.random.default_rng(seed)
    pos = [f for f in frames if f["n_labelled"] > 0]
    free = [f for f in frames if f["free"]]
    n_pos = min(len(pos), int(round(n_total * (1 - free_share))))
    n_free = min(len(free), int(round(n_pos * free_share / (1 - free_share))))
    pick_pos = [pos[i] for i in sorted(rng.choice(len(pos), n_pos, replace=False))] if n_pos < len(pos) else list(pos)
    w = np.array([weight if f["lookalike"] else 1.0 for f in free])
    pick_free = [free[i] for i in sorted(rng.choice(len(free), n_free, replace=False, p=w / w.sum()))] if n_free else []
    homes = sorted({f["home"] for f in pick_pos + pick_free})
    n_val = max(1, int(round(len(homes) * val_share)))
    val = set(homes[i] for i in rng.choice(len(homes), n_val, replace=False))
    out = sorted(({**f, "split": "val" if f["home"] in val else "train"} for f in pick_pos + pick_free),
                 key=lambda f: f["frame_id"])
    stats = {"candidates": len(frames), "positive_candidates": len(pos), "free_candidates": len(free),
             "chosen": len(out), "positive": len(pick_pos), "free": len(pick_free),
             "free_lookalike": sum(f["lookalike"] for f in pick_free), "homes": len(homes), "val_homes": sorted(val),
             "train_frames": sum(f["split"] == "train" for f in out), "val_frames": sum(f["split"] == "val" for f in out)}
    return out, stats


# --- homes ----------------------------------------------------------------------------------------------------------

def setup(cfg_path: Path) -> Dict[str, Any]:
    cfg = read_config(cfg_path)
    ts = cfg["trainset"]
    gen = read_config(config.CONFIG_DIR / ts["datagen"])
    paths = config.paths()
    out = paths["outputs"] / cfg["out_dir"] / "trainset"
    return {"cfg": cfg, "ts": ts, "gen": gen, "paths": paths, "out": out}


def scene_of(home: str) -> str:
    """'00006-HkseAnWCgqk' -> 'HkseAnWCgqk'; an MP3D scene id is its own scene."""
    return home.split("-", 1)[1] if "-" in home else home


def exclusion_lists(s: Dict[str, Any]) -> Dict[str, List[str]]:
    """Photo-test homes (Phase 1 / Phase 9), every HM3D val home (exam), every MP3D scene (practice set + exam B)."""
    p = s["paths"]
    splits = json.loads((config.CONFIG_DIR / s["ts"]["splits_file"]).read_text())
    train = home_splits.train_homes(s["ts"]["split"])
    photo = sorted(set(home_splits.by_prefix(train, splits["rule"]["val_drive"]["phase1_homes"])))
    p9_frames = p["outputs"] / "p9_recognisers" / "frames" / "frames.json"
    if p9_frames.exists():  # the photo-test frame set itself (must be the same 10 homes)
        photo = sorted(set(photo) | {f["home"] for f in json.loads(p9_frames.read_text())})
    val = sorted(d.name for d in (p["hm3d_scenes"] / "val").glob("*-*") if d.is_dir())
    val += sorted(d.name for d in (p["hm3d_semantics"] / "val").glob("*-*") if d.is_dir())
    mp3d = sorted(d.name for d in Path(p["mp3d_scenes"]).iterdir() if d.is_dir()) if Path(p["mp3d_scenes"]).exists() else []
    return {"photo_test": photo, "hm3d_val": sorted(set(val)), "mp3d": mp3d}


def pool_homes(s: Dict[str, Any], excl: Dict[str, List[str]]) -> List[str]:
    """Labelled HM3D train homes (HM3D Semantics present) minus the photo-test homes."""
    lab = sorted(d.name for d in (s["paths"]["hm3d_semantics"] / s["ts"]["split"]).glob("*-*") if d.is_dir())
    return [h for h in lab if h not in set(excl["photo_test"])]


def overlap(pool: Sequence[str], excl: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Pool ∩ each exclusion list, by full id and by scene id (gate row 6: all must be empty)."""
    ps = {scene_of(h) for h in pool}
    return {k: sorted(set(pool) & set(v) | {x for x in v if scene_of(x) in ps}) for k, v in excl.items()}


def cmd_homes(s: Dict[str, Any]) -> None:
    excl = exclusion_lists(s)
    if not excl["mp3d"] or not excl["hm3d_val"] or len(excl["photo_test"]) != 10:
        raise SystemExit(f"exclusion lists incomplete: { {k: len(v) for k, v in excl.items()} }")
    pool = pool_homes(s, excl)
    ov = overlap(pool, excl)
    doc = {"pool": pool, "pool_sha256": home_splits.list_sha256(pool),
           "exclusions": {k: {"homes": v, "sha256": home_splits.list_sha256(v)} for k, v in excl.items()},
           "overlap": ov, "contamination_free": not any(ov.values())}
    pf.write_json_atomic(s["out"] / "homes.json", doc)
    print(json.dumps({"pool": len(pool), **{k: len(v) for k, v in excl.items()}, "overlap": ov}))
    if not doc["contamination_free"]:
        raise SystemExit("CONTAMINATION: pool overlaps an exclusion list")


def load_pool(s: Dict[str, Any]) -> List[str]:
    doc = json.loads((s["out"] / "homes.json").read_text())
    if not doc["contamination_free"] or home_splits.list_sha256(doc["pool"]) != doc["pool_sha256"]:
        raise SystemExit("homes.json: contamination check failed or pool list changed")
    return doc["pool"]


# --- render ---------------------------------------------------------------------------------------------------------

def drive_index(s: Dict[str, Any]) -> Dict[str, List[str]]:
    names: List[str] = []
    for f in s["ts"]["drives"]["split_files"]:
        names += [n.strip() for n in (s["paths"]["mapmad_data"] / f).read_text().split() if n.strip()]
    out: Dict[str, List[str]] = defaultdict(list)
    for n in sorted(set(names)):
        out[n.split("-")[1]].append(n)
    return out


def home_poses(s: Dict[str, Any], home: str, drives: Sequence[str], k: int) -> List[Dict[str, Any]]:
    """Thinned poses of the home's drives, then a seeded subset of at most max_candidates_per_home."""
    d = s["ts"]["drives"]
    root = s["paths"]["mapmad_data"] / d["dataset"]
    out = []
    for drive in drives:
        frames = np.load(root / drive / "mapmad_frames.npz")
        with open(root / drive / "traj_data.pkl", "rb") as f:
            traj = pickle.load(f)
        pos3, yaw = frames["position_habitat"], np.asarray(traj["yaw"], np.float64)
        for t in pf.thin_poses(np.asarray(traj["position"], np.float64), yaw, d["min_move_m"], d["min_turn_deg"]):
            out.append({"drive": drive, "t": int(t), "position": [float(v) for v in pos3[t]], "yaw": float(yaw[t])})
    cap = int(s["ts"]["max_candidates_per_home"])
    if len(out) > cap:
        rng = np.random.default_rng([s["cfg"]["seed"], k])
        out = [out[i] for i in sorted(rng.choice(len(out), cap, replace=False))]
    return out


def render_home(s: Dict[str, Any], home: str, k: int, drives: Sequence[str], gpu: int, mapping: Dict[str, str]) -> Dict[str, Any]:
    import p9_build_frames as p9

    ts = s["ts"]
    hdir = s["out"] / "cand" / home
    hdir.mkdir(parents=True, exist_ok=True)
    poses = home_poses(s, home, drives, k)
    names = p9.home_names(home, ts["split"])
    tables = pl.six_tables(names.values(), mapping)
    p9s = {"gen": s["gen"], "cfg": {"split": ts["split"], "seed": s["cfg"]["seed"]}, "homes": [home]}
    robot, sampler = p9.open_home(p9s, home, gpu)
    frames, no_floor = [], 0
    try:
        for n, pose in enumerate(poses):
            fid = f"p14t-{home}-{n:04d}"
            try:
                obs, inst = p9.render_truth(robot, sampler.labels, names, pose, float(ts["max_depth_m"]))
            except ValueError:  # no floor under a drive pose (mesh hole): skipped, counted
                no_floor += 1
                continue
            roles = {i.id: instance_role(i.name, i.pixels, tables, int(ts["min_pixels"]), ts["lookalike_words"])
                     for i in inst}
            inst_map, cls_map = label_maps(obs["semantic"], roles)
            labelled = Counter(c for r, c in roles.values() if r == "label")
            n_ignore_target = sum(1 for i in inst if roles[i.id][0] == "ignore"
                                  and any(t.role(i.name) == "counted" for t in tables.values()))
            cv2.imwrite(str(hdir / f"{fid}.jpg"), cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, int(ts["jpeg_quality"])])
            cv2.imwrite(str(hdir / f"{fid}_inst.png"), inst_map)
            cv2.imwrite(str(hdir / f"{fid}_cls.png"), cls_map)
            frames.append({"frame_id": fid, "home": home, **pose, "labelled": dict(labelled),
                           "n_labelled": int(sum(labelled.values())), "small_targets": n_ignore_target,
                           "free": not labelled and not n_ignore_target,
                           "lookalike": has_lookalike([i.name for i in inst if i.pixels >= int(ts["min_pixels"])],
                                                      ts["lookalike_words"]),
                           "ignore_px": int((cls_map == IGNORE).sum())})
    finally:
        robot.close()
    doc = {"home": home, "poses": len(poses), "no_floor_skipped": no_floor, "frames": frames,
           "tables": {c: t.as_dict() for c, t in tables.items()}}
    pf.write_json_atomic(s["out"] / "cand" / f"{home}.json", doc)
    return doc


def cmd_render(s: Dict[str, Any], gpu: int, shard: str, limit: int) -> None:
    pool = load_pool(s)
    i, n = (int(x) for x in shard.split("/"))
    drives = drive_index(s)
    mapping = load_map(config.CONFIG_DIR / "objectnav_categories.yaml")
    todo = [(k, h) for k, h in enumerate(pool) if k % n == i]
    if limit:
        todo = todo[:limit]
    for k, home in todo:
        if (s["out"] / "cand" / f"{home}.json").exists():
            continue
        t0 = time.time()
        doc = render_home(s, home, k, drives.get(home[:5], []), gpu, mapping)
        fr = doc["frames"]
        print(f"{home}: {len(fr)} frames ({sum(f['n_labelled'] > 0 for f in fr)} positive, "
              f"{sum(f['free'] for f in fr)} free) in {time.time() - t0:.0f} s", flush=True)


def cmd_select(s: Dict[str, Any]) -> None:
    pool = load_pool(s)
    ts = s["ts"]
    frames = []
    missing = [h for h in pool if not (s["out"] / "cand" / f"{h}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} homes not rendered, e.g. {missing[:3]}")
    for h in pool:
        frames += json.loads((s["out"] / "cand" / f"{h}.json").read_text())["frames"]
    pf.check_unique([f["frame_id"] for f in frames], "frame")
    chosen, stats = select_frames(frames, int(ts["n_frames"]), float(ts["free_share"]), float(ts["lookalike_weight"]),
                                  float(ts["val_share"]), int(s["cfg"]["seed"]))
    per_class = Counter()
    for f in chosen:
        per_class.update(f["labelled"])
    stats["instances_per_class"] = dict(sorted(per_class.items()))
    stats["fallback_fewer_than_20000"] = stats["chosen"] < 20000
    path = s["out"] / "selection.json"
    pf.write_json_atomic(path, {"stats": stats, "frames": chosen})
    (s["out"] / "selection.json.sha256").write_text(pf.sha256_file(path) + "\n")
    save_run_info(s["out"], {"config": s["cfg"]}, int(s["cfg"]["seed"]))
    print(json.dumps(stats))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["homes", "render", "select"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--limit", type=int, default=0, help="at most this many homes of the shard (DRY)")
    a = p.parse_args()
    s = setup(a.config)
    s["out"].mkdir(parents=True, exist_ok=True)
    if a.step == "homes":
        cmd_homes(s)
    elif a.step == "render":
        cmd_render(s, a.gpu, a.shard, a.limit)
    else:
        cmd_select(s)


if __name__ == "__main__":
    main()
