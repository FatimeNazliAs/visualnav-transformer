"""Phase 9 photo-test frame set (block items 2, 3, 6), in naz_mapmad_habitat_p9. Three steps:

    python mapmad/scripts/p9_build_frames.py candidates --gpu 1     # render every candidate pose, record truth (GPU)
    python mapmad/scripts/p9_build_frames.py select                 # items, halves, label tables (CPU)
    python mapmad/scripts/p9_build_frames.py render --gpu 1         # render + save the chosen frames, fingerprint (GPU)

Outputs below <outputs>/p9_recognisers/frames/ (see frames/README written by `render`):
- candidates/<home>.json: every candidate pose (frame id, source, pose, camera matrix, instances in view);
- items_six.json, items_ovon.json, halves.json, label_tables.json, ovon_mapping.json, selection_log.txt;
- img/<frame id>_{rgb,depth,sem}.png (RGB; depth uint16 mm; semantic uint16 ids), frames.json, fingerprint.json,
  sheet_truth.png (24 frames with truth masks).
`render` re-renders every chosen pose and refuses to continue if its instances differ from the candidate pass
(determinism). Steps are resumable: a finished home / file is not redone.
"""

import argparse
import csv
import gzip
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapmad_sim import config, home_splits, photo_frames as pf, photo_labels as pl  # noqa: E402
from mapmad_sim.categories import load_map  # noqa: E402
from mapmad_sim.ovon import SPLITS as OVON_SPLITS  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import read_config  # noqa: E402

CONFIG = config.CONFIG_DIR / "p9_frames.yaml"


def setup(cfg_path: Path) -> Dict[str, Any]:
    cfg = read_config(cfg_path)
    gen = read_config(config.CONFIG_DIR / cfg["datagen"])
    splits = json.loads((config.CONFIG_DIR / cfg["splits_file"]).read_text())
    prefixes = splits["rule"]["val_drive"]["phase1_homes"]
    homes = home_splits.by_prefix(home_splits.train_homes(cfg["split"]), prefixes)
    if any(h not in splits["homes"]["val_drive"] for h in homes):
        raise SystemExit("a Phase 1 home is not in the frozen val-drive split")
    paths = config.paths()
    out = paths["outputs"] / cfg["out_dir"] / "frames"
    return {"cfg": cfg, "gen": gen, "homes": homes, "paths": paths, "out": out}


def home_names(home: str, split: str) -> Dict[int, str]:
    """Semantic id -> raw label from the home's <id>.semantic.txt."""
    scene = home.split("-", 1)[1]
    path = config.paths()["hm3d_scenes"] / split / home / f"{scene}.semantic.txt"
    with open(path, newline="") as f:
        return {int(r[0]): r[2].strip().lower() for r in list(csv.reader(f))[1:] if len(r) >= 3}


# --- candidates ---------------------------------------------------------------------------------------------------

def drive_poses(s: Dict[str, Any], home: str) -> List[Dict[str, Any]]:
    """Thinned poses of the home's Phase 8 test-split drives (source b)."""
    d = s["cfg"]["drives"]
    names = [n.strip() for n in (s["paths"]["mapmad_data"] / d["split_file"]).read_text().split() if n.strip()]
    root = s["paths"]["mapmad_data"] / d["dataset"]
    out = []
    for drive in sorted(n for n in names if n.split("-")[1] == home[:5]):
        frames = np.load(root / drive / "mapmad_frames.npz")
        with open(root / drive / "traj_data.pkl", "rb") as f:
            traj = pickle.load(f)
        pos3, yaw = frames["position_habitat"], np.asarray(traj["yaw"], np.float64)
        for t in pf.thin_poses(np.asarray(traj["position"], np.float64), yaw, d["min_move_m"], d["min_turn_deg"]):
            out.append({"drive": drive, "t": int(t), "position": [float(v) for v in pos3[t]], "yaw": float(yaw[t])})
    return out


def random_poses(sampler: Any, robot: Any, n: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """n seeded random real-floor poses with a random heading and a camera in free space (source a)."""
    out = []
    while len(out) < n:
        p = sampler.random_point()
        if not sampler.on_real_floor(p):
            continue
        yaw = float(rng.uniform(-math.pi, math.pi))
        if robot.camera_clear(p, yaw):
            out.append({"position": [round(float(v), 5) for v in p], "yaw": round(yaw, 5)})
    return out


def render_truth(robot: Any, labels: Any, names: Dict[int, str], pose: Dict[str, Any], max_depth: float):
    """Place the robot, render, and list the instances in view."""
    robot.place(np.asarray(pose["position"], np.float64), pose["yaw"])
    obs = robot.observe()
    ids = np.unique(obs["semantic"])
    boxes = {int(i): labels.box(int(i)) for i in ids if i != 0 and labels.box(int(i)) is not None}
    inst = pf.frame_instances(obs["semantic"], obs["depth"], names, boxes, max_depth)
    return obs, inst


def open_home(s: Dict[str, Any], home: str, gpu: int):
    from mapmad_sim.datagen_v2 import robot_spec
    from mapmad_sim.drive_sampler import DriveSampler
    from mapmad_sim.robot import LimoSim

    gen = s["gen"]
    robot = LimoSim(s["cfg"]["split"], home, robot_spec(gen), semantic=True, depth=True, gpu=gpu, seed=0)
    k = s["homes"].index(home)
    sampler = DriveSampler(robot, gen, np.random.default_rng([s["cfg"]["seed"], k, 0]), goals=[])
    return robot, sampler


def cmd_candidates(s: Dict[str, Any], gpu: int) -> None:
    out = s["out"] / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    md = float(s["cfg"]["max_depth_m"])
    for k, home in enumerate(s["homes"]):
        path = out / f"{home}.json"
        if path.exists():
            continue
        t0 = time.time()
        robot, sampler = open_home(s, home, gpu)
        try:
            names = home_names(home, s["cfg"]["split"])
            sim_names = {i: n for i, n in sampler.labels.names.items() if i in names}
            name_mismatch = sum(pl.normalise(names[i]) != pl.normalise(n) for i, n in sim_names.items())
            rng = np.random.default_rng([s["cfg"]["seed"], k, 1])
            poses = [("rnd", p) for p in random_poses(sampler, robot, int(s["cfg"]["random_poses_per_home"]), rng)]
            poses += [("drv", p) for p in drive_poses(s, home)]
            counters = {"rnd": 0, "drv": 0}
            frames, no_floor = [], 0
            for src, pose in poses:
                fid = pf.frame_id(home, src, counters[src])
                counters[src] += 1
                try:
                    _, inst = render_truth(robot, sampler.labels, names, pose, md)
                except ValueError:  # no floor under a drive pose (mesh hole): skipped, counted
                    no_floor += 1
                    continue
                frames.append({"frame_id": fid, "home": home, "src": src, **pose,
                               "camera_tf": np.round(robot.camera_transform(), 6).tolist(),
                               "floor_y": round(robot.floor_y, 6), "instances": pf.instances_to_json(inst)})
            pf.check_unique([f["frame_id"] for f in frames], "frame")
            pf.write_json_atomic(path, {"home": home, "names": {str(i): n for i, n in sorted(names.items())},
                                        "hfov_deg": robot.spec.hfov_deg, "navmesh_sha256": robot.navmesh_sha256,
                                        "floor_checks": dict(sampler.floor_checks), "no_floor_skipped": no_floor,
                                        "sim_vs_txt_name_mismatches": name_mismatch,
                                        "sim_ids_missing_in_txt": len(set(sampler.labels.names) - set(names)),
                                        "frames": frames})
        finally:
            robot.close()
        print(f"{home}: {len(frames)} candidates ({counters}) in {time.time() - t0:.0f} s", flush=True)


# --- select -------------------------------------------------------------------------------------------------------

def ovon_names(root: Path) -> Dict[str, List[str]]:
    """OVON val category names per split (names only, read from the episode files; no scene is loaded)."""
    out = {}
    for split in OVON_SPLITS:
        names = set()
        for path in sorted((root / "hm3d" / split / "content").glob("*.json.gz")):
            with gzip.open(path, "rt") as f:
                data = json.load(f)
            names |= {key.split(".basis.glb_", 1)[1] for key in data["goals_by_category"]}
        out[split] = sorted(names)
    return out


def cmd_select(s: Dict[str, Any]) -> None:
    cfg, out = s["cfg"], s["out"]
    if (out / "items_six.json").exists():
        print("select: already done")
        return
    cands = [json.loads((out / "candidates" / f"{h}.json").read_text()) for h in s["homes"]]
    all_names = sorted({pl.normalise(n) for c in cands for n in c["names"].values() if n.strip()})
    mapping = load_map(config.CONFIG_DIR / cfg["category_map"])
    six = pl.six_tables(all_names, mapping)
    ov = ovon_names(s["paths"]["mapmad_data"] / cfg["ovon"]["root"])
    ov_all = sorted({n for v in ov.values() for n in v})
    ovt = {n: pl.ovon_table(n, all_names) for n in ov_all}
    mapped = {n: t for n, t in ovt.items() if t.counted}
    frames = []
    for c in cands:
        for f in c["frames"]:
            inst = pf.instances_from_json(f["instances"])
            st = {k: pf.item_status(inst, t) for k, t in six.items()}
            sto = {k: pf.item_status(inst, t) for k, t in mapped.items()}
            frames.append({"frame_id": f["frame_id"], "home": f["home"], "src": f["src"], "status": st, "status_ovon": sto})
    log: List[str] = []
    sel = cfg["select"]
    items_six = pf.select_items(frames, six, pf.SelectSpec(seed=cfg["seed"], **sel["six"]), log=log)
    ovon_frames = [{**f, "status": f["status_ovon"]} for f in frames]
    items_ovon = pf.select_items(ovon_frames, mapped, pf.SelectSpec(seed=cfg["seed"] + 1, **sel["ovon"]), ovon=True, log=log)
    pf.check_unique([i["item_id"] for i in items_six + items_ovon], "item")
    pos_by_home: Dict[str, Dict[str, int]] = {h: {} for h in s["homes"]}
    for i in items_six:
        if i["status"] == "positive":
            pos_by_home[i["home"]][i["key"]] = pos_by_home[i["home"]].get(i["key"], 0) + 1
    a, b, why = pf.split_halves(s["homes"], pos_by_home, cfg["seed"], cfg["halves_min_share"])
    half = {h: "A" for h in a} | {h: "B" for h in b}
    for i in items_six + items_ovon:
        i["half"] = half[i["home"]]
    split_of = {n: [sp for sp, names in ov.items() if n in names] for n in ov_all}
    pf.write_json_atomic(out / "halves.json", {"A": a, "B": b, **why, "seed": cfg["seed"]})
    pf.write_json_atomic(out / "label_tables.json", {
        "category_map": cfg["category_map"], "category_map_sha256": pl.file_sha256(config.CONFIG_DIR / cfg["category_map"]),
        "home_raw_labels": all_names, "six": {k: t.as_dict() for k, t in six.items()},
        "near_words": pl.NEAR_WORDS, "not_near": pl.NOT_NEAR})
    pf.write_json_atomic(out / "ovon_mapping.json", {
        "splits": ov, "mapped": {n: {**t.as_dict(), "splits": split_of[n]} for n, t in sorted(mapped.items())},
        "unmapped": {n: split_of[n] for n in ov_all if n not in mapped}})
    pf.write_json_atomic(out / "items_six.json", items_six)
    pf.write_json_atomic(out / "items_ovon.json", items_ovon)
    (out / "selection_log.txt").write_text("\n".join(log) + "\n")
    print(f"select: {len(items_six)} six-category items, {len(items_ovon)} OVON items; halves A={a} B={b} ({why})")


# --- render -------------------------------------------------------------------------------------------------------

def cmd_render(s: Dict[str, Any], gpu: int) -> None:
    out = s["out"]
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    items = json.loads((out / "items_six.json").read_text()) + json.loads((out / "items_ovon.json").read_text())
    wanted = {i["frame_id"] for i in items}
    half = {i["home"]: i["half"] for i in items}
    md = float(s["cfg"]["max_depth_m"])
    chosen = []
    for home in s["homes"]:
        cand = json.loads((out / "candidates" / f"{home}.json").read_text())
        todo = [f for f in cand["frames"] if f["frame_id"] in wanted]
        if not todo:
            continue
        robot, sampler = open_home(s, home, gpu)
        try:
            names = {int(k): v for k, v in cand["names"].items()}
            for f in todo:
                obs, inst = render_truth(robot, sampler.labels, names, f, md)
                if pf.instances_to_json(inst) != f["instances"]:
                    raise SystemExit(f"{f['frame_id']}: re-rendered instances differ from the candidate pass")
                if int(obs["semantic"].max()) > 65535:
                    raise SystemExit(f"{f['frame_id']}: semantic id > 65535 does not fit uint16")
                base = img / f["frame_id"]
                cv2.imwrite(f"{base}_rgb.png", cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{base}_depth.png", np.clip(np.round(obs["depth"] * 1000.0), 0, 65535).astype(np.uint16))
                cv2.imwrite(f"{base}_sem.png", obs["semantic"].astype(np.uint16))
                chosen.append({**{k: f[k] for k in ("frame_id", "home", "src", "position", "yaw", "camera_tf", "floor_y",
                                                   "instances")},
                               **({"drive": f["drive"], "t": f["t"]} if f["src"] == "drv" else {}),
                               "half": half[home], "hfov_deg": cand["hfov_deg"]})
        finally:
            robot.close()
        print(f"{home}: saved {len(todo)} frames", flush=True)
    chosen.sort(key=lambda f: f["frame_id"])
    pf.check_unique([f["frame_id"] for f in chosen], "frame")
    pf.write_json_atomic(out / "frames.json", chosen)
    write_fingerprint(out)
    truth_sheet(out, chosen, s["cfg"])
    save_run_info(out, {"config": s["cfg"], "homes": s["homes"]}, s["cfg"]["seed"])


def write_fingerprint(out: Path) -> None:
    """Separate fingerprints of the 6-category and the OVON set (items file + the frames' PNGs + frames.json)."""
    fp = {}
    for name in ("six", "ovon"):
        items = json.loads((out / f"items_{name}.json").read_text())
        fids = sorted({i["frame_id"] for i in items})
        files = [f"items_{name}.json", "frames.json"] + [f"img/{f}_{k}.png" for f in fids for k in ("rgb", "depth", "sem")]
        fp[name] = {**pf.fingerprint(out, files), "items": len(items), "frames": len(fids),
                    "items_sha256": pf.sha256_file(out / f"items_{name}.json")}
    fp["frames_json_sha256"] = pf.sha256_file(out / "frames.json")
    pf.write_json_atomic(out / "fingerprint.json", fp)
    print("fingerprint:", json.dumps(fp))


def truth_sheet(out: Path, frames: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """24 seeded frames: RGB with the counted instances of the 6 categories outlined (colour per category)."""
    items = json.loads((out / "items_six.json").read_text())
    tables = json.loads((out / "label_tables.json").read_text())["six"]
    role = {k: set(t["counted"]) for k, t in tables.items()}
    pos = sorted({i["frame_id"] for i in items if i["status"] == "positive"})
    rng = np.random.default_rng(cfg["seed"])
    pick = [pos[k] for k in sorted(rng.choice(len(pos), size=min(cfg["sheet_frames"], len(pos)), replace=False))]
    by_id = {f["frame_id"]: f for f in frames}
    colours = {"chair": (0, 0, 255), "sofa": (0, 160, 255), "bed": (255, 0, 255), "toilet": (255, 255, 0),
               "tv_monitor": (0, 255, 0), "plant": (0, 128, 0)}
    tiles = []
    for fid in pick:
        rgb = cv2.imread(str(out / "img" / f"{fid}_rgb.png"))
        sem = cv2.imread(str(out / "img" / f"{fid}_sem.png"), cv2.IMREAD_UNCHANGED)
        labels = []
        for inst in pf.instances_from_json(by_id[fid]["instances"]):
            for k, names in role.items():
                if pl.normalise(inst.name) in names and inst.counted():
                    m = (sem == inst.id).astype(np.uint8)
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(rgb, cnts, -1, colours[k], 2)
                    labels.append(k)
        cv2.putText(rgb, fid[4:], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(rgb, ",".join(sorted(set(labels))), (3, 234), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tiles.append(rgb)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]
    cv2.imwrite(str(out.parent / "sheets" / "frames_truth.png"), np.vstack(rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["candidates", "select", "render"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", type=Path, default=CONFIG)
    a = p.parse_args()
    s = setup(a.config)
    if a.step == "candidates":
        cmd_candidates(s, a.gpu)
    elif a.step == "select":
        cmd_select(s)
    else:
        cmd_render(s, a.gpu)


if __name__ == "__main__":
    main()
