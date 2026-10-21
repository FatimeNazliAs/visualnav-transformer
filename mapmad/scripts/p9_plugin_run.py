"""Phase 9 plug-in check (G9 row 4, block item 7), in naz_mapmad_habitat_p9: replay Phase 8 dev drives (test split,
labelled Phase 1 homes) pose by pose, let a recogniser draw the "target seen here" layer through Phase 8's fusion,
and write it in the Phase 8 format.

    python mapmad/scripts/p9_plugin_run.py --recogniser truth --gpu 1
    python mapmad/scripts/p9_plugin_run.py --recogniser owlv2 --port 5560 --gpu 1      # threshold / p99 from choice.json

Per drive -> <outputs>/p9_recognisers/plugin/<recogniser>/<drive>/:
- mapmad_target.npz (map_v2.TARGET_FILE layout): `noisy_*` = the recogniser's fused layer (the one the network reads),
  `clean_*` = simulator truth (Phase 8 rule), det_* = raw detections per frame (instance -1 for models; mapped score
  in det_confidence, the model's own score in det_raw_score, det_point_x/y for point models), false_* empty;
- steps.json: latency per step (recogniser round trip, ms), detections per step; video.mp4: picture with the kept
  detections (red) and the recogniser layer around the drive (white = 1.0), truth layer outlined in yellow.
Truth: parity.json = the plug-in truth layer equals the drive's saved Phase 8 clean layer exactly (cell, frame and
value arrays) -> G9 row 4. Drives: 2 seeded search drives of the Phase 8 test split in the 10 dev homes
(--drives to override), the same 2 for every recogniser.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mapmad" / "src"))
sys.path.insert(0, str(REPO / "train"))  # numpy-only NoMaD-side modules (map_v2)
from mapmad_bridge import protocol  # noqa: E402
from mapmad_sim import config, objectnav  # noqa: E402
from mapmad_sim.datagen_v2 import robot_spec  # noqa: E402
from mapmad_sim.depth_map import FirstSeenMap, MapSpec  # noqa: E402
from mapmad_sim.photo_frames import write_json_atomic  # noqa: E402
from mapmad_sim.recogniser import TruthRecogniser, connect, detection_cells, frame_scores  # noqa: E402
from mapmad_sim.robot import LimoSim  # noqa: E402
from mapmad_sim.run_layout import read_config  # noqa: E402
from mapmad_sim.target_layer import NoiseSpec, detect, frame_cells, max_into  # noqa: E402
from vint_train.mapmad.map_v2 import TARGET_FILE, TargetFusion  # noqa: E402

N_DRIVES = 2


def dev_drives(cfg: Dict[str, Any], seed: int) -> List[str]:
    """2 seeded search drives of the Phase 8 test split in the 10 Phase 1 homes."""
    paths = config.paths()
    split = json.loads((config.CONFIG_DIR / cfg["splits_file"]).read_text())["rule"]["val_drive"]["phase1_homes"]
    names = (paths["mapmad_data"] / cfg["drives"]["split_file"]).read_text().split()
    cands = sorted(n for n in names if n.split("-")[1] in split and "-search-" in n)
    return [cands[k] for k in sorted(np.random.default_rng(seed).permutation(len(cands))[:N_DRIVES])]


def window(layer: np.ndarray, xy: np.ndarray, origin: np.ndarray, res: float, pad: int = 10) -> slice:
    ij = np.floor((xy - origin) / res).astype(int)
    lo, hi = np.maximum(ij.min(0) - pad, 0), np.minimum(ij.max(0) + pad + 1, layer.shape)
    return (slice(lo[0], hi[0]), slice(lo[1], hi[1]))


def frame_image(rgb: np.ndarray, dets, rec_layer: np.ndarray, truth_layer: np.ndarray, win, ij, text: str) -> np.ndarray:
    """Video frame: picture with kept detections | the recogniser layer around the drive (+ truth outline, robot)."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for d in dets:
        if d.mask is not None:
            cnts, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
        elif d.point is not None:
            cv2.circle(img, (int(d.point[0]), int(d.point[1])), 5, (0, 0, 255), -1)
    lay = (np.clip(rec_layer[win], 0, 1) * 255).astype(np.uint8)
    view = cv2.cvtColor(lay, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours((truth_layer[win] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(view, cnts, -1, (0, 255, 255), 1)
    r, c = ij[0] - win[0].start, ij[1] - win[1].start
    if 0 <= r < view.shape[0] and 0 <= c < view.shape[1]:
        view[r, c] = (255, 0, 0)
    view = cv2.resize(view, (240 * view.shape[1] // max(view.shape[0], 1), 240), interpolation=cv2.INTER_NEAREST)
    out = np.hstack([img, view])
    cv2.rectangle(out, (0, 0), (out.shape[1], 14), (0, 0, 0), -1)
    cv2.putText(out, text, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out


def replay(drive: str, cfg: Dict[str, Any], gen: Dict[str, Any], rec, gpu: int, out: Path) -> Dict[str, Any]:
    paths = config.paths()
    folder = paths["mapmad_data"] / cfg["drives"]["dataset"] / drive
    meta = json.loads((folder / "mapmad_meta.json").read_text())
    frames = np.load(folder / "mapmad_frames.npz")
    with open(folder / "traj_data.pkl", "rb") as f:
        traj = pickle.load(f)
    p8 = np.load(folder / TARGET_FILE)
    home, cat = meta["home"], meta["category"]
    ids = sorted({g.object_id for g in objectnav.home_goals(meta["split"], home) if g.category == cat})
    spec, nspec = MapSpec.from_config(gen["map"]), NoiseSpec.from_config(gen["target_layer"])
    robot = LimoSim(meta["split"], home, robot_spec(gen), semantic=True, depth=True, gpu=gpu, seed=0)
    out.mkdir(parents=True, exist_ok=True)
    try:
        low, high = robot.pathfinder.get_bounds()
        fmap = FirstSeenMap.around(spec, low, high)
        truth_f, rec_f = TargetFusion(fmap.shape), TargetFusion(fmap.shape)
        det_rows, steps, video = [], [], None
        xy = np.asarray(traj["position"], np.float64)
        win = window(np.zeros(fmap.shape), xy, np.asarray(fmap.origin), spec.resolution)
        for t in range(len(frames["position_habitat"])):
            robot.place(frames["position_habitat"][t], float(traj["yaw"][t]))
            obs = robot.observe()
            depth = obs["depth"]
            cells = frame_cells(depth, robot.spec.hfov_deg, robot.camera_transform(), robot.floor_y, fmap, spec,
                                nspec.max_depth_m)
            truth_scores: Dict[int, float] = {}
            found = detect(obs["semantic"], depth, ids, nspec)
            for inst in sorted(found):  # Phase 8's clean layer, as datagen_v2 draws it
                c = cells.pixel_cell[found[inst]]
                max_into(truth_scores, np.unique(c[c >= 0]), 1.0)
            truth_f.update(t, cells.view, truth_scores)
            t0 = time.perf_counter()
            dets = rec.detect(obs["rgb"], depth, cat, semantic=obs["semantic"])
            ms = (time.perf_counter() - t0) * 1e3
            rec_f.update(t, cells.view, frame_scores(dets, cells, depth.shape))
            for d in dets:
                px = int(d.mask.sum()) if d.mask is not None else 1
                det_rows.append((t, d.instance, px, d.score, d.raw_score,
                                 d.point[0] if d.point else np.nan, d.point[1] if d.point else np.nan,
                                 int(detection_cells(d, cells, depth.shape).size)))
            steps.append({"t": t, "ms": round(ms, 3), "detections": len(dets)})
            ij = np.floor((xy[t] - np.asarray(fmap.origin)) / spec.resolution).astype(int)
            img = frame_image(obs["rgb"], dets, rec_f.grid(), truth_f.grid(), win, ij,
                              f"{rec.name} | {drive[4:]} | '{cat}' | t={t} | {ms:.0f} ms | {len(dets)} det")
            if video is None:
                video = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8, (img.shape[1], img.shape[0]))
            video.write(img)
        if video is not None:
            video.release()
    finally:
        robot.close()
    arr: Dict[str, np.ndarray] = {"shape": np.asarray(fmap.shape, np.int32), "category": np.asarray(cat)}
    for name, fus in (("clean", truth_f), ("noisy", rec_f)):
        for k, v in fus.updates().items():
            arr[f"{name}_{k}"] = v
    d = np.asarray(det_rows, np.float64).reshape(-1, 8)
    arr.update({"det_frame": d[:, 0].astype(np.int32), "det_instance": d[:, 1].astype(np.int64),
                "det_pixels": d[:, 2].astype(np.int32), "det_confidence": d[:, 3].astype(np.float32),
                "det_missed": np.zeros(len(d), bool), "det_raw_score": d[:, 4].astype(np.float32),
                "det_point_x": d[:, 5].astype(np.float32), "det_point_y": d[:, 6].astype(np.float32),
                "det_cells": d[:, 7].astype(np.int32), "false_frame": np.zeros(0, np.int32),
                "false_cell": np.zeros(0, np.int64), "false_value": np.zeros(0, np.float32)})
    np.savez_compressed(out / TARGET_FILE, **arr)
    ms = np.array([s["ms"] for s in steps])
    summary = {"drive": drive, "home": home, "category": cat, "frames": len(steps), "recogniser": rec.name,
               "latency_ms_median": float(np.median(ms)), "latency_ms_p95": float(np.percentile(ms, 95)),
               "detections": int(len(d)), "steps_with_detections": int(sum(s["detections"] > 0 for s in steps))}
    eq = {k: bool(np.array_equal(arr[f"clean_{k}"], p8[f"clean_{k}"])) for k in ("cell", "frame", "value")}
    summary["truth_layer_equals_phase8_clean"] = eq
    if rec.name == "truth":
        summary["plugin_truth_equals_phase8_clean"] = {k: bool(np.array_equal(arr[f"noisy_{k}"], p8[f"clean_{k}"]))
                                                       for k in ("cell", "frame", "value")}
    write_json_atomic(out / "steps.json", {**summary, "steps": steps})
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recogniser", required=True, help="truth or a p9_models.yaml key")
    p.add_argument("--host", default="naz_mapmad_recog")
    p.add_argument("--port", type=int)
    p.add_argument("--gpu", type=int, required=True, help="GPU that renders")
    p.add_argument("--drives", nargs="*")
    p.add_argument("--config", type=Path, default=config.CONFIG_DIR / "p9_frames.yaml")
    a = p.parse_args()
    cfg = read_config(a.config)
    gen = read_config(config.CONFIG_DIR / cfg["datagen"])
    root = config.paths()["outputs"] / cfg["out_dir"]
    drives = a.drives or dev_drives(cfg, cfg["seed"])
    if a.recogniser == "truth":
        rec = None
    else:
        ch = json.loads((root / "photo_test" / "choice.json").read_text())["rows"][a.recogniser]
        rec = connect(a.recogniser, a.host, a.port, ch["threshold"], ch["p99"], ch["kind"] == "vlm", protocol.auth_key())
    results = []
    for drive in drives:
        r = rec
        if a.recogniser == "truth":
            meta = json.loads((config.paths()["mapmad_data"] / cfg["drives"]["dataset"] / drive / "mapmad_meta.json").read_text())
            goals = objectnav.home_goals(meta["split"], meta["home"])
            r = TruthRecogniser({c: [g.object_id for g in goals if g.category == c] for c in {g.category for g in goals}},
                                NoiseSpec.from_config(gen["target_layer"]))
        results.append(replay(drive, cfg, gen, r, a.gpu, root / "plugin" / a.recogniser / drive))
        print(json.dumps(results[-1]), flush=True)
    if rec is not None:
        rec.close()
    write_json_atomic(root / "plugin" / a.recogniser / "summary.json", {"drives": drives, "results": results})


if __name__ == "__main__":
    main()
