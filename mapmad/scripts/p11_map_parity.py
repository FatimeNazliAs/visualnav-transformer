"""Phase 11 item 11.1 / block item 9: the closed-loop online map = the Phase 8 generator's map, on 2 search drives in
dev homes, with the drives' own actions replayed through the simulator server's code (sidecar_server.SimService, in
process).

    python mapmad/scripts/p11_map_parity.py --gpu 1      # inside naz_mapmad_habitat

1. Two search drives are generated with the Phase 8 generator (option A, p8_datagen_v3a.yaml; ids p11x-<home>-search-
   <n>, dev homes 00746 / 00440) into <mapmad_data>/p11_checks_parity/ (check data, rebuilt every time).
2. Each drive is replayed twice: the server is reset on the drive's start with search_v2 outputs and the drive's own
   target-noise stream ([drive seed, 1], as datagen_v2), then (a) "poses": put on every saved pose (Habitat position +
   yaw, float64), the server's bump rule applied to the logged move -- PASS/FAIL; (b) "actions": driven with the logged
   commands (v, w) -- report only, because mapmad_frames.npz stores them as float32 and the replayed pose drifts by
   ~1e-8 m, which can move a depth point across a cell edge. After every move, generator bumps at that frame that the
   rule did not make (the generator's floor-map safety step, which has no student counterpart) are injected and
   counted; a bump the rule makes that the generator does not have fails the check.
3. At every frame the server's map windows ("maps", [2, 4, 64, 64]) must equal, byte for byte, the windows rebuilt from
   the saved drive alone: obstacle / explored from mapmad_map.npz, frontier = map_v2.teacher_frontier_at (with the
   bump events), target = the noisy layer from mapmad_target.npz; the replayed pose must equal traj_data.pkl's.
Result: <outputs>/p11_dagger/checks/map_parity.json (pass = in the pose replay of both drives: 0 mismatched frames,
poses equal, and no bump made by the rule that the generator does not have).
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mapmad_sim import config, home_splits, objectnav
from mapmad_sim.datagen_v2 import HomeGeneratorV2
from mapmad_sim.run_layout import RunLayout, write_text_atomic
from mapmad_sim.search_wire import camera_wire, datagen_config, search_settings
from mapmad_sim.sidecar_server import SimService, limo_factory
from vint_train.mapmad.local_map import known_at
from vint_train.mapmad.map_v2 import TARGET_FILE, far_window, near_window, target_at, teacher_frontier_at

CONFIG = config.CONFIG_DIR / "p11_dagger.yaml"
DRIVES = [("00746", 0), ("00440", 0)]  # dev homes (Phase 1, val-drive split); search drive number
TRAIN_MAP = {"size": 64, "resolution": 0.1, "far_resolution": 0.4}  # train/config/mapmad_v2.yaml `map`


def saved_windows(dm: Dict[str, np.ndarray], tgt: Dict[str, np.ndarray], bumps: np.ndarray, t: int,
                  pose) -> np.ndarray:
    """[2, 4, 64, 64] at frame t from the saved drive alone (module docstring, step 3)."""
    obstacle = known_at(dm["obstacle_first_seen"], t)
    layers = {"obstacle": obstacle, "explored": known_at(dm["explored_first_seen"], t),
              "frontier": teacher_frontier_at(dm, bumps, t),
              "target": target_at(tgt["noisy_cell"], tgt["noisy_frame"], tgt["noisy_value"], t, obstacle.shape)}
    res = float(dm["resolution"])
    return np.stack([near_window(layers, dm["origin"], res, pose, 64, 0.1),
                     far_window(layers, dm["origin"], res, pose, 64, 0.4)]).astype(np.float32)


def episode_of(meta: Dict[str, Any], category_ids: List[int]) -> Dict[str, Any]:
    """The drive's search task in the closed-loop episode format (search_set module docstring)."""
    insts = [{"object_id": i["object_id"], "centre": i["centre"], "size": i["size"], "view_points": i["view_points"]}
             for i in meta["instances"]]
    vps = [vp for i in insts for vp in i["view_points"]]
    return {"episode_id": meta["drive_id"], "split": meta["split"], "home": meta["home"],
            "start_position": meta["start"]["position"], "start_yaw": meta["start"]["yaw"],
            "navmesh_sha256": meta["navmesh_sha256"],
            "target": {"category": meta["category"], "category_ids": category_ids, "instances": insts,
                       "view_points": vps, "point": vps[0]}}


def replay(folder: Path, gpu: int, settings: Dict[str, Any], camera: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Replay one saved drive through SimService (module docstring). mode "poses": the robot is put on every saved
    pose (Habitat position + yaw, float64) and the bump rule runs on the logged moves -> must match exactly; mode
    "actions": the logged commands are driven (stored as float32, so the pose drifts by ~1e-8 m; report only)."""
    from mapmad_sim.robot import Move

    meta = json.loads((folder / "mapmad_meta.json").read_text())
    with np.load(folder / "mapmad_map.npz") as f:
        dm = {k: f[k] for k in f.files}
    with np.load(folder / TARGET_FILE) as f:
        tgt = {k: f[k] for k in f.files}
    with np.load(folder / "mapmad_frames.npz") as f:
        fr = {k: f[k] for k in ("v", "w", "bump_events", "collided", "travelled_m", "position_habitat")}
    with open(folder / "traj_data.pkl", "rb") as f:
        traj = pickle.load(f)
    ids = sorted({g.object_id for g in objectnav.home_goals(meta["split"], meta["home"]) if g.category == meta["category"]})
    service = SimService(limo_factory(gpu))
    sv = dict(settings, noise_seed=[int(meta["seed"]), 1], record=False)
    reply = service.reset(episode_of(meta, ids), camera["hfov_deg"], False, 0.05, {"search_v2": sv}, camera)
    events = np.asarray(fr["bump_events"], dtype=np.float64).reshape(-1, 3)
    n = len(traj["position"])
    mismatched, max_pose, injected, extra_rule = [], 0.0, 0, 0
    for t in range(n):
        pose = (float(traj["position"][t, 0]), float(traj["position"][t, 1]), float(traj["yaw"][t]))
        max_pose = max(max_pose, float(np.abs(np.asarray(service.pose2d()) - pose).max()))
        if not np.array_equal(reply["maps"], saved_windows(dm, tgt, events, t, pose)):
            mismatched.append(t)
        if t == n - 1:
            break
        before = service.search.bumped.copy()
        if mode == "poses":
            service.robot.place(fr["position_habitat"][t + 1], float(traj["yaw"][t + 1]))
            move = Move(v=float(fr["v"][t]), w=float(fr["w"][t]), travelled_m=float(fr["travelled_m"][t]), short_m=0.0,
                        refused_substeps=0, collided=bool(fr["collided"][t]))
        else:
            move = service.robot.drive(float(fr["v"][t]), float(fr["w"][t]))
        service.search.after_move(move, service.pose2d(), service.robot.spec.dt)
        rule_new = np.flatnonzero((service.search.bumped & ~before).ravel())
        want = events[events[:, 0] == t]
        cells, inside = service.search.fmap.cells(want[:, 1:]) if len(want) else (np.zeros(0, np.int64), np.zeros(0, bool))
        want_cells = set(int(c) for c in cells[inside])
        extra_rule += sum(int(c) not in want_cells for c in rule_new)
        for c in sorted(want_cells):
            if not service.search.bumped.flat[c]:
                service.search.bumped.flat[c] = True
                injected += 1
        reply = {**service.observation(), "state": service.state()}
    service.close()
    ok = not mismatched and extra_rule == 0 and (mode != "poses" or max_pose <= 1e-9)
    return {"drive": meta["drive_id"], "mode": mode, "outcome": meta["outcome"], "frames": n,
            "mismatched_frames": len(mismatched), "first_mismatches": mismatched[:5], "max_pose_diff": max_pose,
            "generator_bump_events": int(len(events)), "injected_safety_bumps": injected,
            "rule_bumps_not_in_generator": extra_rule, "noisy_target_updates": int(len(tgt["noisy_value"])),
            "pass": ok}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--config", type=Path, default=CONFIG)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    dg = datagen_config(layout.cfg)
    settings = search_settings(dg, TRAIN_MAP)
    camera = camera_wire(layout.cfg["camera"])
    out = config.paths()["mapmad_data"] / "p11_checks_parity"
    if out.exists():  # our own check output, rebuilt every time
        shutil.rmtree(out)
    homes = home_splits.train_homes(dg["split"])
    report = []
    for prefix5, n in DRIVES:
        home = home_splits.by_prefix(homes, [prefix5])[0]
        did = f"p11x-{home}-search-{n:03d}"
        gen = HomeGeneratorV2(dg, home, labelled=True, gpu=args.gpu)
        try:
            meta = gen.drive(did, "search", out, failed_dir=out / "_failed")
        finally:
            gen.close()
        folder = out / did if meta["kept"] else out / "_failed" / did
        for mode in ("poses", "actions"):
            row = replay(folder, args.gpu, settings, camera, mode)
            report.append(row)
            print(json.dumps(row), flush=True)
    ok = all(r["pass"] for r in report if r["mode"] == "poses")
    path = layout.base / "checks" / "map_parity.json"
    write_text_atomic(path, json.dumps({"drives": report, "pass": ok,
                                        "rule": "pass/fail on the pose replay; the action replay is report only"},
                                       indent=1) + "\n")
    print(json.dumps({"pass": ok, "out": str(path)}))


if __name__ == "__main__":
    main()
