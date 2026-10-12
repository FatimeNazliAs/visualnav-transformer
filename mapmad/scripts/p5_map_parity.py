"""Phase 5 map parity (block item 2): the map MapMaD saw during an exam-style run == the Phase 2 builder's map for
the same poses, read the way training reads it. Inside naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/p5_map_parity.py --config mapmad/configs/p5_dev.yaml --traces <run>/traces/C_wp2 --gpu 0

For every trace (run_exam --trace-maps: the 3 x 64 x 64 tensors the policy received + the exact poses):
1. replay the poses in a fresh LimoSim through Phase 2's own recording step (`datagen.DriveRunner.record_frame`:
   depth -> world points -> first-seen map, frame index = step);
2. write the result as a Phase 2 drive would (`mapmad_map.npz` from `FirstSeenMap.arrays`, `mapmad_meta.json` with
   the object box) into <out>/drives/<episode>/;
3. read it back with the TRAINING classes (`map_sample.DriveStore` + `MapSampleBuilder`, the train config's map
   block, no perturbation) at the pose of traj_data.pkl's convention (frames.to_2d, yaw);
4. compare with the online tensor byte for byte, per layer (obstacle, explored, heat), at t = 0, the middle step
   and the last step (the block's 3 steps) and, as an extra, at every step.
Writes <out>/parity.json (pass = all equal).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mapmad_sim import config
from mapmad_sim.datagen import DriveRecord, DriveRunner
from mapmad_sim.depth_map import FirstSeenMap, MapSpec
from mapmad_sim.episodes import load_episodes
from mapmad_sim.frames import to_2d
from mapmad_sim.robot import LimoSim, RobotSpec
from mapmad_sim.run_layout import RunLayout, parse_run_id, read_config

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
from vint_train.mapmad.config import load_config  # noqa: E402
from vint_train.mapmad.map_sample import DriveStore, MapSampleBuilder  # noqa: E402

LAYERS = ("obstacle", "explored", "heat")


def replay(robot: LimoSim, p2: dict, positions: np.ndarray, yaws: np.ndarray) -> FirstSeenMap:
    """Phase 2's recording step at every logged pose, in order."""
    runner = DriveRunner(robot, p2)
    low, high = robot.pathfinder.get_bounds()
    fmap = FirstSeenMap.around(MapSpec.from_config(p2["map"]), low, high)
    rec = DriveRecord()
    for pos, yaw in zip(positions, yaws):
        robot.place(pos, float(yaw))
        runner.record_frame(rec, fmap, None)
    return fmap, rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    episodes = {e["episode_id"]: e for e in load_episodes(layout.episodes_file)[0]}
    p2 = read_config(config.CONFIG_DIR / layout.cfg["map_settings"]["builder"])
    crop = load_config(str(layout.resolve(layout.cfg["map_settings"]["crop"])))["map"]
    out = args.out or args.traces.parent.parent / "parity"
    report = {"traces": str(args.traces), "rule": "online tensor == Phase 2 builder + training reader, byte for byte, "
              "per layer, at t = 0, middle, last (and every step)", "episodes": []}
    for path in sorted(args.traces.glob("*.npz")):
        ep_id = parse_run_id(path.name)[0]
        e = episodes[ep_id]
        tr = np.load(path)
        maps, positions, yaws = tr["maps"], tr["positions"], tr["yaws"]
        with LimoSim(e["split"], e["home"], RobotSpec.from_config(config.robot()), depth=True, gpu=args.gpu) as robot:
            fmap, rec = replay(robot, p2, positions, yaws)
        drive = out / "drives" / ep_id
        drive.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(drive / "mapmad_map.npz", **fmap.arrays(float(np.median(rec.floor_y))))
        (drive / "mapmad_meta.json").write_text(json.dumps({"target": {
            "kind": "object", "box_center": e["target"]["box_center"], "box_size": e["target"]["box_size"]}}))
        builder = MapSampleBuilder(DriveStore(str(out / "drives"), [ep_id]), **dict(crop, perturb_prob=0.0))
        steps = len(maps)
        equal = []
        for t in range(steps):
            x, y = to_2d(positions[t])
            ref = builder.build(ep_id, t, (float(x), float(y), float(yaws[t])), heat_on=True)
            equal.append([bool(np.array_equal(ref[k], maps[t][k])) and ref.dtype == maps[t].dtype for k in range(3)])
        equal = np.array(equal)
        checked = sorted({0, (steps - 1) // 2, steps - 1})
        row = {"episode_id": ep_id, "trace": path.name, "steps": steps,
               "checked_steps": {str(t): dict(zip(LAYERS, map(bool, equal[t]))) for t in checked},
               "all_steps_equal": dict(zip(LAYERS, map(bool, equal.all(axis=0)))),
               "obstacle_cells_last": int(maps[-1][0].sum()), "explored_cells_last": int(maps[-1][1].sum()),
               "heat_max_last": float(maps[-1][2].max())}
        row["pass"] = bool(all(all(v.values()) for v in row["checked_steps"].values()))
        report["episodes"].append(row)
        print(json.dumps(row), flush=True)
    report["pass"] = bool(report["episodes"]) and all(r["pass"] for r in report["episodes"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity.json").write_text(json.dumps(report, indent=1) + "\n")
    print("PASS" if report["pass"] else "FAIL")


if __name__ == "__main__":
    main()
