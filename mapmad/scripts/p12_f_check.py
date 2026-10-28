"""Phase 12 F check (block item 3; automatic stop if it fails): arm F's teacher, run inside the simulator server exactly
as in the exam (sidecar_server.SimService + search_exam.FrontierAgent, arm F_truth: "seen" = Phase 8's simulator-truth
seen instances), makes the SAME decisions as the Phase 8 teacher on 2 Phase 8 search drives of the dev homes.

    python mapmad/scripts/p12_f_check.py --gpu 0 [--config mapmad/configs/p12_exam.yaml]   # inside naz_mapmad_habitat

Drives: 2 seeded (seed 12301) search drives with an approach of the Phase 8 test split in the 10 Phase 1 dev homes
(never exam homes). Each drive is replayed pose by pose through the server (as p11_map_parity.py, "poses" mode: the
robot is put on every saved pose; the generator's bump events are applied after the move they belong to). At every
frame the agent's decision -- mode, goal, next path point -- must equal the drive's saved teacher_mode, goal_xy and
next_point_xy (NaN = no goal). The arm's settings come from search_wire.arm_wire (as every exam arm's). Result:
<out_dir>/checks/f_check.json (--out); exit 3 when a frame differs.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mapmad_sim import config, objectnav
from mapmad_sim.arm_table import ArmTable
from mapmad_sim.robot import Move
from mapmad_sim.run_layout import RunLayout, write_text_atomic
from mapmad_sim.search_wire import arm_wire
from mapmad_sim.sidecar_server import SimService, limo_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))
from p11_map_parity import episode_of  # noqa: E402
from vint_train.mapmad.config import load_config  # noqa: E402

DRIVE_SEED = 12301
N_DRIVES = 2


def dev_search_drives(n: int, seed: int) -> List[str]:
    """n seeded Phase 8 test-split search drives in the 10 Phase 1 dev homes that reached an approach."""
    paths = config.paths()
    split = json.loads((config.CONFIG_DIR / "p2_home_splits.json").read_text())["rule"]["val_drive"]["phase1_homes"]
    names = (paths["mapmad_data"] / "splits/habitat_mapmad_v2/test/traj_names.txt").read_text().split()
    cands = sorted(x for x in names if x.split("-")[1] in split and "-search-" in x)
    out = []
    for k in np.random.default_rng(seed).permutation(len(cands)):
        folder = paths["mapmad_data"] / "habitat_mapmad_v2" / cands[int(k)]
        with np.load(folder / "mapmad_frames.npz") as f:
            if len(f["seen_events"]) and len(f["v"]) <= 600:  # an approach, and a replay of a few minutes
                out.append(cands[int(k)])
        if len(out) == n:
            break
    return sorted(out)


def same_xy(a, b) -> bool:
    a = np.asarray([np.nan, np.nan] if a is None else a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.array_equal(a, b, equal_nan=True))


def check_drive(folder: Path, gpu: int, settings: Dict[str, Any], camera: Dict[str, Any]) -> Dict[str, Any]:
    meta = json.loads((folder / "mapmad_meta.json").read_text())
    with np.load(folder / "mapmad_frames.npz") as f:
        fr = {k: f[k] for k in f.files}
    with open(folder / "traj_data.pkl", "rb") as f:
        traj = pickle.load(f)
    ids = sorted({g.object_id for g in objectnav.home_goals(meta["split"], meta["home"]) if g.category == meta["category"]})
    names = [str(x) for x in fr["teacher_mode_names"]]
    service = SimService(limo_factory(gpu))
    sv = dict(settings, noise_seed=[int(meta["seed"]), 1], record=False)
    reply = service.reset(episode_of(meta, ids), camera["hfov_deg"], False, 0.05, {"search_v2": sv}, camera)
    events = np.asarray(fr["bump_events"], dtype=np.float64).reshape(-1, 3)
    n = len(traj["position"])
    bad: List[Dict[str, Any]] = []
    modes = {}
    for t in range(n):
        dec = reply["teacher"]
        want = names[int(fr["teacher_mode"][t])]
        modes[want] = modes.get(want, 0) + 1
        ok = dec["mode"] == want and same_xy(dec["goal"], fr["goal_xy"][t]) and same_xy(dec["next_point"], fr["next_point_xy"][t])
        if not ok:
            bad.append({"frame": t, "mode": [dec["mode"], want], "goal": [dec["goal"], fr["goal_xy"][t].tolist()]})
        if t == n - 1:
            break
        service.robot.place(fr["position_habitat"][t + 1], float(traj["yaw"][t + 1]))
        move = Move(v=float(fr["v"][t]), w=float(fr["w"][t]), travelled_m=float(fr["travelled_m"][t]), short_m=0.0,
                    refused_substeps=0, collided=bool(fr["collided"][t]))
        service.search.after_move(move, service.pose2d(), service.robot.spec.dt)
        want_b = events[events[:, 0] == t]
        if len(want_b):
            cells, inside = service.search.fmap.cells(want_b[:, 1:])
            for c in cells[inside]:
                service.search.bumped.flat[int(c)] = True
        reply = {**service.observation(), "state": service.state()}
    service.shutdown()
    return {"drive": meta["drive_id"], "frames": n, "mismatched_frames": len(bad), "first_mismatches": bad[:5],
            "modes": modes, "pass": not bad}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--config", type=Path, default=config.CONFIG_DIR / "p12_exam.yaml")
    p.add_argument("--out", default="checks/f_check.json", help="below out_dir")
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    w = arm_wire(ArmTable.from_layout(layout), "F_truth", load_config)  # arm F_truth exactly as the exam wires arms
    settings, camera = w.search_v2, w.camera
    drives = dev_search_drives(N_DRIVES, DRIVE_SEED)
    rows = []
    for d in drives:
        rows.append(check_drive(config.paths()["mapmad_data"] / "habitat_mapmad_v2" / d, args.gpu, settings, camera))
        print(json.dumps(rows[-1]), flush=True)
    ok = len(rows) == N_DRIVES and all(r["pass"] for r in rows)
    write_text_atomic(layout.base / args.out,
                      json.dumps({"pass": ok, "drives": rows, "seed": DRIVE_SEED,
                                  "rule": "F_truth's decision (mode, goal, next point) = the Phase 8 teacher's at every "
                                          "frame of 2 Phase 8 dev-home search drives (pose replay)"}, indent=1) + "\n")
    print(json.dumps({"pass": ok}), flush=True)
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
