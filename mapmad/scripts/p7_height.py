"""Phase 7.4 Method B: camera height (and pitch) of the NoMaD training robots from their own pictures + odometry.

    python mapmad/scripts/p7_height.py estimate --dataset recon        # one dataset -> <out>/height/method_b_<name>.json
    python mapmad/scripts/p7_height.py synthetic                         # Habitat pictures at known heights (check)

Datasets (NoMaD format: <traj>/<i>.jpg + traj_data.pkl with position (N, 2), yaw (N,); 4 Hz):
- recon: Jackal; published calibration (data_config.yaml camera_metrics: pinhole + radial-tangential, 640 x 480),
  camera 0.45 m ahead of the centre; the check dataset (published height 0.95 m).
- sacson: Roomba-based robot, 170 deg fisheye 160 x 120; lens = p7_camera/lens/sacson_lens.json (fitted in 7.1).
- go_stanford: TurtleBot2, Ricoh THETA S 180 deg fisheye cropped to 160 x 120; no calibration -> generic 180 deg
  equidistant model (focal length swept for a wider range).

Pairs: frames (i, i + k) of one trajectory with a straight forward drive of 0.3-0.8 m (|turn| < 5 deg, |sideways|
< 0.1 m), seeded random choice, at most `per_traj` per trajectory. Per pair: grid search over height x pitch
(mapmad_sim.floor_height); per dataset: median and 5-95 % of the per-pair heights (block item 4) plus the pooled
estimate (one height for all pairs) as a second number.
"""

import argparse
import glob
import json
import math
import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mapmad_sim import config
from mapmad_sim.floor_height import pool, relative_pose, score_grid
from mapmad_sim.lens import DoubleSphere, Equidistant, OpenCVPinhole
from mapmad_sim.run_layout import RunLayout

HEIGHTS = np.round(np.arange(0.08, 1.501, 0.01), 3)
PITCHES = np.arange(-12.0, 12.01, 1.0)
RECON_CAMERA = dict(fx=272.547, fy=266.358, cx=320.0, cy=220.0, width=640, height=480,
                    k1=-0.038483, k2=-0.010456, p1=0.003930, p2=-0.001007)  # train/vint_train/data/data_config.yaml


def p7_dir() -> Path:
    """<outputs>/<out_dir> of the sweep config (p7_sweep.yaml): every Phase 7 output folder lives below it."""
    return RunLayout.load(config.CONFIG_DIR / "p7_sweep.yaml").base


def out_dir() -> Path:
    return p7_dir() / "height"


def sacson_lens() -> DoubleSphere:
    """The double sphere fitted to SACSoN's frames (7.1, p7_lens.py fit)."""
    return DoubleSphere.from_config(json.loads((p7_dir() / "lens" / "sacson_lens.json").read_text())["double_sphere"])


def dataset_spec(name: str, go_stanford_f: float = None) -> Dict[str, Any]:
    """Folder glob, camera model, camera forward offset (m) and frame step for one dataset."""
    p = config.paths()
    if name == "recon":
        return {"glob": str(p["recon"] / "*"), "lens": OpenCVPinhole(**RECON_CAMERA), "forward_m": 0.45}
    if name == "sacson":
        return {"glob": str(p["nomad_data"] / "sacson" / "*"), "lens": sacson_lens(), "forward_m": 0.0}
    if name == "go_stanford":
        f = go_stanford_f or 80.0 / (math.pi / 2)  # 180 deg across the 160 px width
        return {"glob": str(p["nomad_data"] / "raw" / "go_stanford" / "go_stanford" / "no*vc_*"),
                "lens": Equidistant(f=f, cx=80.0, cy=60.0, width=160, height=120), "forward_m": 0.0}
    raise ValueError(name)


def pick_pairs(folders: List[str], n_pairs: int, per_traj: int, seed: int,
               move_m: Tuple[float, float] = (0.3, 0.8)) -> List[Tuple[str, int, int, Tuple[float, float, float]]]:
    """(folder, i, j, odometry of j in i's frame) for straight forward drives of move_m metres."""
    rng = random.Random(seed)
    folders = list(folders)
    rng.shuffle(folders)
    out = []
    for folder in folders:
        try:
            traj = pickle.load(open(Path(folder) / "traj_data.pkl", "rb"))
        except (OSError, EOFError, pickle.UnpicklingError):
            continue
        pos, yaw = np.asarray(traj["position"], dtype=np.float64), np.asarray(traj["yaw"], dtype=np.float64).ravel()
        cands = []
        for i in range(len(pos)):
            for j in range(i + 1, min(i + 12, len(pos))):
                dx, dy, dyaw = relative_pose(pos[i], yaw[i], pos[j], yaw[j])
                if dx > move_m[1]:
                    break
                if dx >= move_m[0] and abs(dy) < 0.1 and abs(math.degrees(dyaw)) < 5.0:
                    cands.append((folder, i, j, (dx, dy, dyaw)))
                    break
        for c in rng.sample(cands, min(per_traj, len(cands))):
            if (Path(folder) / f"{c[1]}.jpg").exists() and (Path(folder) / f"{c[2]}.jpg").exists():
                out.append(c)
        if len(out) >= n_pairs:
            break
    return out[:n_pairs]


def grey(path: Path, blur: float = 1.0) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), blur) if blur > 0 else g


def pair_surface(job: Tuple[Tuple[str, int, int, Tuple[float, float, float]], Any, float]) -> np.ndarray:
    (f, i, j, rel), lens, forward_m = job
    return score_grid(grey(Path(f) / f"{i}.jpg"), grey(Path(f) / f"{j}.jpg"), lens, rel, HEIGHTS, PITCHES,
                      forward_m=forward_m)


def surfaces_for(pairs, lens: Any, forward_m: float, workers: int = 1) -> np.ndarray:
    """(n_pairs, n_pitches, n_heights) NCC surfaces; workers > 1 = a process pool (same numbers, pair order kept)."""
    jobs = [(pair, lens, forward_m) for pair in pairs]
    if workers <= 1:
        return np.stack([pair_surface(j) for j in jobs])
    from multiprocessing import Pool

    with Pool(workers) as pool:
        return np.stack(pool.map(pair_surface, jobs, chunksize=1))


def summarise(surf: np.ndarray, n_boot: int, seed: int) -> Dict[str, Any]:
    """Per-pair argmax heights/pitches (median, 5-95 %) and the pooled estimate."""
    flat = surf.reshape(len(surf), -1)
    best = flat.argmax(axis=1)
    ok = flat.max(axis=1) > 0.3  # pairs whose best NCC is clearly above noise
    k, i = np.unravel_index(best, surf.shape[1:])
    h, p = HEIGHTS[i][ok], PITCHES[k][ok]
    pooled = pool(surf[ok], HEIGHTS, PITCHES, n_boot=n_boot, seed=seed)
    return {"n_pairs": int(len(surf)), "n_pairs_used": int(ok.sum()),
            "per_pair": {"height_median_m": float(np.median(h)), "height_p5_p95_m": [float(v) for v in np.percentile(h, [5, 95])],
                         "pitch_median_deg": float(np.median(p)), "pitch_p5_p95_deg": [float(v) for v in np.percentile(p, [5, 95])]},
            "pooled": asdict(pooled)}


def cmd_estimate(args: argparse.Namespace) -> None:
    spec = dataset_spec(args.dataset, args.go_stanford_f)
    folders = sorted(glob.glob(spec["glob"]))
    pairs = pick_pairs(folders, args.pairs, args.per_traj, args.seed)
    surf = surfaces_for(pairs, spec["lens"], spec["forward_m"], args.workers)
    res = {"dataset": args.dataset, "lens": type(spec["lens"]).__name__, "lens_params": asdict(spec["lens"]),
           "forward_m": spec["forward_m"], "seed": args.seed, "heights_m": [HEIGHTS[0], HEIGHTS[-1], 0.01],
           "pitches_deg": [PITCHES[0], PITCHES[-1], 1.0], "pairs": [[Path(f).name, i, j, list(r)] for f, i, j, r in pairs],
           **summarise(surf, args.boot, args.seed)}
    name = args.dataset + (f"_f{args.go_stanford_f:.1f}" if args.go_stanford_f else "")
    out = out_dir() / f"method_b_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    np.save(out.with_suffix(".surfaces.npy"), surf.astype(np.float32))
    print(json.dumps({k: res[k] for k in ("dataset", "n_pairs", "n_pairs_used", "per_pair", "pooled")}, indent=1, default=float))


def cmd_synthetic(args: argparse.Namespace) -> None:
    """Method B on Habitat pictures with known heights (pinhole 90 deg and the SACSoN-fitted fisheye)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from p7_synthetic_height import synthetic_pairs

    lens = sacson_lens()
    forward_m = float(config.robot()["camera"]["forward_of_base_m"])  # the simulated LIMO's camera offset
    rows = []
    for name, fish, model in [("pinhole90", None, OpenCVPinhole.from_hfov(90.0, 320, 240)), ("fisheye", lens, lens)]:
        for h in args.heights:
            pairs = synthetic_pairs(h, 0.0 if fish else -2.0, fisheye=fish, n_pairs=args.pairs, steps=args.steps)
            import cv2

            surf = np.stack([score_grid(cv2.GaussianBlur(a, (0, 0), 1.0), cv2.GaussianBlur(b, (0, 0), 1.0), model, rel,
                                        HEIGHTS, PITCHES, forward_m=forward_m) for a, b, rel in pairs])
            rows.append({"camera": name, "true_height_m": h, **summarise(surf, args.boot, args.seed)})
            print(json.dumps(rows[-1], default=float))
    (out_dir() / "method_b_synthetic.json").write_text(json.dumps(rows, indent=1, default=float) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--dataset", required=True, choices=["recon", "sacson", "go_stanford"])
    e.add_argument("--pairs", type=int, default=200)
    e.add_argument("--per-traj", type=int, default=2)
    e.add_argument("--seed", type=int, default=7400)
    e.add_argument("--boot", type=int, default=1000)
    e.add_argument("--workers", type=int, default=1, help="processes (results do not depend on it)")
    e.add_argument("--go-stanford-f", type=float, default=None, help="equidistant focal length (px) for go_stanford")
    s = sub.add_parser("synthetic")
    s.add_argument("--heights", type=float, nargs="+", default=[0.18, 0.40, 0.95])
    s.add_argument("--pairs", type=int, default=30)
    s.add_argument("--steps", type=int, default=10, help="0.05 m control steps between the two pictures")
    s.add_argument("--seed", type=int, default=7400)
    s.add_argument("--boot", type=int, default=500)
    args = p.parse_args()
    {"estimate": cmd_estimate, "synthetic": cmd_synthetic}[args.cmd](args)


if __name__ == "__main__":
    main()
