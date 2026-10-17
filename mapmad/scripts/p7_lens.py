"""Phase 7.1 / 7.3: the fisheye lens fitted to SACSoN's real pictures, and the S-lens sheet.

    python mapmad/scripts/p7_lens.py fit       # -> <out>/lens/sacson_lens.json (+ sacson_circle.png)
    python mapmad/scripts/p7_lens.py sheet     # -> <out>/lens/lens_sheet.png (real frames vs renders, also at 96 x 96)

fit: SACSoN's calibration file is not downloadable without a Google sign-in and the bags carry no camera_info, so the
lens is fitted to the pictures: the mean of all converted SACSoN frames shows the dark ring outside the image circle;
a circle fitted to the ring's edge gives the principal point (its centre) and the 85 deg radius (half of the paper's
170 deg); the double sphere is then the one closest to the ideal equidistant curve through that circle
(mapmad_sim.lens.DoubleSphere.fit_equidistant). Resolution = SACSoN's 160 x 120 (already 4:3: NoMaD's crop is a
no-op, the resize to 96 x 96 happens in the policy as for every camera).
"""

import argparse
import dataclasses
import glob
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np

from mapmad_sim import config
from mapmad_sim.lens import DoubleSphere
from mapmad_sim.run_layout import RunLayout

FOV_DEG = 170.0  # SACSoN paper (arXiv 2306.01874, Sec. IV-A): "a 170 deg wide-angle RGB camera"
DARK_SHARE = 0.25  # mean-image pixels darker than this share of the median brightness are outside the lens (0.35+
#                    lets the dark floor into the bottom corners and pulls the centre up: 0.15 / 0.25 / 0.35 / 0.45 -> cy 54.5 /
#                    54.0 / 49.3 / 36.7, radius 97.4 / 96.9 / 95.8 / 96.1 px)


def p7_dir() -> Path:
    """<outputs>/<out_dir> of the sweep config (p7_sweep.yaml): every Phase 7 output folder lives below it."""
    return RunLayout.load(config.CONFIG_DIR / "p7_sweep.yaml").base


def out_dir() -> Path:
    return p7_dir() / "lens"


def mean_image(pattern: str, max_frames: int, seed: int) -> np.ndarray:
    """Mean grey picture of up to max_frames JPEGs (seeded choice)."""
    import cv2

    files = sorted(glob.glob(pattern))
    rng = np.random.default_rng(seed)
    files = [files[i] for i in rng.choice(len(files), min(max_frames, len(files)), replace=False)]
    acc = None
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        acc = g if acc is None else acc + g
    return acc / len(files)


def ring_edge(mean: np.ndarray) -> np.ndarray:
    """(N, 2) pixel centres (u, v) of dark pixels that touch a bright pixel: the edge of the image circle."""
    dark = mean < DARK_SHARE * np.median(mean)
    bright = ~dark
    nb = np.zeros_like(dark)
    nb[1:, :] |= bright[:-1, :]
    nb[:-1, :] |= bright[1:, :]
    nb[:, 1:] |= bright[:, :-1]
    nb[:, :-1] |= bright[:, 1:]
    v, u = np.nonzero(dark & nb)
    return np.stack([u + 0.5, v + 0.5], axis=1)


def fit_circle(points: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares circle (Kasa fit): centre (cx, cy) and radius."""
    x, y = points[:, 0], points[:, 1]
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    return float(cx), float(cy), float(math.sqrt(sol[2] + cx * cx + cy * cy))


def hfov_deg(lens: DoubleSphere) -> float:
    """Horizontal field of view across the picture's full width at the principal point's row (degrees)."""
    rays, _ = DoubleSphere(**{**lens.to_config(), "circle_px": None}).unproject(np.array([[0.0, lens.cy], [lens.width, lens.cy]]))
    return float(sum(math.degrees(math.atan2(abs(r[0]), r[2])) for r in rays))


def cmd_fit(args: argparse.Namespace) -> None:
    import cv2

    pattern = str(config.paths()["nomad_data"] / "sacson" / "*" / "*.jpg")
    mean = mean_image(pattern, args.frames, args.seed)
    h, w = mean.shape
    edge = ring_edge(mean)
    if len(edge) < 20:
        raise RuntimeError(f"only {len(edge)} ring-edge pixels: no dark ring found")
    cx, cy, radius = fit_circle(edge)
    lens = DoubleSphere.fit_equidistant(FOV_DEG, radius, w, h, cx=cx, cy=cy)
    res = {"source": "mean of SACSoN frames (no calibration file available)", "frames": args.frames, "seed": args.seed,
           "fov_deg": FOV_DEG, "dark_share": DARK_SHARE, "ring_edge_pixels": int(len(edge)),
           "circle": {"cx": cx, "cy": cy, "radius_px": radius},
           "circle_fit_rms_px": float(np.sqrt(np.mean((np.hypot(edge[:, 0] - cx, edge[:, 1] - cy) - radius) ** 2))),
           "double_sphere": lens.to_config(), "fov_check_deg": lens.fov_deg(),
           "hfov_deg": hfov_deg(lens)}
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "sacson_lens.json").write_text(json.dumps(res, indent=1) + "\n")
    vis = cv2.cvtColor(np.clip(mean, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
    cv2.circle(vis, (int(cx * 4), int(cy * 4)), int(radius * 4), (0, 0, 255), 1)
    for u, v in edge:
        cv2.circle(vis, (int(u * 4), int(v * 4)), 1, (0, 255, 0), -1)
    cv2.imwrite(str(out / "sacson_circle.png"), vis)
    print(json.dumps(res, indent=1))


# --- S-lens sheet ---------------------------------------------------------------------------------------------------

SHEET_REAL = {  # picked by eye from contact sheets of random frames (lens/pick_*.png, pick_index.txt): real GoStanford
    # (no*vc = real, unflipped) and SACSoN frames in a corridor, a room and near furniture; RECON (outdoors only)
    "go_stanford": [("corridor", "raw/go_stanford/go_stanford/no1vc_17_0/218.jpg"),
                    ("room", "raw/go_stanford/go_stanford/no7vc_29_0/87.jpg"),
                    ("furniture", "raw/go_stanford/go_stanford/no2vc_156_2/20.jpg")],
    "sacson": [("corridor", "sacson/Feb-17-2023-soda3_00000002_3/58.jpg"),
               ("room", "sacson/Jan-12-2023-bww8_00000005_0/145.jpg"),
               ("furniture", "sacson/Dec-06-2022-bww8_00000031_0/149.jpg")],
    "recon": [("room", "jackal_2019-08-08-15-54-38_6_r01/10.jpg"),
              ("corridor", "jackal_2019-08-27-16-56-55_23_r02/10.jpg")],
}
SHEET_HEIGHT = {"go_stanford": 0.55, "sacson": 0.25, "recon": 0.95}  # 7.4 (height_estimates.md); RECON: block 0.95
LAYOUT_HEIGHTS = (0.15, 0.25, 0.40, 0.55)  # block item 4 fallback: layout vs renders at several heights
TILE = (240, 180)


def nomad_input(rgb: np.ndarray) -> np.ndarray:
    """What NoMaD sees: training's centre crop to 4:3 and PIL resize to 96 x 96 (shown enlarged, nearest)."""
    import cv2
    from PIL import Image

    img = Image.fromarray(rgb)
    w, h = img.size
    if w * 3 != h * 4:  # crop to 4:3 as resize_and_aspect_crop
        cw, ch = (int(h * 4 / 3), h) if w * 3 > h * 4 else (w, int(w * 3 / 4))
        img = img.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    small = np.asarray(img.resize((96, 96)))
    return cv2.resize(small, (TILE[1], TILE[1]), interpolation=cv2.INTER_NEAREST)


def find_poses(robot, rng: np.random.Generator, tries: int = 600) -> dict:
    """A corridor view (far ahead, walls close left and right) and a room view (open) from the 90 deg depth camera at
    the render height; both need clear floor ahead (lower band of the picture > 1.2 m away)."""
    import math

    found = {}
    for _ in range(tries):
        p = np.asarray(robot.pathfinder.get_random_navigable_point(), dtype=np.float64)
        yaw = float(rng.uniform(-math.pi, math.pi))
        try:
            robot.place(p, yaw)
        except ValueError:
            continue
        d = robot.observe()["depth"]
        h, w = d.shape
        mid = d[h // 3: 2 * h // 3]
        c, l, r = (np.median(mid[:, a:b]) for a, b in ((2 * w // 5, 3 * w // 5), (0, w // 4), (3 * w // 4, w)))
        clear = np.median(d[2 * h // 3:, w // 4: 3 * w // 4]) > 1.2
        if "corridor" not in found and clear and c > 3.0 and l < 1.3 and r < 1.3:
            found["corridor"] = (p.tolist(), yaw)
        if "room" not in found and clear and 2.5 < c < 6.0 and l > 1.8 and r > 1.8:
            found["room"] = (p.tolist(), yaw)
        if len(found) == 2:
            break
    return found


def furniture_pose(eps: list, home: str, rng: np.random.Generator):
    """A goal-photo pose of a sofa / chair / bed episode in this home (the object ~1 m ahead)."""
    cands = [e for e in eps if e["home"] == home and e["target"]["category"] in ("sofa", "chair", "bed")]
    e = cands[int(rng.integers(len(cands)))]
    return e["goal_photo_position"], e["goal_photo_yaw"]


def render(home: str, pose, setting, gpu: int) -> np.ndarray:
    from mapmad_sim.robot import LimoSim, RobotSpec

    with LimoSim("train", home, setting.apply(RobotSpec.from_config(config.robot())), gpu=gpu) as robot:
        robot.place(pose[0], pose[1])
        return robot.observe()["rgb"]


def label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def cmd_sheet(args: argparse.Namespace) -> None:
    import cv2

    from mapmad_sim.camera_setting import CameraSetting
    from mapmad_sim.robot import LimoSim, RobotSpec

    lens = DoubleSphere.from_config(json.loads((out_dir() / "sacson_lens.json").read_text())["double_sphere"])
    fish = lambda h: CameraSetting(f"fisheye_{h:.2f}", "fisheye", h, 0.0, double_sphere=lens)  # noqa: E731
    eps = json.loads(RunLayout.load(config.CONFIG_DIR / "p7_sweep.yaml").episodes_file.read_text())["episodes"]
    homes = sorted({e["home"] for e in eps})
    rng = np.random.default_rng(args.seed)
    roots = {"go_stanford": config.paths()["nomad_data"], "sacson": config.paths()["nomad_data"], "recon": config.paths()["recon"]}
    rows, picks = [], []
    k = 0
    for ds in ("go_stanford", "sacson", "recon"):
        for kind, rel in SHEET_REAL[ds]:
            spec = level_depth90_spec(RobotSpec.from_config(config.robot()), SHEET_HEIGHT[ds])
            for _ in range(len(homes)):  # corridor / room: the next home that has one (homes rarely have corridors)
                home = homes[k % len(homes)]
                k += 1
                if kind == "furniture":
                    pose = furniture_pose(eps, home, rng)
                    break
                with LimoSim("train", home, spec, depth=True, gpu=args.gpu, seed=args.seed) as robot:
                    poses = find_poses(robot, rng)
                if kind in poses:
                    pose = poses[kind]
                    break
            else:
                raise RuntimeError(f"no {kind} view found in any dev home")
            real = cv2.cvtColor(cv2.imread(str(roots[ds] / rel)), cv2.COLOR_BGR2RGB)
            sim = render(home, pose, fish(SHEET_HEIGHT[ds]), args.gpu)
            picks.append({"dataset": ds, "kind": kind, "real": rel, "home": home, "pose": pose, "height_m": SHEET_HEIGHT[ds]})
            rows.append(np.hstack([label(cv2.resize(real, TILE), f"real {ds} ({kind})"),
                                   label(cv2.resize(sim, TILE), f"render fisheye {SHEET_HEIGHT[ds]:.2f} m {home[:5]}"),
                                   label(nomad_input(real), "real 96x96"), label(nomad_input(sim), "render 96x96")]))
    sheet = np.vstack(rows)
    # layout fallback rows: one real frame per indoor dataset next to one home pose rendered at several heights
    layout_rows = []
    for ds, (kind, rel) in (("sacson", SHEET_REAL["sacson"][1]), ("go_stanford", SHEET_REAL["go_stanford"][1])):
        home = picks[[p["dataset"] for p in picks].index(ds) + 1]["home"]
        pose = picks[[p["dataset"] for p in picks].index(ds) + 1]["pose"]
        real = cv2.cvtColor(cv2.imread(str(roots[ds] / rel)), cv2.COLOR_BGR2RGB)
        tiles = [label(cv2.resize(real, TILE), f"real {ds}")]
        tiles += [label(cv2.resize(render(home, pose, fish(h), args.gpu), TILE), f"fisheye {h:.2f} m") for h in LAYOUT_HEIGHTS]
        layout_rows.append(np.hstack(tiles))
    layout = np.vstack(layout_rows)
    width = max(sheet.shape[1], layout.shape[1])
    pad = lambda im: np.pad(im, ((0, 0), (0, width - im.shape[1]), (0, 0)))  # noqa: E731
    gap = np.full((12, width, 3), 255, np.uint8)
    out = np.vstack([pad(sheet), gap, pad(layout)])
    cv2.imwrite(str(out_dir() / "lens_sheet.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    (out_dir() / "lens_sheet.json").write_text(json.dumps({"seed": args.seed, "lens": lens.to_config(), "rows": picks,
                                                           "layout_heights_m": LAYOUT_HEIGHTS}, indent=1) + "\n")
    print(out_dir() / "lens_sheet.png", out.shape)


def level_depth90_spec(spec, height_m: float):
    """LIMO with a level 90 deg (pinhole) camera at the render height, for picking views by depth."""
    return dataclasses.replace(spec, hfov_deg=90.0, camera_height_m=height_m, pitch_deg=0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--frames", type=int, default=5000)
    f.add_argument("--seed", type=int, default=7100)
    sh = sub.add_parser("sheet")
    sh.add_argument("--seed", type=int, default=7300)
    sh.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    {"fit": cmd_fit, "sheet": cmd_sheet}[args.cmd](args)


if __name__ == "__main__":
    main()
