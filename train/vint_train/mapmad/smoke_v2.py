"""Phase 10 smoke checks (chain step 2, block items 1-3) and the check sheet; any failed check exits with code 4.

1. waypoint gap: metric_waypoint_spacing x waypoint_spacing of habitat_mapmad_v2 = 0.25 m;
2. mode shares over 10,000 seeded draws (the dataset's own per-draw Generator) within 2 points of `habitat_modes`;
3. 200 real samples: map [2, 4, 64, 64] float32, masks [2]; maps shown only in search_map / explore_map; a
   search_map sample comes from a search drive, an explore_map sample from an exploration drive; explore_map's target
   layer is all zero; hidden maps are all zero; GoStanford samples have both maps hidden;
4. check sheet (report only): 8 random search_map samples of the main train split -> <out>/check_sheet.png
   (camera frame t + near / far x obstacle, explored, frontier, target).

    python -m vint_train.mapmad.smoke_v2 --config config/mapmad_v2.yaml --out /outputs/mapmad/p10_v2_train/checks
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from vint_train.mapmad import modes as M
from vint_train.mapmad.config import load_config
from vint_train.mapmad.dataset import build_mapmad_dataset

SHARE_TOL = 0.02
N_DRAWS, N_SAMPLES, N_SHEET = 10_000, 200, 8


def waypoint_gap(cfg: Dict[str, Any]) -> float:
    path = os.path.join(os.path.dirname(os.path.abspath(M.__file__)), "..", "data", "data_config.yaml")
    with open(path) as f:
        spacing = yaml.safe_load(f)["habitat_mapmad_v2"]["metric_waypoint_spacing"]
    return float(spacing) * int(cfg["datasets"]["habitat_mapmad_v2"]["waypoint_spacing"])


def mode_shares(ds, n: int = N_DRAWS, seed_key: int = 10 ** 9) -> Dict[str, float]:
    """Shares of the modes the dataset draws for keys seed_key .. seed_key + n - 1 (no sample is loaded)."""
    names = [M.draw_habitat_mode(ds.mode_p, ds.rng(seed_key + k), M.HABITAT_MODES_V2).name for k in range(n)]
    return {m: names.count(m) / n for m in M.HABITAT_MODES_V2}


def sample_checks(ds, gs, n: int = N_SAMPLES) -> List[str]:
    """Problems found in n real Habitat samples and 20 GoStanford samples (empty = fine)."""
    problems = []
    rng = np.random.default_rng(7)
    for k in range(n):
        i = int(rng.integers(len(ds)))
        key = 2 * 10 ** 9 + k
        mode = M.draw_habitat_mode(ds.mode_p, ds.rng(key), M.HABITAT_MODES_V2)
        s = ds.get(i, key)
        m, mask, mid = s[7], s[9], int(s[10])
        if tuple(m.shape) != (2, 4, 64, 64) or str(m.dtype) != "torch.float32" or tuple(mask.shape) != (2,):
            problems.append(f"sample {k}: shapes {tuple(m.shape)} {m.dtype} {tuple(mask.shape)}")
        if mid != mode.id:
            problems.append(f"sample {k}: mode id {mid} != drawn {mode.id}")
        if mode.map_mask:
            if mask.tolist() != [1, 1] or float(m.abs().sum()) != 0.0:
                problems.append(f"sample {k} ({mode.name}): hidden maps not zero/masked")
        else:
            if mask.tolist() != [0, 0]:
                problems.append(f"sample {k} ({mode.name}): shown maps masked")
            want = "search" if mode.name == "search_map" else "explore"
            # the drawn drive (re-derive the pool index with a fresh Generator of the same key: same draws)
            g = ds.rng(key)
            M.draw_habitat_mode(ds.mode_p, g, M.HABITAT_MODES_V2)
            got = ds.kind(ds.index_to_data[ds.pool_index(i, mode, g)][0])
            if got != want:
                problems.append(f"sample {k} ({mode.name}): drive kind {got}")
            if mode.name == "explore_map" and float(m[:, 3].abs().sum()) != 0.0:
                problems.append(f"sample {k}: explore_map target layer not zero")
            if float(m[0, 1].sum()) == 0.0:
                problems.append(f"sample {k} ({mode.name}): nothing explored around the robot")
    for k in range(20):
        s = gs[k]
        if s[9].tolist() != [1, 1] or float(s[7].abs().sum()) != 0.0 or int(s[10]) not in (5, 6):
            problems.append(f"GoStanford sample {k}: maps not hidden")
    return problems


def check_sheet(ds, out_png: str, n: int = N_SHEET, seed: int = 11) -> List[Dict[str, Any]]:
    """n random search_map samples: camera frame + near/far x 4 layers (report only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pool = ds.pools["search"]
    picks = [int(pool[int(rng.integers(len(pool)))]) for _ in range(n)]
    fig, axes = plt.subplots(n, 9, figsize=(18, 2.1 * n))
    rows = []
    for r, i in enumerate(picks):
        drive, t, _ = ds.index_to_data[i]
        s = ds.habitat_sample(drive, t, M.BY_NAME["search_map"], np.random.default_rng([seed, r]))
        img = s[0][-3:].permute(1, 2, 0).numpy()
        axes[r, 0].imshow(np.clip(img, 0, 1))
        axes[r, 0].set_title(f"{ds.category(drive)} t={t}", fontsize=7)
        for w, wname in enumerate(("near", "far")):
            for k, lname in enumerate(("obstacle", "explored", "frontier", "target")):
                ax = axes[r, 1 + 4 * w + k]
                ax.imshow(s[7][w, k].numpy(), vmin=0, vmax=1, cmap="magma" if lname == "target" else "gray")
                ax.set_title(f"{wname} {lname}", fontsize=7)
                ax.plot([31.5], [31.5], "c+")
        rows.append({"drive": drive, "t": int(t), "target_max_near": float(s[7][0, 3].max()),
                     "target_max_far": float(s[7][1, 3].max()), "frontier_cells_far": int(s[7][1, 2].sum())})
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Phase 10 check sheet: camera frame t, near (6.4 m) / far (25.6 m) windows, forward = up", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--no-sheet", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config, "config/defaults.yaml")
    ds = build_mapmad_dataset(cfg, "habitat_mapmad_v2", args.split, train=True)
    gs = build_mapmad_dataset(cfg, "go_stanford", "test", train=False)
    res: Dict[str, Any] = {"config": args.config, "split": args.split, "drives": len(ds.traj_names), "samples": len(ds)}
    res["waypoint_gap_m"] = waypoint_gap(cfg)
    res["shares"] = mode_shares(ds)
    res["shares_expected"] = dict(cfg["habitat_modes"])
    res["problems"] = sample_checks(ds, gs)
    if abs(res["waypoint_gap_m"] - 0.25) > 1e-9:
        res["problems"].append(f"waypoint gap {res['waypoint_gap_m']} m != 0.25 m")
    for m, share in res["shares"].items():
        if abs(share - float(cfg["habitat_modes"][m])) > SHARE_TOL:
            res["problems"].append(f"mode {m}: share {share:.4f} vs {cfg['habitat_modes'][m]}")
    os.makedirs(args.out, exist_ok=True)
    if not args.no_sheet:
        res["sheet"] = check_sheet(ds, os.path.join(args.out, "check_sheet.png"))
    res["pass"] = not res["problems"]
    with open(os.path.join(args.out, f"smoke_checks_{args.split}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("drives", "samples", "waypoint_gap_m", "shares", "problems", "pass")}, indent=1))
    if not res["pass"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
