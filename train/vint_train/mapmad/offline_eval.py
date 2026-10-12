"""Offline check of the map effect (gate G3 rows 4-5; Phase 3 confirmation items 12, 14, 19): the command line.
The parts: `offline_list.py` (frozen list), `offline_scoring.py` (scoring), `offline_stats.py` (statistics). Style of
`train/ablation/eval_paired.py`: fixed sample lists, fixed noise, deterministic kernels, per-sample scores.

Commands (inside naz_mapmad, from /app/visualnav-transformer/train):

1. `make-list` -- the frozen list of 5,000 Habitat val-drive test samples (collision drives excluded), spread
   evenly over the drives (7-8 each), frames t with t - 15 >= 0 and t + 40 <= last frame (so the same list works
   for waypoint spacing 1, 2 and 5). Written as JSON with its sha256.
2. `score` -- one checkpoint, every arm on every listed sample; `gc_action_loss` = MSE between the expert's 8
   waypoints and the waypoints from the full 10-step reverse diffusion (normalised units, NoMaD's
   `_compute_losses_nomad` formula per sample). Every sample's noise -- starting noise and the scheduler noise of
   each reverse step -- comes from its own id (`diffusion.item_noise`), so it is the same in every arm, batch size
   and subset (Phase 5 code list item 1; until Phase 4 the scheduler noise was drawn per batch).
   Arms (same samples):
     map_goal    photo hidden, map shown, correct heat, unperturbed            (a)
     map_hidden  photo hidden, map hidden (= plain explore)                    (b)
     wrong_heat  heat rotated 180 deg about the robot, obstacles/explored kept (c)
     photo / photo_map / explore_map  per-mode numbers; the photo goal is k waypoints ahead, k drawn per sample
                 from {4..min(19, K)} (inside NoMaD's goal range, action mask 1)  (d)
   Also per sample and map-shown arm: the map attention share = the mean attention weight the other visible
   tokens give to the map token (over 4 layers x 4 heads).
3. `compare` -- from one `score` output: mean(a - b) with the DRIVE-level paired bootstrap 95% CI (10,000
   resamples of drives; the gate uses this one) and the per-sample CI for reference; R = (b - a) / b with its
   drive-level CI; (c) - (a); per target type; per mode; attention share per mode.
4. `gostanford` -- row 5: a fixed seeded list of GoStanford test samples (5,000 or all) with positive photo goals
   k in {4..min(19, K)}; `gc_action_loss` of a MapMaD checkpoint (map hidden, photo shown) vs the official
   nomad.pth, same samples, same noise; pass: MapMaD <= official + 0.10.

Examples:
    python -m vint_train.mapmad.offline_eval make-list --config config/mapmad.yaml \
        --out /outputs/mapmad/p3_model/offline/offline_samples.json
    CUDA_VISIBLE_DEVICES=0 python -m vint_train.mapmad.offline_eval score --config config/mapmad.yaml \
        --checkpoint <run>/ema_latest.pth --list /outputs/mapmad/p3_model/offline/offline_samples.json --out <dir>
    python -m vint_train.mapmad.offline_eval compare --scores <dir>/scores.npz --out <dir>
    CUDA_VISIBLE_DEVICES=0 python -m vint_train.mapmad.offline_eval gostanford --config config/mapmad.yaml \
        --checkpoint <run>/ema_latest.pth --official /outputs/mapmad/weights/official/nomad.pth --out <dir>
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from typing import List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from vint_train.mapmad.closed_loop.metrics import bootstrap_ci  # noqa: E402
from vint_train.mapmad.config import git_commit, load_config  # noqa: E402
from vint_train.mapmad.dataset import build_mapmad_dataset, read_drive_list  # noqa: E402
from vint_train.mapmad.offline_list import (AHEAD_FRAMES, BACK_FRAMES, LIST_SEED, N_SAMPLES,  # noqa: E402
                                            canonical_sha256, load_list, make_list)
from vint_train.mapmad.offline_scoring import (ARMS, NOISE_SEED, GoStanfordPhotoSamples, score_arm,  # noqa: E402
                                               score_gostanford)
from vint_train.mapmad.offline_stats import N_BOOT, compare  # noqa: E402
from vint_train.mapmad.reference_forward import enforce_determinism  # noqa: E402
from vint_train.mapmad.weights import load_checkpoint  # noqa: E402


def cmd_make_list(args) -> None:
    cfg = load_config(args.config)
    d = cfg["datasets"]["habitat_mapmad"]
    excluded = read_drive_list(d.get("exclude_drives"))
    names = [x.strip() for x in open(os.path.join(d["test"], "traj_names.txt")) if x.strip()]
    drives = [x for x in names if x not in excluded]
    samples = make_list(d["data_folder"], drives)
    body = {"samples": samples, "n_drives": len(drives), "n_excluded": len(names) - len(drives), "seed": LIST_SEED,
            "rule": f"t - {BACK_FRAMES} >= 0 and t + {AHEAD_FRAMES} <= last frame",
            "test_traj_names_sha256": hashlib.sha256(open(os.path.join(d["test"], "traj_names.txt"), "rb").read()).hexdigest()}
    body["sha256"] = canonical_sha256(samples)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; the offline list is frozen")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(body, f)
    kinds = {k: sum(s["type"] == k for s in samples) for k in ("spot", "object")}
    print(f"{len(samples)} samples from {len(drives)} drives ({body['n_excluded']} excluded); types {kinds}; "
          f"sha256 {body['sha256']}")


def cmd_score(args) -> None:
    enforce_determinism()
    cfg = load_config(args.config)
    body = load_list(args.list)
    samples = body["samples"][:args.limit] if args.limit else body["samples"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(cfg, args.checkpoint, device, map_input=True)
    ds = build_mapmad_dataset(cfg, "habitat_mapmad", "test", train=False)
    arms = args.arms.split(",") if args.arms else list(ARMS)
    os.makedirs(args.out, exist_ok=True)
    scores = {}
    for arm in arms:
        for k, v in score_arm(model, ds, samples, arm, cfg, device, args.batch_size, args.workers).items():
            scores[f"{arm}.{k}"] = v
    np.savez(os.path.join(args.out, "scores.npz"), ids=np.array([s["id"] for s in samples]),
             drive=np.array([s["drive"] for s in samples]), type=np.array([s["type"] for s in samples]), **scores)
    manifest = {"checkpoint": args.checkpoint, "config": args.config, "list": args.list, "list_sha256": body["sha256"],
                "waypoint_spacing": cfg["datasets"]["habitat_mapmad"].get("waypoint_spacing", 1), "arms": arms,
                "batch_size": args.batch_size, "noise_seed": NOISE_SEED, "limit": args.limit, "git_commit": git_commit()}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print({a: float(np.mean(scores[f"{a}.gc_action_loss"])) for a in arms})


def cmd_compare(args) -> None:
    with np.load(args.scores) as z:
        scores = {k: z[k] for k in z.files}
    res = compare(scores)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "compare.json"), "w") as f:
        json.dump(res, f, indent=1)
    r = res["row4"]
    print(f"map {r['mean_a']:.4f} vs hidden {r['mean_b']:.4f}: diff {r['diff']:+.4f} CI(drive) "
          f"[{r['diff_ci_drive'][0]:+.4f}, {r['diff_ci_drive'][1]:+.4f}] R {r['R']:+.3f} -> row4 "
          f"{'PASS' if r['pass'] else 'FAIL'}")
    print(json.dumps({"means": res["means"], "attention_share": res["attention_share"]}, indent=1))


def cmd_gostanford(args) -> None:
    from vint_train.mapmad.reference_forward import test_dataset

    enforce_determinism()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = test_dataset("go_stanford", cfg, cfg)
    samples = GoStanfordPhotoSamples(ds, args.n)
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    official = score_gostanford(load_checkpoint(cfg, args.official, device, map_input=False), loader, cfg, device, False)
    mapmad = score_gostanford(load_checkpoint(cfg, args.checkpoint, device, map_input=True), loader, cfg, device, True)
    diff = bootstrap_ci(mapmad - official, n_boot=N_BOOT)
    res = {"n": len(samples), "official": float(official.mean()), "mapmad": float(mapmad.mean()),
           "diff": diff[0], "diff_ci_sample": diff[1:], "threshold": 0.10,
           "pass": bool(mapmad.mean() <= official.mean() + 0.10), "items_sha256": canonical_sha256(samples.items),
           "checkpoint": args.checkpoint, "official_weights": args.official, "git_commit": git_commit()}
    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "gostanford_scores.npz"), official=official, mapmad=mapmad)
    with open(os.path.join(args.out, "gostanford.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-list")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default=None, help=f"comma list, default all: {','.join(ARMS)}")
    p.add_argument("--limit", type=int, default=None, help="score only the first N listed samples (checks only)")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("compare")
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("gostanford")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--official", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=N_SAMPLES)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    {"make-list": cmd_make_list, "score": cmd_score, "compare": cmd_compare, "gostanford": cmd_gostanford}[args.cmd](args)


if __name__ == "__main__":
    main()
