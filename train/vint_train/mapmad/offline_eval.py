"""Offline check of the map effect (gate G3 rows 4-5; Phase 3 confirmation items 12, 14, 19). Style of
`train/ablation/eval_paired.py`: fixed sample lists, fixed noise, deterministic kernels, per-sample scores.

Commands (inside naz_mapmad, from /app/visualnav-transformer/train):

1. `make-list` -- the frozen list of 5,000 Habitat val-drive test samples (collision drives excluded), spread
   evenly over the drives (7-8 each), frames t with t - 15 >= 0 and t + 40 <= last frame (so the same list works
   for waypoint spacing 1, 2 and 5). Written as JSON with its sha256.
2. `score` -- one checkpoint, every arm on every listed sample; `gc_action_loss` = MSE between the expert's 8
   waypoints and the waypoints from the full 10-step reverse diffusion (normalised units, NoMaD's
   `_compute_losses_nomad` formula per sample). Same starting noise per sample in every arm (keyed on the sample
   id) and the same scheduler noise per batch (keyed on the batch number; the batch size is in the manifest).
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

import argparse
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset

from vint_train.mapmad import modes as M
from vint_train.mapmad.closed_loop.metrics import bootstrap_ci
from vint_train.mapmad.config import git_commit, load_config
from vint_train.mapmad.dataset import build_mapmad_dataset, read_drive_list
from vint_train.mapmad.reference_forward import IMAGENET, enforce_determinism, make_noise_scheduler
from vint_train.mapmad.train_loop import to_batch
from vint_train.mapmad.weights import load_checkpoint

N_SAMPLES = 5000
LIST_SEED = 20261010
NOISE_SEED = 0
BACK_FRAMES, AHEAD_FRAMES = 15, 40  # 3 context frames x spacing 5 back; 8 waypoints x spacing 5 ahead
N_BOOT = 10_000
PHOTO_K_MIN, PHOTO_K_MAX = 4, 19  # strictly inside NoMaD's action range (3, 20)
ARMS = ("map_goal", "map_hidden", "wrong_heat", "photo", "photo_map", "explore_map")
ARM_MODE = {"map_goal": "map_goal", "map_hidden": "explore", "wrong_heat": "map_goal", "photo": "photo",
            "photo_map": "photo_map", "explore_map": "explore_map"}


# --- the frozen list ---------------------------------------------------------------------------------------------
def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_list(data_folder: str, drives: Sequence[str], n: int = N_SAMPLES, seed: int = LIST_SEED) -> List[Dict[str, Any]]:
    """n samples spread evenly over `drives` (n // D or n // D + 1 each, the +1 drives drawn with the seed)."""
    import pickle

    rng = np.random.default_rng(seed)
    drives = sorted(drives)
    per = np.full(len(drives), n // len(drives))
    per[rng.choice(len(drives), n - per.sum(), replace=False)] += 1
    out = []
    for drive, k in zip(drives, per):
        with open(os.path.join(data_folder, drive, "traj_data.pkl"), "rb") as f:
            length = len(pickle.load(f)["position"])
        with open(os.path.join(data_folder, drive, "mapmad_meta.json")) as f:
            kind = json.load(f)["target"]["kind"]
        valid = np.arange(BACK_FRAMES, length - 1 - AHEAD_FRAMES + 1)
        if len(valid) < k:
            raise ValueError(f"{drive}: only {len(valid)} valid frames for {k} samples")
        for t in np.sort(rng.choice(valid, k, replace=False)):
            out.append({"drive": drive, "t": int(t), "type": kind})
    for i, s in enumerate(out):
        s["id"] = i
    return out


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


def load_list(path: str) -> Dict[str, Any]:
    with open(path) as f:
        body = json.load(f)
    if canonical_sha256(body["samples"]) != body["sha256"]:
        raise SystemExit(f"{path}: sha256 mismatch, the list was changed")
    return body


# --- models ------------------------------------------------------------------------------------------------------
def attention_share(encoder, tokens: torch.Tensor, padding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(encoder output, per-sample map attention share): the TransformerEncoder (norm_first, eval) re-run layer by
    layer to read the attention weights; share = mean over layers, heads and visible non-map queries of the
    weight on the last (map) token."""
    x, shares = tokens, []
    for layer in encoder.layers:
        h = layer.norm1(x)
        a, w = layer.self_attn(h, h, h, key_padding_mask=padding, need_weights=True, average_attn_weights=False)
        x = x + layer.dropout1(a)
        x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
        shares.append(w[..., -1])  # (B, heads, queries)
    if encoder.norm is not None:
        x = encoder.norm(x)
    q = (~padding).float()
    q[:, -1] = 0.0  # the map token's own query does not count
    per_layer_head = torch.stack(shares, 1)  # (B, layers, heads, queries)
    share = (per_layer_head * q[:, None, None, :]).sum(-1) / q.sum(-1)[:, None, None]
    return x, share.mean(dim=(1, 2))


@torch.no_grad()
def sample_actions(model, cond: torch.Tensor, init_noise: torch.Tensor, scheduler, batch_seed: int) -> torch.Tensor:
    """Full reverse diffusion -> waypoints (B, 8, 2) in normalised units (get_action)."""
    from vint_train.training.train_utils import get_action

    device = cond.device
    gen = torch.Generator(device=device).manual_seed(batch_seed)
    naction = init_noise.to(device)
    for k in scheduler.timesteps[:]:
        pred = model("noise_pred_net", sample=naction, timestep=k.unsqueeze(-1).repeat(naction.shape[0]).to(device),
                     global_cond=cond)
        naction = scheduler.step(model_output=pred, timestep=k, sample=naction, generator=gen).prev_sample
    return get_action(naction)


def init_noise(ids: Sequence[int], horizon: int) -> torch.Tensor:
    """One fixed starting-noise tensor per sample id (independent of batching and arm)."""
    return torch.stack([torch.randn((horizon, 2), generator=torch.Generator().manual_seed(NOISE_SEED + int(i)))
                        for i in ids])


def per_sample_loss(actions: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """gc_action_loss and waypoint cosine similarity per sample (the _compute_losses_nomad formulas)."""
    loss = F.mse_loss(actions, labels, reduction="none").mean(dim=(1, 2))
    cos = F.cosine_similarity(actions[:, :, :2], labels[:, :, :2], dim=-1).mean(dim=-1)
    return loss, cos


# --- Habitat arms ------------------------------------------------------------------------------------------------
class ArmSamples(Dataset):
    """The listed samples built in one arm (deterministic: photo goal offset from the sample id)."""

    def __init__(self, ds, samples: List[Dict[str, Any]], arm: str) -> None:
        self.ds, self.samples, self.arm = ds, samples, arm

    def __len__(self) -> int:
        return len(self.samples)

    def photo_offset(self, s: Dict[str, Any]) -> int:
        k_max = min(PHOTO_K_MAX, self.ds.max_goal_offset(s["drive"], s["t"]))
        return int(np.random.default_rng([LIST_SEED, s["id"]]).integers(PHOTO_K_MIN, k_max + 1))

    def __getitem__(self, j: int):
        s = self.samples[j]
        mode = M.BY_NAME[ARM_MODE[self.arm]]
        out = self.ds.habitat_sample(s["drive"], s["t"], mode, np.random.default_rng([LIST_SEED, s["id"]]),
                                     goal_offset=self.photo_offset(s) if mode.goal_mask == 0 else None,
                                     wrong_heat=self.arm == "wrong_heat", perturbation=None)
        return out + (torch.tensor(s["id"]),)


@torch.no_grad()
def score_arm(model, ds, samples, arm: str, cfg, device, batch_size: int, workers: int) -> Dict[str, np.ndarray]:
    scheduler = make_noise_scheduler(cfg)
    scheduler.set_timesteps(cfg["num_diffusion_iters"])
    loader = DataLoader(ArmSamples(ds, samples, arm), batch_size=batch_size, shuffle=False, num_workers=workers)
    enc = model.vision_encoder
    res = {"gc_action_loss": [], "cos_sim": [], "attn_share": [], "action_mask": []}
    for bi, data in enumerate(tqdm.tqdm(loader, desc=arm, dynamic_ncols=True, leave=False)):
        b, ids = to_batch(data[:-1], IMAGENET, device), data[-1]  # the training input recipe
        obs, goal, labels, maps, gm, mm = (b["obs"], b["goal"], b["actions"].to(device), b["map"], b["goal_mask"],
                                           b["map_mask"])
        am = b["action_mask"].cpu()
        tokens, padding = enc.map_tokens(enc._encode_images(obs, goal), gm, maps, mm)
        out_manual, share = attention_share(enc.sa_encoder, tokens, padding)
        cond = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps, input_map_mask=mm)
        if bi == 0:  # the manual layer pass must reproduce nn.TransformerEncoder (up to float32 kernel noise:
            # on the GPU the encoder runs PyTorch's fused fast path, about 1e-4 of the output scale apart)
            ref = enc.sa_encoder(tokens, src_key_padding_mask=padding)
            visible = ~padding
            rel = float((out_manual - ref)[visible].abs().max() / ref[visible].abs().max())
            assert rel < 1e-3, f"manual attention pass differs from the encoder: relative {rel:.2e}"
        actions = sample_actions(model, cond, init_noise(ids.tolist(), cfg["len_traj_pred"]), scheduler, NOISE_SEED + bi)
        loss, cos = per_sample_loss(actions, labels)
        res["gc_action_loss"].append(loss.cpu())
        res["cos_sim"].append(cos.cpu())
        res["attn_share"].append(torch.where(mm.bool(), torch.full_like(share, float("nan")), share).cpu())
        res["action_mask"].append(am.float())
    return {k: torch.cat(v).numpy().astype(np.float64) for k, v in res.items()}


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


# --- statistics --------------------------------------------------------------------------------------------------
def drive_bootstrap(values: Dict[str, np.ndarray], groups: np.ndarray, stat, n_boot: int = N_BOOT,
                    seed: int = 0) -> Tuple[float, float, float]:
    """(estimate, low, high): percentile bootstrap resampling whole drives; stat(sums dict, count) -> float,
    computed from per-drive sums so that a drive drawn twice counts twice."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = {k: np.bincount(inv, weights=v, minlength=len(uniq)) for k, v in values.items()}
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot_sums = {k: s[draws].sum(axis=1) for k, s in sums.items()}
    boot = stat(boot_sums, counts[draws].sum(axis=1))
    est = stat({k: s.sum() for k, s in sums.items()}, counts.sum())
    return float(est), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_stats(a: np.ndarray, b: np.ndarray, drives: np.ndarray) -> Dict[str, Any]:
    """mean(a - b) with drive-level and per-sample CIs, and R = (b - a) / b with its drive-level CI."""
    diff = drive_bootstrap({"d": a - b}, drives, lambda s, n: s["d"] / n)
    rel = drive_bootstrap({"a": a, "b": b}, drives, lambda s, n: (s["b"] - s["a"]) / s["b"])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff": diff[0], "diff_ci_drive": diff[1:],
            "diff_ci_sample": bootstrap_ci(a - b, n_boot=N_BOOT)[1:], "R": rel[0], "R_ci_drive": rel[1:],
            "n_samples": int(len(a)), "n_drives": int(len(np.unique(drives)))}


def compare(scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    drives, types = scores["drive"], scores["type"]
    loss = {arm: scores[f"{arm}.gc_action_loss"] for arm in ARMS if f"{arm}.gc_action_loss" in scores}
    out: Dict[str, Any] = {"means": {k: float(v.mean()) for k, v in loss.items()}}
    a, b = loss["map_goal"], loss["map_hidden"]
    out["row4"] = paired_stats(a, b, drives)
    out["row4"]["pass"] = bool(out["row4"]["diff"] < 0 and out["row4"]["diff_ci_drive"][1] < 0)
    if "wrong_heat" in loss:
        out["wrong_minus_map"] = paired_stats(loss["wrong_heat"], a, drives)
    out["per_type"] = {}
    for t in np.unique(types):
        sel = types == t
        out["per_type"][str(t)] = {"map_vs_hidden": paired_stats(a[sel], b[sel], drives[sel])}
        if "wrong_heat" in loss:
            out["per_type"][str(t)]["wrong_minus_map"] = paired_stats(loss["wrong_heat"][sel], a[sel], drives[sel])
    out["attention_share"] = {arm: float(np.nanmean(scores[f"{arm}.attn_share"])) for arm in loss
                              if not np.all(np.isnan(scores[f"{arm}.attn_share"]))}
    return out


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


# --- GoStanford no-harm (row 5) ----------------------------------------------------------------------------------
class GoStanfordPhotoSamples(Dataset):
    """Fixed GoStanford test samples with a positive photo goal k waypoints ahead (k from the sample's seed)."""

    def __init__(self, ds, n: int, seed: int = LIST_SEED) -> None:
        self.ds = ds
        rng = np.random.default_rng([seed, 1])
        idx = np.arange(len(ds)) if n >= len(ds) else np.sort(rng.choice(len(ds), n, replace=False))
        self.items = []
        for j, i in enumerate(idx):
            drive, t, max_goal = ds.index_to_data[int(i)]
            k_max = min(PHOTO_K_MAX, int(max_goal) // ds.waypoint_spacing)
            if k_max < PHOTO_K_MIN:
                continue
            k = int(np.random.default_rng([seed, 2, j]).integers(PHOTO_K_MIN, k_max + 1))
            self.items.append((drive, int(t), k))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, j: int):
        drive, t, k = self.items[j]
        ds = self.ds
        obs = torch.cat([ds._load_image(drive, ct) for ct in ds._context_times(t)])
        goal = ds._load_image(drive, t + k * ds.waypoint_spacing)
        actions, _ = ds._compute_actions(ds._get_trajectory(drive), t, t + k * ds.waypoint_spacing)
        return obs, goal, torch.as_tensor(actions, dtype=torch.float32), torch.tensor(j)


@torch.no_grad()
def score_gostanford(model, loader, cfg, device, map_input: bool) -> np.ndarray:
    scheduler = make_noise_scheduler(cfg)
    scheduler.set_timesteps(cfg["num_diffusion_iters"])
    losses = []
    for bi, (obs_image, goal_image, labels, ids) in enumerate(tqdm.tqdm(loader, dynamic_ncols=True, leave=False)):
        obs = torch.cat([IMAGENET(o) for o in torch.split(obs_image, 3, dim=1)], dim=1).to(device)
        goal = IMAGENET(goal_image).to(device)
        b = obs.shape[0]
        kwargs = dict(obs_img=obs, goal_img=goal, input_goal_mask=torch.zeros(b, dtype=torch.long, device=device))
        if map_input:
            kwargs.update(map_img=torch.zeros((b, 3, 64, 64), device=device),
                          input_map_mask=torch.ones(b, dtype=torch.long, device=device))
        cond = model("vision_encoder", **kwargs)
        actions = sample_actions(model, cond, init_noise(ids.tolist(), cfg["len_traj_pred"]), scheduler, NOISE_SEED + bi)
        losses.append(per_sample_loss(actions, labels.to(device))[0].cpu())
    return torch.cat(losses).numpy().astype(np.float64)


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
