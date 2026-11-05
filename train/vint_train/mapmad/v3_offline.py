"""Phase 14b-3 offline reports for MapMaD v3 (kickoff §3 item 7 "also report", fallback "reverse waypoints unstable";
p14b3_rules.md §4):

    python -m vint_train.mapmad.v3_offline recovery --config config/mapmad_v3.yaml \
        --checkpoint v21b=<ema_9 of v2.1b> v3=<ema_9 of v3> --out <json>
    python -m vint_train.mapmad.v3_offline past --run <v3 run folder> --out <json>

recovery: the recovery-label loss = offline_scoring.score_arm's action loss (sampled actions vs the label, fixed noise
per state) on n seeded W1 recovery states of round 2 (labels.npz `recovery`; maps-shown states in arm "maps",
maps-hidden states in arm "hidden"; no past token: the same input for both checkpoints) for each checkpoint; the
difference v3 - v2.1b with a drive-bootstrap 95 % CI. "Falls" (pre-set): mean lower AND CI upper bound < 0; else the
fallback "W1-turn" (p14b3_rules.md §4). Round 2 has no held-out split (no rollouts in the dev / photo-test homes), so
these are training states: the check asks whether the network learned the recovery answer at all.
past: the past-move prediction error (MSE of the auxiliary head, normalised units) of the EMA model per epoch on the
in-loop test sets (metrics.jsonl `eval_mode_diffusion_loss.<set>.past_mse`), first and last epoch.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from vint_train.mapmad.dataset_dagger import build_dagger_dataset, recovery_flags


def recovery_samples(ds, n: int, seed: int) -> List[Dict[str, Any]]:
    """n seeded recovery states (labelled, full context) of a DAgger dataset (ids unique within it)."""
    flags: Dict[str, np.ndarray] = {}
    idx = []
    for i, (d, t, _) in enumerate(ds.index_to_data):
        if d not in flags:
            flags[d] = recovery_flags(ds.data_folder, d)
        if flags[d][t]:
            idx.append(i)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(idx), min(n, len(idx)), replace=False)) if idx else []
    return [{"drive": ds.index_to_data[idx[k]][0], "t": int(ds.index_to_data[idx[k]][1]), "id": int(idx[k])}
            for k in pick], len(idx)


def drive_ci(d: np.ndarray, drives: List[str], seed: int, draws: int = 10000) -> List[float]:
    names = sorted(set(drives))
    arr = np.asarray(drives)
    rows = {k: np.flatnonzero(arr == k) for k in names}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        sel = np.concatenate([rows[names[j]] for j in rng.integers(0, len(names), len(names))])
        boot.append(d[sel].mean())
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def cmd_recovery(args) -> Dict[str, Any]:
    from vint_train.mapmad.config import load_config
    from vint_train.mapmad.offline_scoring import score_arm
    from vint_train.mapmad.weights import load_checkpoint

    cfg = load_config(args.config, "config/defaults.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = dict(c.split("=", 1) for c in args.checkpoint)
    sets = []
    for hidden, key, arm in ((False, "dagger_more", "maps"), (True, "dagger_hidden_more", "hidden")):
        block = (cfg["map"].get(key) or [None])[0]
        if block is None:
            continue
        ds = build_dagger_dataset(cfg, "train", train=False, hidden=hidden, block=block)
        samples, total = recovery_samples(ds, args.n, args.seed + int(hidden))
        sets.append((arm, ds, samples, total))
    out: Dict[str, Any] = {"n_per_set": args.n, "seed": args.seed, "sets": {}}
    losses: Dict[str, List[np.ndarray]] = {k: [] for k in ckpts}
    drives: List[str] = []
    for arm, ds, samples, total in sets:
        out["sets"][arm] = {"recovery_states": total, "scored": len(samples)}
        drives += [f"{arm}/{s['drive']}" for s in samples]
    for label, path in ckpts.items():
        cfg_k = load_config(args.configs[label], "config/defaults.yaml") if args.configs and label in args.configs else cfg
        model = load_checkpoint(cfg_k, path, device, map_input=True)
        for arm, ds, samples, total in sets:
            if samples:
                losses[label].append(score_arm(model, ds, samples, arm, cfg, device, 100, 4)["gc_action_loss"])
        del model
    means = {k: float(np.concatenate(v).mean()) if v else None for k, v in losses.items()}
    out["means"] = means
    if len(ckpts) >= 2 and drives:
        a, b = list(ckpts)[:2]
        d = np.concatenate(losses[b]) - np.concatenate(losses[a])
        ci = drive_ci(d, drives, args.seed)
        out["difference"] = {"arms": f"{b} - {a}", "mean": float(d.mean()), "ci95_drive_bootstrap": ci}
        out["falls"] = bool(d.mean() < 0 and ci[1] < 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


def cmd_past(args) -> Dict[str, Any]:
    rows = [json.loads(x) for x in open(os.path.join(args.run, "metrics.jsonl")) if x.strip()]
    per_epoch = []
    for r in rows:
        if "eval_mode_diffusion_loss" in r:
            per_epoch.append({"epoch": r.get("epoch"), **{k: v.get("past_mse") for k, v in r["eval_mode_diffusion_loss"].items()},
                              "train_past_loss": r.get("past_loss")})
    out = {"run": args.run, "per_epoch": per_epoch, "first": per_epoch[0] if per_epoch else None,
           "last": per_epoch[-1] if per_epoch else None,
           "note": "MSE of the auxiliary head on (dx/0.25 m, dy/0.25 m, dtheta rad) per move, EMA model, in-loop test sets"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("recovery")
    q.add_argument("--config", required=True)
    q.add_argument("--checkpoint", nargs="+", required=True, help="label=path; the first two are compared (2nd - 1st)")
    q.add_argument("--configs", nargs="*", default=None, help="label=train config of that checkpoint (default --config)")
    q.add_argument("--n", type=int, default=1000)
    q.add_argument("--seed", type=int, default=14302)
    q.add_argument("--out", required=True)
    q = sub.add_parser("past")
    q.add_argument("--run", required=True)
    q.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.cmd == "recovery" and args.configs:
        args.configs = dict(c.split("=", 1) for c in args.configs)
    {"recovery": cmd_recovery, "past": cmd_past}[args.cmd](args)


if __name__ == "__main__":
    main()
