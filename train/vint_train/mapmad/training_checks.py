"""Phase 4 in-training checks (Phase 4 confirmation items 5 and 7). Same scoring code as the Phase 3 offline
harness (`offline_eval.py`: full 10-step reverse diffusion, all noise fixed per sample id -- since Phase 5 also the
scheduler noise, so a subset item scores exactly as in the full harness -- deterministic kernels), on two frozen
1,000-sample subsets:

- GoStanford no-harm: 1,000 of the 5,000 GoStanford items of the full harness (`GoStanfordPhotoSamples`, the
  list Phase 3 scored; positive photo goal 4-19 waypoints ahead), drawn with seed 4004. Delta = mean(MapMaD map
  hidden, photo shown - official nomad.pth) of `gc_action_loss`, trajectory-level bootstrap CI. The official
  scores are computed once per run, at its start.
- Map effect: ~1,000 of the frozen Phase 3 offline list (5,000 Habitat val-drive samples, sha256 eca78af3...),
  whole drives in a seeded order (seed 4004) until >= 1,000 samples (1,000-1,007; drives hold 7-8).
  Arms map_goal vs map_hidden, R = (hidden - map) / hidden with the drive-level CI; attention share = mean map-token share in map_goal.
- Step-0 sanity (before training): MapMaD with nomad.pth and the map hidden must give the official per-sample
  losses (|Delta| < 0.0005 and every sample within 1e-5); otherwise the run stops (a bug).

Each check appends one row to `<run>/no_harm_trace.csv` and saves its per-sample scores in `<run>/checks/`.

Commands (inside naz_mapmad, from /app/visualnav-transformer/train):

    python -m vint_train.mapmad.training_checks make-subsets --config config/mapmad_p4.yaml \
        --out-dir /outputs/mapmad/p4_train/subsets
    python -m vint_train.mapmad.training_checks gostanford-report \
        --scores <final_eval>/gostanford_scores.npz --subset /outputs/mapmad/p4_train/subsets/gostanford_subset.json \
        --out <final_eval>/gostanford_report.json

`gostanford-report` (selection-bias report, item 7): the full-harness no-harm with per-sample and trajectory-level
CIs, and the same on the 4,000 items outside the selection subset (flagged when > +0.10).
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import copy
import json
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from vint_train.mapmad.closed_loop.metrics import bootstrap_ci
from vint_train.mapmad.config import git_commit, load_config
from vint_train.mapmad.dataset import build_mapmad_dataset
from vint_train.mapmad.check_trace import append_trace, read_trace
from vint_train.mapmad.offline_list import N_SAMPLES, canonical_sha256, load_list
from vint_train.mapmad.offline_scoring import GoStanfordPhotoSamples, score_arm, score_gostanford
from vint_train.mapmad.offline_stats import N_BOOT, drive_bootstrap, paired_stats
from vint_train.mapmad.reference_forward import enforce_determinism, test_dataset
from vint_train.mapmad.weights import load_checkpoint

SUBSET_SEED = 4004
SUBSET_N = 1000
SANITY_MEAN_TOL = 5e-4      # |Delta| prints as 0.000
SANITY_SAMPLE_TOL = 1e-5    # per-sample |MapMaD map hidden - official|
BIAS_FLAG = 0.10


# --- the frozen subsets ------------------------------------------------------------------------------------------
def draw_gostanford_subset(n_items: int, n: int = SUBSET_N, seed: int = SUBSET_SEED) -> List[int]:
    """n of the harness item indices 0..n_items-1, without replacement, sorted."""
    rng = np.random.default_rng([seed, 1])
    return sorted(int(i) for i in rng.choice(n_items, n, replace=False))


def draw_offline_subset(samples: Sequence[Dict[str, Any]], n: int = SUBSET_N, seed: int = SUBSET_SEED) -> List[int]:
    """Sample ids of whole drives: the shortest prefix of a seeded random drive order with >= n samples
    (drives hold 7-8 samples, so 1,000-1,007)."""
    by_drive: Dict[str, List[int]] = {}
    for s in samples:
        by_drive.setdefault(s["drive"], []).append(int(s["id"]))
    drives = sorted(by_drive)
    rng = np.random.default_rng([seed, 2])
    chosen: List[int] = []
    for i in rng.permutation(len(drives)):
        chosen.extend(by_drive[drives[i]])
        if len(chosen) >= n:
            return sorted(chosen)
    raise ValueError(f"only {len(chosen)} samples in all drives, fewer than {n}")


def subset_body_sha(body: Dict[str, Any]) -> str:
    """sha256 over everything in a subset file except the sha256 field itself."""
    return canonical_sha256({k: v for k, v in body.items() if k != "sha256"})


def load_subset(path: str) -> Dict[str, Any]:
    with open(path) as f:
        body = json.load(f)
    if subset_body_sha(body) != body["sha256"]:
        raise SystemExit(f"{path}: sha256 mismatch, the subset was changed")
    return body


def write_frozen(path: str, body: Dict[str, Any]) -> str:
    """Write a subset file once (refuses to overwrite); returns its sha256."""
    body["sha256"] = subset_body_sha(body)
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; the subsets are frozen")
    with open(path, "w") as f:
        json.dump(body, f)
    return body["sha256"]


def cmd_make_subsets(args) -> None:
    cfg = load_config(args.config, "config/defaults.yaml")
    c = cfg["training_checks"]
    os.makedirs(args.out_dir, exist_ok=True)
    harness = GoStanfordPhotoSamples(test_dataset("go_stanford", cfg, checks_config(cfg)), N_SAMPLES)
    gs = {"kind": "gostanford_no_harm", "seed": SUBSET_SEED, "n": SUBSET_N, "harness_n": N_SAMPLES,
          "harness_items_sha256": canonical_sha256(harness.items), "harness_items": harness.items,
          "ids": draw_gostanford_subset(len(harness.items)),
          "test_split": c["gostanford_test"], "rule": "seeded choice without replacement from the harness items"}
    source = load_list(c["offline_list"])
    off = {"kind": "offline_map_effect", "seed": SUBSET_SEED, "n": SUBSET_N, "source_list": c["offline_list"],
           "source_list_sha256": source["sha256"], "ids": draw_offline_subset(source["samples"]),
           "rule": "shortest prefix of whole drives in a seeded order with >= n samples"}
    off["n_drives"] = len({s["drive"] for s in source["samples"] if s["id"] in set(off["ids"])})
    sha_gs = write_frozen(os.path.join(args.out_dir, "gostanford_subset.json"), gs)
    sha_off = write_frozen(os.path.join(args.out_dir, "offline_subset.json"), off)
    print(f"gostanford_subset.json: {len(gs['ids'])} of {len(harness.items)} harness items "
          f"(harness sha256 {gs['harness_items_sha256']}); sha256 {sha_gs}")
    print(f"offline_subset.json: {len(off['ids'])} samples, {off['n_drives']} drives; sha256 {sha_off}")


def cmd_make_offline_subset(args) -> None:
    """Phase 10: only the map-effect subset (whole drives, >= 1,000 samples) of a frozen list; the GoStanford subset
    of Phase 4 is reused unchanged (same GoStanford test split, harness items checked by sha256 at load)."""
    source = load_list(args.list)
    off = {"kind": "offline_map_effect", "seed": SUBSET_SEED, "n": SUBSET_N, "source_list": args.list,
           "source_list_sha256": source["sha256"], "ids": draw_offline_subset(source["samples"]),
           "rule": "shortest prefix of whole drives in a seeded order with >= n samples"}
    off["n_drives"] = len({s["drive"] for s in source["samples"] if s["id"] in set(off["ids"])})
    sha = write_frozen(args.out, off)
    print(f"{args.out}: {len(off['ids'])} samples, {off['n_drives']} drives; sha256 {sha}")


# --- the checker -------------------------------------------------------------------------------------------------
def checks_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """The config with the test splits the subsets come from (smoke runs train on tiny splits)."""
    cfg = copy.deepcopy(config)
    c = cfg["training_checks"]
    cfg["datasets"][c.get("dataset", "habitat_mapmad")]["test"] = c["habitat_test"]
    cfg["datasets"]["go_stanford"]["test"] = c["gostanford_test"]
    return cfg


@contextlib.contextmanager
def deterministic_kernels() -> Iterator[None]:
    """Deterministic kernels for one check, then the training settings again."""
    if torch.cuda.is_available() and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching (deterministic cuBLAS)")
    prev = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled())
    enforce_determinism()
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = prev[0], prev[1]
        torch.use_deterministic_algorithms(prev[2])


def ci_trajectory(diff: np.ndarray, groups: np.ndarray) -> List[float]:
    """Mean of diff with the trajectory (drive) level percentile bootstrap CI: [estimate, low, high]."""
    return list(drive_bootstrap({"d": diff}, groups, lambda s, n: s["d"] / n))


class TrainingChecks:
    """No-harm, map effect and attention share of a model on the frozen subsets (`training_checks` config)."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        c = config["training_checks"]
        self.c, self.device, self.run = c, device, config.get("run_label", config["run_name"])
        self.folder = config["project_folder"]
        self.cfg = checks_config(config)
        limit = c.get("limit")
        gs = load_subset(c["gostanford_subset"])
        harness = GoStanfordPhotoSamples(test_dataset("go_stanford", self.cfg, self.cfg), gs["harness_n"])
        if canonical_sha256(harness.items) != gs["harness_items_sha256"]:
            raise RuntimeError("the GoStanford harness items differ from the frozen subset's")
        ids = gs["ids"][:limit]
        self.gs_set = Subset(harness, ids)
        self.gs_drives = np.array([harness.items[j][0] for j in ids])
        off = load_subset(c["offline_subset"])
        source = load_list(c["offline_list"])
        if source["sha256"] != off["source_list_sha256"]:
            raise RuntimeError("the offline list differs from the one the subset was drawn from")
        by_id = {s["id"]: s for s in source["samples"]}
        self.map_samples = [by_id[i] for i in off["ids"]][:limit]
        self.map_drives = np.array([s["drive"] for s in self.map_samples])
        self.hab_ds = build_mapmad_dataset(self.cfg, c.get("dataset", "habitat_mapmad"), "test", train=False)
        self.arms = tuple(c.get("arms", ("map_goal", "map_hidden")))  # (map shown, map hidden); v2: maps, hidden
        self.official: Optional[np.ndarray] = None
        self.subset_sha = {"gostanford": gs["sha256"], "offline": off["sha256"]}

    def due(self, epoch_1based: int) -> bool:
        return epoch_1based % int(self.c["every"]) == 0

    def has_row(self, epoch_1based: int) -> bool:
        """Is there a trace row for this epoch already (a resumed run re-runs a check lost in a crash)?"""
        path = os.path.join(self.folder, "no_harm_trace.csv")
        return os.path.exists(path) and any(r["epoch"] == epoch_1based for r in read_trace(path))

    def _gs_scores(self, model, map_input: bool) -> np.ndarray:
        loader = DataLoader(self.gs_set, batch_size=self.c.get("batch_size", 100), shuffle=False,
                            num_workers=self.c.get("workers", 4))
        return score_gostanford(model, loader, self.cfg, self.device, map_input)

    def reference(self) -> np.ndarray:
        """Official nomad.pth on the GoStanford subset (once per process)."""
        if self.official is None:
            with deterministic_kernels():
                net = load_checkpoint(self.cfg, self.c["official_weights"], self.device, map_input=False)
                self.official = self._gs_scores(net, map_input=False)
            del net
            np.save(os.path.join(self.folder, "checks", "official_gostanford.npy"), self.official)
        return self.official

    def step0_sanity(self, model) -> Dict[str, Any]:
        """Before training: map hidden == official (else RuntimeError). Returns the epoch-0 trace row."""
        os.makedirs(os.path.join(self.folder, "checks"), exist_ok=True)
        official = self.reference()
        with deterministic_kernels():
            mine = self._gs_scores(model, map_input=True)
        delta, worst = float(np.mean(mine - official)), float(np.max(np.abs(mine - official)))
        print(f"[checks] step-0 sanity: no-harm delta {delta:+.6f}, max per-sample |diff| {worst:.2e}", flush=True)
        if abs(delta) >= SANITY_MEAN_TOL or worst > SANITY_SAMPLE_TOL:
            raise RuntimeError(f"step-0 sanity failed: delta {delta:+.6f}, max per-sample diff {worst:.2e} "
                               f"(map hidden must equal the official model) -> stop, this is a bug")
        return self.run_check(model, 0, "init", gs_scores=mine)

    def run_check(self, model, epoch_1based: int, ema_file: str,
                  gs_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """One trace row (appended to no_harm_trace.csv); per-sample scores to checks/epoch_XX.npz."""
        t0 = time.time()
        os.makedirs(os.path.join(self.folder, "checks"), exist_ok=True)
        official = self.reference()
        with deterministic_kernels():
            mine = self._gs_scores(model, map_input=True) if gs_scores is None else gs_scores
            arms = {arm: score_arm(model, self.hab_ds, self.map_samples, arm, self.cfg, self.device,
                                   self.c.get("batch_size", 100), self.c.get("workers", 4))
                    for arm in self.arms}
        shown, hidden = self.arms
        nh = ci_trajectory(mine - official, self.gs_drives)
        a, b = arms[shown]["gc_action_loss"], arms[hidden]["gc_action_loss"]
        me = paired_stats(a, b, self.map_drives)
        share = np.asarray(arms[shown]["attn_share"], dtype=np.float64)
        if share.ndim == 2:  # v2: near + far = the share of all map tokens; per token in the v2 report below
            self._v2_report(arms, epoch_1based)
            share = share.sum(axis=1)
        row = {"run": self.run, "epoch": epoch_1based, "ema_file": ema_file, "no_harm_delta": nh[0],
               "no_harm_ci_low": nh[1], "no_harm_ci_high": nh[2], "R": me["R"], "R_ci_low": me["R_ci_drive"][0],
               "R_ci_high": me["R_ci_drive"][1],
               "attention_share": float(np.nanmean(share)),
               "minutes": (time.time() - t0) / 60.0, "gs_model": float(mine.mean()),
               "gs_official": float(official.mean()), "map_goal": me["mean_a"], "map_hidden": me["mean_b"],
               "map_diff": me["diff"], "map_diff_ci_low": me["diff_ci_drive"][0],
               "map_diff_ci_high": me["diff_ci_drive"][1], "n_gs": int(len(mine)), "n_map": int(len(a))}
        np.savez(os.path.join(self.folder, "checks", f"epoch_{epoch_1based:02d}.npz"), gostanford=mine,
                 official=official, map_goal=a, map_hidden=b, map_goal_attn=arms[shown]["attn_share"])
        append_trace(os.path.join(self.folder, "no_harm_trace.csv"), row)
        print("[checks] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
        return row

    def _v2_report(self, arms: Dict[str, Dict[str, np.ndarray]], epoch_1based: int) -> None:
        """v2 (report only): offline_v2.compare_v2 on the subset -> checks/epoch_XX_v2.json (forward/turn split,
        per-category loss, attention share per map token)."""
        from vint_train.mapmad.offline_v2 import compare_v2

        scores = {"drive": self.map_drives}
        for arm, res in arms.items():
            scores.update({f"{arm}.{k}": v for k, v in res.items()})
        with open(os.path.join(self.folder, "checks", f"epoch_{epoch_1based:02d}_v2.json"), "w") as f:
            json.dump(compare_v2(scores, self.map_samples), f, indent=1)


# --- selection-bias report (final eval) --------------------------------------------------------------------------
def gostanford_report(official: np.ndarray, mapmad: np.ndarray, subset: Dict[str, Any]) -> Dict[str, Any]:
    """Full harness and outside-the-subset no-harm, each with per-sample and trajectory-level CIs."""
    if len(official) != subset["harness_n"]:
        raise ValueError(f"scores hold {len(official)} items, the harness {subset['harness_n']}")
    drives = np.array([it[0] for it in subset["harness_items"]])
    inside = np.zeros(len(official), dtype=bool)
    inside[subset["ids"]] = True

    def block(sel: np.ndarray) -> Dict[str, Any]:
        d = mapmad[sel] - official[sel]
        return {"n": int(sel.sum()), "official": float(official[sel].mean()), "mapmad": float(mapmad[sel].mean()),
                "diff": float(d.mean()), "diff_ci_sample": list(bootstrap_ci(d, n_boot=N_BOOT)[1:]),
                "diff_ci_trajectory": ci_trajectory(d, drives[sel])[1:]}

    out = {"full": block(np.ones(len(official), dtype=bool)), "outside_subset": block(~inside),
           "inside_subset": block(inside), "flag_threshold": BIAS_FLAG, "subset_sha256": subset["sha256"]}
    out["outside_flag"] = bool(out["outside_subset"]["diff"] > BIAS_FLAG)
    return out


def cmd_gostanford_report(args) -> None:
    with np.load(args.scores) as z:
        official, mapmad = z["official"], z["mapmad"]
    res = gostanford_report(official, mapmad, load_subset(args.subset))
    res.update({"scores": args.scores, "git_commit": git_commit()})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("full", "outside_subset", "outside_flag")}, indent=1))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-subsets")
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p = sub.add_parser("gostanford-report")
    p.add_argument("--scores", required=True)
    p.add_argument("--subset", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("make-offline-subset")
    p.add_argument("--list", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    {"make-subsets": cmd_make_subsets, "gostanford-report": cmd_gostanford_report,
     "make-offline-subset": cmd_make_offline_subset}[args.cmd](args)


if __name__ == "__main__":
    main()
