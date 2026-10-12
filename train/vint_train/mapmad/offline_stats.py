"""Statistics of the offline harness: drive-level paired bootstrap (resampling whole drives), R = (hidden - map) /
hidden, and the `compare` summary of one `score` output (gate row: map goal vs map hidden).
"""

from typing import Any, Dict, Tuple

import numpy as np

from vint_train.mapmad.closed_loop.metrics import bootstrap_ci
from vint_train.mapmad.offline_scoring import ARMS

N_BOOT = 10_000


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
