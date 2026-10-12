"""Phase 5 statistics (block item 11), numpy only.

- Per episode, a metric is first averaged over its seeds (3; the planner has one run that counts for all).
- Paired difference a - b: per episode (same episodes in both arms), then averaged over episodes.
- Primary test = HOME-CLUSTERED bootstrap: resample the homes with replacement (n_boot = 10,000, fixed seed); a
  resample's statistic is the mean over all episodes of the drawn homes (a home drawn twice counts twice). One-sided
  p = share of resamples whose mean difference is <= 0. CI = 2.5% / 97.5% percentiles.
- Episode-level bootstrap (resample episodes) reported next to it.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

N_BOOT = 10_000


def cluster_bootstrap(values: np.ndarray, clusters: Sequence[str], n_boot: int = N_BOOT,
                      seed: int = 0) -> Tuple[float, float, float, float]:
    """(mean, CI low, CI high, one-sided p of mean <= 0) resampling whole clusters (homes)."""
    values = np.asarray(values, dtype=np.float64)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)),
            float(np.mean(boot <= 0.0)))


def episode_bootstrap(values: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> Tuple[float, float, float, float]:
    """The same statistic resampling single episodes (each its own cluster)."""
    values = np.asarray(values, dtype=np.float64)
    return cluster_bootstrap(values, np.arange(len(values)).astype(str), n_boot, seed)


def paired(a: Dict[str, float], b: Dict[str, float], home_of: Dict[str, str], n_boot: int = N_BOOT,
           seed: int = 0) -> Dict[str, float]:
    """a - b over the episodes both have: per-episode values (seed means) keyed by episode id."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {}
    d = np.array([a[i] - b[i] for i in ids])
    h = cluster_bootstrap(d, [home_of[i] for i in ids], n_boot, seed)
    e = episode_bootstrap(d, n_boot, seed)
    return {"n": len(ids), "homes": len({home_of[i] for i in ids}), "diff": h[0], "home_ci": [h[1], h[2]],
            "home_p_one_sided": h[3], "episode_ci": [e[1], e[2]], "episode_p_one_sided": e[3]}
