"""Episode metrics and their confidence intervals.

Per episode: success (geodesic distance to the target point <= success_m at some step; oracle stop), SPL =
success * L / max(L, P) with L = geodesic start -> target point and P = driven path length, collisions (and the
share of steps with one), path length, final geodesic distance; success@N = success within N steps. Per arm: mean with a 95% percentile-bootstrap CI over episodes. Two arms on the same
episodes: paired difference (resample episodes, keep pairs together).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

METRICS = ("success", "spl", "collisions", "collision_share", "path_length_m", "final_geodesic_m")
N_BOOT = 10000
BOOT_SEED = 0


def spl(success: bool, shortest_m: float, path_m: float) -> float:
    """Success weighted by (normalised inverse) path length (Anderson et al. 2018)."""
    if not success:
        return 0.0
    return shortest_m / max(shortest_m, path_m) if shortest_m > 0 else 1.0


def success_at(row: Dict[str, float], steps: int) -> float:
    """1.0 if the episode succeeded within `steps` steps (read from the same run)."""
    return float(row["success_step"] is not None and row["success_step"] <= steps)


def bootstrap_ci(values: Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                 level: float = 0.95) -> Tuple[float, float, float]:
    """(mean, low, high): percentile bootstrap of the mean."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_diff_ci(a: Sequence[float], b: Sequence[float], **kwargs) -> Tuple[float, float, float]:
    """Mean of a - b over the same episodes with its bootstrap CI."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arms need the same episodes")
    return bootstrap_ci(a - b, **kwargs)


def arm_table(rows: List[Dict[str, float]]) -> Dict[str, Tuple[float, float, float]]:
    """metric -> (mean, low, high) over one arm's episode summaries."""
    return {m: bootstrap_ci([float(r[m]) for r in rows]) for m in METRICS}


def paired(rows_a: List[Dict[str, float]], rows_b: List[Dict[str, float]],
           metrics: Sequence[str] = ("success", "spl")) -> Optional[Dict[str, Tuple[float, float, float]]]:
    """a - b per metric over the episodes both arms ran (matched by episode_id); None if there are none."""
    b_by_id = {r["episode_id"]: r for r in rows_b}
    common = [r for r in rows_a if r["episode_id"] in b_by_id]
    if not common:
        return None
    return {m: paired_diff_ci([float(r[m]) for r in common], [float(b_by_id[r["episode_id"]][m]) for r in common])
            for m in metrics}
