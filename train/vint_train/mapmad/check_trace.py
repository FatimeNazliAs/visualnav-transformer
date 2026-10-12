"""The in-training check trace `<run>/no_harm_trace.csv` (Phase 4): one row per check (epoch, no-harm delta, R,
attention share, ...), written by `training_checks.TrainingChecks`, read by `select_checkpoint.py` and
`plot_curves.py`. Stdlib only.
"""

import csv
import os
from typing import Any, Dict, List

TRACE_COLUMNS = ["run", "epoch", "ema_file", "no_harm_delta", "no_harm_ci_low", "no_harm_ci_high", "R", "R_ci_low",
                 "R_ci_high", "attention_share", "minutes", "gs_model", "gs_official", "map_goal", "map_hidden",
                 "map_diff", "map_diff_ci_low", "map_diff_ci_high", "n_gs", "n_map"]


def append_trace(path: str, row: Dict[str, Any]) -> None:
    """Append one row to the trace CSV (header written with the first row)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def read_trace(path: str) -> List[Dict[str, Any]]:
    """Trace rows with numbers parsed (epoch int, the rest float where possible)."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = dict(r)
            for k, v in r.items():
                if k in ("run", "ema_file"):
                    continue
                row[k] = int(v) if k in ("epoch", "n_gs", "n_map") else float(v)
            out.append(row)
    return out
