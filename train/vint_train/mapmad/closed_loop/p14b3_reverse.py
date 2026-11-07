"""Phase 14b-3 Part 3 (report only): how MapMaD v3 drives backward.

The pre-set oscillation rule (practice check, forward / back flips > 20 % of reverse steps) measures flipping, not how
much the robot reverses. This report reads the run logs (one JSON line per step, `policy.reverse`, `position`) of one
seed per arm and gives, per arm:
- the share of reverse steps by episode phase (steps 1-40, 41-200, 201-1000);
- the share of reverse steps when the robot "looks stuck" by the W1 rule (< 0.10 m moved in the last 40 steps) and
  when it does not;
- the lengths of reverse stretches (consecutive reverse steps): median, 90th percentile, max.

    python -m vint_train.mapmad.closed_loop.p14b3_reverse --logs /outputs/mapmad/p14b3_v3/runs/examb/logs \
        --arms M3h_P M3_P --out /outputs/mapmad/p14b3_v3/tables/reverse.json
"""

import argparse
import glob
import json
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

PHASES: Tuple[Tuple[str, int], ...] = (("1-40", 40), ("41-200", 200), ("201-1000", 10 ** 9))
STUCK_WINDOW, STUCK_M = 40, 0.10  # the W1 rule's "little movement" part (recovery_labels.py)


def stretches(reverse: Sequence[bool]) -> List[int]:
    """Lengths of the runs of consecutive True values, in order."""
    out, run = [], 0
    for r in reverse:
        if r:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def read_steps(path: str) -> Tuple[np.ndarray, List[bool]]:
    """(xz positions [T, 2], reverse flag per step) of one run log."""
    pos, rev = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") != "step":
                continue
            pos.append(r["position"][::2])
            rev.append(bool((r.get("policy") or {}).get("reverse")))
    return np.asarray(pos, dtype=float).reshape(-1, 2), rev


def arm_report(files: Sequence[str]) -> Dict[str, object]:
    """The module docstring's numbers over the given run logs."""
    phase = {name: [0, 0] for name, _ in PHASES}
    looks = {True: [0, 0], False: [0, 0]}
    lens: List[int] = []
    for path in files:
        pos, rev = read_steps(path)
        for i, r in enumerate(rev):
            name = next(n for n, end in PHASES if i + 1 <= end)
            phase[name][0] += r
            phase[name][1] += 1
            if i >= STUCK_WINDOW:
                st = bool(np.linalg.norm(pos[i] - pos[i - STUCK_WINDOW]) < STUCK_M)
                looks[st][0] += r
                looks[st][1] += 1
        lens += stretches(rev)
    ln = np.asarray(lens or [0])
    share = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    return {"runs": len(files),
            "reverse_share_by_phase": {k: share(*v) for k, v in phase.items()},
            "reverse_share_looks_stuck": share(*looks[True]), "steps_looks_stuck": looks[True][1],
            "reverse_share_not_stuck": share(*looks[False]), "steps_not_stuck": looks[False][1],
            "stretch_median": float(np.median(ln)), "stretch_p90": float(np.percentile(ln, 90)),
            "stretch_max": int(ln.max())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", required=True, help="runs/<set>/logs folder (one subfolder per arm)")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seed-index", type=int, default=0, help="which seed's runs (file suffix __s<k>.jsonl)")
    ap.add_argument("--n", type=int, default=120, help="runs per arm (seeded random pick; 0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out: Dict[str, object] = {"logs": a.logs, "seed_index": a.seed_index, "n": a.n, "pick_seed": 0, "arms": {}}
    for arm in a.arms:
        files = sorted(glob.glob(f"{a.logs}/{arm}/*__s{a.seed_index}.jsonl"))
        random.Random(0).shuffle(files)
        out["arms"][arm] = arm_report(files[:a.n] if a.n else files)
        print(arm, json.dumps(out["arms"][arm]))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
