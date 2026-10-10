"""Phase 4 checkpoint selection (confirmation item 7; the rule is frozen in /outputs/mapmad/p4_train/p4_rules.md).

A checkpoint qualifies when its subset no-harm point estimate (`no_harm_delta` in the run's no_harm_trace.csv,
unrounded) is <= +0.08. First match wins:
  1. the latest qualifying run-A checkpoint from epochs 12-20 (ema_11, 13, 15, 17, 19);
  2. else the latest qualifying run-B checkpoint from epochs 12-20;
  3. else STOP (decide with Naz; nothing is re-judged).
The map effect, R and attention share are never read here.

    python -m vint_train.mapmad.select_checkpoint --run-a <A folder> --run-b <B folder> \
        --rules /outputs/mapmad/p4_train/p4_rules.md --out /outputs/mapmad/p4_train/selection.json
"""

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from vint_train.mapmad.training_checks import read_trace

NO_HARM_LIMIT = 0.08
EPOCH_FLOOR, EPOCH_CEIL = 12, 20


def qualifying(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows of epochs 12-20 with no_harm_delta <= 0.08 (the last row per epoch if a check was repeated)."""
    last = {r["epoch"]: r for r in rows}
    return [r for e, r in sorted(last.items())
            if EPOCH_FLOOR <= e <= EPOCH_CEIL and r["no_harm_delta"] <= NO_HARM_LIMIT]


def select(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """{"line": 1 | 2 | 3, "run", "epoch", "ema_file"}; line 3 = nothing qualifies (run/epoch None)."""
    for line, run, rows in ((1, "A", rows_a), (2, "B", rows_b)):
        q = qualifying(rows)
        if q:
            return {"line": line, "run": run, "epoch": q[-1]["epoch"], "ema_file": q[-1]["ema_file"],
                    "no_harm_delta": q[-1]["no_harm_delta"]}
    return {"line": 3, "run": None, "epoch": None, "ema_file": None, "no_harm_delta": None}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--rules", required=True, help="p4_rules.md (its sha256 is recorded)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    folders = {"A": args.run_a, "B": args.run_b}
    rows = {k: read_trace(os.path.join(v, "no_harm_trace.csv")) for k, v in folders.items()}
    res = select(rows["A"], rows["B"])
    res.update({"rules": args.rules, "rules_sha256": sha256_file(args.rules), "no_harm_limit": NO_HARM_LIMIT,
                "epochs": [EPOCH_FLOOR, EPOCH_CEIL],
                "candidates": {k: [(r["epoch"], r["no_harm_delta"]) for r in v] for k, v in rows.items()}})
    if res["run"] is not None:
        res["path"] = os.path.join(folders[res["run"]], res["ema_file"])
        res["sha256"] = sha256_file(res["path"])
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; the selection is made once")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    if res["line"] == 3:
        raise SystemExit("rule line 3: no checkpoint qualifies -> STOP and decide with Naz")


if __name__ == "__main__":
    main()
