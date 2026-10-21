#!/usr/bin/env python3
"""Phase 10 timing decision (chain step 3, p10_rules.md §6). Host side, standard library only.

    python3 p10_timing.py <timing run folder> <gpu memory csv> <train log> [--accum A]

Reads the timing run's metrics.jsonl (epoch record: per-batch data wait and compute seconds), the peak GPU memory of
the nvidia-smi samples (csv: time,mib) and the train log (out-of-memory), and prints one JSON line:
{"oom", "peak_mib", "loader_share", "s_per_step", "estimate_h", "decision"} with decision in
"ok" | "memory_fallback" | "more_workers" | "budget_stop".
"""

import argparse
import json
import os

PEAK_LIMIT_MIB = 23.5 * 1024
LOADER_LIMIT = 0.15
BUDGET_STOP_H = 10.5
EPOCHS, STEPS = 20, 782            # optimizer steps per epoch (200,000 / 256)
EVAL_MIN, CHECKS, CHECK_MIN = 0.5, 11, 1.0
OOM_MARKS = ("out of memory", "OutOfMemoryError", "CUDA error: out of memory")


def decide(oom: bool, peak_mib: float, wait_s: float, compute_s: float, accum: int, workers_raised: bool) -> dict:
    per_batch = wait_s + compute_s
    s_per_step = per_batch * accum
    estimate_h = (EPOCHS * (STEPS * s_per_step / 60.0 + EVAL_MIN) + CHECKS * CHECK_MIN) / 60.0
    share = wait_s / per_batch if per_batch > 0 else 0.0
    if oom or peak_mib > PEAK_LIMIT_MIB:
        decision = "memory_fallback"
    elif share > LOADER_LIMIT and not workers_raised:
        decision = "more_workers"
    elif estimate_h > BUDGET_STOP_H:
        decision = "budget_stop"
    else:
        decision = "ok"
    return {"oom": oom, "peak_mib": peak_mib, "loader_share": round(share, 4), "s_per_step": round(s_per_step, 4),
            "estimate_h": round(estimate_h, 2), "decision": decision}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run")
    p.add_argument("mem_csv")
    p.add_argument("log")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--workers-raised", action="store_true")
    a = p.parse_args()
    text = open(a.log, errors="replace").read() if os.path.exists(a.log) else ""
    oom = any(m in text for m in OOM_MARKS)
    peak = 0.0
    if os.path.exists(a.mem_csv):
        for line in open(a.mem_csv):
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[1].strip().isdigit():
                peak = max(peak, float(parts[1]))
    wait_s = compute_s = 0.0
    metrics = os.path.join(a.run, "metrics.jsonl") if a.run else ""
    if metrics and os.path.exists(metrics):
        for line in open(metrics):
            rec = json.loads(line)
            if "timing" in rec:
                wait_s, compute_s = rec["timing"]["data_wait_s"] or 0.0, rec["timing"]["compute_s"] or 0.0
    if not oom and compute_s == 0.0:
        print(json.dumps({"oom": oom, "peak_mib": peak, "decision": "error", "why": "no timing record"}))
        return
    print(json.dumps(decide(oom, peak, wait_s, compute_s, a.accum, a.workers_raised)))


if __name__ == "__main__":
    main()
