#!/usr/bin/env python3
"""Phase 11 STATUS.md writer and run counter (kickoff §2). Host side, standard library only.

    python3 p11_status.py <p11 dir> chain --set step="1 step-0 dev check" gpus="0 1" [--state RUNNING]
    python3 p11_status.py <p11 dir> twin --set --reset step="..."    # driver start: drop the keys of earlier launches
    python3 p11_status.py <p11 dir> count <subdir> <arm> [<arm> ...]     # -> "<done> <crashed>" summaries

Each caller (chain, twin) keeps its state in <p11>/driver/<who>_state.json (key=value pairs merged in). STATUS.md
(atomic: one temp file per writer) shows the chain (step, done/total of the current closed-loop step from the
summaries, GPUs, the fine-tune's epoch x/10 from metrics.jsonl, "come back at"), the step-0 result once V2_FINAL.txt
exists, the twin's line, and STOPPED / AT GATE / CHAIN DONE. Keys used: step, gpus, state, reason, started, eta,
note, run (fine-tune or twin run folder), subdir + arms + total (closed-loop progress).
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple


def load(path: str) -> Dict[str, str]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(path: str, state: Dict[str, str]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def count(p11: str, subdir: str, arms: List[str]) -> Tuple[int, int]:
    """(finished summaries, crash records) of these arms in <p11>/<subdir>/summaries."""
    done = crashed = 0
    for arm in arms:
        d = os.path.join(p11, subdir, "summaries", arm)
        if os.path.isdir(d):
            names = os.listdir(d)
            crashed += sum(n.endswith(".crash.json") for n in names)
            done += sum(n.endswith(".json") and not n.endswith(".crash.json") for n in names)
    return done, crashed


def epochs(run: str) -> Tuple[int, float]:
    """(epochs done, mean minutes per epoch) of a training run folder."""
    path = os.path.join(run or "", "metrics.jsonl")
    if not run or not os.path.exists(path):
        return 0, 0.0
    mins = []
    for line in open(path):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if "train_seconds" in rec:
            mins.append((rec["train_seconds"] + rec.get("eval_seconds", 0)) / 60)
    return len(mins), (sum(mins) / len(mins) if mins else 0.0)


def line_of(p11: str, who: str, st: Dict[str, str]) -> List[str]:
    out = [f"- **{who}**: {st.get('state', '?')} — step {st.get('step', '?')}; GPUs {st.get('gpus', '-')}"]
    if st.get("subdir") and st.get("arms"):
        done, crashed = count(p11, st["subdir"], st["arms"].split())
        out.append(f"  - runs {done}/{st.get('total', '?')} done, {crashed} crashed ({st['subdir']}: {st['arms']})")
    if st.get("run"):
        n, m = epochs(st["run"])
        total = st.get("epochs", "?")
        out.append(f"  - training {os.path.basename(st['run'])}: epoch {n}/{total} done"
                   + (f", {m:.1f} min per epoch" if m else ""))
    for k in ("eta", "note", "reason"):
        if st.get(k):
            out.append(f"  - {k}: {st[k]}")
    return out


def write_status(p11: str) -> None:
    chain = load(os.path.join(p11, "driver", "chain_state.json"))
    twin = load(os.path.join(p11, "driver", "twin_state.json"))
    lines = [f"# Phase 11 STATUS — {chain.get('state', 'NOT STARTED')}", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')} (Istanbul time). Started {chain.get('started', '-')}.", ""]
    if chain.get("come_back"):
        lines += [f"**Come back at: {chain['come_back']}**", ""]
    lines += line_of(p11, "chain", chain)
    v2 = os.path.join(p11, "V2_FINAL.txt")
    if os.path.exists(v2):
        lines += ["", "Step-0 result: " + open(v2).read().strip()]
    lines += [""] + (line_of(p11, "twin (report only)", twin) if twin else ["- **twin (report only)**: not started"])
    if os.path.exists(os.path.join(p11, "STOP_reason.txt")):
        lines += ["", "STOP_reason.txt:", "```", open(os.path.join(p11, "STOP_reason.txt")).read().strip(), "```"]
    lines += ["", "Logs: driver/chain.log, driver/twin.log, driver/gpu_watch.log; `screen -r p11_chain`."]
    tmp = os.path.join(p11, f".STATUS.md.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(p11, "STATUS.md"))


def main() -> None:
    p11, who = sys.argv[1], sys.argv[2]
    if who == "count":
        done, crashed = count(p11, sys.argv[3], sys.argv[4:])
        print(done, crashed)
        return
    path = os.path.join(p11, "driver", f"{who}_state.json")
    st = load(path)
    args = sys.argv[3:]
    if args and args[0] == "--set":
        args = args[1:]
    while args:
        a = args.pop(0)
        if a == "--reset":  # a driver (re)start: stale run / reason / gpus of an earlier launch would show otherwise
            st = {}
        elif a == "--state":
            st["state"] = args.pop(0)
        elif "=" in a:
            k, v = a.split("=", 1)
            st[k] = v
    save(path, st)
    write_status(p11)


if __name__ == "__main__":
    main()
