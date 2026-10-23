#!/usr/bin/env python3
"""Phase 12 STATUS.md writer and run counter (kickoff §2). Host side, standard library only.

    python3 p12_status.py <p12 dir> chain --set --reset step="0 pre-flight" started="..." --state RUNNING
    python3 p12_status.py <p12 dir> chain --set step="5 main exam" gpus="0 1" come_back="Mon 26 Oct ~14:00"
    python3 p12_status.py <p12 dir> count <subdir> <arm> [<arm> ...]      # -> "<done> <crashed>" summaries

The chain keeps its state in <p12>/driver/chain_state.json (key=value pairs merged in; --reset at every driver start
drops the keys of an earlier launch, Phase 11 lesson). STATUS.md (atomic) shows: state (RUNNING / WAITING / STOPPED /
CHAIN DONE / AT GATE), step, GPUs, "come back at", the current step's done/total per arm (from the summaries), the dev
choices once P12_DEV_CHOICES.json exists (headline, deltas, tau; "both Delta <= 0" when so), the headline pair's interim
numbers once driver/headline_interim.txt exists, notes (estimate changes, trims, GPU-only-1), STOP_reason.txt.
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


def count_arm(p12: str, subdir: str, arm: str) -> Tuple[int, int]:
    d = os.path.join(p12, subdir, "summaries", arm)
    if not os.path.isdir(d):
        return 0, 0
    names = os.listdir(d)
    crashed = sum(n.endswith(".crash.json") for n in names)
    return sum(n.endswith(".json") and not n.endswith(".crash.json") for n in names), crashed


def count(p12: str, subdir: str, arms: List[str]) -> Tuple[int, int]:
    done = crashed = 0
    for arm in arms:
        d, c = count_arm(p12, subdir, arm)
        done, crashed = done + d, crashed + c
    return done, crashed


def write_status(p12: str) -> None:
    st = load(os.path.join(p12, "driver", "chain_state.json"))
    lines = [f"# Phase 12 STATUS — {st.get('state', 'NOT STARTED')}", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')} (Istanbul time). Driver started {st.get('started', '-')}.", ""]
    if st.get("come_back"):
        lines += [f"**Come back at: {st['come_back']}**", ""]
    lines.append(f"- step: {st.get('step', '?')}; GPUs: {st.get('gpus', '-') or '-'}")
    if st.get("subdir") and st.get("arms"):
        parts = []
        for arm in st["arms"].split():
            d, c = count_arm(p12, st["subdir"], arm)
            parts.append(f"{arm} {d}" + (f" (+{c} crashed)" if c else ""))
        done, crashed = count(p12, st["subdir"], st["arms"].split())
        lines.append(f"- runs of `{st.get('step_label', st['subdir'])}`: {done}/{st.get('total', '?')} done, {crashed} "
                     f"crashed — {', '.join(parts)} ({st['subdir']})")
    for k in ("eta", "budget", "note", "estimate", "reason"):
        if st.get(k):
            lines.append(f"- {k}: {st[k]}")
    ch = load(os.path.join(p12, "P12_DEV_CHOICES.json"))
    if ch:
        d = ch.get("deltas", {})
        tau = ch.get("tau")
        lines += ["", f"**Dev choices:** headline = **{ch.get('headline')}**; Delta(v2) = {d.get('v2', float('nan')):+.4f}, "
                  f"Delta(v2.1) = {d.get('v21', float('nan')):+.4f}" + (" — **both Delta <= 0**" if ch.get("both_delta_le_0") else "")
                  + (f"; tau = **{tau:.2f}**" if tau is not None else "; tau: pending")]
    interim = os.path.join(p12, "driver", "headline_interim.txt")
    if os.path.exists(interim):
        lines += ["", "**Headline pair (interim, main exam):** " + open(interim).read().strip()]
    for name in ("STOP_reason.txt",):
        f = os.path.join(p12, name)
        if os.path.exists(f):
            lines += ["", f"{name}:", "```", open(f).read().strip(), "```"]
    lines += ["", "Logs: driver/chain.log, driver/gpu_watch.log, phase12_log.md; `screen -r p12_chain`."]
    tmp = os.path.join(p12, f".STATUS.md.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(p12, "STATUS.md"))


def main() -> None:
    p12, who = sys.argv[1], sys.argv[2]
    if who == "count":
        done, crashed = count(p12, sys.argv[3], sys.argv[4:])
        print(done, crashed)
        return
    path = os.path.join(p12, "driver", f"{who}_state.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    st = load(path)
    args = sys.argv[3:]
    if args and args[0] == "--set":
        args = args[1:]
    while args:
        a = args.pop(0)
        if a == "--reset":
            st = {}
        elif a == "--state":
            st["state"] = args.pop(0)
        elif "=" in a:
            k, v = a.split("=", 1)
            st[k] = v
    save(path, st)
    write_status(p12)


if __name__ == "__main__":
    main()
