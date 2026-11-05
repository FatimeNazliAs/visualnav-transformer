#!/usr/bin/env python3
"""Phase 14a STATUS.md writer and run counter (kickoff §2; = mapmad_p13/p13_status.py with Phase 14a names). Host side, standard library only.

    python3 p14_status.py <p14a dir> chain --set --reset step="0 pre-flight" started="..." --state RUNNING
    python3 p14_status.py <p14a dir> chain --set step="13.7 exams" gpus="0 1" done_at="Fri 30 Oct ~23:00"
    python3 p14_status.py <p14a dir> chain --set fallback="13.1: render from re-runs"     # appended, never replaced
    python3 p14_status.py <p14a dir> count <subdir> <arm> [<arm> ...]      # -> "<done> <crashed>" summaries

The chain keeps its state in <p14a>/driver/chain_state.json (key=value pairs merged in; --reset at every driver start
drops the keys of an earlier launch, Phase 11 lesson); `fallback` and `error` values are appended to lists. STATUS.md
(atomic) starts with the line Naz pastes into the chat on return, then: state (RUNNING / WAITING / STOPPED / CHAIN
DONE / AT GATE), the current step and done/total, GPUs and mode, "done at", every fallback taken, the last error,
notes, and STOP_reason.txt with its one-line question at a safety stop.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

PASTE = os.environ.get("P14_PASTE") or ("continue Phase 14a: read STATUS.md and the chain log, then finish the tables, "
                                       "14.6, the G14a table and the summary")
PHASE = os.environ.get("P14_PHASE", "14a")  # Phase 14b-1 reuses this writer (P14_PHASE, P14_PASTE, P14_LOGNAME)
LOGNAME = os.environ.get("P14_LOGNAME", "phase14a_log.md")
LISTS = ("fallback", "error")


def load(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(path: str, state: Dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def count_arm(p13: str, subdir: str, arm: str) -> Tuple[int, int]:
    """(done, crashed) runs of the arm whose episode id starts with $CL_EP_PREFIX (Phase 14a: p13d- / p14d-) and, with
    $CL_SEEDS (Phase 14b-3), whose seed index is one of those."""
    d = os.path.join(p13, subdir, "summaries", arm)
    if not os.path.isdir(d):
        return 0, 0
    names = [n for n in os.listdir(d) if n.startswith(os.environ.get("CL_EP_PREFIX", ""))]
    seeds = os.environ.get("CL_SEEDS", "").split()  # Phase 14b-3: exam B calls per seed index share runs/examb
    if seeds:
        names = [n for n in names if "__s" in n and n.split(".", 1)[0].rsplit("__s", 1)[1] in seeds]
    crashed = sum(n.endswith(".crash.json") for n in names)
    return sum(n.endswith(".json") and not n.endswith(".crash.json") for n in names), crashed


def count(p13: str, subdir: str, arms: List[str]) -> Tuple[int, int]:
    done = crashed = 0
    for arm in arms:
        d, c = count_arm(p13, subdir, arm)
        done, crashed = done + d, crashed + c
    return done, crashed


def write_status(p13: str) -> None:
    st = load(os.path.join(p13, "driver", "chain_state.json"))
    lines = [f"**On return, paste in the VS Code chat:** `{PASTE}`", "",
             f"# Phase {PHASE} STATUS — {st.get('state', 'NOT STARTED')}", "",
             f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')} (Istanbul time). Driver started {st.get('started', '-')}.", ""]
    if st.get("come_back"):
        lines += [f"**CHAIN DONE expected: {st['come_back']}**", ""]
    lines.append(f"- step: {st.get('step', '?')}; GPUs: {st.get('gpus', '-') or '-'}; mode: {st.get('mode', '-')}")
    if st.get("done_at"):
        lines.append(f"- this step done at: {st['done_at']}")
    if st.get("subdir") and st.get("arms"):
        parts = []
        for arm in st["arms"].split():
            d, c = count_arm(p13, st["subdir"], arm)
            parts.append(f"{arm} {d}" + (f" (+{c} crashed)" if c else ""))
        done, crashed = count(p13, st["subdir"], st["arms"].split())
        lines.append(f"- runs of `{st.get('step_label', st['subdir'])}`: {done}/{st.get('total', '?')} done, {crashed} "
                     f"crashed — {', '.join(parts)} ({st['subdir']})")
    for k in ("eta", "budget", "estimate", "note", "reason", "other_user"):
        if st.get(k):
            lines.append(f"- {k}: {st[k]}")
    if st.get("fallback"):
        lines += ["", "**Fallbacks taken (kickoff §3; listed under Decisions to confirm):**"]
        lines += [f"- {x}" for x in st["fallback"]]
    if st.get("error"):
        lines += ["", f"**Last error:** {st['error'][-1]}"]
    f = os.path.join(p13, "STOP_reason.txt")
    if os.path.exists(f):
        lines += ["", "STOP_reason.txt:", "```", open(f).read().strip(), "```"]
    lines += ["", f"Logs: driver/chain.log, driver/gpu_watch.log, {LOGNAME}; `screen -r p{PHASE.replace('-', '')}_chain`."]
    tmp = os.path.join(p13, f".STATUS.md.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, os.path.join(p13, "STATUS.md"))


def main() -> None:
    p13, who = sys.argv[1], sys.argv[2]
    if who == "count":
        done, crashed = count(p13, sys.argv[3], sys.argv[4:])
        print(done, crashed)
        return
    path = os.path.join(p13, "driver", f"{who}_state.json")
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
            if k in LISTS:
                st.setdefault(k, []).append(f"{time.strftime('%m-%d %H:%M')} {v}")
            else:
                st[k] = v
    save(path, st)
    write_status(p13)


if __name__ == "__main__":
    main()
