#!/usr/bin/env python3
"""One GPU placement module (Phase 12 architecture candidate 3): the single answer to "which GPUs may <owner> take
now?" for every chain. Host side, standard library only, never touches a GPU. Two inputs:
- the watcher's free list (gpu_watch.py -> gpu_free.json: free for 20 min + rule 11), re-read on every call;
  a file older than 3 watcher intervals counts as "nothing free" (a dead watcher never hands out GPUs);
- a live nvidia-smi check right now (gpu_watch.check_now: no process of anyone but the owner).

    gpu_place.py free   --owner chain --free-file <gpu_free.json> [--fixed "0 1"]   # the watcher's free list
    gpu_place.py usable --owner chain --free-file <f> [--fixed ..]                 # free AND clean now
    gpu_place.py clean  <gpu> --owner chain                                         # exit 0 if clean now
    gpu_place.py pick   --owner chain --free-file <f> [--prefer 1] [--fixed ..]    # one usable GPU, preferred first;
                                                                                    # exit 1 if none (caller waits)
--fixed (DRY runs only: P13_GPUS) replaces the watcher's list; the live check still applies (except to `free`).
Busy GPUs are explained on stderr (the caller's log). Bash keeps one line per decision (p13_lib.sh).
"""

import argparse
import contextlib
import json
import os
import sys
import time
from typing import Callable, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STALE_INTERVALS = 3


def free_list(free_file: str, owner: str, fixed: Optional[str] = None, now: Optional[float] = None) -> List[str]:
    """GPUs free for `owner` by the watcher's file (or the fixed list); [] if the file is missing, unreadable or
    stale (older than STALE_INTERVALS x its check interval)."""
    if fixed:
        return fixed.split()
    try:
        with open(free_file) as f:
            d = json.load(f)
        age = (time.time() if now is None else now) - os.path.getmtime(free_file)
    except (OSError, ValueError):
        return []
    if age > STALE_INTERVALS * float(d.get("check_s", 300)):
        print(f"gpu_place: {free_file} is {age:.0f} s old (watcher not running?): no GPU free", file=sys.stderr)
        return []
    return [str(g) for g in d.get("free", {}).get(owner, [])]


def live_check(gpu: str, owner: str) -> bool:
    from gpu_watch import check_now  # nvidia-smi; imported here so the tests need no GPU tool
    with contextlib.redirect_stdout(sys.stderr):  # its "GPU g busy: ..." line goes to the log, not into $(...)
        return check_now(gpu, owner) == 0


def usable(free: Sequence[str], owner: str, check: Callable[[str, str], bool] = live_check) -> List[str]:
    """The free GPUs that are also clean right now (in the given order)."""
    return [g for g in free if check(g, owner)]


def pick(free: Sequence[str], owner: str, prefer: Sequence[str] = (),
         check: Callable[[str, str], bool] = live_check) -> Optional[str]:
    """One GPU: the preferred ones first, then the rest of the free list; only free AND clean now; None if none."""
    order = [g for g in prefer if g in free] + [g for g in free if g not in prefer]
    return next((g for g in order if check(g, owner)), None)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["free", "usable", "clean", "pick"])
    p.add_argument("gpu", nargs="?", help="clean: the GPU")
    p.add_argument("--owner", required=True)
    p.add_argument("--free-file")
    p.add_argument("--fixed", default="", help="DRY runs: these GPUs instead of the watcher's list")
    p.add_argument("--prefer", nargs="*", default=[])
    a = p.parse_args(argv)
    if a.cmd == "clean":
        return 0 if live_check(a.gpu, a.owner) else 1
    free = free_list(a.free_file or "", a.owner, a.fixed or None)
    if a.cmd == "free":
        print(" ".join(free))
        return 0
    if a.cmd == "usable":
        print(" ".join(usable(free, a.owner)))
        return 0
    g = pick(free, a.owner, a.prefer)
    if g is None:
        return 1
    print(g)
    return 0


if __name__ == "__main__":
    sys.exit(main())
