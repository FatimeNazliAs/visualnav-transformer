"""Phase 9 GPU watcher (host, in screen; never touches a GPU). v2 plan §8 + Phase 9 block item 9.

Every CHECK_S seconds (default 300) it lists the compute processes on each GPU and sorts them:
- ours: a process inside one of the Phase 9 containers (`--ours`, by the docker id in /proc/<pid>/cgroup);
- phase10: another process of the same user (Naz's Phase 10 jobs in naz_mapmad / naz_mapmad_habitat);
- other: a process of another user.
A GPU is "free" after FREE_CHECKS (4) consecutive checks with no phase10 / other process (20 min). Writes
<dir>/gpu_free.json atomically:
    {"time", "streak": {gpu: n}, "free": [...], "phase10": [...], "other": [...], "ours": [...]}
and logs every change to <dir>/gpu_watch.log. Who may take which GPU is decided by `usable_gpus` (Phase 9 priority:
Phase 10 main run > Phase 9 > Phase 10 twin run; Phase 9 takes a GPU only if afterwards another GPU is still free or
Phase 10 already has a GPU).

    python3 mapmad/scripts/p9_gpu_watch.py <dir> --ours naz_mapmad_habitat_p9 naz_mapmad_recog
"""

import argparse
import json
import os
import pwd
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Set


def run(cmd: Sequence[str]) -> str:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False).stdout


def container_ids(names: Sequence[str]) -> Set[str]:
    """Full docker ids of the named containers that exist."""
    ids = set()
    for n in names:
        out = run(["docker", "inspect", "-f", "{{.Id}}", n]).strip()
        if out:
            ids.add(out)
    return ids


def pid_container(pid: int) -> str:
    """Docker id of the container a host pid runs in ('' = none / unknown)."""
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if "/docker/" in line:
                return line.rsplit("/docker/", 1)[1].strip()
    except OSError:
        pass
    return ""


def pid_user(pid: int) -> str:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (OSError, KeyError):
        return ""


def classify(ours: Set[str]) -> Dict[int, Dict[str, int]]:
    """GPU index -> counts of ours / phase10 / other processes."""
    uuid_idx = {}
    for line in run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]).splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        uuid_idx[uuid] = int(idx)
    out = {i: {"ours": 0, "phase10": 0, "other": 0} for i in uuid_idx.values()}
    me = pwd.getpwuid(os.getuid()).pw_name
    for line in run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]).splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] not in uuid_idx:
            continue
        pid, gpu = int(parts[0]), uuid_idx[parts[1]]
        if pid_container(pid) in ours:
            out[gpu]["ours"] += 1
        elif pid_user(pid) == me:
            out[gpu]["phase10"] += 1
        else:  # another user, or a pid that vanished / can't be read: count as busy (the safe side)
            out[gpu]["other"] += 1
    return out


def usable_gpus(state: Dict, using: Sequence[int]) -> List[int]:
    """GPUs Phase 9 may start new work on now (block item 9): free by the 20-min rule, not already ours, and after
    taking it another GPU is still free or Phase 10 already holds a GPU."""
    free = [int(g) for g in state["free"]]
    phase10_running = bool(state["phase10"])
    out = []
    for g in free:
        if g in using:
            continue
        others_free = [h for h in free if h != g and h not in using]
        if others_free or phase10_running:
            out.append(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--check-s", type=int, default=int(os.environ.get("CHECK_S", 300)))
    ap.add_argument("--free-checks", type=int, default=4)
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    log = d / "gpu_watch.log"

    def say(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a") as f:
            f.write(line + "\n")

    say(f"watcher start: every {a.check_s} s, free = {a.free_checks} consecutive checks without phase10/other")
    streak: Dict[int, int] = {}
    last = None
    while True:
        cls = classify(container_ids(a.ours))
        for g, c in cls.items():
            streak[g] = 0 if (c["phase10"] or c["other"]) else streak.get(g, 0) + 1
        state = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "streak": {str(g): n for g, n in sorted(streak.items())},
                 "free": [g for g in sorted(streak) if streak[g] >= a.free_checks],
                 "phase10": [g for g in sorted(cls) if cls[g]["phase10"]],
                 "other": [g for g in sorted(cls) if cls[g]["other"]],
                 "ours": [g for g in sorted(cls) if cls[g]["ours"]]}
        tmp = d / ".gpu_free.json.tmp"
        tmp.write_text(json.dumps(state) + "\n")
        os.replace(tmp, d / "gpu_free.json")
        key = (state["free"], state["phase10"], state["other"], state["ours"])
        if key != last:
            say(f"free={state['free']} phase10={state['phase10']} other={state['other']} ours={state['ours']} "
                f"streak={state['streak']} usable_now={usable_gpus(state, state['ours'])}")
        last = key
        time.sleep(a.check_s)


if __name__ == "__main__":
    main()
