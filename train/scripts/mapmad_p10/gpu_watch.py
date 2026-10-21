#!/usr/bin/env python3
"""Phase 10 GPU watcher (v2 plan §8 GPU rule, Phase 10 block item 9). Host side, standard library only, never
touches a GPU:

    screen -dmS p10_gpuwatch python3 train/scripts/mapmad_p10/gpu_watch.py /mnt/shared_disk/nazli/nomad_outputs/mapmad/p10_v2_train/driver

Every CHECK_S seconds (default 300) it lists every compute process per GPU (nvidia-smi) and gives each an owner:
- "chain" / "twin": our own Phase 10 processes, marked with MAPMAD_P10_OWNER=chain|twin in their environment
  (docker exec -e ...; /proc/<pid>/environ, readable because every container runs as our uid);
- "other": everything else -- other users' jobs AND Phase 9's processes (they count as busy, block item 9).
Per GPU and per owner it keeps a streak: the number of consecutive checks with no process of anyone else (for
"any": no process at all). A GPU is free for an owner when that streak >= FREE_CHECKS (4 = 20 min, §0.4). Written
atomically to <dir>/gpu_free.json:
    {"time", "check_s", "free_checks", "gpus": {"0": {"procs": [...], "streak": {"any", "chain", "twin"}}},
     "free": {"any": [...], "chain": [...], "twin": [...]}}
Every change of the free sets and every process that appears or ends goes to <dir>/gpu_watch.log (GPU events).
`gpu_watch.py --check <gpu> <owner>`: exit 0 if that GPU holds no process of anyone but the owner right now.
"""

import json
import os
import subprocess
import sys
import time
from typing import Dict, List

OWNERS = ("any", "chain", "twin")


def owner_of(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            env = f.read().split(b"\0")
    except OSError:
        return "other"
    for item in env:
        if item.startswith(b"MAPMAD_P10_OWNER="):
            value = item.split(b"=", 1)[1].decode()
            return value if value in ("chain", "twin") else "other"
    return "other"


def gpu_processes() -> Dict[str, List[Dict]]:
    """{gpu index: [{pid, owner, mib, user, cmd}]} for every GPU."""
    idx = {}
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).splitlines():
        i, uuid = [x.strip() for x in line.split(",")]
        idx[uuid] = i
    out: Dict[str, List[Dict]] = {i: [] for i in idx.values()}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                    "--format=csv,noheader,nounits"], text=True)
    for line in apps.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        try:
            user = subprocess.check_output(["ps", "-o", "user=", "-p", str(pid)], text=True).strip()
            cmd = subprocess.check_output(["ps", "-o", "args=", "-p", str(pid)], text=True).strip()[:160]
        except subprocess.CalledProcessError:
            user, cmd = "?", "?"
        out.setdefault(idx.get(parts[1], parts[1]), []).append(
            {"pid": pid, "owner": owner_of(pid), "mib": parts[2], "user": user, "cmd": cmd})
    return out


def write_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def check_now(gpu: str, owner: str) -> int:
    """Exit code 0 if GPU `gpu` holds no process right now except the owner's own (launchers re-check right before a
    start, so they never share a GPU between two watcher checks)."""
    procs = gpu_processes().get(gpu, [])
    foreign = [p for p in procs if p["owner"] != owner]
    if foreign:
        print(f"GPU {gpu} busy: " + "; ".join(f"{p['pid']} {p['owner']} {p['user']} {p['cmd'][:60]}" for p in foreign))
        return 1
    return 0


def main() -> None:
    if sys.argv[1] == "--check":
        sys.exit(check_now(sys.argv[2], sys.argv[3]))
    folder = sys.argv[1]
    check_s = int(os.environ.get("CHECK_S", "300"))
    need = int(os.environ.get("FREE_CHECKS", "4"))
    os.makedirs(folder, exist_ok=True)
    log_path = os.path.join(folder, "gpu_watch.log")

    def say(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    say(f"watcher start: every {check_s} s, free = {need} consecutive checks; owners by MAPMAD_P10_OWNER")
    streak: Dict[str, Dict[str, int]] = {}
    last_free, last_pids = None, {}
    while True:
        procs = gpu_processes()
        for g, ps in procs.items():
            s = streak.setdefault(g, {o: 0 for o in OWNERS})
            owners = {p["owner"] for p in ps}
            s["any"] = 0 if ps else s["any"] + 1
            for o in ("chain", "twin"):
                s[o] = 0 if owners - {o} else s[o] + 1
        free = {o: sorted(g for g, s in streak.items() if s[o] >= need) for o in OWNERS}
        pids = {(g, p["pid"]): p for g, ps in procs.items() for p in ps}
        for key in pids.keys() - last_pids.keys():
            p = pids[key]
            say(f"GPU {key[0]}: process {p['pid']} appeared (owner {p['owner']}, user {p['user']}, {p['mib']} MiB): {p['cmd']}")
        for key in last_pids.keys() - pids.keys():
            say(f"GPU {key[0]}: process {key[1]} ended (owner {last_pids[key]['owner']})")
        if free != last_free:
            say(f"free={free} streaks={streak}")
        last_free, last_pids = free, pids
        write_atomic(os.path.join(folder, "gpu_free.json"),
                     json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "check_s": check_s, "free_checks": need,
                                 "gpus": {g: {"procs": procs[g], "streak": streak[g]} for g in sorted(procs)},
                                 "free": free}) + "\n")
        time.sleep(check_s)


if __name__ == "__main__":
    main()
