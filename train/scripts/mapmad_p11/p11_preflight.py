#!/usr/bin/env python3
"""Phase 11 chain pre-flight and run counts. Host side (standard library + PyYAML).

    python3 p11_preflight.py check <p11_rules.md> <outputs mount>
    python3 p11_preflight.py totals <run config (mapmad/configs/...)> <p11 dir>    # -> "<step0> <rollouts> <dev_v21>"
    python3 p11_preflight.py first-dev <run config> <p11 dir>   # -> the first dev episode id (determinism episode)

`check`: the rules file's ```json preflight``` block lists every frozen input the chain judges with: checkpoints
(path below <outputs mount>/mapmad -> sha256) and episode sets (path -> fingerprint); each must match (block item 9:
checkpoint sha256 mismatch = automatic stop). Exit 1 on any mismatch.
`totals`: the number of runs of the three closed-loop steps from the config's arms, seeds and the frozen sets.
"""

import hashlib
import json
import os
import re
import sys
from typing import Any, Dict

import yaml


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(episodes) -> str:
    """mapmad_sim.episodes.fingerprint: sha256 of the canonical JSON of the episode list."""
    return hashlib.sha256(json.dumps(list(episodes), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preflight_block(rules: str) -> Dict[str, Any]:
    m = re.search(r"```json preflight\n(.*?)\n```", open(rules).read(), re.S)
    if not m:
        raise SystemExit(f"{rules}: no ```json preflight``` block")
    return json.loads(m.group(1))


def cmd_check(rules: str, out_mount: str) -> int:
    block, bad = preflight_block(rules), []
    for rel, sha in block["checkpoints"].items():
        got = file_sha256(os.path.join(out_mount, "mapmad", rel))
        print(f"checkpoint {rel}: {got[:16]} {'ok' if got == sha else 'MISMATCH'}")
        bad += [] if got == sha else [rel]
    for rel, fp in block["sets"].items():
        doc = json.load(open(os.path.join(out_mount, "mapmad", rel)))
        got = fingerprint(doc["episodes"])
        ok = got == fp and doc["fingerprint"] == fp
        print(f"set {rel}: {got[:16]} {'ok' if ok else 'MISMATCH'}")
        bad += [] if ok else [rel]
    return 1 if bad else 0


def read_config(path: str) -> Dict[str, Any]:
    """mapmad_sim.run_layout.read_config (extends: merged key by key; None removes)."""
    cfg = yaml.safe_load(open(path))
    base = cfg.pop("extends", None)
    if not base:
        return cfg

    def merge(a, b):
        out = dict(a)
        for k, v in b.items():
            if v is None:
                out.pop(k, None)
            elif isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    return merge(read_config(os.path.join(os.path.dirname(path), base)), cfg)


def cmd_totals(config: str, p11: str) -> None:
    cfg = read_config(config)
    n = {s: len(json.load(open(os.path.join(p11, cfg["sets"][s]["folder"], "episodes.json")))["episodes"])
         for s in ("dev", "rollout")}
    seeds = int(cfg["run"]["seeds"])

    def k(arm):
        own = cfg["arms"][arm].get("seeds")
        return len([s for s in range(seeds) if own is None or s in own])

    print(n["dev"] * (k("run1") + k("runC") + k("run1_hidden")), n["rollout"], n["dev"] * k("v21"))


def main() -> None:
    if sys.argv[1] == "check":
        sys.exit(cmd_check(sys.argv[2], sys.argv[3]))
    if sys.argv[1] == "first-dev":
        cfg = read_config(sys.argv[2])
        print(json.load(open(os.path.join(sys.argv[3], cfg["sets"]["dev"]["folder"], "episodes.json")))["episodes"][0]["episode_id"])
        return
    cmd_totals(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
