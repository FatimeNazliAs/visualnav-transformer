#!/usr/bin/env python3
"""Phase 10 STATUS.md writer (kickoff §2). Host side, standard library only. The chain and the twin launcher call it
after every step, every epoch and every GPU event:

    python3 p10_status.py <p10 dir> chain --set step="main run" gpu=1 main_run=/mnt/.../runs/v2_s1_... [--state RUNNING]

Each caller keeps its own state in <p10>/driver/<who>_state.json (key=value pairs merged in; numbers stay strings).
STATUS.md (atomic) shows the chain (step, epoch x/20 with the latest losses from the main run's metrics.jsonl,
GPU, a "come back at" from the measured epoch times + the offline estimate), the twin's line, and STOPPED /
CHAIN DONE. Keys used: step, gpu, state (RUNNING / WAITING / STOPPED / CHAIN DONE), reason, main_run, offline_h,
estimate_h, started, note.
"""

import json
import os
import sys
import time
from typing import Dict, Optional

EPOCHS = 20


def load(path: str) -> Dict[str, str]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(path: str, state: Dict[str, str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)


def run_progress(run: Optional[str]) -> Dict[str, object]:
    """Epochs done, mean minutes per epoch (train + eval + check), the last epoch's losses, the latest step line."""
    out: Dict[str, object] = {"done": 0}
    if not run:
        return out
    path = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(path):
        return out
    epochs, last_step = [], None
    for line in open(path):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("record") == "step":
            last_step = rec
        elif "train_seconds" in rec:
            epochs.append(rec)
    out["done"] = len(epochs)
    if epochs:
        mins = [(e["train_seconds"] + e.get("eval_seconds", 0) + 60 * e.get("training_check", {}).get("minutes", 0)) / 60
                for e in epochs]
        out["min_per_epoch"] = sum(mins) / len(mins)
        last = epochs[-1]
        out["losses"] = {"diffusion": round(last["diffusion_loss"], 4), "dist": round(last["dist_loss"], 3),
                         **{k: round(v, 4) for k, v in last.get("train_mode_diffusion_loss", {}).items()}}
        chk = last.get("training_check")
        if chk:
            out["check"] = {"map_diff": round(chk["map_diff"], 3), "R": round(chk["R"], 3),
                            "gs_delta": round(chk["no_harm_delta"], 3)}
    if last_step:
        out["step"] = f"epoch {last_step['epoch']} step {last_step['step']}, diffusion {last_step['diffusion_loss']:.4f}"
    return out


def come_back(state: Dict[str, str], prog: Dict[str, object]) -> str:
    """Now + remaining main-run epochs x measured minutes + the offline estimate (or the timing estimate)."""
    offline_h = float(state.get("offline_h", 1.5))
    if "min_per_epoch" in prog:
        left_h = (EPOCHS - int(prog["done"])) * float(prog["min_per_epoch"]) / 60 + offline_h
    elif "estimate_h" in state:
        left_h = float(state["estimate_h"]) + offline_h
    else:
        return state.get("come_back", "not estimated yet")
    return time.strftime("%H:%M (%a)", time.localtime(time.time() + 3600 * left_h)) + f" (~{left_h:.1f} h)"


def render(p10: str) -> str:
    chain = load(os.path.join(p10, "driver", "chain_state.json"))
    twin = load(os.path.join(p10, "driver", "twin_state.json"))
    prog = run_progress(chain.get("main_run"))
    lines = [f"# Phase 10 STATUS (written by the driver, {time.strftime('%Y-%m-%d %H:%M:%S')})", ""]
    state = chain.get("state", "RUNNING")
    if state == "STOPPED":
        lines += [f"**STOPPED: {chain.get('reason', '?')}** (STOP_reason.txt in the step's folder / driver/)", ""]
    if state == "CHAIN DONE":
        lines += ["**CHAIN DONE**", ""]
    lines += [f"- Chain step: {chain.get('step', '-')} ({state})",
              f"- Main run: epoch {prog['done']}/{EPOCHS} done" + (f"; now {prog['step']}" if "step" in prog else ""),
              f"- Latest losses: {json.dumps(prog.get('losses', {}))}",
              f"- Latest in-training check: {json.dumps(prog.get('check', {}))}",
              f"- GPU: {chain.get('gpu', 'none yet')}",
              f"- Come back at: {come_back(chain, prog) if state not in ('STOPPED', 'CHAIN DONE') else 'now'}",
              f"- Note: {chain.get('note', '-')}",
              "",
              f"- Twin (seed 2, report only): {twin.get('step', 'not started')} ({twin.get('state', 'WAITING')})"
              + (f"; GPU {twin['gpu']}" if twin.get("gpu") else "")
              + (f"; epoch {run_progress(twin.get('main_run'))['done']}/{EPOCHS}" if twin.get("main_run") else ""),
              "",
              "- Logs: driver/chain.log, driver/twin.log, driver/gpu_watch.log, phase10_log.md; screens p10_chain, "
              "p10_chain_watch, p10_twin, p10_gpuwatch"]
    return "\n".join(lines) + "\n"


def main() -> None:
    p10, who = sys.argv[1], sys.argv[2]
    path = os.path.join(p10, "driver", f"{who}_state.json")
    state = load(path)
    args = sys.argv[3:]
    if args and args[0] == "--set":
        for kv in args[1:]:
            if kv == "--state":
                break
            k, v = kv.split("=", 1)
            state[k] = v
    if "--state" in args:
        state["state"] = args[args.index("--state") + 1]
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(path, state)
    tmp = os.path.join(p10, f".STATUS.md.{os.getpid()}.tmp")  # one temp file per writer (writers can overlap)
    with open(tmp, "w") as f:
        f.write(render(p10))
    os.replace(tmp, os.path.join(p10, "STATUS.md"))


if __name__ == "__main__":
    main()
