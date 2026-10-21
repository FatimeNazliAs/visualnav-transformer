"""Phase 9 driver (host, in screen via p9_job.sh, which adds the EXIT trap): every GPU step of the phase, unattended.

    MAPMAD_SIDECAR_KEY=$(openssl rand -hex 16) bash mapmad/scripts/p9_job.sh driver <p9>/driver \
        python3 mapmad/scripts/p9_driver.py

Steps (state in <p9>/driver/state.json; a finished step is never redone, so a relaunch resumes):
1. frames: candidates (GPU) -> select -> render (GPU) -> fingerprint (p9_build_frames.py);
2. rules: <p9>/p9_rules.md = mapmad/configs/p9_rules.md + the generated appendix (fingerprints, label tables, OVON
   mapping, weights sha256), its sha256 in p9_rules.md.sha256 -- before any model is scored;
3. models, one at a time from the queue (p9_models.yaml order, SAM 3 last; its download is retried once first):
   recogniser server on one GPU (naz_mapmad_recog) + client (naz_mapmad_habitat_p9) raw pass -> half-A threshold ->
   timing pass -> server stopped. A model failing twice is skipped with the reason (model_status.json); a model
   cut short while another process was on our GPU is not a failure: its folder goes to photo_test/_partial/ and it is
   re-run from zero;
4. scoring (CPU): truth through the pipeline, every model scored, choice rule, tables, sheets;
5. plug-in (GPU): truth parity on 2 dev drives (run right after step 2), then R1 and R2 on the same drives.
GPU rule (p9_rules_v2.md A1): a GPU is used only if p9_gpu_watch.py lists it as free (4 x 5-min checks without a
Phase 10 or other user's process) and the flag file <outputs>/mapmad/P10_WANTS_GPU does not exist (while it exists:
no new model starts, a running one finishes, then the GPU is released); checked before every GPU step (the watcher's file, then a live nvidia-smi check of that GPU right
before taking it) and re-checked every 5 min while one runs: if a Phase 10 / other process
appears on our GPU, the current model finishes and no new work starts there (the next step waits for a usable GPU).
One GPU at a time. Budget: > 12 h of GPU holding (1.5 x 8 h) -> STOP. Every GPU event goes to the log.
Refuses uncommitted code under mapmad/ (§0.8).
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from p9_gpu_watch import classify, container_ids, usable_gpus  # noqa: E402

HAB, REC = "naz_mapmad_habitat_p9", "naz_mapmad_recog"
CHECK_S = int(os.environ.get("CHECK_S", 300))
BUDGET_H = 12.0
PORT0 = 5560
SERVER_THREADS = 4  # CPU threads per recogniser server (13:13-15:38 slowdown: oversubscription)


def outputs_host() -> Path:
    out = subprocess.run(["docker", "inspect", "-f", '{{range .Mounts}}{{if eq .Destination "/outputs"}}{{.Source}}{{end}}{{end}}',
                          HAB], capture_output=True, text=True, check=True).stdout.strip()
    return Path(out) / "mapmad" / "p9_recognisers"


P9 = outputs_host()
DRV = P9 / "driver"
P10_WANTS_GPU = P9.parent / "P10_WANTS_GPU"  # p9_rules_v2.md A1 (created by Naz or Phase 10)
STATE = DRV / "state.json"
LOG = P9 / "phase9_log.md"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def say(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)  # p9_job.sh writes stdout to driver/driver.log
    with open(LOG, "a") as f:
        f.write(f"- [driver {now()}] {msg}\n")


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_state() -> Dict[str, Any]:
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": [], "gpu_hours": 0.0}


def save_state(st: Dict[str, Any]) -> None:
    write_atomic(STATE, json.dumps(st, indent=1) + "\n")


def status(st: Dict[str, Any], step: str, gpu: str, error: str = "") -> None:
    """STATUS.md for Naz (atomic)."""
    write_atomic(P9 / "STATUS.md", "\n".join([
        f"# Phase 9 STATUS (driver, updated {now()})",
        f"- Step: {step}",
        f"- Done: {', '.join(st['done']) or 'nothing yet'}",
        f"- GPU: {gpu}; GPU hours used so far: {st['gpu_hours']:.2f} (budget 8, stop at 12)",
        f"- Come back at: {st.get('come_back', 'see the check-in')}",
        f"- Last error: {error or 'none'}",
        "- Driver log: driver/driver.log; screen mapmad_p9_driver", ""]))


def run(cmd: List[str], log: Optional[Path] = None, check: bool = True) -> int:
    say("run: " + " ".join(cmd))
    with open(log or DRV / "commands.log", "a") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = p.wait()
        except BaseException:
            p.terminate()
            raise
    if check and code != 0:
        raise RuntimeError(f"exit {code}: {' '.join(cmd)} (log {log or DRV / 'commands.log'})")
    return code


def hab(*args: str, log: Optional[Path] = None, check: bool = True) -> int:
    return run(["docker", "exec", "-w", "/app/visualnav-transformer", "-e", "MAPMAD_SIDECAR_KEY", HAB, "python", *args],
               log, check)


# --- GPUs ---------------------------------------------------------------------------------------------------------

def gpu_state() -> Dict[str, Any]:
    return json.loads((DRV / "gpu_free.json").read_text())


def live_busy(g: int) -> bool:
    """True if GPU g holds a Phase 10 or another user's process right now (nvidia-smi, not the watcher's file)."""
    c = classify(container_ids([HAB, REC])).get(g, {"phase10": 0, "other": 0})
    return bool(c["phase10"] or c["other"])


def acquire(st: Dict[str, Any], what: str) -> int:
    """Wait until a GPU may be taken (module docstring); log the event."""
    waited = False
    while True:
        s = gpu_state()
        wants = P10_WANTS_GPU.exists()
        usable = usable_gpus(s, [], wants)
        if wants and not waited:
            say(f"GPU event: {P10_WANTS_GPU} exists: Phase 10 wants a GPU, no new Phase 9 work")
        if usable and live_busy(usable[0]):  # the watcher's file can be 5 min old
            say(f"GPU event: GPU {usable[0]} listed free but busy now (live check): not taken")
            usable = []
        if usable:
            g = usable[0]
            if waited:
                say(f"GPU event: GPU {g} usable now ({s})")
            say(f"GPU event: taking GPU {g} for {what} (free={s['free']} phase10={s['phase10']} other={s['other']})")
            run(["nvidia-smi"], DRV / "nvidia_smi.log")
            return g
        if not waited:
            say(f"GPU event: waiting for a usable GPU for {what} (free={s['free']} phase10={s['phase10']} other={s['other']})")
            status(st, f"waiting for a GPU ({what})", f"none usable (free={s['free']}, phase10={s['phase10']}, other={s['other']})")
            waited = True
        time.sleep(CHECK_S)


class Hold:
    """Context: counts GPU hours, re-checks every 5 min whether another process appeared on our GPU."""

    def __init__(self, st: Dict[str, Any], g: int, what: str) -> None:
        self.st, self.g, self.what = st, g, what
        self.contended = False  # another process was seen on our GPU while we held it

    def __enter__(self) -> "Hold":
        self.t0 = time.time()
        status(self.st, self.what, f"GPU {self.g}")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.st["gpu_hours"] += (time.time() - self.t0) / 3600.0
        save_state(self.st)
        if self.st["gpu_hours"] > BUDGET_H:
            raise SystemExit(f"STOP: GPU budget exceeded ({self.st['gpu_hours']:.2f} h > {BUDGET_H} h)")

    def check(self) -> None:
        if P10_WANTS_GPU.exists():
            say(f"GPU event: {P10_WANTS_GPU} exists: finishing {self.what} on GPU {self.g}, then releasing it")
        s = gpu_state()
        if self.g in s["phase10"] or self.g in s["other"] or live_busy(self.g):
            self.contended = True
            say(f"GPU event: another process on our GPU {self.g} (phase10={s['phase10']} other={s['other']}): "
                f"finishing {self.what}, no new work there")


def run_held(hold: Hold, cmd: List[str], log: Path) -> int:
    """Run a long command; every CHECK_S the GPU is re-checked (Hold.check)."""
    say("run: " + " ".join(cmd))
    with open(log, "a") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    return p.wait(timeout=CHECK_S)
                except subprocess.TimeoutExpired:
                    hold.check()
        except BaseException:
            p.terminate()
            raise


# --- steps --------------------------------------------------------------------------------------------------------

def step_frames(st: Dict[str, Any]) -> None:
    g = acquire(st, "frame rendering")
    with Hold(st, g, "9.1/9.2 frame set: candidates + render") as h:
        log = P9 / "frames" / "build.log"
        if run_held(h, ["docker", "exec", "-w", "/app/visualnav-transformer", HAB, "python",
                        "mapmad/scripts/p9_build_frames.py", "candidates", "--gpu", str(g)], log):
            raise RuntimeError("frame candidates failed (frames/build.log)")
        hab("mapmad/scripts/p9_build_frames.py", "select", log=log)
        if run_held(h, ["docker", "exec", "-w", "/app/visualnav-transformer", HAB, "python",
                        "mapmad/scripts/p9_build_frames.py", "render", "--gpu", str(g)], log):
            raise RuntimeError("frame render failed (frames/build.log)")


def step_rules(st: Dict[str, Any]) -> None:
    """Wait for the download job, retry failed downloads once (SAM 3 is retried before its turn), freeze the rules."""
    job = P9 / "jobs" / "download"
    while not ((job / "DONE.txt").exists() or (job / "STOP_reason.txt").exists()):
        status(st, "waiting for the model downloads before freezing the rules", "none")
        time.sleep(60)
    w = json.loads((P9 / "photo_test" / "weights.json").read_text())
    retry = [k for k in models_queue() if k != "sam3" and w.get(k, {}).get("status") != "ok"]
    if retry:
        run(["docker", "exec", "-e", "HOME=/model_cache/home", "-e", "YOLO_OFFLINE=False", "-e", "CUDA_VISIBLE_DEVICES=",
             REC, "python", "mapmad/scripts/p9_download.py", "--only", *retry, "--tries", "1"], DRV / "download_retry.log",
            check=False)
    hab("mapmad/scripts/p9_rules_appendix.py", log=DRV / "rules.log")
    say("rules frozen: " + (P9 / "p9_rules.md.sha256").read_text().strip())


def models_queue() -> List[str]:
    import re
    text = (REPO / "mapmad" / "configs" / "p9_models.yaml").read_text()
    order = re.search(r"run_order: \[([^\]]*)\]", text).group(1)  # p9_rules_v2.md A2
    return [m.strip() for m in order.split(",")]


def model_status() -> Dict[str, Any]:
    p = P9 / "photo_test" / "model_status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def set_model_status(key: str, entry: Dict[str, Any]) -> None:
    ms = model_status()
    ms[key] = {**entry, "time": now()}
    write_atomic(P9 / "photo_test" / "model_status.json", json.dumps(ms, indent=1) + "\n")


def server(key: str, g: int, port: int) -> None:
    log = f"/outputs/mapmad/p9_recognisers/photo_test/{key}/server.log"
    (P9 / "photo_test" / key).mkdir(parents=True, exist_ok=True)
    run(["docker", "exec", "-d", "-e", f"CUDA_VISIBLE_DEVICES={g}", "-e", "MAPMAD_SIDECAR_KEY", "-e", "HOME=/model_cache/home",
         "-e", "HF_HUB_OFFLINE=1", "-e", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
         "-e", f"OMP_NUM_THREADS={SERVER_THREADS}", "-e", f"MKL_NUM_THREADS={SERVER_THREADS}", REC, "bash", "-c",
         f"python -m mapmad_recog.server --model {key} --port {port} >> {log} 2>&1"])


def stop_server(key: str, port: int) -> None:
    run(["docker", "exec", REC, "pkill", "-f", f"mapmad_recog.server --model {key} --port {port}"], check=False)
    time.sleep(5)


def weights_ok(key: str) -> Optional[str]:
    w = P9 / "photo_test" / "weights.json"
    entry = json.loads(w.read_text()).get(key, {}) if w.exists() else {}
    return None if entry.get("status") == "ok" else entry.get("reason", "weights not downloaded")


def step_model(st: Dict[str, Any], key: str) -> None:
    if key == "sam3":  # gated: retry the download once before scoring it (block / Naz 2026-10-21)
        ok = run(["docker", "exec", REC, "test", "-r", "/hf_token/token"], check=False) == 0
        say(f"sam3: HF token file {'readable' if ok else 'NOT readable'} in {REC} (contents never read here)")
        run(["docker", "exec", "-e", "HOME=/model_cache/home", "-e", "YOLO_OFFLINE=False", "-e", "CUDA_VISIBLE_DEVICES=",
             REC, "python", "mapmad/scripts/p9_download.py", "--only", "sam3", "--tries", "1"], DRV / "download_retry.log",
            check=False)
    reason = weights_ok(key)
    if reason:
        set_model_status(key, {"status": "skipped", "reason": f"weights: {reason}"})
        say(f"{key}: skipped ({reason})")
        return
    folder = P9 / "photo_test" / key
    attempt = int(st.get("prior_failed_attempts", {}).get(key, 0))  # p9_rules_v2.md A1 (yolo_world: 1)
    if attempt:
        say(f"{key}: {attempt} failed attempt(s) before this launch count")
    while attempt < 2:
        attempt += 1
        g = acquire(st, f"model {key}")
        port = PORT0 + g
        with Hold(st, g, f"9.4 photo test: {key} (attempt {attempt})") as h:
            server(key, g, port)
            try:
                c = run_held(h, ["docker", "exec", "-w", "/app/visualnav-transformer", "-e", "MAPMAD_SIDECAR_KEY", HAB,
                                 "python", "mapmad/scripts/p9_photo_test.py", "raw", "--model", key, "--host", REC,
                                 "--port", str(port)], folder / "client.log")
                if c == 0:
                    hab("mapmad/scripts/p9_score.py", "tune", "--model", key, log=folder / "score.log")
                    kind = json.loads((folder / "raw" / "stats.json").read_text())["info"]["kind"]
                    if kind != "vlm":
                        t = json.loads((folder / "threshold.json").read_text())["threshold"]
                        c = run_held(h, ["docker", "exec", "-w", "/app/visualnav-transformer", "-e", "MAPMAD_SIDECAR_KEY",
                                         HAB, "python", "mapmad/scripts/p9_photo_test.py", "timing", "--model", key,
                                         "--host", REC, "--port", str(port), "--threshold", str(t)], folder / "client.log")
            finally:
                stop_server(key, port)
        if c != 0:
            h.check()  # a crash can come sooner than the next 5-min check (e.g. out of memory next to a new job)
        if h.contended:  # cut short or timed next to another job: re-run from zero, not an attempt
            dest = P9 / "photo_test" / "_partial" / f"{key}__{time.strftime('%Y%m%d-%H%M%S')}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(folder, dest)
            say(f"{key}: another process shared GPU {g} during the run (exit {c}): moved to {dest}, re-run from zero")
            attempt -= 1
            continue
        if c == 0:
            set_model_status(key, {"status": "ran", "gpu": g, "attempts": attempt})
            say(f"{key}: ran (GPU {g}, attempt {attempt})")
            return
        say(f"{key}: attempt {attempt} failed (exit {c}; see {folder}/client.log, server.log)")
    tail = (folder / "server.log").read_text().strip().splitlines()[-3:] if (folder / "server.log").exists() else []
    set_model_status(key, {"status": "skipped", "reason": "failed twice: " + " | ".join(tail)[-400:]})


EDGES = (0.01, 0.99)


def step_edges(st: Dict[str, Any]) -> None:
    """p9_rules_v2.md A3 (report only): a model whose threshold is on a grid edge gets one extra raw pass with score
    >= 0.001 (raw_edge/), scored at 0.01 / 0.005 / 0.001 -> photo_test/<model>/edge.json."""
    for key, v in model_status().items():
        folder = P9 / "photo_test" / key
        if v["status"] != "ran" or not (folder / "threshold.json").exists() or (folder / "edge.json").exists():
            continue
        if json.loads((folder / "raw" / "stats.json").read_text())["info"]["kind"] == "vlm":
            continue
        t = json.loads((folder / "threshold.json").read_text())["threshold"]
        if t not in EDGES:
            continue
        say(f"{key}: threshold {t} on a grid edge -> extra report-only raw pass at score >= 0.001 (A3)")
        while True:
            g = acquire(st, f"edge pass {key}")
            port = PORT0 + g
            with Hold(st, g, f"A3 edge pass: {key}") as h:
                server(key, g, port)
                try:
                    c = run_held(h, ["docker", "exec", "-w", "/app/visualnav-transformer", "-e", "MAPMAD_SIDECAR_KEY",
                                     HAB, "python", "mapmad/scripts/p9_photo_test.py", "raw", "--model", key, "--host",
                                     REC, "--port", str(port), "--min-score", "0.001", "--raw-dir", "raw_edge"],
                                 folder / "client_edge.log")
                finally:
                    stop_server(key, port)
            if c != 0:
                h.check()
            if h.contended:
                dest = P9 / "photo_test" / "_partial" / f"{key}_raw_edge__{time.strftime('%Y%m%d-%H%M%S')}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if (folder / "raw_edge").exists():
                    os.replace(folder / "raw_edge", dest)
                say(f"{key}: edge pass shared GPU {g}: moved to {dest}, re-run")
                continue
            break
        if c == 0:
            hab("mapmad/scripts/p9_score.py", "edge", "--model", key, log=folder / "score.log")
        else:
            say(f"{key}: edge pass failed (exit {c}); reported as missing (report only)")


def step_scoring(st: Dict[str, Any]) -> None:
    status(st, "9.4/9.5 scoring (CPU)", "none")
    hab("mapmad/scripts/p9_truth_raw.py", log=P9 / "photo_test" / "score.log")
    ran = [k for k, v in model_status().items() if v["status"] == "ran"]
    for key in ran + ["truth"]:
        hab("mapmad/scripts/p9_score.py", "score", "--model", key, log=P9 / "photo_test" / "score.log")
    hab("mapmad/scripts/p9_score.py", "choose", log=P9 / "photo_test" / "score.log")
    for key in ran + ["truth"]:
        hab("mapmad/scripts/p9_score.py", "sheets", "--model", key, log=P9 / "photo_test" / "score.log", check=False)
    ch = json.loads((P9 / "photo_test" / "choice.json").read_text())
    say(f"choice: R1={ch['R1']} R2={ch['R2']} flags={ch['fallback_flags']}")


def step_plugin(st: Dict[str, Any], rec: str) -> None:
    g = acquire(st, f"plug-in {rec}")
    out = P9 / "plugin"
    out.mkdir(exist_ok=True)
    port = PORT0 + g
    with Hold(st, g, f"9.6 plug-in: {rec}") as h:
        args = ["docker", "exec", "-w", "/app/visualnav-transformer", "-e", "MAPMAD_SIDECAR_KEY", HAB, "python",
                "mapmad/scripts/p9_plugin_run.py", "--recogniser", rec, "--gpu", str(g)]
        if rec != "truth":
            server(rec, g, port)
            args += ["--port", str(port)]
        try:
            c = run_held(h, args, out / f"{rec}.log")
        finally:
            if rec != "truth":
                stop_server(rec, port)
    if c != 0:
        raise RuntimeError(f"plug-in {rec} failed (plugin/{rec}.log)")


def main() -> None:
    DRV.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("MAPMAD_SIDECAR_KEY"):
        raise SystemExit("set MAPMAD_SIDECAR_KEY")
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "mapmad", "train"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted code under mapmad/ or train/ (§0.8): commit first\n{dirty}")
    st = load_state()
    st.setdefault("come_back", os.environ.get("COME_BACK", "see the check-in"))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    say(f"driver start (commit {commit}); done so far: {st['done']}")
    # truth parity first: a short GPU step that shows a plug-in problem before the long model runs
    steps = [("frames", step_frames), ("rules", step_rules), ("plugin:truth", lambda s: step_plugin(s, "truth"))]
    steps += [(f"model:{k}", (lambda s, k=k: step_model(s, k))) for k in models_queue()]
    steps += [("edges", step_edges), ("scoring", step_scoring)]
    for name, fn in steps:
        if name in st["done"]:
            continue
        say(f"step {name}: start")
        fn(st)
        st["done"].append(name)
        save_state(st)
        say(f"step {name}: done")
    ch = json.loads((P9 / "photo_test" / "choice.json").read_text())
    for r in ("R1", "R2"):
        name = f"plugin:{r}"
        if name not in st["done"] and ch.get(r):
            say(f"step {name} ({ch[r]}): start")
            step_plugin(st, ch[r])
            st["done"].append(name)
            save_state(st)
    status(st, "AT GATE: driver finished (waiting for Claude Code: gate table + summary)", "none")
    say("DONE")


def cleanup_ours() -> None:
    """Stop our own processes inside our two containers. Killing a host-side `docker exec` does not stop the process in
    the container (18:42: a yolo_world server and client outlived the driver). Matches only our scripts."""
    for container, pattern in ((HAB, "mapmad/scripts/p9_"), (REC, "mapmad_recog.server")):
        subprocess.run(["docker", "exec", container, "pkill", "-f", pattern], capture_output=True, check=False)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # finally blocks stop our servers
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(129))
    try:
        main()
    except BaseException as exc:
        if not (isinstance(exc, SystemExit) and exc.code in (0, None)):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)  # a closing screen must not cut the clean-up short
            cleanup_ours()
            st = load_state()
            status(st, f"STOPPED: {type(exc).__name__}: {exc}", "released", str(exc)[:300])
            say(f"STOPPED: {type(exc).__name__}: {exc}")
        raise
