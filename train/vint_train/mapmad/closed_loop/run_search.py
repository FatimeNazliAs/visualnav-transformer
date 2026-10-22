"""Run Phase 11 search arms on a frozen episode set against one simulator server (inside naz_mapmad).

    python -m vint_train.mapmad.closed_loop.run_search --config /app/visualnav-transformer/mapmad/configs/p11_dagger.yaml \
        --set dev --arms run1 runC run1_hidden --port 5560 --claims <token> --subdir step0 --shard 0

Arms (`arms` in the config): a checkpoint key (`checkpoints`: path + sha256 + train config, or `from_file`: a JSON
with the same keys written by an earlier step into <out_dir>, e.g. V2_FINAL.json) and whether the maps are shown; `seeds` limits an
arm to some seed indices. Work = episodes x seed indices (`run.seeds`) x arms, in that order (one episode's runs stay
together, its home stays loaded). Noise seed = exam_runner.noise_seed(salt, episode, index): the same diffusion noise
and the same target-layer noise for every arm (block item 1).
Shared run queue (§0.9, --claims TOKEN): every runner walks the whole list and takes the next run nobody has claimed
in this launch (claims/<TOKEN>/<arm>/<run>); results do not depend on which pair runs a run. Resumable: a run with a
summary (or crash record) is skipped. Infrastructure errors are retried twice with the same seed on a fresh
connection, then recorded as summaries/<arm>/<run>.crash.json; any other exception stops the runner (a possible bug).
Every network's checkpoint sha256 is checked when it is built, and every run's log carries it.
--save-drives (rollouts): the server records every run and writes it to <mapmad_data>/<rollouts.data_folder>/<episode
id> in the Phase 8 drive format; a folder left by an interrupted run (no summary) is moved to _partial/ first.
--stop-file: the runner takes no new run once this file exists (GPU withdrawal, §8 rule 4); its current run finishes.
Writes below <outputs>/<out_dir>/<subdir>: logs/, summaries/, timing/, progress.<tag>.log, run_info.<tag>.json.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mapmad_bridge.client import SimClient  # noqa: E402
from mapmad_sim import config as mapmad_config  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, run_id, try_claim, write_text_atomic  # noqa: E402
from mapmad_sim.search_wire import camera_wire, datagen_config, search_settings  # noqa: E402
from vint_train.mapmad.closed_loop.exam_runner import noise_seed  # noqa: E402
from vint_train.mapmad.closed_loop.run_arms import deterministic_torch  # noqa: E402
from vint_train.mapmad.closed_loop.search_policy import MapMaDV2Policy  # noqa: E402
from vint_train.mapmad.closed_loop.search_runner import SearchSettings, run_search_episode  # noqa: E402
from vint_train.mapmad.config import load_config  # noqa: E402
from vint_train.mapmad.inference_v2 import MapMaDV2Inference  # noqa: E402

RETRIES = 2
INFRA_ERRORS = (OSError, EOFError, ConnectionError)


def checkpoint_of(layout: RunLayout, key: str) -> Dict[str, str]:
    """{"path", "sha256", "config"} of a checkpoint key (`checkpoints` in the config; `from_file` -> that JSON)."""
    c = layout.cfg["checkpoints"][key]
    if "from_file" in c:
        f = layout.base / c["from_file"]  # written by an earlier step of this phase, in its own output folder
        if not f.exists():
            raise SystemExit(f"checkpoint {key}: {f} does not exist yet")
        c = json.loads(f.read_text())
    return {"path": str(layout.resolve(c["path"])), "sha256": c["sha256"], "config": str(layout.resolve(c["config"]))}


def seed_indices(layout: RunLayout, arm: str, asked: Optional[List[int]]) -> List[int]:
    """The arm's seed indices: --seeds (or range(run.seeds)), limited by the arm's own `seeds`."""
    base = asked if asked is not None else list(range(int(layout.cfg["run"]["seeds"])))
    own = layout.cfg["arms"][arm].get("seeds")
    return [k for k in base if own is None or k in own]


class Policies:
    """One MapMaDV2Policy per arm; one network per checkpoint, sha256-checked when built."""

    def __init__(self, layout: RunLayout, device: torch.device) -> None:
        self.layout, self.device = layout, device
        drive, sim = mapmad_config.robot()["drive"], mapmad_config.robot()["sim"]
        self.limits = dict(max_v=drive["max_v_mps"], max_w=drive["max_w_radps"], rate_hz=sim["control_hz"])
        self.nets: Dict[str, MapMaDV2Inference] = {}
        self.ckpts: Dict[str, Dict[str, str]] = {}
        self.cache: Dict[str, MapMaDV2Policy] = {}

    def get(self, arm: str) -> MapMaDV2Policy:
        if arm not in self.cache:
            a, p = self.layout.cfg["arms"][arm], self.layout.cfg["policy"]
            key = a["checkpoint"]
            if key not in self.nets:
                c = self.ckpts[key] = checkpoint_of(self.layout, key)
                self.nets[key] = MapMaDV2Inference(c["path"], load_config(c["config"]), self.device, c["sha256"],
                                                   num_samples=p["num_samples"], step_m=p["step_m"])
            c = self.ckpts[key]
            self.cache[arm] = MapMaDV2Policy(c["path"], c["sha256"], self.nets[key].cfg, self.device, bool(a["maps"]),
                                             num_samples=p["num_samples"], sample_index=p["sample_index"],
                                             waypoint_index=p["waypoint_index"], context_spacing=p["context_spacing"],
                                             step_m=p["step_m"], net=self.nets[key], **self.limits)
        return self.cache[arm]

    def header(self, arm: str) -> Dict[str, Any]:
        c = self.ckpts[self.layout.cfg["arms"][arm]["checkpoint"]]
        return {"weights_sha256": c["sha256"], "checkpoint": c["path"], "maps": bool(self.layout.cfg["arms"][arm]["maps"])}


def work(episodes: List[Dict[str, Any]], arms: List[str], seeds: Dict[str, List[int]]) -> List[Tuple[Dict, int, str]]:
    """(episode, seed index, arm) in run order."""
    out = []
    every = sorted({k for s in seeds.values() for k in s})
    for e in episodes:
        for k in every:
            for arm in arms:
                if k in seeds[arm]:
                    out.append((e, k, arm))
    return out


def set_file(layout: RunLayout, name: str) -> Path:
    return layout.base / layout.cfg["sets"][name]["folder"] / "episodes.json"


def move_leftover(folder: Path, token: str) -> None:
    """A drive folder of an interrupted run (no summary): moved to <data>/_partial/<id>__<token> (never deleted)."""
    if folder.exists():
        dest = folder.parent / "_partial" / f"{folder.name}__{token}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(folder, dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", choices=["dev", "rollout"], required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--host", default="naz_mapmad_habitat")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--shard", type=int, default=0, help="names this runner's files")
    p.add_argument("--claims", required=True, help="launch token of the shared run queue")
    p.add_argument("--episodes", nargs="*", default=[], help="only these episode ids")
    p.add_argument("--seeds", type=int, nargs="*", help="seed indices (default: range(run.seeds))")
    p.add_argument("--subdir", required=True, help="run outputs below <out_dir>/<subdir>")
    p.add_argument("--save-drives", action="store_true", help="rollouts: record every run as a Phase 8 drive")
    p.add_argument("--stop-file", type=Path, help="take no new run once this file exists (GPU withdrawal)")
    p.add_argument("--shutdown-server", action="store_true")
    args = p.parse_args()

    layout = RunLayout.load(args.config, subdir=args.subdir)
    cfg = layout.cfg
    episodes, fingerprint = load_episodes(set_file(layout, args.set))
    if len({e["episode_id"] for e in episodes}) != len(episodes):
        raise SystemExit(f"{args.set}: duplicate episode ids")
    if args.episodes:
        episodes = [e for e in episodes if e["episode_id"] in set(args.episodes)]
    run = SearchSettings.from_config(cfg["run"])
    seeds = {a: seed_indices(layout, a, args.seeds) for a in args.arms}
    camera = camera_wire(cfg["camera"])
    dg = datagen_config(cfg)
    deterministic_torch()
    policies = Policies(layout, torch.device("cuda"))
    for arm in args.arms:
        policies.get(arm)  # build and sha256-check every network before the first episode
    wire = {a: search_settings(dg, policies.get(a).cfg["map"]) for a in args.arms}
    data_dir = mapmad_config.paths()["mapmad_data"] / cfg["rollouts"]["data_folder"]
    tag = f"{'+'.join(args.arms)}.shard{args.shard}"
    save_run_info(layout.root, {**vars(args), "config": str(args.config), "p11": cfg, "episodes_fingerprint": fingerprint,
                                "checkpoints": policies.ckpts, "search_v2": wire, "camera": camera,
                                "robot": mapmad_config.robot(), "torch": torch.__version__,
                                "gpu": torch.cuda.get_device_name(0)}, 0, name=f"run_info.{tag}.json")
    todo = work(episodes, args.arms, seeds)
    progress = layout.progress(tag)
    sim = SimClient(args.host, args.port)
    t_start, done = time.time(), 0
    try:
        for e, k, arm in todo:
            if args.stop_file is not None and args.stop_file.exists():
                print(f"stop file {args.stop_file}: no new runs", flush=True)
                break
            rid = run_id(e["episode_id"], k)
            summary, crash = layout.summary(arm, rid), layout.crash(arm, rid)
            done += 1
            if summary.exists() or crash.exists():
                continue
            if not try_claim(layout.claim(args.claims, arm, rid)):
                continue
            seed = noise_seed(cfg["run"]["seed_salt"], e["episode_id"], k)
            save_dir = data_dir / e["episode_id"] if args.save_drives else None
            sv = dict(wire[arm], noise_seed=seed, record=save_dir is not None)
            errors = []
            for attempt in range(RETRIES + 1):
                if save_dir is not None:
                    move_leftover(save_dir, f"{args.claims}_{attempt}")
                t0 = time.perf_counter()
                try:
                    result, ms = run_search_episode(sim, policies.get(arm), e, arm, run, k, seed, layout.log(arm, rid),
                                                    fingerprint, sv, camera, policies.header(arm), save_dir)
                    break
                except INFRA_ERRORS + (RuntimeError,) as exc:
                    if isinstance(exc, RuntimeError) and not str(exc).startswith("simulator server"):
                        raise  # not an infrastructure error: stop (a possible bug)
                    errors.append(f"{type(exc).__name__}: {exc}")
                    print(f"[retry {attempt + 1}] {arm} {rid}: {errors[-1]}", flush=True)
                    try:
                        sim.sock.close()
                    except OSError:
                        pass
                    sim = SimClient(args.host, args.port)
            else:
                write_text_atomic(crash, json.dumps({"errors": errors}, indent=1))
                continue
            wall = time.perf_counter() - t0
            write_text_atomic(summary, json.dumps(result, indent=1, sort_keys=True) + "\n")
            timing = layout.timing(arm, args.shard)
            timing.parent.mkdir(parents=True, exist_ok=True)
            arr = np.array(ms) if ms else np.zeros(1)
            with open(timing, "a") as f:
                f.write(json.dumps({"run": rid, "steps": result["steps"], "wall_s": round(wall, 2),
                                    "policy_ms_mean": round(float(arr.mean()), 2), "attempts": len(errors) + 1}) + "\n")
            line = (f"{time.strftime('%H:%M:%S')} {done}/{len(todo)} {arm} {rid}: "
                    f"{'SUCCESS' if result['success'] else 'fail'} {result['steps']} steps, spl {result['spl']:.2f}, "
                    f"coll {result['collisions']}, seen {result['first_in_view_step']} ({wall:.0f} s; "
                    f"elapsed {(time.time() - t_start) / 60:.1f} min)")
            print(line, flush=True)
            with open(progress, "a") as f:
                f.write(line + "\n")
    finally:
        sim.close(shutdown_server=args.shutdown_server)


if __name__ == "__main__":
    main()
