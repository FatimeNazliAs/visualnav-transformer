"""Run Phase 5 arms on a frozen episode set against one simulator server (inside naz_mapmad).

    python -m vint_train.mapmad.closed_loop.run_exam --config /app/visualnav-transformer/mapmad/configs/p5_exam.yaml \
        --arms A B C D E F --port 5550 --shard 0 --num-shards 4

Work: this shard's episodes (every num_shards-th of the set, so one episode's arms and seeds stay together and its
home stays loaded) x seed indices (`run.seeds`; the deterministic planner only index 0) x arms. Noise seed =
exam_runner.noise_seed(salt, episode, index), the same for every arm (block item 7).
Resumable at the episode level: a (arm, episode, seed) with a summary is skipped; a log is complete only once
renamed from .partial. Infrastructure errors (simulator/bridge exceptions, block item 13b) are retried up to 2
times with the same seed on a fresh connection; then summaries/<arm>/<episode>__s<k>.crash.json records the episode (the analysis
drops it from all arms). Any other exception stops the shard (a possible bug: item 13).
Checkpoint sha256 checked when each policy is built (exam_runner start). Writes under <outputs>/<out_dir>/<subdir>:
logs/<arm>/<episode>__s<k>.jsonl, summaries/<arm>/<episode>__s<k>.json, timing/<arm>.shard<i>.jsonl,
run_info.<tag>.json, progress.<tag>.log (tag = arms + shard); with --trace-maps traces/<arm>/<episode>__s<k>.npz (map parity).
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from mapmad_bridge.client import SimClient  # noqa: E402
from mapmad_sim import config as mapmad_config  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.exam_arms import arms_of  # noqa: E402
from mapmad_sim.run_layout import RunLayout, read_config, run_id  # noqa: E402
from vint_train.mapmad.closed_loop.exam_runner import ExamArm, ExamSettings, Trace, noise_seed, run_episode  # noqa: E402
from vint_train.mapmad.closed_loop.mapmad_policy import MapMaDPolicy, check_sha256  # noqa: E402
from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy  # noqa: E402
from vint_train.mapmad.closed_loop.planner_policy import PlannerPolicy  # noqa: E402
from vint_train.mapmad.closed_loop.run_arms import deterministic_torch  # noqa: E402
from vint_train.mapmad.config import load_config  # noqa: E402

RETRIES = 2
INFRA_ERRORS = (OSError, EOFError, ConnectionError)


def map_settings(layout: RunLayout) -> Dict[str, Any]:
    """The map numbers the server needs: Phase 2 builder `map` block + the train config's crop/heat `map` block."""
    ms = layout.cfg["map_settings"]
    builder = read_config(mapmad_config.CONFIG_DIR / ms["builder"])["map"]
    crop = load_config(str(layout.resolve(ms["crop"])))["map"]
    return {"builder": builder, "crop": crop}


class Policies:
    """One policy object per arm; the official NoMaD and the MapMaD model are each loaded once."""

    def __init__(self, layout: RunLayout, device: torch.device) -> None:
        self.layout, self.device = layout, device
        drive, sim = mapmad_config.robot()["drive"], mapmad_config.robot()["sim"]
        self.limits = dict(max_v=drive["max_v_mps"], max_w=drive["max_w_radps"], rate_hz=sim["control_hz"])
        self._nomad: Optional[NomadPolicy] = None
        self._mapmad_model = None
        self.cache: Dict[str, Any] = {}
        self.hashes: Dict[str, str] = {}

    def nomad(self) -> NomadPolicy:
        if self._nomad is None:
            n = self.layout.cfg["nomad"]
            weights = self.layout.resolve(n["weights"])
            self.hashes["nomad"] = check_sha256(str(weights), n["weights_sha256"])
            cfg = yaml.safe_load(self.layout.resolve(n["config"]).read_text())
            self._nomad = NomadPolicy(str(weights), cfg, self.device, num_samples=n["num_samples"],
                                      sample_index=n["sample_index"], waypoint_index=n["waypoint_index"], **self.limits)
        return self._nomad

    def mapmad(self, arm: ExamArm) -> MapMaDPolicy:
        m = self.layout.cfg["mapmad"]
        index = arm.waypoint_index if arm.waypoint_index is not None else m["waypoint_index"]
        if index is None:
            raise ValueError(f"arm {arm.name}: no waypoint index (mapmad.waypoint_index is set from the dev rule)")
        cfg = load_config(str(self.layout.resolve(m["model_config"])))
        pol = MapMaDPolicy(str(self.layout.resolve(m["checkpoint"])), m["checkpoint_sha256"], cfg, self.device,
                           arm.mapmad_goal,
                           num_samples=m["num_samples"], sample_index=m["sample_index"], waypoint_index=int(index),
                           context_spacing=m["context_spacing"], step_m=m["step_m"], model=self._mapmad_model,
                           **self.limits)
        self._mapmad_model = pol.model
        self.hashes["mapmad"] = pol.checkpoint_sha256
        return pol

    def planner(self) -> PlannerPolicy:
        p = self.layout.cfg["planner"]
        return PlannerPolicy(p["inflation_m"], p["lookahead_m"], p["start_clear_m"], **self.limits)

    def get(self, arm: ExamArm):
        if arm.name not in self.cache:
            self.cache[arm.name] = {"nomad": self.nomad, "planner": self.planner}[arm.policy]() \
                if arm.policy != "mapmad" else self.mapmad(arm)
        return self.cache[arm.name]

    def header(self, arm: ExamArm) -> Dict[str, Any]:
        """Weights fingerprint written into every log (G5 row 1: sha256 matched every run)."""
        return {"weights_sha256": self.hashes.get(arm.weights_key)} if arm.weights_key else {}


def work(episodes: List[Dict[str, Any]], arms: List[ExamArm], seeds: List[int], shard: int, num_shards: int):
    """(episode, seed index, arm) in run order for this shard."""
    for e in episodes[shard::num_shards]:
        for k in seeds:
            for arm in arms:
                if k in arm.seed_indices([k]):
                    yield e, k, arm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--host", default="naz_mapmad_habitat")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--episodes", nargs="*", default=[], help="only these episode ids")
    p.add_argument("--seeds", type=int, nargs="*", help="seed indices (default: range(run.seeds))")
    p.add_argument("--limit", type=int, default=0, help="at most this many episodes (0 = all)")
    p.add_argument("--subdir", default="runs", help="run outputs below <out_dir>/<subdir>")
    p.add_argument("--trace-maps", action="store_true", help="keep the map tensors + exact poses (parity check)")
    p.add_argument("--shutdown-server", action="store_true")
    args = p.parse_args()

    layout = RunLayout.load(args.config, subdir=args.subdir)
    cfg = layout.cfg
    episodes, fingerprint = load_episodes(layout.episodes_file)
    if len({e["episode_id"] for e in episodes}) != len(episodes):  # run paths and noise seeds are keyed by the id
        raise SystemExit(f"{layout.episodes_file}: duplicate episode ids")
    if args.episodes:
        episodes = [e for e in episodes if e["episode_id"] in set(args.episodes)]
    if args.limit:
        episodes = episodes[:args.limit]
    arms = arms_of(cfg, args.arms)
    run = ExamSettings.from_config(cfg["run"])
    seeds = args.seeds if args.seeds is not None else list(range(int(cfg["run"]["seeds"])))
    settings = map_settings(layout)
    deterministic_torch()
    policies = Policies(layout, torch.device("cuda"))
    for arm in arms:
        policies.get(arm)  # build (and sha256-check) every policy before the first episode
    tag = f"{'+'.join(args.arms)}.shard{args.shard}"
    save_run_info(layout.root, {**vars(args), "config": str(args.config), "p5": cfg, "episodes_fingerprint": fingerprint,
                                "weights_sha256": policies.hashes, "map_settings": settings, "robot": mapmad_config.robot(),
                                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
                  0, name=f"run_info.{tag}.json")
    todo = list(work(episodes, arms, seeds, args.shard, args.num_shards))
    progress = layout.progress(tag)
    sim = SimClient(args.host, args.port)
    t_start, done = time.time(), 0
    try:
        for e, k, arm in todo:
            rid = run_id(e["episode_id"], k)
            summary = layout.summary(arm.name, rid)
            done += 1
            crash = layout.crash(arm.name, rid)
            if summary.exists() or crash.exists():
                continue
            seed = noise_seed(cfg["run"]["seed_salt"], e["episode_id"], k)
            errors = []
            for attempt in range(RETRIES + 1):
                t0 = time.perf_counter()
                trace = Trace() if args.trace_maps else None
                try:
                    result, ms = run_episode(sim, policies.get(arm), e, arm, run, k, seed, layout.log(arm.name, rid),
                                             fingerprint, settings, policies.header(arm), trace)
                    break
                except INFRA_ERRORS + (RuntimeError,) as exc:
                    if isinstance(exc, RuntimeError) and not str(exc).startswith("simulator server"):
                        raise  # not an infrastructure error: stop (block item 13)
                    errors.append(f"{type(exc).__name__}: {exc}")
                    print(f"[retry {attempt + 1}] {arm.name} {rid}: {errors[-1]}", flush=True)
                    try:
                        sim.sock.close()
                    except OSError:
                        pass
                    sim = SimClient(args.host, args.port)
            else:
                summary.parent.mkdir(parents=True, exist_ok=True)
                crash.write_text(json.dumps({"errors": errors}, indent=1))
                continue
            wall = time.perf_counter() - t0
            if trace is not None and trace.maps:
                trace.save(layout.trace(arm.name, rid))
            summary.parent.mkdir(parents=True, exist_ok=True)
            summary.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
            timing = layout.timing(arm.name, args.shard)
            timing.parent.mkdir(parents=True, exist_ok=True)
            arr = np.array(ms) if ms else np.zeros(1)
            with open(timing, "a") as f:
                f.write(json.dumps({"episode": rid, "steps": result["steps"], "wall_s": round(wall, 2),
                                    "policy_ms_mean": round(float(arr.mean()), 2), "attempts": len(errors) + 1}) + "\n")
            line = (f"{time.strftime('%H:%M:%S')} {done}/{len(todo)} {arm.name} {rid}: "
                    f"{'SUCCESS' if result['success'] else 'fail'} {result['steps']} steps, spl {result['spl']:.2f}, "
                    f"coll {result['collisions']}, final {result['final_vp_geodesic_m']} m ({wall:.0f} s; "
                    f"elapsed {(time.time() - t_start) / 60:.1f} min)")
            print(line, flush=True)
            with open(progress, "a") as f:
                f.write(line + "\n")
    finally:
        sim.close(shutdown_server=args.shutdown_server)


if __name__ == "__main__":
    main()
