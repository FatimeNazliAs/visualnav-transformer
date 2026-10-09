"""Run closed-loop arms on the frozen Phase 1 episodes against a simulator server (inside naz_mapmad).

    python -m vint_train.mapmad.closed_loop.run_arms --arms photo_iv explore_oov --port 5550 --shard 0 --num-shards 2

Seeds: episode seed = run.base_seed + the episode's index in the episode file (the same for every arm, so arms
are paired on identical noise, and independent of sharding). Writes under <outputs>/<out_dir>[/<subdir>]:
logs/<arm>/<episode>.jsonl, summaries/<arm>/<episode>.json, videos/<arm>/<episode>.mp4,
timing/<arm>.shard<i>.jsonl (wall-clock per episode and NoMaD compute time per step, kept out of the logs) and run_info.<arms>.shard<i>.json.
mapmad/src must be on PYTHONPATH (mapmad_bridge, mapmad_sim.config); mapmad/scripts/run_p1.sh sets it.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from mapmad_bridge.client import SimClient  # noqa: E402
from mapmad_sim import config as mapmad_config  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import P1_CONFIG, RunLayout  # noqa: E402
from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy  # noqa: E402
from vint_train.mapmad.closed_loop.runner import Arm, RunSettings, run_episode, write_video  # noqa: E402


def deterministic_torch() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_policy(layout: RunLayout) -> NomadPolicy:
    n = layout.cfg["nomad"]
    cfg = yaml.safe_load(layout.nomad_config.read_text())
    drive = mapmad_config.robot()["drive"]
    sim = mapmad_config.robot()["sim"]
    return NomadPolicy(str(layout.nomad_weights), cfg, torch.device("cuda"), num_samples=n["num_samples"],
                       sample_index=n["sample_index"], waypoint_index=n["waypoint_index"], max_v=drive["max_v_mps"],
                       max_w=drive["max_w_radps"], rate_hz=sim["control_hz"])


def select(episodes: List[Dict[str, Any]], arm: Arm, ids: List[str], shard: int, num_shards: int,
           limit: int) -> List[Dict[str, Any]]:
    """The arm's episodes (with their file index) for this shard."""
    picked = [dict(e, index=i) for i, e in enumerate(episodes) if e["type"] == arm.episodes and (not ids or e["episode_id"] in ids)]
    picked = picked[shard::num_shards]
    return picked[:limit] if limit else picked


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=P1_CONFIG)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--host", default="naz_mapmad_habitat")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--episodes", nargs="*", default=[], help="only these episode ids")
    p.add_argument("--limit", type=int, default=0, help="at most this many episodes per arm (0 = all)")
    p.add_argument("--subdir", default="", help="write under <out_dir>/<subdir> (e.g. checks/determinism_a)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--shutdown-server", action="store_true", help="stop the simulator server at the end")
    args = p.parse_args()

    layout = RunLayout.load(args.config, subdir=args.subdir)
    p1 = layout.cfg
    episodes, fingerprint = load_episodes(layout.episodes_file)
    arms = [Arm(name=a, **p1["arms"][a]) for a in args.arms]
    run = RunSettings(success_m=p1["run"]["success_m"], max_steps=p1["run"]["max_steps"],
                      topdown_m_per_px=p1["run"]["topdown_m_per_px"], video_fps=p1["run"]["video_fps"])
    deterministic_torch()
    policy = make_policy(layout)
    tag = f"{'+'.join(args.arms)}.shard{args.shard}"
    save_run_info(layout.root, {**vars(args), "config": str(args.config), "p1": p1, "episodes_fingerprint": fingerprint,
                        "weights_sha256": hashlib.sha256(layout.nomad_weights.read_bytes()).hexdigest(),
                        "robot": mapmad_config.robot(),
                        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
                  p1["run"]["base_seed"], name=f"run_info.{tag}.json")
    sim = SimClient(args.host, args.port)
    try:
        for arm in arms:
            todo = select(episodes, arm, args.episodes, args.shard, args.num_shards, args.limit)
            timing = layout.timing(arm.name, args.shard)
            timing.parent.mkdir(parents=True, exist_ok=True)
            for e in todo:
                t0 = time.perf_counter()
                seed = p1["run"]["base_seed"] + e["index"]
                result, rec = run_episode(sim, policy, e, arm, run, seed, layout.log(arm.name, e["episode_id"]), fingerprint)
                t_run = time.perf_counter() - t0
                summary = layout.summary(arm.name, e["episode_id"])
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
                if not args.no_video:
                    write_video(layout.video(arm.name, e["episode_id"]), e, arm, result, rec, run)
                ms = np.array(rec.policy_ms) if rec.policy_ms else np.zeros(1)
                with open(timing, "a") as f:
                    f.write(json.dumps({"episode_id": e["episode_id"], "steps": result["steps"], "run_s": round(t_run, 2),
                                        "total_s": round(time.perf_counter() - t0, 2),
                                        "policy_ms_mean": round(float(ms.mean()), 2),
                                        "policy_ms_p95": round(float(np.percentile(ms, 95)), 2),
                                        "policy_ms_max": round(float(ms.max()), 2)}) + "\n")
                print(f"{arm.name} {e['episode_id']}: {'SUCCESS' if result['success'] else 'fail'} in {result['steps']} "
                      f"steps, spl {result['spl']:.2f}, collisions {result['collisions']}, final {result['final_geodesic_m']} m "
                      f"({t_run:.0f} s)", flush=True)
    finally:
        sim.close(shutdown_server=args.shutdown_server)


if __name__ == "__main__":
    main()
