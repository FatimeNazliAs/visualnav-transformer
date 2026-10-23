"""Run search arms (Phases 11-12) on a frozen episode set against one simulator server (inside naz_mapmad).

    python -m vint_train.mapmad.closed_loop.run_search --config /app/visualnav-transformer/mapmad/configs/p11_dagger.yaml \
        --set dev --arms run1 runC run1_hidden --port 5560 --claims <token> --subdir step0 --shard 0

Arms come from the config's arm table (mapmad_sim.arm_table: policy type, checkpoint key + sha256, maps shown or
hidden, what draws the target layer, recogniser, camera, seeds). Policies: `mapmad_v2` (search_policy.MapMaDV2Policy,
config `policy`), `nomad` (nomad_policy.NomadPolicy, explore = goal masked, config `policies.nomad`), `mapmad_v1`
(mapmad_policy.MapMaDPolicy, map goal with `map_zero` layers zeroed, `policies.mapmad_v1`), `frontier`
(frontier_policy.FrontierPolicy: arm F, whose teacher runs in the simulator server). Work = episodes x seed indices x
arms, in that order (one episode's runs stay together, its home stays loaded). Noise seed =
exam_runner.noise_seed(salt, episode, index) with the set's `seed_salt` (else run.seed_salt): the same diffusion noise
and target-layer noise for every arm.
Shared run queue (§0.9, --claims TOKEN): every runner walks the whole list and takes the next run nobody has claimed
in this launch (claims/<TOKEN>/<arm>/<run>); results do not depend on which pair runs a run. Resumable: a run with a
summary (or crash record) is skipped. Infrastructure errors (incl. "recogniser sidecar down") are retried twice with
the same seed on a fresh connection, then recorded as summaries/<arm>/<run>.crash.json; any other exception stops the
runner (a possible bug). Every network's checkpoint sha256 is checked when it is built, and every run's log carries it.
--save-drives (rollouts): the server records every run and writes it to <mapmad_data>/<rollouts.data_folder>/<episode
id> in the Phase 8 drive format; a folder left by an interrupted run (no summary) is moved to _partial/ first.
--stop-file: the runner takes no new run once this file exists (GPU withdrawal, §8 rule 4); its current run finishes.
--recog-port (Phase 12): the recogniser sidecar of this pair (host --recog-host); arm F's tau comes from
P12_DEV_CHOICES.json (or --tau, DRY runs).
Writes below <outputs>/<out_dir>/<subdir>: logs/, summaries/, timing/, progress.<tag>.log, run_info.<tag>.json.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from mapmad_bridge.client import SimClient  # noqa: E402
from mapmad_sim import config as mapmad_config  # noqa: E402
from mapmad_sim.arm_table import Arm, ArmTable  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, read_config, run_id, try_claim, write_text_atomic  # noqa: E402
from mapmad_sim.search_wire import camera_wire, datagen_config, search_settings  # noqa: E402
from vint_train.mapmad.closed_loop.exam_runner import noise_seed  # noqa: E402
from vint_train.mapmad.closed_loop.policy import Policy  # noqa: E402
from vint_train.mapmad.closed_loop.run_arms import deterministic_torch  # noqa: E402
from vint_train.mapmad.closed_loop.search_runner import SearchSettings, run_search_episode  # noqa: E402
from vint_train.mapmad.config import load_config  # noqa: E402

RETRIES = 2
INFRA_ERRORS = (OSError, EOFError, ConnectionError)
V2_TRAIN_CONFIG = "repo:train/config/mapmad_v2.yaml"  # window size / resolutions of the online map v2 (every arm)
MAP_LAYERS_V1 = ("obstacle", "explored", "heat")


def checkpoint_of(layout: RunLayout, key: str) -> Dict[str, str]:
    """{"path", "sha256", "config"} of a checkpoint key, resolved in this container (arm_table)."""
    ck = ArmTable.from_layout(layout).checkpoint(key)
    return {"path": str(ck.path), "sha256": ck.sha256, "config": str(ck.config)}


def seed_indices(layout: RunLayout, arm: str, asked: Optional[List[int]]) -> List[int]:
    """The arm's seed indices: --seeds (or range(run.seeds)), limited by the arm table."""
    base = asked if asked is not None else list(range(int(layout.cfg["run"]["seeds"])))
    return ArmTable.from_layout(layout).seeds(arm, base)


class Policies:
    """One policy per arm; one network per checkpoint, sha256-checked when built (arm_table.ArmTable.verify)."""

    def __init__(self, layout: RunLayout, device: torch.device) -> None:
        self.layout, self.device = layout, device
        self.table = ArmTable.from_layout(layout)
        drive, sim = mapmad_config.robot()["drive"], mapmad_config.robot()["sim"]
        self.limits = dict(max_v=drive["max_v_mps"], max_w=drive["max_w_radps"], rate_hz=sim["control_hz"])
        self.nets: Dict[str, Any] = {}
        self.ckpts: Dict[str, Dict[str, str]] = {}
        self.cache: Dict[str, Policy] = {}

    def arm(self, name: str) -> Arm:
        return self.table.arm(name)

    def _checkpoint(self, key: str) -> Dict[str, str]:
        if key not in self.ckpts:
            ck = self.table.verify(key)
            self.ckpts[key] = {"path": str(ck.path), "sha256": ck.sha256, "config": str(ck.config), "key": ck.key}
        return self.ckpts[key]

    def train_cfg(self, name: str) -> Dict[str, Any]:
        """The train config of the arm's network (mapmad_v2.yaml for arms without a v2 network: window numbers)."""
        a = self.arm(name)
        if a.policy == "mapmad_v2":
            return self.get(name).cfg
        return load_config(str(self.table.resolve(V2_TRAIN_CONFIG)))

    def get(self, name: str) -> Policy:
        if name not in self.cache:
            a = self.arm(name)
            self.cache[name] = getattr(self, f"_build_{a.policy}")(a)
        return self.cache[name]

    def _build_mapmad_v2(self, a: Arm) -> Policy:
        from vint_train.mapmad.closed_loop.search_policy import MapMaDV2Policy
        from vint_train.mapmad.inference_v2 import MapMaDV2Inference

        p, c = self.layout.cfg["policy"], self._checkpoint(a.checkpoint)
        if c["key"] not in self.nets:
            self.nets[c["key"]] = MapMaDV2Inference(c["path"], load_config(c["config"]), self.device, c["sha256"],
                                                    num_samples=p["num_samples"], step_m=p["step_m"])
        net = self.nets[c["key"]]
        return MapMaDV2Policy(c["path"], c["sha256"], net.cfg, self.device, bool(a.maps), num_samples=p["num_samples"],
                              sample_index=p["sample_index"], waypoint_index=p["waypoint_index"],
                              context_spacing=p["context_spacing"], step_m=p["step_m"], net=net, **self.limits)

    def _build_nomad(self, a: Arm) -> Policy:
        from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy, load_nomad

        p, c = self.layout.cfg["policies"]["nomad"], self._checkpoint(a.checkpoint)
        cfg = yaml.safe_load(Path(c["config"]).read_text())
        if c["key"] not in self.nets:
            self.nets[c["key"]] = load_nomad(c["path"], cfg, self.device)
        return NomadPolicy("", cfg, self.device, num_samples=p["num_samples"], sample_index=p["sample_index"],
                           waypoint_index=p["waypoint_index"], context_spacing=p["context_spacing"],
                           model=self.nets[c["key"]], **self.limits)

    def _build_mapmad_v1(self, a: Arm) -> Policy:
        from vint_train.mapmad.closed_loop.mapmad_policy import MapMaDPolicy

        p, c = self.layout.cfg["policies"]["mapmad_v1"], self._checkpoint(a.checkpoint)
        cfg = load_config(c["config"])
        pol = MapMaDPolicy(c["path"], c["sha256"], cfg, self.device, "map", num_samples=p["num_samples"],
                           sample_index=p["sample_index"], waypoint_index=p["waypoint_index"],
                           context_spacing=p["context_spacing"], step_m=p["step_m"], model=self.nets.get(c["key"]),
                           zero_layers=[MAP_LAYERS_V1.index(n) for n in a.map_zero], **self.limits)
        self.nets[c["key"]] = pol.model
        return pol

    def _build_frontier(self, a: Arm) -> Policy:
        from vint_train.mapmad.closed_loop.frontier_policy import FrontierPolicy

        teacher = datagen_config(self.layout.cfg)["teacher"]
        return FrontierPolicy(float(teacher["face_tolerance_deg"]), **self.limits)

    def header(self, name: str) -> Dict[str, Any]:
        """Log-header fields of the arm (Phase 11 arms: exactly Phase 11's fields)."""
        a = self.arm(name)
        if a.policy == "frontier":
            return {"policy_type": "frontier", "target": a.target, "recogniser": a.recogniser}
        c = self._checkpoint(a.checkpoint)
        out = {"weights_sha256": c["sha256"], "checkpoint": c["path"], "maps": bool(a.maps)}
        if self.layout.cfg.get("policies"):  # Phase 12 configs
            out.update(policy_type=a.policy, checkpoint_key=c["key"], target=a.target, recogniser=a.recogniser,
                       map_zero=list(a.map_zero))
        return out


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


def set_salt(layout: RunLayout, name: str) -> str:
    """The noise-seed salt of a set (Phase 12: dev = Phase 11's salt, exam sets = mapmad-p12)."""
    return layout.cfg["sets"][name].get("seed_salt", layout.cfg["run"]["seed_salt"])


def move_leftover(folder: Path, token: str) -> None:
    """A drive folder of an interrupted run (no summary): moved to <data>/_partial/<id>__<token> (never deleted)."""
    if folder.exists():
        dest = folder.parent / "_partial" / f"{folder.name}__{token}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(folder, dest)


def tau_of(layout: RunLayout, asked: Optional[float]) -> float:
    """Arm F's tau: --tau, else P12_DEV_CHOICES.json `tau` (written by the dev decisions before any exam run)."""
    if asked is not None:
        return float(asked)
    ch = ArmTable.from_layout(layout).choices()
    if ch is None:
        raise SystemExit("arm F needs tau: P12_DEV_CHOICES.json does not exist yet (or pass --tau for a DRY run)")
    return float(ch["tau"])


def with_heat_box(episode: Dict[str, Any]) -> Dict[str, Any]:
    """V1's online map needs a heat box (online_map.heat_target); its heat layer is zeroed (map_zero), so the box
    only has to exist: the usable instance whose view point is the target point (else the first one)."""
    t = episode["target"]
    if "box_center" in t:
        return episode
    insts = t["instances"]
    pick = next((i for i in insts if list(t["point"]) in [list(v) for v in i["view_points"]]), insts[0])
    return dict(episode, target=dict(t, box_center=pick["centre"], box_size=pick["size"]))


class Wire:
    """The per-arm server settings: search_v2 (+ Phase 12 target / rule / teacher) and extra outputs (V1's map)."""

    def __init__(self, layout: RunLayout, policies: Policies, arms: List[str], recog_host: str,
                 recog_port: Optional[int], tau: Optional[float]) -> None:
        cfg = layout.cfg
        dg = datagen_config(cfg)
        self.base: Dict[str, Dict[str, Any]] = {}
        self.extra: Dict[str, Dict[str, Any]] = {}
        self.cameras: Dict[str, Dict[str, Any]] = {}
        for name in arms:
            a = policies.arm(name)
            sv = search_settings(dg, policies.train_cfg(name)["map"])
            if cfg.get("policies"):  # Phase 12
                run = cfg["run"]
                sv["rule"] = run["rule_stop"]
                target: Dict[str, Any] = {"source": a.target}
                if a.target == "recogniser":
                    if recog_port is None:
                        raise SystemExit(f"arm {name} needs a recogniser sidecar (--recog-port)")
                    r = policies.table.recogniser(a.recogniser)
                    target["recogniser"] = {"host": recog_host, "port": int(recog_port), "model": r["model"],
                                            "threshold": float(r["threshold"]), "p99": float(r["p99"]),
                                            "vlm": bool(r["vlm"])}
                sv["target"] = target
                if a.policy == "frontier":
                    f = cfg["policies"]["frontier"]
                    sv["teacher"] = {"spec": dg["teacher"], "seen": a.teacher_seen, "mode": a.teacher_mode,
                                     "mark_radius_m": float(f["mark_radius_m"]),
                                     "min_views": int(run["rule_stop"]["min_views"]),
                                     "tau": tau_of(layout, tau) if (a.teacher_seen == "layer" and
                                                                    a.teacher_mode == "search") else 1.0}
                if a.policy == "mapmad_v1":
                    ms = cfg["map_settings"]
                    self.extra[name] = {"map": True, "map_settings": {
                        "builder": read_config(mapmad_config.CONFIG_DIR / ms["builder"])["map"],
                        "crop": load_config(str(policies.table.resolve(ms["crop"])))["map"]}}
            self.base[name] = sv
            cam = policies.table.camera_name(a)
            self.cameras[name] = camera_wire(cam) if cam.endswith(".yaml") else {"name": cam, **cfg["cameras"][cam]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", required=True, help="a key of the config's `sets`")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--host", default="naz_mapmad_habitat")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--recog-host", default="naz_mapmad_recog")
    p.add_argument("--recog-port", type=int, help="Phase 12: this pair's recogniser sidecar")
    p.add_argument("--tau", type=float, help="arm F's tau (DRY runs; else P12_DEV_CHOICES.json)")
    p.add_argument("--live-rule-tau", type=float, help="check runs only: the robot stops at the first rule value >= tau")
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
    run = SearchSettings.from_config(dict(cfg["run"], **({"live_rule_tau": args.live_rule_tau}
                                                         if args.live_rule_tau is not None else {})))
    seeds = {a: seed_indices(layout, a, args.seeds) for a in args.arms}
    salt = set_salt(layout, args.set)
    deterministic_torch()
    policies = Policies(layout, torch.device("cuda"))
    for arm in args.arms:
        policies.get(arm)  # build and sha256-check every network before the first episode
    wire = Wire(layout, policies, args.arms, args.recog_host, args.recog_port, args.tau)
    data_dir = mapmad_config.paths()["mapmad_data"] / cfg["rollouts"]["data_folder"] if args.save_drives else None
    tag = f"{'+'.join(args.arms)}.shard{args.shard}"
    if len(tag) > 120:
        tag = f"{len(args.arms)}arms_{hashlib.sha256(' '.join(args.arms).encode()).hexdigest()[:8]}.shard{args.shard}"
    save_run_info(layout.root, {**vars(args), "config": str(args.config), "phase_config": cfg,
                                "episodes_fingerprint": fingerprint, "seed_salt": salt, "checkpoints": policies.ckpts,
                                "search_v2": wire.base, "outputs_extra": wire.extra, "cameras": wire.cameras,
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
            seed = noise_seed(salt, e["episode_id"], k)
            save_dir = data_dir / e["episode_id"] if data_dir is not None else None
            sv = dict(wire.base[arm], noise_seed=seed, record=save_dir is not None)
            ep = with_heat_box(e) if arm in wire.extra else e
            errors = []
            for attempt in range(RETRIES + 1):
                if save_dir is not None:
                    move_leftover(save_dir, f"{args.claims}_{attempt}")
                t0 = time.perf_counter()
                rec_ms: List[float] = []
                try:
                    result, ms = run_search_episode(sim, policies.get(arm), ep, arm, run, k, seed, layout.log(arm, rid),
                                                    fingerprint, sv, wire.cameras[arm], policies.header(arm), save_dir,
                                                    wire.extra.get(arm), rec_ms)
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
            row = {"run": rid, "steps": result["steps"], "wall_s": round(wall, 2),
                   "policy_ms_mean": round(float(arr.mean()), 2), "attempts": len(errors) + 1}
            if rec_ms:
                r = np.asarray(rec_ms)
                row.update(rec_ms_mean=round(float(r.mean()), 2), rec_ms_p95=round(float(np.percentile(r, 95)), 2),
                           rec_ms_max=round(float(r.max()), 2))
            with open(timing, "a") as f:
                f.write(json.dumps(row) + "\n")
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
