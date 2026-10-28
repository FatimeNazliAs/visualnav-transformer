"""The `search_v2` settings a closed-loop client sends to the simulator server at reset (Phase 11; sidecar_server,
search_v2.SearchSpec.from_wire), and one arm's whole wiring (`arm_wire`, Phase 12: the exam runner, the video replay
and the F check build an arm's settings only through it). Pure Python + PyYAML, so the NoMaD container builds them too
(search_v2 itself needs the Habitat side's packages)."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from mapmad_sim import config
from mapmad_sim.arm_table import ArmTable
from mapmad_sim.run_layout import read_config


def search_settings(datagen_cfg: Dict[str, Any], train_map: Dict[str, Any]) -> Dict[str, Any]:
    """The wire settings from the Phase 8 generator config (p8_datagen_v3a.yaml: `map`, `target_layer`, `motion`,
    `teacher`, `record`) and the train config's `map` block (window size and resolutions)."""
    return {"builder": datagen_cfg["map"], "target_layer": datagen_cfg["target_layer"],
            "bump_ahead_m": datagen_cfg["motion"]["bump_ahead_m"], "size": train_map["size"],
            "near_m": train_map["resolution"], "far_m": train_map["far_resolution"],
            "min_cluster": datagen_cfg["teacher"]["min_cluster"], "jpg_quality": datagen_cfg["record"]["jpg_quality"]}


def camera_wire(name_file: str) -> Dict[str, Any]:
    """A camera setting file of configs/ (e.g. camera_fair.yaml) as the dict the bridge carries (with its "name")."""
    return config.load_yaml(config.CONFIG_DIR / name_file)


def datagen_config(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The Phase 8 generator config a Phase 11 run config names (`datagen`)."""
    return read_config(config.CONFIG_DIR / run_cfg["datagen"])


# --- one arm's wiring (Phase 12 clean-up candidate 2) ------------------------------------------------------------------
V2_TRAIN_CONFIG = "repo:train/config/mapmad_v2.yaml"  # window size / resolutions of the online map v2 (every arm)
RECOG_HOST = "naz_mapmad_recog"


@dataclass(frozen=True)
class ArmWire:
    """What the simulator server needs for one arm: the reset's `search_v2` block (without the per-run noise seed and
    record flag), the extra outputs (V1's v1 map, else None) and the camera."""

    search_v2: Dict[str, Any]
    extra: Optional[Dict[str, Any]]
    camera: Dict[str, Any]


def f_tau(table: ArmTable, asked: Optional[float]) -> float:
    """Arm F's tau: `asked` (DRY runs), else P12_DEV_CHOICES.json `tau` (written before any exam run)."""
    if asked is not None:
        return float(asked)
    ch = table.choices()
    if ch is None:
        raise SystemExit("arm F needs tau: P12_DEV_CHOICES.json does not exist yet (or pass --tau for a DRY run)")
    return float(ch["tau"])


def arm_wire(table: ArmTable, name: str, train_config: Callable[[str], Dict[str, Any]],
             recog: Optional[Tuple[str, int]] = None, tau: Optional[float] = None) -> ArmWire:
    """The server settings of arm `name` (Phases 11-12), the same for the exam runner, the video replay and the F check.
    train_config = the NoMaD side's config loader (vint_train.mapmad.config.load_config: base_config chains), used for
    the arm's network config (map window numbers; mapmad_v2.yaml for arms without a v2 network) and V1's crop;
    recog = (host, port) of the recogniser sidecar (needed by `recogniser` arms); tau = arm F's tau (else the dev
    choice). Phase 11 configs (no `policies` block) get exactly Phase 11's settings."""
    cfg = table.cfg
    a = table.arm(name)
    dg = datagen_config(cfg)
    net_cfg = table.checkpoint(a.checkpoint).config if a.policy == "mapmad_v2" else table.resolve(V2_TRAIN_CONFIG)
    sv = search_settings(dg, train_config(str(net_cfg))["map"])
    extra = None
    if cfg.get("policies"):  # Phase 12
        run = cfg["run"]
        sv["rule"] = run["rule_stop"]
        target: Dict[str, Any] = {"source": a.target}
        if a.target == "recogniser":
            if recog is None:
                raise SystemExit(f"arm {name} needs a recogniser sidecar (--recog-port)")
            r = table.recogniser(a.recogniser)
            target["recogniser"] = {"host": recog[0], "port": int(recog[1]), "model": r["model"],
                                    "threshold": float(r["threshold"]), "p99": float(r["p99"]), "vlm": bool(r["vlm"])}
        sv["target"] = target
        if a.policy == "frontier":
            f = cfg["policies"]["frontier"]
            sv["teacher"] = {"spec": dg["teacher"], "seen": a.teacher_seen, "mode": a.teacher_mode,
                             "mark_radius_m": float(f["mark_radius_m"]), "min_views": int(run["rule_stop"]["min_views"]),
                             "tau": f_tau(table, tau) if (a.teacher_seen == "layer" and a.teacher_mode == "search")
                             else 1.0}
        if a.policy == "mapmad_v1":
            ms = cfg["map_settings"]
            extra = {"map": True, "map_settings": {"builder": read_config(config.CONFIG_DIR / ms["builder"])["map"],
                                                   "crop": train_config(str(table.resolve(ms["crop"])))["map"]}}
    cam = table.camera_name(a)
    camera = camera_wire(cam) if cam.endswith(".yaml") else {"name": cam, **cfg["cameras"][cam]}
    return ArmWire(sv, extra, camera)
