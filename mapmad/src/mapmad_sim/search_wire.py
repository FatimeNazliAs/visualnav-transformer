"""The `search_v2` settings a closed-loop client sends to the simulator server at reset (Phase 11; sidecar_server,
search_v2.SearchSpec.from_wire). Pure Python + PyYAML, so the NoMaD container builds them too (search_v2 itself needs
the Habitat side's packages)."""

from typing import Any, Dict

from mapmad_sim import config
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
