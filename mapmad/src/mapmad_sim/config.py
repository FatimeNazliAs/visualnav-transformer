"""Read MapMaD's config files: container paths (configs/paths.yaml) and the robot (configs/robot_limo.yaml)."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
SPLITS = ("minival", "val", "train")


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def paths(config: Optional[Path] = None) -> Dict[str, Path]:
    """All container paths from paths.yaml; MAPMAD_<NAME> environment variables win over the file."""
    config = Path(config or os.environ.get("MAPMAD_PATHS_CONFIG", CONFIG_DIR / "paths.yaml"))
    values = load_yaml(config)
    return {name: Path(os.environ.get(f"MAPMAD_{name.upper()}", value)) for name, value in values.items()}


def robot(config: Optional[Path] = None) -> Dict[str, Any]:
    """The LIMO facts (camera height, FOV, resolution, lidar) from robot_limo.yaml."""
    return load_yaml(Path(config or os.environ.get("MAPMAD_ROBOT_CONFIG", CONFIG_DIR / "robot_limo.yaml")))


def hm3d_scene_dataset_config(split: str, scenes: Optional[Path] = None) -> Path:
    """HM3D's own per-split dataset config, which ties each home's mesh to its semantic labels.

    Example: /hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json. Its paths are
    relative to its own folder, so it only works next to the homes (the generic
    hm3d_annotated_basis config in the same folder expects a different layout).
    """
    if split not in SPLITS:
        raise ValueError(f"unknown HM3D split {split!r}; expected one of {SPLITS}")
    scenes = scenes or paths()["hm3d_scenes"]
    return scenes / split / f"hm3d_annotated_{split}_basis.scene_dataset_config.json"


def hm3d_scene(split: str, home: str, scenes: Optional[Path] = None) -> Path:
    """The textured mesh of one home, e.g. hm3d_scene("minival", "00800-TEEsavR23oF")."""
    scenes = scenes or paths()["hm3d_scenes"]
    return scenes / split / home / f"{home.split('-', 1)[1]}.basis.glb"


def hm3d_sim_settings(split: str, home: str, scenes: Optional[Path] = None, **settings: Any):
    """habitat-starter SimSettings for one HM3D home, with the matching dataset config (labels line up).

    Example: hm3d_sim_settings("minival", "00800-TEEsavR23oF", width=320, height=240, semantic_sensor=True).
    Note: HM3D object boxes (obj.aabb) are only filled when the colour camera is on (the default).
    """
    from habitat_starter import SimSettings  # imported here: reading paths/robot config needs no habitat-sim

    return SimSettings(scene=str(hm3d_scene(split, home, scenes)),
                       scene_dataset_config=str(hm3d_scene_dataset_config(split, scenes)), **settings)


def labelled_homes(split: str, scenes: Optional[Path] = None) -> list:
    """Homes of a split that have semantic labels (a <id>.semantic.txt next to the mesh), sorted."""
    scenes = scenes or paths()["hm3d_scenes"]
    return sorted(p.parent.name for p in (scenes / split).glob("*/*.semantic.txt"))
