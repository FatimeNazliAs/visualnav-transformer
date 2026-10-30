"""Read MapMaD's config files: container paths (configs/paths.yaml) and the robot (configs/robot_limo.yaml)."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
SPLITS = ("minival", "val", "train")
MP3D_SPLIT = "mp3d"  # Phase 13: every MP3D home runs with split "mp3d" and home = its scene id (e.g. "2azQ1b91cZZ")
MP3D_DATASET_CONFIG = CONFIG_DIR / "mp3d.scene_dataset_config.json"  # habitat-starter's: semantic frames line up


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


def is_mp3d(split: str) -> bool:
    return split == MP3D_SPLIT


def scene_id(home: str) -> str:
    """The scene id of a home: HM3D "00800-TEEsavR23oF" -> "TEEsavR23oF"; an MP3D home is its scene id already."""
    return home.split("-", 1)[1] if "-" in home else home


def mp3d_scene(home: str, scenes: Optional[Path] = None) -> Path:
    """The mesh of one MP3D home, e.g. /mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb (+ .house and _semantic.ply next to it)."""
    scenes = scenes or paths()["mp3d_scenes"]
    return scenes / home / f"{home}.glb"


def sim_settings(split: str, home: str, **settings: Any):
    """SimSettings of one home: HM3D (hm3d_sim_settings) or, for split "mp3d", the MP3D mesh with our copy of
    habitat-starter's mp3d.scene_dataset_config.json (without it the semantic picture does not line up with the
    colour and depth pictures: Phase 13 check, 161 of 162 official view points see their goal with it, 10 without).
    The shipped MP3D .navmesh is never used: LimoSim always rebuilds the floor map for LIMO."""
    if not is_mp3d(split):
        return hm3d_sim_settings(split, home, **settings)
    from habitat_starter import SimSettings

    return SimSettings(scene=str(mp3d_scene(home)), scene_dataset_config=str(MP3D_DATASET_CONFIG), **settings)


def labelled_homes(split: str, scenes: Optional[Path] = None) -> list:
    """Homes of a split that have semantic labels (a <id>.semantic.txt next to the mesh), sorted."""
    scenes = scenes or paths()["hm3d_scenes"]
    return sorted(p.parent.name for p in (scenes / split).glob("*/*.semantic.txt"))
