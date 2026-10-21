"""Layered MapMaD configs: a config may name `base_config: <file>` (relative to its own folder); the base is read
first (recursively) and this file's keys are merged on top, nested blocks key by key (e.g. one dataset's
`waypoint_spacing`). A block with `_replace: true` replaces the base's block instead (Phase 10: v2's `datasets` and
`habitat_modes` drop v1's entries). Used by train.py and the MapMaD tools; configs without `base_config` are
returned unchanged.
"""

import os
import subprocess
from typing import Any, Dict

import yaml

REPLACE_KEY = "_replace"


def git_commit() -> str:
    """Commit hash of the repo this file lives in ('unknown' outside git); saved next to every run's outputs."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:  # noqa: BLE001 - only a label
        return "unknown"


def map_model_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """NoMaD_ViNT's map arguments from a config: map_input, and with it the `map` block's tokens (1 = v1, 2 = v2)
    and layers (3 = v1, 4 = v2)."""
    if not config.get("map_input", False):
        return {"map_input": False}
    m = config.get("map", {})
    return {"map_input": True, "map_tokens": int(m.get("tokens", 1)), "map_channels": int(m.get("layers", 3))}


def deep_merge(base: Dict[str, Any], top: Dict[str, Any]) -> Dict[str, Any]:
    """base with top's keys merged in (dicts merged recursively, everything else replaced; a dict with
    `_replace: true` replaces the base's value, without that key)."""
    out = dict(base)
    for k, v in top.items():
        if isinstance(v, dict) and v.get(REPLACE_KEY):
            out[k] = {kk: vv for kk, vv in v.items() if kk != REPLACE_KEY}
        else:
            out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_base(path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """config (read from `path`) with its base_config chain merged underneath; the key itself is removed."""
    if "base_config" not in config:
        return config
    base_path = os.path.join(os.path.dirname(os.path.abspath(path)), config["base_config"])
    with open(base_path) as f:
        base = resolve_base(base_path, yaml.safe_load(f))
    return deep_merge(base, {k: v for k, v in config.items() if k != "base_config"})


def load_config(path: str, defaults: str = None) -> Dict[str, Any]:
    """defaults.yaml (if given) updated with the resolved config, as train.py builds it."""
    config: Dict[str, Any] = {}
    if defaults:
        with open(defaults) as f:
            config = yaml.safe_load(f)
    with open(path) as f:
        config.update(resolve_base(path, yaml.safe_load(f)))
    return config
