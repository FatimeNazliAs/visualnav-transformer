"""Where a closed-loop run's inputs and outputs live (Phase 1; Phase 5 reuses it with its own config file).

    layout = RunLayout.load()                     # configs/p1_baseline.yaml
    layout.episodes_file                          # <outputs>/p1_baseline/episodes/episodes.json
    layout.log("photo_iv", "p1-iv-00081-000")     # <outputs>/p1_baseline/logs/photo_iv/p1-iv-00081-000.jsonl

With a `subdir` (e.g. "checks/determinism_a") every run output moves below <out_dir>/<subdir>; the episode file
stays in <out_dir>/episodes. A run config may start from another one (`extends: p1_baseline.yaml`, top-level keys
replaced). Needs only PyYAML: used in both containers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from mapmad_sim import config

P1_CONFIG = config.CONFIG_DIR / "p1_baseline.yaml"


def read_config(path: Path) -> Dict[str, Any]:
    """A run config; with `extends: <file>` (next to it) that file's keys come first and these replace them."""
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("extends", None)
    return {**read_config(path.parent / base), **cfg} if base else cfg


@dataclass(frozen=True)
class RunLayout:
    cfg: Dict[str, Any]  # the run config (p1_baseline.yaml)
    base: Path  # <outputs>/<out_dir>
    root: Path  # base / subdir: run outputs
    outputs: Path  # <outputs> (paths.yaml)

    @classmethod
    def load(cls, path: Optional[Path] = None, subdir: str = "", outputs: Optional[Path] = None) -> "RunLayout":
        cfg = read_config(Path(path or P1_CONFIG))
        outputs = Path(outputs or config.paths()["outputs"])
        base = outputs / cfg["out_dir"]
        return cls(cfg=cfg, base=base, root=base / subdir, outputs=outputs)

    # --- inputs ------------------------------------------------------------------------------------------------
    @property
    def episodes_dir(self) -> Path:
        return self.base / "episodes"

    @property
    def episodes_file(self) -> Path:
        return self.episodes_dir / "episodes.json"

    @property
    def nomad_weights(self) -> Path:
        return self.outputs / self.cfg["nomad"]["weights"]

    @property
    def nomad_config(self) -> Path:
        return self.outputs / self.cfg["nomad"]["config"]

    # --- run outputs (below root) ------------------------------------------------------------------------------
    def log(self, arm: str, episode_id: str) -> Path:
        return self.root / "logs" / arm / f"{episode_id}.jsonl"

    def logs(self, arm: Optional[str] = None):
        """Every step log (of one arm), sorted."""
        return sorted((self.root / "logs").glob(f"{arm or '*'}/*.jsonl"))

    def summary(self, arm: str, episode_id: str) -> Path:
        return self.root / "summaries" / arm / f"{episode_id}.json"

    def video(self, arm: str, episode_id: str) -> Path:
        return self.root / "videos" / arm / f"{episode_id}.mp4"

    def timing(self, arm: str, shard: int) -> Path:
        return self.root / "timing" / f"{arm}.shard{shard}.jsonl"

    def anyside(self, arm: str, episode_id: str) -> Path:
        return self.root / "anyside" / arm / f"{episode_id}.json"

    def diagnostics(self, arm: str, episode_id: str, step: int) -> Path:
        return self.root / "diagnostics" / f"{arm}__{episode_id}__{step}"
