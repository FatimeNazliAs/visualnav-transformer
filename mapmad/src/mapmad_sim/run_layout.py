"""Where a closed-loop run's inputs and outputs live (Phase 1; Phase 5 reuses it with its own config file).

    layout = RunLayout.load()                     # configs/p1_baseline.yaml
    layout.episodes_file                          # <outputs>/p1_baseline/episodes/episodes.json
    layout.log("photo_iv", "p1-iv-00081-000")     # <outputs>/p1_baseline/logs/photo_iv/p1-iv-00081-000.jsonl

With a `subdir` (e.g. "checks/determinism_a") every run output moves below <out_dir>/<subdir>; the episode file
stays in <out_dir>/<episodes_dir> (config key, default "episodes"). Config paths are relative to paths.yaml
`outputs`, or to the repository when they start with "repo:" (`resolve`). A run config may start from another one (`extends: p1_baseline.yaml`): its blocks are
merged into that file's (nested keys replaced one by one; a key set to null is removed). Needs only PyYAML: used in
both containers.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from mapmad_sim import config

P1_CONFIG = config.CONFIG_DIR / "p1_baseline.yaml"
REPO = config.CONFIG_DIR.parents[1]


def merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """base with over's keys: dicts merged recursively, other values replaced, None removes the key."""
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def read_config(path: Path) -> Dict[str, Any]:
    """A run config; with `extends: <file>` (next to it) merged into that file (see the module docstring)."""
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("extends", None)
    return merge(read_config(path.parent / base), cfg) if base else cfg


RUN_SEP = "__s"  # Phase 5 run id: <episode_id>__s<seed index>


def run_id(episode_id: str, seed_index: int) -> str:
    """The Phase 5 run id of one (episode, seed index): the stem of its log, summary, crash and trace files."""
    return f"{episode_id}{RUN_SEP}{seed_index}"


def parse_run_id(name: str) -> Tuple[str, int]:
    """(episode id, seed index) of a run id or of one of its file names (e.g. `<id>__s0.crash.json`)."""
    episode_id, k = name.split(".", 1)[0].rsplit(RUN_SEP, 1)
    return episode_id, int(k)


def partial_path(log: Path) -> Path:
    """Where a log is written until its episode is complete (then renamed to `log`)."""
    return log.with_name(log.name + ".partial")


def write_text_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` so a reader (or a resumed run) sees either no file or the whole file: write a temporary
    file next to it, then rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def try_claim(path: Path) -> bool:
    """Phase 7 dynamic hand-out: create the claim file `path`; True only for the one runner that created it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


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
        return self.base / self.cfg.get("episodes_dir", "episodes")

    def resolve(self, value: str) -> Path:
        """A config path: "repo:<path>" -> the repository, otherwise below <outputs>."""
        return REPO / value[len("repo:"):] if value.startswith("repo:") else self.outputs / value

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

    def crash(self, arm: str, run: str) -> Path:
        """Phase 5: the record of a run that still failed after its retries (block item 13b)."""
        return self.root / "summaries" / arm / f"{run}.crash.json"

    def crashed_episodes(self, arms: Optional[Sequence[str]] = None) -> List[str]:
        """Phase 5: episode ids with a crash record in any of these arms (all arms by default), sorted."""
        files = [p for a in (arms or ["*"]) for p in (self.root / "summaries").glob(f"{a}/*.crash.json")]
        return sorted({parse_run_id(p.name)[0] for p in files})

    def claim(self, token: str, arm: str, run: str) -> Path:
        """Phase 7: the claim file of one run in one launch (`token`); a relaunch uses a new token, so claims of runs
        that a stopped launch never finished do not block them."""
        return self.root / "claims" / token / arm / run

    def trace(self, arm: str, run: str) -> Path:
        """Phase 5: the per-step map tensors and poses of one run (--trace-maps, map parity)."""
        return self.root / "traces" / arm / f"{run}.npz"

    def progress(self, tag: str) -> Path:
        return self.root / f"progress.{tag}.log"

    def video(self, arm: str, episode_id: str) -> Path:
        return self.root / "videos" / arm / f"{episode_id}.mp4"

    def timing(self, arm: str, shard: int) -> Path:
        return self.root / "timing" / f"{arm}.shard{shard}.jsonl"

    def anyside(self, arm: str, episode_id: str) -> Path:
        return self.root / "anyside" / arm / f"{episode_id}.json"

    def diagnostics(self, arm: str, episode_id: str, step: int) -> Path:
        return self.root / "diagnostics" / f"{arm}__{episode_id}__{step}"
