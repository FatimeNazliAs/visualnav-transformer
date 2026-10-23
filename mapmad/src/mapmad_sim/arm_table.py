"""One arm table (Phase 12 sub-step 12.0, architecture candidate 2; Phases 11-12): what an arm IS, answered in one place.

    table = ArmTable.from_layout(layout)           # layout = run_layout.RunLayout of a phase config
    arm = table.arm("M2")                          # Arm(policy, checkpoint key, maps, target layer, recogniser, ...)
    ck = table.checkpoint(arm.checkpoint)          # Checkpoint(path, sha256, config): absolute in THIS container
    table.seeds("M2h", [0, 1, 2])                  # the seed indices the arm runs

An arm (config `arms:` entry) = a policy + its weights + what it is shown + what draws its map's target layer:
- policy: `mapmad_v2` (MapMaDV2Inference, Phase 5 MapMaD settings), `nomad` (official NoMaD explore, Phase 5 arm A
  settings), `mapmad_v1` (Phase 4 checkpoint, one map token, Phase 5 MapMaD settings), `frontier` (arm F: the explorer
  teacher on the robot's own map, steered as the v1 planner arm; no weights, deterministic);
- checkpoint: a key of `checkpoints:` (none for `frontier`);
- maps: MapMaD v2 sees its near + far windows (False = both hidden, NoMaD's own computation);
- map_zero: v1 map layers given as 0 (V1: ["heat"]);
- target: what draws the online map's "target seen here" layer, which every arm's map has (the rule stop reads it):
  `truth_noisy` (simulator truth with the Phase 8 noise, Phase 11), `recogniser` (a `recognisers:` entry through the
  sidecar), `none` (no layer: exploration, or a camera the layer cannot be drawn with);
- camera: a `cameras:` entry (None = the config's `camera`, pin025);
- seeds: the seed indices it runs (None = all of `run.seeds`); `seeds_not_headline`: the seeds when the arm is not
  part of the headline pair (Phase 12 block item 3: M2-h 3 seeds if headline, else 1);
- teacher_seen (frontier only): `layer` (a target-layer cell >= tau seen from >= 2 views) or `truth` (Phase 8's
  simulator-truth "seen", the F check); teacher_mode `search` | `explore`.
A checkpoint (config `checkpoints:` entry) is {path, sha256, config} (paths below <outputs> or "repo:"), or
{from_file: <json below out_dir>} (written by an earlier step: V2_FINAL.json, V21.json), or {from_choice: <key of
P12_DEV_CHOICES.json>} (the dev decision names another checkpoint key, e.g. headline -> "v21"). Paths are resolved in
the container that asks, so no caller ever builds a checkpoint path. Pure Python + PyYAML: both containers use it.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

POLICIES = ("mapmad_v2", "nomad", "mapmad_v1", "frontier")
TARGETS = ("truth_noisy", "recogniser", "none")
CHOICES_FILE = "P12_DEV_CHOICES.json"  # Phase 12: the dev decisions (headline, deltas, tau), below out_dir


@dataclass(frozen=True)
class Checkpoint:
    key: str
    path: Path
    sha256: str
    config: Path


@dataclass(frozen=True)
class Arm:
    """One arm of the table (module docstring)."""

    name: str
    policy: str = "mapmad_v2"
    checkpoint: Optional[str] = None
    maps: bool = True
    map_zero: Tuple[str, ...] = ()
    target: str = "truth_noisy"
    recogniser: Optional[str] = None
    camera: Optional[str] = None
    seeds: Optional[Tuple[int, ...]] = None
    seeds_not_headline: Optional[Tuple[int, ...]] = None
    teacher_seen: str = "layer"
    teacher_mode: str = "search"
    label: str = ""
    pair: Optional[str] = None  # Phase 12: the checkpoint pair this arm belongs to (v2 | v21), for the headline rule

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"arm {self.name}: policy {self.policy!r} not in {POLICIES}")
        if self.target not in TARGETS:
            raise ValueError(f"arm {self.name}: target {self.target!r} not in {TARGETS}")
        if (self.target == "recogniser") != (self.recogniser is not None):
            raise ValueError(f"arm {self.name}: target 'recogniser' needs a recogniser (and only it)")
        if (self.policy == "frontier") == (self.checkpoint is not None):
            raise ValueError(f"arm {self.name}: every policy but 'frontier' needs a checkpoint")
        if self.policy == "frontier" and self.target == "none" and self.teacher_mode == "search":
            raise ValueError(f"arm {self.name}: a searching frontier arm needs a target layer")
        if self.teacher_seen not in ("layer", "truth") or self.teacher_mode not in ("search", "explore"):
            raise ValueError(f"arm {self.name}: teacher_seen layer|truth, teacher_mode search|explore")
        if "__" in self.name or "." in self.name or "/" in self.name:
            raise ValueError(f"arm {self.name}: no '__', '.' or '/' in an arm name (run ids and file names)")

    @classmethod
    def from_config(cls, name: str, spec: Dict[str, Any]) -> "Arm":
        s = dict(spec)
        for k in ("seeds", "seeds_not_headline", "map_zero"):
            if s.get(k) is not None:
                s[k] = tuple(s[k])
        return cls(name=name, **s)

    @property
    def deterministic(self) -> bool:
        """No random numbers: its seed-index-0 run stands for every seed index (Phase 5 planner rule)."""
        return self.policy == "frontier"


@dataclass
class ArmTable:
    """The arms, checkpoints and recognisers of one phase config (module docstring)."""

    cfg: Dict[str, Any]
    base: Path  # <outputs>/<out_dir>
    outputs: Path
    repo: Path
    arms: Dict[str, Arm] = field(default_factory=dict)

    @classmethod
    def from_layout(cls, layout: Any) -> "ArmTable":
        from mapmad_sim.run_layout import REPO

        return cls.from_config(layout.cfg, layout.base, layout.outputs, REPO)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], base: Path, outputs: Path, repo: Path) -> "ArmTable":
        arms = {n: Arm.from_config(n, s) for n, s in (cfg.get("arms") or {}).items()}
        return cls(cfg, Path(base), Path(outputs), Path(repo), arms)

    # --- arms ---------------------------------------------------------------------------------------------------
    def arm(self, name: str) -> Arm:
        if name not in self.arms:
            raise KeyError(f"unknown arm {name!r} (known: {sorted(self.arms)})")
        return self.arms[name]

    def choices(self) -> Optional[Dict[str, Any]]:
        """P12_DEV_CHOICES.json below out_dir, or None before the dev decisions."""
        f = self.base / CHOICES_FILE
        return json.loads(f.read_text()) if f.exists() else None

    def is_headline(self, arm: Arm) -> bool:
        """The arm belongs to the headline pair (P12_DEV_CHOICES.json `headline`; False before it exists)."""
        ch = self.choices()
        return ch is not None and arm.pair is not None and arm.pair == ch["headline"]

    def seeds(self, name: str, run_seeds: Sequence[int]) -> List[int]:
        """The seed indices this arm runs, out of run_seeds: its own `seeds` (or `seeds_not_headline` when it is not
        in the headline pair); a deterministic arm runs index 0 only."""
        a = self.arm(name)
        own = a.seeds
        if a.seeds_not_headline is not None and self.choices() is not None and not self.is_headline(a):
            own = a.seeds_not_headline
        out = [k for k in run_seeds if own is None or k in own]
        return [k for k in out if k == 0] if a.deterministic else out

    # --- checkpoints --------------------------------------------------------------------------------------------
    def resolve(self, value: str) -> Path:
        """A config path: "repo:<path>" -> the repository, "/..." -> as is, otherwise below <outputs>."""
        if value.startswith("repo:"):
            return self.repo / value[len("repo:"):]
        return Path(value) if value.startswith("/") else self.outputs / value

    def checkpoint_entry(self, key: str) -> Tuple[str, Dict[str, Any]]:
        """(final key, {path, sha256, config}) of a checkpoint key, following from_choice / from_file."""
        seen = []
        while True:
            if key in seen:
                raise ValueError(f"checkpoint keys loop: {seen + [key]}")
            seen.append(key)
            c = self.cfg["checkpoints"][key]
            if "from_choice" in c:
                ch = self.choices()
                if ch is None:
                    raise SystemExit(f"checkpoint {key}: {self.base / CHOICES_FILE} does not exist yet")
                key = ch[c["from_choice"]]
                continue
            if "from_file" in c:
                f = self.base / c["from_file"]  # written by an earlier step of this phase, in its own output folder
                if not f.exists():
                    raise SystemExit(f"checkpoint {key}: {f} does not exist yet")
                c = json.loads(f.read_text())
            return key, c

    def checkpoint(self, key: str) -> Checkpoint:
        final, c = self.checkpoint_entry(key)
        return Checkpoint(final, self.resolve(c["path"]), c["sha256"], self.resolve(c["config"]))

    def verify(self, key: str) -> Checkpoint:
        """The checkpoint, after checking its file's sha256 (a mismatch raises: automatic stop)."""
        ck = self.checkpoint(key)
        got = file_sha256(ck.path)
        if got != ck.sha256:
            raise RuntimeError(f"checkpoint {key}: {ck.path} sha256 {got} != {ck.sha256} -- not the frozen checkpoint")
        return ck

    # --- recognisers and cameras ----------------------------------------------------------------------------------
    def recogniser(self, name: str) -> Dict[str, Any]:
        """A `recognisers:` entry: {model (sidecar key), threshold, p99, vlm}."""
        return dict(self.cfg["recognisers"][name], name=name)

    def camera_name(self, arm: Arm) -> str:
        """The camera file / entry of the arm (the config's `camera` by default)."""
        return arm.camera or self.cfg["camera"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
