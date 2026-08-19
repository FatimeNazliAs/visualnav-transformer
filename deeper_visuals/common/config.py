# deeper_visuals/common/config.py
"""
Resolve a phase's config.yaml into a PhaseConfig.

Each phase folder holds a tiny config.yaml:

    sample: gentle_left     # a key from common/samples.yaml, or {traj:…, frame:…}
    checkpoint: latest      # latest | ema
    # …any phase-specific knobs, e.g. num_seeds: 6

load_config() turns that into one frozen object carrying everything the phase's
run_model.py and the page builder need: the scene, the resolved weight file, the
frame indices, and the output directory. Phase-specific knobs stay reachable
through .extra so this module never needs to know about them.

The output directory is tagged by checkpoint (out/p1/latest/, out/p1/ema/) so
switching `checkpoint:` never overwrites the other variant's PNGs — you can hold
both and compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from deeper_visuals.common import settings

COMMON_DIR   = Path(__file__).resolve().parent
SAMPLES_PATH = COMMON_DIR / "samples.yaml"
# deeper_visuals/out/<phase>/<ckpt_tag>/ — generated, gitignored.
OUT_ROOT     = COMMON_DIR.parent / "out"


@dataclass(frozen=True)
class PhaseConfig:
    """Everything a phase needs, resolved from its config.yaml."""

    phase: str                  # "p1", "p3", …  (used for out/ paths and titles)
    sample_key: str             # "gentle_left", or "custom" for an inline scene
    traj: str                   # go_stanford trajectory folder name
    frame: int                  # the current frame, t
    description: str            # human-readable scene description
    checkpoint_tag: str         # "latest" | "ema"
    checkpoint: Path            # the actual .pth resolved from the tag
    run_dir: Path               # the training run folder the weights came from
    out_dir: Path               # where PNGs + facts.json are written
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Derived frame indices — one definition, so every phase agrees ─────────
    @property
    def obs_idxs(self) -> list[int]:
        """The CONTEXT_SIZE+1 observation frames [t-3 … t] fed to encoder psi."""
        return list(range(self.frame - settings.CONTEXT_SIZE, self.frame + 1))

    @property
    def goal_idx(self) -> int:
        """The goal frame t+NUM_ACTIONS fed to encoder phi."""
        return self.frame + settings.NUM_ACTIONS

    @property
    def traj_dir(self) -> Path:
        return settings.RAW_DATA_DIR / self.traj

    def summary(self) -> str:
        return (
            f"phase={self.phase}  sample={self.sample_key} "
            f"({self.traj} @ f{self.frame})  checkpoint={self.checkpoint_tag} "
            f"({self.checkpoint.name})"
        )


def _load_samples() -> dict[str, dict]:
    with open(SAMPLES_PATH) as fh:
        return yaml.safe_load(fh)


def _resolve_sample(spec: Any) -> tuple[str, str, int, str]:
    """
    Accept either a library key ("gentle_left") or an inline scene
    ({traj: …, frame: …}). Returns (sample_key, traj, frame, description).
    """
    if isinstance(spec, dict):
        if "traj" not in spec or "frame" not in spec:
            raise ValueError(
                f"inline sample needs both 'traj' and 'frame', got: {spec}"
            )
        return (
            "custom",
            str(spec["traj"]),
            int(spec["frame"]),
            str(spec.get("description", "inline sample (not from samples.yaml)")),
        )

    samples = _load_samples()
    if spec not in samples:
        raise KeyError(
            f"unknown sample {spec!r}. Known scenes: {', '.join(sorted(samples))}. "
            f"Or inline one as {{traj: …, frame: …}}."
        )
    s = samples[spec]
    return str(spec), str(s["traj"]), int(s["frame"]), str(s["description"])


def _resolve_checkpoint(tag: str, run: str) -> tuple[Path, Path]:
    """
    Map a checkpoint tag onto a real weight file inside the run folder.

    The run is settled (see settings.DEFAULT_RUN), so this is a fixed two-entry
    map rather than a search:

        ema    -> ema_99.pth    the EMA weights at the final epoch. The default,
                                because EMA is what NoMaD's own evaluate_nomad
                                runs on and it gives cleaner advisor figures.
        latest -> latest.pth    raw epoch-99 weights, kept as the fallback.

    Two details worth not rediscovering:

    1. `latest.pth` is not a distinct checkpoint — train_eval_loop.py saves
       model.state_dict() to both {epoch}.pth and latest.pth back-to-back, so it
       is the same tensors as 99.pth.
    2. There is no ema_latest.pth to point at. train_eval_loop.py builds that
       path and prints "Saved EMA model to …" but never calls torch.save on it
       (upstream bug), which is why the EMA entry names the numbered file.
    """
    run_dir = settings.RUNS_DIR / run
    if not run_dir.is_dir():
        raise FileNotFoundError(f"training run not found: {run_dir}")

    tag = tag.lower()
    if tag not in settings.CHECKPOINT_FILES:
        raise ValueError(
            f"checkpoint must be one of "
            f"{' | '.join(sorted(settings.CHECKPOINT_FILES))}, got {tag!r}"
        )

    ckpt = run_dir / settings.CHECKPOINT_FILES[tag]
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    return ckpt, run_dir


def phase_for(phase_dir: Path | str) -> str:
    """
    The phase id for a folder: everything before the first underscore, so
    p1_inputs -> "p1" and p3_transformer_masking -> "p3".

    Every caller derives it through here — run_model.py, build_page.py and
    update.sh — because two rules that disagree send the forward pass and the
    page builder to different out/ directories, which fails confusingly.
    """
    return Path(phase_dir).name.split("_")[0]


def config_from_dict(raw: dict, phase: str) -> PhaseConfig:
    """
    Resolve an already-parsed config mapping. Split out from load_config so
    callers without a phase folder (common/smoke_test.py) share the exact same
    resolution rules rather than reimplementing them.
    """
    sample_key, traj, frame, description = _resolve_sample(raw.get("sample", "gentle_left"))
    tag = str(raw.get("checkpoint", settings.DEFAULT_CHECKPOINT))
    run = str(raw.get("run", settings.DEFAULT_RUN))
    ckpt, run_dir = _resolve_checkpoint(tag, run)

    # Anything not consumed above is a phase-specific knob (e.g. num_seeds).
    extra = {k: v for k, v in raw.items() if k not in {"sample", "checkpoint", "run"}}

    return PhaseConfig(
        phase=phase,
        sample_key=sample_key,
        traj=traj,
        frame=frame,
        description=description,
        checkpoint_tag=tag,
        checkpoint=ckpt,
        run_dir=run_dir,
        out_dir=OUT_ROOT / phase / tag,
        extra=extra,
    )


def load_config(phase_dir: Path | str, phase: str | None = None) -> PhaseConfig:
    """
    Read <phase_dir>/config.yaml and resolve it.

    `phase` defaults to the leading pN of the folder name, so p1_inputs -> "p1".
    """
    phase_dir = Path(phase_dir).resolve()
    cfg_path = phase_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"no config.yaml in {phase_dir}")

    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh) or {}

    return config_from_dict(raw, phase or phase_for(phase_dir))
