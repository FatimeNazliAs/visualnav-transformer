# deeper_visuals/common/config.py
"""
Resolve a phase's config.yaml into a PhaseConfig.

Each phase folder holds a tiny config.yaml:

    sample: gentle_left     # a key from common/samples.yaml, or {traj:…, frame:…}
    checkpoint: latest      # latest | ema

load_config() turns that into one frozen object carrying everything the phase's
run_model.py and the page builder need: the scene, which weights to use, the
frame indices, and the output directory. A phase-specific knob (P5's num_seeds)
is read by that phase straight from its own config.yaml; PhaseConfig previously
carried an `extra` dict for them that nothing ever read.

The output directory is tagged by checkpoint (out/p1/latest/, out/p1/ema/) so
switching `checkpoint:` never overwrites the other variant's PNGs — you can hold
both and compare.

Nothing here touches the filesystem under /outputs. The weight file and its run
folder are properties on PhaseConfig, resolved the first time something actually
asks for them — which in practice is only model.load_model. That keeps reading a
config free of the training run, so build_page.py and any test can run on a
machine with no GPU and no mounts, which is the whole point of the two-script
split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deeper_visuals.common import settings

# yaml is imported where it is used, not here, so that phase_for() — a pure
# string operation on a path — costs nothing but the standard library. update.sh
# calls it on the host before every run; a module-level import would put that
# behind PyYAML being installed there, and so behind a docker round-trip.

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
    checkpoint_tag: str         # "latest" | "ema" — validated at load time
    run: str                    # training run folder name, e.g. nomad_2026_…
    out_dir: Path               # where PNGs + facts.json are written

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

    # ── Weights — resolved on demand, never at load ───────────────────────────
    # These two are the only members that touch /outputs, and they are reached
    # exactly once, by model.load_model. Resolving them eagerly in load_config
    # would mean that *reading a config file* requires the 16 GB training run to
    # be mounted — which put build_page.py, the half of the pipeline explicitly
    # designed to run without a GPU, behind the same requirement as the forward
    # pass. Keeping them lazy is what lets pages (and tests) build on any
    # machine. The tag itself is still validated eagerly in config_from_dict,
    # because that is a dict lookup and a typo should fail immediately.

    @property
    def run_dir(self) -> Path:
        """The training run folder. Raises if it is not mounted."""
        run_dir = settings.RUNS_DIR / self.run
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"training run not found: {run_dir}. Is /outputs mounted? "
                f"Only the forward pass needs it — page builds do not."
            )
        return run_dir

    @property
    def checkpoint(self) -> Path:
        """
        The .pth the tag resolves to. Raises if it is missing.

        Two details worth not rediscovering:

        1. `latest.pth` is not a distinct checkpoint — train_eval_loop.py saves
           model.state_dict() to both {epoch}.pth and latest.pth back-to-back,
           so it is the same tensors as 99.pth.
        2. There is no ema_latest.pth to point at. train_eval_loop.py builds
           that path and prints "Saved EMA model to …" but never calls
           torch.save on it (upstream bug), which is why the EMA entry in
           settings.CHECKPOINT_FILES names the numbered file.
        """
        ckpt = self.run_dir / settings.CHECKPOINT_FILES[self.checkpoint_tag]
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt

    def weights_provenance(self) -> dict:
        """
        Which weights this phase is pinned to, as a facts.json "model" block.

        Touches .checkpoint, so it also proves the file is on disk — a phase
        that never loads the weights still fails here if /outputs is unmounted
        or the tag is wrong, instead of publishing a page that quietly cites a
        checkpoint nobody checked. Loaders extend this dict with what only
        loading can know (parameter count, device, key mismatches).
        """
        return {
            "checkpoint_tag":  self.checkpoint_tag,
            "checkpoint_file": self.checkpoint.name,
            "run":             self.run,
        }

    def summary(self) -> str:
        # Deliberately does not touch .checkpoint — summarising a config must
        # not require the weights to be present.
        return (
            f"phase={self.phase}  sample={self.sample_key} "
            f"({self.traj} @ f{self.frame})  checkpoint={self.checkpoint_tag} "
            f"({settings.CHECKPOINT_FILES[self.checkpoint_tag]})"
        )


def _load_samples() -> dict[str, dict]:
    import yaml

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


def _validate_checkpoint_tag(tag: str) -> str:
    """
    Normalise and check a `checkpoint:` value. Pure — no filesystem.

    The run is settled (see settings.DEFAULT_RUN), so this is a fixed two-entry
    map rather than a search:

        ema    -> ema_99.pth    the EMA weights at the final epoch. The default,
                                because EMA is what NoMaD's own evaluate_nomad
                                runs on and it gives cleaner advisor figures.
        latest -> latest.pth    raw epoch-99 weights, kept as the fallback.

    Whether those files actually exist is PhaseConfig.checkpoint's problem, and
    it only asks when something needs the weights.
    """
    tag = tag.lower()
    if tag not in settings.CHECKPOINT_FILES:
        raise ValueError(
            f"checkpoint must be one of "
            f"{' | '.join(sorted(settings.CHECKPOINT_FILES))}, got {tag!r}"
        )
    return tag


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
    tag = _validate_checkpoint_tag(str(raw.get("checkpoint", settings.DEFAULT_CHECKPOINT)))
    run = str(raw.get("run", settings.DEFAULT_RUN))

    return PhaseConfig(
        phase=phase,
        sample_key=sample_key,
        traj=traj,
        frame=frame,
        description=description,
        checkpoint_tag=tag,
        run=run,
        out_dir=OUT_ROOT / phase / tag,
    )


def read_raw(phase_dir: Path | str) -> dict:
    """
    The phase's config.yaml, parsed and otherwise untouched.

    Exposed because the keys PhaseConfig does not model are read by the phase
    that owns them — P4's `seed`, P5's `num_seeds` — and they need the same file
    and the same "no config.yaml here" message as everything else. The
    alternative was each phase opening the file a second time for one key, which
    is two readers with two error messages for one config.
    """
    phase_dir = Path(phase_dir).resolve()
    cfg_path = phase_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"no config.yaml in {phase_dir}")

    import yaml

    with open(cfg_path) as fh:
        return yaml.safe_load(fh) or {}


def load_config(phase_dir: Path | str, phase: str | None = None) -> PhaseConfig:
    """
    Read <phase_dir>/config.yaml and resolve it.

    `phase` defaults to the leading pN of the folder name, so p1_inputs -> "p1".
    """
    phase_dir = Path(phase_dir).resolve()
    return config_from_dict(read_raw(phase_dir), phase or phase_for(phase_dir))
