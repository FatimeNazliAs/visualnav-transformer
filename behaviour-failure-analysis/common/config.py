# behaviour-failure-analysis/common/config.py
"""
Resolve a task's config.yaml, and the cases it runs on.

This is the one module in common/ that is written rather than copied from
deeper_visuals. Its predecessor, PhaseConfig, was a single frozen object holding
both "which weights" and "which scene", because a visualization phase had
exactly one scene and ran it once. A behaviour analysis has one set of weights
and MANY cases, and it varies things a phase never varied — so the two halves
are split:

    ExperimentConfig   which weights, how many seeds, which conditions, where
                       results go. One per run.
    Case               one situation: a trajectory, a frame, a goal distance.
                       Many per run.

Both keep PhaseConfig's two best properties, because they were right:

  1. Nothing here touches /outputs at load time. The weight file and its run
     folder are resolved the first time something asks — in practice only
     common.model.load_model. Reading a config, listing cases and analysing
     saved results therefore all work on a machine with no GPU and no mounts.
  2. `weights_provenance()` touches .checkpoint, so a run that records its
     provenance has also proved the file is on disk.

Two things PhaseConfig conflated and this does not:

  * Goal distance vs prediction horizon. PhaseConfig's goal_idx was
    frame + NUM_ACTIONS. The head always emits NUM_ACTIONS waypoints, but the
    goal may sit anywhere in [GOAL_DISTANCE_MIN, GOAL_DISTANCE_MAX] — that is
    the range it was trained over. Case takes goal_distance explicitly.
  * Split membership. deeper_visuals never asked which split a scene came from,
    and nine of its ten curated scenes were training trajectories. Case
    validation refuses a trajectory that is not in the split the experiment
    declares.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import settings

# The three goal conditions an experiment can run. Navigation is the subject;
# the other two exist to answer "did the goal influence this prediction at
# all?", which no measurement of navigation alone can answer.
#
#   navigation   the goal token takes part in attention (the real behaviour)
#   exploration  the goal token is masked out of attention (NoMaD's own
#                undirected mode — the same checkpoint, GOAL_HIDDEN)
#   wrong_goal   the goal token is visible, but the goal image comes from a
#                DIFFERENT trajectory. Distinguishes "the model used the goal"
#                from "the model produced its usual path for this observation".
CONDITIONS = ("navigation", "exploration", "wrong_goal")


# ── Cases ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Case:
    """
    One situation to probe: a trajectory, a frame in it, and a goal distance.

    Deliberately holds no pixels and no model. It is a *selection*, small enough
    to sit in a YAML file, be listed in results, and be compared across runs.
    common.data.load_sample turns one into a Scene.

    `expectation` is free text recording why this case was chosen (what a human
    reading the frames expects to see). It is provenance, never an input to any
    measurement — a case is not labelled coherent or suspicious by its author.
    """

    case_id: str
    traj: str
    frame: int
    goal_distance: int = settings.NUM_ACTIONS
    expectation: str = ""

    def __post_init__(self) -> None:
        lo, hi = settings.GOAL_DISTANCE_MIN, settings.GOAL_DISTANCE_MAX
        if not lo <= self.goal_distance <= hi:
            raise ValueError(
                f"case {self.case_id!r}: goal_distance {self.goal_distance} is "
                f"outside the range the checkpoint was trained over ({lo}..{hi}). "
                f"Outside it the goal is out of distribution and any anomaly "
                f"says more about the sampling than about the model."
            )

    # ── Derived frame indices — one definition, so every task agrees ──────────
    @property
    def obs_idxs(self) -> List[int]:
        """The CONTEXT_SIZE+1 observation frames [t-3 … t] fed to encoder psi."""
        return list(range(self.frame - settings.CONTEXT_SIZE, self.frame + 1))

    @property
    def goal_idx(self) -> int:
        """The goal frame, `goal_distance` frames ahead of the current one."""
        return self.frame + self.goal_distance

    @property
    def horizon_idx(self) -> int:
        """
        The last frame the recorded ground-truth path reaches.

        NUM_ACTIONS ahead regardless of the goal — the head's output length is
        fixed by `len_traj_pred` and does not follow the goal.
        """
        return self.frame + settings.NUM_ACTIONS

    @property
    def traj_dir(self) -> Path:
        return settings.RAW_DATA_DIR / self.traj

    def to_dict(self) -> Dict[str, Any]:
        """The case as it appears in results and provenance files."""
        return {
            "case_id":       self.case_id,
            "traj":          self.traj,
            "frame":         self.frame,
            "goal_distance": self.goal_distance,
            "goal_idx":      self.goal_idx,
            "obs_idxs":      self.obs_idxs,
            "expectation":   self.expectation,
        }

    def summary(self) -> str:
        return (f"{self.case_id}: {self.traj} @ f{self.frame} "
                f"(goal +{self.goal_distance} -> f{self.goal_idx})")


# ── Splits ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def load_split(name: str) -> Tuple[str, ...]:
    """
    The trajectory names in a split, in file order.

    Cached because case validation asks once per case and the file is 740 or
    2956 lines. Returns a tuple so the cache cannot be mutated by a caller.
    """
    paths = {"train": settings.TRAIN_SPLIT, "test": settings.TEST_SPLIT}
    if name not in paths:
        raise ValueError(f"split must be train | test, got {name!r}")
    path = paths[name]
    if not path.is_file():
        raise FileNotFoundError(f"split list not found: {path}. Is /data mounted?")
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())


def split_of(traj: str) -> Optional[str]:
    """Which split a trajectory belongs to, or None if it is in neither."""
    for name in ("train", "test"):
        if traj in load_split(name):
            return name
    return None


# ── Experiment ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentConfig:
    """
    One run of one task: which weights, how it samples, and where it writes.

    Frozen, and every field lands in provenance.json, so a result can always be
    traced back to the exact settings that produced it.
    """

    task: str
    checkpoint_tag: str
    run: str
    split: str
    num_seeds: int
    goal_distance: int
    conditions: Tuple[str, ...]
    results_root: Path
    run_id: str

    @property
    def out_dir(self) -> Path:
        """Where this run writes. Tagged by run_id, so runs never overwrite."""
        return self.results_root / self.task / self.run_id

    @property
    def seeds(self) -> List[int]:
        """
        The seeds, 0 … num_seeds-1.

        Fixed and contiguous on purpose: the whole point of naming seeds is that
        a suspicious sample can be regenerated on its own, and a range makes the
        set reproducible from one number.
        """
        return list(range(self.num_seeds))

    # ── Weights — resolved on demand, never at load ───────────────────────────
    @property
    def run_dir(self) -> Path:
        """The training run folder. Raises if it is not mounted."""
        run_dir = settings.RUNS_DIR / self.run
        if not run_dir.is_dir():
            raise FileNotFoundError(
                f"training run not found: {run_dir}. Is /outputs mounted? "
                f"Only the forward pass needs it — analysis of saved results "
                f"does not."
            )
        return run_dir

    @property
    def checkpoint(self) -> Path:
        """
        The .pth the tag resolves to. Raises if it is missing.

        Two details worth not rediscovering, both verified on this checkpoint on
        2026-08-25 by comparing the zip entries inside the files:

        1. `latest.pth` is not a distinct checkpoint. Its 649 entries have
           byte-identical CRCs to 99.pth's. Their md5 sums and file sizes DO
           differ, but only because torch names the archive's root directory
           after the file ("latest/…" vs "99/…"); the weights are the same.
        2. There is no ema_latest.pth to point at. train_eval_loop.py builds
           that path and prints "Saved EMA model to …" but never calls
           torch.save on it (upstream bug), which is why the EMA entry in
           settings.CHECKPOINT_FILES names the numbered file.
        """
        ckpt = self.run_dir / settings.CHECKPOINT_FILES[self.checkpoint_tag]
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt

    def weights_provenance(self) -> Dict[str, Any]:
        """
        Which weights this run is pinned to, as a provenance block.

        Touches .checkpoint, so it also proves the file is on disk — a run that
        records its provenance cannot be citing a checkpoint nobody checked.
        common.model.load_model extends this with what only loading can know
        (parameter count, device, key mismatches).
        """
        return {
            "checkpoint_tag":  self.checkpoint_tag,
            "checkpoint_file": self.checkpoint.name,
            "run":             self.run,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Everything except the lazily-resolved weights, for provenance.json."""
        return {
            "task":          self.task,
            "run":           self.run,
            "checkpoint":    self.checkpoint_tag,
            "split":         self.split,
            "num_seeds":     self.num_seeds,
            "seeds":         self.seeds,
            "goal_distance": self.goal_distance,
            "conditions":    list(self.conditions),
            "results_root":  str(self.results_root),
            "run_id":        self.run_id,
            "out_dir":       str(self.out_dir),
        }

    def summary(self) -> str:
        # Deliberately does not touch .checkpoint — summarising must not require
        # the weights to be present.
        return (
            f"task={self.task}  checkpoint={self.checkpoint_tag} "
            f"({settings.CHECKPOINT_FILES[self.checkpoint_tag]})  split={self.split}  "
            f"seeds={self.num_seeds}  goal=+{self.goal_distance}  "
            f"conditions={','.join(self.conditions)}"
        )


# ── Loading ───────────────────────────────────────────────────────────────────

def _validate_checkpoint_tag(tag: str) -> str:
    """Normalise and check a `checkpoint:` value. Pure — no filesystem."""
    tag = tag.lower()
    if tag not in settings.CHECKPOINT_FILES:
        raise ValueError(
            f"checkpoint must be one of "
            f"{' | '.join(sorted(settings.CHECKPOINT_FILES))}, got {tag!r}"
        )
    return tag


def _validate_conditions(raw: Sequence[str]) -> Tuple[str, ...]:
    """Check every requested condition is one we know how to run."""
    conditions = tuple(str(c).lower() for c in raw)
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(
            f"unknown condition(s) {unknown}. Known: {', '.join(CONDITIONS)}"
        )
    if "navigation" not in conditions:
        raise ValueError(
            "conditions must include 'navigation' — the other two are controls "
            "that are only meaningful as a comparison against it."
        )
    return conditions


def read_yaml(path: Path) -> Dict[str, Any]:
    """Parse a YAML file, with a message that names it if it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such config file: {path}")

    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def make_run_id(stamp: Optional[str] = None) -> str:
    """
    A run id: UTC timestamp to the second.

    Every run gets its own output directory rather than overwriting the last,
    because comparing a result against the previous one is the whole workflow
    and an overwritten result cannot be compared with anything.
    """
    if stamp:
        return stamp
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_experiment(task_dir: Path, *, run_id: Optional[str] = None) -> ExperimentConfig:
    """Read <task_dir>/config.yaml and resolve it into an ExperimentConfig."""
    task_dir = Path(task_dir).resolve()
    raw = read_yaml(task_dir / "config.yaml")

    split = str(raw.get("split", "test")).lower()
    if split not in ("train", "test"):
        raise ValueError(f"split must be train | test, got {split!r}")

    goal_distance = int(raw.get("goal_distance", settings.NUM_ACTIONS))
    lo, hi = settings.GOAL_DISTANCE_MIN, settings.GOAL_DISTANCE_MAX
    if not lo <= goal_distance <= hi:
        raise ValueError(f"goal_distance must be in {lo}..{hi}, got {goal_distance}")

    num_seeds = int(raw.get("num_seeds", 32))
    if num_seeds < 2:
        raise ValueError(
            f"num_seeds must be at least 2, got {num_seeds}. Every measurement "
            f"here is about the spread across samples; one sample has none."
        )

    return ExperimentConfig(
        task=task_dir.name,
        checkpoint_tag=_validate_checkpoint_tag(
            str(raw.get("checkpoint", settings.DEFAULT_CHECKPOINT))),
        run=str(raw.get("run", settings.DEFAULT_RUN)),
        split=split,
        num_seeds=num_seeds,
        goal_distance=goal_distance,
        conditions=_validate_conditions(raw.get("conditions", CONDITIONS)),
        results_root=Path(str(raw.get("results_root", settings.RESULTS_ROOT))),
        run_id=make_run_id(run_id),
    )


def load_cases(task_dir: Path, cfg: ExperimentConfig, *,
               validate_split: bool = True) -> List[Case]:
    """
    Read <task_dir>/cases.yaml into Cases, checking each against cfg.split.

    The split check is the reason this function exists rather than a list
    comprehension at the call site. A case silently drawn from `train` is the
    single failure mode that would invalidate every number this project
    produces, and it is invisible in the results — the figures look identical.
    So it fails here, loudly, before the model is ever loaded.
    """
    task_dir = Path(task_dir).resolve()
    raw = read_yaml(task_dir / "cases.yaml")
    entries = raw.get("cases", raw if isinstance(raw, list) else [])
    if not entries:
        raise ValueError(f"no cases in {task_dir / 'cases.yaml'}")

    cases = [
        Case(
            case_id=str(entry["case_id"]),
            traj=str(entry["traj"]),
            frame=int(entry["frame"]),
            goal_distance=int(entry.get("goal_distance", cfg.goal_distance)),
            expectation=str(entry.get("expectation", "")),
        )
        for entry in entries
    ]

    duplicates = {c.case_id for c in cases if
                  sum(1 for o in cases if o.case_id == c.case_id) > 1}
    if duplicates:
        raise ValueError(f"duplicate case_id(s) in cases.yaml: {sorted(duplicates)}")

    if validate_split:
        misplaced = [(c.case_id, c.traj, split_of(c.traj)) for c in cases
                     if c.traj not in load_split(cfg.split)]
        if misplaced:
            lines = "\n".join(
                f"    {cid}: {traj} is in {found or 'NEITHER split'}"
                for cid, traj, found in misplaced)
            raise ValueError(
                f"cases.yaml declares split={cfg.split}, but these are not in it:\n"
                f"{lines}\n"
                f"  A case drawn from the training split measures memorisation, "
                f"not behaviour. Pick from {getattr(settings, cfg.split.upper() + '_SPLIT')}."
            )

    return cases
