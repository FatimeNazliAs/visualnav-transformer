"""The task set: one fixed, seeded list of navigation problems, built once.

The fairness protocol (plan §7) is a single sentence — *identical task set
across every checkpoint* — and this module is what makes it true rather than
intended. A task set is built once, written to disk with the seeds that
produced it, and every arm afterwards is pointed at that directory. Nothing
re-samples at evaluation time, so no arm can face an easier draw than another.

A **task** is one of P2's reference-path topomaps plus its metadata: a start
pose to be dropped at, a trail of goal images to follow, a goal pose success is
measured from, and the geodesic length SPL divides by. Building one is P2's
job, unchanged — this only decides *which* ones exist, and remembers.

Two properties are worth knowing, because both are load-bearing:

  * **A task's seed is a pure function of its scene and its index**, never of
    how many tasks were asked for. So a three-task set is the first three tasks
    of a ten-task set, exactly — which is what lets `p3_1_score_test.py` prove
    the pipeline on a small set that is genuinely part of the real one.
  * **A built task set is immutable.** The config that produced it is
    fingerprinted into the manifest, and a build against a changed config
    refuses rather than quietly rebuilding. Half a table scored on one task set
    and half on another is not a comparison, and it is invisible in the CSV.
"""

import hashlib
import json
import shutil
from pathlib import Path

import yaml

import topomap_builder
from topomap_builder import TopomapConfig

SIM_EVAL_DIR = Path(__file__).resolve().parent
MANIFEST_NAME = "manifest.json"

# Seeds are laid out `base + scene_index * SEEDS_PER_SCENE + task_index`, so
# every scene gets its own block and no two tasks can collide however many are
# asked for. It also keeps the seed readable: task 3 of scene 1 at base 1000 is
# 2003, not a hash.
SEEDS_PER_SCENE = 1000


class TaskSetError(RuntimeError):
    """Raised when a task set cannot be built, or would not be the same one."""


def task_seed(base_seed, scene_index, task_index):
    """The seed for one task — see SEEDS_PER_SCENE."""
    return int(base_seed) + int(scene_index) * SEEDS_PER_SCENE + int(task_index)


def task_id(scene, task_index):
    """`Rs_00` — sorts in build order and says which house it is in."""
    return "{}_{:02d}".format(scene, task_index)


class Task:
    """One navigation problem: where to start, what trail to follow, where it ends.

    A thin reader over a topomap directory. The frames are loaded on demand
    rather than held, because a task set is twenty of these and an episode
    needs one at a time.
    """

    def __init__(self, task_id, scene, directory, seed, metadata=None):
        self.task_id = task_id
        self.scene = scene
        self.directory = Path(directory)
        self.seed = int(seed)
        self._metadata = metadata

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = topomap_builder.load_metadata(self.directory)
        return self._metadata

    @property
    def world_config(self):
        """The world this trail was driven in, pinned beside it by P2."""
        return self.directory / topomap_builder.WORLD_CONFIG_NAME

    @property
    def geodesic_length_m(self):
        """The shortest path from start to goal — SPL's numerator, and the
        length the timeout is tied to."""
        return float(self.metadata["geodesic_length_m"])

    @property
    def goal_xy(self):
        """Where success is measured from: the pose the goal image was taken at.

        Not `planned_goal_xy`. The planner's target is a point on a grid the
        reference drive stopped near; the goal *image* at the end of the trail —
        the one the model is actually chasing — was rendered from this pose.
        """
        goal = self.metadata["goal_pose"]
        return [float(goal["x"]), float(goal["y"])]

    @property
    def node_count(self):
        return len(self.metadata["nodes"])

    def start_pose(self, floor_height):
        """(position, orientation) for `SimBody.place` — the reference start.

        The floor's height is not in the metadata because it belongs to the
        scene, not to the task; the caller reads it from the open scene, as P2
        does when it places the robot for the reference drive.
        """
        start = self.metadata["start_pose"]
        return ([float(start["x"]), float(start["y"]), float(floor_height)],
                [0.0, 0.0, float(start["yaw"])])

    def topomap(self):
        """The trail, as the bridge's loader reads it."""
        import bridge

        return bridge.load_topomap(self.directory)

    def summary(self):
        return "{:<8} {:<4} geodesic {:.2f} m · {} nodes · seed {}".format(
            self.task_id, self.scene, self.geodesic_length_m,
            self.node_count, self.seed)


class TaskSetConfig:
    """Which tasks exist: the scenes, how many per scene, and the seed they start at.

    The trails themselves are built with P2's knobs, read from
    `topomap_config` and amended by `overrides`. Pointing at P2's file rather
    than copying its contents means there is one definition of what a trail is;
    the fingerprint below is what stops that sharing from becoming a way for a
    task set to change under a half-finished comparison.
    """

    def __init__(self, topomap_config, scenes, tasks_per_scene=10, base_seed=1000,
                 overrides=None, adopt_from=None):
        self.topomap_config = Path(topomap_config)
        if not self.topomap_config.is_absolute():
            self.topomap_config = SIM_EVAL_DIR / self.topomap_config
        self.scenes = list(scenes)
        self.tasks_per_scene = int(tasks_per_scene)
        self.base_seed = int(base_seed)
        self.overrides = dict(overrides or {})
        # Built task sets whose trails this one may take over rather than
        # drive again — see `adoptable`. Provenance, not identity: an adopted
        # task is the task this config would build, so it is left out of the
        # fingerprint, and the manifest records where each one came from.
        self.adopt_from = [_resolve(directory) for directory in (adopt_from or [])]

        if not self.scenes:
            raise TaskSetError("a task set needs at least one scene")
        if self.tasks_per_scene < 1:
            raise TaskSetError("tasks_per_scene must be at least 1")

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def adoptable(self, topomap_config, identifier):
        """(source directory, manifest entry) of a built task that is exactly
        the one `topomap_config` describes, or None.

        Exactly, and checked three ways, because an adopted trail is scored as
        if this config had driven it:

          * the source set was built from the same P2 knobs (its manifest
            records them all, `max_ticks` included, which a task's own metadata
            does not);
          * it has this task id under this task's seed;
          * the world saved beside the trail is the world this task loads.

        The build is deterministic in the seed (`topomap_builder.build_topomap`),
        so a task that passes all three is the trail a rebuild would drive.
        """
        knobs = _normalized(self.base_knobs())
        for source in self.adopt_from:
            manifest, _tasks = load(source)
            if _normalized(manifest["topomap_config"]) != knobs:
                continue
            for entry in manifest["tasks"]:
                if entry["task_id"] != identifier or entry["seed"] != topomap_config.seed:
                    continue
                world = source / entry["directory"] / topomap_builder.WORLD_CONFIG_NAME
                if world.exists() and yaml.safe_load(world.read_text()) == \
                        topomap_config.world_config():
                    return source, entry
        return None

    def base_knobs(self):
        """P2's topomap knobs, with this task set's overrides applied."""
        with open(self.topomap_config, "r") as handle:
            knobs = yaml.safe_load(handle)
        knobs.update(self.overrides)
        if knobs.get("start") is not None:
            raise TaskSetError(
                "{} pins a start/goal pair, so every task in the set would be "
                "the same trail. A task set samples its tasks — set start and "
                "goal to null, or override them here."
                .format(self.topomap_config))
        return knobs

    def topomap_config_for(self, scene, scene_index, task_index):
        """P2's config for one task: this scene, this task's seed."""
        knobs = self.base_knobs()
        knobs["scene_id"] = scene
        knobs["seed"] = task_seed(self.base_seed, scene_index, task_index)
        return TopomapConfig(**knobs)

    def world(self):
        """The iGibson world every task is driven in, as its contents.

        `scene_id` is left as the file has it: each task overrides it with its
        own scene, and the scene list is hashed separately.
        """
        return TopomapConfig(**self.base_knobs()).world_config()

    def fingerprint(self):
        """A hash of everything that decides which tasks exist.

        Covers P2's knobs as well as the scene list, because the trails are
        half of what a task is: the same start/goal pair at a different node
        spacing is a different problem. If this changes, the task set on disk
        is not the one this config describes, and the build says so instead of
        silently producing a second, incomparable set.

        It covers the **world file's contents**, not just its path. The camera,
        the robot and the physics are the other half of a task: every trail
        image is rendered through that camera, and every episode reopens the
        copy of the world saved beside its trail. Hashing only the path, as
        this did until the P5 review, meant an edit to `vertical_fov` in
        `configs/locobot_rs_bridge.yaml` matched the old fingerprint — so an
        existing set was silently reused at the old camera, and a new set was
        indistinguishable from it.
        """
        payload = {
            "scenes": self.scenes,
            "tasks_per_scene": self.tasks_per_scene,
            "base_seed": self.base_seed,
            "topomap": self.base_knobs(),
            "world": self.world(),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def summary(self):
        return "{} x {} tasks from base seed {} ({})".format(
            ", ".join(self.scenes), self.tasks_per_scene, self.base_seed,
            self.topomap_config.name)


def _resolve(directory):
    directory = Path(directory)
    return directory if directory.is_absolute() else SIM_EVAL_DIR / directory


def _normalized(knobs):
    """Knobs as JSON would store them, so a manifest compares with a config."""
    return json.loads(json.dumps(knobs, sort_keys=True, default=str))


def adopt(source, entry, directory):
    """Copy an adopted task into `directory`; return its manifest entry,
    marked with where it came from."""
    target = directory / entry["task_id"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source / entry["directory"], target)
    return dict(entry, directory=entry["task_id"], adopted_from=str(source))


def load(directory):
    """Read a built task set back off disk."""
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise TaskSetError(
            "{} has no {}, so there is no task set here yet. Build one with "
            "run_eval.py (it builds the set before it scores anything)."
            .format(directory, MANIFEST_NAME))
    manifest = json.loads(manifest_path.read_text())
    tasks = [Task(entry["task_id"], entry["scene"],
                  directory / entry["directory"], entry["seed"])
             for entry in manifest["tasks"]]
    return manifest, tasks


def build(config, directory, open_body, on_task=None):
    """Build every task in the set, and write the manifest that pins it.

    One simulator per scene, not one per task: opening iGibson costs tens of
    seconds and the whole scene's tasks are driven in the same world. Each task
    still gets its own `world.yaml` written beside it, so a task directory
    stays self-contained. A task found already built under `adopt_from` is
    copied instead of driven, and a scene whose tasks are all adopted never
    opens a simulator at all.

    `open_body` is passed in rather than imported so that the caller owns the
    GPU (it has already pinned and verified it) and so this stays testable
    against a fake simulator.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    entries = []
    for scene_index, scene in enumerate(config.scenes):
        body = None
        try:
            for task_index in range(config.tasks_per_scene):
                identifier = task_id(scene, task_index)
                task_dir = directory / identifier
                topomap_config = config.topomap_config_for(
                    scene, scene_index, task_index)

                found = config.adoptable(topomap_config, identifier)
                if found is not None:
                    entries.append(adopt(*found, directory))
                    if on_task is not None:
                        on_task(entries[-1])
                    continue

                # The first task in a scene opens the simulator; the rest reuse
                # it. `open_body` also writes the world config into the task
                # directory, which is why the later tasks still write theirs.
                if body is None:
                    body = open_body(topomap_config, task_dir)
                else:
                    topomap_builder.write_world_config(
                        topomap_config.world_config(), task_dir)

                metadata = topomap_builder.build_topomap(
                    body, topomap_config, task_dir)
                entries.append({
                    "task_id": identifier,
                    "scene": scene,
                    "directory": identifier,
                    "seed": topomap_config.seed,
                    "geodesic_length_m": metadata["geodesic_length_m"],
                    "nodes": len(metadata["nodes"]),
                })
                if on_task is not None:
                    on_task(entries[-1])
        finally:
            if body is not None:
                body.close()

    manifest = {
        "fingerprint": config.fingerprint(),
        "scenes": config.scenes,
        "tasks_per_scene": config.tasks_per_scene,
        "base_seed": config.base_seed,
        "topomap_config": config.base_knobs(),
        "tasks": entries,
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, [Task(entry["task_id"], entry["scene"],
                           directory / entry["directory"], entry["seed"])
                      for entry in entries]


def ensure(config, directory, open_body, on_task=None, rebuild=False):
    """Return the task set at `directory`, building it if it is not there yet.

    An existing set is reused, never rebuilt, and is checked against the
    config's fingerprint first. A mismatch is an error rather than a rebuild:
    the checkpoints already scored against the old set cannot be re-scored by a
    later one, and a table half from each says nothing.
    """
    directory = Path(directory)
    if rebuild or not (directory / MANIFEST_NAME).exists():
        return build(config, directory, open_body, on_task=on_task)

    manifest, tasks = load(directory)
    if manifest.get("fingerprint") != config.fingerprint():
        raise TaskSetError(
            "the task set in {} was built from a different config "
            "(fingerprint {} on disk, {} now). Checkpoints already scored "
            "against it cannot be compared with ones scored against a new "
            "set — so either restore the config, or build the new set in a "
            "new directory and re-run every arm against it."
            .format(directory, manifest.get("fingerprint"), config.fingerprint()))
    return manifest, tasks


def group_by_scene(tasks):
    """Tasks grouped in build order, so a scene is opened once and reused."""
    groups = {}
    for task in tasks:
        groups.setdefault(task.scene, []).append(task)
    return groups
