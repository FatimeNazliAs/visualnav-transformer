"""The word-goal task set: fixed, seeded object-reach problems every arm faces.

A topomap task (task_set.py) is a start, a trail and a goal pose. A word has
no trail and no goal pose, so the language-goal sim (Phase 7) needs its own
kind of task:

    start pose  ->  reach any instance of <word>, within success_radius_m

and every arm — word, photo, masked — faces the same ones. A photo arm is
also told *which* instance: the one geodesically nearest the start, whose
goal photo is rendered once, at build time, and frozen with the set.

**Starts** (plan, Phase 7): 2-5 m of geodesic from the nearest instance, with
the target in view or one turn away. Operationally:

  * the category's distance field (objects.py) at the start is inside
    [min_geodesic_m, max_geodesic_m], on the scene's main component;
  * the straight line to the nearest instance is clear (`line_of_sight`), so
    the target is not behind a wall;
  * the heading is the bearing to the target plus a uniform offset of at most
    `yaw_offset_deg` — 0 faces it, 90 is a quarter turn away.

Each task's start is drawn from its own seed over the list of every
qualifying cell, so a task does not depend on how many were asked for (the
first 3 tasks of a 3-task and a 5-task set are the same 3).

**Frozen and fingerprinted**, like task_set.py: the fingerprint covers the
knobs, the world file's contents *and the scene's annotations*, so moving a
chair after an arm has been scored makes a new set rather than silently
changing the old one.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

import objects

SIM_EVAL_DIR = Path(__file__).resolve().parent
MANIFEST_NAME = "manifest.json"
WORLD_CONFIG_NAME = "world.yaml"


class ObjectTaskSetError(RuntimeError):
    """Raised for a word task set that cannot be built or does not match its config."""


class StartRules:
    """Where a word task may start, relative to its nearest target instance."""

    def __init__(self, min_geodesic_m=2.0, max_geodesic_m=5.0, yaw_offset_deg=90.0,
                 line_of_sight=True, min_clearance_m=0.3, min_pairwise_start_m=0.0,
                 min_in_view=0, min_out_of_view=0, forward_clear_m=0.0):
        self.min_geodesic_m = float(min_geodesic_m)
        self.max_geodesic_m = float(max_geodesic_m)
        self.yaw_offset_deg = float(yaw_offset_deg)
        self.line_of_sight = bool(line_of_sight)
        # From the house mesh, not the trav map (objects.MeshClearance): the
        # map marks some furniture as floor.
        self.min_clearance_m = float(min_clearance_m)
        # A word's tasks start at least this far apart (straight line).
        self.min_pairwise_start_m = float(min_pairwise_start_m)
        # A word's tasks include at least this many with the target inside the
        # camera's horizontal field of view at the start, and this many with it
        # outside (at most yaw_offset_deg away: one turn). Task index decides:
        # the first min_in_view are in view, the next min_out_of_view are not,
        # the rest are drawn over the whole +-yaw_offset_deg.
        self.min_in_view = int(min_in_view)
        self.min_out_of_view = int(min_out_of_view)
        # The start heading must look down open floor: the straight segment
        # this long ahead is free on the trav map and clears the house mesh by
        # min_clearance_m all along it (0 turns the rule off). Without it an
        # out-of-view heading can face a wall, and a policy with no obstacle
        # avoidance pins itself before it can show whether it uses the goal.
        self.forward_clear_m = float(forward_clear_m)
        if not 0 < self.min_geodesic_m < self.max_geodesic_m:
            raise ValueError("need 0 < min_geodesic_m < max_geodesic_m")

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))

    def as_dict(self):
        return {"min_geodesic_m": self.min_geodesic_m, "max_geodesic_m": self.max_geodesic_m,
                "yaw_offset_deg": self.yaw_offset_deg, "line_of_sight": self.line_of_sight,
                "min_clearance_m": self.min_clearance_m,
                "min_pairwise_start_m": self.min_pairwise_start_m,
                "min_in_view": self.min_in_view, "min_out_of_view": self.min_out_of_view,
                "forward_clear_m": self.forward_clear_m}

    def view_class(self, task_index):
        """"in", "out" or "any": where task `task_index`'s target must be at its start."""
        if task_index < self.min_in_view:
            return "in"
        if task_index < self.min_in_view + self.min_out_of_view:
            return "out"
        return "any"


class ObjectTaskSetConfig:
    """Which word tasks exist: houses, words, how many, from which seed."""

    # What a word may override: how many tasks it has, and its view mix.
    PER_WORD_KEYS = ("tasks", "min_in_view", "min_out_of_view")

    def __init__(self, scenes, words, tasks_per_word=5, base_seed=7000,
                 world_config="configs/locobot_rs_bridge.yaml", start=None, floor=0,
                 annotations=objects.ANNOTATIONS_PATH, per_word=None):
        self.scenes = list(scenes)
        self.words = list(words)
        self.tasks_per_word = int(tasks_per_word)
        self.base_seed = int(base_seed)
        path = Path(world_config)
        self.world_config = path if path.is_absolute() else SIM_EVAL_DIR / path
        self.start = start if isinstance(start, StartRules) else StartRules.from_dict(start)
        self.floor = int(floor)
        self.annotations = Path(annotations)
        # {word: {tasks, min_in_view, min_out_of_view}}: a word whose house has
        # less room gets fewer tasks or another view mix; every other rule is
        # the set's own.
        self.per_word = {word: dict(values) for word, values in (per_word or {}).items()}
        for word, values in self.per_word.items():
            if word not in self.words:
                raise ValueError("per_word names {!r}, which is not in words".format(word))
            unknown = set(values) - set(self.PER_WORD_KEYS)
            if unknown:
                raise ValueError("per_word.{}: unknown keys {}".format(word, sorted(unknown)))
        if self.tasks_per_word < 1 or not self.words or not self.scenes:
            raise ValueError("a word task set needs scenes, words and >= 1 task per word")
        if any(self.tasks_for(word) < 1 for word in self.words):
            raise ValueError("every word needs >= 1 task")

    def tasks_for(self, word):
        return int(self.per_word.get(word, {}).get("tasks", self.tasks_per_word))

    def rules_for(self, word):
        """The start rules for one word: the set's, with its view-mix overrides."""
        overrides = {key: value for key, value in self.per_word.get(word, {}).items()
                     if key != "tasks"}
        if not overrides:
            return self.start
        return StartRules.from_dict(dict(self.start.as_dict(), **overrides))

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def world(self, scene):
        """The iGibson world for one house: the world file with its scene_id."""
        with open(self.world_config, "r") as handle:
            world = yaml.safe_load(handle)
        world["scene_id"] = scene
        return world

    def task_seed(self, scene_index, word_index, task_index):
        return self.base_seed + scene_index * 1000 + word_index * 100 + task_index

    def fingerprint(self):
        with open(self.annotations, "r") as handle:
            annotations = yaml.safe_load(handle) or {}
        payload = {
            "scenes": self.scenes,
            "words": self.words,
            "tasks_per_word": self.tasks_per_word,
            "per_word": self.per_word,
            "base_seed": self.base_seed,
            "start": self.start.as_dict(),
            "floor": self.floor,
            "worlds": {scene: self.world(scene) for scene in self.scenes},
            "annotations": {scene: annotations.get(scene) for scene in self.scenes},
            "field": {"source_reach_m": objects.SOURCE_REACH_M,
                      "los_margin_m": objects.LOS_MARGIN_M},
            # How a start's clearance is measured decides which starts exist.
            "clearance": {"body_band_m": list(objects.BODY_BAND_M),
                          "resolution_m": objects.CLEARANCE_RES_M},
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def summary(self):
        return ("{} · words {} · {} per word · seeds from {} · start {:.1f}-{:.1f} m, "
                "yaw +-{:.0f} deg{}".format(
                    ", ".join(self.scenes), ", ".join(self.words), self.tasks_per_word,
                    self.base_seed, self.start.min_geodesic_m, self.start.max_geodesic_m,
                    self.start.yaw_offset_deg,
                    ", line of sight" if self.start.line_of_sight else ""))


class ObjectTask:
    """One word-goal problem, read back from the manifest."""

    def __init__(self, entry, directory):
        self.entry = dict(entry)
        self.directory = Path(directory)
        self.task_id = entry["task_id"]
        self.scene = entry["scene"]
        self.seed = int(entry["seed"])
        self.word = entry["word"]
        self.target_instance = entry["target_instance"]

    @property
    def metadata(self):
        return self.entry

    @property
    def world_config(self):
        return self.directory / self.scene / WORLD_CONFIG_NAME

    @property
    def geodesic_length_m(self):
        """Geodesic from the start to the nearest instance of the word: SPL's
        shortest path, and the length the timeout is tied to."""
        return float(self.entry["geodesic_length_m"])

    @property
    def node_count(self):
        return 0

    @property
    def photo_path(self):
        return self.directory / self.entry["photo"]

    def photo(self):
        from PIL import Image

        return Image.open(self.photo_path).convert("RGB")

    def start_pose(self, floor_height):
        start = self.entry["start_pose"]
        return ([float(start["x"]), float(start["y"]), float(floor_height)],
                [0.0, 0.0, float(start["yaw"])])

    def summary(self):
        return "{:<14} {:<4} {:<6} geodesic {:.2f} m (target {}) · seed {}".format(
            self.task_id, self.scene, self.word, self.geodesic_length_m,
            self.target_instance, self.seed)


def task_id(scene, word, task_index):
    return "{}_{}_{:02d}".format(scene, word.replace(" ", "-"), task_index)


def start_candidates(scene_objects, word, rules, has_node=None, clearance=None):
    """Every free cell a `word` task may start from, as [(x, y, instance_id, geodesic)].

    In row-major order, so a seed indexes the same cell on every build.
    """
    grid = scene_objects.grid
    field = scene_objects.category_field(word)
    rows, cols = np.nonzero(grid.free)
    candidates = []
    for row, col in sorted(zip(rows.tolist(), cols.tolist())):
        xy = grid.centre(row, col)
        if not rules.min_geodesic_m <= field.values[row, col] <= rules.max_geodesic_m:
            continue
        # The same lookup success and final distance use, so a task's length
        # and its episodes' distances are one ruler.
        geodesic = field.distance(xy)
        if geodesic is None or not rules.min_geodesic_m <= geodesic <= rules.max_geodesic_m:
            continue
        if has_node is not None and not has_node(xy):
            continue
        if clearance is not None and clearance.clearance(xy) < rules.min_clearance_m:
            continue
        instance, _ = scene_objects.nearest_instance(word, xy)
        # Line of sight: nothing on the segment to the object is an obstacle
        # until it reaches the object's own erosion margin.
        if rules.line_of_sight and not objects.clear_line(grid, xy, instance):
            continue
        candidates.append((float(xy[0]), float(xy[1]), instance.id, float(geodesic)))
    return candidates


def draw_offset(rng, rules, view, half_fov_deg):
    """The heading's offset from the target bearing, in degrees, for a view class.

    "in": inside the camera's half field of view; "out": beyond it, up to
    yaw_offset_deg (one turn away), either side; "any": the whole range.
    """
    limit = rules.yaw_offset_deg
    if view == "in":
        return rng.uniform(-min(half_fov_deg, limit), min(half_fov_deg, limit))
    if view == "out":
        if half_fov_deg >= limit:
            raise ObjectTaskSetError("no out-of-view heading: half FOV {:.1f} deg >= "
                                     "yaw_offset_deg {}".format(half_fov_deg, limit))
        magnitude = rng.uniform(half_fov_deg, limit)
        return magnitude if rng.uniform() < 0.5 else -magnitude
    return rng.uniform(-limit, limit)


def forward_is_clear(grid, clearance, start, length_m, min_clearance_m, step_m=0.05):
    """Is the straight segment `length_m` ahead of a start free and clear of the mesh?"""
    steps = int(round(length_m / step_m))
    for k in range(steps + 1):
        point = (start["x"] + k * step_m * math.cos(start["yaw"]),
                 start["y"] + k * step_m * math.sin(start["yaw"]))
        if not grid.is_free(point):
            return False
        if clearance is not None and clearance.clearance(point) < min_clearance_m:
            return False
    return True


def draw_start(candidates, scene_objects, word, seed, rules, taken=(), touching=None,
               view="any", half_fov_deg=90.0, forward_clear=None):
    """One task's start from its own seed: a candidate cell and a heading.

    `taken` holds the (x, y) of the word's earlier starts: a candidate closer
    than `rules.min_pairwise_start_m` to any of them is not available. The
    heading is the bearing to the target plus an offset drawn for `view`
    (`draw_offset`). `forward_clear(start)` says whether the heading looks down
    open floor; an out-of-view heading that does not is tried mirrored (the
    same offset, the other side), and a candidate with neither is dropped.
    `touching(start)`
    places the robot there and says whether it is already in contact with
    something; such a start is dropped and the same generator draws again, so
    the result is still a function of the seed (and the earlier starts) alone.
    Returns (start, instance_id, geodesic, rejected).
    """
    rng = np.random.RandomState(seed)
    available = [c for c in candidates if spaced(c, taken, rules.min_pairwise_start_m)]
    rejected = 0
    while available:
        x, y, instance_id, geodesic = available.pop(rng.randint(len(available)))
        target = scene_objects.instances[instance_id].nearest_point([x, y])
        bearing = math.atan2(target[1] - y, target[0] - x)
        offset = draw_offset(rng, rules, view, half_fov_deg)
        offsets = [offset, -offset] if view == "out" else [offset]
        for offset_deg in offsets:
            yaw = (bearing + math.radians(offset_deg) + math.pi) % (2 * math.pi) - math.pi
            start = {"x": round(x, 4), "y": round(y, 4), "yaw": round(yaw, 4)}
            if forward_clear is not None and not forward_clear(start):
                continue
            break
        else:
            continue
        if touching is not None and touching(start):
            rejected += 1
            continue
        return start, instance_id, geodesic, rejected
    raise ObjectTaskSetError(
        "no start left for {!r}: {} candidates, {} earlier starts at >= {} m spacing, "
        "{} touching".format(word, len(candidates), len(taken), rules.min_pairwise_start_m,
                             rejected))


def spaced(candidate, taken, min_spacing_m):
    """Is a candidate at least `min_spacing_m` from every earlier start, and not one?"""
    x, y = round(candidate[0], 4), round(candidate[1], 4)
    return all((x, y) != (tx, ty) and math.hypot(x - tx, y - ty) >= min_spacing_m
               for tx, ty in taken)


def check_view_mix(rules, entries):
    """Refuse a word whose starts miss the in-view / out-of-view counts."""
    in_view = sum(1 for e in entries if e["target_in_view"])
    out_of_view = len(entries) - in_view
    if in_view < rules.min_in_view or out_of_view < rules.min_out_of_view:
        raise ObjectTaskSetError(
            "{}: {} starts in view, {} out of view; need >= {} and >= {}".format(
                entries[0]["word"] if entries else "?", in_view, out_of_view,
                rules.min_in_view, rules.min_out_of_view))


def target_bearing(start, instance):
    """The target's nearest point relative to the start heading, in degrees (-180..180]."""
    target = instance.nearest_point([start["x"], start["y"]])
    bearing = math.atan2(target[1] - start["y"], target[0] - start["x"]) - start["yaw"]
    return math.degrees((bearing + math.pi) % (2 * math.pi) - math.pi)


def horizontal_fov_deg(intrinsics):
    """The camera's horizontal field of view, from its vertical one and aspect."""
    half_vertical = math.radians(intrinsics["vertical_fov_deg"]) / 2
    aspect = intrinsics["width"] / intrinsics["height"]
    return math.degrees(2 * math.atan(math.tan(half_vertical) * aspect))


def start_touches(body, start):
    """Place the robot at `start`, hold still one tick: is it in contact?"""
    body.reset()
    body.place([start["x"], start["y"], body.scene.floor_height], [0.0, 0.0, start["yaw"]])
    return bool(body.command(0.0, 0.0))


def build(config, directory, open_body, on_task=None):
    """Build the set: world files, goal photos, starts, manifest.

    `open_body(world_path)` opens the simulator on a written world file, so a
    world.yaml in the set is the file iGibson actually loaded.
    """
    import goals

    directory = Path(directory)
    # Any content refuses, not only a manifest: a build that failed half way
    # leaves world files and photos behind, and a set built over them would
    # be a mix of two builds.
    if directory.exists() and any(directory.iterdir()):
        raise ObjectTaskSetError("{} exists and is not empty; a task set is built into "
                                 "a new folder".format(directory))
    entries = []
    for scene_index, scene in enumerate(config.scenes):
        scene_dir = directory / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        world_path = scene_dir / WORLD_CONFIG_NAME
        with open(world_path, "w") as handle:
            yaml.safe_dump(config.world(scene), handle, default_flow_style=False,
                           sort_keys=False)
        body = open_body(world_path)
        try:
            scene_objects = objects.SceneObjects.for_scene(body.scene)
            clearance = objects.MeshClearance.for_scene(scene, body.scene.floor_height)
            half_fov = horizontal_fov_deg(body.intrinsics) / 2
            forward_clear = None
            if config.start.forward_clear_m > 0:
                forward_clear = lambda start: forward_is_clear(  # noqa: E731
                    scene_objects.grid, clearance, start, config.start.forward_clear_m,
                    config.start.min_clearance_m)
            photos = {}
            for word_index, word in enumerate(config.words):
                rules = config.rules_for(word)
                candidates = start_candidates(scene_objects, word, rules,
                                              has_node=body.scene.has_node,
                                              clearance=clearance)
                taken = set()
                for task_index in range(config.tasks_for(word)):
                    seed = config.task_seed(scene_index, word_index, task_index)
                    start, instance_id, geodesic, rejected = draw_start(
                        candidates, scene_objects, word, seed, rules, taken,
                        touching=lambda start: start_touches(body, start),
                        view=rules.view_class(task_index), half_fov_deg=half_fov,
                        forward_clear=forward_clear)
                    taken.add((start["x"], start["y"]))
                    if instance_id not in photos:
                        photo_rel = Path(scene) / "photos" / "{}.png".format(instance_id)
                        (directory / photo_rel).parent.mkdir(parents=True, exist_ok=True)
                        goals.render_goal_photo(
                            body, scene_objects.instances[instance_id]).save(directory / photo_rel)
                        photos[instance_id] = str(photo_rel)
                    entry = {
                        "task_id": task_id(scene, word, task_index), "scene": scene,
                        "word": word, "seed": seed, "start_pose": start,
                        "target_instance": instance_id,
                        "geodesic_length_m": round(geodesic, 4),
                        "candidates": len(candidates), "touching_rejected": rejected,
                        "clearance_m": round(clearance.clearance([start["x"], start["y"]]), 3),
                        # Reported, not a rule: line of sight is the rule, and
                        # a start up to yaw_offset_deg off is "one turn away".
                        "target_bearing_deg": round(target_bearing(
                            start, scene_objects.instances[instance_id]), 1),
                        "photo": photos[instance_id],
                    }
                    entry["target_in_view"] = abs(entry["target_bearing_deg"]) <= half_fov
                    entry["view_class"] = rules.view_class(task_index)
                    entries.append(entry)
                    if on_task is not None:
                        on_task(entry)
                check_view_mix(rules, [e for e in entries
                                              if e["scene"] == scene and e["word"] == word])
        finally:
            body.close()
    manifest = {"fingerprint": config.fingerprint(), "summary": config.summary(),
                "tasks": entries}
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return manifest, [ObjectTask(entry, directory) for entry in entries]


def load(directory):
    directory = Path(directory)
    path = directory / MANIFEST_NAME
    if not path.exists():
        raise ObjectTaskSetError("{} has no {}; build the set first".format(directory,
                                                                         MANIFEST_NAME))
    manifest = json.loads(path.read_text())
    return manifest, [ObjectTask(entry, directory) for entry in manifest["tasks"]]


def ensure(config, directory, open_body, on_task=None):
    """The set at `directory`, building it if absent; refuse a fingerprint mismatch."""
    directory = Path(directory)
    if not (directory / MANIFEST_NAME).exists():
        return build(config, directory, open_body, on_task=on_task)
    manifest, tasks = load(directory)
    if manifest["fingerprint"] != config.fingerprint():
        raise ObjectTaskSetError(
            "the word task set in {} was built from a different config or different "
            "annotations (fingerprint {} on disk, {} now). Build the new set in a new "
            "directory and re-run every arm on it.".format(
                directory, manifest["fingerprint"], config.fingerprint()))
    return manifest, tasks


def group_by_scene(tasks):
    groups = {}
    for task in tasks:
        groups.setdefault(task.scene, []).append(task)
    return groups
