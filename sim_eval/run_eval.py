"""Score one checkpoint over the whole task set, unattended.

This is P3's driver, and the shape of every run from here on: build the task
set if it is not there, load one checkpoint, run every task as an episode, and
append one row per episode to a CSV as it finishes. Nothing about it is
interactive, and nothing about it is per-arm — the arm is an argument, and the
tasks it faces were fixed before it was named (plan §7).

    ./sim_eval/run_p3_score_test.sh                        # the small proof
    ./sim_eval/run_eval.sh --checkpoint clean_stock        # one full arm

The GPU is an argument too. P3 evaluates one checkpoint on one device; P6
splits the headline pair across both by running this twice, once per GPU, on
the same task set — which is why the device is a parameter here rather than
something the driver decides for itself.

Rows are appended as each episode ends rather than written at the end: a run is
tens of slow episodes, and a crash on the last one must not cost the rest. The
table can be read with `tail -f` while it fills.
"""

import argparse
import traceback
from pathlib import Path

import yaml

import checkpoints
import episode_runner
import gpu
import metrics
import recorder as recording
import task_set
from driver import DriverConfig
from episode_runner import EpisodeRules
from recorder import RecordingConfig
from task_set import TaskSetConfig, TaskSetError
from topomap_builder import TopomapError

SIM_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SIM_EVAL_DIR / "configs" / "eval.yaml"
DEFAULT_VIDEO_DIR = "outputs/videos"


def resolve_path(path):
    """Config paths are relative to `sim_eval/`, so a config means the same
    thing wherever it is run from."""
    path = Path(path)
    return path if path.is_absolute() else SIM_EVAL_DIR / path


def read_layered_yaml(path):
    """A config file laid over the one it names in `extends:`, if any.

    What lets a run say only what is different about it. P6's config extends
    `eval.yaml` and changes where things go, which houses and what is filmed —
    so the episode rules and the driver both arms face have one definition
    shared with every earlier phase, not a copy that could drift from it.
    `extends` is relative to the file that names it.
    """
    path = Path(path)
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    base = data.pop("extends", None)
    if base is None:
        return data
    return deep_merge(read_layered_yaml(path.parent / base), data)


def deep_merge(base, override):
    """`override` laid over `base`: mappings merge key by key, anything else
    (a list included) is replaced whole."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class EvalConfig:
    """`configs/eval.yaml`: which problems, which arms, and when an episode ends."""

    def __init__(self, tasks, task_directory, rules, floor, checkpoint_names,
                 output_dir, recording, driver, seed_offsets=(0,)):
        self.tasks = tasks
        self.task_directory = task_directory
        self.rules = rules
        self.floor = floor
        self.checkpoint_names = checkpoint_names
        self.output_dir = output_dir
        # P4's layer: whether an episode is filmed while it is scored. It can
        # change nothing about the scoring, which is why it is a section of its
        # own rather than another episode rule.
        self.recording = recording
        # How the policy steers. A section of its own because it belongs to the
        # driver, not to the problem: the same task set is faced with it, and
        # every row of the table records which settings faced it.
        self.driver = driver
        # Every task is run once per offset, and every arm runs the same ones:
        # the task set is tasks x seeds episodes. [0] is one episode per task
        # under its own seed, which is every run before P7. See
        # `EpisodeRunner.seed_offset`.
        self.seed_offsets = validate_seed_offsets(seed_offsets)

    @classmethod
    def from_dict(cls, data):
        task_section = dict(data["task_set"])
        episode = dict(data.get("episode") or {})
        record_section = dict(data.get("recording") or {})
        record_section["directory"] = resolve_path(
            record_section.get("directory", DEFAULT_VIDEO_DIR))
        return cls(
            task_directory=resolve_path(task_section.pop("directory")),
            tasks=TaskSetConfig.from_dict(task_section),
            floor=int(episode.pop("floor", 0)),
            # Whatever is left of the episode section is the rules, so a typo
            # in a knob name fails here rather than being silently ignored.
            rules=EpisodeRules.from_dict(episode),
            checkpoint_names=list(data.get("checkpoints") or []),
            output_dir=resolve_path(data.get("output_dir", "outputs")),
            recording=RecordingConfig.from_dict(record_section),
            driver=DriverConfig.from_dict(data.get("driver")),
            seed_offsets=data.get("seed_offsets", [0]),
        )

    @classmethod
    def from_yaml(cls, path=DEFAULT_CONFIG):
        return cls.from_dict(read_layered_yaml(path))

    def csv_path(self, checkpoint_name):
        return self.output_dir / "{}.csv".format(checkpoint_name)


def validate_seed_offsets(offsets):
    """The offsets as ints, or a message: at least one, and no repeats — a
    repeated offset would score the same episode twice."""
    offsets = [int(offset) for offset in offsets]
    if not offsets:
        raise ValueError("seed_offsets is empty: nothing would be scored")
    if len(set(offsets)) != len(offsets):
        raise ValueError("seed_offsets repeats an offset: {}".format(offsets))
    return offsets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="eval config to run (default: %(default)s)")
    parser.add_argument("--checkpoint", default=None,
                        help="name from configs/checkpoints.yaml; default is "
                             "the first in the eval config")
    parser.add_argument("--task-set", type=Path, default=None,
                        help="task set directory (default: from the config)")
    parser.add_argument("--tasks", type=int, default=None,
                        help="override tasks per scene — a smaller set is a "
                             "prefix of a larger one, task for task")
    parser.add_argument("--output", type=Path, default=None,
                        help="metrics CSV to write (default: from the config)")
    parser.add_argument("--rebuild-tasks", action="store_true",
                        help="rebuild the task set even if one is already there")
    parser.add_argument("--build-only", action="store_true",
                        help="build the task set and stop, without scoring")
    parser.add_argument("--resume", action="store_true",
                        help="carry on an existing metrics CSV instead of "
                             "starting it over: episodes already in it are "
                             "skipped, and a half-written last row is dropped")
    parser.add_argument("--quiet", action="store_true",
                        help="one line per episode instead of one per tick")
    # Single-goal mode (configs/word_goal.yaml) only.
    parser.add_argument("--arm", default=None,
                        help="object-goal mode: <checkpoint>+<word|photo|masked>; "
                             "default is every arm in the config")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="object-goal mode: directory for the arms' tables")
    parser.add_argument("--seed-offsets", default=None,
                        help="object-goal mode: comma-separated offsets overriding "
                             "the config's")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to use; never guesses")
    # P4's recorder. `None` means "whatever the config says", so neither flag
    # has to be repeated to keep the config's answer.
    parser.add_argument("--record", dest="record", action="store_true",
                        default=None, help="film every recorded episode")
    parser.add_argument("--no-record", dest="record", action="store_false",
                        help="score without filming (the default)")
    parser.add_argument("--record-fps", type=int, default=None,
                        help="video frame rate; the sim runs at 4 Hz, so 4 is "
                             "real time (default: from the config)")
    parser.add_argument("--record-tasks", default=None,
                        help="which episodes to film: all, a count, or a "
                             "comma-separated list of task ids")
    return parser.parse_args()


def apply_recording_flags(config, args):
    """Let the command line override the config's `recording:` block."""
    if args.record is not None:
        config.recording.enabled = args.record
    if args.record_fps is not None:
        config.recording.fps = args.record_fps
    if args.record_tasks is not None:
        config.recording.tasks = recording.parse_subset(args.record_tasks)
    return config.recording


def build_task_set(config, directory, rebuild=False):
    """Make sure the task set exists, and return its tasks.

    The simulator is opened through P2's own `open_body`, so a task built here
    is built exactly as `p2_1_build_test.py` builds one: same world file, same
    acceptance rules, same code.
    """
    import topomap_builder

    print("task set:   {}".format(config.tasks.summary()))
    print("            {}".format(directory))
    return task_set.ensure(
        config.tasks, directory, topomap_builder.open_body, rebuild=rebuild,
        on_task=lambda entry: print(
            "  {} {:<8} geodesic {:.2f} m, {} nodes, seed {}{}".format(
                "adopted" if "adopted_from" in entry else "built",
                entry["task_id"], entry["geodesic_length_m"],
                entry["nodes"], entry["seed"],
                " (from {})".format(Path(entry["adopted_from"]).name)
                if "adopted_from" in entry else "")))


class EpisodeCrashError(RuntimeError):
    """Raised at the end of a run in which some episodes crashed.

    At the end, not at the crash: the episodes after it still ran and scored.
    The crashed ones have no row, so a `--resume` run retries exactly those.
    """

    def __init__(self, episodes):
        self.episodes = list(episodes)
        super().__init__(
            "{} episode(s) crashed and were left unscored: {}. Every other "
            "episode is in the table; rerun with --resume to retry only these."
            .format(len(self.episodes), ", ".join(self.episodes)))


def score_task(task, task_index, runner, checkpoint_name, table, film,
               verbose=True):
    """Run one task as an episode, film it if asked, and append its row.

    The tick loop is here, in the consumer, rather than inside the runner —
    which is the seam P4's recorder attaches to. Recording is one line of it
    (`video.capture`) rather than a callback the runner would have to know
    about, and an unrecorded run gets a `NullRecording` whose capture does
    nothing, so the loop is the same loop either way. Nothing is kept: a record
    is drawn, printed and released, so a scene's worth of episodes never
    accumulates frames.
    """
    episode = runner.episode(task, checkpoint_name)
    with film.episode(task, checkpoint_name, runner.body.scene,
                      task_index=task_index,
                      seed_offset=runner.seed_offset) as video:
        for record in episode:
            video.capture(record)
            if verbose:
                print(record.summary())
        result = episode.result()
        # A single-goal film ends on a held success/timeout stamp; a topomap
        # film has no `finish` and ends as P4 filmed it.
        if hasattr(video, "finish"):
            video.finish(result.metrics)
    table.append(result.metrics, trace=result.trace())
    print(result.metrics.summary())
    if video.path is not None:
        print(video.summary())


def run_scene(scene, tasks, runner, checkpoint_name, table, film,
              verbose=True):
    """Score every task in one scene not already in the table, under the
    runner's seed offset. Returns the episodes that crashed, as `task (seed)`.

    A task whose (checkpoint, task, seed) already has a row is skipped, which
    is what makes a run resumable: rerun it and it picks up where it stopped.
    A crash inside one episode is printed and the scene carries on — twenty
    slow rollouts must not hang on one — but it is never swallowed: it has no
    row, so it is retried on resume, and the run ends in `EpisodeCrashError`.
    """
    print("\n=== {}: {} tasks, seed offset {} ===".format(
        scene, len(tasks), runner.seed_offset))
    scored = table.completed()
    crashed = []

    # Indexed before skipping: "film the first N tasks" counts from the
    # scene's first task, not from wherever a resumed run starts.
    for task_index, task in enumerate(tasks):
        seed = runner.seed_for(task)
        if (checkpoint_name, task.task_id, seed) in scored:
            print("\n--- {} seed {} ({}) already scored, skipped ---".format(
                task.task_id, seed, checkpoint_name))
            continue
        print("\n--- {} · run seed {} ({}) ---".format(
            task.summary(), seed, checkpoint_name))
        try:
            score_task(task, task_index, runner, checkpoint_name, table, film,
                       verbose=verbose)
        except Exception:  # noqa: BLE001 — isolated, reported, and retried
            traceback.print_exc()
            episode = "{} (seed {})".format(task.task_id, seed)
            print("CRASHED:    {} ({}) — left unscored".format(
                episode, checkpoint_name))
            crashed.append(episode)
    return crashed


def score_checkpoint(config, tasks, policy, checkpoint_name, table,
                     selected_gpu, film, verbose=True):
    """Run every task under every seed offset for one checkpoint, one scene's
    simulator at a time. Returns the episodes that crashed.

    The offsets loop inside the scene, so a house is opened once per arm
    however many seeds it is run under.

    The world is opened from the first task's own `world.yaml` — the file P2
    wrote when it drove the trail — so an episode runs in the world its task
    was defined in, rather than in whatever a config happens to say today.
    """
    import bridge

    crashed = []
    for scene, scene_tasks in task_set.group_by_scene(tasks).items():
        body = bridge.SimBody(config_path=scene_tasks[0].world_config,
                              floor=config.floor)
        try:
            body.verify_gpu(selected_gpu)
            for seed_offset in config.seed_offsets:
                runner = episode_runner.EpisodeRunner(
                    policy, body, rules=config.rules, seed_offset=seed_offset)
                crashed += run_scene(scene, scene_tasks, runner,
                                     checkpoint_name, table, film,
                                     verbose=verbose)
        finally:
            body.close()
    return crashed


def build_recorder(config, policy, caption_for=None):
    """P4's recorder for this run, or a disabled one.

    The waypoint scale is read off the checkpoint that is about to drive — the
    overlay draws the model's own samples in metres, and a checkpoint trained
    without normalization means them in metres already. `caption_for` is
    `recorder.Recorder`'s optional per-episode caption.
    """
    import bridge
    import pd_control

    if not config.recording.enabled:
        return recording.disabled()
    return recording.Recorder(
        config.recording,
        waypoint_scale_m=bridge.waypoint_scale_m(
            policy.model_params, pd_control.RobotLimits.from_config()),
        success_radius_m=config.rules.success_radius_m, caption_for=caption_for)


def evaluate(config, checkpoint_name, csv_path, selected_gpu, task_directory=None,
             rebuild_tasks=False, resume=False, verbose=True, build_only=False):
    """The whole of a scoring run: task set, checkpoint, every episode, the table."""
    import torch

    from nomad_policy import NomadPolicy

    directory = task_directory or config.task_directory
    _manifest, tasks = build_task_set(config, directory, rebuild=rebuild_tasks)
    print("            {} tasks\n".format(len(tasks)))
    if build_only:
        return None

    spec = checkpoints.load(checkpoint_name)
    print("checkpoint: {}".format(spec.summary()))
    print("rules:      {}".format(config.rules.summary()))
    print("seeds:      offsets {} — {} episodes".format(
        config.seed_offsets, len(tasks) * len(config.seed_offsets)))
    print("driver:     {}".format(config.driver.summary()))
    print("recording:  {}".format(config.recording.summary()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:     {}".format(device))
    policy = NomadPolicy(spec, device, config.driver)

    table = metrics.MetricsTable(csv_path).open(resume=resume)
    if table.rows:
        print("resuming:   {} episode(s) already scored".format(table.rows))
    crashed = score_checkpoint(config, tasks, policy, checkpoint_name, table,
                               selected_gpu, build_recorder(config, policy),
                               verbose=verbose)

    print()
    print(metrics.format_summary(
        checkpoint_name, metrics.summarize(metrics.read_table(csv_path))))
    print()
    print("wrote:      {}".format(csv_path))
    print("            {}".format(table.trace_path))
    if crashed:
        raise EpisodeCrashError(crashed)
    return table


# --- single-goal mode (the language-goal sim, Phase 7) ------------------------
#
# `mode: object_goal` in a config switches this driver from topomap tasks to
# the word task set (object_tasks.py). An arm is `<checkpoint>+<goal>`, e.g.
# `clip_v2b+word`: the goal kind is part of what the arm is, as its image size
# and stride are, and it names the arm's table (`clip_v2b+word.csv`). Every
# arm faces the same word tasks under the same seed offsets, so the fairness
# gate (comparison.py) applies to these tables unchanged.

OBJECT_GOAL_MODE = "object_goal"


def parse_arm(arm):
    """`checkpoint+goal` -> (checkpoint, goal kind)."""
    import goals

    checkpoint, sep, kind = str(arm).partition("+")
    if not sep or kind not in goals.GOAL_KINDS:
        raise SystemExit("FAILED: arm {!r} is not <checkpoint>+<{}>".format(
            arm, "|".join(goals.GOAL_KINDS)))
    return checkpoint, kind


class ObjectEvalConfig:
    """A `mode: object_goal` config: the word task set, the arms, the rules."""

    def __init__(self, tasks, task_directory, rules, arms, output_dir, recording,
                 driver, seed_offsets=(0,)):
        self.tasks = tasks
        self.task_directory = task_directory
        self.rules = rules
        self.arms = list(arms)
        self.output_dir = output_dir
        self.recording = recording
        self.driver = driver
        self.seed_offsets = validate_seed_offsets(seed_offsets)

    @classmethod
    def from_dict(cls, data):
        import object_tasks

        task_section = dict(data["object_tasks"])
        episode = dict(data.get("episode") or {})
        floor = int(episode.pop("floor", 0))
        task_section.setdefault("floor", floor)
        record_section = dict(data.get("recording") or {})
        record_section["directory"] = resolve_path(
            record_section.get("directory", DEFAULT_VIDEO_DIR))
        for arm in data.get("arms") or []:
            parse_arm(arm)
        return cls(
            task_directory=resolve_path(task_section.pop("directory")),
            tasks=object_tasks.ObjectTaskSetConfig.from_dict(task_section),
            rules=EpisodeRules.from_dict(episode),
            arms=data.get("arms") or [],
            output_dir=resolve_path(data.get("output_dir", "outputs")),
            recording=RecordingConfig.from_dict(record_section),
            driver=DriverConfig.from_dict(data.get("driver")),
            seed_offsets=data.get("seed_offsets", [0]),
        )

    def csv_path(self, arm):
        return self.output_dir / "{}.csv".format(arm)

    # What comparison.load_fair_tables reads off a config.
    @property
    def checkpoint_names(self):
        return self.arms


def open_world(world_path, floor=0):
    import bridge

    return bridge.SimBody(config_path=world_path, floor=floor)


def build_object_task_set(config, directory=None):
    import object_tasks

    directory = directory or config.task_directory
    print("task set:   {}".format(config.tasks.summary()))
    print("            {}".format(directory))
    return object_tasks.ensure(
        config.tasks, directory, lambda world: open_world(world, config.tasks.floor),
        on_task=lambda entry: print(
            "  built {:<14} start ({:+.2f}, {:+.2f}) yaw {:+.2f}  geodesic {:.2f} m "
            "to {} ({} start cells)".format(
                entry["task_id"], entry["start_pose"]["x"], entry["start_pose"]["y"],
                entry["start_pose"]["yaw"], entry["geodesic_length_m"],
                entry["target_instance"], entry["candidates"])))


def goal_provider(kind, spec, clip):
    """`task -> Goal` for one arm, each distinct goal built once."""
    import goals

    cache = {}

    def goal_for(task):
        key = (task.word, task.target_instance) if kind == "photo" else task.word
        if key not in cache:
            photo = task.photo() if kind == "photo" else None
            cache[key] = goals.build_goal(kind, spec, clip, word=task.word, photo=photo)
        return cache[key]

    return goal_for


RUN_MANIFEST = "run_manifest.json"


def run_manifest_path(directory):
    return Path(directory) / RUN_MANIFEST


def read_run_manifest(directory):
    """A run folder's manifest, or SystemExit: every reader of a run needs it."""
    import json

    path = run_manifest_path(directory)
    if not path.exists():
        raise SystemExit("FAILED: {} has no {}; it was not written by a reviewed "
                         "object-goal run".format(directory, RUN_MANIFEST))
    return json.loads(path.read_text())


def record_arm(config, arm, spec, task_set_manifest, task_directory, csv_path, resume):
    """Tie an arm's table to the run it belongs to, or refuse.

    A run folder holds one `run_manifest.json`: the task set (directory and
    fingerprint), the seed offsets, the driver and the episode rules every
    arm in it faced, plus one entry per arm (its weights and how its goal
    feeding was established). A second arm must match the run's shared
    fields; a resumed arm must match its own entry too. That is what keeps a
    resume or a report from mixing two runs.

    Without `--resume` an existing arm table is refused rather than truncated.
    """
    import json

    run = {
        "task_set": {"directory": str(task_directory),
                     "fingerprint": task_set_manifest["fingerprint"]},
        "seed_offsets": config.seed_offsets,
        "driver": config.driver.label(),
        "rules": config.rules.as_dict(),
    }
    entry = {
        "checkpoint": spec.name, "weights": str(spec.weights_path),
        "goal_provenance": spec.goal_provenance,
        "goal_config": {key: spec.model_params[key]
                        for key in checkpoints.GOAL_CONFIG_KEYS},
        "table": Path(csv_path).name,
    }
    path = run_manifest_path(Path(csv_path).parent)
    existing = json.loads(path.read_text()) if path.exists() else None
    if existing is not None:
        shared = {key: existing[key] for key in run}
        if shared != json.loads(json.dumps(run)):
            raise SystemExit("FAILED: {} belongs to another run (task set, seeds, "
                             "driver or rules differ):\n  on disk {}\n  now     {}"
                             .format(path.parent, shared, run))
    arms = dict(existing["arms"]) if existing is not None else {}
    if Path(csv_path).exists() and not resume:
        raise SystemExit("FAILED: {} exists; pass --resume to carry it on, or use a "
                         "new output directory".format(csv_path))
    if resume and arm in arms and arms[arm] != json.loads(json.dumps(entry)):
        raise SystemExit("FAILED: resuming {} with a different checkpoint or goal "
                         "config than its table was scored with".format(arm))
    arms[arm] = entry
    run["arms"] = arms
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, default=str))
    return run


def evaluate_object_arm(config, arm, csv_path, selected_gpu, task_directory=None,
                        resume=False, verbose=True):
    """Score one arm over the whole word task set, one house at a time."""
    import torch

    import bridge
    import object_tasks
    import objects as scene_objects_module
    import recorder
    from nomad_policy import NomadPolicy

    checkpoint_name, kind = parse_arm(arm)
    task_directory = task_directory or config.task_directory
    task_set_manifest, tasks = build_object_task_set(config, task_directory)
    print("            {} tasks\n".format(len(tasks)))

    spec = checkpoints.load_goal_arm(checkpoint_name)
    record_arm(config, arm, spec, task_set_manifest, task_directory, csv_path, resume)
    print("arm:        {} ({} goal)".format(arm, kind))
    print("checkpoint: {}".format(spec.summary()))
    print("rules:      {}".format(config.rules.summary()))
    print("seeds:      offsets {} — {} episodes".format(
        config.seed_offsets, len(tasks) * len(config.seed_offsets)))
    print("driver:     {}".format(config.driver.summary()))
    print("recording:  {}".format(config.recording.summary()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = NomadPolicy(spec, device, config.driver)
    goal_for = goal_provider(kind, spec, policy.clip)

    table = metrics.MetricsTable(csv_path).open(resume=resume)
    if table.rows:
        print("resuming:   {} episode(s) already scored".format(table.rows))
    crashed = []
    for scene, scene_tasks in object_tasks.group_by_scene(tasks).items():
        body = bridge.SimBody(config_path=scene_tasks[0].world_config,
                              floor=config.tasks.floor)
        try:
            body.verify_gpu(selected_gpu)
            house = scene_objects_module.SceneObjects.for_scene(body.scene)
            film = recorder.disabled()
            if config.recording.enabled:
                film = recorder.ObjectGoalRecorder(
                    config.recording,
                    waypoint_scale_m=bridge.waypoint_scale_m(
                        policy.model_params, body.limits),
                    success_radius_m=config.rules.success_radius_m,
                    objects=house, goal_for=goal_for)
            for seed_offset in config.seed_offsets:
                runner = episode_runner.ObjectGoalRunner(
                    policy, body, house, goal_for, rules=config.rules,
                    seed_offset=seed_offset)
                crashed += run_scene(scene, scene_tasks, runner, arm, table, film,
                                     verbose=verbose)
        finally:
            body.close()

    print()
    print(metrics.format_summary(arm, metrics.summarize(metrics.read_table(csv_path))))
    print()
    print("wrote:      {}".format(csv_path))
    print("            {}".format(table.trace_path))
    if crashed:
        raise EpisodeCrashError(crashed)
    return table


def main_object_goal(args, data, selected_gpu):
    import object_tasks

    config = ObjectEvalConfig.from_dict(data)
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed_offsets is not None:
        config.seed_offsets = validate_seed_offsets(args.seed_offsets.split(","))
    apply_recording_flags(config, args)
    if args.build_only:
        build_object_task_set(config, args.task_set)
        return
    arms = [args.arm] if args.arm else config.arms
    if not arms:
        raise SystemExit("FAILED: no arm. Name one with --arm or list `arms:`.")
    if args.output is not None and len(arms) > 1:
        raise SystemExit("FAILED: --output names one table, but {} arms would write "
                         "into it; use --arm or --output-dir".format(len(arms)))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        evaluate_object_arm(config, arm, csv_path=args.output or config.csv_path(arm),
                            selected_gpu=selected_gpu, task_directory=args.task_set,
                            resume=args.resume, verbose=not args.quiet)


def main():
    args = parse_args()

    # Before importing iGibson or torch: both read the device variables when
    # they initialise, not when they are used.
    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))

    data = read_layered_yaml(args.config)
    print("config:     {}".format(args.config))
    if data.get("mode") == OBJECT_GOAL_MODE:
        return main_object_goal(args, data, selected_gpu)
    config = EvalConfig.from_dict(data)
    if args.tasks is not None:
        config.tasks.tasks_per_scene = args.tasks
    apply_recording_flags(config, args)

    checkpoint_name = args.checkpoint or (config.checkpoint_names or [None])[0]
    if checkpoint_name is None and not args.build_only:
        raise SystemExit(
            "FAILED: no checkpoint to evaluate. Name one with --checkpoint or "
            "list one under `checkpoints:` in {}.".format(args.config))

    evaluate(config, checkpoint_name,
             csv_path=args.output or config.csv_path(checkpoint_name),
             selected_gpu=selected_gpu,
             task_directory=args.task_set,
             rebuild_tasks=args.rebuild_tasks,
             resume=args.resume,
             verbose=not args.quiet,
             build_only=args.build_only)


if __name__ == "__main__":
    # A bad GPU pin, an unknown checkpoint or a task set that does not match
    # its config are all messages to a person, not defects to debug.
    try:
        main()
    except (gpu.GpuSelectionError, checkpoints.CheckpointError, TaskSetError,
            TopomapError, EpisodeCrashError) as error:
        raise SystemExit("FAILED: {}".format(error))
    except RuntimeError as error:
        # object_tasks.ObjectTaskSetError, without importing it at module scope.
        if type(error).__name__ != "ObjectTaskSetError":
            raise
        raise SystemExit("FAILED: {}".format(error))
