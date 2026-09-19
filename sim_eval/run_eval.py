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


class EvalConfig:
    """`configs/eval.yaml`: which problems, which arms, and when an episode ends."""

    def __init__(self, tasks, task_directory, rules, floor, checkpoint_names,
                 output_dir, recording, driver, seed_offset=0):
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
        # 0 for anything that is scored; see `EpisodeRunner.seed_offset`.
        self.seed_offset = int(seed_offset)

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
            seed_offset=data.get("seed_offset", 0),
        )

    @classmethod
    def from_yaml(cls, path=DEFAULT_CONFIG):
        with open(path, "r") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def csv_path(self, checkpoint_name):
        return self.output_dir / "{}.csv".format(checkpoint_name)


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
                        help="append to an existing metrics CSV instead of "
                             "starting it over")
    parser.add_argument("--quiet", action="store_true",
                        help="one line per episode instead of one per tick")
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
            "  built {:<8} geodesic {:.2f} m, {} nodes, seed {}".format(
                entry["task_id"], entry["geodesic_length_m"],
                entry["nodes"], entry["seed"])))


def run_scene(scene, tasks, runner, checkpoint_name, table, film,
              verbose=True):
    """Score every task in one scene, through one already-open simulator.

    The tick loop is here, in the consumer, rather than inside the runner —
    which is the seam P4's recorder attaches to. Recording is one line of it
    (`video.capture`) rather than a callback the runner would have to know
    about, and an unrecorded run gets a `NullRecording` whose capture does
    nothing, so the loop is the same loop either way. Nothing is kept: a record
    is drawn, printed and released, so a scene's worth of episodes never
    accumulates frames.
    """
    print("\n=== {}: {} tasks ===".format(scene, len(tasks)))

    for task_index, task in enumerate(tasks):
        print("\n--- {} ({}) ---".format(task.summary(), checkpoint_name))
        episode = runner.episode(task, checkpoint_name)
        with film.episode(task, checkpoint_name, runner.body.scene,
                          task_index=task_index) as video:
            for record in episode:
                video.capture(record)
                if verbose:
                    print(record.summary())
        result = episode.result()
        table.append(result.metrics, trace=result.trace())
        print(result.metrics.summary())
        if video.path is not None:
            print(video.summary())


def score_checkpoint(config, tasks, policy, checkpoint_name, table,
                     selected_gpu, film, verbose=True):
    """Run every task for one checkpoint, one scene's simulator at a time.

    The world is opened from the first task's own `world.yaml` — the file P2
    wrote when it drove the trail — so an episode runs in the world its task
    was defined in, rather than in whatever a config happens to say today.
    """
    import bridge

    for scene, scene_tasks in task_set.group_by_scene(tasks).items():
        body = bridge.SimBody(config_path=scene_tasks[0].world_config,
                              floor=config.floor)
        try:
            body.verify_gpu(selected_gpu)
            runner = episode_runner.EpisodeRunner(
                policy, body, rules=config.rules,
                seed_offset=config.seed_offset)
            run_scene(scene, scene_tasks, runner, checkpoint_name, table,
                      film, verbose=verbose)
        finally:
            body.close()


def build_recorder(config, policy):
    """P4's recorder for this run, or a disabled one.

    The waypoint scale is read off the checkpoint that is about to drive — the
    overlay draws the model's own samples in metres, and a checkpoint trained
    without normalization means them in metres already.
    """
    import bridge
    import pd_control

    if not config.recording.enabled:
        return recording.disabled()
    return recording.Recorder(
        config.recording,
        waypoint_scale_m=bridge.waypoint_scale_m(
            policy.model_params, pd_control.RobotLimits.from_config()),
        success_radius_m=config.rules.success_radius_m)


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
    print("driver:     {}".format(config.driver.summary()))
    print("recording:  {}".format(config.recording.summary()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:     {}".format(device))
    policy = NomadPolicy(spec, device, config.driver)

    table = metrics.MetricsTable(csv_path).open(resume=resume)
    score_checkpoint(config, tasks, policy, checkpoint_name, table,
                     selected_gpu, build_recorder(config, policy), verbose=verbose)

    print()
    print(metrics.format_aggregate(
        checkpoint_name, metrics.aggregate(metrics.read_table(csv_path))))
    print()
    print("wrote:      {}".format(csv_path))
    print("            {}".format(table.trace_path))
    return table


def main():
    args = parse_args()

    # Before importing iGibson or torch: both read the device variables when
    # they initialise, not when they are used.
    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))

    config = EvalConfig.from_yaml(args.config)
    print("config:     {}".format(args.config))
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
            TopomapError) as error:
        raise SystemExit("FAILED: {}".format(error))
