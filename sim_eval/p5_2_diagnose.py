"""A few episodes, filmed and taken apart — P5's second question.

P0-P4 built a ruler. It works, and the first thing it measured was one success
in three. Plan §9 calls P5 "confirm NoMaD behaves sanely despite the domain
gap", and the only way to confirm or deny that is to watch a handful of
episodes closely rather than to run twenty and average them.

So this runs a **small** set — three to five tasks in one scene, one
checkpoint — with the recorder on, and writes a bundle that answers "why" for
each one:

    <checkpoint>.csv              the metrics table P3 would have written
    <checkpoint>.jsonl            the same episodes with their pose traces
    ticks/<task_id>.csv           one row per control tick, per episode
    videos/<checkpoint>/*.mp4     P4's three-panel replay of each
    p5_2_diagnosis.csv            one named failure mode per episode

**It tunes nothing and changes nothing.** The driver settings are the config's
(`navigate.py`'s defaults — plan decision E), the episode rules are the
config's, and the task set is built the way P2 builds every task set. The only
departures from `run_eval.py` are that the set is small, the recorder is on,
and the traces are read back afterwards and named. A run of this and a run of
`run_eval.sh --record` over the same tasks produce the same rows.

The per-tick CSV is the part P3 and P4 did not have. The `.jsonl` already
carried most of it, but nested inside a JSON object per episode, which is not
something you can sort, plot or scroll. One flat file per episode is — and it
is where the two columns P5 added live: what the distance head read for the
node it localized onto, and for the node it steered at. Those decide whether
the trail advances at all, and until now they existed only as a number drawn on
a video frame.

Run (from the repo root, on the host):
    ./sim_eval/run_p5_2_diagnose.sh
    ./sim_eval/run_p5_2_diagnose.sh --checkpoint clean_stock --tasks 5
"""

import argparse
import csv
import json
from pathlib import Path

import checkpoints
import diagnosis
import gpu
import run_eval
import task_set
from task_set import TaskSetError
from topomap_builder import TopomapError

SIM_EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SIM_EVAL_DIR / "outputs"
DEFAULT_BUNDLE = OUTPUT_DIR / "p5_2_diagnostics"
DEFAULT_TASK_SET = OUTPUT_DIR / "p5_2_task_set"
DEFAULT_CHECKPOINT = "best_combined"

# Small on purpose. The point is to watch every one of them, and a set nobody
# opens is a set that taught nothing. Plan §B's twenty is P6's number.
DEFAULT_TASKS = 3

# The per-tick log, in the order a person reads it: when, where, what the model
# thought, what it decided, what the robot did.
TICK_COLUMNS = ("tick", "x", "y", "yaw", "node", "subgoal", "dist_closest",
                "dist_subgoal", "waypoint_x_m", "waypoint_y_m", "v", "w",
                "collided", "contact_bearing_deg")

DIAGNOSIS_COLUMNS = ("task_id", "mode", "explanation", "outcome", "ticks",
                     "final_distance_m", "path_length_m", "geodesic_length_m",
                     "path_ratio", "trail_progress", "contact_tick_fraction",
                     "collision_events", "declared_arrival_tick",
                     "final_displacement_m", "mean_v", "mean_abs_w",
                     "turning_tick_fraction", "mean_dist_closest",
                     "min_dist_closest", "first_contact_tick",
                     "first_contact_bearing_deg", "first_contact_side",
                     "off_trail_at_first_contact_m",
                     "max_off_trail_before_contact_m",
                     "mean_abs_off_trail_before_contact_m",
                     "mean_signed_off_trail_before_contact_m",
                     "longest_one_side_run_ticks",
                     "head_lost_after_contact", "full_speed_after_contact")


def read_traces(path):
    """The episodes of one run, in the order they were scored."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_tick_log(trace, path):
    """One episode's ticks as a flat CSV — the per-step log.

    The pose is joined back on here. The trace keeps poses in their own array
    (one per tick, plus the pose the last tick ended at) because that is what
    redraws a path; a person reading a single tick wants it on the line.
    """
    poses = trace.get("poses") or []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TICK_COLUMNS)
        writer.writeheader()
        for tick in trace.get("ticks_log") or []:
            index = int(tick["tick"])
            pose = poses[index] if index < len(poses) else (None, None, None)
            waypoint = tick.get("waypoint_m") or (None, None)
            writer.writerow({
                "tick": index,
                "x": pose[0], "y": pose[1], "yaw": pose[2],
                "node": tick.get("node"),
                "subgoal": tick.get("subgoal"),
                "dist_closest": tick.get("dist_closest"),
                "dist_subgoal": tick.get("dist_subgoal"),
                "waypoint_x_m": waypoint[0], "waypoint_y_m": waypoint[1],
                "v": tick.get("v"), "w": tick.get("w"),
                "collided": int(bool(tick.get("collided"))),
                # Several contact points on one tick are usually one obstacle;
                # the CSV keeps them all, separated so the cell stays one cell.
                "contact_bearing_deg": ";".join(
                    str(value) for value in tick.get("contact_bearing_deg") or []),
            })
    return path


def write_diagnoses(diagnoses, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSIS_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for entry in diagnoses:
            writer.writerow(entry.as_row())
    return path


def report(diagnoses, driver, bundle):
    """The whole point of the phase, in one screen."""
    print("\n=== why each episode ended where it did ===")
    for entry in diagnoses:
        print("  " + entry.summary())

    print("\n=== modes ===")
    for mode, count in diagnosis.tally(diagnoses):
        print("  {:>2} x {}".format(count, mode))

    print("\n=== how the policy was steered ===")
    print("  {}".format(driver.summary()))
    if not driver.is_deployment_default():
        print("  NOTE: these are not navigate.py's defaults, so this run is "
              "not comparable with one that used them (plan decision E).")

    print("\nbundle:     {}".format(bundle))
    print("            watch the videos before reading the numbers — a mode is "
          "a label on\n            something you can see.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=run_eval.DEFAULT_CONFIG,
                        help="eval config to run (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="name from configs/checkpoints.yaml (default: %(default)s)")
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS,
                        help="how many tasks in the scene (default: %(default)s)")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE,
                        help="where the diagnostics go (default: %(default)s)")
    parser.add_argument("--task-set", type=Path, default=DEFAULT_TASK_SET,
                        help="where the task set lives (default: %(default)s)")
    parser.add_argument("--rebuild-tasks", action="store_true",
                        help="rebuild the task set even if one is already there")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="added to every task's seed, to replay the same "
                             "tasks under different diffusion noise and measure "
                             "run-to-run variation; 0 is the fair, scored "
                             "setting (default: %(default)s)")
    parser.add_argument("--no-record", dest="record", action="store_false",
                        default=True,
                        help="skip the videos (the numbers are the same either way)")
    parser.add_argument("--quiet", action="store_true",
                        help="one line per episode instead of one per tick")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to use; never guesses")
    return parser.parse_args()


def main():
    args = parse_args()

    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))

    config = run_eval.EvalConfig.from_yaml(args.config)
    config.tasks.tasks_per_scene = args.tasks
    config.recording.enabled = args.record
    config.recording.tasks = "all"
    config.recording.directory = args.bundle / "videos"
    config.seed_offset = args.seed_offset

    bundle = args.bundle
    bundle.mkdir(parents=True, exist_ok=True)
    csv_path = bundle / "{}.csv".format(args.checkpoint)

    table = run_eval.evaluate(
        config, args.checkpoint, csv_path=csv_path, selected_gpu=selected_gpu,
        task_directory=args.task_set, rebuild_tasks=args.rebuild_tasks,
        verbose=not args.quiet)

    traces = read_traces(table.trace_path)
    for trace in traces:
        write_tick_log(trace, bundle / "ticks" / "{}.csv".format(trace["task_id"]))
    # The trail each episode was asked to follow, so the diagnosis can say how
    # far off it the agent was when something first touched it.
    _manifest, tasks = task_set.load(args.task_set)
    reference_paths = {task.task_id: task.metadata["driven_path"] for task in tasks}
    diagnoses = diagnosis.diagnose_all(traces, reference_paths,
                                       config.driver.close_threshold)
    write_diagnoses(diagnoses, bundle / "p5_2_diagnosis.csv")

    report(diagnoses, config.driver, bundle)


if __name__ == "__main__":
    try:
        main()
    except (gpu.GpuSelectionError, checkpoints.CheckpointError, TaskSetError,
            TopomapError) as error:
        raise SystemExit("FAILED: {}".format(error))
