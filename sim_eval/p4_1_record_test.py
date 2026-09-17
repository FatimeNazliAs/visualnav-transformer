"""P4's claim: any episode can be replayed as a three-panel video, for free.

Runs **one** episode twice on the same task, same seed and same checkpoint:
once with the recorder on and once with it off. That is the whole phase in one
experiment, because P4 makes two claims and they pull against each other —

  * the recorded run must produce a watchable MP4 whose frames line up with the
    ticks that were scored, and
  * the unrecorded run must be the run P3 already had: same metrics, same
    numbers, to the last decimal.

The second is the one worth testing mechanically. A recorder that draws from
the same stream the scorer scores could so easily consume a frame, advance a
generator or reseed something and move a metric by a hair, and nothing in the
table would look wrong — it would just be a different number than the one P3
would have written. So the two rows are compared column for column, and any
disagreement fails the phase.

What comes out (from `sim_eval/outputs/`):

    videos/<checkpoint>/<task_id>.mp4   the three-panel replay, to scrub
    p4_1_record_test.csv                the recorded run's metrics row
    p4_1_unrecorded.csv                 the same episode, unfilmed

Run (from the repo root, on the host):
    ./sim_eval/run_p4_record_test.sh
"""

import argparse
import time
from pathlib import Path

import checkpoints
import gpu
import metrics
import recorder
import run_eval
from task_set import TaskSetError, task_id as make_task_id
from topomap_builder import TopomapError

SIM_EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SIM_EVAL_DIR / "outputs"
DEFAULT_TASK_SET = OUTPUT_DIR / "p4_1_task_set"
DEFAULT_CSV = OUTPUT_DIR / "p4_1_record_test.csv"
UNRECORDED_CSV = OUTPUT_DIR / "p4_1_unrecorded.csv"
DEFAULT_CHECKPOINT = "best_combined"

# One task, and it is the first task of every set built from this base seed —
# `Rs_00`, the episode P3 scored as a success in 58 ticks. Short enough to
# watch in one sitting, and it reaches its goal, so the video shows the whole
# arc rather than a robot stuck against a wall.
TASKS = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=run_eval.DEFAULT_CONFIG,
                        help="eval config to run (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="name from configs/checkpoints.yaml (default: %(default)s)")
    parser.add_argument("--fps", type=int, default=None,
                        help="video frame rate; the sim runs at 4 Hz, so 4 is "
                             "real time (default: from the config)")
    parser.add_argument("--task-set", type=Path, default=DEFAULT_TASK_SET,
                        help="where the one-task set lives (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV,
                        help="metrics CSV for the recorded run (default: %(default)s)")
    parser.add_argument("--rebuild-tasks", action="store_true",
                        help="rebuild the task set even if one is already there")
    parser.add_argument("--skip-off-check", action="store_true",
                        help="only run the recorded episode, and do not prove "
                             "that recording off leaves the metrics alone")
    parser.add_argument("--quiet", action="store_true",
                        help="one line per episode instead of one per tick")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to use; never guesses")
    return parser.parse_args()


def score_once(config, args, selected_gpu, csv_path, record):
    """Run the one episode, filmed or not, and return (row, wall-clock seconds).

    The clock covers the whole call — opening the simulator, loading the
    checkpoint, the episode — so it is a ceiling on the cost of recording
    rather than a measurement of it. Both calls pay the same setup, because
    the trail they share was built before either of them started.
    """
    config.recording.enabled = record
    started = time.time()
    run_eval.evaluate(
        config, args.checkpoint, csv_path=csv_path, selected_gpu=selected_gpu,
        task_directory=args.task_set, verbose=not args.quiet)
    elapsed = time.time() - started

    rows = metrics.read_table(csv_path)
    if len(rows) != 1:
        raise SystemExit("FAILED: expected 1 episode in {}, found {}.".format(
            csv_path, len(rows)))
    return rows[0], elapsed


def check_video(path, row, fps):
    """The acceptance test: a readable MP4 with one frame per scored tick.

    Frame count is the sync claim made mechanical. Every panel in a frame is
    drawn from one `TickRecord`, so they cannot disagree with each other; what
    they *could* disagree with is the episode, by dropping or doubling a frame
    somewhere in the encoder. One frame per tick is the proof that they do not.
    """
    import imageio

    if not path.exists():
        raise SystemExit("FAILED: no video at {}.".format(path))

    reader = imageio.get_reader(str(path))
    try:
        frames = reader.count_frames()
        first = reader.get_data(0)
    finally:
        reader.close()

    ticks = int(row["ticks"])
    if frames != ticks:
        raise SystemExit(
            "FAILED: {} holds {} frames but the episode ran {} ticks — the "
            "panels are not in sync with the run.".format(path, frames, ticks))

    height, width = first.shape[:2]
    expected = (recorder.FRAME_HEIGHT_PX, recorder.FRAME_WIDTH_PX)
    if (height, width) != expected:
        raise SystemExit(
            "FAILED: {} is {}x{}, not the {}x{} that was drawn — ffmpeg "
            "resized the canvas.".format(path, width, height, expected[1],
                                         expected[0]))

    megabytes = path.stat().st_size / 1e6
    print("video:      {}".format(path))
    print("            {} frames at {} fps = {:.1f} s of video, {}x{}, {:.1f} MB"
          .format(frames, fps, frames / fps, width, height, megabytes))
    return frames


def check_metrics_unchanged(recorded, unrecorded):
    """Recording off must leave P3's table exactly as it was."""
    differences = [
        "  {}: {!r} recorded, {!r} not".format(column, recorded[column], value)
        for column, value in unrecorded.items()
        if recorded.get(column) != value
    ]
    if differences:
        raise SystemExit(
            "FAILED: filming changed the episode — these columns disagree:\n"
            + "\n".join(differences))
    print("unchanged:  every column of the metrics row is identical filmed "
          "and unfilmed")


def report(row, recorded_s, unrecorded_s):
    print()
    print("episode:    {} · {} · {} ticks · {}".format(
        row["task_id"], row["checkpoint"], row["ticks"], row["outcome"]))
    print("            SPL {} · {} m from the goal · {} collision ticks".format(
        row["spl"], row["final_geodesic_distance_m"] or "-",
        row["collision_ticks"]))
    if unrecorded_s is None:
        return
    print("wall clock: {:.1f} s filmed, {:.1f} s not (setup included in both)"
          .format(recorded_s, unrecorded_s))


def main():
    args = parse_args()

    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))

    config = run_eval.EvalConfig.from_yaml(args.config)
    config.tasks.tasks_per_scene = TASKS
    if args.fps is not None:
        config.recording.fps = args.fps
    # Named rather than left as "all", so the subset knob is exercised by the
    # phase's own test rather than only documented.
    task_id = make_task_id(config.tasks.scenes[0], 0)
    config.recording.tasks = [task_id]

    # Build the trail before either run is timed. It is P2 work, it happens
    # once, and leaving it inside the first run would charge the recorder for
    # a reference drive and make the two wall clocks incomparable.
    run_eval.evaluate(config, args.checkpoint, csv_path=args.output,
                      selected_gpu=selected_gpu, task_directory=args.task_set,
                      rebuild_tasks=args.rebuild_tasks, build_only=True)

    print("\n=== the episode, filmed ===")
    recorded_row, recorded_s = score_once(
        config, args, selected_gpu, args.output, record=True)
    video = config.recording.directory / args.checkpoint / "{}.mp4".format(task_id)
    check_video(video, recorded_row, config.recording.fps)

    unrecorded_s = None
    if not args.skip_off_check:
        print("\n=== the same episode, unfilmed ===")
        unrecorded_row, unrecorded_s = score_once(
            config, args, selected_gpu, UNRECORDED_CSV, record=False)
        check_metrics_unchanged(recorded_row, unrecorded_row)

    report(recorded_row, recorded_s, unrecorded_s)

    print()
    print("PASSED: the episode replays as a three-panel video, and recording "
          "off leaves the scoring exactly as P3 left it.")


if __name__ == "__main__":
    try:
        main()
    except (gpu.GpuSelectionError, checkpoints.CheckpointError, TaskSetError,
            TopomapError) as error:
        raise SystemExit("FAILED: {}".format(error))
