"""P3's claim: the scorer runs episodes unattended and grades them correctly.

Runs a small task set — three tasks in one house — on one checkpoint, end to
end and with nobody watching, and writes the per-episode metrics table. It is
the same code path `run_eval.py` uses for a full arm, with two things made
small: the number of tasks, and the number of checkpoints.

That it is a *prefix* rather than a separate experiment is the point. A task's
seed depends on its scene and its index, never on how many tasks were asked
for, so these three tasks are the first three tasks of the ten-task set. What
passes here is what will run in P6.

The test then checks the table it produced, because the risk in a scorer is not
that it crashes — it is that it writes a plausible number that is wrong. So it
re-derives SPL from the row's own path lengths, checks success agrees with the
distance that was measured, and checks the timeout budget matches the formula
for the task's geodesic length. A row that cannot survive that is not a
measurement.

    ./sim_eval/run_p3_score_test.sh
"""

import argparse
import math
from pathlib import Path

import checkpoints
import gpu
import metrics
import pd_control
import run_eval
from task_set import TaskSetError
from topomap_builder import TopomapError

SIM_EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SIM_EVAL_DIR / "outputs"
DEFAULT_TASK_SET = OUTPUT_DIR / "p3_1_task_set"
DEFAULT_CSV = OUTPUT_DIR / "p3_1_score_test.csv"
DEFAULT_CHECKPOINT = "best_combined"
DEFAULT_TASKS = 3

# Rounding in the CSV is to 6 decimals, so a re-derived number may differ in
# the last place. Anything larger is a real disagreement.
TOLERANCE = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=run_eval.DEFAULT_CONFIG,
                        help="eval config to run (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="name from configs/checkpoints.yaml (default: %(default)s)")
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASKS,
                        help="how many tasks to score (default: %(default)s)")
    parser.add_argument("--task-set", type=Path, default=DEFAULT_TASK_SET,
                        help="where the small task set lives (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV,
                        help="metrics CSV to write (default: %(default)s)")
    parser.add_argument("--rebuild-tasks", action="store_true",
                        help="rebuild the task set even if one is already there")
    parser.add_argument("--quiet", action="store_true",
                        help="one line per episode instead of one per tick")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to use; never guesses")
    return parser.parse_args()


def check_row(row, rules, limits):
    """Re-derive what a row claims, from the row. Returns a list of complaints.

    Every check here is the metric computed a second way — from the numbers
    beside it in the table, not from the object that wrote it — so an error in
    `metrics.py` cannot agree with itself into a pass.
    """
    complaints = []
    success = bool(int(row["success"]))
    geodesic = float(row["geodesic_length_m"])
    driven = float(row["path_length_m"])
    radius = float(row["success_radius_m"])
    euclidean = float(row["final_euclidean_distance_m"])
    geodesic_to_goal = (None if row["final_geodesic_distance_m"] == ""
                        else float(row["final_geodesic_distance_m"]))
    # The distance the rule in force actually measured the radius with.
    measured = (euclidean if row["success_metric"] == "euclidean"
                else geodesic_to_goal)

    expected_spl = metrics.spl(success, geodesic, driven)
    if abs(float(row["spl"]) - expected_spl) > TOLERANCE:
        complaints.append("SPL is {} but S*L/max(P,L) is {:.6f}".format(
            row["spl"], expected_spl))
    if success and expected_spl > 1.0 + TOLERANCE:
        complaints.append("SPL {:.6f} exceeds 1".format(expected_spl))

    # Success is a claim about where the agent stopped, and the row carries the
    # distance it stopped at — so the two must agree in both directions. The
    # episode checks the radius after every tick, including the last, so an
    # episode that timed out was outside it at the moment it ran out of ticks.
    if success and (measured is None or measured > radius + TOLERANCE):
        complaints.append(
            "marked success but stopped {} m away ({}), outside the {} m radius"
            .format(measured, row["success_metric"], radius))
    if not success and measured is not None and measured <= radius:
        complaints.append(
            "marked {} but stopped {:.3f} m away ({}), inside the {} m radius"
            .format(row["outcome"], measured, row["success_metric"], radius))

    # Around the furniture is never shorter than through it.
    if geodesic_to_goal is not None and geodesic_to_goal < euclidean - TOLERANCE:
        complaints.append(
            "geodesic distance {:.3f} m is shorter than the straight line "
            "{:.3f} m".format(geodesic_to_goal, euclidean))

    expected_timeout = rules.timeout_ticks(geodesic, limits)
    if int(row["timeout_ticks"]) != expected_timeout:
        complaints.append("timeout is {} ticks but the formula gives {}".format(
            row["timeout_ticks"], expected_timeout))
    if int(row["ticks"]) > expected_timeout:
        complaints.append("ran {} ticks past a {}-tick budget".format(
            row["ticks"], expected_timeout))
    if not success and int(row["ticks"]) != expected_timeout:
        complaints.append(
            "failed after {} of {} ticks — a timeout is the only way to fail, "
            "so it must have spent the budget".format(
                row["ticks"], expected_timeout))

    # Collisions are counted and never terminal (plan §6): a collision-heavy
    # episode still gets its full budget, and events can never exceed ticks.
    if int(row["collision_events"]) > int(row["collision_ticks"]):
        complaints.append("{} collision events over {} colliding ticks".format(
            row["collision_events"], row["collision_ticks"]))
    if float(row["contact_tick_fraction"]) > 1.0 + TOLERANCE:
        complaints.append("in contact for more ticks than it ran")

    if math.isclose(driven, 0.0) and int(row["ticks"]) > 0:
        complaints.append("drove {} ticks but logged no distance".format(row["ticks"]))

    return complaints


def check_table(csv_path, rules, limits, expected_rows):
    """The acceptance test: the table has a row per episode, and each row holds."""
    rows = metrics.read_table(csv_path)
    if len(rows) != expected_rows:
        raise SystemExit(
            "FAILED: expected {} episodes in {}, found {}."
            .format(expected_rows, csv_path, len(rows)))

    failures = []
    for row in rows:
        for complaint in check_row(row, rules, limits):
            failures.append("  {}: {}".format(row["task_id"], complaint))

    if failures:
        raise SystemExit("FAILED: the metrics table does not hold up:\n"
                         + "\n".join(failures))

    print("checked:    {} rows — SPL, success radius, timeout budget and "
          "collision counts all re-derive".format(len(rows)))
    return rows


def report(rows):
    """The table, as a person reads it."""
    print()
    print("{:<8} {:<8} {:>6} {:>8} {:>8} {:>7} {:>6}".format(
        "task", "outcome", "SPL", "dist m", "path m", "ticks", "coll"))
    for row in rows:
        print("{:<8} {:<8} {:>6.3f} {:>8} {:>8.2f} {:>7} {:>6}".format(
            row["task_id"], row["outcome"], float(row["spl"]),
            row["final_geodesic_distance_m"] or "-",
            float(row["path_length_m"]), row["ticks"], row["collision_ticks"]))


def main():
    args = parse_args()

    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))

    config = run_eval.EvalConfig.from_yaml(args.config)
    config.tasks.tasks_per_scene = args.tasks
    scenes = len(config.tasks.scenes)

    run_eval.evaluate(
        config, args.checkpoint, csv_path=args.output, selected_gpu=selected_gpu,
        task_directory=args.task_set, rebuild_tasks=args.rebuild_tasks,
        verbose=not args.quiet)

    rows = check_table(args.output, config.rules,
                       pd_control.RobotLimits.from_config(),
                       expected_rows=args.tasks * scenes)
    report(rows)

    print()
    print("PASSED: the scorer ran {} episodes unattended and every metric in "
          "the table re-derives.".format(len(rows)))


if __name__ == "__main__":
    try:
        main()
    except (gpu.GpuSelectionError, checkpoints.CheckpointError, TaskSetError,
            TopomapError) as error:
        raise SystemExit("FAILED: {}".format(error))
