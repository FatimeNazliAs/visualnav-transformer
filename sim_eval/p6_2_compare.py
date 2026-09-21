"""The headline table: every checkpoint of a run, side by side, mean ± SE.

P6's last step, and GPU-free: it reads the per-episode tables the scoring runs
wrote and nothing else. Before it reports a number it checks the comparison is
one (plan §7) — every arm has a row for every task in the task set, under the
same seed, and every row was steered and judged under the same settings. A
table that fails that check is refused, not printed with a footnote.

    ./sim_eval/run_p6_headline.sh           # runs this at the end
    ./sim_eval/run_p6_2_compare.sh          # or on its own, after the fact

Writes three files beside the metrics:

    p6_2_comparison.md    the table to read: mean ± SE (n) per checkpoint
    p6_2_comparison.csv   the same numbers, one row per (statistic, checkpoint)
    p6_2_episodes.csv     every per-episode row of every arm, task by task
"""

import argparse
import csv
from pathlib import Path

import checkpoints
import metrics
import run_eval
import task_set

DEFAULT_CONFIG = run_eval.SIM_EVAL_DIR / "configs" / "p6_headline.yaml"
COMPARISON_MD = "p6_2_comparison.md"
COMPARISON_CSV = "p6_2_comparison.csv"
EPISODES_CSV = "p6_2_episodes.csv"

# Columns that must read the same in every row of every arm: what steered the
# robot and what counted as arriving. A difference in any of them means the
# arms were not measured with the same ruler.
SHARED_SETTINGS = ("driver", "success_radius_m", "success_metric")


class ComparisonError(RuntimeError):
    """Raised when the tables do not add up to a fair comparison."""


def expected_episodes(tasks, seed_offset):
    """(task_id, seed) for every task in the set — what each arm must have run."""
    return {(task.task_id, task.seed + seed_offset) for task in tasks}


def fairness_problems(tables, expected):
    """Everything that stops `tables` ({checkpoint: rows}) being a comparison.

    An empty list means every arm ran exactly the task set, once per task,
    under the same seeds and the same settings.
    """
    problems = []
    for name, rows in tables.items():
        ran = [(row["task_id"], int(row["seed"])) for row in rows]
        missing = sorted(expected - set(ran))
        extra = sorted(set(ran) - expected)
        repeated = sorted({episode for episode in ran if ran.count(episode) > 1})
        for label, episodes in (("missing", missing), ("not in the task set", extra),
                                ("scored twice", repeated)):
            if episodes:
                problems.append("{}: {} episode(s) {}: {}".format(
                    name, len(episodes), label,
                    ", ".join("{} (seed {})".format(*episode) for episode in episodes)))

    all_rows = [row for rows in tables.values() for row in rows]
    for column in SHARED_SETTINGS:
        values = sorted({str(row[column]) for row in all_rows})
        if len(values) > 1:
            problems.append("`{}` differs between rows: {}".format(
                column, ", ".join(values)))
    return problems


def comparison_rows(tables):
    """One row per (statistic, checkpoint): mean, SE and n, for the CSV."""
    rows = []
    summaries = {name: metrics.summarize(table)
                 for name, table in tables.items()}
    for statistic, _column, _over in metrics.EPISODE_STATISTICS:
        for name, summary in summaries.items():
            mean, se, count = summary[statistic]
            rows.append({"statistic": statistic, "checkpoint": name,
                         "mean": _round(mean), "se": _round(se), "n": count})
    return rows


def _round(value, places=4):
    return "" if value is None else round(value, places)


def format_cell(mean, se, count):
    """`0.450 ± 0.114 (20)` — or what is missing, said plainly."""
    if count == 0:
        return "n/a (0)"
    if se in ("", None):
        return "{:.3f} ± n/a ({})".format(float(mean), count)
    return "{:.3f} ± {:.3f} ({})".format(float(mean), float(se), count)


def markdown_table(rows, names, provenance):
    """The comparison as a Markdown table, one column per checkpoint."""
    cells = {(row["statistic"], row["checkpoint"]): row for row in rows}
    lines = ["# P6 headline comparison", "", "mean ± SE across tasks (n)", ""]
    lines += ["- **{}**: {}".format(name, provenance[name]) for name in names]
    lines += ["", "| statistic | " + " | ".join(names) + " |",
              "|---" * (len(names) + 1) + "|"]
    for statistic, _column, _over in metrics.EPISODE_STATISTICS:
        lines.append("| {} | {} |".format(statistic, " | ".join(
            format_cell(cells[statistic, name]["mean"], cells[statistic, name]["se"],
                        cells[statistic, name]["n"])
            for name in names)))
    return "\n".join(lines) + "\n"


def episodes_side_by_side(tables):
    """Every arm's rows in one list, ordered task by task, then by arm."""
    order = {name: index for index, name in enumerate(tables)}
    rows = [row for table in tables.values() for row in table]
    return sorted(rows, key=lambda row: (row["task_id"], order[row["checkpoint"]]))


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def provenance(name):
    """What an arm is, in one line — its render resolution read from its own
    training config (plan §7), not typed in."""
    spec = checkpoints.load(name)
    return "{}x{}, context {} — {}".format(
        spec.image_size[0], spec.image_size[1], spec.context_size,
        spec.weights_path)


def compare(config):
    """Check the run is a comparison, then write the three files. Returns the
    Markdown table."""
    _manifest, tasks = task_set.load(config.task_directory)
    names = config.checkpoint_names
    tables = {name: metrics.read_table(config.csv_path(name)) for name in names}

    problems = fairness_problems(
        tables, expected_episodes(tasks, config.seed_offset))
    if problems:
        raise ComparisonError(
            "these tables are not a fair comparison yet:\n  " + "\n  ".join(problems))

    rows = comparison_rows(tables)
    table = markdown_table(rows, names, {name: provenance(name) for name in names})
    output = config.output_dir
    (output / COMPARISON_MD).write_text(table)
    write_csv(output / COMPARISON_CSV, rows, ("statistic", "checkpoint", "mean", "se", "n"))
    write_csv(output / EPISODES_CSV, episodes_side_by_side(tables), metrics.CSV_COLUMNS)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="the run's eval config (default: %(default)s)")
    args = parser.parse_args()

    config = run_eval.EvalConfig.from_yaml(args.config)
    print(compare(config))
    for name in (COMPARISON_MD, COMPARISON_CSV, EPISODES_CSV):
        print("wrote:      {}".format(config.output_dir / name))


if __name__ == "__main__":
    try:
        main()
    except (ComparisonError, task_set.TaskSetError, checkpoints.CheckpointError,
            FileNotFoundError) as error:
        raise SystemExit("FAILED: {}".format(error))
