"""The headline table: every checkpoint of a run, side by side, mean ± SE.

P6's last step, and GPU-free: it reads the per-episode tables the scoring runs
wrote and nothing else. Before it reports a number it checks the comparison is
one (plan §7) — every arm has a row for every task in the task set under every
seed offset of the run, the same seeds for every arm, and every row was steered
and judged under the same settings. A table that fails that check is refused,
not printed with a footnote (`comparison.load_fair_tables`, the check
`p7_contrasts.py` makes too). Any number of arms: one column each.

    ./sim_eval/run_p6_headline.sh           # runs this at the end
    ./sim_eval/run_p6_2_compare.sh          # or on its own, after the fact

Writes three files beside the metrics, named by the config's `comparison:`
section (P6's own names when it has none):

    <name>_comparison.md    the table to read: mean ± SE (n) per checkpoint,
                            over every house and then house by house
    <name>_comparison.csv   the same numbers, one row per
                            (scope, statistic, checkpoint) — scope is `all` or
                            a house
    <name>_episodes.csv     every per-episode row of every arm, task by task
    <name>_tasks.csv        each arm's value per task, its seeds averaged —
                            the unit every mean and SE here is taken over

The task, not the episode, is the unit (`task_stats`): a task's seeds are
averaged first, and the mean ± SE is across tasks. With one seed per task —
every run up to P6 — that is the same number the episodes give.
"""

import argparse
from pathlib import Path

import checkpoints
import comparison
import metrics
import run_eval
import task_set
import task_stats
from comparison import ALL_HOUSES, ComparisonNaming

DEFAULT_CONFIG = run_eval.SIM_EVAL_DIR / "configs" / "p6_headline.yaml"


def comparison_rows(tables, scope=ALL_HOUSES):
    """One row per (statistic, checkpoint): mean, SE and n tasks, for the CSV."""
    rows = []
    summaries = {name: task_stats.summarize(table)
                 for name, table in tables.items()}
    for statistic, _column, _over in metrics.EPISODE_STATISTICS:
        for name, summary in summaries.items():
            mean, se, count = summary[statistic]
            rows.append({"scope": scope, "statistic": statistic, "checkpoint": name,
                         "mean": comparison.round_cell(mean), "se": comparison.round_cell(se), "n": count})
    return rows


def per_house_rows(tables):
    """`comparison_rows` again for each house on its own: whether a ranking
    holds in every house or is carried by one."""
    rows = []
    for house in comparison.houses(tables):
        in_house = {name: [row for row in table if row["scene"] == house]
                    for name, table in tables.items()}
        rows += comparison_rows(in_house, scope=house)
    return rows


def format_cell(mean, se, count):
    """`0.450 ± 0.114 (20)` — or what is missing, said plainly."""
    if count == 0:
        return "n/a (0)"
    if se in ("", None):
        return "{:.3f} ± n/a ({})".format(float(mean), count)
    return "{:.3f} ± {:.3f} ({})".format(float(mean), float(se), count)


def statistics_table(rows, names):
    """One scope's rows as Markdown table lines, one column per checkpoint."""
    cells = {(row["statistic"], row["checkpoint"]): row for row in rows}
    lines = ["| statistic | " + " | ".join(names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for statistic, _column, _over in metrics.EPISODE_STATISTICS:
        lines.append("| {} | {} |".format(statistic, " | ".join(
            format_cell(cells[statistic, name]["mean"], cells[statistic, name]["se"],
                        cells[statistic, name]["n"])
            for name in names)))
    return lines


def markdown_table(rows, names, provenance, title=ComparisonNaming().title):
    """The comparison as Markdown: every house together, then each alone."""
    lines = ["# " + title, "",
             "mean ± SE across tasks (n = tasks); a task's seeds are averaged first", ""]
    lines += ["- **{}**: {}".format(name, provenance[name]) for name in names]
    scopes = list(dict.fromkeys(row["scope"] for row in rows))
    for scope in scopes:
        heading = "All houses" if scope == ALL_HOUSES else "House: " + scope
        lines += ["", "## " + heading, ""]
        lines += statistics_table([row for row in rows if row["scope"] == scope], names)
    return "\n".join(lines) + "\n"


def episodes_side_by_side(tables):
    """Every arm's rows in one list, ordered task by task and seed by seed,
    then by arm — so one episode's arms sit on adjacent rows."""
    order = {name: index for index, name in enumerate(tables)}
    rows = [row for table in tables.values() for row in table]
    return sorted(rows, key=lambda row: (row["task_id"], int(row["seed"]),
                                         order[row["checkpoint"]]))


def per_task_rows(tables):
    """Each arm's per-task value of every statistic: what the means average."""
    rows = []
    for name, table in tables.items():
        scene_of = {row["task_id"]: row["scene"] for row in table}
        seeds = task_stats.seed_counts(table)
        values = {statistic: task_stats.task_values(table, statistic)
                  for statistic, _column, _over in metrics.EPISODE_STATISTICS}
        for task in sorted(scene_of):
            row = {"checkpoint": name, "scene": scene_of[task], "task_id": task,
                   "seeds": seeds[task]}
            row.update({statistic: comparison.round_cell(per_task.get(task))
                        for statistic, per_task in values.items()})
            rows.append(row)
    return rows


def provenance(name):
    """What an arm is, in one line — its render resolution and context
    stride read from its own training config (plan §7), not typed in."""
    spec = checkpoints.load(name)
    return "{}x{}, context {} (stride {}) — {}".format(
        spec.image_size[0], spec.image_size[1], spec.context_size,
        spec.context_stride, spec.weights_path)


def compare(config, naming=None):
    """Check the run is a comparison, then write the four files. Returns the
    Markdown table."""
    naming = naming or ComparisonNaming()
    names = config.checkpoint_names
    tables = comparison.load_fair_tables(
        config, {name: config.csv_path(name) for name in names})

    rows = comparison_rows(tables) + per_house_rows(tables)
    table = markdown_table(rows, names, {name: provenance(name) for name in names},
                           title=naming.title)
    output = config.output_dir
    (output / naming.markdown).write_text(table)
    comparison.write_csv(output / naming.csv, rows,
                         ("scope", "statistic", "checkpoint", "mean", "se", "n"))
    comparison.write_csv(output / naming.episodes, episodes_side_by_side(tables),
                         metrics.CSV_COLUMNS)
    comparison.write_csv(output / naming.tasks, per_task_rows(tables),
                         ("checkpoint", "scene", "task_id", "seeds")
                         + tuple(statistic for statistic, _c, _o
                                 in metrics.EPISODE_STATISTICS))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="the run's eval config (default: %(default)s)")
    args = parser.parse_args()

    config = run_eval.EvalConfig.from_yaml(args.config)
    naming = ComparisonNaming.from_yaml(args.config)
    print(compare(config, naming))
    for name in naming.files:
        print("wrote:      {}".format(config.output_dir / name))


if __name__ == "__main__":
    try:
        main()
    except (comparison.ComparisonError, task_set.TaskSetError, checkpoints.CheckpointError,
            FileNotFoundError) as error:
        raise SystemExit("FAILED: {}".format(error))
