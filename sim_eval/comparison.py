"""What makes a set of per-episode tables a comparison, and how one is named.

Two scripts read the tables a scoring run wrote: `p6_2_compare.py` sets each
arm's means side by side, and `p7_contrasts.py` pairs arms task by task. Both
must refuse the same things before they report a number (plan §7) — an arm
short of an episode, an episode scored twice, two arms steered differently —
so the refusal lives here, once, behind `load_fair_tables`.

    tables = load_fair_tables(eval_config, {"ctx03": path, "bc30": path})

Also here, because both scripts write them: the `comparison:` naming every
file is written under, the house list the per-house splits iterate, and the
CSV writer.
"""

import csv

import episode_runner
import metrics
import run_eval
import task_set

ALL_HOUSES = "all"

# Columns that must read the same in every row of every arm: what steered the
# robot and what counted as arriving. A difference in any of them means the
# arms were not measured with the same ruler.
SHARED_SETTINGS = ("driver", "success_radius_m", "success_metric")


class ComparisonError(RuntimeError):
    """Raised when the tables do not add up to a fair comparison."""


class ComparisonNaming:
    """What a comparison's files and heading are called — a config's
    `comparison:` section, so a P7 run writes p7_* files, not P6's."""

    def __init__(self, name="p6_2", title="P6 headline comparison"):
        self.name = name
        self.title = title

    @classmethod
    def from_yaml(cls, path):
        return cls(**(run_eval.read_layered_yaml(path).get("comparison") or {}))

    @property
    def markdown(self):
        return "{}_comparison.md".format(self.name)

    @property
    def csv(self):
        return "{}_comparison.csv".format(self.name)

    @property
    def episodes(self):
        return "{}_episodes.csv".format(self.name)

    @property
    def tasks(self):
        return "{}_tasks.csv".format(self.name)

    @property
    def files(self):
        return (self.markdown, self.csv, self.episodes, self.tasks)

    def output(self, suffix):
        """`<name>_<suffix>` — the files `p7_contrasts.py` adds beside these."""
        return "{}_{}".format(self.name, suffix)


def expected_episodes(tasks, seed_offsets):
    """(task_id, seed) for every task under every offset — tasks x seeds, what
    each arm must have run. The seed comes from the runner's own rule."""
    return {(task.task_id, episode_runner.episode_seed(task, offset))
            for task in tasks for offset in seed_offsets}


def fairness_problems(tables, expected):
    """Everything that stops `tables` ({checkpoint: rows}) being a comparison.

    An empty list means every arm ran exactly the task set, once per
    (task, seed), under the same seeds and the same settings.
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


def restrict(rows, expected):
    """Only the rows whose (task, seed) is one of `expected`."""
    return [row for row in rows if (row["task_id"], int(row["seed"])) in expected]


def load_fair_tables(eval_config, table_paths, restrict_to_task_set=False):
    """{name: rows} for every table in `table_paths`, in its order — or
    `ComparisonError` naming everything that stops them being a comparison.

    The expectation is the config's own: its task set, under its seed offsets.
    `restrict_to_task_set` first cuts each table down to exactly those
    episodes, for a table scored on a larger set that contains this one (see
    `p7_contrasts.ContrastsConfig`); missing episodes are refused either way.
    """
    _manifest, tasks = task_set.load(eval_config.task_directory)
    expected = expected_episodes(tasks, eval_config.seed_offsets)
    tables = {name: metrics.read_table(path) for name, path in table_paths.items()}
    if restrict_to_task_set:
        tables = {name: restrict(rows, expected) for name, rows in tables.items()}
    problems = fairness_problems(tables, expected)
    if problems:
        raise ComparisonError(
            "these tables are not a fair comparison yet:\n  " + "\n  ".join(problems))
    return tables


def houses(tables):
    """The houses the tables cover, in the order they first appear."""
    seen = {}
    for rows in tables.values():
        for row in rows:
            seen.setdefault(row["scene"], None)
    return list(seen)


def round_cell(value, places=4):
    """A number as a CSV cell: rounded, or blank when there is none."""
    return "" if value is None else round(value, places)


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
