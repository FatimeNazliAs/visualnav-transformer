"""Statistics with the task as the unit of analysis.

Once a task is run under several seeds, its episodes are not independent: they
share a start, a goal, a trail and a house, and differ only in the diffusion
noise. Counting them as separate draws would shrink every standard error by
roughly the square root of the seed count and credit the seeds with evidence
only the tasks can give. So everything here first averages a task's seeds
into one number per task — its success fraction, its mean SPL — and only then
takes a mean, a standard error or a paired difference *across tasks*. With one
seed per task that is exactly the episode-level statistic P6 reported.

Pure numpy and scipy over rows as `metrics.read_table` returns them: no GPU,
no simulator.

    task_values(rows, "success")          # {task_id: success fraction}
    summarize(rows)                       # {statistic: (mean, se, n_tasks)}
    contrast({"a": rows_a, "b": rows_b}, {"a": 1, "b": -1}, "spl")
"""

import numpy as np
from scipy import stats

import metrics

# Which way is better, per statistic: +1 higher, -1 lower, 0 neither. Only the
# additivity verdict reads it, to name an interaction super- or sub-additive.
# Path length has no better direction: shorter is efficient on a success and
# is also what a robot that stalls at the start produces.
BETTER = {
    "success_rate": +1,
    "spl": +1,
    "collision_events": -1,
    "collision_events_per_m": -1,
    "contact_tick_fraction": -1,
    "final_geodesic_distance_m": -1,
    "path_length_m": 0,
    "ticks": -1,
    "ticks_to_goal": -1,
}

# A difference within this many standard errors of zero is read as noise —
# the rule every earlier table here used.
NOISE_SE = 2.0

# Per-task differences closer to zero than this are ties. A contrast sums
# several task means, so arms that tie exactly can leave 1e-17 behind, which a
# signed-rank test would otherwise rank as a real difference.
TIE_TOLERANCE = 1e-9


def _statistic(name):
    """(column, over) for one of `metrics.EPISODE_STATISTICS`."""
    for statistic, column, over in metrics.EPISODE_STATISTICS:
        if statistic == name:
            return column, over
    raise KeyError("unknown statistic {!r}".format(name))


def task_values(rows, statistic):
    """{task_id: the task's mean over its seeds} for one statistic.

    `ticks_to_goal` averages only a task's successful seeds, and a task with
    none has no value — the same rule `metrics.summarize` applies to episodes.
    Blank cells are left out, never read as zero.
    """
    column, over = _statistic(statistic)
    by_task = {}
    for row in rows:
        if over == "successes" and not int(row["success"]):
            continue
        if row[column] in ("", None):
            continue
        by_task.setdefault(row["task_id"], []).append(float(row[column]))
    return {task: float(np.mean(values)) for task, values in by_task.items()}


def summarize(rows):
    """{statistic: (mean, se, n_tasks)} — `metrics.summarize`, one level up."""
    rows = list(rows)
    return {statistic: metrics.mean_se(list(task_values(rows, statistic).values()))
            for statistic, _column, _over in metrics.EPISODE_STATISTICS}


class Contrast:
    """A weighted sum of arms, task by task, and its spread across tasks.

    `bc30 - ctx03` is weights {bc30: 1, ctx03: -1}; the additivity interaction
    `bc30 - stride3 - img160x120 + ctx03` is four weights. Only tasks with a
    value in every arm count, so each contrast carries its own `n`.
    """

    def __init__(self, per_task):
        self.per_task = per_task
        values = list(per_task.values())
        self.mean, self.se, self.n = metrics.mean_se(values)
        self.p_paired_t = _paired_t_p(values)
        self.p_wilcoxon = _wilcoxon_p(values)

    @property
    def within_noise(self):
        """|mean| <= 2 SE — or no SE to judge by, which is not evidence either."""
        if self.se is None:
            return True
        return abs(self.mean) <= NOISE_SE * self.se


def contrast(tables, weights, statistic):
    """The `Contrast` of `weights` ({arm: weight}) over `tables` ({arm: rows})."""
    values = {arm: task_values(tables[arm], statistic) for arm in weights}
    common = set.intersection(*(set(per_arm) for per_arm in values.values()))
    return Contrast({task: sum(weight * values[arm][task]
                               for arm, weight in weights.items())
                     for task in sorted(common)})


def _paired_t_p(differences):
    """Two-sided one-sample t test of the per-task differences against zero.

    None below two tasks. Differences that are all equal have no spread to
    test against: all zero is no evidence (p = 1), anything else is None
    rather than a t statistic divided by zero.
    """
    if len(differences) < 2:
        return None
    if np.ptp(differences) <= TIE_TOLERANCE:
        return 1.0 if abs(differences[0]) <= TIE_TOLERANCE else None
    return float(stats.ttest_1samp(differences, 0.0).pvalue)


def _wilcoxon_p(differences):
    """Two-sided Wilcoxon signed-rank p on the per-task differences.

    Tasks where the arms tie are dropped (scipy's default, Wilcoxon's own
    rule): they say nothing about which arm is better. All tied is p = 1.
    """
    nonzero = [value for value in differences if abs(value) > TIE_TOLERANCE]
    if not nonzero:
        return 1.0
    return float(stats.wilcoxon(nonzero).pvalue)


def mcnemar(rows_a, rows_b):
    """(a_only, b_only, exact two-sided p) on success, one episode per task.

    Only defined with one seed per task: McNemar pairs *episodes*, and under
    several seeds that pairs the seeds, which is the independence this module
    exists to refuse. Under multi-seed, success is judged by `contrast`'s
    Wilcoxon on the success fractions instead.
    """
    a, b = _single_episode(rows_a), _single_episode(rows_b)
    common = set(a) & set(b)
    a_only = sum(1 for task in common if a[task] and not b[task])
    b_only = sum(1 for task in common if b[task] and not a[task])
    discordant = a_only + b_only
    p = 1.0 if discordant == 0 else float(
        stats.binomtest(a_only, discordant, 0.5).pvalue)
    return a_only, b_only, p


def _single_episode(rows):
    """{task_id: success} — refusing a table with more than one seed a task."""
    repeated = sorted(task for task, count in seed_counts(rows).items() if count > 1)
    if repeated:
        raise ValueError("McNemar needs one episode per task; {} have more"
                         .format(", ".join(repeated)))
    return {row["task_id"]: int(row["success"]) for row in rows}


def seed_counts(rows):
    """{task_id: how many episodes (seeds) the task has}."""
    counts = {}
    for row in rows:
        counts[row["task_id"]] = counts.get(row["task_id"], 0) + 1
    return counts


def additivity_verdict(statistic, interaction):
    """How the combination compares with the sum of its parts.

    `interaction` is Δboth − (Δa + Δb). Within noise, the parts add up. Beyond
    it, super-additive means the combination is *better* than the sum
    predicts, in the statistic's own better direction, and sub-additive worse.
    A statistic with no better direction says only which way it differs.
    """
    if interaction.within_noise:
        return "additive"
    direction = BETTER[statistic] * np.sign(interaction.mean)
    if direction > 0:
        return "super-additive"
    if direction < 0:
        return "sub-additive"
    return "non-additive ({})".format("+" if interaction.mean > 0 else "-")
