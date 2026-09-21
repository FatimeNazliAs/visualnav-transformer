"""What an episode scored: the five metrics of plan §6, and the table they go in.

This module is deliberately arithmetic and nothing else. It never touches
iGibson, torch, a checkpoint or a GPU — it takes the numbers an episode
produced and turns them into the row that lands in the CSV. That keeps
`tests/test_metrics.py` able to pin every formula in milliseconds, which
matters because a metric that is quietly wrong does not crash: it produces a
plausible table, and the whole comparison downstream is built on it.

The five metrics (plan §6):

  * **success** — the agent finished within the goal radius.
  * **collision rate** — distinct collisions per episode and per metre driven,
    plus the fraction of ticks spent in contact. Count-and-continue: a
    collision never ends an episode.
  * **SPL** — success weighted by path length (Anderson et al. 2018):
    `S * L / max(P, L)`, with `L` the geodesic shortest path from the task's
    metadata and `P` the path the agent actually drove.
  * **final distance-to-goal** — how far from the goal it stopped. Geodesic, so
    a metre of wall between agent and goal counts as the detour it is.
  * **steps / time-to-goal** — ticks spent, and the robot-time they represent.

Everything else in a row (checkpoint, scene, task, seed) is provenance: the
fairness protocol (plan §7) is only checkable if each row says which arm ran
which task under which seed.
"""

import csv
import json
from pathlib import Path

import numpy as np

# The order columns appear in the CSV. Provenance first, then the five metrics,
# then the context needed to read them (what the rules were, how it ended).
# `DictWriter` uses this as its field list, so a metric added to `as_row`
# without a column here fails loudly instead of being dropped.
CSV_COLUMNS = (
    # provenance — which arm ran which task, under what seed, steered how
    "checkpoint",
    "scene",
    "task_id",
    "seed",
    "driver",
    # the five metrics of plan §6
    "success",
    "collision_ticks",
    "collision_events",
    "contact_tick_fraction",
    "collision_events_per_m",
    "spl",
    "final_geodesic_distance_m",
    "final_euclidean_distance_m",
    "ticks",
    "seconds",
    # how to read them
    "path_length_m",
    "geodesic_length_m",
    "outcome",
    "success_radius_m",
    "success_metric",
    "timeout_ticks",
    "declared_arrival_tick",
    "final_node",
    "node_count",
)

# How an episode ended. "timeout" is the only failure the runner produces:
# collisions are counted and never terminal (plan §6), so there is no third
# outcome to hide a failure behind.
OUTCOME_SUCCESS = "success"
OUTCOME_TIMEOUT = "timeout"


def path_length(poses):
    """Metres travelled along a sequence of (x, y, yaw) poses.

    Summed over consecutive pairs, so the caller must include the pose the
    agent finished at — the tick records hold the pose each tick *started*
    from, and leaving the final one out drops the last leg. This is SPL's
    denominator, so that missing leg would silently inflate every score.
    """
    points = np.asarray([pose[:2] for pose in poses], dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def count_collision_events(collided_flags):
    """Distinct collisions, counting a run of contact ticks as one event.

    Ticks-in-contact and collisions-are-separate-events answer different
    questions and neither subsumes the other: a robot scraping a wall for
    twenty ticks is one mistake that lasted five seconds, while twenty taps on
    twenty different chair legs is twenty mistakes. Both are logged, and this
    is the rising edge of the flag — the second reading.
    """
    flags = [bool(flag) for flag in collided_flags]
    return sum(1 for index, flag in enumerate(flags)
               if flag and not (index > 0 and flags[index - 1]))


def spl(success, geodesic_length_m, path_length_m):
    """Success weighted by path efficiency — Anderson et al. 2018, per episode.

        SPL = S * L / max(P, L)

    `max` is in the definition, not defensive: an agent that cuts the corner
    the grid planner had to go round drives *less* than the geodesic, and
    without the clamp its SPL exceeds 1. That happens here rather than rarely —
    the reference drives in P2 are already shorter than their own geodesics,
    because A* zigzags between cell centres and a smooth follower does not.

    A failed episode scores 0 whatever it drove, which is the metric's whole
    point: efficiency only counts if you arrived.
    """
    if not success:
        return 0.0
    geodesic = float(geodesic_length_m)
    driven = float(path_length_m)
    if geodesic <= 0.0:
        raise ValueError(
            "geodesic length must be positive to weight a path by it; got {}"
            .format(geodesic_length_m))
    return geodesic / max(driven, geodesic)


def _rate(count, total):
    """count / total, or 0.0 when there is no total to divide by."""
    return float(count) / float(total) if total > 0 else 0.0


class EpisodeMetrics:
    """One scored episode: the row that goes in the table.

    Built by `EpisodeRunner`, which supplies the raw counts; every derived
    number (SPL, the two collision rates, seconds) is computed here, once, so
    the CSV and any later analysis cannot disagree about what a rate meant.
    """

    def __init__(self, checkpoint, task, seed, driver, success, collision_ticks,
                 collision_events, path_length_m, final_geodesic_distance_m,
                 final_euclidean_distance_m, ticks, seconds, timeout_ticks,
                 success_radius_m, success_metric, declared_arrival_tick,
                 final_node):
        self.checkpoint = checkpoint
        self.task = task
        self.seed = int(seed)
        # How the policy was steered — `nomad_policy.DriverConfig.label()`, e.g.
        # "n8w2r4t3". In the row because those four numbers decide the action as
        # much as the checkpoint does, and P5 is expected to move them: two
        # tables from two tunings must not be indistinguishable.
        self.driver = str(driver)
        self.success = bool(success)
        self.collision_ticks = int(collision_ticks)
        self.collision_events = int(collision_events)
        self.path_length_m = float(path_length_m)
        self.final_geodesic_distance_m = final_geodesic_distance_m
        self.final_euclidean_distance_m = float(final_euclidean_distance_m)
        self.ticks = int(ticks)
        self.seconds = float(seconds)
        self.timeout_ticks = int(timeout_ticks)
        self.success_radius_m = float(success_radius_m)
        # Which of the two distances the radius was measured with. In the row
        # because it changes what `success` means, and a table that does not
        # say so cannot be compared with another one.
        self.success_metric = success_metric
        # Which tick the policy first localized onto the goal node, or None. It
        # is a diagnostic, never an end condition: the episode ends on where the
        # agent *is*, not on where it believes it is (see episode_runner).
        self.declared_arrival_tick = declared_arrival_tick
        self.final_node = int(final_node)

    @property
    def outcome(self):
        return OUTCOME_SUCCESS if self.success else OUTCOME_TIMEOUT

    @property
    def spl(self):
        return spl(self.success, self.task.geodesic_length_m, self.path_length_m)

    @property
    def contact_tick_fraction(self):
        """How much of the episode was spent touching something."""
        return _rate(self.collision_ticks, self.ticks)

    @property
    def collision_events_per_m(self):
        """Distinct collisions per metre driven — plan §6's collision rate.

        Per-episode counts favour whichever arm gave up earliest, so the rate
        is normalized by distance. **Events**, not colliding ticks: a robot
        wedged against a chair for a minute is one mistake, and dividing its
        300 contact ticks by the 5 m it managed gives 60 "collisions per
        metre", which is a number about being stuck rather than about hitting
        things. How stuck it was is `contact_tick_fraction`, next door.

        A zero-length path has no rate, not an infinite one.
        """
        return _rate(self.collision_events, self.path_length_m)

    def as_row(self):
        """The CSV row. Keys must match CSV_COLUMNS exactly."""
        return {
            "checkpoint": self.checkpoint,
            "scene": self.task.scene,
            "task_id": self.task.task_id,
            "seed": self.seed,
            "driver": self.driver,
            "success": int(self.success),
            "collision_ticks": self.collision_ticks,
            "collision_events": self.collision_events,
            "contact_tick_fraction": round(self.contact_tick_fraction, 6),
            "collision_events_per_m": round(self.collision_events_per_m, 6),
            "spl": round(self.spl, 6),
            "final_geodesic_distance_m": (
                "" if self.final_geodesic_distance_m is None
                else round(self.final_geodesic_distance_m, 4)),
            "final_euclidean_distance_m": round(self.final_euclidean_distance_m, 4),
            "ticks": self.ticks,
            "seconds": round(self.seconds, 3),
            "path_length_m": round(self.path_length_m, 4),
            "geodesic_length_m": round(self.task.geodesic_length_m, 4),
            "outcome": self.outcome,
            "success_radius_m": self.success_radius_m,
            "success_metric": self.success_metric,
            "timeout_ticks": self.timeout_ticks,
            "declared_arrival_tick": (
                "" if self.declared_arrival_tick is None
                else self.declared_arrival_tick),
            "final_node": self.final_node,
            "node_count": self.task.node_count,
        }

    def summary(self):
        distance = ("?" if self.final_geodesic_distance_m is None
                    else "{:.2f}".format(self.final_geodesic_distance_m))
        return ("{:<10} {:<12} {:<8} SPL {:.3f}  {} m from goal  "
                "{} ticks ({:.0f} s)  {} collision ticks".format(
                    self.task.task_id, self.checkpoint, self.outcome.upper(),
                    self.spl, distance, self.ticks, self.seconds,
                    self.collision_ticks))


class MetricsTable:
    """The per-episode CSV, written a row at a time.

    Appended rather than collected and dumped at the end, because a run is
    twenty slow episodes: a crash on episode nineteen must not cost the
    eighteen that already scored, and a run in progress should be readable
    with `tail`. The header is written only when the file is created, so a
    reopened table continues rather than restarting.
    """

    def __init__(self, csv_path, trace_path=None):
        self.csv_path = Path(csv_path)
        # One JSON object per episode beside the table: the same metrics plus
        # the pose trace and the per-tick decisions. The CSV is the result; this
        # is the evidence behind it, and what P4 replays into a video.
        self.trace_path = (Path(trace_path) if trace_path is not None
                           else self.csv_path.with_suffix(".jsonl"))
        self.rows = 0

    def open(self, resume=False):
        """Create the table, or reopen an existing one to carry on from it.

        Resuming first repairs what an interrupted run can leave behind — see
        `_repair` — so the table that is appended to holds only whole episodes.
        """
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.csv_path.exists():
            self._repair()
            return self
        self._write([], [])
        return self

    def append(self, metrics, trace=None):
        """Append one scored episode, and flush it — see the class docstring.

        The trace goes first and the CSV row last: the row is the commit
        marker. A run killed between the two leaves a trace with no row, which
        `_repair` drops and the resumed run re-scores — never a row whose
        evidence is missing.
        """
        if trace is not None:
            with open(self.trace_path, "a") as handle:
                handle.write(json.dumps(trace) + "\n")
        with open(self.csv_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writerow(metrics.as_row())
        self.rows += 1
        return self

    def completed(self):
        """The episodes already scored, as `episode_key`s — what a resumed run skips."""
        return {episode_key(row) for row in read_table(self.csv_path)}

    def _repair(self):
        """Keep the whole episodes of an interrupted table, and nothing else.

        A process killed mid-write can leave a truncated last row, and one
        killed between an episode's trace and its row leaves a trace with no
        row. Both are dropped, so the episode they belonged to is simply not
        done yet. A header that is not this module's is refused rather than
        repaired: that table was written under different columns, and
        appending to it would mix two schemas in one file.
        """
        reader = csv.DictReader(_whole_lines(self.csv_path))
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                "{} has different columns from this scorer's, so it cannot "
                "be resumed. Move it aside and start the table over."
                .format(self.csv_path))
        rows = [row for row in reader if _is_whole(row)]

        keys = {episode_key(row) for row in rows}
        traces = {}
        for trace in _read_traces(self.trace_path):
            if episode_key(trace) in keys:
                traces[episode_key(trace)] = trace
        self._write(rows, [traces[episode_key(row)] for row in rows
                           if episode_key(row) in traces])
        self.rows = len(rows)

    def _write(self, rows, traces):
        with open(self.csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        self.trace_path.write_text(
            "".join(json.dumps(trace) + "\n" for trace in traces))


def episode_key(row):
    """What identifies one episode: which arm ran which task under which seed.

    The seed is part of it because the fairness protocol (plan §7) is about the
    pair, not the task alone — the same task rerun under another seed is a
    different episode, and must not be skipped as if it were already done.
    """
    return (str(row["checkpoint"]), str(row["task_id"]), int(row["seed"]))


def _whole_lines(path):
    """The lines of a file that were finished, i.e. ended with a newline.

    A write cut off by a crash is always the last line and never ends in one —
    and it can look complete: a row truncated inside its final column still
    has every field, just the wrong number in the last.
    """
    lines = Path(path).read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        return lines[:-1]
    return lines


def _is_whole(row):
    """False for a row with fields missing (None) or extra (under key None)."""
    return None not in row.values() and None not in row


def _read_traces(trace_path):
    """Every finished, parseable trace line."""
    if not Path(trace_path).exists():
        return []
    traces = []
    for line in _whole_lines(trace_path):
        try:
            traces.append(json.loads(line))
        except ValueError:
            continue
    return traces


def read_table(csv_path):
    """Read a metrics table back — the counterpart to `MetricsTable.append`."""
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))


# The statistics a checkpoint is summarized by: (name, CSV column, which
# episodes it averages over). This table is the only definition of each name.
# The comparison table and the block a scoring run ends with both read it, so a
# name means the same number wherever it is printed.
#
# Every statistic is a mean over episodes of a per-episode column, never a
# ratio of totals. Each arm runs each task once, so a mean over episodes is a
# mean over tasks: every task counts once, and the mean has a standard error
# across tasks, which is what a difference between arms is read against. A
# ratio of totals (sum of contact ticks / sum of ticks) lets one long timeout
# outweigh several short successes, and has no per-task spread to put beside it.
#
# Success rate and SPL average over every episode: a failure is a zero, not a
# row left out. Time to goal averages the successes only (plan section 6),
# because a mean that took in timeouts would describe the timeout formula, not
# the agent.
EPISODE_STATISTICS = (
    ("success_rate", "success", "all"),
    ("spl", "spl", "all"),
    ("collision_events", "collision_events", "all"),
    ("collision_events_per_m", "collision_events_per_m", "all"),
    ("contact_tick_fraction", "contact_tick_fraction", "all"),
    ("final_geodesic_distance_m", "final_geodesic_distance_m", "all"),
    ("path_length_m", "path_length_m", "all"),
    ("ticks", "ticks", "all"),
    ("ticks_to_goal", "ticks", "successes"),
)


def mean_se(values):
    """(mean, standard error, n) of a sample; SE is None below two values.

    The standard error of the mean, `s / sqrt(n)` with the sample standard
    deviation (ddof=1): the spread of the *mean* over a redraw of the tasks,
    which is what a difference between two arms has to be read against. One
    value has no spread to estimate, so it gets none rather than a zero.
    """
    values = np.asarray([float(value) for value in values], dtype=float)
    count = len(values)
    if count == 0:
        return None, None, 0
    if count == 1:
        return float(values[0]), None, 1
    return (float(values.mean()),
            float(values.std(ddof=1) / np.sqrt(count)), count)


def summarize(rows):
    """Every `EPISODE_STATISTICS` entry for one checkpoint's rows, as mean_se.

    Blank cells (an episode that ended off the nav mesh has no geodesic
    distance) are left out of that one statistic rather than counted as zero,
    so each statistic carries its own `n`.
    """
    rows = list(rows)
    summary = {}
    for name, column, over in EPISODE_STATISTICS:
        chosen = [row for row in rows
                  if over == "all" or int(row["success"])]
        summary[name] = mean_se(row[column] for row in chosen
                                if row[column] not in ("", None))
    return summary


def format_summary(checkpoint, summary):
    """`summarize`'s means as a short block: what a scoring run ends with.

    Means only, so it reads at a glance; the standard errors are in the
    comparison table. Both come from the same `summarize`, so the numbers here
    are the table's means.
    """
    episodes = summary["success_rate"][2]
    if not episodes:
        return "{}: no episodes".format(checkpoint)
    mean = {name: value[0] for name, value in summary.items()}
    ticks = ("n/a (no successes)" if mean["ticks_to_goal"] is None
             else "{:.0f}".format(mean["ticks_to_goal"]))
    distance = ("n/a (none on the nav mesh)"
                if mean["final_geodesic_distance_m"] is None
                else "{:.2f} m from the goal".format(
                    mean["final_geodesic_distance_m"]))
    return "\n".join([
        "{} over {} episodes (means per episode)".format(checkpoint, episodes),
        "  success rate:   {:.0%}".format(mean["success_rate"]),
        "  SPL:            {:.3f}".format(mean["spl"]),
        "  collisions:     {:.1f} per episode, {:.3f} per metre".format(
            mean["collision_events"], mean["collision_events_per_m"]),
        "  in contact:     {:.0%} of an episode's ticks".format(
            mean["contact_tick_fraction"]),
        "  final distance: {}".format(distance),
        "  ticks to goal:  {} (successes only)".format(ticks),
    ])
