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
        """Create the table (or keep appending to an existing one)."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and self.csv_path.exists():
            return self
        with open(self.csv_path, "w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
        self.trace_path.write_text("")
        return self

    def append(self, metrics, trace=None):
        """Append one scored episode, and flush it — see the class docstring."""
        with open(self.csv_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writerow(metrics.as_row())
        if trace is not None:
            with open(self.trace_path, "a") as handle:
                handle.write(json.dumps(trace) + "\n")
        self.rows += 1
        return self


def read_table(csv_path):
    """Read a metrics table back — the counterpart to `MetricsTable.append`."""
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate(rows):
    """Roll per-episode rows up into the headline numbers for one checkpoint.

    P6 reports these; P3 prints them so a scoring run can be read at a glance.
    Success rate and mean SPL average over *all* episodes (a failure is a zero,
    not an omission), while time-to-goal averages over the successes only —
    plan §6 defines it that way, and a mean that mixed in timeouts would say
    more about the timeout formula than about the agent.
    """
    rows = list(rows)
    if not rows:
        return {}
    successes = [row for row in rows if int(row["success"])]
    total_distance = sum(float(row["path_length_m"]) for row in rows)
    total_ticks = sum(int(row["ticks"]) for row in rows)
    contact_ticks = sum(int(row["collision_ticks"]) for row in rows)
    events = sum(int(row["collision_events"]) for row in rows)
    return {
        "episodes": len(rows),
        "success_rate": len(successes) / len(rows),
        "spl": float(np.mean([float(row["spl"]) for row in rows])),
        "collision_events_per_episode": events / len(rows),
        "collision_events_per_m": _rate(events, total_distance),
        "contact_tick_fraction": _rate(contact_ticks, total_ticks),
        # An episode that ended off the traversable component has no geodesic
        # distance, and it must not turn the whole column into a blank: those
        # rows are skipped rather than counted as zero or as NaN.
        "final_distance_m": _mean_distance(rows),
        "ticks_to_goal": (float(np.mean([int(row["ticks"]) for row in successes]))
                          if successes else None),
    }


def _mean_distance(rows):
    """Mean final distance over the episodes that have one."""
    distances = [float(row["final_geodesic_distance_m"]) for row in rows
                 if row["final_geodesic_distance_m"] not in ("", None)]
    return float(np.mean(distances)) if distances else None


def format_aggregate(checkpoint, summary):
    """The rolled-up numbers as a short block — what a scoring run ends with."""
    if not summary:
        return "{}: no episodes".format(checkpoint)
    ticks = ("n/a (no successes)" if summary["ticks_to_goal"] is None
             else "{:.0f}".format(summary["ticks_to_goal"]))
    distance = ("n/a (none on the nav mesh)" if summary["final_distance_m"] is None
                else "{:.2f} m from the goal".format(summary["final_distance_m"]))
    return "\n".join([
        "{} over {} episodes".format(checkpoint, summary["episodes"]),
        "  success rate:   {:.0%}".format(summary["success_rate"]),
        "  SPL:            {:.3f}".format(summary["spl"]),
        "  collisions:     {:.1f} per episode, {:.3f} per metre".format(
            summary["collision_events_per_episode"],
            summary["collision_events_per_m"]),
        "  in contact:     {:.0%} of ticks".format(summary["contact_tick_fraction"]),
        "  final distance: {}".format(distance),
        "  ticks to goal:  {} (successes only)".format(ticks),
    ])
