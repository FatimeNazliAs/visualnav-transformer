"""Why an episode failed — a name, and the evidence for it.

P3's table says an episode timed out 1.5 m from its goal. That is the *result*,
and it is the same result for an agent wedged under a sofa, an agent spinning
on the spot, an agent that drove confidently past the goal, and an agent whose
distance head decided at tick 57 that it had arrived. Those are four different
problems with four different answers, and a success rate cannot tell them
apart — which is the whole difficulty of plan §9's P5, "confirm NoMaD behaves
sanely despite the domain gap".

So this reads one episode's trace and names the mode, from the evidence already
in it. Nothing here runs anything, nothing here is a fix, and nothing here
changes a metric: a diagnosis is a reading of a table, and the table is the
same table P3 wrote.

**It is deliberately ordered, and the order is the argument.** The modes are
not mutually exclusive — a wedged agent has also failed to advance along its
trail, and an agent that declared arrival early usually stopped advancing too —
so the first match wins, and they are tried most-specific first. What that
buys is that the name points at the *earliest* thing that went wrong rather
than at its consequence.

The thresholds below are for reading, not for scoring. Nothing in the metrics
table depends on them, and moving one renames an episode without changing a
single number that P6 will report.
"""

import math

# --- when an agent counts as stuck against something ------------------------
# Plan §6 counts collisions and never ends an episode for one, so an agent that
# wedges spends its entire budget in contact: P3's second episode logged 298
# colliding ticks out of 341. Half the episode in contact is far past anything
# a passing scrape produces.
WEDGED_CONTACT_FRACTION = 0.5
# ...and has not gone anywhere since. Measured over the last quarter of the
# episode, so a run that hit something early and drove on is not called wedged.
WEDGED_DISPLACEMENT_M = 0.3

# --- when it is turning instead of going ------------------------------------
# A differential drive lining up with its path spends up to pi / max_w = 7.9 s
# turning on the spot, which is healthy. Doing it for most of an episode is
# not: it means the waypoint keeps coming back off to the side and never in
# front, which is what a policy does when it cannot read the scene at all.
SPINNING_TICK_FRACTION = 0.6
SPINNING_MAX_V = 0.02

# --- when it never got along its trail --------------------------------------
# The fraction of the trail the distance head localized onto. Under a quarter
# means the agent spent the episode believing it was still near the start.
TRAIL_STALLED_FRACTION = 0.25

# --- when it drove a long way to nowhere ------------------------------------
# Path length against the shortest path. Twice is generous: P3's successful
# episode drove 2.7 m of a 4.4 m geodesic (the planner's grid zigzags, so
# driving *less* is normal), and its worst failure drove 5.6 m of 3.9 m.
WANDERING_PATH_RATIO = 2.0


class Diagnosis:
    """One episode's failure mode, with the numbers that named it."""

    def __init__(self, task_id, mode, explanation, evidence):
        self.task_id = task_id
        self.mode = mode
        self.explanation = explanation
        self.evidence = evidence

    def summary(self):
        return "{:<10} {:<18} {}".format(self.task_id, self.mode, self.explanation)

    def as_row(self):
        row = {"task_id": self.task_id, "mode": self.mode,
               "explanation": self.explanation}
        row.update(self.evidence)
        return row


def _float(value, default=None):
    """A CSV/JSON field as a float, tolerating the blanks the table writes."""
    if value is None or value == "":
        return default
    return float(value)


def _displacement_m(poses, fraction=0.25):
    """How far the agent actually got during the last `fraction` of the episode.

    Straight-line start to end of that window, not path length: an agent
    grinding against a wall racks up path length while going nowhere, and going
    nowhere is the thing being measured.
    """
    poses = list(poses or [])
    if len(poses) < 2:
        return 0.0
    window = max(2, int(len(poses) * fraction))
    first, last = poses[-window], poses[-1]
    return math.hypot(last[0] - first[0], last[1] - first[1])


def _trail_progress(trace):
    """The fraction of the trail the distance head localized onto, 0 to 1.

    `final_node` against the last node, not against the node count, because the
    last node *is* the goal: an agent that reaches node 19 of 20 has covered
    all of it.
    """
    last_node = max(1, int(trace.get("node_count", 1)) - 1)
    return min(1.0, float(trace.get("final_node", 0)) / last_node)


def _motion(ticks_log):
    """What the controller was doing, averaged over the episode."""
    ticks = list(ticks_log or [])
    if not ticks:
        return {"mean_v": 0.0, "mean_abs_w": 0.0, "turning_tick_fraction": 0.0}
    speeds = [abs(float(tick["v"])) for tick in ticks]
    turns = [abs(float(tick["w"])) for tick in ticks]
    turning = sum(1 for speed in speeds if speed <= SPINNING_MAX_V)
    return {
        "mean_v": round(sum(speeds) / len(ticks), 4),
        "mean_abs_w": round(sum(turns) / len(ticks), 4),
        "turning_tick_fraction": round(turning / len(ticks), 4),
    }


def _distance_head(ticks_log):
    """What the distance head read, averaged over the episode.

    A head reading far above `close_threshold` every tick is a head that never
    lets the subgoal advance — the trail then cannot progress no matter how
    well the agent drives, and no other column in the table says so.
    """
    readings = [_float(tick.get("dist_closest")) for tick in (ticks_log or [])]
    readings = [value for value in readings if value is not None]
    if not readings:
        return {"mean_dist_closest": None, "min_dist_closest": None}
    return {
        "mean_dist_closest": round(sum(readings) / len(readings), 3),
        "min_dist_closest": round(min(readings), 3),
    }


def evidence(trace):
    """Everything a diagnosis is allowed to look at, as one flat dict."""
    ticks_log = trace.get("ticks_log") or []
    facts = {
        "outcome": trace.get("outcome"),
        "ticks": int(trace.get("ticks", 0)),
        "final_distance_m": _float(trace.get("final_geodesic_distance_m")),
        "path_length_m": _float(trace.get("path_length_m"), 0.0),
        "geodesic_length_m": _float(trace.get("geodesic_length_m"), 0.0),
        "contact_tick_fraction": _float(trace.get("contact_tick_fraction"), 0.0),
        "collision_events": int(_float(trace.get("collision_events"), 0)),
        "declared_arrival_tick": _float(trace.get("declared_arrival_tick")),
        "trail_progress": round(_trail_progress(trace), 3),
        "final_displacement_m": round(_displacement_m(trace.get("poses")), 3),
    }
    facts.update(_motion(ticks_log))
    facts.update(_distance_head(ticks_log))
    facts["path_ratio"] = (
        round(facts["path_length_m"] / facts["geodesic_length_m"], 2)
        if facts["geodesic_length_m"] else None)
    return facts


def diagnose(trace):
    """Name one episode's mode. First match wins — see the module docstring."""
    facts = evidence(trace)
    task_id = trace.get("task_id", "?")

    def named(mode, explanation):
        return Diagnosis(task_id, mode, explanation, facts)

    if trace.get("success") in (1, "1", True):
        return named("reached the goal",
                     "{} ticks, {:.2f} m of a {:.2f} m shortest path".format(
                         facts["ticks"], facts["path_length_m"],
                         facts["geodesic_length_m"]))

    if (facts["contact_tick_fraction"] >= WEDGED_CONTACT_FRACTION
            and facts["final_displacement_m"] < WEDGED_DISPLACEMENT_M):
        return named("wedged",
                     "in contact for {:.0%} of the episode and moved {:.2f} m "
                     "in its last quarter".format(
                         facts["contact_tick_fraction"],
                         facts["final_displacement_m"]))

    if (facts["turning_tick_fraction"] >= SPINNING_TICK_FRACTION
            and facts["final_displacement_m"] < WEDGED_DISPLACEMENT_M):
        return named("turning on the spot",
                     "{:.0%} of ticks below {} m/s, mean |w| {:.2f} rad/s — the "
                     "waypoint never lands in front".format(
                         facts["turning_tick_fraction"], SPINNING_MAX_V,
                         facts["mean_abs_w"]))

    if facts["declared_arrival_tick"] is not None:
        return named("false arrival",
                     "the distance head localized onto the last node at tick "
                     "{:.0f} of {}, and the agent finished {} away".format(
                         facts["declared_arrival_tick"], facts["ticks"],
                         _metres(facts["final_distance_m"])))

    if facts["trail_progress"] < TRAIL_STALLED_FRACTION:
        return named("trail not advanced",
                     "localized only {:.0%} along the trail in {} ticks; the "
                     "head read {} at the closest node on average".format(
                         facts["trail_progress"], facts["ticks"],
                         _number(facts["mean_dist_closest"])))

    if (facts["path_ratio"] is not None
            and facts["path_ratio"] >= WANDERING_PATH_RATIO):
        return named("wandered",
                     "drove {:.2f} m of a {:.2f} m shortest path ({}x) and "
                     "finished {} away".format(
                         facts["path_length_m"], facts["geodesic_length_m"],
                         facts["path_ratio"], _metres(facts["final_distance_m"])))

    return named("ran out of ticks",
                 "made it {:.0%} along the trail and finished {} away, with no "
                 "single thing going wrong".format(
                     facts["trail_progress"], _metres(facts["final_distance_m"])))


def _metres(value):
    """A distance for a sentence, or the honest absence of one.

    A blank `final_geodesic_distance_m` means the agent has no path to the goal
    at all — it left the traversable component, which in Rs means it climbed
    onto the furniture. Printing 0 there would read as "arrived".
    """
    return "{:.2f} m".format(value) if value is not None else "no path from"


def _number(value):
    return "{:.1f}".format(value) if value is not None else "nothing"


def diagnose_all(traces):
    return [diagnose(trace) for trace in traces]


def tally(diagnoses):
    """How many episodes fell into each mode, most common first."""
    counts = {}
    for diagnosis in diagnoses:
        counts[diagnosis.mode] = counts.get(diagnosis.mode, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
