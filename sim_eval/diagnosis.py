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

from driver import DEFAULT_CLOSE_THRESHOLD

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


# --- which side a contact was on ---------------------------------------------
# A contact within this many degrees of the heading counts as "front". 30 is
# the 45-degree-vertical camera's own horizontal half field of view (28.9
# degrees, rounded): anything wider was outside what P1-P5's camera could see.
FRONT_HALF_ANGLE_DEG = 30.0
REAR_HALF_ANGLE_DEG = 30.0


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


def signed_offset_from_path(point, path):
    """How far (x, y) is from a polyline, signed by which side of it: + is left
    of the direction the path runs, - is right.

    Measured to the segments, not the vertices, so a sparse path does not read
    as further away than it is. The sign is what turns "off the trail" into
    *drift*: a robot that wanders either side of its trail is noisy, and one
    that sits on the same side of it tick after tick is holding an offset.
    """
    x, y = float(point[0]), float(point[1])
    vertices = [(float(p[0]), float(p[1])) for p in path]
    if len(vertices) == 1:
        return math.hypot(x - vertices[0][0], y - vertices[0][1])
    best, signed = math.inf, 0.0
    for (ax, ay), (bx, by) in zip(vertices, vertices[1:]):
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0 else max(0.0, min(1.0, (
            (x - ax) * dx + (y - ay) * dy) / length_sq))
        distance = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
        if distance < best:
            side = dx * (y - ay) - dy * (x - ax)
            best, signed = distance, math.copysign(distance, side)
    return signed


def distance_to_path(point, path):
    """Shortest distance from (x, y) to a polyline."""
    return abs(signed_offset_from_path(point, path))


# How far off the trail counts as being on one side of it, for the longest
# one-sided run. 2 cm is well under the 15-30 cm offsets the failures hold,
# and well over the millimetres a robot straddling its trail wobbles by.
ONE_SIDE_M = 0.02


def longest_one_side_run(offsets, threshold=ONE_SIDE_M):
    """The most consecutive ticks spent on the same side of the trail.

    "Ticks to cross back", taken to its end: a run is broken by a tick on the
    other side, or by one within `threshold` of the trail itself. A policy
    that recentres keeps these short; P5's failures held one side for 32-46
    ticks straight until something touched them.
    """
    longest = current = 0
    side = 0
    for offset in offsets:
        this = 0 if abs(offset) <= threshold else (1 if offset > 0 else -1)
        current = current + 1 if this != 0 and this == side else (1 if this else 0)
        side = this
        longest = max(longest, current)
    return longest


def mean_bearing_deg(bearings):
    """The circular mean of some bearings — +179 and -179 average to 180, not 0."""
    if not bearings:
        return None
    sines = sum(math.sin(math.radians(value)) for value in bearings)
    cosines = sum(math.cos(math.radians(value)) for value in bearings)
    return math.degrees(math.atan2(sines, cosines))


def side_of(bearing_deg):
    """front / left / right / rear, from a bearing in the ROS convention."""
    if bearing_deg is None:
        return None
    if abs(bearing_deg) <= FRONT_HALF_ANGLE_DEG:
        return "front"
    if abs(bearing_deg) >= 180.0 - REAR_HALF_ANGLE_DEG:
        return "rear"
    return "left" if bearing_deg > 0 else "right"


def _first_contact(ticks_log, poses, reference_path, close_threshold):
    """The first collision, and what happened to the policy after it.

    This is the wedge -> freeze mechanism made into columns: where the contact
    was, how well the agent had been tracking its trail until then, and — for
    every tick after it — whether the distance head had lost the trail (its
    closest reading at or past `close_threshold`, so the subgoal cannot
    advance) and whether the robot was still driving at full speed.
    """
    first = next((tick for tick in ticks_log if tick.get("collided")), None)
    before = [tick for tick in ticks_log
              if first is None or tick["tick"] < first["tick"]]
    facts = {
        "first_contact_tick": None if first is None else int(first["tick"]),
        "first_contact_bearing_deg": None,
        "first_contact_side": None,
        "off_trail_at_first_contact_m": None,
        "max_off_trail_before_contact_m": None,
        "mean_abs_off_trail_before_contact_m": None,
        "mean_signed_off_trail_before_contact_m": None,
        "longest_one_side_run_ticks": None,
        "head_lost_after_contact": None,
        "full_speed_after_contact": None,
    }
    if first is not None:
        bearing = mean_bearing_deg(first.get("contact_bearing_deg") or [])
        facts["first_contact_bearing_deg"] = (
            None if bearing is None else round(bearing, 1))
        facts["first_contact_side"] = side_of(bearing)
        after = [tick for tick in ticks_log if tick["tick"] > first["tick"]]
        if after:
            lost = [tick for tick in after if tick.get("dist_closest") is not None
                    and float(tick["dist_closest"]) >= close_threshold]
            fast = [tick for tick in after if abs(float(tick["v"])) >= 0.199]
            facts["head_lost_after_contact"] = round(len(lost) / len(after), 3)
            facts["full_speed_after_contact"] = round(len(fast) / len(after), 3)

    if reference_path and poses:
        def signed(tick):
            index = int(tick["tick"])
            return signed_offset_from_path(poses[index], reference_path) \
                if index < len(poses) else None
        offsets = [value for value in map(signed, before) if value is not None]
        if offsets:
            facts["max_off_trail_before_contact_m"] = round(
                max(abs(value) for value in offsets), 3)
            facts["mean_abs_off_trail_before_contact_m"] = round(
                sum(abs(value) for value in offsets) / len(offsets), 3)
            facts["mean_signed_off_trail_before_contact_m"] = round(
                sum(offsets) / len(offsets), 3)
            facts["longest_one_side_run_ticks"] = longest_one_side_run(offsets)
        if first is not None and signed(first) is not None:
            facts["off_trail_at_first_contact_m"] = round(abs(signed(first)), 3)
    return facts


def evidence(trace, reference_path=None, close_threshold=DEFAULT_CLOSE_THRESHOLD):
    """Everything a diagnosis is allowed to look at, as one flat dict.

    `reference_path` is the trail's own driven path (the task's metadata), and
    is optional: without it the off-trail columns are blank rather than guessed.
    """
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
    facts.update(_first_contact(ticks_log, trace.get("poses") or [],
                                reference_path, close_threshold))
    facts["path_ratio"] = (
        round(facts["path_length_m"] / facts["geodesic_length_m"], 2)
        if facts["geodesic_length_m"] else None)
    return facts


def diagnose(trace, reference_path=None, close_threshold=DEFAULT_CLOSE_THRESHOLD):
    """Name one episode's mode. First match wins — see the module docstring."""
    facts = evidence(trace, reference_path, close_threshold)
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


def diagnose_all(traces, reference_paths=None, close_threshold=DEFAULT_CLOSE_THRESHOLD):
    """Diagnose every trace; `reference_paths` maps task id -> driven path."""
    reference_paths = reference_paths or {}
    return [diagnose(trace, reference_paths.get(trace.get("task_id")),
                     close_threshold)
            for trace in traces]


def tally(diagnoses):
    """How many episodes fell into each mode, most common first."""
    counts = {}
    for diagnosis in diagnoses:
        counts[diagnosis.mode] = counts.get(diagnosis.mode, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
