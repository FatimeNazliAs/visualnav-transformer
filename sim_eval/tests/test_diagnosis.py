"""Pin which failure mode a trace gets named, and in what order.

A diagnosis changes no metric, so nothing downstream breaks if one is wrong —
which is exactly why it needs pinning. A mislabelled episode does not fail; it
sends a person looking at the wrong thing, and P5's whole output is a list of
these labels.

The order is the part worth testing. The modes are not mutually exclusive: a
wedged agent has also failed to advance along its trail, and an agent whose
distance head declared arrival early usually stopped advancing too. First match
wins and they are tried most-specific first, so the name points at the earliest
thing that went wrong rather than at its consequence — and that is a property
of the sequence, not of any one threshold.

No GPU, no simulator, no checkpoint: these are dictionaries.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diagnosis  # noqa: E402


def a_trace(**overrides):
    """A plausible failed episode, with everything a diagnosis reads.

    Deliberately a *failure* with nothing much wrong — "ran out of ticks" — so
    that each test switches on exactly one thing and the mode it produces is
    attributable to that one thing.
    """
    trace = {
        "task_id": "Rs_00",
        "success": 0,
        "outcome": "timeout",
        "ticks": 100,
        "node_count": 21,
        "final_node": 15,
        "final_geodesic_distance_m": 1.4,
        "path_length_m": 4.0,
        "geodesic_length_m": 4.0,
        "contact_tick_fraction": 0.0,
        "collision_events": 0,
        "declared_arrival_tick": None,
        "poses": [[float(i) * 0.05, 0.0, 0.0] for i in range(101)],
        "ticks_log": [
            {"tick": i, "node": i // 5, "subgoal": i // 5 + 1,
             "dist_closest": 5.0, "dist_subgoal": 6.0,
             "waypoint_m": [0.05, 0.0], "v": 0.2, "w": 0.0, "collided": False}
            for i in range(100)
        ],
    }
    trace.update(overrides)
    return trace


def still_ticks(count=100):
    """A tick log for an agent that turns without going anywhere."""
    return [{"tick": i, "node": 0, "subgoal": 1, "dist_closest": 8.0,
             "dist_subgoal": 9.0, "waypoint_m": [0.0, 0.05], "v": 0.0,
             "w": 0.4, "collided": False} for i in range(count)]


def one_pose_repeated(count=101):
    return [[1.0, 1.0, 0.0] for _ in range(count)]


# --- the plain readings -----------------------------------------------------

def test_a_success_is_named_by_its_outcome_not_by_what_else_went_on():
    trace = a_trace(success=1, outcome="success", contact_tick_fraction=0.9)
    assert diagnosis.diagnose(trace).mode == "reached the goal"


def test_an_episode_with_nothing_singular_wrong_simply_ran_out():
    assert diagnosis.diagnose(a_trace()).mode == "ran out of ticks"


# --- the specific modes -----------------------------------------------------

def test_an_agent_in_contact_and_going_nowhere_is_wedged():
    trace = a_trace(contact_tick_fraction=0.87, collision_events=1,
                    poses=one_pose_repeated())
    result = diagnosis.diagnose(trace)
    assert result.mode == "wedged"
    assert "87%" in result.explanation


def test_contact_alone_is_not_wedged_if_it_kept_moving():
    """P3's failing episode scraped along a wall for most of its ticks and kept
    driving. That is a different problem from being stuck under the furniture,
    and calling both 'wedged' would hide it."""
    assert diagnosis.diagnose(a_trace(contact_tick_fraction=0.87)).mode != "wedged"


def test_an_agent_turning_in_place_all_episode_is_named_for_that():
    trace = a_trace(ticks_log=still_ticks(), poses=one_pose_repeated())
    assert diagnosis.diagnose(trace).mode == "turning on the spot"


def test_a_healthy_turn_at_the_start_is_not_spinning():
    """Lining up with the trail takes up to pi / max_w = 7.9 s of pure
    rotation, which is 32 ticks of a 100-tick episode — well under the
    threshold, and the agent goes somewhere afterwards."""
    ticks = still_ticks(32) + a_trace()["ticks_log"][32:]
    assert diagnosis.diagnose(a_trace(ticks_log=ticks)).mode != "turning on the spot"


def test_the_head_claiming_arrival_far_from_the_goal_is_a_false_arrival():
    trace = a_trace(declared_arrival_tick=57, final_geodesic_distance_m=1.5)
    result = diagnosis.diagnose(trace)
    assert result.mode == "false arrival"
    assert "tick 57" in result.explanation


def test_an_agent_that_never_got_along_its_trail_says_so():
    trace = a_trace(final_node=2)
    result = diagnosis.diagnose(trace)
    assert result.mode == "trail not advanced"
    assert "10%" in result.explanation


def test_driving_far_further_than_the_shortest_path_is_wandering():
    trace = a_trace(path_length_m=12.0, geodesic_length_m=4.0)
    result = diagnosis.diagnose(trace)
    assert result.mode == "wandered"
    assert "3.0x" in result.explanation


# --- the order --------------------------------------------------------------

def test_wedged_beats_the_consequences_of_being_wedged():
    """An agent stuck against a wall also fails to advance its trail and also
    drives a strange path. The name has to point at the cause."""
    trace = a_trace(contact_tick_fraction=0.87, poses=one_pose_repeated(),
                    final_node=1, path_length_m=20.0)
    assert diagnosis.diagnose(trace).mode == "wedged"


def test_a_false_arrival_outranks_a_stalled_trail():
    """A head that has localized onto the last node has by definition advanced
    the trail to its end, but the interesting fact is the claim, not the
    progress — and an agent that then stops advancing would otherwise be filed
    under the milder name."""
    trace = a_trace(declared_arrival_tick=5, final_node=1)
    assert diagnosis.diagnose(trace).mode == "false arrival"


# --- the evidence -----------------------------------------------------------

def test_an_unreachable_goal_is_said_rather_than_printed_as_zero():
    """A blank distance means the agent left the traversable component — in Rs,
    that it climbed onto the furniture. A 0 there would read as 'arrived'."""
    trace = a_trace(final_geodesic_distance_m="", declared_arrival_tick=10)
    result = diagnosis.diagnose(trace)
    assert result.evidence["final_distance_m"] is None
    assert "no path from" in result.explanation


def test_the_evidence_carries_what_the_distance_head_read():
    facts = diagnosis.diagnose(a_trace()).evidence
    assert facts["mean_dist_closest"] == 5.0
    assert facts["min_dist_closest"] == 5.0


def test_a_trace_from_before_p5_has_no_readings_and_does_not_crash():
    """The two distance columns arrived in P5; a P3 or P4 trace has neither,
    and the diagnosis of an old run should still be readable."""
    ticks = [{k: v for k, v in tick.items()
              if k not in ("dist_closest", "dist_subgoal")}
             for tick in a_trace()["ticks_log"]]
    facts = diagnosis.diagnose(a_trace(ticks_log=ticks)).evidence
    assert facts["mean_dist_closest"] is None


def test_the_tally_counts_modes_most_common_first():
    traces = [a_trace(), a_trace(), a_trace(success=1, outcome="success")]
    counts = diagnosis.tally(diagnosis.diagnose_all(traces))
    assert counts == [("ran out of ticks", 2), ("reached the goal", 1)]


# --- the first contact --------------------------------------------------------
# P5's field-of-view experiment asks where each episode was first touched, how
# well it had been tracking its trail until then, and whether the wedge ->
# freeze mechanism followed. These are the columns that answer it.

def contact_at(tick_index, bearing, count=100):
    """A tick log whose only collisions start at `tick_index`, then the head
    loses the trail (reading 9, past close_threshold 3) at full speed."""
    ticks = a_trace()["ticks_log"][:count]
    for tick in ticks[tick_index:]:
        tick.update(collided=True, contact_bearing_deg=[bearing],
                    dist_closest=9.0, v=0.2)
    return ticks


def test_a_contact_at_the_left_shoulder_is_named_left():
    facts = diagnosis.diagnose(a_trace(ticks_log=contact_at(40, 75.0))).evidence
    assert facts["first_contact_tick"] == 40
    assert facts["first_contact_bearing_deg"] == 75.0
    assert facts["first_contact_side"] == "left"


def test_the_front_band_is_the_old_cameras_view():
    """28.9 deg was the 45-degree camera's horizontal half field of view; a
    contact inside it was something the model could have seen."""
    assert diagnosis.side_of(25.0) == "front"
    assert diagnosis.side_of(-35.0) == "right"
    assert diagnosis.side_of(175.0) == "rear"
    assert diagnosis.side_of(None) is None


def test_bearings_either_side_of_behind_average_to_behind_not_ahead():
    assert abs(diagnosis.mean_bearing_deg([179.0, -179.0])) == pytest.approx(180.0)


def test_the_mechanism_after_contact_is_measured():
    facts = diagnosis.diagnose(a_trace(ticks_log=contact_at(40, 75.0))).evidence
    assert facts["head_lost_after_contact"] == 1.0
    assert facts["full_speed_after_contact"] == 1.0


def test_off_trail_is_measured_to_the_path_segments():
    """The fixture drives along y = 0; a reference path along y = 0.15 is
    0.15 m off everywhere, including between its two far-apart vertices."""
    reference = [[-10.0, 0.15], [10.0, 0.15]]
    facts = diagnosis.diagnose(a_trace(ticks_log=contact_at(40, 75.0)),
                               reference_path=reference).evidence
    assert facts["off_trail_at_first_contact_m"] == pytest.approx(0.15)
    assert facts["max_off_trail_before_contact_m"] == pytest.approx(0.15)


def test_an_episode_without_contact_has_blank_contact_columns():
    facts = diagnosis.diagnose(a_trace(), reference_path=[[0, 0], [5, 0]]).evidence
    assert facts["first_contact_tick"] is None
    assert facts["first_contact_side"] is None
    assert facts["max_off_trail_before_contact_m"] == 0.0


# --- drift: which side of the trail, and for how long ---------------------------

def test_left_of_the_trail_is_positive_right_is_negative():
    east = [[0.0, 0.0], [10.0, 0.0]]
    assert diagnosis.signed_offset_from_path((5.0, 0.3), east) == pytest.approx(0.3)
    assert diagnosis.signed_offset_from_path((5.0, -0.3), east) == pytest.approx(-0.3)
    assert diagnosis.distance_to_path((5.0, -0.3), east) == pytest.approx(0.3)


def test_the_side_follows_the_direction_the_trail_runs():
    west = [[10.0, 0.0], [0.0, 0.0]]
    assert diagnosis.signed_offset_from_path((5.0, 0.3), west) == pytest.approx(-0.3)


def test_a_held_offset_is_one_long_run():
    assert diagnosis.longest_one_side_run([0.1] * 40) == 40


def test_crossing_back_breaks_the_run():
    offsets = [0.1] * 10 + [-0.1] * 3 + [0.1] * 12
    assert diagnosis.longest_one_side_run(offsets) == 12


def test_riding_the_trail_itself_is_no_side_at_all():
    """Within 2 cm of the trail is on it — a straddling robot wobbles by
    millimetres, and those wobbles are not drift."""
    assert diagnosis.longest_one_side_run([0.01, -0.015, 0.005] * 10) == 0
    assert diagnosis.longest_one_side_run([0.1, 0.01, 0.1, 0.1]) == 2


def test_the_drift_columns_come_from_the_ticks_before_contact():
    """The fixture drives along y = 0; a reference along y = -0.15 puts the
    robot 0.15 m to its left for every tick before the contact at tick 40."""
    facts = diagnosis.diagnose(a_trace(ticks_log=contact_at(40, 75.0)),
                               reference_path=[[-10.0, -0.15], [10.0, -0.15]]
                               ).evidence
    assert facts["mean_abs_off_trail_before_contact_m"] == pytest.approx(0.15)
    assert facts["mean_signed_off_trail_before_contact_m"] == pytest.approx(0.15)
    assert facts["longest_one_side_run_ticks"] == 40
