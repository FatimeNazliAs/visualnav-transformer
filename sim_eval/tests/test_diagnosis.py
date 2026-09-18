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
