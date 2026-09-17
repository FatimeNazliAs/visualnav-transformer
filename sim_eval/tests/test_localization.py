"""Pin the localization window — the arithmetic that picks which node to steer at.

This is the highest-stakes untested logic the P1 review found, for one reason:
upstream has already shipped this bug. Commit 7b5b24c ("Fix closest node update
for topomap localization in navigate.py") replaced

    closest_node = np.argmin(distances)      # window-relative
with
    closest_node = start + min_dist_idx      # absolute

An off-by-one between those two kinds of index does not raise. It steers the
robot confidently toward the wrong node, which looks plausible in a replay and
silently corrupts every metric computed from the run. No rollout can catch it;
only these assertions can.

The functions under test are pure numpy on purpose, so this file needs no
torch, no checkpoint and no GPU — unlike the model they are called from.

    python -m pytest sim_eval/tests/test_localization.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver import DEFAULT_CLOSE_THRESHOLD, DEFAULT_RADIUS  # noqa: E402
from nomad_policy import (  # noqa: E402
    localization_window,
    localize,
    window_size,
)

# A 15-node trail, matching the hand-made P1 topomap.
GOAL_NODE = 14


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------

def test_window_is_symmetric_away_from_both_ends():
    """Mid-trail, the window is radius behind and radius+1 ahead."""
    start, end = localization_window(closest_node=7, goal_node=GOAL_NODE, radius=4)
    assert (start, end) == (3, 12)


def test_window_floors_at_the_first_node():
    """At the start of an episode closest_node is 0, so the window cannot run negative."""
    start, end = localization_window(closest_node=0, goal_node=GOAL_NODE, radius=4)
    assert start == 0
    assert end == 5


def test_window_ceilings_at_the_goal_node():
    """It must never score a node past the goal — there is nothing there."""
    start, end = localization_window(closest_node=13, goal_node=GOAL_NODE, radius=4)
    assert end == GOAL_NODE


def test_window_at_the_goal_still_contains_the_goal():
    """The stopping condition is `closest_node == goal_node`, so it has to stay reachable."""
    start, end = localization_window(
        closest_node=GOAL_NODE, goal_node=GOAL_NODE, radius=4)
    assert start <= GOAL_NODE <= end
    assert window_size(start, end) >= 1


def test_window_size_matches_the_slice_the_caller_takes():
    """act() slices [start:end+1]; the encoder output has one row per node in it.

    `localize` clamps the subgoal against that row count, so if these two
    disagree the clamp is wrong.
    """
    for closest_node in range(0, GOAL_NODE + 1):
        start, end = localization_window(closest_node, GOAL_NODE, DEFAULT_RADIUS)
        nodes = list(range(GOAL_NODE + 1))[start:end + 1]
        assert window_size(start, end) == len(nodes)


def test_a_two_node_trail_still_produces_a_usable_window():
    """The degenerate trail: the window must not come back empty."""
    start, end = localization_window(closest_node=0, goal_node=1, radius=4)
    assert window_size(start, end) == 2


# --------------------------------------------------------------------------
# the pick — this is the part that had the bug
# --------------------------------------------------------------------------

def test_closest_node_is_absolute_not_window_relative():
    """THE regression test for upstream bug 7b5b24c.

    The nearest node is at offset 1 of a window starting at 3, so the answer is
    node 4 — not node 1. Returning the offset is the original bug, and it is
    what makes the robot chase a node ten places behind where it actually is.
    """
    distances = np.array([9.0, 0.5, 8.0, 7.0])
    closest_node, _offset = localize(
        distances, start=3, num_window_nodes=4, close_threshold=DEFAULT_CLOSE_THRESHOLD)
    assert closest_node == 4
    assert closest_node != int(np.argmin(distances))


def test_start_of_zero_makes_the_two_index_kinds_agree():
    """A window at the trail's head is the one case where the bug is invisible.

    Worth pinning precisely because it is why the original bug survived review.
    """
    distances = np.array([9.0, 0.5, 8.0])
    closest_node, _offset = localize(
        distances, start=0, num_window_nodes=3, close_threshold=DEFAULT_CLOSE_THRESHOLD)
    assert closest_node == 1


def test_subgoal_steps_one_node_ahead_when_the_current_node_is_close():
    """`dists[min_idx] < close_threshold` means "arrived", so aim at the next one."""
    distances = np.array([9.0, 1.0, 8.0, 7.0])
    _closest, offset = localize(
        distances, start=3, num_window_nodes=4, close_threshold=3)
    assert offset == 2


def test_subgoal_stays_put_while_the_current_node_is_still_far():
    distances = np.array([9.0, 5.0, 8.0, 7.0])
    _closest, offset = localize(
        distances, start=3, num_window_nodes=4, close_threshold=3)
    assert offset == 1


def test_subgoal_is_clamped_to_the_last_node_in_the_window():
    """Stepping ahead from the final node must not index past the encoder output."""
    distances = np.array([9.0, 8.0, 0.5])
    _closest, offset = localize(
        distances, start=12, num_window_nodes=3, close_threshold=3)
    assert offset == 2


def test_the_subgoal_offset_always_indexes_the_window():
    """Whatever the scores, the offset is a legal row of the encoder output."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        num_nodes = int(rng.integers(1, 10))
        distances = rng.uniform(0.0, 20.0, size=num_nodes)
        _closest, offset = localize(
            distances, start=int(rng.integers(0, 12)),
            num_window_nodes=num_nodes,
            close_threshold=DEFAULT_CLOSE_THRESHOLD)
        assert 0 <= offset < num_nodes


def test_the_subgoal_is_never_behind_the_current_node():
    """Localization may hold position or advance; it may not walk the trail backwards."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        num_nodes = int(rng.integers(1, 10))
        distances = rng.uniform(0.0, 20.0, size=num_nodes)
        start = int(rng.integers(0, 12))
        closest_node, offset = localize(
            distances, start=start, num_window_nodes=num_nodes,
            close_threshold=DEFAULT_CLOSE_THRESHOLD)
        assert start + offset >= closest_node


@pytest.mark.parametrize("ties", [
    np.array([1.0, 1.0, 1.0]),
    np.array([0.0, 0.0]),
])
def test_ties_resolve_to_the_earliest_node(ties):
    """`argmin` takes the first minimum; pinned so a "nicer" tie-break cannot sneak in.

    Preferring the later node on a tie would let the robot skip trail nodes.
    """
    closest_node, _offset = localize(
        ties, start=5, num_window_nodes=len(ties), close_threshold=0)
    assert closest_node == 5
