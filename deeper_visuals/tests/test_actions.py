# deeper_visuals/tests/test_actions.py
"""
The action-space arithmetic — the highest-risk code in the library.

`common/actions.py` is the chain the checkpoint was fitted against. Getting the
*order* of its three steps wrong changes the shape of a path rather than only its
size, which is exactly the failure mode of the frozen
`debug_visuals/visualize_stage5.py`. These tests need no GPU, no checkpoint and
no dataset, and they run in milliseconds.
"""

import numpy as np

from deeper_visuals.common import actions, settings


def _route(n=None, seed=0):
    """A plausible route in waypoint units — cumulative, so it curves."""
    n = n or settings.NUM_ACTIONS
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=0.6, size=(n, settings.ACTION_DIM)), axis=0)


def test_round_trip_is_exact():
    """to_waypoints undoes to_normalised_deltas, up to the metric scale."""
    route = _route()
    back = actions.to_waypoints(actions.to_normalised_deltas(route))
    expected = route * settings.METRIC_WAYPOINT_SPACING
    assert np.allclose(back, expected, atol=1e-12), np.abs(back - expected).max()


def test_order_of_operations_actually_matters():
    """
    The trap, pinned.

    Un-normalising is affine with a non-zero offset, so it does not commute with
    the cumulative sum. If someone "simplifies" to_waypoints by summing first and
    rescaling after, this test fails — which is the whole reason the docstring
    says it is a function and not a call site.
    """
    normalised = actions.to_normalised_deltas(_route())
    low = np.asarray(settings.ACTION_MIN, dtype=float)
    high = np.asarray(settings.ACTION_MAX, dtype=float)

    correct = actions.to_waypoints(normalised)
    wrong = ((normalised.cumsum(axis=-2) + 1) / 2 * (high - low) + low) \
        * settings.METRIC_WAYPOINT_SPACING

    assert not np.allclose(correct, wrong), (
        "cumsum-then-rescale produced the same answer as rescale-then-cumsum; "
        "the offset that makes the order matter has gone missing"
    )


def test_leading_axes_are_left_alone():
    """A whole descent decodes the same as its steps decoded one at a time."""
    batch = np.stack([actions.to_normalised_deltas(_route(seed=s)) for s in range(4)])
    together = actions.to_waypoints(batch)
    assert together.shape == batch.shape
    for i in range(len(batch)):
        assert np.allclose(together[i], actions.to_waypoints(batch[i]))


def test_a_straight_route_stays_straight():
    """Constant deltas decode to evenly spaced points on a line."""
    steps = np.tile([1.0, 0.0], (settings.NUM_ACTIONS, 1)).cumsum(axis=0)
    path = actions.to_waypoints(actions.to_normalised_deltas(steps))
    gaps = np.diff(path, axis=0)
    assert np.allclose(gaps, gaps[0]), gaps
    assert np.allclose(path[:, 1], 0.0), "a straight-ahead route drifted sideways"


def test_travelled_measures_from_the_robot():
    """The first leg counts: a path is measured from (0, 0), not from step 1."""
    one_step = np.array([[0.5, 0.0]])
    assert np.isclose(actions.travelled(one_step), 0.5)

    straight = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert np.isclose(actions.travelled(straight), 3.0)


def test_travelled_is_at_least_the_endpoint_distance():
    """A path can wander, but it can never be shorter than the crow flies."""
    for seed in range(5):
        path = actions.to_waypoints(actions.to_normalised_deltas(_route(seed=seed)))
        assert actions.travelled(path) >= np.linalg.norm(path[-1]) - 1e-12
