# behaviour-failure-analysis/common/actions.py
"""
The action space, both directions — and nothing else.

This module is deliberately **pure**: numpy and `settings`, no torch, no
diffusers, no filesystem. That is its whole reason for existing separately from
`denoise.py`.

The arithmetic here is the highest-risk code in the library and the cheapest to
check. It is the exact chain the checkpoint was fitted against, and getting the
*order* wrong changes the shape of a path rather than only its size — which is
the mistake the frozen `debug_visuals/visualize_stage5.py` makes. But until this
split, reaching `to_waypoints` meant importing `denoise`, which imports torch and
diffusers at module scope; so the one function that needs neither a GPU nor a
checkpoint to verify could only be exercised by running a full forward pass and
looking at a PNG.

Now it can be imported with numpy alone, and 01-goal-conditioned-behaviour's
sanity_check.py pins the round trip before any experiment runs.

Copied verbatim from deeper_visuals/common/actions.py (branch
feature/deeper-visualization) apart from the import path. Do not replace it with
the frozen debug_visuals/visualize_stage5.py version, which is wrong.

The two directions:

    to_normalised_deltas   waypoint units  ->  [-1, 1] deltas   (what training
                                               corrupts; the ground-truth side)
    to_waypoints           [-1, 1] deltas  ->  metres           (what the head
                                               emits; the inference side)

They are inverses up to the metric scale, and `tests/test_actions.py` asserts it.
"""

from __future__ import annotations

import numpy as np

from common import settings


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    """The normalisation range the checkpoint was trained under, as arrays."""
    return (np.asarray(settings.ACTION_MIN, dtype=float),
            np.asarray(settings.ACTION_MAX, dtype=float))


def to_waypoints(actions: np.ndarray) -> np.ndarray:
    """
    Turn what the head emits into a path on the floor, in metres.

    The head does not predict positions. It predicts *deltas*, squashed to
    [-1, 1] — train_utils.get_delta differences the waypoints and
    normalize_data rescales them before the loss ever sees them. Undoing that
    is three steps, and skipping any one of them silently changes the shape of
    the path rather than only its size:

        1. Undo the squash, back to waypoint-unit deltas. This is affine with a
           non-zero offset, so it does NOT commute with the cumulative sum —
           cumsum-then-rescale accumulates the offset into a drift that bends
           the whole path. That is why this is a function and not a call site.
        2. Accumulate them, which is what turns deltas into waypoints.
        3. Scale to metres by the dataset's waypoint spacing.

    Mirrors train_utils.get_action, which is the inverse the trained model was
    fitted against; adapted rather than imported, per the reuse rule.

    Input is (…, NUM_ACTIONS, ACTION_DIM); the leading axes are left alone.
    Output is metres in the robot's frame at the current instant: +x ahead,
    +y to its left, starting from (0, 0).
    """
    low, high = _bounds()
    deltas = (np.asarray(actions, dtype=float) + 1) / 2 * (high - low) + low
    return deltas.cumsum(axis=-2) * settings.METRIC_WAYPOINT_SPACING


def to_normalised_deltas(waypoints: np.ndarray) -> np.ndarray:
    """
    The forward direction: a real route, in the space the head predicts in.

    `waypoints` are in *waypoint units* — metres divided by
    METRIC_WAYPOINT_SPACING — measured in the robot's frame at the current
    instant. Differencing then squashing to [-1, 1] reproduces
    train_utils.get_delta followed by normalize_data, which is what the
    diffusion head's training target actually is.

    This is the tensor `denoise.corrupt` buries under noise. It is the exact
    inverse of `to_waypoints` up to the metric scale:

        to_waypoints(to_normalised_deltas(u)) == u * METRIC_WAYPOINT_SPACING

    Input is (…, NUM_ACTIONS, ACTION_DIM); the leading axes are left alone.
    """
    low, high = _bounds()
    values = np.asarray(waypoints, dtype=float)
    leading = values.shape[:-2]
    origin = np.zeros((*leading, 1, values.shape[-1]))
    deltas = np.diff(values, axis=-2, prepend=origin)
    return (deltas - low) / (high - low) * 2 - 1


def travelled(path: np.ndarray) -> float:
    """
    How far the robot moves along a path, in metres — the sum of its steps.

    `path` is metres from the robot outward, so the first leg is measured from
    (0, 0): a path that started at its own first waypoint would be missing its
    first step, which is the same reason `figures.draw_path` prepends the origin.
    """
    legs = np.diff(path, axis=0, prepend=np.zeros((1, path.shape[1])))
    return float(np.linalg.norm(legs, axis=1).sum())
