"""
The spread arithmetic — what P5's page quotes and what its arrows point at.

`common/measure.widest_gap` and `widest_pair` are the only numbers on P5's page
that are not a length or a distance from the recorded route, and the pair is
what the hero figure draws an arrow between. If those two ever disagree the
figure annotates one pair of runs while the caption quotes another, and nothing
about the resulting picture looks wrong. Pinned here instead.

No GPU, no checkpoint, no dataset.
"""

import numpy as np

from deeper_visuals.common import measure


def _fan(n_paths=6, n_steps=8, seed=0):
    """Paths that leave one origin together and separate as they go."""
    rng = np.random.default_rng(seed)
    spreading = np.linspace(0.0, 1.0, n_steps)[None, :, None]
    return np.cumsum(rng.normal(size=(n_paths, n_steps, 2)) * spreading, axis=1)


def test_widest_gap_is_the_largest_pairwise_distance():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    assert np.isclose(measure.widest_gap(points), 5.0)


def test_widest_pair_names_the_points_the_gap_measures():
    """The figure's arrow and the page's number must describe the same pair."""
    for seed in range(5):
        points = np.random.default_rng(seed).normal(size=(7, 2))
        first, second = measure.widest_pair(points)
        assert np.isclose(measure.widest_gap(points),
                          np.linalg.norm(points[first] - points[second]))


def test_a_lone_point_has_no_gap():
    assert measure.widest_gap(np.array([[1.0, 2.0]])) == 0.0
    assert measure.widest_pair(np.array([[1.0, 2.0]])) == (0, 0)


def test_identical_points_have_no_gap():
    assert measure.widest_gap(np.zeros((5, 2))) == 0.0


def test_the_gap_does_not_depend_on_the_order_of_the_runs():
    """Sampled runs are exchangeable; a measure over them must be too."""
    points = np.random.default_rng(3).normal(size=(6, 2))
    shuffled = points[np.random.default_rng(4).permutation(len(points))]
    assert np.isclose(measure.widest_gap(points), measure.widest_gap(shuffled))


def test_the_gap_is_not_a_spread_about_the_mean():
    """
    The trap this measure exists to avoid.

    A standard deviation describes variation about a mean, and the mean of
    several sampled paths is the averaged-away answer a diffusion policy is
    chosen in order not to produce. Two points four apart have a gap of four
    and a std of two; if someone "simplifies" widest_gap into a spread, the
    page starts quoting half of what its own figure shows.
    """
    points = np.array([[-2.0, 0.0], [2.0, 0.0]])
    assert np.isclose(measure.widest_gap(points), 4.0)
    assert not np.isclose(measure.widest_gap(points),
                          np.linalg.norm(points - points.mean(axis=0), axis=1).mean())


def test_gap_per_step_is_one_gap_for_each_step():
    paths = _fan(n_paths=5, n_steps=8)
    gaps = measure.gap_per_step(paths)
    assert gaps.shape == (8,)
    for step in range(8):
        assert np.isclose(gaps[step], measure.widest_gap(paths[:, step]))


def test_paths_that_never_differ_never_have_a_gap():
    one = _fan(n_paths=1)[0]
    assert np.allclose(measure.gap_per_step(np.stack([one, one, one])), 0.0)


def test_error_per_step_leaves_leading_axes_alone():
    """One path and a stack of them are the same call — P4 and P5's shared need."""
    reference = _fan(n_paths=1)[0]
    stack = _fan(n_paths=4)
    together = measure.error_per_step(stack, reference)
    assert together.shape == (4, stack.shape[1])
    for i in range(len(stack)):
        assert np.allclose(together[i], measure.error_per_step(stack[i], reference))


def test_a_path_has_no_error_against_itself():
    path = _fan(n_paths=1)[0]
    assert np.allclose(measure.error_per_step(path, path), 0.0)
