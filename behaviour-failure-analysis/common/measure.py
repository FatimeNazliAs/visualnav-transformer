# behaviour-failure-analysis/common/measure.py
"""
Measurement helpers shared by the phases.

Everything here returns a NUMBER, never a formatted string. That is the rule
this module exists to hold: facts.json stores what was measured, and how it
looks is decided at the page seam by build_page.fill's format specs. A helper
that returned "48%" would put presentation back on the far side of a boundary
that needs a GPU to re-cross.

Cosine similarity lived in four places before this module — twice
character-for-character identical (p2_encoders/occlusion.py and
p3_transformer_masking/run_model.py) and twice as row-normalised variants. Two
advisor pages quote a cosine; four definitions of it meant nothing in the code
said those numbers were comparable.
"""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_to_each(base: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between one vector and each row of a 2-D array.

    Returns (len(rows),). The batched form of `cosine`, kept separate rather
    than special-cased inside it, because the callers differ: one asks about a
    pair, the other sweeps hundreds of perturbations.
    """
    unit_base = base / np.linalg.norm(base)
    unit_rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    return unit_rows @ unit_base


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean cosine similarity between two sets of row vectors.

    When both sides are the same set the diagonal is dropped — a vector's
    similarity to itself is 1 by definition and would only pull the average up.
    """
    unit_a = a / np.linalg.norm(a, axis=1, keepdims=True)
    unit_b = b / np.linalg.norm(b, axis=1, keepdims=True)
    similarity = unit_a @ unit_b.T
    if a.shape == b.shape and np.array_equal(a, b):
        return float(similarity[~np.eye(len(a), dtype=bool)].mean())
    return float(similarity.mean())


def millions(n: int) -> float:
    """A parameter count in millions, as a number for the page to format."""
    return float(n) / 1e6


def widest_gap(points: np.ndarray) -> float:
    """
    The largest distance between any two of a set of points, in their own units.

    `points` is (n, dim). This is how far apart a set of independent samples
    actually are, and it is deliberately not a standard deviation: a std is a
    spread about a mean, and the mean of several sampled paths is a fiction —
    it is the averaged-away answer a diffusion policy exists in order not to
    produce. The widest gap needs no centre to be defined against, so it can be
    quoted on a page without implying one.

    Defined by `widest_pair`, so that a figure annotating the gap and a page
    quoting it are provably talking about the same two samples.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    first, second = widest_pair(points)
    return float(np.linalg.norm(points[first] - points[second]))


def widest_pair(points: np.ndarray) -> tuple[int, int]:
    """
    Which two of a set of points are furthest apart — their indices.

    Separate from `widest_gap` because a figure needs the pair, not the
    distance: an arrow drawn between two samples chosen independently of the
    quoted number is exactly how a picture ends up annotating a gap it is not
    showing. O(n^2), which is nothing at the handful of runs a phase samples.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return (0, 0)
    deltas = points[:, None, :] - points[None, :, :]
    flat = int(np.linalg.norm(deltas, axis=-1).argmax())
    return divmod(flat, len(points))


def gap_per_step(paths: np.ndarray) -> np.ndarray:
    """
    How far apart a set of paths are at each step along them — (n_steps,).

    `paths` is (n_paths, n_steps, dim), all measured from the same origin. The
    answer is one `widest_gap` per step, which is the series a phase plots when
    the question is not "do these differ" but "differ WHERE": paths that leave
    the same point can only disagree by degrees, and the degree is what grows.
    """
    paths = np.asarray(paths, dtype=float)
    return np.array([widest_gap(paths[:, step]) for step in range(paths.shape[1])])


def error_per_step(paths: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    How far a path is from a reference path, at each step along it — metres.

    `paths` is (..., n_steps, dim) and `reference` is (n_steps, dim); the leading
    axes are left alone, so one path and a stack of them are the same call. The
    answer keeps its per-step shape rather than reducing, because the two phases
    that need it reduce differently: one takes the endpoint of a single path, the
    other the worst endpoint across several.

    This is the measurement that says whether a predicted path is any good, and
    both P4 and P5 quote it on their advisor pages. Each had written its own
    `measure_against_truth` — same name, same quantity, different reductions,
    different key names, different units — which is exactly the situation this
    module was created to end for cosine similarity ("four definitions of it
    meant nothing in the code said those numbers were comparable"). Two pages a
    reader sees back to back now cite one definition.
    """
    return np.linalg.norm(
        np.asarray(paths, dtype=float) - np.asarray(reference, dtype=float),
        axis=-1)
