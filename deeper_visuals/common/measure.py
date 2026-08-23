# deeper_visuals/common/measure.py
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
