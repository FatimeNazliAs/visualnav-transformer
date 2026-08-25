"""
The seam rule, enforced instead of documented.

`facts.json` holds what was measured; how it reads is decided at the page by
`build_page.fill`'s format specs. That rule is stated in three modules'
docstrings — and was broken fifteen times, in ways nobody could see: a
parameter count stored as "3.16M" put its rounding and its unit behind a GPU
run, and ten of the fifteen pre-formatted facts were quoted by no page at all.

A docstring cannot fail a build. This can.
"""

import re

import numpy as np

from deeper_visuals.common import facts, measure

# A string that is really a number: digits, an optional decimal part, an
# optional unit or suffix. Deliberately narrow. "11 × 96 × 96" is a shape and
# survives (see facts.shape_str); "way 1 → goal" is an English clause and
# survives; "right" and "global_cond" are words. "3.16M", "48%", "0.12 m" are
# quantities wearing a costume, and they do not.
LOOKS_LIKE_A_NUMBER = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*(?:[A-Za-z]{1,3}|%)?\s*$")


def _walk(node, path=""):
    """Every leaf of a facts payload, with the dotted key that reaches it."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}{key}.")
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}{index}.")
        return
    yield path.rstrip("."), node


def offending_facts(payload: dict) -> list:
    """The keys whose value is a number stored as a string."""
    return [key for key, value in _walk(payload)
            if isinstance(value, str) and LOOKS_LIKE_A_NUMBER.match(value)]


def test_a_quantity_stored_as_a_string_is_caught():
    """The exact bug this test exists for: n_params_m == "3.16M"."""
    assert offending_facts({"transformer": {"n_params_m": "3.16M"}}) \
        == ["transformer.n_params_m"]
    assert offending_facts({"a": {"goal_share": "48%"}}) == ["a.goal_share"]
    assert offending_facts({"a": {"spacing": "0.12 m"}}) == ["a.spacing"]


def test_the_things_that_are_legitimately_strings_survive():
    """
    Shapes, words and English clauses are not quantities.

    A shape has no format spec that would render it, which is why
    facts.shape_str exists; the rest are vocabulary, not measurements.
    """
    payload = {
        "input":   {"obs_tensor": facts.shape_str(12, 96, 96)},
        "descent": {"side": "right", "shape": facts.shape_str(11, 8, 2)},
        "cond":    {"enters_as": "global_cond",
                    "conditioning": "FiLM bias per residual block"},
        "attn":    {"favourites": "way 1 → goal, way 2 → t − 3"},
    }
    assert offending_facts(payload) == []


def test_numbers_stored_as_numbers_are_never_flagged():
    payload = {"m": {"n_params_m": measure.millions(3_158_000),
                     "down_dims": [64, 128, 256],
                     "share": 0.48, "steps": 8}}
    assert offending_facts(payload) == []


def test_shape_str_is_the_one_spelling():
    """Three phases spelled this three ways before it was promoted."""
    assert facts.shape_str(12, 96, 96) == "12 × 96 × 96"
    assert facts.shape_str(*np.zeros((11, 8, 2)).shape) == "11 × 8 × 2"
