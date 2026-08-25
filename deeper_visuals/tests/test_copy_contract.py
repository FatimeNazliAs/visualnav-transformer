# deeper_visuals/tests/test_copy_contract.py
"""
The advisor-page contract and the facts seam.

`build_page.py` never imports torch — that is a stated design goal and it is
what lets `--page-only` rebuild in under a second. But its pure half was still
only reachable through `build()`, which wants a real facts.json, real PNGs and
the templates, so the code most likely to be edited was the code hardest to
exercise. These tests reach it directly.
"""

import numpy as np

from deeper_visuals.common import build_page, measure


# ── the substitution seam ─────────────────────────────────────────────────────

def test_flatten_makes_dotted_keys():
    flat = build_page.flatten({"scene": {"traj": "x", "frame": 54}, "phase": "p4"})
    assert flat == {"scene.traj": "x", "scene.frame": 54, "phase": "p4"}


def test_fill_substitutes_and_formats():
    flat = {"a.n": 12.0306, "a.name": "right"}
    assert build_page.fill("{a.name}", flat) == "right"
    assert build_page.fill("{a.n:.0f}", flat) == "12"
    assert build_page.fill("{a.n:.2f} steps", flat) == "12.03 steps"


def test_fill_renders_a_list_per_element():
    """The reason no phase pre-joins a progression into a string."""
    flat = {"a.per_layer": [0.01, 0.05]}
    assert build_page.fill("{a.per_layer:.0%}", flat) == "1% → 5%"


def test_fill_names_the_missing_key_and_lists_what_exists():
    try:
        build_page.fill("{nope.here}", {"scene.traj": "x"})
    except KeyError as exc:
        assert "nope.here" in str(exc) and "scene.traj" in str(exc)
    else:
        raise AssertionError("a missing key must raise")


def test_fill_reports_a_spec_that_cannot_apply():
    try:
        build_page.fill("{a.name:.2f}", {"a.name": "right"})
    except ValueError as exc:
        assert "a.name" in str(exc)
    else:
        raise AssertionError("a bad format spec must raise")


# ── the reverse check: facts nobody quotes ────────────────────────────────────

def test_unread_facts_finds_what_no_page_quotes():
    page = {"heading": "x {descent.n_steps}", "points": ["{accuracy.mean_cm:.0f} cm"]}
    flat = {"descent.n_steps": 10, "accuracy.mean_cm": 2.2, "descent.n_rises": 2}
    assert build_page.unread_facts(page, flat) == ["descent.n_rises"]


def test_provenance_is_never_unread():
    """The footer reads these, not page.yaml, so they must not be reported."""
    page = {"heading": "x"}
    flat = {"phase": "p4", "generated": "…", "figures": ["a.png"],
            "scene.traj": "no11vc_9_1", "model.checkpoint_file": "ema_99.pth"}
    assert build_page.unread_facts(page, flat) == []


# ── the copy caps ─────────────────────────────────────────────────────────────

def test_one_line_collapses_a_folded_block():
    assert build_page.one_line("a\n  b\nc\n") == "a b c"


def test_bold_markup_does_not_count_against_a_cap():
    """Bolding a term must never be what trips the limit."""
    body = "x" * build_page.COPY_LIMITS["point"]
    build_page.within_length("point", f"<b>{body}</b>")


def test_over_length_copy_is_refused_with_the_count():
    body = "x" * (build_page.COPY_LIMITS["heading"] + 1)
    try:
        build_page.within_length("heading", body)
    except ValueError as exc:
        assert str(len(body)) in str(exc)
    else:
        raise AssertionError("an over-length heading must raise")


def test_list_counts_are_bounded_both_ways():
    low, high = build_page.LIST_LIMITS["points"]
    build_page.within_count("points", ["p"] * low)
    build_page.within_count("points", ["p"] * high)
    for n in (low - 1, high + 1):
        try:
            build_page.within_count("points", ["p"] * n)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{n} points must raise")


def test_as_text_escapes_but_as_rich_keeps_bold():
    assert "<b>" not in build_page.as_text("a <b>b</b>")
    assert "<b>" in build_page.as_rich("a <b>b</b>")


# ── measure.py: pure, and previously untested ─────────────────────────────────

def test_cosine_is_one_for_a_vector_with_itself():
    v = np.array([1.0, 2.0, 3.0])
    assert np.isclose(measure.cosine(v, v), 1.0)
    assert np.isclose(measure.cosine(v, -v), -1.0)
    assert np.isclose(measure.cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)


def test_mean_cosine_drops_the_diagonal_only_for_the_same_set():
    """
    The branch that picks a statistic by inspecting the data.

    Two identical-but-distinct sets take the self-similarity branch, which is
    worth pinning: a caller passing `.copy()` gets a different number, and
    nothing in the signature says so.
    """
    rows = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(measure.mean_cosine(rows, rows), 0.0)   # diagonal dropped
    assert np.isclose(measure.mean_cosine(rows, rows.copy()), 0.0)
    assert np.isclose(measure.mean_cosine(rows, np.array([[1.0, 0.0]])), 0.5)


def test_cosine_to_each_matches_cosine_row_by_row():
    base = np.array([1.0, 2.0, 3.0])
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    got = measure.cosine_to_each(base, rows)
    for i, row in enumerate(rows):
        assert np.isclose(got[i], measure.cosine(base, row))


def test_millions_returns_a_number_not_a_string():
    value = measure.millions(5_288_548)
    assert isinstance(value, float) and np.isclose(value, 5.288548)
