"""Pin the check that stands between two tables and a headline number.

`p6_2_compare.py` refuses to report unless the tables are a comparison (plan
§7): every arm ran every task of the set exactly once, under the task's seed,
steered and judged under the same settings. Each of those can fail without
anything crashing — a resumed run that scored a task twice, an arm that
stopped two tasks short — and each would bias the mean it feeds.

No GPU, no simulator: rows are plain dicts, as `metrics.read_table` returns.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import p6_2_compare as compare  # noqa: E402

EXPECTED = {("Rs_00", 1000), ("Rs_01", 1001)}


def row(checkpoint, task_id, seed, **overrides):
    values = {"checkpoint": checkpoint, "task_id": task_id, "seed": str(seed),
              "driver": "n8w2r4t3", "success_radius_m": "1.0",
              "success_metric": "geodesic", "success": "1", "spl": "0.9",
              "collision_events": "0", "collision_events_per_m": "0.0",
              "contact_tick_fraction": "0.0", "final_geodesic_distance_m": "0.3",
              "path_length_m": "4.0", "ticks": "40"}
    values.update(overrides)
    return values


def arm(checkpoint, **overrides):
    return [row(checkpoint, "Rs_00", 1000, **overrides),
            row(checkpoint, "Rs_01", 1001, **overrides)]


def test_two_complete_matching_arms_are_a_comparison():
    tables = {"best_combined": arm("best_combined"), "clean_stock": arm("clean_stock")}
    assert compare.fairness_problems(tables, EXPECTED) == []


def test_an_arm_short_of_a_task_is_refused():
    tables = {"best_combined": arm("best_combined"),
              "clean_stock": arm("clean_stock")[:1]}
    [problem] = compare.fairness_problems(tables, EXPECTED)
    assert "clean_stock" in problem and "missing" in problem and "Rs_01" in problem


def test_a_task_scored_twice_is_refused():
    tables = {"best_combined": arm("best_combined") + arm("best_combined")[:1]}
    [problem] = compare.fairness_problems(tables, EXPECTED)
    assert "scored twice" in problem


def test_a_task_under_the_wrong_seed_is_both_missing_and_foreign():
    tables = {"best_combined": [row("best_combined", "Rs_00", 1000),
                                row("best_combined", "Rs_01", 9999)]}
    problems = compare.fairness_problems(tables, EXPECTED)
    assert any("missing" in problem for problem in problems)
    assert any("not in the task set" in problem for problem in problems)


def test_arms_steered_differently_are_refused():
    tables = {"best_combined": arm("best_combined"),
              "clean_stock": arm("clean_stock", driver="n8w3r4t3")}
    [problem] = compare.fairness_problems(tables, EXPECTED)
    assert "driver" in problem


def test_the_table_has_a_cell_for_every_statistic_and_arm():
    tables = {"best_combined": arm("best_combined"), "clean_stock": arm("clean_stock")}
    rows = compare.comparison_rows(tables)
    table = compare.markdown_table(rows, list(tables),
                                   {name: "x" for name in tables})
    assert "| success_rate | 1.000 ± 0.000 (2) | 1.000 ± 0.000 (2) |" in table
    assert len(rows) == 2 * len(compare.metrics.EPISODE_STATISTICS)


def test_a_cell_says_what_is_missing():
    assert compare.format_cell("", "", 0) == "n/a (0)"
    assert compare.format_cell(0.5, None, 1) == "0.500 ± n/a (1)"


def test_episodes_are_listed_task_by_task_with_the_arms_side_by_side():
    tables = {"best_combined": arm("best_combined"), "clean_stock": arm("clean_stock")}
    ordered = compare.episodes_side_by_side(tables)
    assert [(r["task_id"], r["checkpoint"]) for r in ordered] == [
        ("Rs_00", "best_combined"), ("Rs_00", "clean_stock"),
        ("Rs_01", "best_combined"), ("Rs_01", "clean_stock")]
