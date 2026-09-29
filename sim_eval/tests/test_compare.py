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

import comparison  # noqa: E402
import p6_2_compare as compare  # noqa: E402

EXPECTED = {("Rs_00", 1000), ("Rs_01", 1001)}


def row(checkpoint, task_id, seed, **overrides):
    values = {"checkpoint": checkpoint, "scene": task_id.split("_")[0],
              "task_id": task_id, "seed": str(seed),
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
    assert comparison.fairness_problems(tables, EXPECTED) == []


def test_an_arm_short_of_a_task_is_refused():
    tables = {"best_combined": arm("best_combined"),
              "clean_stock": arm("clean_stock")[:1]}
    [problem] = comparison.fairness_problems(tables, EXPECTED)
    assert "clean_stock" in problem and "missing" in problem and "Rs_01" in problem


def test_a_task_scored_twice_is_refused():
    tables = {"best_combined": arm("best_combined") + arm("best_combined")[:1]}
    [problem] = comparison.fairness_problems(tables, EXPECTED)
    assert "scored twice" in problem


def test_a_task_under_the_wrong_seed_is_both_missing_and_foreign():
    tables = {"best_combined": [row("best_combined", "Rs_00", 1000),
                                row("best_combined", "Rs_01", 9999)]}
    problems = comparison.fairness_problems(tables, EXPECTED)
    assert any("missing" in problem for problem in problems)
    assert any("not in the task set" in problem for problem in problems)


def test_arms_steered_differently_are_refused():
    tables = {"best_combined": arm("best_combined"),
              "clean_stock": arm("clean_stock", driver="n8w3r4t3")}
    [problem] = comparison.fairness_problems(tables, EXPECTED)
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


def test_expected_episodes_are_every_task_under_every_offset():
    tasks = [type("Task", (), {"task_id": "Rs_00", "seed": 1000})(),
             type("Task", (), {"task_id": "Rs_01", "seed": 1001})()]
    assert comparison.expected_episodes(tasks, [0]) == EXPECTED
    assert comparison.expected_episodes(tasks, [0, 100000]) == EXPECTED | {
        ("Rs_00", 101000), ("Rs_01", 101001)}


def seeded_arm(checkpoint, offsets=(0, 100000)):
    return [row(checkpoint, task_id, seed + offset)
            for task_id, seed in sorted(EXPECTED) for offset in offsets]


EXPECTED_TWO_SEEDS = {(task_id, seed + offset) for task_id, seed in EXPECTED
                      for offset in (0, 100000)}


def test_four_arms_under_two_seeds_are_a_comparison():
    names = ("ctx03", "stride3", "img160x120", "bc30")
    tables = {name: seeded_arm(name) for name in names}
    assert comparison.fairness_problems(tables, EXPECTED_TWO_SEEDS) == []
    table = compare.markdown_table(compare.comparison_rows(tables), list(names),
                                   {name: "x" for name in names})
    # n is the 2 tasks, not the 4 episodes: a task's seeds are averaged first.
    assert "| success_rate | " + " | ".join(["1.000 ± 0.000 (2)"] * 4) + " |" in table


def test_an_arm_that_skipped_a_seed_is_refused():
    tables = {"ctx03": seeded_arm("ctx03"), "bc30": seeded_arm("bc30", offsets=(0,))}
    [problem] = comparison.fairness_problems(tables, EXPECTED_TWO_SEEDS)
    assert "bc30" in problem and "missing" in problem and "seed 101000" in problem


def test_episodes_are_listed_task_by_task_with_the_arms_side_by_side():
    tables = {"best_combined": arm("best_combined"), "clean_stock": arm("clean_stock")}
    ordered = compare.episodes_side_by_side(tables)
    assert [(r["task_id"], r["checkpoint"]) for r in ordered] == [
        ("Rs_00", "best_combined"), ("Rs_00", "clean_stock"),
        ("Rs_01", "best_combined"), ("Rs_01", "clean_stock")]



def test_a_task_s_seeds_are_listed_in_order_with_the_arms_side_by_side():
    tables = {"ctx03": seeded_arm("ctx03"), "bc30": seeded_arm("bc30")}
    ordered = compare.episodes_side_by_side(tables)
    assert [(r["task_id"], int(r["seed"]), r["checkpoint"]) for r in ordered[:4]] == [
        ("Rs_00", 1000, "ctx03"), ("Rs_00", 1000, "bc30"),
        ("Rs_00", 101000, "ctx03"), ("Rs_00", 101000, "bc30")]



# --- naming and the per-house split (P7) -------------------------------------

def test_a_config_without_a_comparison_section_keeps_p6s_file_names():
    naming = comparison.ComparisonNaming()
    assert naming.files == ("p6_2_comparison.md", "p6_2_comparison.csv",
                            "p6_2_episodes.csv", "p6_2_tasks.csv")


def test_a_named_comparison_writes_its_own_files(tmp_path):
    config = tmp_path / "run.yaml"
    config.write_text("comparison:\n  name: p7_1_rescore\n  title: Rescore\n")
    naming = comparison.ComparisonNaming.from_yaml(config)
    assert naming.markdown == "p7_1_rescore_comparison.md"
    assert naming.title == "Rescore"


def test_each_house_gets_its_own_rows_beside_the_whole():
    maben = [row("bc30", "Maben_00", 2000, success="0")]
    tables = {"bc30": arm("bc30") + maben}
    rows = compare.comparison_rows(tables) + compare.per_house_rows(tables)
    success = {r["scope"]: r["mean"] for r in rows if r["statistic"] == "success_rate"}
    assert success == {"all": round(2 / 3, 4), "Rs": 1.0, "Maben": 0.0}
    table = compare.markdown_table(rows, ["bc30"], {"bc30": "x"}, title="T")
    assert table.startswith("# T\n")
    assert "## All houses" in table and "## House: Rs" in table
    assert "## House: Maben" in table
