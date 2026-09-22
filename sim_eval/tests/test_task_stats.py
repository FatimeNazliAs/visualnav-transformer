"""Pin the task as the unit of analysis.

The mistake this module exists to prevent is silent: treating a task's three
seeds as three independent episodes produces a smaller SE, a smaller p and a
confident-looking table, and nothing crashes. So these pin that seeds are
averaged within a task before anything is taken across tasks — and that with
one seed per task the numbers are exactly P6's episode-level ones.

No GPU, no simulator: rows are plain dicts, as `metrics.read_table` returns.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics  # noqa: E402
import task_stats  # noqa: E402


def row(checkpoint, task_id, seed, success, spl=None, ticks=100, **overrides):
    values = {"checkpoint": checkpoint, "scene": task_id.split("_")[0],
              "task_id": task_id, "seed": str(seed), "success": str(success),
              "spl": str(spl if spl is not None else 0.8 * success),
              "collision_events": "0", "collision_events_per_m": "0.0",
              "contact_tick_fraction": "0.0", "final_geodesic_distance_m": "0.5",
              "path_length_m": "3.0", "ticks": str(ticks)}
    values.update(overrides)
    return values


def arm(checkpoint, successes, offsets=(0, 100000, 200000)):
    """{task_index: [success per seed]} as rows."""
    return [row(checkpoint, "Rs_{:02d}".format(task), 1000 + task + offset, success)
            for task, per_seed in successes.items()
            for offset, success in zip(offsets, per_seed)]


def test_a_tasks_seeds_are_averaged_into_one_value():
    rows = arm("a", {0: [1, 1, 0], 1: [0, 0, 0]})
    assert task_stats.task_values(rows, "success_rate") == {
        "Rs_00": pytest.approx(2 / 3), "Rs_01": 0.0}


def test_the_n_and_the_se_are_across_tasks_not_episodes():
    rows = arm("a", {0: [1, 1, 1], 1: [0, 0, 0]})
    mean, se, n = task_stats.summarize(rows)["success_rate"]
    assert n == 2
    assert mean == 0.5
    # Across the two task means (1 and 0): sd 0.707, / sqrt(2) = 0.5. Treating
    # the six episodes as independent would give 0.224 — the error refused.
    assert se == pytest.approx(0.5)


def test_one_seed_per_task_is_exactly_the_episode_level_summary():
    rows = [row("a", "Rs_{:02d}".format(task), 1000 + task, task % 2,
                ticks=50 + 7 * task) for task in range(10)]
    by_task, by_episode = task_stats.summarize(rows), metrics.summarize(rows)
    for statistic in by_episode:
        assert by_task[statistic] == pytest.approx(by_episode[statistic]), statistic


def test_ticks_to_goal_averages_only_a_tasks_successful_seeds():
    rows = [row("a", "Rs_00", 1000, 1, ticks=40), row("a", "Rs_00", 101000, 0, ticks=400),
            row("a", "Rs_00", 201000, 1, ticks=60), row("a", "Rs_01", 1001, 0, ticks=400)]
    assert task_stats.task_values(rows, "ticks_to_goal") == {"Rs_00": 50.0}


def test_a_blank_cell_is_left_out_not_read_as_zero():
    rows = [row("a", "Rs_00", 1000, 0, final_geodesic_distance_m=""),
            row("a", "Rs_00", 101000, 0, final_geodesic_distance_m="2.0")]
    assert task_stats.task_values(rows, "final_geodesic_distance_m") == {"Rs_00": 2.0}


def test_a_paired_contrast_differences_the_task_means():
    tables = {"a": arm("a", {0: [1, 1, 0], 1: [1, 0, 0], 2: [1, 1, 1]}),
              "b": arm("b", {0: [0, 0, 0], 1: [1, 0, 0], 2: [1, 1, 0]})}
    result = task_stats.contrast(tables, {"a": 1, "b": -1}, "success_rate")
    assert result.per_task == pytest.approx(
        {"Rs_00": 2 / 3, "Rs_01": 0.0, "Rs_02": 1 / 3})
    assert result.n == 3
    assert result.mean == pytest.approx(1 / 3)


def test_the_interaction_is_the_four_way_per_task_contrast():
    """bc − stride − img + base: zero exactly when the gains add up."""
    per_arm = {"base": 0.2, "stride": 0.4, "img": 0.3, "both": 0.5}
    tables = {name: [row(name, "Rs_{:02d}".format(task), 1000 + task, 0,
                         spl=value + 0.01 * task) for task in range(4)]
              for name, value in per_arm.items()}
    interaction = task_stats.contrast(
        tables, {"both": 1, "stride": -1, "img": -1, "base": 1}, "spl")
    assert interaction.mean == pytest.approx(0.0)
    assert task_stats.additivity_verdict("spl", interaction) == "additive"


def test_only_tasks_every_arm_has_a_value_for_count():
    tables = {"a": [row("a", "Rs_00", 1000, 1, ticks=40), row("a", "Rs_01", 1001, 1, ticks=50)],
              "b": [row("b", "Rs_00", 1000, 1, ticks=45), row("b", "Rs_01", 1001, 0)]}
    result = task_stats.contrast(tables, {"a": 1, "b": -1}, "ticks_to_goal")
    assert result.per_task == {"Rs_00": -5.0}


def contrast_of(values):
    return task_stats.Contrast({"Rs_{:02d}".format(i): v for i, v in enumerate(values)})


def test_beyond_two_se_is_flagged_and_within_is_not():
    assert not contrast_of([0.5, 0.6, 0.55, 0.45]).within_noise
    assert contrast_of([0.5, -0.4, 0.1, -0.3]).within_noise


def test_the_verdict_follows_each_statistics_better_direction():
    better = contrast_of([0.5, 0.6, 0.55, 0.45])
    assert task_stats.additivity_verdict("success_rate", better) == "super-additive"
    assert task_stats.additivity_verdict("collision_events_per_m", better) == "sub-additive"
    assert task_stats.additivity_verdict("path_length_m", better) == "non-additive (+)"


def test_all_tied_tasks_are_no_evidence():
    tied = contrast_of([0.0, 0.0, 0.0])
    assert tied.p_wilcoxon == 1.0 and tied.p_paired_t == 1.0


def test_mcnemar_counts_the_discordant_tasks():
    a = [row("a", "Rs_{:02d}".format(t), 1000 + t, s) for t, s in enumerate([1, 1, 1, 0, 1])]
    b = [row("b", "Rs_{:02d}".format(t), 1000 + t, s) for t, s in enumerate([0, 0, 1, 1, 1])]
    a_only, b_only, p = task_stats.mcnemar(a, b)
    assert (a_only, b_only) == (2, 1)
    assert p == pytest.approx(1.0)


def test_mcnemar_refuses_several_seeds_per_task():
    rows = arm("a", {0: [1, 0, 1]})
    with pytest.raises(ValueError):
        task_stats.mcnemar(rows, rows)
