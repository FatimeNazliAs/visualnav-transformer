"""Pin the five metrics, because a wrong one does not crash — it publishes.

Every number in `metrics.py` ends up in a table that a conclusion is drawn
from. A scorer that divides by the wrong length, counts a scrape along a wall
as twenty mistakes, or hands a failed episode a non-zero SPL still produces a
complete, plausible CSV, and nothing downstream would query it. So the
formulas are pinned here, against hand-computed numbers.

Needs no GPU, no iGibson and no checkpoint:

    ./sim_eval/run_tests.sh
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics  # noqa: E402


class FakeTask:
    """The three things a row needs from the task it scored."""

    def __init__(self, task_id="Rs_00", scene="Rs", geodesic_length_m=4.0,
                 node_count=12):
        self.task_id = task_id
        self.scene = scene
        self.geodesic_length_m = geodesic_length_m
        self.node_count = node_count


def make_metrics(**overrides):
    values = dict(
        checkpoint="best_combined", task=FakeTask(), seed=1000,
        driver="n8w2r4t3", success=True,
        collision_ticks=0, collision_events=0, path_length_m=5.0,
        final_geodesic_distance_m=0.4, final_euclidean_distance_m=0.4,
        ticks=80, seconds=20.0, timeout_ticks=352, success_radius_m=1.0,
        success_metric="geodesic", declared_arrival_tick=70, final_node=11)
    values.update(overrides)
    return metrics.EpisodeMetrics(**values)


# --- path length -------------------------------------------------------------

def test_path_length_sums_the_legs_and_ignores_yaw():
    """Turning on the spot buys no distance, and SPL must not pay for it."""
    poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 1.5), (1.0, 2.0, 3.0)]
    assert metrics.path_length(poses) == pytest.approx(3.0)


def test_a_single_pose_has_no_path():
    assert metrics.path_length([(0.0, 0.0, 0.0)]) == 0.0
    assert metrics.path_length([]) == 0.0


# --- SPL ---------------------------------------------------------------------

def test_spl_is_the_ratio_of_shortest_to_driven():
    assert metrics.spl(True, 4.0, 8.0) == pytest.approx(0.5)


def test_a_failed_episode_scores_zero_however_efficiently_it_drove():
    assert metrics.spl(False, 4.0, 4.0) == 0.0


def test_spl_is_clamped_at_one_when_the_agent_beats_the_grid_planner():
    """A* zigzags between cell centres; a smooth drive is genuinely shorter.

    Without `max(P, L)` in the denominator this returns 1.33 — a score above
    the metric's own ceiling, which would quietly inflate any mean it entered.
    """
    assert metrics.spl(True, 4.0, 3.0) == pytest.approx(1.0)


def test_spl_refuses_a_task_with_no_geodesic_length():
    """Dividing by it would make every episode on that task look perfect."""
    with pytest.raises(ValueError):
        metrics.spl(True, 0.0, 5.0)


# --- collisions --------------------------------------------------------------

def test_a_run_of_contact_ticks_is_one_collision_event():
    """Scraping a wall for four ticks is one mistake, not four."""
    assert metrics.count_collision_events([0, 1, 1, 1, 0]) == 1


def test_separate_contacts_are_separate_events():
    assert metrics.count_collision_events([1, 0, 1, 0, 1]) == 3


def test_contact_from_the_first_tick_counts_as_an_event():
    assert metrics.count_collision_events([1, 1, 0]) == 1


def test_no_contact_is_no_events():
    assert metrics.count_collision_events([0, 0, 0]) == 0


def test_the_collision_rate_counts_events_per_metre_not_contact_ticks():
    """A robot wedged against a chair for 300 ticks is one mistake. Dividing
    its contact ticks by the distance it managed would report 60 "collisions
    per metre", which describes being stuck rather than hitting things."""
    scored = make_metrics(collision_ticks=300, collision_events=1,
                          ticks=341, path_length_m=5.0)
    assert scored.collision_events_per_m == pytest.approx(0.2)
    assert scored.contact_tick_fraction == pytest.approx(300 / 341)


def test_an_episode_that_went_nowhere_has_a_rate_of_zero_not_infinity():
    scored = make_metrics(collision_ticks=3, collision_events=1, ticks=0,
                          path_length_m=0.0)
    assert scored.contact_tick_fraction == 0.0
    assert scored.collision_events_per_m == 0.0


# --- the row -----------------------------------------------------------------

def test_the_row_fills_every_declared_column_and_invents_none():
    """DictWriter drops unknown keys silently and errors on missing ones, so a
    metric added to the row without a column would vanish from the table."""
    assert set(make_metrics().as_row()) == set(metrics.CSV_COLUMNS)


def test_a_missing_geodesic_distance_is_blank_rather_than_a_number():
    """An agent off the traversable component has no geodesic to the goal.
    Writing 0 there would read as "arrived"."""
    row = make_metrics(final_geodesic_distance_m=None).as_row()
    assert row["final_geodesic_distance_m"] == ""


def test_outcome_names_the_only_two_ways_an_episode_can_end():
    assert make_metrics(success=True).outcome == metrics.OUTCOME_SUCCESS
    assert make_metrics(success=False).outcome == metrics.OUTCOME_TIMEOUT


# --- the table ---------------------------------------------------------------

def test_the_table_writes_a_header_and_one_row_per_episode(tmp_path):
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics(), trace={"task_id": "Rs_00"})
    table.append(make_metrics(success=False))

    rows = metrics.read_table(table.csv_path)
    assert [row["outcome"] for row in rows] == ["success", "timeout"]
    assert rows[0]["checkpoint"] == "best_combined"
    assert table.trace_path.read_text().count("\n") == 1


def test_reopening_a_table_to_resume_keeps_the_rows_already_scored(tmp_path):
    """Nineteen slow episodes must survive a crash on the twentieth."""
    path = tmp_path / "scores.csv"
    metrics.MetricsTable(path).open().append(make_metrics())
    metrics.MetricsTable(path).open(resume=True).append(make_metrics())
    assert len(metrics.read_table(path)) == 2


def test_a_resumed_table_knows_which_episodes_are_already_scored(tmp_path):
    path = tmp_path / "scores.csv"
    metrics.MetricsTable(path).open().append(make_metrics())
    table = metrics.MetricsTable(path).open(resume=True)
    assert table.completed() == {("best_combined", "Rs_00", 1000)}
    assert table.rows == 1


def test_the_same_task_under_another_seed_is_not_already_scored(tmp_path):
    path = tmp_path / "scores.csv"
    metrics.MetricsTable(path).open().append(make_metrics(seed=1000))
    assert ("best_combined", "Rs_00", 1001) not in \
        metrics.MetricsTable(path).open(resume=True).completed()


def test_resuming_drops_a_row_cut_off_mid_write(tmp_path):
    """A kill mid-row can leave every field present and the last one wrong,
    so the test for a finished row is its newline, not its field count."""
    path = tmp_path / "scores.csv"
    table = metrics.MetricsTable(path).open()
    table.append(make_metrics(), trace={"checkpoint": "best_combined",
                                        "task_id": "Rs_00", "seed": 1000})
    whole = path.read_text()
    table.append(make_metrics(task=FakeTask(task_id="Rs_01")))
    path.write_text(path.read_text()[:-3])

    resumed = metrics.MetricsTable(path).open(resume=True)
    assert path.read_text() == whole
    assert resumed.completed() == {("best_combined", "Rs_00", 1000)}


def test_resuming_drops_a_trace_whose_row_was_never_written(tmp_path):
    """The trace is written first and the row last, so a crash between them
    leaves an orphan trace — the episode is not done, and a retry would
    otherwise leave two traces for it."""
    path = tmp_path / "scores.csv"
    table = metrics.MetricsTable(path).open()
    table.append(make_metrics(), trace={"checkpoint": "best_combined",
                                        "task_id": "Rs_00", "seed": 1000})
    with open(table.trace_path, "a") as handle:
        handle.write('{"checkpoint": "best_combined", "task_id": "Rs_01", "seed": 1001}\n')
        handle.write('{"checkpoint": "best_comb')

    metrics.MetricsTable(path).open(resume=True)
    lines = table.trace_path.read_text().splitlines()
    assert [json.loads(line)["task_id"] for line in lines] == ["Rs_00"]


def test_a_table_with_other_columns_is_refused_rather_than_resumed(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("checkpoint,task_id,seed\nbest_combined,Rs_00,1000\n")
    with pytest.raises(ValueError, match="different columns"):
        metrics.MetricsTable(path).open(resume=True)


def test_reopening_without_resume_starts_the_table_over(tmp_path):
    path = tmp_path / "scores.csv"
    metrics.MetricsTable(path).open().append(make_metrics())
    metrics.MetricsTable(path).open()
    assert metrics.read_table(path) == []


# --- the roll-up -------------------------------------------------------------

def test_every_statistic_is_a_mean_over_episodes_not_a_ratio_of_totals(tmp_path):
    """Each task counts once. Pooled, the 300-tick episode would outweigh the
    100-tick one three to one (contact 300/400 = 0.75) and the long drive would
    dilute the collision (1 event / 10 m = 0.1). Per episode, it is 0.5 and
    0.25: the two definitions really do give different numbers, which is why
    only one of them may carry the name."""
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics(collision_ticks=300, collision_events=1,
                              ticks=300, path_length_m=2.0))
    table.append(make_metrics(collision_ticks=0, collision_events=0,
                              ticks=100, path_length_m=8.0))

    summary = metrics.summarize(metrics.read_table(table.csv_path))
    assert summary["contact_tick_fraction"][0] == pytest.approx(0.5)
    assert summary["collision_events_per_m"][0] == pytest.approx(0.25)
    assert summary["collision_events"][0] == pytest.approx(0.5)


def test_success_rate_and_spl_average_over_every_episode(tmp_path):
    """A failure is a zero in the mean, not a row left out of it."""
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics(success=True, path_length_m=4.0))
    table.append(make_metrics(success=False, path_length_m=9.0))

    summary = metrics.summarize(metrics.read_table(table.csv_path))
    assert summary["success_rate"][0] == pytest.approx(0.5)
    assert summary["spl"][0] == pytest.approx(0.5)


def test_time_to_goal_averages_the_successes_only(tmp_path):
    """Plan §6 defines it for episodes that reached the goal; a mean including
    timeouts would describe the timeout formula, not the agent."""
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics(success=True, ticks=50))
    table.append(make_metrics(success=False, ticks=352))

    summary = metrics.summarize(metrics.read_table(table.csv_path))
    assert summary["ticks_to_goal"] == (50.0, None, 1)
    assert summary["ticks"][0] == pytest.approx(201.0)


def test_time_to_goal_is_absent_rather_than_zero_when_nothing_succeeded(tmp_path):
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics(success=False))
    summary = metrics.summarize(metrics.read_table(table.csv_path))

    assert summary["ticks_to_goal"] == (None, None, 0)
    assert "n/a (no successes)" in metrics.format_summary("best_combined", summary)


def test_an_episode_with_no_geodesic_distance_does_not_blank_the_column(tmp_path):
    """One agent stranded off the nav mesh must not turn the mean distance of
    a whole run into NaN: it drops out of that one statistic's n."""
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics(final_geodesic_distance_m=None))
    table.append(make_metrics(final_geodesic_distance_m=2.0))

    summary = metrics.summarize(metrics.read_table(table.csv_path))
    assert summary["final_geodesic_distance_m"] == (2.0, None, 1)
    assert summary["success_rate"][2] == 2


def test_a_run_where_nothing_reached_the_nav_mesh_reports_no_distance(tmp_path):
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics(final_geodesic_distance_m=None))
    summary = metrics.summarize(metrics.read_table(table.csv_path))

    assert summary["final_geodesic_distance_m"] == (None, None, 0)
    assert "n/a (none on the nav mesh)" in metrics.format_summary(
        "best_combined", summary)


def test_a_run_with_no_episodes_says_so():
    assert metrics.format_summary("clean_stock", metrics.summarize([])) == (
        "clean_stock: no episodes")


def test_the_block_a_run_ends_with_prints_the_comparison_tables_means(tmp_path):
    """The end-of-run block and the comparison table read one `summarize`, so
    a statistic cannot mean one number in an arm's log and another in the
    table beside it: the defect this module used to have."""
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics(collision_ticks=300, collision_events=1,
                              ticks=300, path_length_m=2.0))
    table.append(make_metrics(collision_ticks=0, collision_events=0,
                              ticks=100, path_length_m=8.0))

    block = metrics.format_summary(
        "best_combined", metrics.summarize(metrics.read_table(table.csv_path)))
    assert "over 2 episodes" in block
    assert "50% of an episode's ticks" in block      # not the pooled 75%
    assert "0.250 per metre" in block                # not the pooled 0.100


# --- mean ± SE (P6) ----------------------------------------------------------

def test_the_standard_error_is_the_sample_sd_over_root_n():
    mean, se, count = metrics.mean_se([1, 0, 1, 0])
    assert (mean, count) == (0.5, 4)
    assert se == pytest.approx(0.57735 / 2, abs=1e-5)


def test_one_value_has_a_mean_but_no_standard_error():
    assert metrics.mean_se([0.7]) == (0.7, None, 1)


def test_no_values_have_neither():
    assert metrics.mean_se([]) == (None, None, 0)


def test_the_summary_reports_every_statistic_and_nothing_else(tmp_path):
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    table.append(make_metrics())
    summary = metrics.summarize(metrics.read_table(table.csv_path))

    assert set(summary) == {name for name, _, _ in metrics.EPISODE_STATISTICS}
