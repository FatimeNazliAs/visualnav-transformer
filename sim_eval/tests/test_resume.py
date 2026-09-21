"""Pin what makes a long run survivable: skip what is done, isolate what crashes.

P6 is forty slow rollouts on a shared machine. Two things must hold whatever
goes wrong part-way, and both are silent when they break:

  * **a resumed run skips exactly the episodes already in the table** — keyed
    on (checkpoint, task, seed), so a finished episode is never scored twice
    and an unfinished one is never skipped.
  * **one crashed rollout costs one episode, not the run.** The rest of the
    scene still scores, the crashed task has no row (so a resume retries it),
    and the crash is reported rather than swallowed.

Runs `run_eval.run_scene` against fakes — no GPU, no simulator, no checkpoint.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

import metrics  # noqa: E402
import recorder  # noqa: E402
import run_eval  # noqa: E402


class FakeTask:
    def __init__(self, index):
        self.task_id = "Rs_{:02d}".format(index)
        self.scene = "Rs"
        self.seed = 1000 + index
        self.geodesic_length_m = 4.0
        self.node_count = 10

    def summary(self):
        return self.task_id


class FakeResult:
    def __init__(self, episode_metrics):
        self.metrics = episode_metrics

    def trace(self):
        return self.metrics.as_row()


class FakeEpisode:
    """No ticks; scores a success — or raises, for a task marked to crash."""

    def __init__(self, task, checkpoint, seed, crash):
        self.task, self.checkpoint, self.seed, self.crash = task, checkpoint, seed, crash

    def __iter__(self):
        if self.crash:
            raise RuntimeError("simulated rollout crash on " + self.task.task_id)
        return iter(())

    def result(self):
        return FakeResult(metrics.EpisodeMetrics(
            checkpoint=self.checkpoint, task=self.task, seed=self.seed,
            driver="n8w2r4t3", success=True, collision_ticks=0,
            collision_events=0, path_length_m=4.0,
            final_geodesic_distance_m=0.3, final_euclidean_distance_m=0.3,
            ticks=40, seconds=10.0, timeout_ticks=200, success_radius_m=1.0,
            success_metric="geodesic", declared_arrival_tick=None, final_node=9))


class FakeRunner:
    def __init__(self, crash_on=()):
        self.crash_on = set(crash_on)
        self.ran = []
        self.body = type("Body", (), {"scene": None})()

    def seed_for(self, task):
        return task.seed

    def episode(self, task, checkpoint_name):
        self.ran.append(task.task_id)
        return FakeEpisode(task, checkpoint_name, task.seed,
                           crash=task.task_id in self.crash_on)


def run(table, runner, count=4):
    tasks = [FakeTask(index) for index in range(count)]
    return run_eval.run_scene("Rs", tasks, runner, "best_combined", table,
                              recorder.disabled(), verbose=False)


def scored_ids(table):
    return [row["task_id"] for row in metrics.read_table(table.csv_path)]


def test_a_crashed_rollout_costs_that_episode_and_nothing_else(tmp_path):
    table = metrics.MetricsTable(tmp_path / "scores.csv").open()
    runner = FakeRunner(crash_on={"Rs_01"})

    crashed = run(table, runner)

    assert crashed == ["Rs_01"]
    assert runner.ran == ["Rs_00", "Rs_01", "Rs_02", "Rs_03"]
    assert scored_ids(table) == ["Rs_00", "Rs_02", "Rs_03"]


def test_a_resumed_run_retries_only_what_is_not_in_the_table(tmp_path):
    path = tmp_path / "scores.csv"
    run(metrics.MetricsTable(path).open(), FakeRunner(crash_on={"Rs_01"}))

    retry = FakeRunner()
    crashed = run(metrics.MetricsTable(path).open(resume=True), retry)

    assert crashed == []
    assert retry.ran == ["Rs_01"]
    assert sorted(scored_ids(metrics.MetricsTable(path))) == \
        ["Rs_00", "Rs_01", "Rs_02", "Rs_03"]


def test_a_complete_table_reruns_nothing(tmp_path):
    path = tmp_path / "scores.csv"
    run(metrics.MetricsTable(path).open(), FakeRunner())
    again = FakeRunner()
    run(metrics.MetricsTable(path).open(resume=True), again)
    assert again.ran == []


def test_the_crash_is_reported_by_name_rather_than_swallowed():
    error = run_eval.EpisodeCrashError(["Rs_01", "Rs_07"])
    assert "Rs_01, Rs_07" in str(error) and "--resume" in str(error)


def test_an_interrupt_is_not_mistaken_for_a_crashed_rollout(tmp_path):
    """Ctrl-C must stop the run, not be logged as one bad episode and skipped."""

    class Interrupting(FakeRunner):
        def episode(self, task, checkpoint_name):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(metrics.MetricsTable(tmp_path / "scores.csv").open(), Interrupting())
