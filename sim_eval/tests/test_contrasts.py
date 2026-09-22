"""Pin the contrasts layer: which tables it reads, and what each row holds.

The arithmetic is `task_stats`'s and pinned there. What is pinned here is the
wiring around it — that a contrast naming a table the run does not have fails
before anything is computed, that the additivity table's columns are the
contrasts they claim to be, and that McNemar appears only where it is defined.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comparison  # noqa: E402
import p7_contrasts as contrasts  # noqa: E402
from test_task_stats import row  # noqa: E402


def factorial(values, seeds=(0,)):
    """{arm: rows}, each task's spl = the arm's value + the task index / 100."""
    return {name: [row(name, "Rs_{:02d}".format(task), 1000 + task + offset, 1,
                       spl=value + task / 100)
                   for task in range(4) for offset in seeds]
            for name, value in values.items()}


SPEC = {"baseline": "ctx03", "parts": ["stride3", "img160x120"], "combined": "bc30"}


def test_the_additivity_columns_are_the_contrasts_they_name():
    tables = factorial({"ctx03": 0.2, "stride3": 0.3, "img160x120": 0.25, "bc30": 0.5})
    [spl] = [r for r in contrasts.additivity_rows(SPEC, tables) if r["statistic"] == "spl"]
    assert spl["first"] == pytest.approx(0.1)
    assert spl["second"] == pytest.approx(0.05)
    assert spl["sum"] == pytest.approx(0.15)
    assert spl["both"] == pytest.approx(0.3)
    assert spl["interaction"] == pytest.approx(0.15)
    assert spl["n_tasks"] == 4


def test_mcnemar_is_reported_for_one_seed_and_withheld_for_several():
    one = factorial({"a": 0.5, "b": 0.4})
    [success] = [r for r in contrasts.pair_rows("a", "b", one, []) if
                 r["statistic"] == "success_rate"]
    assert success["mcnemar_p"] != ""

    three = factorial({"a": 0.5, "b": 0.4}, seeds=(0, 100000, 200000))
    [success] = [r for r in contrasts.pair_rows("a", "b", three, []) if
                 r["statistic"] == "success_rate"]
    assert success["mcnemar_p"] == ""
    assert success["n_tasks"] == 4


def test_a_contrast_naming_an_unknown_table_is_refused():
    config = contrasts.ContrastsConfig({"a": "a.csv"}, pairs=[["a", "ghost"]])
    with pytest.raises(ValueError, match="ghost"):
        config.check_names()


def test_additivity_needs_exactly_two_parts():
    config = contrasts.ContrastsConfig(
        dict.fromkeys(["ctx03", "stride3", "bc30"], "x.csv"), pairs=[],
        additivity={"baseline": "ctx03", "parts": ["stride3"], "combined": "bc30"})
    with pytest.raises(ValueError, match="two"):
        config.check_names()


def test_restricting_keeps_only_the_shared_episodes():
    rows = [row("best_combined", task, seed, 1) for task, seed in
            (("Rs_00", 1000), ("Rs_00", 101000), ("Rs_10", 1010), ("Denmark_00", 3000))]
    kept = comparison.restrict(rows, {("Rs_00", 1000), ("Rs_00", 101000)})
    assert [(r["task_id"], int(r["seed"])) for r in kept] == [
        ("Rs_00", 1000), ("Rs_00", 101000)]


def test_restricting_is_off_unless_asked_for():
    assert not contrasts.ContrastsConfig({}, pairs=[]).restrict_to_task_set
