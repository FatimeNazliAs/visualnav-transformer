"""Pin the config that decides everything a scoring run does.

`EvalConfig` composes the other four config objects — the task set, the episode
rules, the driver and the recorder — and every command in this workstream goes
through it. It was also, until P4's architecture pass, the only one of the five
with no tests at all: `TaskSetConfig`, `TopomapConfig`, `EpisodeRules` and
`RecordingConfig` had 86 between them, and the thing that assembles them had
none.

What is worth pinning is not that it parses YAML — it is the three behaviours
that are silent when they break:

  * **a typo must fail at load, not be ignored.** A misspelled episode rule that
    is quietly dropped means a whole run scored under a rule nobody chose, and
    the table looks perfectly normal.
  * **paths resolve against `sim_eval/`, not the working directory.** The
    scripts are run from the repo root and the configs name `outputs/...`;
    resolving against the cwd would scatter results wherever someone happened
    to stand.
  * **a missing section still produces a usable default** — and, for the
    recorder, a default that is *off*. A config with no `recording:` block that
    somehow recorded would cost an hour and a gigabyte on a 20-task run.

Runs against YAML strings — no GPU, no simulator, no checkpoint.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_eval  # noqa: E402

SIM_EVAL_DIR = Path(__file__).resolve().parents[1]

MINIMAL = """
task_set:
  topomap_config: configs/topomap.yaml
  scenes: [Rs]
  tasks_per_scene: 3
  base_seed: 1000
  directory: outputs/test_task_set
"""


def load(text):
    return run_eval.EvalConfig.from_dict(yaml.safe_load(text))


# --- the defaults a missing section leaves behind ----------------------------

def test_a_config_with_only_a_task_set_is_usable():
    config = load(MINIMAL)
    assert config.tasks.scenes == ["Rs"]
    assert config.tasks.tasks_per_scene == 3
    assert config.floor == 0
    assert config.checkpoint_names == []


def test_recording_defaults_to_off():
    """The expensive default must be the safe one."""
    config = load(MINIMAL)
    assert config.recording.enabled is False
    assert config.recording.summary() == "off"


def test_the_driver_defaults_to_the_real_robots_settings():
    config = load(MINIMAL)
    assert config.driver.is_deployment_default()
    assert config.driver.label() == "n8w2r4t3"


def test_episode_rules_default_to_the_plans_numbers():
    rules = load(MINIMAL).rules
    assert rules.success_radius_m == 1.0
    assert rules.success_metric == "geodesic"


# --- a typo fails at load ----------------------------------------------------

def test_an_unknown_episode_rule_is_refused_rather_than_ignored():
    with pytest.raises(TypeError):
        load(MINIMAL + """
episode:
  success_radius_metres: 1.0
""")


def test_an_unknown_driver_knob_is_refused():
    with pytest.raises(TypeError):
        load(MINIMAL + """
driver:
  waypoints: 2
""")


def test_an_unknown_recording_knob_is_refused():
    with pytest.raises(TypeError):
        load(MINIMAL + """
recording:
  frames_per_second: 10
""")


def test_a_bad_recording_subset_is_refused_with_a_message():
    with pytest.raises(ValueError, match="task ids"):
        load(MINIMAL + """
recording:
  tasks: every
""")


def test_a_bad_success_metric_is_refused_with_a_message():
    with pytest.raises(ValueError, match="geodesic"):
        load(MINIMAL + """
episode:
  success_metric: euclidian
""")


# --- paths resolve against sim_eval/, wherever the script was run from -------

def test_relative_paths_resolve_against_the_package_not_the_cwd():
    config = load(MINIMAL)
    assert config.task_directory == SIM_EVAL_DIR / "outputs/test_task_set"
    assert config.output_dir == SIM_EVAL_DIR / "outputs"
    assert config.recording.directory == SIM_EVAL_DIR / "outputs/videos"


def test_an_absolute_path_is_left_alone():
    config = load("""
task_set:
  topomap_config: configs/topomap.yaml
  scenes: [Rs]
  directory: /mnt/shared_disk/somewhere
output_dir: /mnt/shared_disk/results
""")
    assert config.task_directory == Path("/mnt/shared_disk/somewhere")
    assert config.output_dir == Path("/mnt/shared_disk/results")


def test_the_floor_is_read_out_of_the_episode_section_not_passed_to_the_rules():
    """`floor` sits under `episode:` for the reader's sake but belongs to the
    scene, so it must be removed before the rest becomes the rules."""
    config = load(MINIMAL + """
episode:
  floor: 2
  success_radius_m: 1.5
""")
    assert config.floor == 2
    assert config.rules.success_radius_m == 1.5


# --- the shipped config is one of the things worth pinning -------------------

def test_the_checked_in_config_loads_and_still_mirrors_the_real_robot():
    """configs/eval.yaml is what P6 will run. A driver knob edited and left
    behind in it would silently become the headline comparison's settings."""
    config = run_eval.EvalConfig.from_yaml()
    assert config.driver.is_deployment_default(), config.driver.summary()
    assert config.recording.enabled is False
    assert config.checkpoint_names == ["best_combined", "clean_stock"]


def test_one_csv_per_checkpoint_under_the_output_dir():
    config = load(MINIMAL + "\noutput_dir: outputs/p3_2_metrics\n")
    assert config.csv_path("clean_stock") == \
        SIM_EVAL_DIR / "outputs/p3_2_metrics/clean_stock.csv"



# --- layering: a run says only what is different about it (P6) --------------

def test_a_config_that_extends_another_overrides_only_what_it_names(tmp_path):
    (tmp_path / "base.yaml").write_text(MINIMAL + """
episode:
  success_radius_m: 1.5
output_dir: outputs/base
""")
    (tmp_path / "run.yaml").write_text("""
extends: base.yaml
task_set:
  scenes: [Rs, Other]
  directory: outputs/run_task_set
output_dir: outputs/run
""")
    config = run_eval.EvalConfig.from_yaml(tmp_path / "run.yaml")
    assert config.tasks.scenes == ["Rs", "Other"]
    assert config.tasks.tasks_per_scene == 3
    assert config.task_directory == SIM_EVAL_DIR / "outputs/run_task_set"
    assert config.rules.success_radius_m == 1.5
    assert config.output_dir == SIM_EVAL_DIR / "outputs/run"


def test_a_list_is_replaced_whole_not_merged():
    merged = run_eval.deep_merge({"a": {"b": [1, 2], "c": 1}}, {"a": {"b": [3]}})
    assert merged == {"a": {"b": [3], "c": 1}}


def test_the_p6_config_shares_the_rules_and_driver_of_eval_yaml():
    """P6 faces both arms with the deployment settings — inherited, not copied."""
    base = run_eval.EvalConfig.from_yaml()
    headline = run_eval.EvalConfig.from_yaml(
        SIM_EVAL_DIR / "configs" / "p6_headline.yaml")
    assert headline.driver.is_deployment_default()
    assert headline.driver.label() == base.driver.label()
    assert headline.rules.as_dict() == base.rules.as_dict()
    assert headline.tasks.tasks_per_scene == 10 and len(headline.tasks.scenes) == 2
    assert headline.recording.enabled and headline.recording.tasks == "all"
    assert headline.checkpoint_names == ["best_combined", "clean_stock"]



# --- seeds per task (P7) -----------------------------------------------------

def test_one_run_per_task_under_its_own_seed_is_the_default():
    assert load(MINIMAL).seed_offsets == [0]


def test_seed_offsets_are_read_as_a_list():
    config = load(MINIMAL + "seed_offsets: [0, 100000, 200000]\n")
    assert config.seed_offsets == [0, 100000, 200000]


def test_an_empty_seed_list_is_refused():
    with pytest.raises(ValueError):
        load(MINIMAL + "seed_offsets: []\n")


def test_a_repeated_seed_offset_is_refused():
    """It would score the same episode twice."""
    with pytest.raises(ValueError):
        load(MINIMAL + "seed_offsets: [0, 100000, 0]\n")


def test_the_checked_in_configs_still_run_one_seed_per_task():
    """P3-P6 were one seed per task; the new knob must not change them."""
    for name in ("eval.yaml", "p6_headline.yaml"):
        config = run_eval.EvalConfig.from_yaml(SIM_EVAL_DIR / "configs" / name)
        assert config.seed_offsets == [0], name
