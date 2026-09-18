"""Pin the fairness protocol: the same tasks, for every arm, every time.

Plan §7 is one sentence — *identical task set across every checkpoint* — and it
is the kind of claim that is easy to make and easy to break silently. Two ways
it breaks: a task's identity quietly depending on how many tasks were asked
for, and a task set being rebuilt under a comparison that is already half done.
Both produce a full, plausible table of numbers that cannot be compared. Both
are pinned here.

No simulator: the builder is handed a fake body and P2's drive is stubbed, so
what is under test is which tasks exist and what is remembered about them.

    ./sim_eval/run_tests.sh
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import task_set  # noqa: E402
import topomap_builder  # noqa: E402
from task_set import TaskSetConfig, TaskSetError  # noqa: E402

SIM_EVAL_DIR = Path(__file__).resolve().parents[1]


def make_config(tmp_path, **overrides):
    """A task set config over a real (small) copy of P2's topomap config."""
    knobs = yaml.safe_load((SIM_EVAL_DIR / "configs" / "topomap.yaml").read_text())
    knobs.update(overrides.pop("topomap", {}))
    path = tmp_path / "topomap.yaml"
    path.write_text(yaml.safe_dump(knobs))

    values = {"topomap_config": path, "scenes": ["Rs"], "tasks_per_scene": 3,
              "base_seed": 1000}
    values.update(overrides)
    return TaskSetConfig.from_dict(values)


class FakeBody:
    """Records that it was opened and closed, and nothing else."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def stub_build(monkeypatch, bodies):
    """Replace P2's drive with a note of what it was asked to build.

    Driving a reference path needs a scene, a robot and a GPU, and P2's own
    tests already pin what comes out of it. What this file is about is which
    tasks get built, in which world, under which seed.
    """
    built = []

    def open_body(config, directory):
        body = FakeBody()
        bodies.append((body, config.scene_id, Path(directory)))
        return body

    def build_topomap(body, config, directory, **kwargs):
        built.append({"scene": config.scene_id, "seed": config.seed,
                      "directory": Path(directory)})
        return {"geodesic_length_m": 4.0 + 0.1 * config.seed % 1,
                "nodes": [{"node": index} for index in range(5)]}

    monkeypatch.setattr(topomap_builder, "build_topomap", build_topomap)
    monkeypatch.setattr(topomap_builder, "write_world_config",
                        lambda world, directory: Path(directory) / "world.yaml")
    return built, open_body


# --- seeds -------------------------------------------------------------------

def test_a_task_seed_depends_on_its_scene_and_index_only():
    assert task_set.task_seed(1000, 0, 0) == 1000
    assert task_set.task_seed(1000, 0, 3) == 1003
    assert task_set.task_seed(1000, 1, 3) == 2003


def test_a_small_task_set_is_the_prefix_of_a_large_one(tmp_path):
    """This is what lets the three-task score test be a real rehearsal for the
    ten-task run: task 2 is the same problem in both."""
    small = make_config(tmp_path, tasks_per_scene=3)
    large = make_config(tmp_path, tasks_per_scene=10)
    seeds = lambda config: [config.topomap_config_for("Rs", 0, index).seed  # noqa: E731
                            for index in range(config.tasks_per_scene)]
    assert seeds(small) == seeds(large)[:3]


def test_every_scene_gets_its_own_block_of_seeds(tmp_path):
    config = make_config(tmp_path, scenes=["Rs", "Bolton"], tasks_per_scene=3)
    first = {config.topomap_config_for("Rs", 0, index).seed for index in range(3)}
    second = {config.topomap_config_for("Bolton", 1, index).seed for index in range(3)}
    assert not (first & second)


def test_the_scene_reaches_the_topomap_config(tmp_path):
    config = make_config(tmp_path, scenes=["Bolton"])
    assert config.topomap_config_for("Bolton", 0, 0).scene_id == "Bolton"


# --- what a task set may be --------------------------------------------------

def test_a_pinned_start_goal_pair_is_refused(tmp_path):
    """Every task in the set would be the same trail, driven ten times."""
    config = make_config(tmp_path, topomap={"start": [0.0, 0.0], "goal": [1.0, 1.0]})
    with pytest.raises(TaskSetError):
        config.base_knobs()


def test_a_task_set_needs_a_scene_and_a_task(tmp_path):
    with pytest.raises(TaskSetError):
        make_config(tmp_path, scenes=[])
    with pytest.raises(TaskSetError):
        make_config(tmp_path, tasks_per_scene=0)


def test_overrides_amend_p2s_knobs_rather_than_replacing_them(tmp_path):
    config = make_config(tmp_path, overrides={"spacing_ticks": 8})
    knobs = config.base_knobs()
    assert knobs["spacing_ticks"] == 8
    assert knobs["max_ticks"] == 400          # still P2's


# --- the fingerprint ---------------------------------------------------------

def test_the_fingerprint_changes_with_anything_that_changes_the_tasks(tmp_path):
    base = make_config(tmp_path).fingerprint()
    assert make_config(tmp_path, base_seed=2000).fingerprint() != base
    assert make_config(tmp_path, scenes=["Bolton"]).fingerprint() != base
    assert make_config(tmp_path, tasks_per_scene=5).fingerprint() != base
    # The trail knobs are half of what a task is: the same start and goal at a
    # different node spacing is a different problem.
    assert make_config(tmp_path, topomap={"spacing_ticks": 8}).fingerprint() != base


def test_the_fingerprint_is_stable_across_identical_configs(tmp_path):
    assert make_config(tmp_path).fingerprint() == make_config(tmp_path).fingerprint()


def write_world(tmp_path, **changes):
    """A copy of the bridge's world config, with some of it changed."""
    world = yaml.safe_load(
        (SIM_EVAL_DIR / "configs" / "locobot_rs_bridge.yaml").read_text())
    world.update(changes)
    path = tmp_path / "world_{}.yaml".format(len(list(tmp_path.glob("world_*"))))
    path.write_text(yaml.safe_dump(world))
    return path


def test_the_fingerprint_changes_when_the_camera_does(tmp_path):
    """Every trail image is rendered through the world's camera and every
    episode reopens that world. The fingerprint used to hash the world file's
    *path*, so a camera edit matched the old set and was silently ignored."""
    same_path = tmp_path / "world.yaml"
    same_path.write_text(write_world(tmp_path).read_text())
    before = make_config(tmp_path, topomap={"scene_config": str(same_path)})
    fingerprint = before.fingerprint()

    edited = yaml.safe_load(same_path.read_text())
    edited["vertical_fov"] = 90
    same_path.write_text(yaml.safe_dump(edited))

    assert before.fingerprint() != fingerprint


def test_the_fingerprint_reads_what_is_in_the_world_file(tmp_path):
    """Any knob of the world moves it, not only the camera — the robot, the
    render size and the physics all decide what a trail image looks like."""
    # Each config gets its own directory: `make_config` writes topomap.yaml
    # there, and the knobs are read lazily, so a shared file would be read back
    # as whichever config wrote it last.
    def config_over(name, **changes):
        directory = tmp_path / name
        directory.mkdir()
        world = write_world(directory, **changes)
        return make_config(directory, topomap={"scene_config": str(world)})

    first, second = config_over("first"), config_over("second")
    assert first.world() == second.world()
    changed = config_over("changed", image_width=320)
    assert changed.world() != first.world()
    assert changed.fingerprint() != first.fingerprint()


# --- building and reusing ----------------------------------------------------

def test_building_writes_a_manifest_and_one_directory_per_task(tmp_path, monkeypatch):
    bodies = []
    built, open_body = stub_build(monkeypatch, bodies)
    config = make_config(tmp_path, tasks_per_scene=3)

    manifest, tasks = task_set.build(config, tmp_path / "set", open_body)

    assert [entry["seed"] for entry in built] == [1000, 1001, 1002]
    assert [task.task_id for task in tasks] == ["Rs_00", "Rs_01", "Rs_02"]
    assert manifest["fingerprint"] == config.fingerprint()
    assert json.loads((tmp_path / "set" / "manifest.json").read_text())["tasks"]


def test_one_simulator_is_opened_per_scene_not_per_task(tmp_path, monkeypatch):
    """Opening iGibson costs tens of seconds, and a scene's tasks share a world."""
    bodies = []
    _built, open_body = stub_build(monkeypatch, bodies)
    config = make_config(tmp_path, scenes=["Rs", "Bolton"], tasks_per_scene=3)

    task_set.build(config, tmp_path / "set", open_body)

    assert [scene for _body, scene, _dir in bodies] == ["Rs", "Bolton"]
    assert all(body.closed for body, _scene, _dir in bodies)


def test_an_existing_task_set_is_reused_rather_than_rebuilt(tmp_path, monkeypatch):
    bodies = []
    built, open_body = stub_build(monkeypatch, bodies)
    config = make_config(tmp_path)

    task_set.ensure(config, tmp_path / "set", open_body)
    task_set.ensure(config, tmp_path / "set", open_body)

    assert len(built) == config.tasks_per_scene


def test_a_task_set_built_from_a_different_config_is_refused(tmp_path, monkeypatch):
    """Half a table scored on one task set and half on another is not a
    comparison, and nothing in the CSV would say so."""
    bodies = []
    _built, open_body = stub_build(monkeypatch, bodies)
    task_set.ensure(make_config(tmp_path), tmp_path / "set", open_body)

    with pytest.raises(TaskSetError):
        task_set.ensure(make_config(tmp_path, base_seed=7), tmp_path / "set",
                        open_body)


def test_rebuilding_is_possible_but_has_to_be_asked_for(tmp_path, monkeypatch):
    bodies = []
    built, open_body = stub_build(monkeypatch, bodies)
    config = make_config(tmp_path)

    task_set.ensure(config, tmp_path / "set", open_body)
    task_set.ensure(config, tmp_path / "set", open_body, rebuild=True)

    assert len(built) == 2 * config.tasks_per_scene


def test_loading_a_directory_with_no_manifest_says_so(tmp_path):
    with pytest.raises(TaskSetError):
        task_set.load(tmp_path)


def test_tasks_are_grouped_by_scene_in_build_order(tmp_path, monkeypatch):
    bodies = []
    _built, open_body = stub_build(monkeypatch, bodies)
    config = make_config(tmp_path, scenes=["Rs", "Bolton"], tasks_per_scene=2)

    _manifest, tasks = task_set.build(config, tmp_path / "set", open_body)
    groups = task_set.group_by_scene(tasks)

    assert list(groups) == ["Rs", "Bolton"]
    assert [task.task_id for task in groups["Bolton"]] == ["Bolton_00", "Bolton_01"]


# --- reading a task back -----------------------------------------------------

def test_a_task_reads_its_goal_from_the_pose_the_goal_image_was_taken_at(tmp_path):
    """Not `planned_goal_xy`: the planner's target is a point on a grid the
    reference drive stopped near, and the model is chasing the picture."""
    task = task_set.Task("Rs_00", "Rs", tmp_path, seed=1000, metadata={
        "goal_pose": {"x": 1.5, "y": -2.0, "yaw": 0.3},
        "planned_goal_xy": [9.9, 9.9],
        "start_pose": {"x": 0.0, "y": 0.0, "yaw": 1.0},
        "geodesic_length_m": 4.2,
        "nodes": [{"node": 0}, {"node": 1}],
    })
    assert task.goal_xy == [1.5, -2.0]
    assert task.geodesic_length_m == 4.2
    assert task.node_count == 2
    position, orientation = task.start_pose(floor_height=0.7)
    assert position == [0.0, 0.0, 0.7]
    assert orientation == [0.0, 0.0, 1.0]
