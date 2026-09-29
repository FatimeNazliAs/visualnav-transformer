"""Pin the single-goal (language-goal, Phase 7) runner, run folder and film gate.

  * **F1 — how close it got.** The oracle ends a success at the radius, so a
    failure's closest approach is only in the trace: every tick logs its
    distance to the word, and the trace carries the minimum and its tick.
  * **F2/F3 — a table belongs to one run.** An arm's table is never silently
    truncated, a second arm must face the run's task set, seeds, driver and
    rules, and a resumed arm must be the same checkpoint and goal feeding.
  * **F4** — one `--output` table cannot take several arms.
  * **F8 — a film that is not the scored episode never passes for it.** The
    whole row must match, and a mismatched film is renamed, not stacked.

Fakes from tests/test_episode_runner.py drive the real GoalBridge: a body that
moves 0.05 m east per tick. No GPU, no iGibson.

    ./sim_eval/run_tests.sh
"""

import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoints  # noqa: E402
import episode_runner  # noqa: E402
import metrics  # noqa: E402
import object_tasks  # noqa: E402
import objects  # noqa: E402
import run_eval  # noqa: E402
from nomad_policy import PolicyStep  # noqa: E402
from test_episode_runner import FakeBody, FakePolicy, STRAIGHT_AHEAD  # noqa: E402

RES = 0.1
SIZE = 120  # 12 m square, world -6..6
CHAIR_X = 5.0  # the chair's west face; the robot starts at x = 0 facing it


def chair_house():
    trav = np.full((SIZE, SIZE), objects.FREE, dtype=np.uint8)
    grid = objects.GridMap(trav, RES)
    chair = objects.Instance("chair_0", "chair", [[CHAIR_X, CHAIR_X + 0.4, -0.2, 0.2]],
                             view_from=[CHAIR_X - 1.0, 0.0])
    for row in range(SIZE):
        for col in range(SIZE):
            x, y = grid.centre(row, col)
            if CHAIR_X - 0.2 <= x <= CHAIR_X + 0.6 and -0.4 <= y <= 0.4:
                trav[row, col] = 0
    return objects.SceneObjects("Test", objects.GridMap(trav, RES), [chair])


def chair_task(tmp_path, geodesic_m=CHAIR_X):
    entry = {"task_id": "Test_chair_00", "scene": "Test", "word": "chair", "seed": 7000,
             "start_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}, "target_instance": "chair_0",
             "geodesic_length_m": geodesic_m, "photo": "photo.png"}
    return object_tasks.ObjectTask(entry, tmp_path)


class GoalPolicy(FakePolicy):
    """Straight ahead for `stop_after` ticks (all of them if None), then stands."""

    def __init__(self, stop_after=None):
        super().__init__()
        self.stop_after = stop_after

    def act_goal(self, context_frames, goal):
        moving = self.stop_after is None or self.calls < self.stop_after
        self.calls += 1
        return PolicyStep(waypoint=list(STRAIGHT_AHEAD) if moving else [0.0, 0.0],
                          closest_node=0, subgoal_node=0, distances=[], samples=[])


def run(tmp_path, policy):
    house = chair_house()
    goal = types.SimpleNamespace(kind="word", label="chair")
    runner = episode_runner.ObjectGoalRunner(
        policy, FakeBody(), house, lambda _task: goal,
        rules=episode_runner.EpisodeRules(success_radius_m=1.0))
    episode = runner.episode(chair_task(tmp_path), "fake+word")
    for _record in episode:
        pass
    return episode.result()


# --- F1 ------------------------------------------------------------------------

def test_success_logs_its_distance_every_tick(tmp_path):
    result = run(tmp_path, GoalPolicy())
    trace = result.trace()
    assert result.metrics.success
    distances = [tick["goal_distance_m"] for tick in trace["ticks_log"]]
    assert all(d is not None for d in distances)
    assert distances == sorted(distances, reverse=True)  # driving straight at it
    assert distances[-1] <= 1.0 < distances[-2]
    assert trace["min_geodesic_distance_m"] == pytest.approx(distances[-1], abs=1e-3)
    assert trace["min_distance_tick"] == trace["ticks_log"][-1]["tick"]
    assert result.metrics.final_geodesic_distance_m == pytest.approx(
        trace["min_geodesic_distance_m"], abs=1e-3)


def test_timeout_keeps_the_closest_approach(tmp_path):
    # 30 ticks at 0.05 m, then stands 3.5 m from the chair until the timeout.
    result = run(tmp_path, GoalPolicy(stop_after=30))
    trace = result.trace()
    assert not result.metrics.success
    assert result.metrics.ticks == result.metrics.timeout_ticks
    assert trace["min_geodesic_distance_m"] == pytest.approx(CHAIR_X - 30 * 0.05, abs=0.05)
    assert trace["min_distance_tick"] <= 30
    # The CSV row is the shared one: nothing single-goal leaks into its columns.
    assert list(result.metrics.as_row()) == list(metrics.CSV_COLUMNS)


# --- F2 / F3 / F4 ----------------------------------------------------------------

def fake_config(offsets=(0, 100000)):
    return types.SimpleNamespace(
        seed_offsets=list(offsets),
        driver=types.SimpleNamespace(label=lambda: "n8w2r4t3"),
        rules=episode_runner.EpisodeRules(success_radius_m=1.0))


def fake_spec(name="clip_v2b", weights="/w/ema_29.pth"):
    params = {"goal_type": "clip", "clip_fusion": "none", "context_size": 3,
              "context_stride": 1, "image_size": [96, 96]}
    return types.SimpleNamespace(name=name, weights_path=Path(weights), model_params=params,
                                 goal_provenance={k: "train log" for k in params})


def record(tmp_path, arm="clip_v2b+word", fingerprint="abc", config=None, spec=None,
           resume=False):
    return run_eval.record_arm(config or fake_config(), arm, spec or fake_spec(),
                               {"fingerprint": fingerprint}, "/tasks", tmp_path / "{}.csv".format(arm),
                               resume)


def test_run_manifest_collects_the_arms_of_one_run(tmp_path):
    record(tmp_path, "clip_v2b+word")
    record(tmp_path, "clip_v2b+masked")
    run = json.loads((tmp_path / run_eval.RUN_MANIFEST).read_text())
    assert run["task_set"] == {"directory": "/tasks", "fingerprint": "abc"}
    assert run["seed_offsets"] == [0, 100000]
    assert set(run["arms"]) == {"clip_v2b+word", "clip_v2b+masked"}
    assert run_eval.read_run_manifest(tmp_path) == run


def test_existing_table_is_refused_not_truncated(tmp_path):
    record(tmp_path)
    table = tmp_path / "clip_v2b+word.csv"
    table.write_text("scored rows\n")
    with pytest.raises(SystemExit, match="--resume"):
        record(tmp_path)
    assert table.read_text() == "scored rows\n"
    record(tmp_path, resume=True)  # carrying it on is allowed


def test_another_run_cannot_share_the_folder(tmp_path):
    record(tmp_path)
    with pytest.raises(SystemExit, match="another run"):
        record(tmp_path, "ctx03+photo", fingerprint="other")
    with pytest.raises(SystemExit, match="another run"):
        record(tmp_path, "ctx03+photo", config=fake_config(offsets=(0,)))


def test_resume_must_be_the_same_arm(tmp_path):
    record(tmp_path)
    (tmp_path / "clip_v2b+word.csv").write_text("rows\n")
    with pytest.raises(SystemExit, match="different checkpoint"):
        record(tmp_path, spec=fake_spec(weights="/w/ema_28.pth"), resume=True)


def test_missing_run_manifest_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="run_manifest"):
        run_eval.read_run_manifest(tmp_path)


def test_one_output_table_takes_one_arm(tmp_path):
    data = {"mode": "object_goal",
            "object_tasks": {"scenes": ["Rs"], "words": ["chair"], "directory": str(tmp_path)},
            "arms": ["ctx03+photo", "clip_v2b+word"]}
    args = types.SimpleNamespace(output_dir=None, seed_offsets=None, record=None,
                                 record_fps=None, record_tasks=None, build_only=False,
                                 arm=None, output=tmp_path / "one.csv", task_set=None,
                                 resume=False, quiet=True)
    with pytest.raises(SystemExit, match="--output"):
        run_eval.main_object_goal(args, data, selected_gpu=0)


# --- F8 --------------------------------------------------------------------------

def row(**changes):
    values = {column: "1" for column in metrics.CSV_COLUMNS}
    values.update(changes)
    return values


def test_rows_match_on_every_column():
    lg7_4_film = pytest.importorskip("lg7_4_film")
    assert lg7_4_film.rows_match(row(), row())
    # A column the old three-column check never looked at.
    assert not lg7_4_film.rows_match(row(), row(collision_ticks="2"))
    assert not lg7_4_film.rows_match(row(), row(final_geodesic_distance_m="0.99"))


def test_mismatched_film_is_renamed_and_nothing_passes(tmp_path):
    lg7_4_film = pytest.importorskip("lg7_4_film")
    good, bad = tmp_path / "Rs_chair_00.mp4", tmp_path / "Rs_chair_00+100000.mp4"
    good.write_bytes(b"film")
    bad.write_bytes(b"film")
    matches = [{"match": 1, "video": str(good)}, {"match": 0, "video": str(bad)}]
    assert lg7_4_film.quarantine_mismatches(matches) is False
    assert good.exists() and not bad.exists()
    assert (tmp_path / "Rs_chair_00+100000.MISMATCH.mp4").exists()
    assert matches[1]["video"].endswith(".MISMATCH.mp4")
    assert lg7_4_film.quarantine_mismatches([{"match": 1, "video": str(good)}]) is True


# --- start spread ------------------------------------------------------------------

def test_a_words_starts_are_spread_and_seeded():
    house = chair_house()
    # Candidates every 0.5 m along a line, all facing the chair.
    candidates = [(-5.0 + 0.5 * k, 0.0, "chair_0", 5.0) for k in range(16)]
    rules = object_tasks.StartRules(2.0, 9.0, 0.0, True, 0.0, min_pairwise_start_m=1.0)

    def draw_all():
        taken, starts = set(), []
        for k in range(5):
            start, _, _, _ = object_tasks.draw_start(candidates, house, "chair", 7000 + k,
                                                     rules, taken)
            taken.add((start["x"], start["y"]))
            starts.append((start["x"], start["y"]))
        return starts

    starts = draw_all()
    assert starts == draw_all()  # a function of the seeds alone
    gaps = [np.hypot(a[0] - b[0], a[1] - b[1]) for i, a in enumerate(starts)
            for b in starts[i + 1:]]
    assert min(gaps) >= 1.0
    # 16 points 0.5 m apart cannot hold 5 starts 4 m apart.
    wide = object_tasks.StartRules(2.0, 9.0, 0.0, True, 0.0, min_pairwise_start_m=4.0)
    taken = set()
    with pytest.raises(object_tasks.ObjectTaskSetError, match="spacing"):
        for k in range(5):
            start, _, _, _ = object_tasks.draw_start(candidates, house, "chair", 7000 + k,
                                                     wide, taken)
            taken.add((start["x"], start["y"]))


def test_builder_refuses_a_non_empty_folder(tmp_path):
    (tmp_path / "Rs").mkdir()
    (tmp_path / "Rs" / "world.yaml").write_text("left by a failed build\n")
    config = object_tasks.ObjectTaskSetConfig(["Rs"], ["chair"])

    def never_opened(_world):
        raise AssertionError("the simulator must not be opened")

    with pytest.raises(object_tasks.ObjectTaskSetError, match="not empty"):
        object_tasks.build(config, tmp_path, never_opened)
    with pytest.raises(object_tasks.ObjectTaskSetError, match="not empty"):
        object_tasks.ensure(config, tmp_path, never_opened)
    assert (tmp_path / "Rs" / "world.yaml").read_text() == "left by a failed build\n"


def test_view_mix_by_task_index():
    house = chair_house()
    half_fov = 28.9
    candidates = [(-5.0 + 0.5 * k, 0.0, "chair_0", 5.0) for k in range(16)]
    rules = object_tasks.StartRules(2.0, 9.0, 90.0, True, 0.0, min_pairwise_start_m=0.3,
                                    min_in_view=2, min_out_of_view=2)
    assert [rules.view_class(k) for k in range(5)] == ["in", "in", "out", "out", "any"]
    assert "min_in_view" in rules.as_dict() and "min_out_of_view" in rules.as_dict()
    taken, entries = set(), []
    for k in range(5):
        start, instance_id, _, _ = object_tasks.draw_start(
            candidates, house, "chair", 7000 + k, rules, taken,
            view=rules.view_class(k), half_fov_deg=half_fov)
        taken.add((start["x"], start["y"]))
        bearing = object_tasks.target_bearing(start, house.instances[instance_id])
        entries.append({"word": "chair", "target_in_view": abs(bearing) <= half_fov})
        if rules.view_class(k) == "in":
            assert abs(bearing) <= half_fov
        elif rules.view_class(k) == "out":
            assert half_fov < abs(bearing) <= 90.0 + 1e-6
    object_tasks.check_view_mix(rules, entries)  # passes
    with pytest.raises(object_tasks.ObjectTaskSetError, match="in view"):
        object_tasks.check_view_mix(rules, [dict(e, target_in_view=True) for e in entries])


def test_forward_clearance_mirrors_then_drops():
    house = chair_house()
    grid = house.grid

    def blocked_house(boxes):
        trav = np.where(grid.free, objects.FREE, 0).astype(np.uint8)
        for row in range(SIZE):
            for col in range(SIZE):
                x, y = grid.centre(row, col)
                if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in boxes):
                    trav[row, col] = 0
        return objects.SceneObjects("Test", objects.GridMap(trav, RES),
                                    list(house.instances.values()))

    half_fov = 28.9
    rules = object_tasks.StartRules(2.0, 9.0, 90.0, True, 0.0, forward_clear_m=1.0,
                                    min_out_of_view=1)
    assert rules.as_dict()["forward_clear_m"] == 1.0
    candidates = [(0.0, 0.0, "chair_0", 5.0)]  # the chair is due east
    # A wall on the north-east diagonal: every left (+) out-of-view heading is blocked.
    one_side = blocked_house([(0.1, 1.0, 0.1, 1.0)])

    def clear(start, objs=one_side):
        return object_tasks.forward_is_clear(objs.grid, None, start, 1.0, 0.0)

    mirrored = 0
    for seed in range(7000, 7010):
        rng = np.random.RandomState(seed)
        rng.randint(1)
        first = object_tasks.draw_offset(rng, rules, "out", half_fov)
        start, _, _, _ = object_tasks.draw_start(candidates, one_side, "chair", seed, rules,
                                                 view="out", half_fov_deg=half_fov,
                                                 forward_clear=clear)
        bearing = object_tasks.target_bearing(start, one_side.instances["chair_0"])
        assert clear(start) and half_fov < abs(bearing) <= 90.0 + 1e-6
        assert bearing > 0  # heading to the right of the chair: the clear side
        mirrored += first > 0
    assert mirrored > 0  # some seeds drew the blocked side first and were mirrored
    both_sides = blocked_house([(0.1, 1.0, 0.1, 1.0), (0.1, 1.0, -1.0, -0.1)])
    with pytest.raises(object_tasks.ObjectTaskSetError):
        object_tasks.draw_start(candidates, both_sides, "chair", 7000, rules, view="out",
                                half_fov_deg=half_fov,
                                forward_clear=lambda s: clear(s, both_sides))


def test_per_word_overrides_count_mix_and_fingerprint():
    base = dict(scenes=["Rs"], words=["chair", "sofa"], tasks_per_word=5,
                start={"min_geodesic_m": 2.5, "min_in_view": 1, "min_out_of_view": 4})
    plain = object_tasks.ObjectTaskSetConfig(**base)
    over = object_tasks.ObjectTaskSetConfig(
        per_word={"sofa": {"tasks": 4, "min_in_view": 2, "min_out_of_view": 2}}, **base)
    assert (over.tasks_for("chair"), over.tasks_for("sofa")) == (5, 4)
    assert over.rules_for("chair") is over.start
    sofa = over.rules_for("sofa")
    assert [sofa.view_class(k) for k in range(4)] == ["in", "in", "out", "out"]
    assert sofa.min_geodesic_m == 2.5  # every other rule is the set's own
    assert over.fingerprint() != plain.fingerprint()
    with pytest.raises(ValueError, match="not in words"):
        object_tasks.ObjectTaskSetConfig(per_word={"table": {"tasks": 3}}, **base)
    with pytest.raises(ValueError, match="unknown keys"):
        object_tasks.ObjectTaskSetConfig(per_word={"sofa": {"min_geodesic_m": 1.0}}, **base)


def test_clip_selection_prefers_word_succeeds_masked_fails(tmp_path):
    lg7_4_film = pytest.importorskip("lg7_4_film")

    def table(name, rows):
        with open(tmp_path / "{}.csv".format(name), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=metrics.CSV_COLUMNS)
            writer.writeheader()
            for task_id, seed, success, spl, ticks in rows:
                writer.writerow(row(task_id=task_id, seed=seed, success=success, spl=spl,
                                    ticks=ticks))

    table("clip_v2b+word", [("A", 1, 1, "1.0", 40), ("B", 2, 1, "0.9", 60), ("C", 3, 1, "0.8", 70)])
    table("clip_v2b+masked", [("A", 1, 1, "1.0", 41), ("B", 2, 0, "0.0", 300), ("C", 3, 0, "0.0", 300)])
    task_id, seed, selection = lg7_4_film.pick_clip(tmp_path)
    assert (task_id, seed) == ("B", 2)  # best of the contrast episodes, not the best overall
    assert selection["word_succeeds_masked_fails"] == 2
    assert selection["caption"].startswith("selected example: word succeeds, masked fails")
    table("clip_v2b+masked", [("A", 1, 1, "1.0", 41), ("B", 2, 1, "1.0", 50), ("C", 3, 1, "1.0", 50)])
    task_id, seed, selection = lg7_4_film.pick_clip(tmp_path)
    assert (task_id, seed) == ("A", 1)
    assert selection["caption"].startswith("selected example: best word success")
