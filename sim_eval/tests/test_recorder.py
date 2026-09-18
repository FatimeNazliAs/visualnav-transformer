"""Pin the parts of the recorder that can be wrong without looking wrong.

A video is judged by watching it, and most of what could break here breaks
visibly — a blank panel, a frozen image. Three things do not:

  * **which episodes get filmed.** `enabled: false` with a subset that names a
    task must film nothing; a run that quietly records twenty episodes because
    a flag was read in the wrong order costs an hour and a gigabyte.
  * **the map's world bounds** (below). The other silent failure — the
    distance head's number under the subgoal image — moved to `PolicyStep`
    in P5, because the episode trace needs the same reading; it is pinned in
    `test_localization.py` beside the index convention it undoes.
  * **the map's world bounds.** The top-down panel plots world metres straight
    onto an image, so the extent is the only thing holding the path, the goal
    and the walls in the same frame. Off by half a pixel is invisible; off by a
    factor is a path that floats beside the house. (The scene these come from
    is `SimScene`, pinned in `test_sim_scene.py`; what is pinned here is the
    geometry the panel does with them.)

All of it runs without a GPU, a simulator, a checkpoint or ffmpeg.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import recorder  # noqa: E402


def fake_trav_map(size=100, patch=(40, 60)):
    """A square traversability map with a known traversable patch."""
    trav_map = np.zeros((size, size), dtype=np.uint8)
    low, high = patch
    trav_map[low:high, low:high] = 255
    return trav_map


# --- which episodes get filmed ----------------------------------------------

def test_disabled_records_nothing_however_the_subset_is_spelled():
    for tasks in ("all", 10, ["Rs_00"]):
        config = recorder.RecordingConfig(enabled=False, tasks=tasks)
        assert not config.records("Rs_00", 0)


def test_all_records_every_task():
    config = recorder.RecordingConfig(enabled=True, tasks="all")
    assert config.records("Rs_00", 0)
    assert config.records("Rs_19", 19)


def test_a_count_records_the_first_n_tasks():
    config = recorder.RecordingConfig(enabled=True, tasks=2)
    assert [config.records("Rs_{:02d}".format(i), i) for i in range(4)] == [
        True, True, False, False]


def test_a_list_records_exactly_those_task_ids():
    config = recorder.RecordingConfig(enabled=True, tasks=["Rs_00", "Rs_07"])
    assert config.records("Rs_07", 7)
    assert not config.records("Rs_01", 1)


def test_a_nonsense_subset_is_refused_at_load_rather_than_mid_run():
    with pytest.raises(ValueError):
        recorder.RecordingConfig(enabled=True, tasks="every")
    with pytest.raises(ValueError):
        recorder.RecordingConfig(enabled=True, fps=0)


def test_the_cli_and_the_yaml_spell_subsets_the_same_way():
    assert recorder.parse_subset("all") == "all"
    assert recorder.parse_subset("3") == 3
    assert recorder.parse_subset("Rs_00,Rs_07") == ["Rs_00", "Rs_07"]


def test_a_null_recording_swallows_every_tick():
    with recorder.NullRecording() as film:
        film.capture(object())
    assert recorder.NullRecording().path is None


class FakeTask:
    """Only what the recorder asks a task for before it decides to film it."""

    task_id = "Rs_00"


def test_a_disabled_recorder_hands_back_null_recordings():
    assert isinstance(recorder.disabled().episode(FakeTask(), "arm", scene=None),
                      recorder.NullRecording)


# --- the map's world bounds -------------------------------------------------

def test_the_extent_puts_the_world_origin_at_the_centre_of_the_map():
    left, right, bottom, top = recorder.floor_extent(fake_trav_map(size=100), 0.1)
    assert left == pytest.approx(-5.05)
    assert right == pytest.approx(4.95)
    assert (bottom, top) == (left, right)
    # Centre pixel 50 spans [4.95, 5.05] of the way across, i.e. world 0 +- 0.05.
    assert left + 50 * 0.1 == pytest.approx(-0.05)


def test_the_view_box_covers_the_traversable_floor_plus_its_margin():
    bounds = recorder.traversable_bounds(
        fake_trav_map(size=100, patch=(40, 60)), 0.1, margin_m=1.0)
    left, right, bottom, top = bounds
    # Traversable columns 40..59 are world x -1.05 .. 0.85, plus a metre either
    # side.
    assert left == pytest.approx(-2.05)
    assert right == pytest.approx(1.85)
    assert (bottom, top) == (left, right)


def test_a_floor_with_no_traversable_pixels_has_no_view_box():
    assert recorder.traversable_bounds(fake_trav_map(size=20, patch=(0, 0)), 0.1) is None
