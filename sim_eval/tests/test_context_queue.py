"""Pin the rolling observation window.

navigate.py's context queue is three lines and they are easy to get subtly
wrong: an off-by-one in the capacity feeds the encoder the wrong number of
frames, and a reversed order feeds it the future as the past. Both produce a
model that runs and steers badly, which is the worst kind of bug to find in a
rollout.

`ContextQueue` lives in bridge.py, which imports iGibson and pybullet lazily
precisely so this can be tested without a GPU or a simulator.

    ./sim_eval/run_tests.sh
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bridge import ContextQueue  # noqa: E402


def test_capacity_is_context_size_plus_the_current_frame():
    assert ContextQueue(context_size=3).capacity == 4


def test_seeding_fills_the_window_with_the_first_frame():
    """The cold start: at t=0 there is no history, so frame 0 stands in for it."""
    queue = ContextQueue(context_size=3)
    queue.seed("f0")
    assert queue.is_ready
    assert queue.frames == ["f0"] * 4


def test_new_frames_push_the_oldest_out_and_arrive_last():
    queue = ContextQueue(context_size=3)
    queue.seed("f0")
    queue.push("f1")
    assert queue.frames == ["f0", "f0", "f0", "f1"]
    queue.push("f2")
    assert queue.frames == ["f0", "f0", "f1", "f2"]


def test_the_window_never_grows_past_its_capacity():
    queue = ContextQueue(context_size=3)
    queue.seed("f0")
    for index in range(10):
        queue.push("f{}".format(index))
    assert len(queue.frames) == 4
    assert queue.frames == ["f6", "f7", "f8", "f9"]


def test_an_unseeded_queue_fills_before_it_is_ready():
    """navigate.py's original behaviour: wait for the camera stream to fill up."""
    queue = ContextQueue(context_size=2)
    for index in range(2):
        queue.push("f{}".format(index))
        assert not queue.is_ready
    queue.push("f2")
    assert queue.is_ready


def test_frames_are_a_copy_so_callers_cannot_mutate_the_window():
    queue = ContextQueue(context_size=1)
    queue.seed("f0")
    queue.frames.append("bogus")
    assert len(queue.frames) == 2


# --- the stride: training's spacing, not navigate.py's -----------------------
# vint_dataset.py `_context_times`: range(curr - context_size * spacing,
# curr + 1, spacing). These pin that the queue feeds exactly those ticks.


class PreStrideQueue:
    """ContextQueue as it was before the stride existed, verbatim.

    Kept as the reference the stride-1 queue must match frame for frame: every
    stride-1 arm (clean_stock, ctx03, img160x120) was scored with this, and the
    fix must not change a single frame of what they are fed.
    """

    def __init__(self, context_size):
        self.capacity = context_size + 1
        self._frames = []

    def seed(self, frame):
        self._frames = [frame] * self.capacity

    def push(self, frame):
        if len(self._frames) < self.capacity:
            self._frames.append(frame)
        else:
            self._frames.pop(0)
            self._frames.append(frame)

    @property
    def frames(self):
        return list(self._frames)

    @property
    def is_ready(self):
        return len(self._frames) == self.capacity


def training_context_times(curr_time, context_size, stride):
    """vint_dataset.py `_context_times` with waypoint_spacing 1, clamped at 0.

    Clamping is the cold start: before t = context_size * stride there is no
    such tick, and the seeded queue stands frame 0 in for it.
    """
    spacing = stride
    return [max(0, time) for time in
            range(curr_time - context_size * spacing, curr_time + 1, spacing)]


def test_stride_one_feeds_exactly_what_the_pre_stride_queue_fed():
    for context_size in (1, 3, 5):
        for seeded in (True, False):
            new, old = ContextQueue(context_size, stride=1), PreStrideQueue(context_size)
            if seeded:
                new.seed("f0")
                old.seed("f0")
            for tick in range(1, 30):
                assert new.frames == old.frames
                assert new.is_ready == old.is_ready
                new.push("f{}".format(tick))
                old.push("f{}".format(tick))


def test_the_default_stride_is_one():
    assert ContextQueue(context_size=3).stride == 1


def test_stride_three_feeds_the_ticks_training_sampled():
    queue = ContextQueue(context_size=3, stride=3)
    queue.seed(0)
    for tick in range(1, 25):
        queue.push(tick)
        assert queue.frames == training_context_times(tick, 3, 3), tick


def test_stride_three_still_feeds_context_size_plus_one_frames():
    queue = ContextQueue(context_size=3, stride=3)
    queue.seed("f0")
    assert queue.capacity == 4
    for tick in range(20):
        queue.push("f{}".format(tick))
        assert len(queue.frames) == 4


def test_seeding_at_a_stride_repeats_frame_zero_in_every_slot():
    """The cold start at training's spacing: history is frame 0 all the way back."""
    queue = ContextQueue(context_size=3, stride=3)
    queue.seed("f0")
    assert queue.is_ready
    assert queue.frames == ["f0"] * 4
    queue.push("f1")
    assert queue.frames == ["f0", "f0", "f0", "f1"]
    queue.push("f2")
    queue.push("f3")
    assert queue.frames == ["f0", "f0", "f0", "f3"]
    queue.push("f4")
    assert queue.frames == ["f0", "f0", "f1", "f4"]


def test_an_unseeded_strided_queue_waits_for_its_whole_window():
    """The look-back is context_size * stride ticks, so that is what must arrive."""
    queue = ContextQueue(context_size=2, stride=3)
    for tick in range(6):
        queue.push(tick)
        assert not queue.is_ready
    queue.push(6)
    assert queue.is_ready
    assert queue.frames == [0, 3, 6]


def test_a_stride_below_one_is_refused():
    with pytest.raises(ValueError):
        ContextQueue(context_size=3, stride=0)
