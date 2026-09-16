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
