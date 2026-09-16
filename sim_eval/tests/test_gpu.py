"""Pin down the GPU-selection rules.

These run anywhere — no GPU, no iGibson, no container — which is the point of
keeping `resolve_gpu` free of side effects. The rule being protected is
"never silently pick a GPU nobody asked for", and it is the kind of rule that
is easy to relax by accident while making an unrelated change.

    python -m pytest sim_eval/tests/test_gpu.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gpu  # noqa: E402


def test_explicit_argument_wins_over_everything():
    environ = {gpu.RENDER_DEVICE_ENV: "0", gpu.WORKSTREAM_ENV: "0"}
    assert gpu.resolve_gpu(explicit=1, environ=environ) == 1


def test_render_device_env_beats_workstream_env():
    environ = {gpu.RENDER_DEVICE_ENV: "1", gpu.WORKSTREAM_ENV: "0"}
    assert gpu.resolve_gpu(environ=environ) == 1


def test_falls_back_to_workstream_env():
    assert gpu.resolve_gpu(environ={gpu.WORKSTREAM_ENV: "1"}) == 1


def test_empty_values_are_treated_as_unset():
    environ = {gpu.RENDER_DEVICE_ENV: "  ", gpu.WORKSTREAM_ENV: "1"}
    assert gpu.resolve_gpu(environ=environ) == 1


def test_refuses_to_guess_when_nothing_is_set():
    # The whole point: no fallback to 0. GPU 0 also drives the X server, and a
    # wrong pin on a shared machine is silent.
    with pytest.raises(gpu.GpuSelectionError):
        gpu.resolve_gpu(environ={})


def test_rejects_a_non_numeric_index():
    with pytest.raises(gpu.GpuSelectionError):
        gpu.resolve_gpu(environ={gpu.RENDER_DEVICE_ENV: "all"})


def test_resolve_does_not_touch_the_real_environment(monkeypatch):
    monkeypatch.delenv(gpu.RENDER_DEVICE_ENV, raising=False)
    monkeypatch.delenv(gpu.TORCH_DEVICE_ENV, raising=False)
    gpu.resolve_gpu(explicit=1)
    assert gpu.RENDER_DEVICE_ENV not in os.environ
    assert gpu.TORCH_DEVICE_ENV not in os.environ


class FakeRenderer:
    def __init__(self, device_minor):
        self.device_minor = device_minor


def test_verify_accepts_a_matching_renderer():
    assert gpu.verify_renderer(FakeRenderer(1), 1) == 1


def test_verify_rejects_the_wrong_gpu():
    # This is the failure that actually happened in P0 and rendered on GPU 0.
    with pytest.raises(gpu.GpuSelectionError):
        gpu.verify_renderer(FakeRenderer(0), 1)


def test_verify_rejects_a_renderer_that_cannot_be_checked():
    with pytest.raises(gpu.GpuSelectionError):
        gpu.verify_renderer(object(), 1)
