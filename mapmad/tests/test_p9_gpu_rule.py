"""p9_rules_v2.md A1: which GPUs Phase 9 may take (the driver also checks the chosen GPU live with nvidia-smi)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p9_gpu_watch.py"


def usable():
    spec = importlib.util.spec_from_file_location("p9_gpu_watch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.usable_gpus


def state(free, phase10=(), other=()):
    return {"free": list(free), "phase10": list(phase10), "other": list(other), "ours": []}


def test_free_gpu_when_phase10_holds_none():
    assert usable()(state([1], other=[0]), []) == [1]  # the 18:20 case: GPU 1 idle, Phase 10 holds nothing


def test_free_gpu_when_phase10_holds_the_other():
    assert usable()(state([0], phase10=[1]), []) == [0]


def test_not_free_or_already_ours():
    assert usable()(state([]), []) == []
    assert usable()(state([0, 1]), [0]) == [1]


def test_flag_blocks_every_gpu():
    assert usable()(state([0, 1]), [], p10_wants_gpu=True) == []
