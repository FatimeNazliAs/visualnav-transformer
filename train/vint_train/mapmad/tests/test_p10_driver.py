"""Phase 10 driver helpers (train/scripts/mapmad_p10, standard library only): the timing decision (p10_rules.md §6),
STATUS.md rendering, the GPU watcher's owner marker."""

import importlib.util
import json
import os

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "mapmad_p10")


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_timing_decisions():
    t = script("p10_timing")
    assert t.decide(False, 20000, 0.01, 0.9, 1, False)["decision"] == "ok"
    ok = t.decide(False, 20000, 0.01, 0.9, 1, False)
    assert ok["estimate_h"] == pytest.approx((20 * (782 * 0.91 / 60 + 0.5) + 11) / 60, abs=0.01)
    assert t.decide(False, 24100, 0.01, 0.9, 1, False)["decision"] == "memory_fallback"  # > 23.5 GiB
    assert t.decide(True, 1000, 0.01, 0.9, 1, False)["decision"] == "memory_fallback"
    assert t.decide(False, 20000, 0.2, 0.8, 1, False)["decision"] == "more_workers"  # loader 20% of the step
    assert t.decide(False, 20000, 0.2, 0.8, 1, True)["decision"] == "ok"  # workers raised once already
    assert t.decide(False, 20000, 0.0, 2.5, 1, False)["decision"] == "budget_stop"  # ~11.3 h
    assert t.decide(False, 20000, 0.0, 0.5, 2, False)["s_per_step"] == pytest.approx(1.0)  # accumulation: 2 batches


def test_status_renders_progress_and_states(tmp_path):
    s = script("p10_status")
    run = tmp_path / "runs" / "v2_s1_x"
    run.mkdir(parents=True)
    with open(run / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"record": "step", "epoch": 0, "step": 100, "diffusion_loss": 0.05}) + "\n")
        f.write(json.dumps({"epoch": 0, "train_seconds": 720.0, "eval_seconds": 30.0, "diffusion_loss": 0.05,
                            "dist_loss": 3.0, "train_mode_diffusion_loss": {"search_map": 0.06}}) + "\n")
    (tmp_path / "driver").mkdir()
    s.save(str(tmp_path / "driver" / "chain_state.json"), {"step": "4 main run", "gpu": "1", "main_run": str(run)})
    text = s.render(str(tmp_path))
    assert "epoch 1/20 done" in text and "GPU: 1" in text and "Come back at" in text and "search_map" in text
    s.save(str(tmp_path / "driver" / "chain_state.json"), {"state": "STOPPED", "reason": "NaN loss"})
    assert "**STOPPED: NaN loss**" in s.render(str(tmp_path))
    prog = s.run_progress(str(run))
    assert prog["done"] == 1 and prog["min_per_epoch"] == pytest.approx(12.5)


def test_watcher_owner_marker():
    w = script("gpu_watch")
    assert w.owner_of(os.getpid()) in ("chain", "twin", "other")
    assert w.owner_of(10 ** 9) == "other"  # no such process: busy, never ours
