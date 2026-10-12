"""Golden tests for Phase 5 run identity and arm rules (written before the clean-up; must hold before and after).

The strings below are the file names the exam actually wrote (runs/logs/<arm>/<episode>__s<k>.jsonl, ...).
"""

from pathlib import Path

import torch

from mapmad_sim.run_layout import RunLayout
from vint_train.mapmad.closed_loop.exam_runner import ExamArm
from vint_train.mapmad.closed_loop.run_exam import Policies, work

CONFIG = Path(__file__).resolve().parents[4] / "mapmad" / "configs" / "p5_exam.yaml"


def layout(tmp_path: Path) -> RunLayout:
    return RunLayout.load(CONFIG, subdir="runs", outputs=tmp_path)


def arms(lay: RunLayout):
    return [ExamArm(name=a, **lay.cfg["arms"][a]) for a in "ABCDEF"]


def test_run_paths_golden(tmp_path):
    lay = layout(tmp_path)
    root = tmp_path / "p5_exam" / "runs"
    run = "p5e-00800-sofa-2__s1"
    assert lay.log("C", run) == root / "logs" / "C" / "p5e-00800-sofa-2__s1.jsonl"
    assert lay.summary("C", run) == root / "summaries" / "C" / "p5e-00800-sofa-2__s1.json"
    assert lay.summary("C", run).with_suffix(".crash.json") == root / "summaries" / "C" / "p5e-00800-sofa-2__s1.crash.json"
    assert lay.timing("C", 3) == root / "timing" / "C.shard3.jsonl"


def test_work_order_golden(tmp_path):
    eps = [{"episode_id": f"e{i}"} for i in range(3)]
    got = [(e["episode_id"], k, a.name) for e, k, a in work(eps, arms(layout(tmp_path)), [0, 1, 2], 1, 2)]
    want = [("e1", 0, a) for a in "ABCDEF"] + [("e1", k, a) for k in (1, 2) for a in "ABCDF"]
    assert got == want  # episode -> seed -> arm; the planner (E) only at seed index 0


def test_arm_outputs_and_weights_golden(tmp_path):
    lay = layout(tmp_path)
    by = {a.name: a for a in arms(lay)}
    assert by["A"].outputs == {"depth": True, "map": False, "grid": False}
    assert by["C"].outputs == {"depth": True, "map": True, "grid": False}
    assert by["D"].outputs == {"depth": True, "map": False, "grid": False}
    assert by["E"].outputs == {"depth": True, "map": False, "grid": True}
    pol = Policies(lay, torch.device("cpu"))
    pol.hashes = {"nomad": "n", "mapmad": "m"}
    assert [pol.header(by[a]) for a in "ABCDEF"] == [{"weights_sha256": "n"}] * 2 + [{"weights_sha256": "m"}] * 2 + \
        [{}] + [{"weights_sha256": "m"}]


def test_run_identity_new_interface(tmp_path):
    from mapmad_sim.run_layout import parse_run_id, partial_path, run_id

    lay = layout(tmp_path)
    root = tmp_path / "p5_exam" / "runs"
    rid = run_id("p5e-00800-tv_monitor-2", 1)
    assert rid == "p5e-00800-tv_monitor-2__s1"
    for name in (rid, rid + ".jsonl", rid + ".json", rid + ".crash.json", rid + ".npz"):
        assert parse_run_id(name) == ("p5e-00800-tv_monitor-2", 1)
    assert lay.crash("C", rid) == lay.summary("C", rid).with_suffix(".crash.json")  # the name the exam wrote
    assert lay.trace("C", rid) == root / "traces" / "C" / f"{rid}.npz"
    assert lay.progress("A+B.shard0") == root / "progress.A+B.shard0.log"
    assert partial_path(lay.log("C", rid)).name == f"{rid}.jsonl.partial"
    lay.crash("C", rid).parent.mkdir(parents=True)
    lay.crash("C", rid).write_text("{}")
    assert lay.crashed_episodes() == ["p5e-00800-tv_monitor-2"] and lay.crashed_episodes(["A"]) == []


def test_arm_rules_new_interface(tmp_path):
    by = {a.name: a for a in arms(layout(tmp_path))}
    assert [by[a].seed_indices([0, 1, 2]) for a in "AE"] == [[0, 1, 2], [0]]
    assert [by[a].run_seed_index(2) for a in "AE"] == [2, 0]
    assert [by[a].mapmad_goal for a in "CDF"] == ["map", "masked", "photo"]
    assert [by[a].reset_goal for a in "ABCF"] == ["masked", "photo", "masked", "photo"]
    assert [by[a].weights_key for a in "ABCDEF"] == ["nomad", "nomad", "mapmad", "mapmad", None, "mapmad"]
    assert [a for a in "ABCDEF" if by[a].map_given] == ["C"]
