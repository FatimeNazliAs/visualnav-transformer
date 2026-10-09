"""The shared step-log reader (mapmad_bridge.steplog) and the run layout. Stdlib + PyYAML only: runs in both containers.

    cd /app/visualnav-transformer/mapmad && python -m pytest -q -p no:cacheprovider tests/test_steplog.py
"""

import json
from pathlib import Path

from mapmad_bridge import steplog
from mapmad_sim.run_layout import RunLayout


def write_log(path: Path, collided):
    lines = [{"kind": "episode", "arm": "a", "start": {"position": [0.0, 0.0, 0.0], "yaw": 0.5}}]
    lines += [{"kind": "step", "step": i + 1, "collided": c, "position": [0.05 * (i + 1), 0.0, 0.0], "yaw": 0.5}
              for i, c in enumerate(collided)]
    lines += [{"kind": "result", "success": False}]
    path.write_text("".join(json.dumps(l) + "\n" for l in lines))
    return path


def test_read_and_parts(tmp_path):
    log = steplog.read(write_log(tmp_path / "e.jsonl", [False, True, True]))
    assert steplog.header(log)["arm"] == "a" and steplog.result(log)["success"] is False
    assert [s["step"] for s in steplog.steps(log)] == [1, 2, 3]
    poses = steplog.poses(log)
    assert len(poses) == 4 and poses[0] == ([0.0, 0.0, 0.0], 0.5) and poses[2][0][0] == 0.1


def test_runs_and_stuck_streaks(tmp_path):
    assert list(steplog.runs([True, True, False, True])) == [(0, 2), (3, 1)]
    assert list(steplog.runs([])) == []
    log = steplog.read(write_log(tmp_path / "s.jsonl", [False] * 3 + [True] * 5 + [False] + [True] * 2))
    assert steplog.stuck_streaks(log, 5) == [(4, 5)] and steplog.first_stuck(log, 5) == 4
    assert steplog.stuck_streaks(log, 2) == [(4, 5), (10, 2)] and steplog.first_stuck(log, 6) is None


def test_run_layout_paths(tmp_path):
    layout = RunLayout.load(subdir="checks/x", outputs=tmp_path)
    assert layout.base == tmp_path / "p1_baseline" and layout.root == layout.base / "checks/x"
    assert layout.episodes_file == tmp_path / "p1_baseline/episodes/episodes.json"  # never below subdir
    assert layout.log("photo_iv", "e1") == layout.root / "logs/photo_iv/e1.jsonl"
    assert layout.summary("photo_iv", "e1").parent == layout.root / "summaries/photo_iv"
    assert layout.nomad_weights == tmp_path / "weights/official/nomad.pth"
    (layout.root / "logs/b").mkdir(parents=True)
    (layout.root / "logs/b/e2.jsonl").write_text("")
    assert layout.logs() == [layout.root / "logs/b/e2.jsonl"] and layout.logs("a") == []
