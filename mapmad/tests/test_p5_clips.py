"""Phase 5 meeting clips (scripts/p5_meeting_clips.py): the stuck-streak cut and the fastest-success pick."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import p5_meeting_clips as M  # noqa: E402


def driver(last, streaks):
    return SimpleNamespace(last=last, streaks=streaks)


def test_cut_keeps_the_first_seconds_and_the_final_pose():
    # video 12's C: one streak from step 377 for 624 steps (to the end, step 1000); 30 s at 12 fps = 360 poses
    assert M.skipped_ranges([driver(1000, [(377, 624)]), driver(129, [])], 360) == [(737, 1000)]
    assert M.skipped_ranges([driver(1000, [(100, 200)])], 360) == []  # short streak: nothing cut
    assert M.skipped_ranges([driver(1000, [(377, 624)])], 0) == []  # no cut asked
    # overlapping streaks of two drivers merge into one skipped stretch
    assert M.skipped_ranges([driver(900, [(0, 500)]), driver(900, [(50, 600)])], 100) == [(100, 650)]


def test_pick_fastest_success_in_bin(tmp_path):
    d = tmp_path / "summaries" / "C"
    d.mkdir(parents=True)
    runs = {"e1__s0": (True, 200, 1), "e2__s0": (True, 124, 1), "e3__s0": (True, 50, 0), "e4__s1": (True, 10, 1),
            "e5__s0": (False, None, 1), "e0__s0": (True, 124, 1)}
    for rid, (ok, step, b) in runs.items():
        (d / f"{rid}.json").write_text(json.dumps({"episode_id": rid.split("__")[0], "success": ok,
                                                   "success_step": step, "distance_bin": b}))
    (d / "e6__s0.crash.json").write_text("{}")
    # fewest steps at seed index 0 in bin 1; the tie at 124 goes to the smaller episode id
    assert M.pick_fastest(SimpleNamespace(root=tmp_path), "C", 1, 0) == ("e0", 124)
