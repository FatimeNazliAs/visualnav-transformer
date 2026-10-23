"""Phase 11 (Habitat side): DAgger labels on hand-made maps (look-ahead, frozen blacklist, skips), the online map's
bump rule, set id checks, and -- with a GPU (MAPMAD_TEST_GPU=<gpu>) -- the online map = generator map replay on a
saved dev drive (scripts/p11_map_parity.py)."""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from mapmad_sim.dagger_labels import (ARRIVED, REASONS, LabelSpec, arc, collision_frames, frozen_blacklist,
                                      label_poses, lookahead, motion_step, waypoints_of)
from mapmad_sim.explorer import (BLACKLIST_EXHAUSTED, SEARCH, Decision, OwnMap, TeacherSpec, TeacherState,
                                 teacher_step)
from mapmad_sim.expert import PursuitSpec
from mapmad_sim.search_set import rollout_episode_id, search_set_ok

RES = 0.1
PURSUIT = PursuitSpec(max_v=0.2, max_w=0.4, dt=0.25)


def corridor(known_to: int, length: int = 100, width: int = 12) -> OwnMap:
    """A corridor along +X (rows = X): walls on columns 0 and width - 1, rows < known_to explored."""
    obstacle = np.zeros((length, width), bool)
    obstacle[:, 0] = obstacle[:, width - 1] = True
    explored = np.zeros_like(obstacle)
    explored[:known_to, :] = True
    return OwnMap(obstacle, explored, (0.0, 0.0), RES)


# --- look-ahead ---------------------------------------------------------------------------------------------------

def test_arc_matches_a_straight_line_and_a_quarter_turn():
    x, y, yaw = arc((0.0, 0.0, 0.0), 0.2, 0.0, 0.25)
    assert math.isclose(x, 0.05) and abs(y) < 1e-12 and yaw == 0.0
    p = (0.0, 0.0, 0.0)
    for _ in range(int(round((math.pi / 2) / (0.4 * 0.25)))):  # v = 0.2, w = 0.4: radius 0.5 m, left turn
        p = arc(p, 0.2, 0.4, 0.25)
    assert abs(p[0] - 0.5) < 0.02 and abs(p[1] - 0.5) < 0.02  # quarter circle to the LEFT (+Y)


def test_label_in_a_corridor_drives_straight_ahead_at_full_speed():
    own = corridor(60)
    pose = (1.0, 0.6, 0.0)
    dec, st = teacher_step(own, pose, TeacherState())
    for replan in (False, True):
        spec = LabelSpec(replan=replan)
        wp = waypoints_of(label_poses(own, dec, st, pose, [], TeacherSpec(), PURSUIT, spec), spec.every)
        assert wp.shape == (8, 2)
        assert np.all(np.diff(wp[:, 0]) > 0)  # always forward
        assert np.abs(wp[:, 1]).max() < 0.1  # stays in the corridor's middle
        assert 1.6 < wp[-1, 0] < 2.05  # ~0.2 m/s x 10 s (pure pursuit slows little on a straight)


def test_label_turns_on_the_spot_first_for_a_new_goal_behind():
    own = corridor(60)
    pose = (3.0, 0.6, math.pi)  # facing -X; the frontier is at +X
    dec, st = teacher_step(own, pose, TeacherState())
    assert dec.new_choice
    poses = label_poses(own, dec, st, pose, [], TeacherSpec(), PURSUIT, LabelSpec())
    turning = np.hypot(*(poses[1:8, :2] - poses[0, :2]).T)
    assert turning.max() < 1e-9  # retarget: spot turn while the aim is > 30 deg off
    assert poses[-1, 0] > 3.2  # then drives toward +X


def test_arrived_label_turns_to_face_and_stands():
    dec = Decision(ARRIVED, np.asarray([[1.0, 1.0]]), face_yaw=math.pi / 2)
    poses = lookahead(dec, (1.0, 1.0, 0.0), None, PURSUIT, LabelSpec(replan=False))
    assert np.allclose(poses[:, :2], [1.0, 1.0])
    assert abs(poses[-1, 2] - math.pi / 2) <= math.radians(15) + 1e-9
    assert motion_step(dec, (1.0, 1.0, math.pi / 2), False, PURSUIT, LabelSpec()) == (0.0, 0.0, False)


# --- teacher state: frozen blacklist -------------------------------------------------------------------------------

def test_frozen_blacklist_never_drops_a_goal_the_student_ignores():
    own = corridor(40)
    pose = (1.0, 0.6, 0.0)
    normal, frozen = TeacherSpec(blacklist_window_steps=40), frozen_blacklist(TeacherSpec(blacklist_window_steps=40))
    s_n = s_f = TeacherState()
    dropped_n = dropped_f = 0
    for _ in range(120):  # the student stands still: no progress toward the committed frontier
        d_n, s_n = teacher_step(own, pose, s_n, spec=normal)
        d_f, s_f = teacher_step(own, pose, s_f, spec=frozen)
        dropped_n += d_n.dropped == "blacklisted"
        dropped_f += d_f.dropped == "blacklisted"
    assert dropped_n >= 1 and dropped_f == 0
    assert d_f.mode == SEARCH and s_f.committed is not None and not s_f.blacklist


def test_collision_frames_marks_only_long_streaks():
    c = np.zeros(60, bool)
    c[5:15] = True  # 10 steps: kept
    c[30:52] = True  # 22 steps: skipped
    out = collision_frames(c, 20)
    assert not out[:30].any() and out[30:52].all() and not out[52:].any()
    assert REASONS[0] == "ok" and BLACKLIST_EXHAUSTED in REASONS


# --- online map: bump rule ----------------------------------------------------------------------------------------

class _Move:
    def __init__(self, v, travelled, collided):
        self.v, self.w, self.travelled_m, self.collided = v, 0.0, travelled, collided


def test_bump_rule_marks_the_cell_ahead_only_for_a_stopped_forward_move():
    from mapmad_sim.search_v2 import SearchMap

    sm = SearchMap.__new__(SearchMap)  # only the bump parts (no simulator)
    from mapmad_sim.depth_map import FirstSeenMap, MapSpec
    from mapmad_sim.search_v2 import SearchSpec
    from mapmad_sim.target_layer import NoiseSpec

    sm.fmap = FirstSeenMap(MapSpec(), (0.0, 0.0), (50, 50))
    sm.bumped = np.zeros((50, 50), bool)
    sm.spec = SearchSpec(MapSpec(), NoiseSpec())
    sm.rec, sm.t = None, 3
    sm.after_move(_Move(0.2, 0.01, True), (1.05, 1.05, 0.0), 0.25)  # 0.01 m < half of 0.05 m: stopped -> bump
    assert sm.bumped[13, 10] and sm.bumped.sum() == 1  # cell under (1.35, 1.05)
    sm.after_move(_Move(0.2, 0.02, True), (1.05, 2.05, math.pi / 2), 0.25)
    assert sm.bumped[10, 23]  # 0.3 m along +Y: (1.05, 2.35)
    sm.after_move(_Move(0.2, 0.049, False), (2.0, 2.0, 0.0), 0.25)  # no collision: nothing
    sm.after_move(_Move(0.0, 0.0, True), (2.0, 2.0, 0.0), 0.25)  # turning on the spot: nothing
    sm.after_move(_Move(0.2, 0.03, True), (2.0, 2.0, 0.0), 0.25)  # travelled >= half: no bump
    assert sm.bumped.sum() == 2


# --- sets -----------------------------------------------------------------------------------------------------------

def test_set_ids_must_be_unique_and_parseable():
    e = {"episode_id": rollout_episode_id("00006-HkseAnWCgqk", "chair", 3), "target": {"view_points": [[0, 0, 0]]}}
    assert e["episode_id"] == "p11r-00006-chair-003"
    assert search_set_ok([e]) is None
    assert "duplicate" in search_set_ok([e, dict(e)])
    assert search_set_ok([dict(e, episode_id="p11r-1__s0")]) is not None
    assert search_set_ok([dict(e, target={"view_points": []})]) is not None


# --- re-check comparison (scripts/p11_recheck_compare.py) ------------------------------------------------------------

def test_recheck_compare_nan_equal_and_timing_ignored(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import p11_recheck_compare as R

    assert R.same_array(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
    assert not R.same_array(np.array([1.0, np.nan]), np.array([2.0, np.nan]))
    assert R.same_array(np.array(["a", "b"]), np.array(["a", "b"]))
    assert R.strip_timing({"steps": 3, "seconds": 1.2, "run": {"wall_time": 5, "n": 1}}) == {"steps": 3, "run": {"n": 1}}
    a, b = tmp_path / "a", tmp_path / "b"
    for d, t in ((a, 1.0), (b, 9.0)):
        d.mkdir()
        (d / "0.jpg").write_bytes(b"jpg")
        np.savez(d / "labels.npz", goal_xy=np.array([np.nan, 1.0]))
        (d / "mapmad_meta.json").write_text(f'{{"steps": 3, "seconds": {t}}}')
    assert R.check_drive(a, b)["pass"]
    (b / "0.jpg").write_bytes(b"other")
    assert R.check_drive(a, b)["differing"] == ["0.jpg"]


# --- GPU: online map = generator map ---------------------------------------------------------------------------------

PARITY_DRIVE = Path("/mapmad_data/p11_checks_parity/p11x-00746-RTV2n6fXB2w-search-000")


@pytest.mark.skipif(not os.environ.get("MAPMAD_TEST_GPU") or not PARITY_DRIVE.exists(),
                    reason="needs a GPU (MAPMAD_TEST_GPU=<gpu>) and the parity drive (scripts/p11_map_parity.py)")
def test_online_map_equals_the_generator_map_on_a_dev_drive():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import p11_map_parity as P
    from mapmad_sim.run_layout import RunLayout
    from mapmad_sim.search_wire import camera_wire, datagen_config, search_settings

    layout = RunLayout.load(P.CONFIG)
    row = P.replay(PARITY_DRIVE, int(os.environ["MAPMAD_TEST_GPU"]),
                   search_settings(datagen_config(layout.cfg), P.TRAIN_MAP), camera_wire(layout.cfg["camera"]), "poses")
    assert row["pass"] and row["mismatched_frames"] == 0 and row["frames"] > 100
