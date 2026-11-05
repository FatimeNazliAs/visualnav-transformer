"""Phase 14b-3: W1 recovery labels (stuck rule, motion, direction), the reverse switch in the arm table, the run plan
(seeds, exam order), the round-2 set and contamination helpers."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mapmad_sim import recovery_labels as rl  # noqa: E402
from mapmad_sim.arm_table import Arm  # noqa: E402
from mapmad_sim.dagger_labels import waypoints_of  # noqa: E402
from mapmad_sim.explorer import Decision, OwnMap  # noqa: E402

sets = pytest.importorskip("make_p14b3_sets")


def open_room(n: int = 80, wall_col: int = None) -> OwnMap:
    """An 8 m x 8 m room (0.1 m cells, origin 0): walls on the border, optionally a wall across at column wall_col."""
    obst = np.zeros((n, n), bool)
    obst[[0, -1], :] = obst[:, [0, -1]] = True
    if wall_col is not None:
        obst[:, wall_col] = True
    return OwnMap(obst, np.ones((n, n), bool), (0.0, 0.0), 0.1)


def test_stuck_rule_collisions_and_still():
    spec = rl.RecoverySpec()
    n = 60
    xy = np.stack([0.05 * np.arange(n), np.zeros(n)], axis=1)
    coll = np.zeros(n, bool)
    coll[10:26] = True  # 16 collisions: the 20 moves before t = 26 hold all of them, before t = 25 only 15
    s = rl.stuck_frames(coll, xy, spec)
    assert s[26] and s[30] and not s[25] and not s[5]
    still = np.zeros((n, 2))
    s2 = rl.stuck_frames(np.zeros(n, bool), still, spec)
    assert not s2[39] and s2[40]  # < 0.10 m over the last 40 steps


def test_reverse_turn_label_backs_up_then_turns_to_the_path():
    spec = rl.RecoverySpec()
    pose = (4.0, 4.0, 0.0)  # facing +X
    path = np.array([[4.0, 4.0], [4.0, 4.5], [4.0, 5.5]])  # the teacher wants to go left (+Y)
    dec = Decision(mode="search", path=path)
    poses = rl.recovery_poses(pose, dec, open_room(), spec, 40)
    assert poses.shape == (41, 3)
    assert poses[10, 0] == pytest.approx(3.75) and poses[10, 1] == pytest.approx(4.0)  # 0.25 m straight back
    wp = waypoints_of(poses, 5)
    assert wp[0, 0] == pytest.approx(-0.125) and wp[1, 0] == pytest.approx(-0.25)  # behind the robot
    end_heading = math.atan2(*(poses[-1, :2] - poses[-2, :2])[::-1])
    target = math.atan2(4.5 - 4.0, 4.0 - 3.75)
    assert end_heading == pytest.approx(target, abs=1e-6)  # turned toward the path, then forward
    assert wp[-1, 1] > 0.3  # the turn shows in the 2D waypoints (left)


def test_turn_only_fallback_has_no_backward_waypoints_and_free_ray_without_teacher():
    spec = rl.RecoverySpec(mode="turn")
    room = open_room(wall_col=45)  # a wall 0.5 m ahead (+Y side is free)
    pose = (2.0, 4.0, math.pi / 2)  # cell rows = X, cols = Y; facing +Y toward the wall at Y = 4.5
    poses = rl.recovery_poses(pose, None, room, spec, 40)
    wp = waypoints_of(poses, 5)
    assert (wp[:, 0] >= -1e-9).all()  # never behind
    heading = rl.free_ray(room, pose, spec)
    assert abs(math.cos(heading - pose[2])) < 0.99  # not straight into the wall
    with pytest.raises(ValueError):
        rl.RecoverySpec(mode="back")


def test_reverse_switch_is_for_network_arms_and_shows_in_switches():
    a = Arm("M3_P", checkpoint="v3", target="recogniser", recogniser="picked", unstuck="w4", mark_drive="amark",
            reverse="behind")
    assert a.switches == {"unstuck": "w4", "mark_drive": "amark", "reverse": "behind"}
    assert Arm("M21b", checkpoint="v21b", target="recogniser", recogniser="yoloe").switches == {}
    with pytest.raises(ValueError):
        Arm("F", policy="frontier", target="recogniser", recogniser="yoloe", reverse="behind")
    with pytest.raises(ValueError):
        Arm("X", checkpoint="v3", reverse="always")


def test_exam_order_puts_rows_1_3_first_and_one_seed_per_call():
    from mapmad_sim.run_plan_p14b3 import EXAM_ARMS, EXAM_ORDER

    labels = [x[0] for x in EXAM_ORDER]
    first = [x for x in EXAM_ORDER[:6]]
    assert {a for _, arms, _ in first for a in arms} == {"M3_P", "M3h_P", "M21b_w4_d_am"}
    assert all(len(seeds) == 1 for _, _, seeds in EXAM_ORDER)
    assert labels.index("ex_n_s0") < labels.index("ex_f") < labels.index("ex_m21bh_s0")
    covered = {(a, k) for _, arms, seeds in EXAM_ORDER for a in arms for k in seeds}
    for a in EXAM_ARMS:
        assert {k for b, k in covered if b == a} == ({0} if a == "F_d" else {0, 1, 2})


def test_overlap_by_full_id_and_scene():
    excl = {"photo_test": ["00081-5biL7VEkByM"], "mp3d": ["17DRP5sb8fy"], "hm3d_val": ["00800-TEEsavR23oF"]}
    assert not any(sets.overlap(["00006-HkseAnWCgqk"], excl).values())
    assert sets.overlap(["00999-5biL7VEkByM"], excl)["photo_test"] == ["00081-5biL7VEkByM"]  # same scene id
    assert sets.overlap(["17DRP5sb8fy"], excl)["mp3d"] == ["17DRP5sb8fy"]


def test_drive_homes_from_a_split_list(tmp_path):
    f = tmp_path / "traj_names.txt"
    f.write_text("p8d-00006-HkseAnWCgqk-explore-000\np11r-00009-chair-001\n")
    assert sets.drive_homes(f) == ["00006-HkseAnWCgqk", "00009"]
