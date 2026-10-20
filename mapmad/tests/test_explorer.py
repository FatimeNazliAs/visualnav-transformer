"""The explorer teacher (Phase 8.2, block item 1) on hand-made maps: nearest frontier, tie -> straight ahead,
commitment, blacklist, no frontier, approach with unknown = free, arrival and facing. No simulator."""

import math
import time

import numpy as np

from mapmad_sim.explorer import (APPROACH, ARRIVED, BLACKLIST_EXHAUSTED, NO_FRONTIER, SEARCH, OwnMap, SeenInstance,
                                 TeacherSpec, TeacherState, teacher_step)

RES = 0.1


def own_map(obstacle: np.ndarray, explored: np.ndarray) -> OwnMap:
    """Map with origin (0, 0): cell [i, j] centre = (0.05 + 0.1 i, 0.05 + 0.1 j)."""
    return OwnMap(obstacle.astype(bool), explored.astype(bool), (0.0, 0.0), RES)


def corridor(known_to: int, length: int = 100, width: int = 12) -> tuple:
    """A corridor along +X (rows = X): walls on columns 0 and width - 1, rows < known_to explored."""
    obstacle = np.zeros((length, width), bool)
    obstacle[:, 0] = obstacle[:, width - 1] = True
    explored = np.zeros_like(obstacle)
    explored[:known_to, :] = True
    return obstacle, explored


def test_goes_to_the_frontier_at_the_end_of_the_known_corridor():
    m = own_map(*corridor(40))
    d, st = teacher_step(m, (1.0, 0.6, 0.0), TeacherState())
    assert d.mode == SEARCH and d.new_choice
    assert math.isclose(d.goal[0], 3.95)  # frontier = last explored row (row 39)
    assert 2.5 < d.path_length_m < 3.5
    assert np.allclose(d.path[0], (1.0, 0.6)) and st.committed == d.goal
    assert d.next_point is not None and math.hypot(d.next_point[0] - 1.0, d.next_point[1] - 0.6) >= 0.45


def test_blind_zone_connects_the_robot_to_its_first_view():
    # frame 0 at pin025: only cells >= 0.55 m ahead are explored; the robot must still reach the wedge's frontier
    obstacle = np.zeros((100, 100), bool)
    explored = np.zeros_like(obstacle)
    explored[56:90, 35:65] = True  # X 5.6..9.0 ahead of a robot at X = 5.0
    d, _ = teacher_step(own_map(obstacle, explored), (5.0, 5.0, 0.0), TeacherState())
    assert d.mode == SEARCH and d.goal is not None
    d, _ = teacher_step(own_map(obstacle, explored), (5.0, 5.0, 0.0), TeacherState(), spec=TeacherSpec(blind_m=0.3))
    assert d.mode == NO_FRONTIER  # without the blind zone it would fail at once


def test_exploration_drive_label():
    d, _ = teacher_step(own_map(*corridor(40)), (1.0, 0.6, 0.0), TeacherState(), searching=False)
    assert d.mode == "explore"


def two_frontiers(left_end: int, right_end: int) -> tuple:
    """A T: a known cross corridor (rows 40-51) with unknown beyond columns left_end / right_end."""
    obstacle = np.zeros((100, 100), bool)
    explored = np.zeros_like(obstacle)
    explored[39:53, left_end:right_end] = True  # walls are explored too (obstacle cells always are)
    obstacle[39, left_end:right_end] = obstacle[52, left_end:right_end] = True
    return obstacle, explored


def test_nearest_frontier_wins_and_tie_goes_straight_ahead():
    obstacle, explored = two_frontiers(20, 90)  # left frontier 2.5 m from x = 5.0, right one 4.4 m
    pose = (4.6, 4.6, 0.0)  # cell (46, 46), facing +X (across the corridor, both frontiers 90 deg off)
    d, _ = teacher_step(own_map(obstacle, explored), pose, TeacherState())
    assert d.goal[1] < 4.6  # the nearer, left (-Y) end
    obstacle, explored = two_frontiers(25, 68)  # both ends ~2.1-2.2 m away: a tie within 0.5 m
    for yaw, side in ((-math.pi / 2, -1), (math.pi / 2, 1)):  # facing -Y -> the -Y end; facing +Y -> the +Y end
        d, _ = teacher_step(own_map(obstacle, explored), (4.6, 4.6, yaw), TeacherState())
        assert math.copysign(1, d.goal[1] - 4.6) == side


def test_commitment_survives_a_nearer_frontier_appearing():
    obstacle, explored = two_frontiers(20, 90)
    m = own_map(obstacle, explored)
    d, st = teacher_step(m, (4.6, 4.6, 0.0), TeacherState())
    goal = d.goal
    explored2 = explored.copy()
    explored2[40:52, 20:95] = True  # the far end moves; the committed (near) end stays a frontier
    d2, st2 = teacher_step(own_map(obstacle, explored2), (4.6, 4.4, 0.0), st)
    assert d2.goal == goal and not d2.new_choice and len(st2.lengths) == 2


def test_blacklist_after_40_steps_without_progress():
    obstacle, explored = two_frontiers(20, 90)
    m = own_map(obstacle, explored)
    pose = (4.6, 4.6, 0.0)
    d, st = teacher_step(m, pose, TeacherState())
    first = d.goal
    for _ in range(40):  # the robot does not move: no progress
        d, st = teacher_step(m, pose, st)
    assert d.dropped == "blacklisted" and d.new_choice and d.goal != first
    assert st.blacklist == ((*first, 0.5),)


def test_no_frontier_in_a_fully_known_room():
    obstacle = np.zeros((30, 30), bool)
    obstacle[0, :] = obstacle[-1, :] = obstacle[:, 0] = obstacle[:, -1] = True
    d, _ = teacher_step(own_map(obstacle, np.ones_like(obstacle)), (1.5, 1.5, 0.0), TeacherState())
    assert d.mode == NO_FRONTIER and d.goal is None


def test_blacklist_exhausted_when_every_frontier_is_dropped():
    m = own_map(*corridor(40))
    d, _ = teacher_step(m, (1.0, 0.6, 0.0), TeacherState(blacklist=((3.95, 0.55, 1.0),)))
    assert d.mode == BLACKLIST_EXHAUSTED


def test_reached_frontier_is_dropped_for_good():
    m = own_map(*corridor(40))
    d0, st = teacher_step(m, (1.0, 0.6, 0.0), TeacherState())
    d_goal = d0.goal
    d, st = teacher_step(m, (3.6, 0.6, 0.0), st)  # within 0.5 m of the committed goal (3.95, ...)
    assert d.dropped == "reached" and st.blacklist == ((*d_goal, 0.2),)


def test_approach_goes_through_unknown_and_arrives_facing_the_object():
    obstacle, explored = corridor(20)
    m = own_map(obstacle, explored)
    chair = SeenInstance(7, centre=(6.0, 0.6), view_points=((5.0, 0.6), (5.0, 0.4)))
    d, st = teacher_step(m, (1.0, 0.6, 0.0), TeacherState(), seen=[chair])
    assert d.mode == APPROACH and d.new_choice and d.goal == (5.0, 0.6)  # beyond the known part: unknown = free
    assert np.allclose(d.path[-1], (5.0, 0.6))
    d2, st2 = teacher_step(m, (2.0, 0.6, 0.0), st, seen=[chair])
    assert not d2.new_choice and d2.goal == d.goal  # same seen set: no re-evaluation
    d3, _ = teacher_step(m, (4.95, 0.6, 0.3), st2, seen=[chair])
    assert d3.mode == ARRIVED and math.isclose(d3.face_yaw, 0.0, abs_tol=1e-9)


def test_new_instance_triggers_re_evaluation():
    m = own_map(*corridor(20))
    far = SeenInstance(1, centre=(9.0, 0.6), view_points=((8.0, 0.6),))
    near = SeenInstance(2, centre=(3.0, 0.6), view_points=((2.5, 0.6),))
    d, st = teacher_step(m, (1.0, 0.6, 0.0), TeacherState(), seen=[far])
    assert d.goal == (8.0, 0.6)
    d, st = teacher_step(m, (1.2, 0.6, 0.0), st, seen=[far, near])
    assert d.new_choice and d.goal == (2.5, 0.6)


def test_speed_on_a_storey_sized_map():
    """Report only (block: <= 10 ms per step is a target, not a gate)."""
    rng = np.random.default_rng(0)
    obstacle = rng.random((300, 300)) < 0.03
    explored = np.zeros_like(obstacle)
    explored[50:250, 50:250] = True
    m = own_map(obstacle & explored, explored)
    t0 = time.perf_counter()
    for _ in range(5):
        teacher_step(m, (15.0, 15.0, 0.0), TeacherState())
    ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"teacher step on a 20 m x 20 m known area: {ms:.1f} ms")
    assert ms < 500
