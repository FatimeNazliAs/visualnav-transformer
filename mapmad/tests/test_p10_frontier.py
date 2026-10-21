"""Phase 10 frontier parity on hand-made maps: the live teacher's frontier (explorer.teacher_frontier: planning window,
bumped cells as obstacles) = the NoMaD side's map_v2.teacher_frontier_at from the saved arrays, cell for cell."""

import numpy as np

from mapmad_sim.depth_map import FirstSeenMap, MapSpec
from mapmad_sim.explorer import OwnMap, TeacherSpec, teacher_frontier
from vint_train.mapmad.local_map import known_at
from vint_train.mapmad.map_v2 import teacher_frontier_at

RES = 0.1


def saved_map(seed: int, shape=(60, 80), frames: int = 12):
    """Random rooms seen over `frames` frames (first-seen indices; obstacles also explored, as depth_map)."""
    rng = np.random.default_rng(seed)
    explored = np.full(shape, -1, np.int32)
    for t in range(frames):  # blobs of explored floor
        ci, cj, r = rng.integers(10, shape[0] - 10), rng.integers(10, shape[1] - 10), rng.integers(4, 12)
        ii, jj = np.mgrid[0:shape[0], 0:shape[1]]
        blob = ((ii - ci) ** 2 + (jj - cj) ** 2 <= r * r) & (explored < 0)
        explored[blob] = t
    wall = (rng.random(shape) < 0.08) & (explored >= 0)
    obstacle = np.where(wall, explored, -1).astype(np.int32)
    return {"obstacle_first_seen": obstacle, "explored_first_seen": explored, "origin": np.array([-1.0, 2.0]),
            "resolution": np.float64(RES)}


def test_live_teacher_frontier_equals_the_saved_one_with_bumps():
    spec = TeacherSpec()
    for seed in range(5):
        m = saved_map(seed)
        shape = m["explored_first_seen"].shape
        ex = np.argwhere(m["explored_first_seen"] >= 0)
        rng = np.random.default_rng(100 + seed)
        bumps = []
        for k in range(6):  # bump points on explored cells, logged at frames 0..10
            i, j = ex[rng.integers(len(ex))]
            bumps.append([float(rng.integers(0, 11)), -1.0 + (i + 0.3) * RES, 2.0 + (j + 0.7) * RES])
        bumps = np.array(bumps)
        fmap = FirstSeenMap(MapSpec(resolution=RES), m["origin"], shape)
        for t in (3, 7, 11):
            bumped = np.zeros(shape, bool)
            for k, x, y in bumps:
                if k < t:
                    flat, inside = fmap.cells(np.array([[x, y]]))
                    bumped.flat[flat[0]] = bool(inside[0])
            own = OwnMap(known_at(m["obstacle_first_seen"], t), known_at(m["explored_first_seen"], t),
                         tuple(m["origin"]), RES, bumped)
            i, j = ex[rng.integers(len(ex))]
            pose = (-1.0 + (i + 0.5) * RES, 2.0 + (j + 0.5) * RES, 0.3)
            live = teacher_frontier(own, pose, None, spec)
            assert np.array_equal(live, teacher_frontier_at(m, bumps, t, spec.min_cluster)), (seed, t)
