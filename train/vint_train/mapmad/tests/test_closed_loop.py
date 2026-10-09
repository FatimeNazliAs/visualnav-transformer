"""Closed-loop pieces on the NoMaD side: metrics, the episode runner with a fake simulator, and NoMaD itself
(official weights from configs/p1_baseline.yaml `nomad`; skipped where they are missing or there is no GPU).

    docker exec -w /app/visualnav-transformer/train -e PYTHONPATH=/app/visualnav-transformer/mapmad/src naz_mapmad \
        python -m pytest -q -p no:cacheprovider vint_train/mapmad/tests
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from mapmad_bridge.client import Frame  # noqa: E402
from mapmad_sim.run_layout import RunLayout  # noqa: E402
from vint_train.mapmad.closed_loop import metrics  # noqa: E402
from vint_train.mapmad.closed_loop.policy import Command, Observation, Policy  # noqa: E402
from vint_train.mapmad.closed_loop.runner import Arm, RunSettings, run_episode, write_video  # noqa: E402

LAYOUT = RunLayout.load()
EPISODE = {"episode_id": "e0", "type": "in_view", "home": "h", "split": "minival", "start_position": [0.0, 0.0, 0.0],
           "start_yaw": 0.0, "goal_photo_position": [0, 0, 0], "goal_photo_yaw": 0.0,
           "target": {"category": "chair", "point": [3.0, 0.0, 0.0], "box_center": [3.5, 0.5, 0.0],
                      "box_size": [0.5, 1.0, 0.5]}}


def test_spl_and_bootstrap():
    assert metrics.spl(True, 4.0, 5.0) == pytest.approx(0.8) and metrics.spl(True, 4.0, 3.0) == 1.0
    assert metrics.spl(False, 4.0, 4.0) == 0.0
    mean, lo, hi = metrics.bootstrap_ci([0, 1] * 20)
    assert mean == 0.5 and 0.3 < lo < 0.5 < hi < 0.7
    assert metrics.bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)
    d = metrics.paired([{"episode_id": i, "success": 1.0, "spl": 0.5} for i in "abc"],
                       [{"episode_id": i, "success": 0.0, "spl": 0.5} for i in "abcd"])
    assert d["success"] == (1.0, 1.0, 1.0) and d["spl"][0] == 0.0


class FakeSim:
    """Straight corridor along +x: the target point is 3 m ahead; v moves the robot v * 0.25 m."""

    def __init__(self, navmesh_sha256=None):
        self.x, self.navmesh_sha256 = 0.0, navmesh_sha256

    def state(self):
        return {"position": [self.x, 0.0, 0.0], "yaw": 0.0, "geodesic_m": 3.0 - self.x, "floor_below_feet_m": 0.16}

    def reset(self, episode, hfov_deg, goal_photo, topdown_m_per_px=0.05):
        self.x = 0.0
        topdown = {"grid": np.ones((80, 100), bool), "origin": [-1.0, -2.0], "m_per_px": 0.05}
        return Frame(rgb=np.zeros((240, 320, 3), np.uint8), state=self.state(),
                     goal_rgb=np.full((240, 320, 3), 200, np.uint8) if goal_photo else None, topdown=topdown,
                     navmesh_sha256=self.navmesh_sha256)

    def step(self, v, w):
        self.x += v * 0.25
        move = {"v": v, "w": w, "travelled_m": v * 0.25, "short_m": 0.0, "refused_substeps": 0, "collided": False}
        return Frame(rgb=np.full((240, 320, 3), int(self.x * 50) % 256, np.uint8), state=self.state(), move=move)


class NoisyForward(Policy):
    """Drives forward with a random speed, so the log depends on the seed."""

    context_frames = 4

    def reset(self, meta):
        self.rng = np.random.default_rng(meta["seed"])
        self.seen = []

    def act(self, obs: Observation) -> Command:
        assert len(obs.frames) <= 4 and not obs.extras  # no target / geodesic for the policy
        self.seen.append(obs.frames[-1])
        return Command(float(self.rng.uniform(0.1, 0.2)), 0.0, {"frames": len(obs.frames)})


RUN = RunSettings(success_m=1.0, max_steps=200, topdown_m_per_px=0.05, video_fps=8)
ARM = Arm(name="photo_iv", goal="photo", episodes="in_view", hfov_deg=66.5)


def test_runner_stops_at_success_and_logs_identically(tmp_path):
    result, rec = run_episode(FakeSim(), NoisyForward(), EPISODE, ARM, RUN, 7, tmp_path / "a.jsonl", "fp")
    run_episode(FakeSim(), NoisyForward(), EPISODE, ARM, RUN, 7, tmp_path / "b.jsonl", "fp")
    run_episode(FakeSim(), NoisyForward(), EPISODE, ARM, RUN, 8, tmp_path / "c.jsonl", "fp")
    assert result["success"] and result["final_geodesic_m"] <= 1.0
    assert result["path_length_m"] == pytest.approx(3.0 - result["final_geodesic_m"])
    assert result["spl"] == 1.0  # L = 3 m geodesic start -> target point > path length (~2 m)
    assert result["success_step"] == result["steps"] and metrics.success_at(result, 500) == 1.0
    assert metrics.success_at(result, result["steps"] - 1) == 0.0
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert (tmp_path / "a.jsonl").read_bytes() != (tmp_path / "c.jsonl").read_bytes()
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    assert len(lines) == result["steps"] + 2 and len(rec.frames) == result["steps"] + 1


def test_runner_refuses_an_episode_from_another_floor_map(tmp_path):
    """An episode keeps the hash of the floor map it was built on; on a changed floor map (e.g. a new robot size)
    the runner refuses it unless check_p1_episodes.py passed it on that floor map, for this episode file."""
    import json

    from mapmad_sim.episodes import FLOOR_MAP_CHECK, checked_floor_maps
    from vint_train.mapmad.closed_loop.runner import FloorMapMismatch

    episode = dict(EPISODE, navmesh_sha256="old")
    run_episode(FakeSim("old"), NoisyForward(), episode, ARM, RUN, 0, tmp_path / "same.jsonl", "fp")
    with pytest.raises(FloorMapMismatch):
        run_episode(FakeSim("new"), NoisyForward(), episode, ARM, RUN, 0, tmp_path / "x.jsonl", "fp")
    assert not (tmp_path / "x.jsonl").exists()  # refused before anything is logged

    report = {"fingerprint": "fp", "navmesh_sha256": {"h": {"stored": "old", "now": "new"}},
              "episodes": {"e0": {"home": "h", "pass": True}, "e1": {"home": "h", "pass": False}}}
    (tmp_path / FLOOR_MAP_CHECK).write_text(json.dumps(report))
    checked = checked_floor_maps(tmp_path, "fp")
    assert checked == {"e0": "new"}
    result, _ = run_episode(FakeSim("new"), NoisyForward(), episode, ARM, RUN, 0, tmp_path / "ok.jsonl", "fp", checked)
    assert result["success"]
    with pytest.raises(FloorMapMismatch):  # passed on "new", not on yet another floor map
        run_episode(FakeSim("newer"), NoisyForward(), episode, ARM, RUN, 0, tmp_path / "y.jsonl", "fp", checked)
    with pytest.raises(FloorMapMismatch):  # a failed episode stays refused
        run_episode(FakeSim("new"), NoisyForward(), dict(episode, episode_id="e1"), ARM, RUN, 0, tmp_path / "z.jsonl", "fp", checked)
    assert checked_floor_maps(tmp_path, "another file") == {}  # the report belongs to one episode file


def test_timeout_and_video(tmp_path):
    run = RunSettings(success_m=1.0, max_steps=5, topdown_m_per_px=0.05, video_fps=8)
    result, rec = run_episode(FakeSim(), NoisyForward(), EPISODE, ARM, run, 0, tmp_path / "t.jsonl", "fp")
    assert not result["success"] and result["steps"] == 5 and result["spl"] == 0.0
    assert result["success_step"] is None and len(rec.policy_ms) == 5
    write_video(tmp_path / "v.mp4", EPISODE, ARM, result, rec, run)
    assert (tmp_path / "v.mp4").stat().st_size > 1000


def test_policy_gets_raw_frames_without_video_overlays(tmp_path):
    """Overlays (goal inset, text, map) are drawn on copies for the video only."""
    sim, policy = FakeSim(), NoisyForward()
    result, rec = run_episode(sim, policy, EPISODE, ARM, RUN, 3, tmp_path / "r.jsonl", "fp")
    before = [f.copy() for f in rec.frames]
    write_video(tmp_path / "r.mp4", EPISODE, ARM, result, rec, RUN)
    assert all(np.array_equal(a, b) for a, b in zip(before, rec.frames))
    assert all(np.unique(f).size == 1 for f in policy.seen)  # FakeSim pictures are flat: nothing drawn on them


needs_weights = pytest.mark.skipif(not LAYOUT.nomad_weights.exists() or not torch.cuda.is_available(),
                                   reason="official NoMaD weights or GPU missing")


@pytest.fixture(scope="module")
def nomad():
    from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy

    torch.use_deterministic_algorithms(True)
    cfg = yaml.safe_load(LAYOUT.nomad_config.read_text())
    return NomadPolicy(str(LAYOUT.nomad_weights), cfg, torch.device("cuda"))


def pictures(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(n)]


@needs_weights
def test_nomad_waits_for_four_frames_then_drives_within_limits(nomad):
    nomad.reset({"seed": 0})
    assert nomad.act(Observation(frames=pictures(3), goal_image=None, step=0)) == Command(0.0, 0.0, {"waiting_for_frames": 3})
    for goal in (None, pictures(1, seed=5)[0]):
        cmd = nomad.act(Observation(frames=pictures(4), goal_image=goal, step=3))
        assert 0.0 <= cmd.v <= 0.2 and -0.4 <= cmd.w <= 0.4 and len(cmd.info["path_m"]) == 8


@needs_weights
def test_nomad_is_deterministic_per_seed(nomad):
    def paths(seed, goal):
        nomad.reset({"seed": seed})
        return nomad.sample_paths(Observation(frames=pictures(4), goal_image=goal, step=0))

    goal = pictures(1, seed=5)[0]
    assert np.array_equal(paths(3, goal), paths(3, goal))
    assert np.array_equal(paths(3, None), paths(3, None))
    assert not np.array_equal(paths(3, goal), paths(4, goal))
    assert not np.array_equal(paths(3, goal), paths(3, None))  # the goal photo changes the paths


def test_stuck_streaks_and_failure_types():
    from vint_train.mapmad.closed_loop import analysis

    def log(collided, geos, success):
        st = [{"kind": "step", "step": i + 1, "collided": c, "geodesic_m": g} for i, (c, g) in enumerate(zip(collided, geos))]
        return [{"kind": "episode"}] + st + [{"kind": "result", "success": success}]

    stuck = log([False] * 5 + [True] * 120, [5.0] * 125, False)
    assert analysis.first_stuck(stuck) == 6 and analysis.failure_type(stuck) == "fail_stuck"
    assert analysis.first_stuck(log([True] * 99 + [False], [5.0] * 100, False)) is None
    assert analysis.failure_type(log([False] * 10, [1.5] * 10, False)) == "fail_near"
    assert analysis.failure_type(log([False] * 10, [4.0] * 10, False)) == "fail_far"
    assert analysis.failure_type(log([True] * 150, [0.9] * 150, True)) == "success"
