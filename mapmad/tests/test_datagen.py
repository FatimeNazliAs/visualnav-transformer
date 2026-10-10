"""Phase 2 generator parts: frames, depth -> map, planning grid, expert driver; with the HM3D mount, one real drive."""

import json
import math
import pickle

import numpy as np
import pytest

from mapmad_sim import config, frames
from mapmad_sim.camera import centre_depth, project_points
from mapmad_sim.depth_map import FirstSeenMap, MapSpec, camera_points, rotation, world_points
from mapmad_sim.expert import PursuitSpec, PurePursuit, TrackedPath, is_spot_turn, smooth_path, spot_turn
from mapmad_sim.floor_grid import FloorGrid
from mapmad_sim.robot import forward
from mapmad_sim.route import max_turn, plan_route
from mapmad_sim.run_layout import read_config

needs_hm3d = pytest.mark.skipif(not (config.paths()["hm3d_scenes"] / "train").exists(), reason="HM3D mount missing")


# --- frames ---------------------------------------------------------------------------------------------------

def test_frames_habitat_to_nomad():
    assert np.allclose(frames.to_2d([1.0, 5.0, -2.0]), [2.0, -1.0])
    assert np.allclose(frames.to_3d([2.0, -1.0], 5.0), [1.0, 5.0, -2.0])
    for yaw in (0.0, 0.7, -2.0):  # Habitat's facing direction maps onto heading(yaw): same yaw in both frames
        f = forward(yaw)
        assert np.allclose(frames.to_2d(f), frames.heading(yaw))
    assert np.allclose(frames.to_robot(np.array([[1.0, 1.0]]), (0.0, 0.0, math.pi / 2)), [[1.0, -1.0]])


# --- depth -> map -----------------------------------------------------------------------------------------------

def test_camera_rotation_matches_habitat_forward():
    r = rotation(0.4, -2.0)
    assert np.allclose(r @ [0.0, 0.0, -1.0], forward(0.4) * math.cos(math.radians(2)) + [0, -math.sin(math.radians(2)), 0])


def flat_floor_depth(h: int, w: int, hfov: float, cam_height: float) -> np.ndarray:
    """Planar depth of a level camera above an infinite floor (0 above the horizon)."""
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    rows = np.arange(h)[:, None] + 0.5 - h / 2.0
    with np.errstate(divide="ignore"):
        d = np.where(rows > 0, cam_height * f / rows, 0.0)
    return np.repeat(d, w, axis=1)


def test_floor_is_explored_not_obstacle_and_wall_is_obstacle():
    spec = MapSpec(resolution=0.1, max_depth_m=4.0)
    depth = flat_floor_depth(240, 320, 66.5, 0.18)
    depth[:, 150:170] = np.where(depth[:, 150:170] > 2.0, 2.0, depth[:, 150:170])  # a strip of wall 2 m ahead
    cam = np.array([0.0, 0.18, 0.0])  # Habitat, looking along -z = +X, floor at y = 0
    pts = world_points(depth, 66.5, spec.max_depth_m, cam, rotation(0.0, 0.0))
    m = FirstSeenMap(spec, origin=(-5.0, -5.0), shape=(100, 100))
    m.add(pts, floor_y=0.0, t=3)
    a = m.arrays(0.0)
    obst = np.argwhere(a["obstacle_first_seen"] == 3)
    assert len(obst) and np.allclose(np.unique(obst[:, 0]), [70])  # X in [2.0, 2.1): 2 m ahead
    assert set(np.unique(obst[:, 1])) <= {49, 50}  # straight ahead, Y around 0
    assert (a["explored_first_seen"][58:68, 47:53] == 3).all()  # floor 0.8-1.8 m ahead, |Y| < 0.3 m: explored
    assert (a["obstacle_first_seen"][55:68, :] == -1).all()  # ... and not obstacle


def test_first_seen_keeps_the_first_frame_and_unknown_depth():
    spec = MapSpec(min_points=3)
    m = FirstSeenMap(spec, origin=(0.0, 0.0), shape=(10, 10))
    p = np.array([[-0.55, 0.3, -0.55]] * 2)  # 2 points at (X, Y) = (0.55, 0.55), 0.3 m up: below the threshold
    m.add(p, 0.0, 0)
    assert m.arrays(0.0)["obstacle_first_seen"][5, 5] == -1
    m.add(p, 0.0, 1)  # 4 points now
    m.add(p, 0.0, 2)
    assert m.arrays(0.0)["obstacle_first_seen"][5, 5] == 1
    pts, keep = camera_points(np.array([[0.0, 5.0], [1.0, 2.0]]), 90.0, 4.0)
    assert len(pts) == 2 and list(keep) == [2, 3]  # 0 (hole) and 5 m (> max) are unknown


def test_height_band():
    m = FirstSeenMap(MapSpec(min_points=1), origin=(0.0, 0.0), shape=(10, 10))
    heights = [-0.5, -0.05, 0.02, 0.3, 0.6]  # ignored, floor, floor, obstacle, ignored
    for k, h in enumerate(heights):
        m.add(np.array([[-0.05 - 0.1 * k, h, -0.05]]), 0.0, 0)
    a = m.arrays(0.0)
    assert list(a["explored_first_seen"][0, :5]) == [-1, 0, 0, 0, -1]
    assert list(a["obstacle_first_seen"][0, :5]) == [-1, -1, -1, 0, -1]


# --- planning grid + expert -----------------------------------------------------------------------------------

def l_corridor() -> FloorGrid:
    """An L-shaped corridor, 1 m wide, 5 cm cells: along +X then turning left (+Y)."""
    mask = np.zeros((200, 200), bool)  # rows = z, cols = x; low corner (x, z) = (-6, -6)
    # 2D (X, Y) = (-z, -x): X from 0..4 at Y in [-0.5, 0.5]  -> z in [-4, 0], x in [-0.5, 0.5]
    mask[int((-4.5 + 6) / 0.05):int((0.5 + 6) / 0.05), int((-0.5 + 6) / 0.05):int((0.5 + 6) / 0.05)] = True
    # then Y from 0..4 at X in [3.5, 4.5] -> x in [-4, 0], z in [-4.5, -3.5]
    mask[int((-4.5 + 6) / 0.05):int((-3.5 + 6) / 0.05), int((-4.0 + 6) / 0.05):int((0.5 + 6) / 0.05)] = True
    return FloorGrid(mask, (-6.0, -6.0), 0.05, 0.25, 4.0)


def test_plan_keeps_to_the_middle():
    g = l_corridor()
    path = g.plan([0.0, 0.2], [4.0, 3.5])
    assert path is not None
    mid = path[(path[:, 0] > 1.0) & (path[:, 0] < 3.0)]
    # where the corridor has room, every cell keeps >= 0.25 m to spare (preferred clearance), i.e. |Y| <= 0.25
    assert min(g.clearance_at(p) for p in mid) >= 0.25 - 0.05 and np.abs(mid[:, 1]).max() <= 0.25
    assert all(g.clearance_at(p) > 0 for p in path)


def unicycle(pose, v, w, dt):
    x, y, yaw = pose
    if abs(w) < 1e-9:
        return x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw
    return (x + v / w * (math.sin(yaw + w * dt) - math.sin(yaw)), y - v / w * (math.cos(yaw + w * dt) - math.cos(yaw)),
            yaw + w * dt)


def test_pursuit_follows_the_l_without_spot_turns():
    g = l_corridor()
    path = smooth_path(g.plan([0.0, 0.0], [4.0, 3.5]), g.clearance_at)
    sp = PursuitSpec(max_v=0.2, max_w=0.4, dt=0.25)
    pp = PurePursuit(sp, TrackedPath(path))
    pose, steps, min_clear = (0.0, 0.0, 0.0), 0, 1.0
    while steps < 600:
        c = pp.command(pose)
        if c.done:
            break
        assert not is_spot_turn(c.v, c.w) and c.v >= 0.0
        assert abs(c.w) <= 0.4 + 1e-9 and c.v <= 0.2 + 1e-9
        pose = unicycle(pose, c.v, c.w, 0.25)
        min_clear = min(min_clear, g.clearance_at(pose[:2]))
        steps += 1
    assert c.done and min_clear > 0.1 and steps < 200  # ~7.5 m at <= 0.2 m/s


def test_spot_turn_lands_on_the_heading():
    assert spot_turn(0.05, 0.4, 0.25) == pytest.approx(0.2)
    assert spot_turn(-3.0, 0.4, 0.25) == -0.4
    assert is_spot_turn(0.0, 0.4) and not is_spot_turn(0.05, 0.4) and not is_spot_turn(0.0, 0.05)


# --- with HM3D --------------------------------------------------------------------------------------------------

@needs_hm3d
def test_camera_transform_matches_our_rotation():
    from mapmad_sim.robot import LimoSim, RobotSpec

    spec = RobotSpec.from_config(config.robot())
    with LimoSim("train", "00081-5biL7VEkByM", spec, depth=True) as r:
        r.place(np.array(r.pathfinder.get_random_navigable_point()), 0.8)
        t = r.camera_transform()
        assert np.allclose(t[:3, :3], rotation(0.8, spec.pitch_deg), atol=1e-5)
        assert t[1, 3] - r.floor_y == pytest.approx(spec.camera_height_m, abs=1e-4)
        assert np.allclose(t[[0, 2], 3], r.position[[0, 2]] + spec.camera_forward_m * forward(0.8)[[0, 2]], atol=1e-4)


@needs_hm3d
def test_one_real_drive(tmp_path):
    """One spot drive in a train home: files in NoMaD's format, frames match the map's frame count."""
    from mapmad_sim.datagen import generate_home
    from mapmad_sim.run_layout import read_config
    from mapmad_sim import home_splits

    cfg = read_config(config.CONFIG_DIR / "p2_datagen.yaml")
    homes = home_splits.train_homes()
    home = "00155-iLDo95ZbDJq"
    s = generate_home(cfg, tmp_path, home, homes.index(home), False, 0, {"unlabelled": {"spot": 1}})
    assert s["drives"] == 1 and (tmp_path / "_homes" / f"{home}.json").exists()
    d = tmp_path / f"{home}_000"
    traj = pickle.loads((d / "traj_data.pkl").read_bytes())
    n = len(traj["yaw"])
    assert traj["position"].shape == (n, 2) and len(list(d.glob("*.jpg"))) == n
    meta = json.loads((d / "mapmad_meta.json").read_text())
    assert meta["frames"] == n and meta["end_error_m"] <= 0.06
    with np.load(d / "mapmad_map.npz") as m:
        assert m["obstacle_first_seen"].max() <= n - 1 and m["explored_first_seen"].min() == -1
    # a second call is a no-op (resumable)
    again = generate_home(cfg, tmp_path, home, homes.index(home), False, 0, {"unlabelled": {"spot": 1}})
    assert again["seconds"] == s["seconds"] and again["drives"] == 1


def test_project_points_and_centre_depth():
    tf = np.eye(4)
    tf[:3, :3] = rotation(0.0, 0.0)
    tf[:3, 3] = [0.0, 0.18, 0.0]
    uv = project_points(np.array([[0.0, 0.18, -2.0], [1.0, 0.18, -2.0], [0.0, 0.18, 1.0]]), tf, 320, 240, 90.0)
    assert np.allclose(uv[0], [160.0, 120.0]) and np.allclose(uv[1], [240.0, 120.0])  # f = 160 px at 90 deg
    assert np.isnan(uv[2]).all()  # behind the camera
    d = np.zeros((240, 320), np.float32)
    assert centre_depth(d, 20) == 0.0
    d[110:130, 150:170] = 0.4
    d[115, 155] = 0.0  # a hole is skipped
    assert centre_depth(d, 20) == pytest.approx(0.4)


def test_plan_route_with_and_without_detour():
    cfg = read_config(config.CONFIG_DIR / "p2_datagen.yaml")
    g = l_corridor()
    plain = plan_route(g, np.array([0.0, 0.0]), np.array([4.0, 3.5]), 0, cfg, np.random.default_rng(0))
    assert plain.detour_points == [] and np.allclose(plain.path[-1], [4.0, 3.5])
    dc = cfg["variations"]["detour"]
    for seed in range(5):  # whenever a detour is placed, the route obeys the detour rules (module docstring)
        r = plan_route(g, np.array([0.0, 0.0]), np.array([4.0, 3.5]), 1, cfg, np.random.default_rng(seed))
        for p in r.detour_points:
            assert g.clearance_at(p) >= dc["min_clearance_m"]
        if r.detour_points:
            assert r.max_deviation_m >= dc["realised_m"]
            assert max_turn(r.path, dc["turn_window_m"]) <= math.radians(dc["max_turn_deg"]) + 1e-9
        assert all(g.clearance_at(p) > 0 for p in r.path)
