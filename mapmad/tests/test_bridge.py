"""The simulator bridge: wire format, request handling with a fake robot, a real TCP round trip, and (with the
HM3D mount) the real server. Runs in both containers (numpy only, apart from the HM3D test):

    cd /app/visualnav-transformer/mapmad && python -m pytest -q -p no:cacheprovider tests/test_bridge.py
"""

import json
import socket
import struct
import threading
from dataclasses import dataclass

import numpy as np
import pytest

from mapmad_bridge import protocol
from mapmad_bridge.client import SimClient
from mapmad_sim import config
from mapmad_sim.sidecar_server import SimService, listen, serve

KEY = "ab" * 32
EPISODE = {"episode_id": "e0", "split": "minival", "home": "00800-TEEsavR23oF", "start_position": [1.0, 0.1, 2.0],
           "start_yaw": 0.5, "goal_photo_position": [0.0, 0.1, 0.0], "goal_photo_yaw": 0.0,
           "target": {"point": [3.0, 0.1, 2.0]}}


@dataclass
class FakeMove:
    v: float
    w: float
    travelled_m: float
    short_m: float = 0.0
    refused_substeps: int = 0
    collided: bool = False


class FakeRobot:
    """Moves along x by v * 0.25 per step; the picture's value encodes the step count."""

    navmesh_sha256 = "f" * 64

    def __init__(self, split, home, hfov_deg):
        self.home, self.hfov_deg, self.steps, self.closed = home, hfov_deg, 0, False
        self.position, self.yaw, self.floor_below_feet = np.zeros(3), 0.0, 0.16

    def place(self, position, yaw):
        self.position, self.yaw = np.array(position, dtype=float), float(yaw)

    def observe(self):
        return {"rgb": np.full((240, 320, 3), self.steps % 256, np.uint8)}

    def drive(self, v, w):
        self.steps += 1
        self.position = self.position + [v * 0.25, 0.0, 0.0]
        self.yaw += w * 0.25
        return FakeMove(v, w, v * 0.25)

    def geodesic(self, a, b):
        return float(np.linalg.norm(np.subtract(a, b)))

    def topdown(self, m_per_px, height):
        return np.ones((10, 20), bool), np.array([-1.0, -2.0])

    def close(self):
        self.closed = True


def wire(message):
    """encode -> split like recv does -> decode."""
    data = protocol.encode(message)
    (n,) = struct.unpack(">I", data[:4])
    return protocol.decode(json.loads(data[4:4 + n]), data[4 + n:])


def test_encode_round_trip_keeps_arrays_and_values():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    b = np.random.default_rng(0).random((2, 3, 4)).astype(np.float32)
    msg = {"x": a, "nested": {"list": [b, 1, "s", None], "flag": np.bool_(True)}, "f": np.float64(0.5),
           "grid": np.zeros((5, 7), bool)}
    out = wire(msg)
    assert np.array_equal(out["x"], a) and out["x"].dtype == np.uint8
    assert np.array_equal(out["nested"]["list"][0], b) and out["nested"]["list"][1:] == [1, "s", None]
    assert out["nested"]["flag"] is True and out["f"] == 0.5 and out["grid"].dtype == bool


def test_wire_format_is_json_plus_bytes_not_pickle():
    data = protocol.encode({"cmd": "x", "a": np.ones(3, np.uint8)})
    (n,) = struct.unpack(">I", data[:4])
    header = json.loads(data[4:4 + n])  # plain JSON header
    assert header["arrays"] == [{"dtype": "|u1", "shape": [3], "nbytes": 3}] and data[4 + n:] == b"\x01\x01\x01"


def test_service_reset_step_and_home_reload():
    made = []
    service = SimService(lambda s, h, f: made.append(FakeRobot(s, h, f)) or made[-1])
    r = service.handle({"cmd": "reset", "episode": EPISODE, "hfov_deg": 66.5, "goal_photo": True, "topdown_m_per_px": 0.05})
    assert r["rgb"].shape == (240, 320, 3) and r["goal_rgb"] is not None and r["navmesh_sha256"] == "f" * 64
    assert r["state"]["geodesic_m"] == pytest.approx(2.0) and r["topdown"]["origin"] == [-1.0, -2.0]
    s = service.handle({"cmd": "step", "v": 0.2, "w": 0.0})
    assert s["move"]["travelled_m"] == pytest.approx(0.05) and s["state"]["geodesic_m"] == pytest.approx(1.95)
    service.handle({"cmd": "reset", "episode": EPISODE, "hfov_deg": 66.5, "goal_photo": False, "topdown_m_per_px": 0.05})
    assert len(made) == 1  # same home + FOV: no reload
    service.handle({"cmd": "reset", "episode": EPISODE, "hfov_deg": 120.0, "goal_photo": False, "topdown_m_per_px": 0.05})
    assert len(made) == 2 and made[0].closed  # new FOV: new robot, old one closed


def test_step_before_reset_is_an_error():
    with pytest.raises(RuntimeError):
        SimService(FakeRobot).handle({"cmd": "step", "v": 0.1, "w": 0.0})


def serve_in_thread(service):
    listener = listen("127.0.0.1", 0)
    thread = threading.Thread(target=serve, args=(listener, service, KEY), daemon=True)
    thread.start()
    return listener, thread


def test_tcp_round_trip(monkeypatch):
    monkeypatch.setenv(protocol.KEY_ENV, KEY)
    service = SimService(FakeRobot)
    listener, thread = serve_in_thread(service)
    port = listener.getsockname()[1]
    with SimClient("127.0.0.1", port) as sim:  # first client leaves with `bye`
        first = sim.reset(EPISODE, hfov_deg=66.5, goal_photo=True)
        assert first.rgb.shape == (240, 320, 3) and first.goal_rgb.shape == (240, 320, 3)
        assert first.topdown["grid"].shape == (10, 20) and first.move is None
        frames = [sim.step(0.2, 0.1) for _ in range(3)]
        assert [int(f.rgb[0, 0, 0]) for f in frames] == [1, 2, 3]
        assert frames[-1].move["v"] == pytest.approx(0.2)
    sim = SimClient("127.0.0.1", port)  # a second client is served after the first said bye
    with pytest.raises(RuntimeError, match="unknown command"):
        sim._call("jump")
    sim.close(shutdown_server=True)
    thread.join(timeout=5)
    assert not thread.is_alive() and service.robot is None  # simulator closed by the server thread
    listener.close()


def test_wrong_key_is_refused_and_server_survives(monkeypatch):
    listener, thread = serve_in_thread(SimService(FakeRobot))
    port = listener.getsockname()[1]
    monkeypatch.setenv(protocol.KEY_ENV, "cd" * 32)
    with pytest.raises(RuntimeError, match="wrong or missing key"):
        SimClient("127.0.0.1", port, connect_timeout_s=1)
    raw = socket.create_connection(("127.0.0.1", port))  # skipping hello is refused too
    protocol.send(raw, {"cmd": "step", "v": 0.1, "w": 0.0})
    assert "error" in protocol.recv(raw)
    raw.close()
    monkeypatch.setenv(protocol.KEY_ENV, KEY)
    SimClient("127.0.0.1", port).close(shutdown_server=True)
    thread.join(timeout=5)
    listener.close()


HOME = "00006-HkseAnWCgqk"
needs_hm3d = pytest.mark.skipif(not config.hm3d_scene_dataset_config("train").exists(), reason="HM3D mount (/hm3d) missing")


@needs_hm3d
def test_real_server_with_a_frozen_episode(monkeypatch):
    from mapmad_sim.episodes import load_episodes
    from mapmad_sim.sidecar_server import limo_factory

    path = config.paths()["outputs"] / "p1_baseline" / "episodes" / "episodes.json"
    if not path.exists():
        pytest.skip("no Phase 1 episode file yet")
    monkeypatch.setenv(protocol.KEY_ENV, KEY)
    episode = load_episodes(path)[0][0]
    listener, thread = serve_in_thread(SimService(limo_factory(gpu=0)))
    sim = SimClient("127.0.0.1", listener.getsockname()[1])
    first = sim.reset(episode, hfov_deg=66.5, goal_photo=True)
    assert first.rgb.shape == (240, 320, 3) and first.rgb.dtype == np.uint8
    assert first.state["geodesic_m"] == pytest.approx(episode["start_geodesic_m"], abs=0.01)
    # the episode keeps the hash of the floor map it was built on; a robot-size change rebuilds the floor map, so
    # only the form is checked here. The runner refuses a mismatch (test_closed_loop: FloorMapMismatch).
    assert len(first.navmesh_sha256) == 64 and int(first.navmesh_sha256, 16) >= 0
    frame = sim.step(0.2, 0.0)
    assert frame.move["travelled_m"] <= 0.05 + 1e-4  # navmesh snapping adds micrometres
    sim.close(shutdown_server=True)
    thread.join(timeout=30)
    listener.close()
