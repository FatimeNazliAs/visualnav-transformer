"""Simulator server for closed-loop runs: the virtual LIMO behind a TCP socket on mapmad-net (protocol in
mapmad_bridge/protocol.py). Runs in naz_mapmad_habitat; the policy runs in naz_mapmad and connects by name.

    MAPMAD_SIDECAR_KEY=<hex> python -m mapmad_sim.sidecar_server --port 5550 --gpu 0

Listens on all interfaces of the container, which only mapmad-net reaches (never publish the port with -p).
One client at a time; after `bye` the next client may connect, `close` ends the server.
"""

import argparse
import hmac
import math
import socket
import traceback
from dataclasses import asdict
from typing import Any, Callable, Dict, Optional, Tuple

from mapmad_bridge import protocol

RobotFactory = Callable[[str, str, float], Any]  # (split, home, hfov_deg) -> LimoSim-like robot


def limo_factory(gpu: int, seed: int = 0) -> RobotFactory:
    """Robots built from configs/robot_limo.yaml (LIMO-sized floor map) with the requested camera FOV."""
    from mapmad_sim import config
    from mapmad_sim.robot import LimoSim, RobotSpec

    def make(split: str, home: str, hfov_deg: float) -> Any:
        return LimoSim(split, home, RobotSpec.from_config(config.robot(), hfov_deg=hfov_deg), gpu=gpu, seed=seed)

    return make


class SimService:
    """Answers protocol requests; keeps one home loaded and reloads it only when home or camera FOV change."""

    def __init__(self, factory: RobotFactory) -> None:
        self.factory = factory
        self.robot: Optional[Any] = None
        self.key: Optional[Tuple[str, str, float]] = None
        self.target = None

    def handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        cmd = msg["cmd"]
        if cmd == "hello":
            return {"version": protocol.PROTOCOL_VERSION, "home": self.key[1] if self.key else None,
                    "hfov_deg": self.key[2] if self.key else None}
        if cmd == "reset":
            return self.reset(msg["episode"], msg["hfov_deg"], msg["goal_photo"], msg["topdown_m_per_px"])
        if cmd == "step":
            if self.robot is None or self.target is None:
                raise RuntimeError("step before reset")
            move = self.robot.drive(msg["v"], msg["w"])
            return {"rgb": self.robot.observe()["rgb"], "state": self.state(), "move": asdict(move)}
        raise ValueError(f"unknown command {cmd!r}")

    def reset(self, episode: Dict[str, Any], hfov_deg: float, goal_photo: bool, m_per_px: float) -> Dict[str, Any]:
        key = (episode["split"], episode["home"], float(hfov_deg))
        if key != self.key:
            self.close()
            self.robot, self.key = self.factory(*key), key
        self.target = episode["target"]["point"]
        goal_rgb = None
        if goal_photo:  # rendered with this arm's camera (a 120 deg photo for the 120 deg arms), same pose
            self.robot.place(episode["goal_photo_position"], episode["goal_photo_yaw"])
            goal_rgb = self.robot.observe()["rgb"]
        self.robot.place(episode["start_position"], episode["start_yaw"])
        grid, origin = self.robot.topdown(m_per_px, float(episode["start_position"][1]))
        return {"rgb": self.robot.observe()["rgb"], "goal_rgb": goal_rgb, "state": self.state(),
                "topdown": {"grid": grid, "origin": [float(origin[0]), float(origin[1])], "m_per_px": m_per_px},
                "navmesh_sha256": self.robot.navmesh_sha256}

    def state(self) -> Dict[str, Any]:
        geo = self.robot.geodesic(self.robot.position, self.target)
        return {"position": [float(x) for x in self.robot.position], "yaw": float(self.robot.yaw),
                "geodesic_m": geo if math.isfinite(geo) else None,
                "floor_below_feet_m": float(self.robot.floor_below_feet)}

    def close(self) -> None:
        if self.robot is not None:
            self.robot.close()
        self.robot, self.key, self.target = None, None, None


def serve(listener: socket.socket, service: SimService, key: str) -> None:
    """Answer clients one after another until one sends `close`; then close the simulator (in this thread,
    which created it: habitat-sim must not be torn down from another thread)."""
    try:
        while True:
            conn, _ = listener.accept()
            with conn:
                if not answer(conn, service, key):
                    return
    finally:
        service.close()


def answer(conn: socket.socket, service: SimService, key: str) -> bool:
    """Serve one client; False once it asked the server to stop. A first message other than `hello` with the
    right key ends the connection."""
    try:
        first = protocol.recv(conn)
    except (EOFError, ValueError):
        return True
    if first.get("cmd") != "hello" or not hmac.compare_digest(str(first.get("key", "")), key):
        protocol.send(conn, {"error": "PermissionError: wrong or missing key"})
        print("refused a client without the right key", flush=True)
        return True
    protocol.send(conn, service.handle(first))
    while True:
        try:
            msg = protocol.recv(conn)
        except EOFError:
            return True
        if msg["cmd"] in ("bye", "close"):
            protocol.send(conn, {})
            return msg["cmd"] == "bye"
        try:
            reply = service.handle(msg)
        except Exception as exc:  # report to the client instead of dying
            traceback.print_exc()
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        protocol.send(conn, reply)


def listen(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    return sock


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--host", default="0.0.0.0", help="inside the container; only mapmad-net reaches it")
    p.add_argument("--gpu", type=int, default=0, help="GPU that renders")
    args = p.parse_args()
    key = protocol.auth_key()
    with listen(args.host, args.port) as listener:
        print(f"ready on port {args.port}, GPU {args.gpu}", flush=True)
        serve(listener, SimService(limo_factory(args.gpu)), key)


if __name__ == "__main__":
    main()
