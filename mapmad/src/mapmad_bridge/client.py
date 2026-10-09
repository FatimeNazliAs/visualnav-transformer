"""Client side of the simulator bridge: talks to mapmad_sim.sidecar_server over mapmad-net (see protocol.py).

    with SimClient("naz_mapmad_habitat", 5550) as sim:
        first = sim.reset(episode, hfov_deg=66.5, goal_photo=True)
        frame = sim.step(0.2, 0.0)

Needs only numpy (runs in the NoMaD container, Python 3.8).
"""

import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from mapmad_bridge import protocol


@dataclass
class Frame:
    """One observation from the simulator."""

    rgb: np.ndarray  # (H, W, 3) uint8, LIMO's camera
    state: Dict[str, Any]  # position, yaw, geodesic_m, floor_below_feet_m (runner and metrics only, never the policy)
    move: Optional[Dict[str, Any]] = None  # what the last command did (None after reset)
    goal_rgb: Optional[np.ndarray] = None  # goal photo (reset only, if asked for)
    topdown: Optional[Dict[str, Any]] = None  # {"grid": bool (rows z, cols x), "origin": [x, z], "m_per_px"} (reset)
    navmesh_sha256: Optional[str] = None  # floor map of the home (reset)


class SimClient:
    """Connection to one simulator server; retries while the server is still starting."""

    def __init__(self, host: str, port: int, connect_timeout_s: float = 180.0) -> None:
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self.sock = socket.create_connection((host, int(port)), timeout=600)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)
        hello = self._call("hello", key=protocol.auth_key())
        if hello["version"] != protocol.PROTOCOL_VERSION:
            raise RuntimeError(f"server speaks protocol {hello['version']}, client {protocol.PROTOCOL_VERSION}")

    def _call(self, cmd: str, **kwargs: Any) -> Dict[str, Any]:
        protocol.send(self.sock, {"cmd": cmd, **kwargs})
        reply = protocol.recv(self.sock)
        if "error" in reply:
            raise RuntimeError(f"simulator server: {reply['error']}")
        return reply

    def reset(self, episode: Dict[str, Any], hfov_deg: float, goal_photo: bool, topdown_m_per_px: float = 0.05) -> Frame:
        """Load the episode's home (if needed) with this camera FOV and put the robot on its start."""
        r = self._call("reset", episode=episode, hfov_deg=float(hfov_deg), goal_photo=bool(goal_photo),
                       topdown_m_per_px=float(topdown_m_per_px))
        return Frame(rgb=r["rgb"], state=r["state"], goal_rgb=r["goal_rgb"], topdown=r["topdown"],
                     navmesh_sha256=r["navmesh_sha256"])

    def step(self, v: float, w: float) -> Frame:
        """Drive (v m/s, w rad/s) for one control period (0.25 s) and observe."""
        r = self._call("step", v=float(v), w=float(w))
        return Frame(rgb=r["rgb"], state=r["state"], move=r["move"])

    def close(self, shutdown_server: bool = False) -> None:
        """Leave; with shutdown_server the server process exits too."""
        try:
            self._call("close" if shutdown_server else "bye")
        finally:
            self.sock.close()

    def __enter__(self) -> "SimClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
