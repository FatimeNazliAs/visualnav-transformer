"""Wire format between the simulator server (naz_mapmad_habitat, Python 3.9) and its client (naz_mapmad, 3.8).

Transport: TCP on the Docker network mapmad-net (never published with `docker run -p`). No pickle: every message
is length-prefixed JSON plus raw array bytes:

    [4 bytes: header length N, big-endian][N bytes: JSON header][array 0 bytes][array 1 bytes]...

In the JSON body each numpy array is replaced by {"__array__": i}; header["arrays"][i] = {"dtype", "shape",
"nbytes"}. The first request of a connection must be `hello` with the shared key (environment variable
MAPMAD_SIDECAR_KEY, hex); a wrong key gets an error and the connection is closed.

Requests ({"cmd": ...}) and replies:
- hello(key)                             -> {"version", "home", "hfov_deg"}
- reset(episode, hfov_deg, goal_photo, topdown_m_per_px)
                                         -> {"rgb", "goal_rgb" (or None), "state", "topdown", "navmesh_sha256"}
- step(v, w)                             -> {"rgb", "state", "move"}
- bye                                    -> {} (client leaves, server waits for the next one)
- close                                  -> {} (server exits)
Any failure -> {"error": "<type>: <message>"}.
`state` = {"position": [x, y, z], "yaw", "geodesic_m", "floor_below_feet_m"}; `move` = mapmad_sim.robot.Move fields.
"""

import json
import os
import socket
import struct
from typing import Any, Dict, List

import numpy as np

PROTOCOL_VERSION = 2
KEY_ENV = "MAPMAD_SIDECAR_KEY"
MAX_HEADER = 1 << 24  # 16 MB of JSON is far more than any message needs


def auth_key() -> str:
    """The shared key (hex) from MAPMAD_SIDECAR_KEY; raises if it is missing."""
    value = os.environ.get(KEY_ENV)
    if not value:
        raise RuntimeError(f"set {KEY_ENV} (hex) for both the simulator server and its client")
    return value


def encode(message: Dict[str, Any]) -> bytes:
    """Message dict (may hold numpy arrays at any depth) -> bytes on the wire."""
    arrays: List[np.ndarray] = []

    def strip(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            arrays.append(np.ascontiguousarray(value))
            return {"__array__": len(arrays) - 1}
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [strip(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    body = strip(message)
    header = json.dumps({"body": body, "arrays": [{"dtype": a.dtype.str, "shape": list(a.shape), "nbytes": a.nbytes}
                                                  for a in arrays]}).encode()
    return b"".join([struct.pack(">I", len(header)), header] + [a.tobytes() for a in arrays])


def decode(header: Dict[str, Any], payload: bytes) -> Dict[str, Any]:
    """Inverse of encode, given the parsed JSON header and the array bytes that followed it."""
    arrays, offset = [], 0
    for spec in header["arrays"]:
        n = spec["nbytes"]
        arrays.append(np.frombuffer(payload[offset:offset + n], dtype=np.dtype(spec["dtype"])).reshape(spec["shape"]))
        offset += n

    def fill(value: Any) -> Any:
        if isinstance(value, dict):
            return arrays[value["__array__"]] if set(value) == {"__array__"} else {k: fill(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fill(v) for v in value]
        return value

    return fill(header["body"])


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            raise EOFError("connection closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def send(sock: socket.socket, message: Dict[str, Any]) -> None:
    sock.sendall(encode(message))


def recv(sock: socket.socket) -> Dict[str, Any]:
    (n,) = struct.unpack(">I", recv_exact(sock, 4))
    if n > MAX_HEADER:
        raise ValueError(f"header of {n} bytes refused")
    header = json.loads(recv_exact(sock, n))
    payload = recv_exact(sock, sum(a["nbytes"] for a in header["arrays"]))
    return decode(header, payload)
