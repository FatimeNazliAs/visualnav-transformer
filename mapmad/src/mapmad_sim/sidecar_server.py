"""Simulator server for closed-loop runs: the virtual LIMO behind a TCP socket on mapmad-net (protocol in
mapmad_bridge/protocol.py). Runs in naz_mapmad_habitat; the policy runs in naz_mapmad and connects by name.

    MAPMAD_SIDECAR_KEY=<hex> python -m mapmad_sim.sidecar_server --port 5550 --gpu 0

Listens on all interfaces of the container, which only mapmad-net reaches (never publish the port with -p).
One client at a time; after `bye` the next client may connect, `close` ends the server.
"""

import argparse
import hmac
import json
import math
import socket
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from mapmad_bridge import protocol

RobotFactory = Callable[..., Any]  # (split, home, hfov_deg[, depth=True][, camera={...}][, semantic=True]) -> LimoSim-like


def limo_factory(gpu: int, seed: int = 0) -> RobotFactory:
    """Robots built from configs/robot_limo.yaml (LIMO-sized floor map) with the requested camera FOV; depth=True
    adds the depth camera the online map needs (Phase 5); camera = a Phase 7 camera setting (CameraSetting config
    dict with its "name": lens, height, pitch, floor map); semantic=True adds the semantic camera (Phase 11 search
    map: simulator-truth detections)."""
    from mapmad_sim import config
    from mapmad_sim.camera_setting import CameraSetting
    from mapmad_sim.robot import LimoSim, RobotSpec

    def make(split: str, home: str, hfov_deg: float, depth: bool = False, camera: Optional[Dict[str, Any]] = None,
             semantic: bool = False) -> Any:
        spec = RobotSpec.from_config(config.robot(), hfov_deg=hfov_deg)
        if camera is not None:
            spec = CameraSetting.from_wire(camera).apply(spec)
        return LimoSim(split, home, spec, semantic=semantic, depth=depth, gpu=gpu, seed=seed)

    return make


class SimService:
    """Answers protocol requests; keeps one home loaded and reloads it only when home, camera FOV, the need for a
    depth camera or the Phase 7 camera setting (`camera` at reset; absent = robot_limo.yaml's camera) change.

    Phase 5 (`outputs` at reset): "map" -> every reply carries the policy's 3 x 64 x 64 map tensor ("map"), "grid"
    -> the whole-storey grids for the planner ("grid", plus "goal_rect" at reset), both from mapmad_sim.online_map
    with `map_settings` = {"builder": p2_datagen.yaml `map`, "crop": the train config's `map`}. Episodes with
    `target.view_points` (snapped navmesh points) also get the walking distance to the nearest one in `state`
    ("vp_geodesic_m", the Phase 5 primary metric).

    Phase 11 (`outputs` = {"search_v2": settings}, settings from search_v2.search_settings + "noise_seed" + "record"):
    the robot gets depth and semantic cameras; every reply carries the v2 map input "maps" [2, 4, 64, 64]
    (search_v2.SearchMap, simulator-truth target layer with the Phase 8 noise seeded by noise_seed) and `state` the
    truth "in_view" / "first_in_view_step" / "seen_instances"; `save` writes a recorded episode in the Phase 8 drive
    format. The episode's `target.instances` and `target.category_ids` say what counts as the target.
    """

    def __init__(self, factory: RobotFactory) -> None:
        self.factory = factory
        self.robot: Optional[Any] = None
        self.key: Optional[Tuple[str, str, float, bool, Optional[str]]] = None
        self.target = None
        self.view_points: Optional[Any] = None
        self.online_map: Optional[Any] = None
        self.search: Optional[Any] = None  # Phase 11: search_v2.SearchMap of the current episode
        self.episode: Optional[Dict[str, Any]] = None
        self.outputs: Dict[str, Any] = {}

    def handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        cmd = msg["cmd"]
        if cmd == "hello":
            return {"version": protocol.PROTOCOL_VERSION, "home": self.key[1] if self.key else None,
                    "hfov_deg": self.key[2] if self.key else None}
        if cmd == "reset":
            return self.reset(msg["episode"], msg["hfov_deg"], msg["goal_photo"], msg["topdown_m_per_px"],
                              msg.get("outputs") or {}, msg.get("camera"))
        if cmd == "step":
            if self.robot is None or self.target is None:
                raise RuntimeError("step before reset")
            move = self.robot.drive(msg["v"], msg["w"])
            if self.search is not None:
                self.search.after_move(move, self.pose2d(), self.robot.spec.dt)
            return {**self.observation(), "state": self.state(), "move": asdict(move)}
        if cmd == "save":
            return self.save(msg["folder"], msg.get("meta") or {})
        raise ValueError(f"unknown command {cmd!r}")

    def reset(self, episode: Dict[str, Any], hfov_deg: float, goal_photo: bool, m_per_px: float,
              outputs: Dict[str, Any], camera: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        searching = bool(outputs.get("search_v2"))
        depth = bool(outputs.get("map") or outputs.get("grid")) or searching
        camera_key = json.dumps(camera, sort_keys=True) if camera is not None else None
        key = (episode["split"], episode["home"], float(hfov_deg), depth, camera_key, searching)
        if key != self.key:
            self.close()
            kwargs: Dict[str, Any] = {"depth": True} if depth else {}
            if camera is not None:
                kwargs["camera"] = camera
            if searching:
                kwargs["semantic"] = True
            self.robot = self.factory(*key[:3], **kwargs)
            self.key = key
        self.target = episode["target"]["point"]
        self.view_points = episode["target"].get("view_points")
        self.outputs = outputs
        goal_rgb = None
        if goal_photo:  # rendered with this arm's camera (a 120 deg photo for the 120 deg arms), same pose
            self.robot.place(episode["goal_photo_position"], episode["goal_photo_yaw"])
            goal_rgb = self.robot.observe()["rgb"]
        self.robot.place(episode["start_position"], episode["start_yaw"])
        self.online_map = None
        self.search = None
        self.episode = episode
        if searching:
            from mapmad_sim.search_v2 import SearchMap, SearchSpec, instances_of

            sv = outputs["search_v2"]
            self.search = SearchMap(self.robot, SearchSpec.from_wire(sv), episode["target"]["category_ids"],
                                    instances_of(episode), np.random.default_rng(sv["noise_seed"]),
                                    record=bool(sv.get("record")))
        extra: Dict[str, Any] = {}
        if outputs.get("map") or outputs.get("grid"):  # Phase 5 online map (v1 tensor / planner grid)
            from mapmad_sim.depth_map import MapSpec
            from mapmad_sim.online_map import OnlineMap

            settings = outputs["map_settings"]
            self.online_map = OnlineMap(self.robot, episode, MapSpec.from_config(settings["builder"]), settings["crop"])
            if outputs.get("grid"):
                extra["goal_rect"] = list(self.online_map.goal_rect())
        grid, origin = self.robot.topdown(m_per_px, float(episode["start_position"][1]))
        return {**self.observation(), **extra, "goal_rgb": goal_rgb, "state": self.state(),
                "topdown": {"grid": grid, "origin": [float(origin[0]), float(origin[1])], "m_per_px": m_per_px},
                "navmesh_sha256": self.robot.navmesh_sha256}

    def observation(self) -> Dict[str, Any]:
        """The picture at the current pose; with an online map also its new frame and the requested map outputs."""
        obs = self.robot.observe()
        out: Dict[str, Any] = {"rgb": obs["rgb"]}
        if self.online_map is not None:
            self.online_map.add_frame(self.robot, obs["depth"])
            if self.outputs.get("map"):
                out["map"] = self.online_map.tensor(self.robot)
            if self.outputs.get("grid"):
                out["grid"] = self.online_map.grid(self.robot)
        if self.search is not None:
            self.search.add_frame(self.robot, obs)
            out["maps"] = self.search.windows(self.pose2d())
        return out

    def pose2d(self) -> Tuple[float, float, float]:
        """(X, Y, yaw) of the turning centre in the 2D frame of traj_data.pkl."""
        from mapmad_sim.frames import to_2d

        x, y = to_2d(self.robot.position)
        return float(x), float(y), float(self.robot.yaw)

    def save(self, folder: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 11: write the recorded search episode (Phase 8 drive format) to `folder` (must not exist)."""
        if self.search is None:
            raise RuntimeError("save without a search episode")
        e = self.episode
        full = {**meta, "home": e["home"], "split": e["split"], "kind": "search", "category": e["target"]["category"],
                "instances": e["target"]["instances"], "navmesh_sha256": self.robot.navmesh_sha256,
                "floor_y": round(float(np.median(self.search.rec.floor_y)), 4), "student_drive": True,
                "frontier_ignore_trajectory_m": 0.0, "metric_waypoint_spacing": 0.05,
                "frame_convention": "traj_data.pkl: X = -z, Y = -x of Habitat, yaw unchanged (mapmad_sim/frames.py)"}
        return {"bytes": self.search.write(Path(folder), full, e["target"]["category"])}

    def state(self) -> Dict[str, Any]:
        geo = self.robot.geodesic(self.robot.position, self.target)
        out = {"position": [float(x) for x in self.robot.position], "yaw": float(self.robot.yaw),
               "geodesic_m": geo if math.isfinite(geo) else None,
               "floor_below_feet_m": float(self.robot.floor_below_feet)}
        if self.view_points:
            vp = self.robot.geodesic_to_any(self.robot.position, self.view_points)
            out["vp_geodesic_m"] = vp if math.isfinite(vp) else None
        if self.search is not None:
            out.update(in_view=self.search.in_view, first_in_view_step=self.search.first_in_view,
                       seen_instances=sorted(self.search.seen))
        return out

    def close(self) -> None:
        if self.robot is not None:
            self.robot.close()
        self.robot, self.key, self.target, self.view_points, self.online_map = None, None, None, None, None
        self.search, self.episode = None, None


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
