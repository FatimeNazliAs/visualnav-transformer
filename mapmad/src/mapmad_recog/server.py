"""Recogniser sidecar: one Phase 9 recogniser (+ MobileSAM for box models) behind a TCP socket on mapmad-net, in
naz_mapmad_recog. The Habitat side (mapmad_sim.recogniser.SidecarRecogniser) connects by container name.

    docker exec -e CUDA_VISIBLE_DEVICES=1 -e MAPMAD_SIDECAR_KEY=<hex> -e HOME=/model_cache/home naz_mapmad_recog \
        python -m mapmad_recog.server --model grounding_dino --port 5560

Wire format = mapmad_bridge.protocol (length-prefixed JSON + raw arrays; first message `hello` with the shared key;
never publish the port with -p). Requests:
- hello(key)                                 -> {"model", "name", "family", "kind", "output"}
- detect(rgb (H, W, 3) uint8, prompt[, min_score][, max_det])
      -> {"scores" (N,) f32, "boxes" (N, 4) f32 (NaN rows for point detections), "points" (N, 2) f32 (NaN rows
          for box / mask detections), "masks" (N, ceil(H*W/8)) uint8 packed bits or absent, "model_ms", "sam_ms",
          "text" (VLM answer; "" otherwise)}
  min_score: raw pass = the config floor (0.01); driving / timing pass = the model's tuned threshold.
- stats -> {"peak_vram_mb", "reserved_vram_mb"};  reset_stats -> {}
- bye -> {} (client leaves);  close -> {} (server exits)
"""

import argparse
import hmac
import socket
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from mapmad_bridge import protocol
from mapmad_recog.models import MaskMaker, RawDet, build

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "p9_models.yaml"


def pack(dets: List[RawDet], shape) -> Dict[str, np.ndarray]:
    """RawDets -> the reply arrays (module docstring)."""
    n = len(dets)
    out = {"scores": np.array([d.score for d in dets], np.float32),
           "boxes": np.full((n, 4), np.nan, np.float32), "points": np.full((n, 2), np.nan, np.float32)}
    for k, d in enumerate(dets):
        if d.box is not None:
            out["boxes"][k] = d.box
        if d.point is not None:
            out["points"][k] = d.point
    if n and all(d.mask is not None for d in dets):
        out["masks"] = np.stack([np.packbits(d.mask.reshape(-1)) for d in dets])
    return out


class RecogService:
    """Holds one recogniser and, for box models, MobileSAM."""

    def __init__(self, key: str, cfg: Dict[str, Any], device: str = "cuda:0") -> None:
        import torch

        self.torch, self.key = torch, key
        self.sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
        self.spec = cfg["models"][key]
        cache = Path(cfg["cache"])
        self.model = build(key, self.spec, cache, device, float(cfg["min_score"]), int(cfg["max_detections"]))
        self.defaults = (float(cfg["min_score"]), int(cfg["max_detections"]))
        self.sam = None
        if self.spec["output"] == "box":
            ms = cfg["mobile_sam"]
            weights = next((cache / "hf").glob(f"hub/models--{ms['hf'].replace('/', '--')}/snapshots/{ms['revision']}/{ms['file']}"))
            self.sam = MaskMaker(weights, device)

    def handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        cmd = msg["cmd"]
        if cmd == "hello":
            return {"model": self.key, **{k: self.spec[k] for k in ("name", "family", "kind", "output")}}
        if cmd == "detect":
            return self.detect(msg)
        cuda = self.torch.cuda.is_available()
        if cmd == "stats":
            return {"peak_vram_mb": self.torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0,
                    "reserved_vram_mb": self.torch.cuda.max_memory_reserved() / 2**20 if cuda else 0.0}
        if cmd == "reset_stats":
            if cuda:
                self.torch.cuda.reset_peak_memory_stats()
            return {}
        raise ValueError(f"unknown command {cmd!r}")

    def detect(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        rgb = np.ascontiguousarray(msg["rgb"], dtype=np.uint8)
        self.model.min_score = float(msg.get("min_score", self.defaults[0]))
        self.model.max_det = int(msg.get("max_det", self.defaults[1]))
        self.sync()
        t0 = time.perf_counter()
        dets = self.model.detect(rgb, str(msg["prompt"]))
        self.sync()
        t1 = time.perf_counter()
        if self.sam is not None and dets:
            for d, m in zip(dets, self.sam.masks(rgb, [d.box for d in dets])):
                d.mask = m
            self.sync()
        t2 = time.perf_counter()
        return {**pack(dets, rgb.shape[:2]), "model_ms": (t1 - t0) * 1e3, "sam_ms": (t2 - t1) * 1e3,
                "text": getattr(self.model, "last_text", "")}


def answer(conn: socket.socket, service: RecogService, key: str) -> bool:
    """Serve one client; False once it asked the server to stop."""
    try:
        first = protocol.recv(conn)
    except (EOFError, ValueError):
        return True
    if first.get("cmd") != "hello" or not hmac.compare_digest(str(first.get("key", "")), key):
        protocol.send(conn, {"error": "PermissionError: wrong or missing key"})
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
        except Exception as exc:  # reported to the client, which records it
            traceback.print_exc()
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        protocol.send(conn, reply)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=5560)
    p.add_argument("--host", default="0.0.0.0", help="inside the container; only mapmad-net reaches it")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--device", default="cuda:0", help="cuda:0 = the GPU in CUDA_VISIBLE_DEVICES; cpu for smoke tests")
    a = p.parse_args()
    key = protocol.auth_key()
    service = RecogService(a.model, yaml.safe_load(a.config.read_text()), a.device)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((a.host, a.port))
    sock.listen(1)
    print(f"ready: {a.model} on port {a.port}", flush=True)
    with sock:
        while True:
            conn, _ = sock.accept()
            with conn:
                if not answer(conn, service, key):
                    return


if __name__ == "__main__":
    main()
