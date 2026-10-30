"""BLIP-2 image-text matching (ITM) sidecar for Phase 13 arm VF (mapmad_sim.search_vf), in naz_mapmad_recog: one server
per GPU, shared by that GPU's simulator servers (one thread per client, one GPU lock).

    docker exec -e CUDA_VISIBLE_DEVICES=1 -e MAPMAD_SIDECAR_KEY=<hex> -e HOME=/model_cache/home \
        -e HF_HOME=/model_cache/hf naz_mapmad_recog python -m mapmad_recog.itm_server --port 5941

Model: configs/p13_vf.yaml `blip2_itm` = Hugging Face Salesforce/blip2-itm-vit-g at a pinned revision,
transformers Blip2ForImageTextRetrieval with use_image_text_matching_head=False: the ITC cosine (max over the 32 query
embeddings of the normalised image-query / text-CLS projections) = LAVIS blip2_image_text_matching "pretrain"
forward(..., match_head="itc"), the quantity VLFM's BLIP2ITM.cosine returns (vlfm/vlm/blip2itm.py:37-54). The text
goes through LAVIS's "eval" caption processor first (`pre_caption`: lower case, . ! " ( ) * # : ; ~ -> space, at most
50 words), as VLFM's text_processors["eval"]; the picture through the HF processor (224 x 224 bicubic, CLIP mean/std,
as LAVIS blip_image_eval). Weights only from the local cache (HF_HOME, local_files_only). Precision: the whole model
fp16 on cuda:0 (LAVIS runs the ViT in fp16 and the Q-Former in fp32). Deterministic inference:
torch.backends.cudnn.deterministic = True, cudnn.benchmark = False; TF32 left at PyTorch's defaults (matmul off, cudnn
on; irrelevant in fp16).

Wire format = mapmad_bridge.protocol (first message `hello` with the shared key MAPMAD_SIDECAR_KEY; never publish the
port with -p). Requests:
- hello(key)                          -> {"model", "revision", "dtype"}
- itm(rgb (H, W, 3) uint8, text)      -> {"cosine": float, "model_ms"}
- stats                               -> {"peak_vram_mb", "reserved_vram_mb", "calls", "mean_ms"}
- bye -> {} (client leaves);  close -> {} (server exits)
"""

import argparse
import hmac
import os
import re
import socket
import socketserver
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from mapmad_bridge import protocol

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "p13_vf.yaml"


def pre_caption(caption: str, max_words: int = 50) -> str:
    """LAVIS BlipCaptionProcessor.pre_caption (the "eval" text processor VLFM's BLIP2ITM uses)."""
    caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption)
    caption = caption.rstrip("\n").strip(" ")
    words = caption.split(" ")
    return " ".join(words[:max_words]) if len(words) > max_words else caption


class ITMModel:
    """BLIP-2 ITC cosine of one picture and one text (module docstring)."""

    def __init__(self, spec: Dict[str, Any], device: str = "cuda:0") -> None:
        import torch
        from transformers import AutoProcessor, Blip2ForImageTextRetrieval

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.torch, self.device, self.spec = torch, device, spec
        self.dtype = getattr(torch, spec["dtype"]) if device.startswith("cuda") else torch.float32
        kw = dict(revision=spec["revision"], local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(spec["hf"], **kw)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(spec["hf"], dtype=self.dtype, **kw).to(device).eval()
        self.max_words = int(spec.get("max_words", 50))

    def cosine(self, rgb: np.ndarray, text: str) -> float:
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(np.ascontiguousarray(rgb, np.uint8)),
                                text=pre_caption(text, self.max_words), return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with self.torch.inference_mode():
            out = self.model(**inputs, use_image_text_matching_head=False)
        return float(out.logits_per_image.float().reshape(-1)[0].item())


class ITMService:
    """Answers protocol requests; one GPU lock for every client thread."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.lock = threading.Lock()
        self.calls, self.total_ms = 0, 0.0
        torch = getattr(model, "torch", None)
        self.cuda = torch is not None and torch.cuda.is_available()
        self.sync = torch.cuda.synchronize if self.cuda else (lambda: None)

    def handle(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        cmd = msg["cmd"]
        if cmd == "hello":
            spec = getattr(self.model, "spec", {})
            return {"model": spec.get("hf"), "revision": spec.get("revision"), "dtype": spec.get("dtype")}
        if cmd == "itm":
            with self.lock:
                self.sync()
                t0 = time.perf_counter()
                c = self.model.cosine(msg["rgb"], str(msg["text"]))
                self.sync()
                ms = (time.perf_counter() - t0) * 1e3
                self.calls += 1
                self.total_ms += ms
                if self.calls % 500 == 0:
                    print(f"{self.calls} calls, mean {self.total_ms / self.calls:.1f} ms", flush=True)
            return {"cosine": c, "model_ms": ms}
        if cmd == "stats":
            t = getattr(self.model, "torch", None)
            return {"peak_vram_mb": t.cuda.max_memory_allocated() / 2**20 if self.cuda else 0.0,
                    "reserved_vram_mb": t.cuda.max_memory_reserved() / 2**20 if self.cuda else 0.0,
                    "calls": self.calls, "mean_ms": self.total_ms / self.calls if self.calls else 0.0}
        raise ValueError(f"unknown command {cmd!r}")


def answer(conn: socket.socket, service: ITMService, key: str) -> bool:
    """Serve one client; False once it asked the server to stop (as mapmad_recog.server.answer)."""
    try:
        first = protocol.recv(conn)
    except (EOFError, ValueError, OSError):
        return True
    if first.get("cmd") != "hello" or not hmac.compare_digest(str(first.get("key", "")), key):
        protocol.send(conn, {"error": "PermissionError: wrong or missing key"})
        return True
    protocol.send(conn, service.handle(first))
    while True:
        try:
            msg = protocol.recv(conn)
        except (EOFError, OSError):
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


class ITMServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr: Any, service: ITMService, key: str) -> None:
        super().__init__(addr, _Handler)
        self.service, self.key = service, key


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        if not answer(self.request, self.server.service, self.server.key):
            threading.Thread(target=self.server.shutdown, daemon=True).start()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=5941)
    p.add_argument("--host", default="0.0.0.0", help="inside the container; only mapmad-net reaches it")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--device", default="cuda:0", help="cuda:0 = the GPU in CUDA_VISIBLE_DEVICES; cpu for smoke tests")
    a = p.parse_args()
    key = protocol.auth_key()
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    spec = yaml.safe_load(a.config.read_text())["blip2_itm"]
    service = ITMService(ITMModel(spec, a.device))
    with ITMServer((a.host, a.port), service, key) as server:
        print(f"ready: blip2_itm {spec['revision'][:12]} on port {a.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
