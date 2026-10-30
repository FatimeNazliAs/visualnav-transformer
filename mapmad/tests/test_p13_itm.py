"""Phase 13 BLIP-2 ITM server (naz_mapmad_recog): LAVIS's caption processor, the protocol round trip with a stand-in
model, and a GPU smoke test of the real model (skipped without torch + CUDA + the cached weights).

    docker exec -w /app/visualnav-transformer -e HF_HOME=/model_cache/hf -e CUDA_VISIBLE_DEVICES=1 naz_mapmad_recog \
        python -m pytest mapmad/tests/test_p13_itm.py -q
"""

import os
import socket
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml

from mapmad_recog.itm_server import CONFIG, ITMServer, ITMService, pre_caption


def test_pre_caption_is_lavis():
    assert pre_caption("Seems like there is a couch ahead.") == "seems like there is a couch ahead"
    assert pre_caption("A  (tv)!  monitor;") == "a tv monitor"
    assert pre_caption(" ".join(["w"] * 60)) == " ".join(["w"] * 50)


class Stand:
    spec = {"hf": "stand-in", "revision": "0", "dtype": "float32"}

    def cosine(self, rgb, text):
        return float(rgb.mean()) / 255.0 + len(text) / 1000.0


def test_server_round_trip_two_clients(monkeypatch):
    from mapmad_bridge import protocol

    monkeypatch.setenv("MAPMAD_SIDECAR_KEY", "ab12")
    server = ITMServer(("127.0.0.1", 0), ITMService(Stand()), "ab12")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    socks = [socket.create_connection(("127.0.0.1", port)) for _ in range(2)]  # two pairs share one server
    for s in socks:
        protocol.send(s, {"cmd": "hello", "key": "ab12"})
        assert protocol.recv(s)["model"] == "stand-in"
    rgb = np.full((4, 4, 3), 51, np.uint8)
    for s in socks:
        protocol.send(s, {"cmd": "itm", "rgb": rgb, "text": "abcd"})
        assert protocol.recv(s)["cosine"] == pytest.approx(0.2 + 0.004)
    protocol.send(socks[0], {"cmd": "stats"})
    assert protocol.recv(socks[0])["calls"] == 2
    bad = socket.create_connection(("127.0.0.1", port))
    protocol.send(bad, {"cmd": "hello", "key": "wrong"})
    assert "error" in protocol.recv(bad)
    for s in socks + [bad]:
        s.close()
    server.shutdown()
    server.server_close()


def _weights_ok() -> bool:
    try:
        import torch
    except ImportError:
        return False
    spec = yaml.safe_load(Path(CONFIG).read_text())["blip2_itm"]
    snap = Path(os.environ.get("HF_HOME", "/model_cache/hf")) / "hub" / \
        f"models--{spec['hf'].replace('/', '--')}" / "snapshots" / spec["revision"] / "model.safetensors"
    return torch.cuda.is_available() and snap.exists()


@pytest.mark.skipif(not _weights_ok(), reason="needs torch + CUDA + the cached BLIP-2 weights")
def test_real_model_smoke():
    from mapmad_recog.itm_server import ITMModel

    m = ITMModel(yaml.safe_load(Path(CONFIG).read_text())["blip2_itm"], "cuda:0")
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    a = m.cosine(rgb, "Seems like there is a couch ahead.")
    assert -1.0 <= a <= 1.0
    assert m.cosine(rgb, "Seems like there is a couch ahead.") == a  # deterministic on one GPU
