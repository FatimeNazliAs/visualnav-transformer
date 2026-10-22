"""Phase 9 photo test, client side (naz_mapmad_habitat_p9): sends query items through the recogniser sidecar and saves
the raw outputs (block items 4, 5; task 9.4). The server (mapmad_recog.server) holds one model on one GPU.

    python mapmad/scripts/p9_photo_test.py raw    --model owlv2 --port 5560        # every item, min score 0.01
    python mapmad/scripts/p9_photo_test.py timing --model owlv2 --port 5560 --threshold 0.23

raw: the 6-category items, plus the OVON items for open-vocabulary models (not RF-DETR), in item-id order; top 10
detections per item with score >= 0.01, with masks (MobileSAM for box models). Saved in chunks of 250 items:
<outputs>/p9_recognisers/photo_test/<model>/raw/part_<k>.npz (+ part_<k>.json: VLM answers; format and timing
fields: mapmad_bridge.detections), resumable.
Peak VRAM after the pass -> raw/stats.json.
timing: the driving setting (min score = the model's tuned threshold) on 350 seeded half-B 6-category items; the first
50 calls are warm-up and not counted. Round trip per item = client send -> reply (includes MobileSAM and the
socket); -> timing.json. VLMs have no threshold: their raw pass is already the driving setting, so their latency
comes from the raw pass (half-B 6-category items, the first 50 calls of the pass excluded).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapmad_sim import config  # noqa: E402
from mapmad_sim.photo_frames import write_json_atomic  # noqa: E402
from mapmad_sim.recogniser import SidecarRecogniser, connect  # noqa: E402
from mapmad_bridge import detections, protocol  # noqa: E402

CHUNK = 250
WARMUP = 50
TIMING_ITEMS = 350


def out_root() -> Path:
    return config.paths()["outputs"] / "p9_recognisers"


def load_items(kind: str) -> List[Dict[str, Any]]:
    frames = out_root() / "frames"
    items = json.loads((frames / "items_six.json").read_text())
    if kind != "fixed":
        items += json.loads((frames / "items_ovon.json").read_text())
    return sorted(items, key=lambda i: i["item_id"])


class Images:
    """RGB pictures of the frame set, read once."""

    def __init__(self) -> None:
        self.dir = out_root() / "frames" / "img"
        self.cache: Dict[str, np.ndarray] = {}

    def rgb(self, frame_id: str) -> np.ndarray:
        if frame_id not in self.cache:
            self.cache[frame_id] = np.ascontiguousarray(cv2.imread(str(self.dir / f"{frame_id}_rgb.png"))[..., ::-1])
        return self.cache[frame_id]


EMPTY = {"scores": np.zeros(0, np.float32), "boxes": np.zeros((0, 4), np.float32), "points": np.zeros((0, 2), np.float32),
         "model_ms": float("nan"), "sam_ms": float("nan"), "text": ""}


def call(client: SidecarRecogniser, rgb: np.ndarray, prompt: str, min_score: float):
    """One query item; a server-side error is recorded as an item without detections (error text kept)."""
    t0 = time.perf_counter()
    try:
        r = client.request({"cmd": "detect", "rgb": rgb, "prompt": prompt, "min_score": min_score})
    except RuntimeError as exc:
        r = {**EMPTY, "text": f"ERROR {exc}"}
    return r, (time.perf_counter() - t0) * 1e3


def cmd_raw(a: argparse.Namespace, client: SidecarRecogniser) -> None:
    folder = out_root() / "photo_test" / a.model / a.raw_dir
    folder.mkdir(parents=True, exist_ok=True)
    items = load_items(client.info["kind"])
    imgs = Images()
    client.request({"cmd": "reset_stats"})
    hw = detections.FRAME_HW
    for k in range(0, (len(items) + CHUNK - 1) // CHUNK):
        if (folder / f"part_{k:03d}.npz").exists():
            continue
        rows = []
        for it in items[k * CHUNK:(k + 1) * CHUNK]:
            rgb = imgs.rgb(it["frame_id"])
            hw = rgb.shape[:2]
            r, ms = call(client, rgb, it["prompt"], a.min_score)
            rows.append({**r, "item_id": it["item_id"], "rtt_ms": ms})
        detections.save_part(folder, k, rows, hw)
        print(f"{a.model}: chunk {k} done ({min((k + 1) * CHUNK, len(items))}/{len(items)})", flush=True)
    errors = sum(sum(t.startswith("ERROR") for t in json.loads(f.read_text()).values()) for f in folder.glob("part_*.json"))
    write_json_atomic(folder / "stats.json", {**client.request({"cmd": "stats"}), "items": len(items), "errors": errors,
                                              "info": client.info, "chunks": (len(items) + CHUNK - 1) // CHUNK})


def timing_items(seed: int) -> List[Dict[str, Any]]:
    items = [i for i in json.loads((out_root() / "frames" / "items_six.json").read_text()) if i["half"] == "B"]
    items.sort(key=lambda i: i["item_id"])
    pick = np.random.default_rng(seed).permutation(len(items))[:TIMING_ITEMS]
    return [items[k] for k in pick]


def cmd_timing(a: argparse.Namespace, client: SidecarRecogniser) -> None:
    folder = out_root() / "photo_test" / a.model
    imgs = Images()
    client.request({"cmd": "reset_stats"})
    rows = []
    for it in timing_items(a.seed):
        r, ms = call(client, imgs.rgb(it["frame_id"]), it["prompt"], a.threshold)
        rows.append({"item_id": it["item_id"], "rtt_ms": ms, "model_ms": r["model_ms"], "sam_ms": r["sam_ms"],
                     "kept": int(len(r["scores"]))})
    rtt = np.array([r["rtt_ms"] for r in rows[WARMUP:]])
    write_json_atomic(folder / "timing.json", {
        "threshold": a.threshold, "items": len(rows), "warmup": WARMUP, "median_ms": float(np.median(rtt)),
        "p95_ms": float(np.percentile(rtt, 95)), **client.request({"cmd": "stats"}), "rows": rows})
    print(f"{a.model}: timing median {np.median(rtt):.1f} ms, p95 {np.percentile(rtt, 95):.1f} ms", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["raw", "timing"])
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="naz_mapmad_recog")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--min-score", type=float, default=0.01)
    p.add_argument("--threshold", type=float)
    p.add_argument("--raw-dir", default="raw", help="raw_edge: p9_rules_v2.md A3 extra pass (score >= 0.001)")
    p.add_argument("--seed", type=int, default=20261021)
    a = p.parse_args()
    with connect(a.model, a.host, a.port, 0.0, 1.0, False, protocol.auth_key()) as client:
        if client.info["model"] != a.model:
            raise SystemExit(f"server holds {client.info['model']}, not {a.model}")
        cmd_raw(a, client) if a.step == "raw" else cmd_timing(a, client)


if __name__ == "__main__":
    main()
