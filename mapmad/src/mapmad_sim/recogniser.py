"""The recogniser plug-in (Phase 9 block item 7): what draws the map's "target seen here" layer outside the network.

    Recogniser.detect(rgb, depth, category[, semantic]) -> list[Detection(mask | point, score)]

Implementations:
- TruthRecogniser: simulator truth with Phase 8's rule (an instance of the category with >= 200 pixels at depth
  (0, 4] m; its mask = those pixels; score 1.0), instances = the home's ObjectNav goal ids of the category, exactly as
  Phase 8's generator (target_layer.detect). Needs the semantic picture.
- SidecarRecogniser: a Phase 9 model in naz_mapmad_recog (mapmad_recog.server) over mapmad-net. It asks the model's
  word for the category (photo_labels.PROMPTS), keeps detections with score >= the model's threshold t (tuned on
  half A) and maps a kept score s to 0.4 + 0.6 * clip((s - t) / (p99 - t), 0, 1) (p99 = 99th percentile of the
  model's kept scores on half A); a VLM's score 1 -> 1.0. The raw score is kept too.
The layer: each detection's cells = the storey cells under its mask pixels (target_layer.frame_cells: pixels with
depth (0, 4] m above the floor limit), or under its point's pixel for point models; a cell's score in a frame = the
maximum over the detections covering it; then map_v2.TargetFusion fuses over views (Phase 8 code, unchanged).
"""

import socket
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from mapmad_bridge import detections, protocol
from mapmad_sim.photo_labels import PROMPTS
from mapmad_sim.target_layer import FrameCells, NoiseSpec, detect as truth_detect, max_into


@dataclass
class Detection:
    """One detection in one frame. score = the value drawn on the map; raw_score = the model's own score."""

    score: float
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # (H, W) bool
    point: Optional[Tuple[float, float]] = None  # pixel x, y (point models)
    raw_score: float = 1.0
    box: Optional[Tuple[float, float, float, float]] = None
    instance: int = -1  # truth only: the semantic id


class Recogniser(Protocol):
    name: str

    def detect(self, rgb: np.ndarray, depth: np.ndarray, category: str,
               semantic: Optional[np.ndarray] = None) -> List[Detection]:
        ...


def map_score(s: float, t: float, p99: float) -> float:
    """Kept raw score s (>= t) -> map value in [0.4, 1.0] (module docstring)."""
    if p99 <= t:
        return 1.0
    return 0.4 + 0.6 * float(np.clip((s - t) / (p99 - t), 0.0, 1.0))


class TruthRecogniser:
    """Simulator truth, Phase 8's rule and instance ids (module docstring)."""

    name = "truth"

    def __init__(self, ids_by_category: Dict[str, Sequence[int]], spec: NoiseSpec = NoiseSpec()) -> None:
        self.ids = {c: sorted(int(i) for i in v) for c, v in ids_by_category.items()}
        self.spec = spec

    def detect(self, rgb: np.ndarray, depth: np.ndarray, category: str,
               semantic: Optional[np.ndarray] = None) -> List[Detection]:
        if semantic is None:
            raise ValueError("TruthRecogniser needs the semantic picture")
        found = truth_detect(semantic, depth, self.ids.get(category, []), self.spec)
        out = []
        for inst in sorted(found):
            mask = np.zeros(depth.size, dtype=bool)
            mask[found[inst]] = True
            out.append(Detection(1.0, mask.reshape(depth.shape), instance=inst))
        return out


class SidecarRecogniser:
    """A Phase 9 model behind the recogniser sidecar (module docstring). Use as a context manager."""

    def __init__(self, name: str, host: str, port: int, threshold: float, p99: float, vlm: bool, key: str,
                 timeout_s: float = 120.0) -> None:
        self.name, self.threshold, self.p99, self.vlm = name, float(threshold), float(p99), vlm
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        protocol.send(self.sock, {"cmd": "hello", "key": key})
        self.info = self._reply()
        self.last: Dict = {}

    def _reply(self) -> Dict:
        msg = protocol.recv(self.sock)
        if "error" in msg:
            raise RuntimeError(f"recogniser sidecar: {msg['error']}")
        return msg

    def request(self, msg: Dict) -> Dict:
        protocol.send(self.sock, msg)
        return self._reply()

    def detect(self, rgb: np.ndarray, depth: np.ndarray, category: str,
               semantic: Optional[np.ndarray] = None) -> List[Detection]:
        return self.detect_prompt(rgb, PROMPTS[category])

    def detect_prompt(self, rgb: np.ndarray, prompt: str) -> List[Detection]:
        """detect() with the word given as is (Phase 12: OVON names, or PROMPTS[category] resolved by the caller)."""
        r = self.request({"cmd": "detect", "rgb": np.ascontiguousarray(rgb, np.uint8), "prompt": prompt,
                          "min_score": 0.0 if self.vlm else self.threshold})
        self.last = r
        return unpack(r, rgb.shape[:2], lambda s: 1.0 if self.vlm else map_score(s, self.threshold, self.p99),
                      0.0 if self.vlm else self.threshold)

    def close(self) -> None:
        try:
            protocol.send(self.sock, {"cmd": "bye"})
            protocol.recv(self.sock)
        except (OSError, EOFError):
            pass
        self.sock.close()

    def __enter__(self) -> "SidecarRecogniser":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def connect(name: str, host: str, port: int, threshold: float, p99: float, vlm: bool, key: str,
            wait_s: float = 1800.0) -> SidecarRecogniser:
    """SidecarRecogniser, retrying every 5 s while the server is still loading its model (up to wait_s)."""
    deadline = time.time() + wait_s
    while True:
        try:
            return SidecarRecogniser(name, host, port, threshold, p99, vlm, key, timeout_s=600)
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(5)


def unpack(reply: Dict, shape: Tuple[int, int], score_of, min_score: float = 0.0) -> List[Detection]:
    """Sidecar reply arrays -> Detections with score >= min_score (raw score kept)."""
    masks = reply.get("masks")
    out = []
    for k, s in enumerate(np.asarray(reply["scores"], np.float64)):
        if s < min_score:
            continue
        m = detections.unpack_mask(masks[k], shape) if masks is not None else None
        b = detections.present(reply["boxes"][k])
        p = detections.present(reply["points"][k])
        out.append(Detection(score_of(float(s)), m, None if p is None else (float(p[0]), float(p[1])),
                             float(s), None if b is None else tuple(float(v) for v in b)))
    return out


def detection_cells(det: Detection, cells: FrameCells, shape: Tuple[int, int]) -> np.ndarray:
    """Storey cells under a detection: its mask pixels, or its point's pixel (point models), through the frame's
    pixel -> cell table (only pixels with depth (0, 4] m above the floor limit have a cell)."""
    h, w = shape
    if det.mask is not None:
        px = np.flatnonzero(det.mask.reshape(-1))
    elif det.point is not None:
        x, y = int(round(det.point[0])), int(round(det.point[1]))
        px = np.array([min(max(y, 0), h - 1) * w + min(max(x, 0), w - 1)], dtype=np.int64)
    else:
        return np.zeros(0, np.int64)
    c = cells.pixel_cell[px]
    return np.unique(c[c >= 0])


def frame_scores(dets: Sequence[Detection], cells: FrameCells, shape: Tuple[int, int]) -> Dict[int, float]:
    """Cell -> score of one frame (maximum over the detections covering the cell), as target_layer.frame_scores."""
    scores: Dict[int, float] = {}
    for d in dets:
        max_into(scores, detection_cells(d, cells, shape), float(d.score))
    return scores
