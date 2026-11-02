"""Phase 14a stop evidence of one closed-loop run (simulator side; search_v2 setting `log_evidence`): everything the
offline stop bench (14.1), the Qwen check (14.2), the diagnosis (14.3) and the mark check (14.5) need, recorded while
the robot drives, so no depth replay is needed (the Phase 13 dets files hold boxes but no masks).

Per frame t (after search_v2.SearchMap.add_frame, from the same arrays the fusion used):
- pose (x, y, yaw of the turning centre, 2D frame);
- view cells: the storey cells the frame saw (target_layer.FrameCells.view; every one counts as a look of the fused
  layer, map_v2.TargetFusion);
- recogniser detections: raw score, map value, mask pixels (the "size" of the look), box, and the storey cells under
  the mask (recogniser.detection_cells), each with its distance to the robot;
- true targets: every instance of the episode's category in the semantic picture (video_events.true_boxes: id, box,
  pixels) and the storey cells under its pixels.
At the end (`arrays`): the whole-storey first-seen maps (obstacle, explored: frame of first sight, -1 = never) and the
bump events, so the robot's own map at any step can be rebuilt (14.3), plus the map's origin, shape and resolution.
Written by the server as one compressed npz per run (`save_evidence`). Only records: nothing here changes a run.
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from mapmad_sim.recogniser import detection_cells
from mapmad_sim.video_events import true_boxes

EVIDENCE_VERSION = 1


class EvidenceRecorder:
    """Collects one run's per-frame evidence (module docstring)."""

    def __init__(self, shape: Tuple[int, int], origin: Sequence[float], res: float, category_ids: Sequence[int]) -> None:
        self.shape, self.origin, self.res = (int(shape[0]), int(shape[1])), np.asarray(origin, np.float64), float(res)
        self.ids = sorted(int(i) for i in category_ids)
        self.pose: List[Tuple[float, float, float]] = []
        self.view: List[np.ndarray] = []
        self.det_rows: List[Tuple[int, float, float, int, float, float, float, float]] = []  # t, raw, value, px, box
        self.det_cells: List[np.ndarray] = []
        self.true_rows: List[Tuple[int, int, int, float, float, float, float]] = []  # t, id, px, box
        self.true_cells: List[np.ndarray] = []

    def cell_distance(self, cells: np.ndarray, pose: Sequence[float]) -> np.ndarray:
        """Distance (m) from the pose to the centre of each flat cell."""
        i, j = np.divmod(cells.astype(np.int64), self.shape[1])
        cx, cy = self.origin[0] + (i + 0.5) * self.res, self.origin[1] + (j + 0.5) * self.res
        return np.hypot(cx - pose[0], cy - pose[1])

    def add(self, t: int, pose: Sequence[float], cells: Any, dets: Sequence[Any], semantic: np.ndarray) -> None:
        """Frame t: pose, the frame's FrameCells, the recogniser detections drawn into the layer, the semantic picture."""
        if t != len(self.pose):
            raise ValueError(f"evidence frame {t} out of order (expected {len(self.pose)})")
        self.pose.append((float(pose[0]), float(pose[1]), float(pose[2])))
        self.view.append(np.asarray(cells.view, np.int32))
        hw = semantic.shape[:2]
        for d in dets:
            c = detection_cells(d, cells, hw)
            px = int(d.mask.sum()) if d.mask is not None else 1
            box = d.box if d.box is not None else (np.nan,) * 4
            self.det_rows.append((t, float(d.raw_score), float(d.score), px, *[float(v) for v in box]))
            self.det_cells.append(c.astype(np.int32))
        flat = semantic.reshape(-1)
        for tb in true_boxes(semantic, self.ids):
            px_idx = np.flatnonzero(flat == tb["id"])
            c = cells.pixel_cell[px_idx]
            c = np.unique(c[c >= 0])
            self.true_rows.append((t, int(tb["id"]), int(tb["px"]), *[float(v) for v in tb["box"]]))
            self.true_cells.append(c.astype(np.int32))

    def arrays(self, first_seen: Dict[str, np.ndarray], bumps: Sequence[Tuple[int, float, float]]) -> Dict[str, np.ndarray]:
        """Everything as flat arrays (ragged lists as values + offsets)."""
        def ragged(parts: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            ptr = np.zeros(len(parts) + 1, np.int64)
            ptr[1:] = np.cumsum([p.size for p in parts]) if parts else []
            return (np.concatenate(parts).astype(np.int32) if parts else np.zeros(0, np.int32)), ptr

        view, view_ptr = ragged(self.view)
        dc, dc_ptr = ragged(self.det_cells)
        tc, tc_ptr = ragged(self.true_cells)
        det = np.asarray(self.det_rows, np.float64).reshape(-1, 8)
        tru = np.asarray(self.true_rows, np.float64).reshape(-1, 7)
        bump = np.asarray(bumps, np.float64).reshape(-1, 3)
        return {"version": np.array(EVIDENCE_VERSION), "shape": np.asarray(self.shape, np.int64), "origin": self.origin,
                "res": np.array(self.res), "pose": np.asarray(self.pose, np.float64).reshape(-1, 3),
                "view": view, "view_ptr": view_ptr,
                "det_t": det[:, 0].astype(np.int32), "det_raw": det[:, 1], "det_value": det[:, 2],
                "det_px": det[:, 3].astype(np.int32), "det_box": det[:, 4:8], "det_cells": dc, "det_cells_ptr": dc_ptr,
                "true_t": tru[:, 0].astype(np.int32), "true_id": tru[:, 1].astype(np.int32),
                "true_px": tru[:, 2].astype(np.int32), "true_box": tru[:, 3:7], "true_cells": tc, "true_cells_ptr": tc_ptr,
                "first_obstacle": np.asarray(first_seen["obstacle"], np.int32),
                "first_explored": np.asarray(first_seen["explored"], np.int32), "bumps": bump}

    def save(self, path: Path, first_seen: Dict[str, np.ndarray], bumps: Sequence[Tuple[int, float, float]]) -> int:
        """Write the compressed npz atomically (tmp + rename); returns its size in bytes."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez_compressed(buf, **self.arrays(first_seen, bumps))
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path)
        return len(buf.getvalue())


def load(path: Path) -> Dict[str, np.ndarray]:
    """One run's evidence arrays (read side: stop bench, diagnosis, mark check)."""
    with np.load(Path(path)) as z:
        return {k: z[k] for k in z.files}


def ragged_rows(values: np.ndarray, ptr: np.ndarray) -> List[np.ndarray]:
    return [values[ptr[k]:ptr[k + 1]] for k in range(len(ptr) - 1)]
