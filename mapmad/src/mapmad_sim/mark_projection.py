"""Phase 14b-1 fix F1: where a detection's mask lands on the storey map, inside the stop module only (kickoff §3 item 2,
p14b1_rules.md §3). The network's own map input is not touched.

Today (base, Phases 12-14a): every mask pixel goes through its depth onto a 10 cm storey cell
(target_layer.frame_cells -> recogniser.detection_cells). Pixels on the object's rim really hit the wall behind it,
so one box marks a wedge shaped like the camera's view cone, 1-4 m long (14a smear note).

Variants of one mask (all use the frame's own pixel -> cell table, so a kept pixel lands on the same cell as today):
- base: every mask pixel (today; parity with the 14a stop evidence);
- med_d<d> (F1-med, cheap): drop mask pixels whose depth differs from the mask's median depth (pixels with depth > 0)
  by more than d metres (d in {0.2, 0.3, 0.5});
- clu_k<k>_m<m> (F1-clu, VLFM object_point_cloud_map.py + ConceptGraphs): erode the mask (3 x 3 kernel, k iterations),
  back-project the remaining pixels with 0 < depth <= 4 m to world points, DBSCAN (eps 0.2 m, min m pixels) and keep
  the largest cluster only (most pixels; empty -> no mark from this look); its pixels' cells are the mark. DBSCAN runs
  on 5 cm voxels weighted by their pixel count (kickoff fallback "DBSCAN too slow", taken up front: a mask has up to
  ~77k pixels, ~7 M per run), so "min m pixels" keeps its pixel meaning: a voxel is a core point when the pixels of the
  voxels within eps (itself included) are >= m. A border voxel joins the cluster of its first core neighbour (lowest
  voxel index; voxels sorted by their integer key).
  Tentative (VLFM): the mask's box lies in an outer third of the picture (x1 <= W / 3 or x0 >= 2 W / 3), or the
  cluster's median depth is beyond 0.95 x 4 m; the stop module later deletes a tentative look's cells when they are
  in view within 3 m without a detection (stop_variants.py).
numpy + scipy + OpenCV (Habitat container; imported where used, so ProjSpec also loads in naz_mapmad).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ProjSpec:
    """p14b1_rules.md §3 numbers."""

    med_d: Tuple[float, ...] = (0.2, 0.3, 0.5)
    erode_k: Tuple[int, ...] = (1, 2, 3)
    min_px: Tuple[int, ...] = (20, 50, 100)
    eps_m: float = 0.2
    voxel_m: float = 0.05
    max_depth_m: float = 4.0
    far_frac: float = 0.95

    def med_names(self) -> List[str]:
        return [med_name(d) for d in self.med_d]

    def clu_names(self) -> List[str]:
        return [clu_name(k, m) for k in self.erode_k for m in self.min_px]

    def names(self) -> List[str]:
        """Every projection variant, base first."""
        return ["base"] + self.med_names() + self.clu_names()


def med_name(d: float) -> str:
    return f"med_d{d:.1f}"


def clu_name(k: int, m: int) -> str:
    return f"clu_k{k}_m{m}"


@dataclass
class FrameGeometry:
    """One frame's pixels on the map: depth (H, W), the server's pixel -> cell table (-1 = none) and the world point
    of every pixel with 0 < depth <= max_depth (NaN elsewhere)."""

    depth: np.ndarray
    pixel_cell: np.ndarray
    world: np.ndarray  # (H * W, 3)

    @classmethod
    def build(cls, depth: np.ndarray, pixel_cell: np.ndarray, hfov_deg: float, cam_tf: np.ndarray,
              max_depth_m: float) -> "FrameGeometry":
        from mapmad_sim.depth_map import camera_points

        pts, keep = camera_points(depth, hfov_deg, max_depth_m)
        world = np.full((depth.size, 3), np.nan)
        world[keep] = pts @ np.asarray(cam_tf)[:3, :3].T + np.asarray(cam_tf)[:3, 3]
        return cls(np.asarray(depth), np.asarray(pixel_cell, np.int64), world)


def cells_of(px: np.ndarray, pixel_cell: np.ndarray) -> np.ndarray:
    """Sorted unique storey cells under flat pixel indices (pixels without a cell dropped)."""
    c = pixel_cell[px]
    return np.unique(c[c >= 0]).astype(np.int32)


def median_filter(px: np.ndarray, depth_flat: np.ndarray, d: float) -> np.ndarray:
    """F1-med: the mask pixels whose depth is within d of the median depth of the mask's pixels with depth > 0."""
    z = depth_flat[px]
    ok = z > 0
    if not ok.any():
        return px[:0]
    med = float(np.median(z[ok]))
    return px[ok & (np.abs(z - med) <= d + 1e-9)]


def erode(mask: np.ndarray, k: int) -> np.ndarray:
    """The (H, W) bool mask eroded k times by a 3 x 3 kernel (OpenCV; pixels outside the picture count as set, as
    cv2.erode's default border)."""
    import cv2

    if k <= 0:
        return mask
    return cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(k)).astype(bool)


@dataclass
class Voxels:
    """Weighted voxels of a point set: centroid, pixel count, and each point's voxel."""

    centre: np.ndarray  # (V, 3)
    weight: np.ndarray  # (V,)
    of_point: np.ndarray  # (N,)


def voxelize(points: np.ndarray, size: float) -> Voxels:
    """Points (N, 3) -> voxels of `size` metres (sorted by their integer key; size 0 = one voxel per point)."""
    if size <= 0:
        n = len(points)
        return Voxels(points.astype(np.float64), np.ones(n, np.int64), np.arange(n))
    key = np.floor(points / size).astype(np.int64)
    _, of_point, weight = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    of_point = of_point.reshape(-1)
    centre = np.zeros((len(weight), 3))
    np.add.at(centre, of_point, points)
    return Voxels(centre / weight[:, None], weight.astype(np.int64), of_point)


def neighbour_graph(centre: np.ndarray, eps: float):
    """Symmetric adjacency (scipy csr, no self loops) of the voxels within eps of each other."""
    from scipy import sparse
    from scipy.spatial import cKDTree

    n = len(centre)
    pairs = cKDTree(centre).query_pairs(eps, output_type="ndarray") if n > 1 else np.zeros((0, 2), np.int64)
    i, j = pairs[:, 0], pairs[:, 1]
    data = np.ones(2 * len(pairs), bool)
    return sparse.csr_matrix((data, (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n))


def largest_cluster(vox: Voxels, graph, min_px: int) -> np.ndarray:
    """Weighted DBSCAN (module docstring) -> bool per voxel: in the largest cluster (most pixels; ties: the cluster
    holding the lowest voxel index). No core voxel -> all False."""
    from scipy.sparse.csgraph import connected_components

    n = len(vox.weight)
    out = np.zeros(n, bool)
    if n == 0:
        return out
    reach = graph @ vox.weight + vox.weight  # pixels within eps, itself included
    core = reach >= min_px
    if not core.any():
        return out
    ci = np.flatnonzero(core)
    sub = graph[ci][:, ci]
    _, lab_core = connected_components(sub, directed=False)
    label = np.full(n, -1, np.int64)
    label[ci] = lab_core
    border = np.flatnonzero(~core)
    if border.size:  # first core neighbour (lowest voxel index; ci is sorted)
        g = graph[border][:, ci].tocsr()
        g.sort_indices()
        has = np.diff(g.indptr) > 0
        label[border[has]] = lab_core[g.indices[g.indptr[:-1][has]]]
    lab = label[label >= 0]
    size = np.bincount(lab, weights=vox.weight[label >= 0])
    best = np.flatnonzero(size == size.max())
    first = {int(lb): int(np.flatnonzero(label == lb)[0]) for lb in best}
    win = min(first, key=first.get)
    out[label == win] = True
    return out


def outer_third(box: Optional[Sequence[float]], width: int) -> bool:
    """The box lies in the left or right third of the picture (VLFM's 'too offset')."""
    if box is None or not np.all(np.isfinite(box)):
        return False
    x0, _, x1, _ = (float(v) for v in box)
    return x1 <= width / 3.0 or x0 >= 2.0 * width / 3.0


@dataclass
class RowMarks:
    """One detection (or true instance) in one frame: cells per variant, tentative flag per clu variant, and the
    median depth of the cluster per clu variant (NaN = empty)."""

    cells: Dict[str, np.ndarray]
    tentative: Dict[str, bool]
    cluster_depth: Dict[str, float]


def project_mask(mask: np.ndarray, box: Optional[Sequence[float]], geo: FrameGeometry, spec: ProjSpec) -> RowMarks:
    """Every projection variant of one (H, W) bool mask (module docstring)."""
    h, w = mask.shape
    dflat = geo.depth.reshape(-1)
    px = np.flatnonzero(mask.reshape(-1))
    cells = {"base": cells_of(px, geo.pixel_cell)}
    for d in spec.med_d:
        cells[med_name(d)] = cells_of(median_filter(px, dflat, d), geo.pixel_cell)
    tent, cdep = {}, {}
    offset = outer_third(box, w)
    for k in spec.erode_k:
        epx = np.flatnonzero(erode(mask, k).reshape(-1)) if px.size else px
        pts_ok = np.isfinite(geo.world[epx, 0]) if epx.size else np.zeros(0, bool)
        epx = epx[pts_ok]
        vox = voxelize(geo.world[epx], spec.voxel_m) if epx.size else Voxels(np.zeros((0, 3)), np.zeros(0, np.int64),
                                                                            np.zeros(0, np.int64))
        graph = neighbour_graph(vox.centre, spec.eps_m) if epx.size else None
        for m in spec.min_px:
            name = clu_name(k, m)
            if not epx.size:
                cells[name], tent[name], cdep[name] = np.zeros(0, np.int32), False, float("nan")
                continue
            keep_vox = largest_cluster(vox, graph, m)
            cpx = epx[keep_vox[vox.of_point]]
            cells[name] = cells_of(cpx, geo.pixel_cell)
            med = float(np.median(dflat[cpx])) if cpx.size else float("nan")
            cdep[name] = med
            tent[name] = bool(cpx.size) and (offset or med > spec.far_frac * spec.max_depth_m)
    return RowMarks(cells, tent, cdep)


def box_fill_mask(box: Sequence[float], depth: np.ndarray, band_m: float = 0.3) -> np.ndarray:
    """Kickoff fallback "F1-box": the box's pixels whose depth is within band_m of the box's median depth (> 0)."""
    h, w = depth.shape
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    out = np.zeros((h, w), bool)
    if x1 <= x0 or y1 <= y0:
        return out
    z = depth[y0:y1, x0:x1]
    ok = z > 0
    if not ok.any():
        return out
    med = float(np.median(z[ok]))
    out[y0:y1, x0:x1] = ok & (np.abs(z - med) <= band_m + 1e-9)
    return out


class MarkTable:
    """Collects the rows of one run (detections or true instances) as ragged arrays per variant (npz-ready)."""

    def __init__(self, prefix: str, spec: ProjSpec) -> None:
        self.prefix, self.spec = prefix, spec
        self.cells: Dict[str, List[np.ndarray]] = {v: [] for v in spec.names()}
        self.tent: Dict[str, List[bool]] = {v: [] for v in spec.clu_names()}
        self.cdep: Dict[str, List[float]] = {v: [] for v in spec.clu_names()}
        self.extra: Dict[str, List[float]] = {}

    def add(self, row: RowMarks, **extra: float) -> None:
        for v in self.cells:
            self.cells[v].append(np.asarray(row.cells[v], np.int32))
        for v in self.tent:
            self.tent[v].append(bool(row.tentative[v]))
            self.cdep[v].append(float(row.cluster_depth[v]))
        for k, x in extra.items():
            self.extra.setdefault(k, []).append(float(x))

    def __len__(self) -> int:
        return len(self.cells["base"])

    def arrays(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        p = self.prefix
        for v, parts in self.cells.items():
            ptr = np.zeros(len(parts) + 1, np.int64)
            ptr[1:] = np.cumsum([x.size for x in parts]) if parts else []
            out[f"{p}_cells_{v}"] = np.concatenate(parts).astype(np.int32) if parts else np.zeros(0, np.int32)
            out[f"{p}_ptr_{v}"] = ptr
        for v in self.tent:
            out[f"{p}_tent_{v}"] = np.asarray(self.tent[v], bool)
            out[f"{p}_cdepth_{v}"] = np.asarray(self.cdep[v], np.float64)
        for k, xs in self.extra.items():
            out[f"{p}_{k}"] = np.asarray(xs, np.float64)
        return out
