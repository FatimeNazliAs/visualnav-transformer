"""Phase 14a stop rules, scored offline from one run's stop evidence (stop_evidence.py; kickoff 14.1, p14a_rules.md §3).
The robot's driving does not depend on the stop rule (every run drives 1,000 steps), so every rule is judged on the
same runs: `EvidenceReplay` rebuilds, frame by frame, what each look at each storey cell said, and `fires` lists, per
step t >= 1 (frame t = the picture after step t), the cells on which a rule wants to stop.

A look = one frame in which a cell is in view (TargetFusion: the view cells plus the cells under a detection). Its
detection = the recogniser detection covering the cell with the largest map value in that frame (ties: more mask
pixels, then the first); a look without one is a miss.
- today (Phases 12-13): value = mean over looks of the map value (miss = 0), float32 as TargetFusion; stop if a cell
  <= 1.0 m ahead (+-30 deg of the heading) seen from >= 2 looks has value >= tau (0.45). `today_value` reproduces the
  logged rule_value (the parity check of the replay).
- Rule 1 (voting, Perception Matters 2408.02297): a look is a hit if the largest raw recogniser score covering the
  cell is >= s; stop if a cell <= 1.0 m ahead (+-30 deg) has looks >= n and hits / looks >= h.
- Rule 2 (weighted voting + close-and-confident, Perception Matters + ApexNav 2504.14478): a hit look weighs
  w = value x size x near, size = min(mask pixels / 1,000, 1), near = 1 at <= 1.5 m falling linearly to 0.25 at 4 m
  (distance robot -> cell centre in that frame); a miss weighs its geometric weight near (size 1: the cell was in
  view) with value 0; cell value = sum(w x value) / sum(w); stop if the robot is <= 1.0 m from the cell (any
  direction), the value >= v and >= g hit looks had w >= 0.3.
The fired cell of a step = the qualifying cell with the largest rule value (Rule 1: hit share, then looks); ties: the
first in row-major order (as rule_value). Rejected spots (14.2): `fires(..., exclude=[(x, y), ...])` drops cells
within spot_m (0.5 m) of a rejected cell. numpy only (both containers).
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

R1_S = (0.1, 0.25, 0.4)
R1_N = (2, 3, 4, 5)
R1_H = (0.3, 0.4, 0.5, 0.6, 0.7)
R2_V = tuple(round(0.30 + 0.05 * k, 2) for k in range(9))  # 0.30 ... 0.70
R2_G = (1, 2, 3)


@dataclass(frozen=True)
class StopSpec:
    """p14a_rules.md §3 numbers."""

    ahead_m: float = 1.0
    half_angle_deg: float = 30.0
    min_views: int = 2
    tau: float = 0.45
    near_m: float = 1.0  # Rule 2: robot -> cell distance
    size_px: float = 1000.0
    full_m: float = 1.5
    far_m: float = 4.0
    far_weight: float = 0.25
    min_w: float = 0.3
    spot_m: float = 0.5


def r1_key(s: float, n: int, h: float) -> str:
    return f"r1_s{s:.2f}_n{n}_h{h:.1f}"


def r2_key(v: float, g: int) -> str:
    return f"r2_v{v:.2f}_g{g}"


def r1_grid() -> List[Tuple[str, float, int, float]]:
    return [(r1_key(s, n, h), s, n, h) for s in R1_S for n in R1_N for h in R1_H]


def r2_grid() -> List[Tuple[str, float, int]]:
    return [(r2_key(v, g), v, g) for v in R2_V for g in R2_G]


def near_weight(d: np.ndarray, spec: StopSpec) -> np.ndarray:
    """Distance factor of Rule 2 (module docstring)."""
    frac = np.clip((np.asarray(d, np.float64) - spec.full_m) / (spec.far_m - spec.full_m), 0.0, 1.0)
    return 1.0 - (1.0 - spec.far_weight) * frac


@dataclass
class Fire:
    """One step's candidate cells for one rule: flat cell ids (best first) and their rule values."""

    cells: np.ndarray
    values: np.ndarray


class EvidenceReplay:
    """Per-cell look statistics of one run, frame by frame (module docstring)."""

    def __init__(self, ev: Dict[str, np.ndarray], spec: StopSpec = StopSpec(), s_grid: Sequence[float] = R1_S) -> None:
        self.ev, self.spec = ev, spec
        self.shape = (int(ev["shape"][0]), int(ev["shape"][1]))
        self.origin, self.res = np.asarray(ev["origin"], np.float64), float(ev["res"])
        n = self.shape[0] * self.shape[1]
        self.count = np.zeros(n, np.int32)
        self.total = np.zeros(n, np.float64)
        self.s_grid = tuple(s_grid)
        self.hits = np.zeros((len(self.s_grid), n), np.int32)
        self.sw = np.zeros(n, np.float64)
        self.swv = np.zeros(n, np.float64)
        self.strong = np.zeros(n, np.int32)
        self.ever = np.zeros(n, bool)
        self.t = -1
        self.frames = len(ev["pose"])
        self._det_by_t = self._group(ev["det_t"])
        i, j = np.divmod(np.arange(n), self.shape[1])
        self.cx, self.cy = self.origin[0] + (i + 0.5) * self.res, self.origin[1] + (j + 0.5) * self.res

    @staticmethod
    def _group(ts: np.ndarray) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for k, t in enumerate(np.asarray(ts).tolist()):
            out.setdefault(int(t), []).append(k)
        return out

    def advance(self) -> int:
        """Add the next frame's looks; returns its index t."""
        self.t += 1
        t, ev, spec = self.t, self.ev, self.spec
        pose = ev["pose"][t]
        view = ev["view"][ev["view_ptr"][t]:ev["view_ptr"][t + 1]].astype(np.int64)
        best: Dict[int, Tuple[float, int, float]] = {}  # cell -> (value, px, max raw)
        for k in self._det_by_t.get(t, []):
            cells = ev["det_cells"][ev["det_cells_ptr"][k]:ev["det_cells_ptr"][k + 1]].tolist()
            val, px, raw = float(ev["det_value"][k]), int(ev["det_px"][k]), float(ev["det_raw"][k])
            for c in cells:
                old = best.get(c)
                if old is None:
                    best[c] = (val, px, raw)
                else:
                    keep = (val, px) if (val, px) > (old[0], old[1]) else (old[0], old[1])
                    best[c] = (keep[0], keep[1], max(raw, old[2]))
        hit_cells = np.fromiter(best.keys(), np.int64, len(best))
        looks = np.union1d(view, hit_cells)
        if looks.size == 0:
            return t
        d = np.hypot(self.cx[looks] - pose[0], self.cy[looks] - pose[1])
        near = near_weight(d, spec)
        self.count[looks] += 1
        self.sw[looks] += near  # misses: geometric weight; hits corrected below
        if hit_cells.size:
            vals = np.array([best[c][0] for c in hit_cells.tolist()])
            pxs = np.array([best[c][1] for c in hit_cells.tolist()], np.float64)
            raws = np.array([best[c][2] for c in hit_cells.tolist()])
            pos = np.searchsorted(looks, hit_cells)
            w = vals * np.minimum(pxs / spec.size_px, 1.0) * near[pos]
            self.total[hit_cells] += vals
            self.sw[hit_cells] += w - near[pos]
            self.swv[hit_cells] += w * vals
            self.strong[hit_cells] += (w >= spec.min_w - 1e-12)
            for si, s in enumerate(self.s_grid):
                self.hits[si, hit_cells] += (raws >= s - 1e-12)
            self.ever[hit_cells] |= vals > 0
        return t

    # --- cells near the robot -----------------------------------------------------------------------------------
    def _near_cells(self, pose: Sequence[float], radius: float, half_angle_deg: Optional[float]) -> np.ndarray:
        """Flat ids of the cells whose centre is <= radius from the pose (and within +-half_angle of its heading),
        in row-major order (search_exam.ahead_cells' maths)."""
        r = int(math.ceil(radius / self.res)) + 1
        i0 = int(math.floor((pose[0] - self.origin[0]) / self.res))
        j0 = int(math.floor((pose[1] - self.origin[1]) / self.res))
        ii, jj = np.mgrid[max(0, i0 - r):min(self.shape[0], i0 + r + 1), max(0, j0 - r):min(self.shape[1], j0 + r + 1)]
        cx, cy = self.origin[0] + (ii + 0.5) * self.res, self.origin[1] + (jj + 0.5) * self.res
        dx, dy = cx - pose[0], cy - pose[1]
        ok = np.hypot(dx, dy) <= radius
        if half_angle_deg is not None:
            ang = np.arctan2(dy, dx) - pose[2]
            ok &= np.abs(np.arctan2(np.sin(ang), np.cos(ang))) <= math.radians(half_angle_deg)
        return (ii[ok] * self.shape[1] + jj[ok]).astype(np.int64)

    def ahead(self) -> np.ndarray:
        return self._near_cells(self.ev["pose"][self.t], self.spec.ahead_m, self.spec.half_angle_deg)

    def around(self) -> np.ndarray:
        return self._near_cells(self.ev["pose"][self.t], self.spec.near_m, None)

    # --- rules at the current frame -------------------------------------------------------------------------------
    def value(self, cells: np.ndarray) -> np.ndarray:
        """Today's fused value (float32, TargetFusion's maths)."""
        c = np.maximum(self.count[cells], 1)
        return (self.total[cells] / c).astype(np.float32)

    def today_value(self) -> float:
        """The logged rule_value of this frame (largest value among ahead cells with >= min_views looks)."""
        cells = self.ahead()
        cells = cells[self.count[cells] >= self.spec.min_views]
        return float(self.value(cells).max()) if cells.size else 0.0

    def today(self, tau: float) -> Fire:
        cells = self.ahead()
        cells = cells[(self.count[cells] >= self.spec.min_views) & self.ever[cells]]
        v = self.value(cells).astype(np.float64)
        return _ranked(cells, v, v >= tau - 1e-12)

    def rule1(self, si: int, n: int, h: float) -> Fire:
        cells = self.ahead()
        cells = cells[self.ever[cells]]
        looks = self.count[cells]
        share = self.hits[si, cells] / np.maximum(looks, 1)
        return _ranked(cells, share + 1e-9 * looks, (looks >= n) & (share >= h - 1e-12))

    def rule2_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cells within near_m that were ever hit, their Rule 2 value, their strong-look count)."""
        cells = self.around()
        cells = cells[self.ever[cells]]
        v = np.where(self.sw[cells] > 0, self.swv[cells] / np.maximum(self.sw[cells], 1e-12), 0.0)
        return cells, v, self.strong[cells]

    def rule2(self, v_min: float, g: int) -> Fire:
        cells, v, strong = self.rule2_values()
        return _ranked(cells, v, (v >= v_min - 1e-12) & (strong >= g))

    def cell_xy(self, k: int) -> Tuple[float, float]:
        return float(self.cx[k]), float(self.cy[k])

    def excluded(self, cells: np.ndarray, spots: Sequence[Tuple[float, float]]) -> np.ndarray:
        """True for cells within spot_m of a rejected spot."""
        out = np.zeros(cells.size, bool)
        for x, y in spots:
            out |= np.hypot(self.cx[cells] - x, self.cy[cells] - y) <= self.spec.spot_m
        return out


def _ranked(cells: np.ndarray, value: np.ndarray, ok: np.ndarray) -> Fire:
    """The qualifying cells, best first (stable: row-major order among equal values)."""
    c, v = cells[ok], np.asarray(value, np.float64)[ok]
    order = np.argsort(-v, kind="stable")
    return Fire(c[order], v[order])


def first_stops(ev: Dict[str, np.ndarray], spec: StopSpec = StopSpec(),
                logged_rule: Optional[Sequence[Optional[float]]] = None) -> Dict[str, object]:
    """Every rule variant's first stop of one run: {key: (step, cell xy) or None}, plus "parity": the share of steps
    whose rebuilt today-value equals the logged rule_value (|diff| <= 1e-5; logged_rule[t - 1] = step t's value)."""
    rep = EvidenceReplay(ev, spec)
    r1, r2 = r1_grid(), r2_grid()
    out: Dict[str, object] = {"today": None}
    out.update({k: None for k, *_ in r1})
    out.update({k: None for k, *_ in r2})
    same = checked = 0
    rep.advance()  # frame 0 (the start): no stop before the first step
    for t in range(1, rep.frames):
        rep.advance()
        if logged_rule is not None and t - 1 < len(logged_rule):
            checked += 1
            same += abs(rep.today_value() - float(logged_rule[t - 1] or 0.0)) <= 1e-5
        if not rep.ever.any():
            continue
        if out["today"] is None:
            f = rep.today(spec.tau)
            if f.cells.size:
                out["today"] = (t, rep.cell_xy(int(f.cells[0])))
        for key, s, n, h in r1:
            if out[key] is None:
                f = rep.rule1(rep.s_grid.index(s), n, h)
                if f.cells.size:
                    out[key] = (t, rep.cell_xy(int(f.cells[0])))
        open2 = [x for x in r2 if out[x[0]] is None]
        if open2:
            cells, v, strong = rep.rule2_values()
            for key, vmin, g in open2:
                ok = (v >= vmin - 1e-12) & (strong >= g)
                if ok.any():
                    f = _ranked(cells, v, ok)
                    out[key] = (t, rep.cell_xy(int(f.cells[0])))
    out["parity"] = {"checked": checked, "same": same}
    return out


def rule_fires(ev: Dict[str, np.ndarray], key: str, spec: StopSpec = StopSpec()) -> Iterator[Tuple[int, "EvidenceReplay", Fire]]:
    """Every step t >= 1 at which the rule `key` (today | r1_... | r2_...) has qualifying cells (14.2's veto loop)."""
    rep = EvidenceReplay(ev, spec)
    rep.advance()
    fn = rule_function(key)
    for t in range(1, rep.frames):
        rep.advance()
        if not rep.ever.any():
            continue
        f = fn(rep)
        if f.cells.size:
            yield t, rep, f


def rule_function(key: str):
    """The replay method of a rule key."""
    if key == "today":
        return lambda rep: rep.today(rep.spec.tau)
    if key.startswith("r1_"):
        s, n, h = parse_r1(key)
        return lambda rep: rep.rule1(rep.s_grid.index(s), n, h)
    if key.startswith("r2_"):
        v, g = parse_r2(key)
        return lambda rep: rep.rule2(v, g)
    raise ValueError(key)


def parse_r1(key: str) -> Tuple[float, int, float]:
    _, s, n, h = key.split("_")
    return float(s[1:]), int(n[1:]), float(h[1:])


def parse_r2(key: str) -> Tuple[float, int]:
    _, v, g = key.split("_")
    return float(v[1:]), int(g[1:])
