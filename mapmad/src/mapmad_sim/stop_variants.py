"""Phase 14b-1: the stop module's marks rebuilt with fixes F1 / F2 / F3 (+ c_cat), scored offline on the Phase 14a
runs (kickoff §3, p14b1_rules.md §3-§4). Nothing here changes a run or the network's map input: every variant is
scored on the same logs, from the 14a stop evidence (poses, view cells, detections) and the 14b-1 replay's projection
file (mark_projection.py: cells per projection variant of every detection and true instance).

A variant = (marks, projection, score, c_cat):
- marks: "det" (the recogniser) or "true" (simulator-truth instances, score 1.0: the ceiling);
- projection: base | med_d<d> | clu_k<k>_m<m> (mark_projection.ProjSpec.names()); clu looks can be tentative;
- score (F2): "map" = today's map value 0.4 + 0.6 (s - 0.01) / (0.788 - 0.01) (Rule 1 reads the raw score, as 14a)
  or "cal" = the raw score temperature-scaled, sigmoid(logit(s) / T) (value and Rule 1 both read it);
- c_cat: None, or a minimum calibrated score per category (detections below it are dropped).
F3 is a rule option, not a variant: a stop cell must also be <= 1.0 m, within +-30 deg ahead and in view in the
current frame (its view cells) -- for today's rule, Rule 1 and Rule 2 alike. Both F3 off and on are scored in the same
pass (the looks are the same).

Grids (rules §4): today tau 0.05 ... 0.95; Rule 1 s in {0.05, 0.1, 0.25, 0.4, 0.55} x n {2..5} x h {0.3..0.7};
Rule 2 v 0.05 ... 0.70 x g {1, 2, 3}. The 14a points are inside (parity: projection base, score map, F3 off = the 14a
bench's per-run results).
Tentative looks (VLFM): a look whose detection is tentative is pending; at a later frame where the cell is in view
within 3 m of the robot and no detection covers it, every pending look on the cell is turned into a miss.
numpy only (both containers).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.stop_rules import EvidenceReplay, StopSpec, near_weight, r1_key, r2_key

TAU_GRID = tuple(round(0.05 * k, 2) for k in range(1, 20))  # 0.05 ... 0.95
S_GRID = (0.05, 0.1, 0.25, 0.4, 0.55)
N_GRID = (2, 3, 4, 5)
H_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
V_GRID = tuple(round(0.05 * k, 2) for k in range(1, 15))  # 0.05 ... 0.70
G_GRID = (1, 2, 3)
DELETE_M = 3.0  # tentative looks: deleted when the cell is in view within this distance without a detection
F3_TAG = "|f3"


def t_key(tau: float) -> str:
    return f"t_tau{tau:.2f}"


def rule_keys() -> List[str]:
    """Every rule grid point (F3 off); the F3-on twin of a key is key + F3_TAG."""
    return ([t_key(t) for t in TAU_GRID] + [r1_key(s, n, h) for s in S_GRID for n in N_GRID for h in H_GRID]
            + [r2_key(v, g) for v in V_GRID for g in G_GRID])


def all_keys() -> List[str]:
    k = rule_keys()
    return k + [x + F3_TAG for x in k]


def family(key: str) -> str:
    """today | r1 | r2 of a rule key."""
    k = key.split("|")[0]
    return "today" if k.startswith("t_") else k[:2]


@dataclass(frozen=True)
class Variant:
    marks: str = "det"  # det | true
    proj: str = "base"
    score: str = "map"  # map | cal
    ccat: Optional[float] = None

    @property
    def name(self) -> str:
        c = "" if self.ccat is None else f"|c{self.ccat:.2f}"
        return f"{self.marks}|{self.proj}|{self.score}{c}"

    @classmethod
    def parse(cls, name: str) -> "Variant":
        parts = name.split("|")
        return cls(parts[0], parts[1], parts[2], float(parts[3][1:]) if len(parts) > 3 else None)


def calibrated(raw: np.ndarray, temperature: float) -> np.ndarray:
    """F2: sigmoid(logit(s) / T); logit recovered from the score (kickoff fallback: no raw logits logged)."""
    s = np.clip(np.asarray(raw, np.float64), 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-np.log(s / (1.0 - s)) / float(temperature)))


def variant_evidence(ev: Dict[str, np.ndarray], proj: Dict[str, np.ndarray], v: Variant,
                     temperature: float = 1.0) -> Dict[str, np.ndarray]:
    """The run's evidence with the variant's marks in the det_* arrays (+ det_tent: tentative flag per row)."""
    out = dict(ev)
    if v.marks == "true":
        n = len(ev["true_t"])
        out.update({"det_t": ev["true_t"], "det_raw": np.ones(n), "det_value": np.ones(n), "det_px": ev["true_px"],
                    "det_box": ev["true_box"]})
        pre = "true"
    else:
        raw = np.asarray(ev["det_raw"], np.float64)
        if v.score == "cal":
            cal = calibrated(raw, temperature)
            out["det_value"], out["det_raw"] = cal, cal
        pre = "det"
    cells, ptr = proj[f"{pre}_cells_{v.proj}"], proj[f"{pre}_ptr_{v.proj}"]
    if len(ptr) - 1 != len(out["det_t"]):
        raise ValueError(f"projection rows {len(ptr) - 1} != evidence rows {len(out['det_t'])} ({v.name})")
    if v.ccat is not None:
        keep = np.asarray(out["det_value"]) >= v.ccat - 1e-12
        sizes = np.where(keep, np.diff(ptr), 0)
        cells = np.concatenate([cells[ptr[k]:ptr[k + 1]] for k in np.flatnonzero(keep)]) if keep.any() \
            else np.zeros(0, np.int32)
        ptr = np.concatenate([[0], np.cumsum(sizes)])
    out["det_cells"], out["det_cells_ptr"] = cells, ptr
    tk = f"{pre}_tent_{v.proj}"
    out["det_tent"] = proj[tk] if tk in proj else np.zeros(len(out["det_t"]), bool)
    return out


class RunGeometry:
    """Per frame, the cells the rules look at (shared by every variant of one run): ahead (<= 1.0 m, +-30 deg),
    around (<= 1.0 m) and ahead-and-in-view-now (F3), each row-major."""

    def __init__(self, ev: Dict[str, np.ndarray], spec: StopSpec = StopSpec()) -> None:
        rep = EvidenceReplay(ev, spec, s_grid=())
        self.ahead, self.around, self.f3, self.near3 = [], [], [], []
        for t in range(len(ev["pose"])):
            pose = ev["pose"][t]
            view = ev["view"][ev["view_ptr"][t]:ev["view_ptr"][t + 1]].astype(np.int64)
            a = rep._near_cells(pose, spec.ahead_m, spec.half_angle_deg)
            self.ahead.append(a)
            self.around.append(rep._near_cells(pose, spec.near_m, None))
            self.f3.append(a[np.isin(a, view)])
            d = np.hypot(rep.cx[view] - pose[0], rep.cy[view] - pose[1])
            self.near3.append(view[d <= DELETE_M])


class VariantReplay(EvidenceReplay):
    """EvidenceReplay with tentative looks (module docstring) and a hit counter (so a deleted look can clear `ever`)."""

    def __init__(self, ev: Dict[str, np.ndarray], spec: StopSpec = StopSpec(), geo: Optional[RunGeometry] = None) -> None:
        super().__init__(ev, spec, s_grid=S_GRID)
        self.geo = geo
        self.nhit = np.zeros(self.count.size, np.int32)
        self.pending: Dict[int, List[Tuple[float, float, float, bool, np.ndarray]]] = {}  # val, w, near, strong, hits
        self.tent = np.asarray(ev.get("det_tent", np.zeros(len(ev["det_t"]), bool)), bool)

    def advance(self) -> int:
        self.t += 1
        t, ev, spec = self.t, self.ev, self.spec
        pose = ev["pose"][t]
        view = ev["view"][ev["view_ptr"][t]:ev["view_ptr"][t + 1]].astype(np.int64)
        best: Dict[int, Tuple[float, int, float, bool]] = {}  # cell -> (value, px, max raw, tentative)
        for k in self._det_by_t.get(t, []):
            a, b = ev["det_cells_ptr"][k], ev["det_cells_ptr"][k + 1]
            if a == b:
                continue
            val, px, raw, tn = float(ev["det_value"][k]), int(ev["det_px"][k]), float(ev["det_raw"][k]), bool(self.tent[k])
            for c in ev["det_cells"][a:b].tolist():
                old = best.get(c)
                if old is None:
                    best[c] = (val, px, raw, tn)
                elif (val, px) > (old[0], old[1]):
                    best[c] = (val, px, max(raw, old[2]), tn)
                else:
                    best[c] = (old[0], old[1], max(raw, old[2]), old[3])
        hit_cells = np.fromiter(best.keys(), np.int64, len(best))
        if self.pending:
            self._delete(hit_cells)
        looks = np.union1d(view, hit_cells)
        if looks.size == 0:
            return t
        d = np.hypot(self.cx[looks] - pose[0], self.cy[looks] - pose[1])
        near = near_weight(d, spec)
        self.count[looks] += 1
        self.sw[looks] += near
        if hit_cells.size:
            rows = [best[c] for c in hit_cells.tolist()]
            vals = np.array([r[0] for r in rows])
            pxs = np.array([r[1] for r in rows], np.float64)
            raws = np.array([r[2] for r in rows])
            tns = np.array([r[3] for r in rows], bool)
            pos = np.searchsorted(looks, hit_cells)
            w = vals * np.minimum(pxs / spec.size_px, 1.0) * near[pos]
            strong = w >= spec.min_w - 1e-12
            self.total[hit_cells] += vals
            self.sw[hit_cells] += w - near[pos]
            self.swv[hit_cells] += w * vals
            self.strong[hit_cells] += strong
            hit_s = np.stack([raws >= s - 1e-12 for s in self.s_grid]) if self.s_grid else np.zeros((0, len(raws)), bool)
            self.hits[:, hit_cells] += hit_s
            pos_v = vals > 0
            self.nhit[hit_cells] += pos_v
            self.ever[hit_cells] |= pos_v
            for i in np.flatnonzero(tns):
                c = int(hit_cells[i])
                self.pending.setdefault(c, []).append((vals[i], w[i], near[pos[i]], bool(strong[i]), hit_s[:, i]))
        return t

    def _delete(self, hit_cells: np.ndarray) -> None:
        """Pending tentative looks on cells in view within DELETE_M now and not covered by a detection -> misses."""
        near3 = self.geo.near3[self.t] if self.geo is not None else np.zeros(0, np.int64)
        if near3.size == 0:
            return
        keys = np.fromiter(self.pending.keys(), np.int64, len(self.pending))
        gone = keys[np.isin(keys, near3) & ~np.isin(keys, hit_cells)]
        for c in gone.tolist():
            for val, w, nr, strong, hs in self.pending.pop(c):
                self.total[c] -= val
                self.sw[c] -= w - nr
                self.swv[c] -= w * val
                self.strong[c] -= int(strong)
                self.hits[:, c] -= hs
                if val > 0:
                    self.nhit[c] -= 1
            self.ever[c] = self.nhit[c] > 0


def _first_ok(order_cells: np.ndarray, ok: np.ndarray) -> Optional[int]:
    i = np.flatnonzero(ok)
    return int(order_cells[i[0]]) if i.size else None


def first_stops(ev: Dict[str, np.ndarray], geo: RunGeometry, spec: StopSpec = StopSpec()) -> Dict[str, Optional[Tuple[int, int]]]:
    """Every rule key's first stop (all_keys(): F3 off and on) of one variant's evidence: {key: (step, flat cell) or
    None}. Same choices as stop_rules.first_stops (fired cell = the best qualifying cell, ties row-major)."""
    rep = VariantReplay(ev, spec, geo)
    keys = rule_keys()
    out: Dict[str, Optional[Tuple[int, int]]] = {k: None for k in all_keys()}
    open_keys = set(out)
    rep.advance()
    for t in range(1, rep.frames):
        rep.advance()
        if not open_keys:
            break
        if not rep.ever.any():
            continue
        for tag, ahead, around in (("", geo.ahead[t], geo.around[t]), (F3_TAG, geo.f3[t], geo.f3[t])):
            _today(rep, ahead, tag, t, out, open_keys)
            _rule1(rep, ahead, tag, t, out, open_keys)
            _rule2(rep, around, tag, t, out, open_keys)
    return out


def _today(rep: VariantReplay, ahead: np.ndarray, tag: str, t: int, out, open_keys) -> None:
    ks = [(t_key(x), x) for x in TAU_GRID if t_key(x) + tag in open_keys]
    if not ks:
        return
    c = ahead[(rep.count[ahead] >= rep.spec.min_views) & rep.ever[ahead]]
    if not c.size:
        return
    v = rep.value(c).astype(np.float64)
    i = int(np.argmax(v))  # first max = row-major tie rule
    for k, tau in ks:
        if v[i] >= tau - 1e-12:
            out[k + tag] = (t, int(c[i]))
            open_keys.discard(k + tag)


def _rule1(rep: VariantReplay, ahead: np.ndarray, tag: str, t: int, out, open_keys) -> None:
    c = ahead[rep.ever[ahead]]
    if not c.size:
        return
    looks = rep.count[c]
    for si, s in enumerate(S_GRID):
        ks = [(r1_key(s, n, h), n, h) for n in N_GRID for h in H_GRID if r1_key(s, n, h) + tag in open_keys]
        if not ks:
            continue
        share = rep.hits[si, c] / np.maximum(looks, 1)
        order = np.argsort(-(share + 1e-9 * looks), kind="stable")
        oc, osh, olk = c[order], share[order], looks[order]
        for k, n, h in ks:
            cell = _first_ok(oc, (olk >= n) & (osh >= h - 1e-12))
            if cell is not None:
                out[k + tag] = (t, cell)
                open_keys.discard(k + tag)


def _rule2(rep: VariantReplay, around: np.ndarray, tag: str, t: int, out, open_keys) -> None:
    ks = [(r2_key(v, g), v, g) for v in V_GRID for g in G_GRID if r2_key(v, g) + tag in open_keys]
    if not ks:
        return
    c = around[rep.ever[around]]
    if not c.size:
        return
    v = np.where(rep.sw[c] > 0, rep.swv[c] / np.maximum(rep.sw[c], 1e-12), 0.0)
    strong = rep.strong[c]
    order = np.argsort(-v, kind="stable")
    oc, ov, og = c[order], v[order], strong[order]
    for k, vmin, g in ks:
        cell = _first_ok(oc, (ov >= vmin - 1e-12) & (og >= g))
        if cell is not None:
            out[k + tag] = (t, cell)
            open_keys.discard(k + tag)


def cell_xy(ev: Dict[str, np.ndarray], cell: int) -> Tuple[float, float]:
    """Centre (X, Y) of a flat storey cell."""
    i, j = divmod(int(cell), int(ev["shape"][1]))
    res = float(ev["res"])
    return float(ev["origin"][0] + (i + 0.5) * res), float(ev["origin"][1] + (j + 0.5) * res)


def layer_at(ev: Dict[str, np.ndarray], steps: Sequence[int], spec: StopSpec = StopSpec()) -> Dict[int, np.ndarray]:
    """The fused target layer (today's value: mean over looks, float32) of a variant's evidence at the given frames
    (videos: old vs new marks)."""
    rep = VariantReplay(ev, spec, RunGeometry(ev, spec) if "det_tent" in ev and np.any(ev["det_tent"]) else None)
    want, out = set(int(s) for s in steps), {}
    for t in range(rep.frames):
        rep.advance()
        if t in want:
            out[t] = (rep.total / np.maximum(rep.count, 1)).astype(np.float32)
    return out
