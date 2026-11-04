"""Phase 14a step 14.5: the mark check (report only). Inside naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/p14_videos.py code-check --out <p14a>/videos_marks           # does p13_videos draw the layer?
    python mapmad/scripts/p14_videos.py counts --config mapmad/configs/p14a_finish.yaml --subdir runs/baseline \
        --arms M21b M21bh --workers 16 --out <p14a>/videos_marks                       # mark_counts.json + .md
    python mapmad/scripts/p14_videos.py pick --config ... --subdir runs/baseline --out <p14a>/videos_marks
    python mapmad/scripts/p14_videos.py render --config ... --subdir runs/baseline --gpu G --recog-port P \
        --out <p14a>/videos_marks [--shard k/n]
    python mapmad/scripts/p13_videos.py check --out <p14a>/videos_marks             # boxes + poses (Phase 13 check)
    python mapmad/scripts/p14_videos.py render --zoom --picks <p14a>/videos_marks/picks.json ... \
        --out <p14a>/videos_marks/zoom                                               # + the near window (below)

Zoom (Naz's Part 3 review): the storey map shrinks to 440 px and the network's near window (64 x 64 cells of 10 cm,
robot-centred, forward up; the same cells as map_v2.near_window) is drawn beside it at 8 px per cell. Every mark >= 0.05
is an enlarged square with an outline (red when rule-ready). Under the storey map, each step prints every YOLOE box's
raw score and its mark strength (the largest layer value now on the cells under its mask, from the stop evidence),
plus the same strength on each true target's cells.

Videos = Phase 13's checkable videos (p13_videos.render_one, unchanged replay, boxes, timeline, checks) with three
changes for the mark check: every target-layer cell with a value > 0 drawn in a bright colour scale by its strength
(0 -> 1, matplotlib's "plasma" by 5 key colours; rule-ready cells red as before), a strength colour bar with the tau
tick in the legend, and the step's "max mark strength ahead" (the logged rule value: the largest layer value among
cells <= 1.0 m ahead, +-30 deg, seen from >= 2 views) printed on the map panel. Picks (seed 14501, M21b practice
runs): 4 runs with a YOLOE box on a true target (IoU >= 0.3) and 4 runs whose rule stop (tau 0.45) is on a wrong
object, from the stop bench's per-run files. Counts: mapmad_sim/mark_check.py over every run of the arms.
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p13_videos as pv  # noqa: E402

from mapmad_bridge import steplog  # noqa: E402
from mapmad_sim.arm_table import ArmTable  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.mark_check import mark_sightings  # noqa: E402
from mapmad_sim.run_layout import RunLayout, parse_run_id, run_id, write_text_atomic  # noqa: E402
from mapmad_sim.stop_evidence import load as load_evidence  # noqa: E402

PICK_SEED = 14501
N_PICK = 4
_RULE_VALUES: List[float] = []
_TAU = [0.45]


PLASMA = np.array([[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]], np.float64)


def strength_colours(v: np.ndarray) -> np.ndarray:
    """(..., ) strengths in [0, 1] -> (..., 3) RGB uint8: matplotlib's "plasma" through 5 key colours (linear in
    between); an empty input gives an empty output."""
    x = np.clip(np.asarray(v, np.float64), 0.0, 1.0) * (len(PLASMA) - 1)
    k = np.minimum(np.floor(x).astype(int), len(PLASMA) - 2)
    f = (x - k)[..., None]
    return np.round(PLASMA[k] * (1 - f) + PLASMA[k + 1] * f).astype(np.uint8)


def layer_image(fv, obst, expl, value, ready, tau):
    """p13_videos.layer_image with every value > 0 in the strength colour scale (module docstring)."""
    img = np.empty((fv.side, fv.side, 3), np.float32)
    img[:] = pv.OUTSIDE_PX
    img[fv.walkable] = pv.FLOOR_PX
    g = lambda a: a[fv.flat] & fv.inside  # noqa: E731
    img[g(expl)] = pv.EXPL_PX
    img[g(obst)] = pv.OBST_PX
    v = np.where(fv.inside, value[fv.flat], 0.0)
    on = v > 0
    img[on] = strength_colours(v[on]).astype(np.float32)
    img[on & (v >= tau - 1e-12) & g(ready)] = pv.READY_C
    return img.astype(np.uint8)


_legend_p13 = pv.legend_panel


def legend_panel(category, prompt, tau):
    """Phase 13's legend with the strength colour bar in place of the gradient and the ">= tau" rows."""
    img = _legend_p13(category, prompt, tau).copy()
    x, y = 12 + 488, 8 + 3 * 28
    img[y:y + 56, x:x + 460] = 0
    bar = strength_colours(np.linspace(0, 1, 200))[None].repeat(18, axis=0)
    img[y + 4:y + 22, x:x + 200] = bar
    tx = x + int(round(200 * tau))
    cv2.line(img, (tx, y + 22), (tx, y + 30), (255, 255, 255), 2)
    text = pv.Writer()
    text.add((x + 210, y), "mark strength 0 … 1", 20)
    text.add((x, y + 30), f"▲ tau {tau:.2f} (today's stop); red = rule-ready", 18)
    return text.draw(img)


_caption_p13 = pv.caption


def caption(text, episode, name, t, n, events):
    _caption_p13(text, episode, name, t, n, events)
    if t < len(_RULE_VALUES):
        v = _RULE_VALUES[t]
        text.add((pv.MAP_XY[0] + 8, pv.MAP_XY[1] + 32),
                 f"max mark strength ahead: {v:.2f}  (stop at ≥ {_TAU[0]:.2f})", 22,
                 pv.HIGH_C if v >= _TAU[0] - 1e-12 else (230, 230, 230))


_render_one_p13 = pv.render_one


def render_one(service, boxes, layout, table, arm, episode, k, args, tau, vname):
    log = steplog.read(layout.log(arm, run_id(episode["episode_id"], k)))
    _RULE_VALUES[:] = [float(log[0].get("start_rule", {}).get("value") or 0.0)] + \
        [float(s["rule_value"] or 0.0) for s in steplog.steps(log)]
    _TAU[0] = tau
    return _render_one_p13(service, boxes, layout, table, arm, episode, k, args, tau, vname)


def patch() -> None:
    pv.layer_image, pv.legend_panel, pv.caption, pv.render_one = layer_image, legend_panel, caption, render_one


# --- zoom: the network's near window beside the storey (render --zoom, Naz's Part 3 review) --------------------------
ZOOM_N, ZOOM_CELL_M, ZOOM_PX = 64, 0.1, 8  # 64 x 64 cells of 10 cm (6.4 m), 8 px per cell
ZOOM = ZOOM_N * ZOOM_PX  # 512
STOREY = 440  # the whole-storey map shrinks from 760 px to make room
RIGHT_X = 962
ZOOM_XY = (RIGHT_X + STOREY + 6, pv.HEAD + 4)  # (1408, 104): right edge 1920
NUM_XY = (RIGHT_X, pv.HEAD + 4 + STOREY + 8)  # the per-step numbers under the storey map
MARK_MIN = 0.05
_Z: Dict[str, Any] = {}  # the current step's arrays, filled by the patched layer_image / map_panel / render_one


def window_cells(pose: Sequence[float], origin: Sequence[float], res: float, shape: Sequence[int],
                 n: int = ZOOM_N, cell_m: float = ZOOM_CELL_M) -> Tuple[np.ndarray, np.ndarray]:
    """(flat storey cell, inside) of every pixel of the network's near window at pose (X, Y, yaw), each (n, n): the
    stored cell under the pixel's centre, forward = up, as map_v2.sample_layers."""
    from vint_train.mapmad.local_map import cell_centres, local_to_world

    fwd, left = cell_centres(n, cell_m)
    wx, wy = local_to_world(fwd, left, tuple(pose))
    i = np.floor((wx - origin[0]) / res).astype(np.int64)
    j = np.floor((wy - origin[1]) / res).astype(np.int64)
    inside = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    return np.where(inside, i * int(shape[1]) + j, 0), inside


def zoom_px(xy: Sequence[float], pose: Sequence[float], n: int = ZOOM_N, cell_m: float = ZOOM_CELL_M,
            px: int = ZOOM_PX) -> Tuple[float, float]:
    """(col, row) zoom pixel of a 2D point (X, Y); the robot is at (n / 2, n / 2) * px."""
    from vint_train.mapmad.local_map import world_to_local

    f, l_ = world_to_local(np.asarray(xy[0]), np.asarray(xy[1]), tuple(pose))
    return float((n / 2 - l_ / cell_m) * px), float((n / 2 - f / cell_m) * px)


def zoom_image(pose: Sequence[float], ev: Dict[str, Any], obst: np.ndarray, expl: np.ndarray, value: np.ndarray,
               ready: np.ndarray, tau: float, episode: Dict[str, Any], rule_cell: Optional[Sequence[float]],
               stop_cell: Optional[Sequence[float]]) -> np.ndarray:
    """The near window at ZOOM_PX px per cell: unknown / explored / obstacle as on the storey map, every mark cell
    >= MARK_MIN as an enlarged square (ZOOM_PX + 2) with an outline (black; red when rule-ready), weakest drawn first,
    the true objects' boxes (green), the rule's cell (ring) / stop cell (X), the robot (centre, facing up)."""
    from mapmad_sim.frames import to_2d

    flat, inside = window_cells(pose, ev["origin"], float(ev["res"]), ev["shape"])
    img = np.empty((ZOOM_N, ZOOM_N, 3), np.uint8)
    img[:] = pv.OUTSIDE_PX
    img[inside & expl[flat]] = pv.EXPL_PX
    img[inside & obst[flat]] = pv.OBST_PX
    img = cv2.resize(img, (ZOOM, ZOOM), interpolation=cv2.INTER_NEAREST)
    v = np.where(inside, value[flat], 0.0)
    rows, cols = np.nonzero(v >= MARK_MIN)
    order = np.argsort(v[rows, cols], kind="stable")
    for r, c in zip(rows[order], cols[order]):
        x0, y0 = c * ZOOM_PX - 1, r * ZOOM_PX - 1
        colour = tuple(int(q) for q in strength_colours(v[r, c]))
        cv2.rectangle(img, (x0, y0), (x0 + ZOOM_PX + 1, y0 + ZOOM_PX + 1), colour, -1)
        edge = pv.READY_C if (v[r, c] >= tau - 1e-12 and ready[flat[r, c]]) else (0, 0, 0)
        cv2.rectangle(img, (x0, y0), (x0 + ZOOM_PX + 1, y0 + ZOOM_PX + 1), edge, 1)
    for inst in episode["target"]["instances"]:
        c, s = inst["centre"], inst["size"]
        corners = [to_2d([c[0] + a * s[0] / 2, 0.0, c[2] + b * s[2] / 2]) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        poly = np.round([zoom_px(p, pose) for p in corners]).astype(np.int32)
        cv2.polylines(img, [poly], True, pv.TRUE_C, 2, cv2.LINE_AA)
    if rule_cell is not None:
        cv2.circle(img, tuple(int(round(q)) for q in zoom_px(rule_cell, pose)), 9, pv.RULE_C, 2, cv2.LINE_AA)
    if stop_cell is not None:
        cv2.drawMarker(img, tuple(int(round(q)) for q in zoom_px(stop_cell, pose)), pv.RULE_C,
                       cv2.MARKER_TILTED_CROSS, 22, 4)
    h = ZOOM // 2
    tri = np.array([[h, h - 14], [h - 9, h + 8], [h + 9, h + 8]], np.int32)
    cv2.fillPoly(img, [tri], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(img, [tri], True, (0, 0, 0), 2, cv2.LINE_AA)
    bar = int(round(1.0 / ZOOM_CELL_M * ZOOM_PX))  # 1 m
    cv2.line(img, (12, ZOOM - 14), (12 + bar, ZOOM - 14), (255, 255, 255), 3)
    return img


def step_numbers(ev: Dict[str, Any], t: int, value: np.ndarray) -> Tuple[List[Tuple[float, float, float, int]],
                                                                         List[Tuple[int, float, int]]]:
    """Frame t: per YOLOE detection (raw score, look = its map value (recogniser.map_score), mark strength = the
    largest layer value now on the cells under its mask, cells), strongest raw first; per true target in the picture
    (id, the same strength on its cells, cells)."""
    def strength(cells: np.ndarray) -> float:
        return float(value[cells].max()) if cells.size else 0.0

    dets = []
    for k in np.flatnonzero(ev["det_t"] == t):
        cells = ev["det_cells"][ev["det_cells_ptr"][k]:ev["det_cells_ptr"][k + 1]]
        dets.append((float(ev["det_raw"][k]), float(ev["det_value"][k]), strength(cells), int(cells.size)))
    trues = []
    for k in np.flatnonzero(ev["true_t"] == t):
        cells = ev["true_cells"][ev["true_cells_ptr"][k]:ev["true_cells_ptr"][k + 1]]
        trues.append((int(ev["true_id"][k]), strength(cells), int(cells.size)))
    return sorted(dets, key=lambda d: -d[0]), trues


_layer_image_marks = layer_image
_map_panel_p13 = pv.map_panel
_draw_cursor_p13 = pv.draw_cursor
_freeze_frame_p13 = pv.freeze_frame


def layer_image_zoom(fv, obst, expl, value, ready, tau):
    _Z.update(obst=obst, expl=expl, value=value, ready=ready, tau=tau)
    return _layer_image_marks(fv, obst, expl, value, ready, tau)


def map_panel_zoom(fv, base, episode, path, yaw, rule_cell, stop_cell):
    _Z.update(rule_cell=rule_cell, stop_cell=stop_cell)
    return _map_panel_p13(fv, base, episode, path, yaw, rule_cell, stop_cell)


def draw_cursor_zoom(frame: np.ndarray, t: int, n: int) -> None:
    """The timeline cursor, then the zoom panel of step t into the frame (draw_cursor is the last hook that sees the
    frame before the text is drawn)."""
    _draw_cursor_p13(frame, t, n)
    ev = _Z["ev"]
    z = zoom_image(ev["pose"][t], ev, _Z["obst"], _Z["expl"], _Z["value"], _Z["ready"], _Z["tau"], _Z["episode"],
                   _Z["rule_cell"], _Z["stop_cell"])
    x, y = ZOOM_XY
    frame[y:y + ZOOM, x:x + ZOOM] = z
    frame[NUM_XY[1]:pv.TL[1] - 4, RIGHT_X:ZOOM_XY[0] - 4] = 0


def caption_zoom(text, episode, name, t, n, events):
    """caption (with the max mark strength ahead) + zoom titles + the step's YOLOE raw scores and mark strengths."""
    caption(text, episode, name, t, n, events)
    x, y = ZOOM_XY
    text.add((x + 8, y + 6), "network's near window (6.4 m, 10 cm cells)", 18)
    text.add((x + 14, y + ZOOM - 42), "1 m", 18)
    text.add((x + 8, y + ZOOM + 6), f"marks ≥ {MARK_MIN:.2f}: purple → yellow by strength", 18)
    text.add((x + 8, y + ZOOM + 30), "red edge = rule-ready; green = true object", 18)
    dets, trues = step_numbers(_Z["ev"], t, _Z["value"])
    nx, ny = NUM_XY
    text.add((nx, ny), f"step {t}: YOLOE boxes on this picture", 20)
    text.add((nx, ny + 24), "raw = YOLOE score; look = its map value;", 16, (200, 200, 200))
    text.add((nx, ny + 44), "mark = layer now on the box's cells", 16, (200, 200, 200))
    rows = [(f"raw {r:.3f} · look {v:.2f} → mark {s:.2f} ({c} cells)",
             pv.HIGH_C if s >= _TAU[0] - 1e-12 else pv.YOLO) for r, v, s, c in dets[:6]]
    if len(dets) > 6:
        rows.append((f"… {len(dets) - 6} more", (200, 200, 200)))
    if not dets:
        rows.append(("no YOLOE box", (160, 160, 160)))
    rows += [(f"TRUE #{i} → mark {s:.2f} ({c} cells)", pv.TRUE_C) for i, s, c in trues[:3]]
    for k, (s, colour) in enumerate(rows):
        text.add((nx + 8, ny + 70 + 22 * k), s, 17, colour)


def freeze_frame_zoom(frame: np.ndarray, kind: str, lines: List[str], colour: Tuple[int, int, int]) -> np.ndarray:
    """Phase 13's freeze with the whole right block (storey, zoom, numbers) kept bright."""
    out = _freeze_frame_p13(frame, kind, lines, colour)
    y0, y1 = pv.HEAD, pv.TL[1] - 4
    out[y0:y1, RIGHT_X - 2:] = frame[y0:y1, RIGHT_X - 2:]
    cv2.rectangle(out, (RIGHT_X - 2, y0), (pv.SIZE[0] - 1, y1 - 1), colour, 4)
    return out


def render_one_zoom(service, boxes, layout, table, arm, episode, k, args, tau, vname):
    ev = load_evidence(layout.root / "evidence" / arm / f"{run_id(episode['episode_id'], k)}.npz")
    _Z.clear()
    _Z.update(ev=ev, episode=episode)
    return render_one(service, boxes, layout, table, arm, episode, k, args, tau, vname)


def patch_zoom() -> None:
    """patch() + the zoom: storey map 440 px at the left of the right half, the near window beside it."""
    patch()
    pv.MAP, pv.MAP_XY = STOREY, (RIGHT_X, pv.HEAD + 4)
    pv.layer_image, pv.map_panel, pv.draw_cursor = layer_image_zoom, map_panel_zoom, draw_cursor_zoom
    pv.caption, pv.freeze_frame, pv.render_one = caption_zoom, freeze_frame_zoom, render_one_zoom


# --- commands ---------------------------------------------------------------------------------------------------------
def cmd_code_check(args) -> None:
    src = (Path(__file__).resolve().parent / "p13_videos.py").read_text().splitlines()
    lines = {name: next(i + 1 for i, l in enumerate(src) if l.startswith(name))
             for name in ("def layer_image", "def legend_panel", "def render_one")}
    call = next(i + 1 for i, l in enumerate(src) if "value = target_at(" in l)
    ans = {"draws_target_layer": True, "p13_videos.py": lines, "layer rebuilt per step (target_at)": call,
           "how": "layer_image colours cells with 0 < value < tau as an orange blend (35-100 % by value / tau), "
                  "value >= tau yellow, rule-ready (>= tau, >= 2 views) red; legend rows 'target layer, 0 … tau', "
                  "'target value ≥ tau', 'rule-ready'. Weak marks (< tau) are dim orange on a grey floor and easy "
                  "to miss; there is no strength scale and no per-step strength number."}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_text_atomic(Path(args.out) / "code_check.json", json.dumps(ans, indent=1) + "\n")
    print(json.dumps(ans, indent=1))


def set_of(layout: RunLayout) -> Dict[str, str]:
    """episode id -> its set name (practice parts)."""
    out = {}
    for name in layout.cfg["sets"]["practice"]["parts"]:
        eps, _ = load_episodes(layout.base / layout.cfg["sets"][name]["folder"] / "episodes.json")
        out.update({e["episode_id"]: name for e in eps})
    return out


def _counts_one(job: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    ev_file, rid = job
    return rid, mark_sightings(load_evidence(Path(ev_file)))


def cmd_counts(args) -> None:
    layout = RunLayout.load(args.config, subdir=args.subdir)
    res: Dict[str, Any] = {}
    lines = ["# 14.5 mark counts (report only)", "",
             "Object sighting = a frame with a YOLOE box on a true target (IoU >= 0.3); its strength = the largest "
             "target-layer value on the object's cells in that frame and the next 20 steps; marked = strength >= 0.45 "
             "(today's tau). Wrong sighting = a YOLOE box on nothing true (IoU < 0.3 with every true box); strength = "
             "the same over the cells under its mask.", "",
             "| arm | runs | true sightings | marked | share | strength p25 / p50 / p75 | wrong sightings | marked | "
             "share | strength p50 / p90 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in args.arms:
        files = sorted((layout.root / "evidence" / arm).glob("*.npz"))
        with Pool(args.workers) as pool:
            per = dict(pool.map(_counts_one, [(str(f), f.name[:-4]) for f in files], chunksize=2))
        ts = [x["strength"] for r in per.values() for x in r["true"]]
        ws = [x["strength"] for r in per.values() for x in r["wrong"]]
        tm = sum(x["marked"] for r in per.values() for x in r["true"])
        wm = sum(x["marked"] for r in per.values() for x in r["wrong"])
        runs_hit = sum(bool(r["true"]) for r in per.values())
        runs_marked = sum(any(x["marked"] for x in r["true"]) for r in per.values())
        q = lambda a, p: float(np.percentile(a, p)) if a else float("nan")  # noqa: E731
        res[arm] = {"runs": len(files), "true_sightings": len(ts), "true_marked": tm, "wrong_sightings": len(ws),
                    "wrong_marked": wm, "runs_with_true_hit": runs_hit, "runs_with_marked_hit": runs_marked,
                    "true_strength_quartiles": [q(ts, 25), q(ts, 50), q(ts, 75)],
                    "wrong_strength_p50_p90": [q(ws, 50), q(ws, 90)],
                    "true_strength_hist": np.histogram(ts, bins=10, range=(0, 1))[0].tolist(),
                    "wrong_strength_hist": np.histogram(ws, bins=10, range=(0, 1))[0].tolist()}
        lines.append(f"| {arm} | {len(files)} | {len(ts)} | {tm} | {tm / max(len(ts), 1):.3f} | "
                     f"{q(ts, 25):.2f} / {q(ts, 50):.2f} / {q(ts, 75):.2f} | {len(ws)} | {wm} | "
                     f"{wm / max(len(ws), 1):.3f} | {q(ws, 50):.2f} / {q(ws, 90):.2f} |")
    lines += ["", "Runs with >= 1 true sighting / with >= 1 marked true sighting: " +
              "; ".join(f"{a} {r['runs_with_true_hit']} / {r['runs_with_marked_hit']}" for a, r in res.items()),
              "", "Strength histograms (10 bins of 0.1): " + json.dumps({a: {"true": r["true_strength_hist"],
                                                                             "wrong": r["wrong_strength_hist"]}
                                                                         for a, r in res.items()})]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "mark_counts.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(out / "mark_counts.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def cmd_pick(args) -> None:
    layout = RunLayout.load(args.config, subdir=args.subdir)
    sets = set_of(layout)
    bench = layout.base / "stopbench" / "runs" / "M21b"
    hit, wrong = [], []
    for f in sorted((layout.root / "evidence" / "M21b").glob("*.npz")):
        rid = f.name[:-4]
        if mark_sightings(load_evidence(f))["true"]:
            hit.append(rid)
        b = json.loads((bench / f"{rid}.json").read_text())
        if b["rules"]["today"]["wrong_object"]:
            wrong.append(rid)
    rng = np.random.default_rng(PICK_SEED)
    ph = [hit[i] for i in sorted(rng.choice(len(hit), min(N_PICK, len(hit)), replace=False))]
    rest = [r for r in wrong if r not in ph]
    pw = [rest[i] for i in sorted(rng.choice(len(rest), min(N_PICK, len(rest)), replace=False))]
    picks = [{"name": f"true_hit_{k + 1}", "episode": parse_run_id(r)[0], "set": sets[parse_run_id(r)[0]]}
             for k, r in enumerate(ph)] + \
            [{"name": f"wrong_object_stop_{k + 1}", "episode": parse_run_id(r)[0], "set": sets[parse_run_id(r)[0]]}
             for k, r in enumerate(pw)]
    doc = {"seed": PICK_SEED, "arm": "M21b", "subdir": args.subdir, "runs_with_true_hit": len(hit),
           "runs_with_wrong_object_stop": len(wrong), "picks": picks,
           "note": "seeded picks, NOT chosen for clarity; representative within each group"}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_text_atomic(Path(args.out) / "picks.json", json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))


def cmd_render(args) -> None:
    patch_zoom() if args.zoom else patch()
    picks = json.loads((args.picks or Path(args.out) / "picks.json").read_text())["picks"]
    k, n = (int(x) for x in args.shard.split("/"))
    picks = [p for i, p in enumerate(picks) if i % n == k]
    for set_name in sorted({p["set"] for p in picks}):
        args.set = set_name
        args.seed_index = 0
        pv.render_all(args, [(f"{p['name']}__{args.arm}", args.arm, p["episode"]) for p in picks if p["set"] == set_name])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["code-check", "counts", "pick", "render"])
    p.add_argument("--config", type=Path)
    p.add_argument("--subdir", default="runs/baseline")
    p.add_argument("--arms", nargs="+", default=["M21b", "M21bh"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--recog-port", type=int)
    p.add_argument("--itm-port", type=int)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--zoom", action="store_true", help="render: add the network's near window beside the storey")
    p.add_argument("--picks", type=Path, help="render: the picks file (default <out>/picks.json)")
    p.add_argument("--arm", default="M21b", help="render: the arm whose runs are drawn (Phase 14b-2: M21b_w4_d_am)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"code-check": cmd_code_check, "counts": cmd_counts, "pick": cmd_pick, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
