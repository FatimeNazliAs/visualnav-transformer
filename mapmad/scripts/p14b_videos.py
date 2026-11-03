"""Phase 14b-1 step 14b1.5: the 8 Phase 14a mark-check runs (videos_marks/picks.json, M21b) re-rendered with the
fixed marks: the network's near window twice, side by side -- left the 14a marks (today's fused layer, as the 14a
zoom videos), right the marks of the recommended stop-module variant (stop_variants.layer_at: today's fusion = mean
over looks, of that variant's projection and scores). Report only; the runs are not changed. Inside naz_mapmad_habitat
(one YOLOE sidecar per shard, as p14_videos.py render):

    python mapmad/scripts/p14b_videos.py --config mapmad/configs/p14a_finish.yaml --marks-config \
        mapmad/configs/p14b1_marks.yaml --picks <p14a>/videos_marks/picks.json --gpu G --recog-port P \
        --out <p14b1>/videos_marks_fixed [--shard k/n] [--variant det|clu_k1_m20|cal]

Frame = p14_videos.py render --zoom (Phase 13's checkable video: camera, boxes, timeline, checks), with the storey map
replaced by the old near window and the near windows drawn at 7 px per cell (64 x 64 cells of 10 cm, forward up);
marks >= 0.05 as enlarged squares (purple -> yellow by strength); the variant's name above the right window. The
variant = RESULTS.json's overall recommendation (its per-category c_cat variant for c_cat rows) unless --variant.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p13_videos as pv  # noqa: E402
import p14_videos as zv  # noqa: E402

from mapmad_sim import stop_evidence  # noqa: E402
from mapmad_sim.run_layout import RunLayout, run_id  # noqa: E402
from mapmad_sim.stop_variants import RunGeometry, Variant, VariantReplay, variant_evidence  # noqa: E402

PX = 7
NEW: Dict[str, Any] = {}


class LayerStepper:
    """The variant's fused layer (mean over looks) at frame t, advanced frame by frame (restarts if t goes back)."""

    def __init__(self, ev: Dict[str, np.ndarray]) -> None:
        self.ev = ev
        self.geo = RunGeometry(ev) if np.any(ev.get("det_tent", [])) else None
        self.rep: Optional[VariantReplay] = None

    def at(self, t: int) -> np.ndarray:
        if self.rep is None or self.rep.t > t:
            self.rep = VariantReplay(self.ev, geo=self.geo)
        while self.rep.t < t:
            self.rep.advance()
        return (self.rep.total / np.maximum(self.rep.count, 1)).astype(np.float32)


def recommended(marks_out: Path) -> Dict[str, Any]:
    res = json.loads((marks_out / "stopbench" / "RESULTS.json").read_text())
    o = res["recommended"]["overall"]
    sel = res["rows"][o["family"]][o["row"]]
    return {"variant": sel["variant"], "per_category": sel["per_category"], "label": f"{o['family']} · {o['row']}",
            "key": sel["key"]}


def draw_cursor_compare(frame: np.ndarray, t: int, n: int) -> None:
    """Old near window over the storey map, new one beside it (after Phase 13's cursor)."""
    zv._draw_cursor_p13(frame, t, n)
    z = zv._Z
    ev = z["ev"]
    pose = ev["pose"][t]
    old = zv.zoom_image(pose, ev, z["obst"], z["expl"], z["value"], z["ready"], z["tau"], z["episode"],
                        z["rule_cell"], z["stop_cell"])
    new = zv.zoom_image(pose, ev, z["obst"], z["expl"], NEW["stepper"].at(t), np.zeros_like(z["ready"]), z["tau"],
                        z["episode"], None, None)
    y = pv.HEAD + 4
    frame[y:y + zv.ZOOM, zv.RIGHT_X:zv.RIGHT_X + zv.ZOOM] = old
    x2 = zv.RIGHT_X + zv.ZOOM + 8
    frame[y:y + zv.ZOOM, x2:x2 + zv.ZOOM] = new
    frame[zv.NUM_XY[1]:pv.TL[1] - 4, zv.RIGHT_X:pv.SIZE[0]] = 0


def caption_compare(text, episode, name, t, n, events):
    y = pv.HEAD + 4
    storey = {"whole storey, north up (fixed)", "1 m"}  # the storey map's labels: that panel is replaced here
    text.items = [it for it in text.items if not (it[1] in storey and it[0][0] < zv.RIGHT_X + 40)]
    zv.caption(text, episode, name, t, n, events)  # Phase 13 caption + max mark strength ahead (14a rule value)
    x2 = zv.RIGHT_X + zv.ZOOM + 8
    text.add((zv.RIGHT_X + 8, y + 6), "OLD marks (14a)", 18)
    text.add((x2 + 8, y + 6), f"NEW marks: {NEW['label']}", 18)
    text.add((x2 + 8, y + 28), NEW["variant"][:52], 15, (220, 220, 220))
    nx, ny = zv.NUM_XY
    text.add((nx, ny), "near windows: 6.4 m, 10 cm cells, forward up; marks >= 0.05 purple -> yellow by strength", 17)
    text.add((nx, ny + 22), "green = true object; red edge (old) = rule-ready at tau 0.45; ring / X = 14a rule cell / stop", 17)
    dets, trues = zv.step_numbers(zv._Z["ev"], t, zv._Z["value"])
    new_v = NEW["stepper"].at(t)
    rows = []
    for r, v, s, c in dets[:5]:
        rows.append((f"YOLOE raw {r:.3f}: old mark {s:.2f} ({c} cells)", pv.YOLO))
    if not dets:
        rows.append(("no YOLOE box", (160, 160, 160)))
    tv = NEW["truth_cells"].get(t, [])
    for (i, s, c), cells in zip(trues[:3], tv[:3]):
        ns = float(new_v[cells].max()) if cells.size else 0.0
        rows.append((f"TRUE #{i}: old mark {s:.2f} · new mark {ns:.2f}", pv.TRUE_C))
    for k, (s, colour) in enumerate(rows):
        text.add((nx + 8, ny + 50 + 22 * k), s, 17, colour)


def render_one_compare(service, boxes, layout, table, arm, episode, k, args, tau, vname):
    rid = run_id(episode["episode_id"], k)
    ev = stop_evidence.load(layout.root / "evidence" / arm / f"{rid}.npz")
    proj = stop_evidence.load(Path(args.marks_out) / "replay" / "baseline" / arm / f"{rid}.npz")
    rec = NEW["rec"]
    name = (rec["per_category"] or {}).get(episode["target"]["category"], rec["variant"])
    vev = variant_evidence(ev, proj, Variant.parse(name), NEW["T"])
    NEW.update(stepper=LayerStepper(vev), variant=name)
    tc: Dict[int, list] = {}
    for q, t in enumerate(ev["true_t"].tolist()):
        tc.setdefault(t, []).append(ev["true_cells"][ev["true_cells_ptr"][q]:ev["true_cells_ptr"][q + 1]])
    NEW["truth_cells"] = tc
    return zv.render_one_zoom(service, boxes, layout, table, arm, episode, k, args, tau, vname)


def patch_compare() -> None:
    zv.patch_zoom()
    zv.ZOOM_PX, zv.ZOOM = PX, zv.ZOOM_N * PX  # 448 px per window
    zv.zoom_px.__defaults__ = (zv.ZOOM_N, zv.ZOOM_CELL_M, PX)
    zv.NUM_XY = (zv.RIGHT_X, pv.HEAD + 4 + zv.ZOOM + 10)
    pv.draw_cursor, pv.caption, pv.render_one = draw_cursor_compare, caption_compare, render_one_compare


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="the runs' config (p14a_finish.yaml)")
    p.add_argument("--marks-config", type=Path, required=True, help="p14b1_marks.yaml (replay, RESULTS.json, T)")
    p.add_argument("--subdir", default="runs/baseline")
    p.add_argument("--picks", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--recog-port", type=int, required=True)
    p.add_argument("--itm-port", type=int)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--variant", help="a stop_variants.Variant name instead of the recommendation")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    marks_out = RunLayout.load(args.marks_config).base
    args.marks_out = str(marks_out)
    NEW["T"] = float(json.loads((marks_out / "calib" / "calib.json").read_text())["T"])
    NEW["rec"] = {"variant": args.variant, "per_category": None, "label": "chosen variant"} if args.variant \
        else recommended(marks_out)
    NEW["label"] = NEW["rec"]["label"]
    patch_compare()
    picks = json.loads(args.picks.read_text())["picks"]
    k, n = (int(x) for x in args.shard.split("/"))
    picks = [q for i, q in enumerate(picks) if i % n == k]
    args.out.mkdir(parents=True, exist_ok=True)
    for set_name in sorted({q["set"] for q in picks}):
        args.set, args.seed_index = set_name, 0
        pv.render_all(args, [(f"{q['name']}__M21b", "M21b", q["episode"]) for q in picks if q["set"] == set_name])


if __name__ == "__main__":
    main()
