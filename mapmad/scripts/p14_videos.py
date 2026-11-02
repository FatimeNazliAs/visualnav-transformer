"""Phase 14a step 14.5: the mark check (report only). Inside naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/p14_videos.py code-check --out <p14a>/videos_marks           # does p13_videos draw the layer?
    python mapmad/scripts/p14_videos.py counts --config mapmad/configs/p14a_finish.yaml --subdir runs/baseline \
        --arms M21b M21bh --workers 16 --out <p14a>/videos_marks                       # mark_counts.json + .md
    python mapmad/scripts/p14_videos.py pick --config ... --subdir runs/baseline --out <p14a>/videos_marks
    python mapmad/scripts/p14_videos.py render --config ... --subdir runs/baseline --gpu G --recog-port P \
        --out <p14a>/videos_marks [--shard k/n]
    python mapmad/scripts/p13_videos.py check --out <p14a>/videos_marks             # boxes + poses (Phase 13 check)

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
from typing import Any, Dict, List, Optional, Tuple

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
    patch()
    picks = json.loads((Path(args.out) / "picks.json").read_text())["picks"]
    k, n = (int(x) for x in args.shard.split("/"))
    picks = [p for i, p in enumerate(picks) if i % n == k]
    for set_name in sorted({p["set"] for p in picks}):
        args.set = set_name
        args.seed_index = 0
        pv.render_all(args, [(f"{p['name']}__M21b", "M21b", p["episode"]) for p in picks if p["set"] == set_name])


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
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    {"code-check": cmd_code_check, "counts": cmd_counts, "pick": cmd_pick, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
