"""Phase 9 scoring from the saved raw outputs (rules: p9_rules.md; maths: mapmad_sim/photo_scoring.py), in
naz_mapmad_habitat_p9 (CPU only):

    python mapmad/scripts/p9_score.py tune --model owlv2       # labels + half-A threshold (before the timing pass)
    python mapmad/scripts/p9_score.py score --model owlv2      # threshold (half A), metrics (half B), OVON
    python mapmad/scripts/p9_score.py choose                   # choice rule -> photo_test/choice.{json,md}, tables
    python mapmad/scripts/p9_score.py sheets --model owlv2     # 8 hits, 8 misses, 8 false alarms (seeded picks)

score writes photo_test/<model>/scores.json and labels.json; the OVON numbers (open-vocabulary models, at the
model's 6-category threshold, all OVON items) go to ovon_test/<model>.json. Re-running gives identical files
(§0.8 re-check: re-score from the raw outputs).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapmad_sim import config, photo_labels as pl  # noqa: E402
from mapmad_sim import photo_scoring as ps  # noqa: E402
from mapmad_sim.categories import load_map  # noqa: E402

MODELS = config.CONFIG_DIR / "p9_models.yaml"
PHASE12_CALLS = 5_000_000
MAX_LATENCY_S = 0.25


def root() -> Path:
    return config.paths()["outputs"] / "p9_recognisers"


def model_info(key: str) -> Dict[str, Any]:
    if key == "truth":
        return {"name": "Simulator truth (Phase 8 rule)", "kind": "truth", "family": "truth", "output": "mask"}
    return yaml.safe_load(MODELS.read_text())["models"][key]


def tables() -> Dict[str, pl.LabelTable]:
    fr = root() / "frames"
    six = json.loads((fr / "label_tables.json").read_text())["six"]
    ov = json.loads((fr / "ovon_mapping.json").read_text())["mapped"]
    out = {k: pl.LabelTable(k, t["prompt"], t["counted"], t["ignore"]) for k, t in six.items()}
    out.update({f"ovon:{k}": pl.LabelTable(k, t["prompt"], t["counted"], t["ignore"]) for k, t in ov.items()})
    return out


def load_items(kind: str) -> List[Dict[str, Any]]:
    fr = root() / "frames"
    items = [{**i, "table": i["key"], "set": "six"} for i in json.loads((fr / "items_six.json").read_text())]
    if kind not in ("fixed",):
        items += [{**i, "table": f"ovon:{i['key']}", "set": "ovon"} for i in json.loads((fr / "items_ovon.json").read_text())]
    return items


def label_all(key: str, items: List[Dict[str, Any]], tabs: Dict[str, pl.LabelTable]) -> List[ps.Det]:
    fr = root() / "frames"
    frames = {f["frame_id"]: f for f in json.loads((fr / "frames.json").read_text())}
    raw = ps.load_raw(root() / "photo_test" / key / "raw", (240, 320))
    missing = [i["item_id"] for i in items if i["item_id"] not in raw]
    if missing:
        raise SystemExit(f"{key}: {len(missing)} items without raw output, e.g. {missing[:3]}")
    dets: List[ps.Det] = []
    cache: Dict[str, ps.FrameData] = {}
    for it in sorted(items, key=lambda i: i["frame_id"]):
        fid = it["frame_id"]
        if fid not in cache:
            cache = {fid: ps.FrameData.load(frames[fid], fr / "img")}  # items are grouped by frame
        fd = cache[fid]
        truth = ps.item_truth(fd.instances, tabs[it["table"]])
        r = raw[it["item_id"]]
        cam = np.asarray(frames[fid]["camera_tf"])[:3, 3]
        dets += ps.label_item(it["item_id"], fd, truth, r["scores"], r["boxes"], r["points"], r["masks"], cam)
    return dets


def infos(items: List[Dict[str, Any]], tabs: Dict[str, pl.LabelTable]) -> Dict[str, ps.ItemInfo]:
    fr = root() / "frames"
    frames = {f["frame_id"]: f for f in json.loads((fr / "frames.json").read_text())}
    from mapmad_sim.photo_frames import instances_from_json
    out = {}
    for it in items:
        inst = instances_from_json(frames[it["frame_id"]]["instances"])
        out[it["item_id"]] = ps.ItemInfo(it["item_id"], it["table"], it["status"], it["half"], it["src"],
                                         ps.item_truth(inst, tabs[it["table"]]).counted)
    return out


def latency(key: str, kind: str) -> Dict[str, Any]:
    """Median / p95 round trip (s) and peak VRAM: timing pass, or (VLMs) the raw pass on half-B 6-category items."""
    d = root() / "photo_test" / key
    stats = json.loads((d / "raw" / "stats.json").read_text()) if (d / "raw" / "stats.json").exists() else {}
    if kind == "truth":
        return {"median_s": 0.0, "p95_s": 0.0, "peak_vram_mb": 0.0, "source": "n/a (simulator)"}
    if kind == "vlm":
        half_b = {i["item_id"] for i in json.loads((root() / "frames" / "items_six.json").read_text()) if i["half"] == "B"}
        rows = []
        for p in sorted((d / "raw").glob("part_*.npz")):
            z = np.load(p)
            rows += list(zip(z["item_id"].tolist(), z["rtt_ms"].tolist()))
        rtt = np.array([ms for k, (item, ms) in enumerate(rows) if k >= 50 and item in half_b])
        return {"median_s": float(np.median(rtt)) / 1e3, "p95_s": float(np.percentile(rtt, 95)) / 1e3,
                "peak_vram_mb": stats.get("peak_vram_mb"), "source": f"raw pass, {len(rtt)} half-B items after 50 warm-up"}
    t = json.loads((d / "timing.json").read_text())
    return {"median_s": t["median_ms"] / 1e3, "p95_s": t["p95_ms"] / 1e3,
            "peak_vram_mb": max(t.get("peak_vram_mb", 0.0), stats.get("peak_vram_mb", 0.0)),
            "source": f"timing pass at t, {t['items'] - t['warmup']} items after {t['warmup']} warm-up"}


def labels(key: str, items: List[Dict[str, Any]], tabs: Dict[str, pl.LabelTable], fresh: bool) -> List[ps.Det]:
    """The detection labels of a model: computed from the raw outputs (cached in labels.json; --fresh recomputes)."""
    path = root() / "photo_test" / key / "labels.json"
    if path.exists() and not fresh:
        return [ps.Det(**{**d, "hits": tuple(d["hits"])}) for d in json.loads(path.read_text())]
    dets = label_all(key, items, tabs)
    ps.write_json(path, [vars(x) for x in dets])
    return dets


def cmd_tune(key: str, fresh: bool) -> None:
    """Half-A threshold only (needed before the timing pass) -> photo_test/<model>/threshold.json."""
    kind = model_info(key)["kind"]
    tabs = tables()
    items = load_items(kind)
    inf = infos(items, tabs)
    a = [inf[i["item_id"]] for i in items if i["set"] == "six" and i["half"] == "A"]
    t, f1_a = ps.tune(labels(key, items, tabs, fresh), a, kind in ("vlm", "truth"))
    ps.write_json(root() / "photo_test" / key / "threshold.json", {"threshold": t, "f1_half_a": f1_a})
    print(f"{key}: threshold {t} (half-A F1 {f1_a:.4f})")


def cmd_score(key: str, fresh: bool = False) -> None:
    info = model_info(key)
    kind = info["kind"]
    vlm = kind == "vlm"
    tabs = tables()
    items = load_items(kind)
    dets = labels(key, items, tabs, fresh)
    inf = infos(items, tabs)
    six = [inf[i["item_id"]] for i in items if i["set"] == "six"]
    a = [i for i in six if i.half == "A"]
    b = [i for i in six if i.half == "B"]
    t, f1_a = ps.tune(dets, a, vlm or kind == "truth")
    kept_a = sorted(d.score for d in dets if d.item in {i.item_id for i in a} and d.score >= t)
    p99 = float(np.percentile(kept_a, 99)) if kept_a else 1.0
    out: Dict[str, Any] = {"model": key, **info, "threshold": t, "f1_half_a": f1_a, "p99_kept_half_a": p99,
                           "half_b": ps.metrics(dets, b, t, with_ap=not vlm), "per_category": {},
                           "per_category_thresholds_report_only": {}, "by_source": {},
                           "recall_breakdown": ps.recall_breakdown(dets, b, t),
                           "map_hits_by_distance": {}}
    for c in pl.CATEGORIES:
        bc = [i for i in b if i.key == c]
        out["per_category"][c] = ps.metrics(dets, bc, t, with_ap=not vlm)
        tc, _ = ps.tune(dets, [i for i in a if i.key == c], vlm or kind == "truth")
        out["per_category_thresholds_report_only"][c] = {"threshold": tc, "half_b": ps.metrics(dets, bc, tc, with_ap=False)}
    for src in ("rnd", "drv"):
        out["by_source"][src] = ps.metrics(dets, [i for i in b if i.src == src], t, with_ap=not vlm)
    kept_b = [d for d in dets if d.item in {i.item_id for i in b} and d.score >= t and d.map in ("hit", "miss")]
    for lo, hi in ps.DIST_BINS + ((4.0, 99.0),):
        sel = [d for d in kept_b if lo <= d.dist_m < hi]
        out["map_hits_by_distance"][f"{lo:g}-{hi:g}m"] = {"n": len(sel), "H": (sum(d.map == "hit" for d in sel) / len(sel)) if sel else None}
    cat_of = {pl.normalise(n): c for n, c in load_map(config.CONFIG_DIR / "objectnav_categories.yaml").items()}
    out["confusions"] = ps.confusions(dets, b, t, cat_of)
    out["latency"] = latency(key, kind)
    out["phase12_cost_h"] = PHASE12_CALLS * out["latency"]["median_s"] / 3600.0
    d = root() / "photo_test" / key
    ps.write_json(d / "scores.json", out)
    if kind in ("open", "vlm", "truth"):
        ps.write_json(root() / "ovon_test" / f"{key}.json", ovon_scores(dets, items, inf, t, vlm))
    print(f"{key}: t={t} F1(A)={f1_a:.4f} F1(B)={out['half_b']['F1']:.4f} median {out['latency']['median_s']:.3f} s")


def ovon_scores(dets: List[ps.Det], items: List[Dict[str, Any]], inf: Dict[str, ps.ItemInfo], t: float,
                vlm: bool) -> Dict[str, Any]:
    """OVON photo test at the 6-category threshold, all OVON items (no tuning on OVON); coverage per split."""
    mp = json.loads((root() / "frames" / "ovon_mapping.json").read_text())
    min_pos = 5
    ov = [i for i in items if i["set"] == "ovon"]
    pos = {}
    for i in ov:
        pos[i["key"]] = pos.get(i["key"], 0) + (i["status"] == "positive")
    covered = sorted(k for k, n in pos.items() if n >= min_pos)
    out = {"threshold": t, "all": ps.metrics(dets, [inf[i["item_id"]] for i in ov], t, with_ap=not vlm),
           "covered_categories": covered, "splits": {}}
    for split, names in mp["splits"].items():
        cats = [n for n in names if n in covered]
        sel = [inf[i["item_id"]] for i in ov if i["key"] in cats]
        out["splits"][split] = {"categories": len(names), "mapped": sum(n in mp["mapped"] for n in names),
                                "covered": len(cats), **ps.metrics(dets, sel, t, with_ap=not vlm)}
    return out


def fmt(v: Any, nd: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def cmd_choose() -> None:
    cfg = yaml.safe_load(MODELS.read_text())
    status = json.loads((root() / "photo_test" / "model_status.json").read_text())
    rows: Dict[str, Dict[str, Any]] = {}
    for key, spec in cfg["models"].items():
        p = root() / "photo_test" / key / "scores.json"
        if status.get(key, {}).get("status") == "ran" and p.exists():
            s = json.loads(p.read_text())
            rows[key] = {"F1": s["half_b"]["F1"], "median_s": s["latency"]["median_s"], "kind": spec["kind"],
                         "family": spec["family"], "ran": True, "threshold": s["threshold"], "p99": s["p99_kept_half_a"]}
        else:
            rows[key] = {"F1": float("nan"), "median_s": float("nan"), "kind": spec["kind"], "family": spec["family"],
                         "ran": False, "reason": status.get(key, {}).get("reason", "not run")}
    ch = ps.choose(rows, MAX_LATENCY_S)
    for r in ("R1", "R2"):
        m = ch[r]
        if m is not None:
            ch[f"{r}_detail"] = {"model": m, "name": cfg["models"][m]["name"], "threshold": rows[m]["threshold"],
                                 "p99": rows[m]["p99"], "median_s": rows[m]["median_s"], "F1": rows[m]["F1"],
                                 "family": rows[m]["family"], "kind": rows[m]["kind"]}
    ch["rows"] = rows
    ps.write_json(root() / "photo_test" / "choice.json", ch)
    lines = ["# Phase 9 choice (p9_rules.md §choice, applied mechanically)", "",
             f"- **R1 = {ch['R1']}**, **R2 = {ch['R2']}**",
             f"- qualifying (median <= {MAX_LATENCY_S} s, open vocabulary): {', '.join(ch['qualifying']) or 'none'}"]
    lines += [f"- fallback: {f}" for f in ch["fallback_flags"]] + [f"- {l}" for l in ch["log"]]
    (root() / "photo_test" / "choice.md").write_text("\n".join(lines) + "\n")
    write_tables(cfg, rows)
    print("\n".join(lines))


def write_tables(cfg: Dict[str, Any], rows: Dict[str, Dict[str, Any]]) -> None:
    hdr = "| Model | Family | t | F1 (map) | H | R (map) | AP50 | P | R (pix) | FA/free | median s | p95 s | VRAM MB | Phase 12 h |"
    out = ["# Phase 9 photo test, half B, 6 categories", "", hdr, "|" + "---|" * 14]
    keys = list(cfg["models"]) + ["truth"]
    for k in keys:
        p = root() / "photo_test" / k / "scores.json"
        if not p.exists():
            r = rows.get(k, {})
            out.append(f"| {k} | {r.get('family', '')} | skipped: {r.get('reason', '')} |" + " |" * 11)
            continue
        s = json.loads(p.read_text())
        b, lat = s["half_b"], s["latency"]
        out.append(f"| {s['name']} | {s['family']} | {s['threshold']:.2f} | {fmt(b['F1'])} | {fmt(b['H'])} | {fmt(b['R'])} | "
                   f"{fmt(b['ap50'])} | {fmt(b['precision'])} | {fmt(b['recall'])} | {fmt(b['fa_per_free'])} | "
                   f"{fmt(lat['median_s'])} | {fmt(lat['p95_s'])} | {fmt(lat['peak_vram_mb'], 0)} | {fmt(s['phase12_cost_h'], 1)} |")
    out += ["", "## Per category (map F1 at the model's threshold)", "",
            "| Model | " + " | ".join(pl.CATEGORIES) + " |", "|---|" + "---|" * len(pl.CATEGORIES)]
    for k in keys:
        p = root() / "photo_test" / k / "scores.json"
        if p.exists():
            s = json.loads(p.read_text())
            out.append(f"| {k} | " + " | ".join(fmt(s["per_category"][c]["F1"]) for c in pl.CATEGORIES) + " |")
    (root() / "photo_test" / "table.md").write_text("\n".join(out) + "\n")
    ov = ["# OVON photo test (all OVON items, model's 6-category threshold)", "",
          "| Model | F1 all | " + " | ".join(f"{s} F1 (covered/mapped/all)" for s in ("val_seen", "val_seen_synonyms", "val_unseen")) + " |",
          "|---|---|---|---|---|"]
    for k in keys:
        p = root() / "ovon_test" / f"{k}.json"
        if p.exists():
            o = json.loads(p.read_text())
            ov.append(f"| {k} | {fmt(o['all']['F1'])} | " + " | ".join(
                f"{fmt(o['splits'][s]['F1'])} ({o['splits'][s]['covered']}/{o['splits'][s]['mapped']}/{o['splits'][s]['categories']})"
                for s in ("val_seen", "val_seen_synonyms", "val_unseen")) + " |")
    (root() / "ovon_test" / "table.md").write_text("\n".join(ov) + "\n")


def cmd_sheets(key: str, seed: int = 20261021) -> None:
    """8 hits, 8 misses (positive items with no kept map hit), 8 false alarms (kept pixel FP on target-free items),
    half B, seeded picks; kept detections outlined in red (points as dots), counted truth instances in yellow."""
    s = json.loads((root() / "photo_test" / key / "scores.json").read_text())
    t = s["threshold"]
    dets = [ps.Det(**{**d, "hits": tuple(d["hits"])}) for d in json.loads((root() / "photo_test" / key / "labels.json").read_text())]
    items = {i["item_id"]: i for i in load_items(model_info(key)["kind"]) if i["set"] == "six" and i["half"] == "B"}
    kept = [d for d in dets if d.item in items and d.score >= t]
    rng = np.random.default_rng(seed)
    hits = sorted({d.item for d in kept if d.map == "hit"})
    fas = sorted({d.item for d in kept if d.pix == "FP" and items[d.item]["status"] == "free"})
    hit_items = {d.item for d in kept if d.map == "hit"}
    misses = sorted(i for i, it in items.items() if it["status"] == "positive" and i not in hit_items)
    pick = lambda xs: [xs[k] for k in sorted(rng.permutation(len(xs))[:8])] if xs else []
    raw = ps.load_raw(root() / "photo_test" / key / "raw", (240, 320))
    fr = root() / "frames"
    frames = {f["frame_id"]: f for f in json.loads((fr / "frames.json").read_text())}
    tabs = tables()
    tiles = []
    for group, ids in (("hit", pick(hits)), ("miss", pick(misses)), ("false alarm", pick(fas))):
        for item in ids:
            it = items[item]
            img = cv2.imread(str(fr / "img" / f"{it['frame_id']}_rgb.png"))
            sem = cv2.imread(str(fr / "img" / f"{it['frame_id']}_sem.png"), cv2.IMREAD_UNCHANGED)
            r = raw[item]
            for k in np.nonzero(r["scores"] >= t)[0]:
                m = r["masks"][k]
                if m is not None:
                    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cnts, -1, (0, 0, 255), 1)
                elif not np.isnan(r["points"][k]).any():
                    cv2.circle(img, tuple(int(v) for v in r["points"][k]), 4, (0, 0, 255), -1)
            from mapmad_sim.photo_frames import instances_from_json
            truth = ps.item_truth(instances_from_json(frames[it["frame_id"]]["instances"]), tabs[it["key"]])
            for inst in truth.counted:
                cnts, _ = cv2.findContours((sem == inst.id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 255), 1)
            tiles.append(annotate(img, f"{group}: {it['prompt']} {item[4:]}"))
    while len(tiles) < 24:
        tiles.append(np.zeros((240, 320, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[r:r + 8]) for r in range(0, 24, 8)])
    cv2.imwrite(str(root() / "sheets" / f"{key}.png"), sheet)


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (320, 14), (0, 0, 0), -1)
    cv2.putText(img, text[:52], (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["tune", "score", "choose", "sheets"])
    p.add_argument("--model")
    p.add_argument("--fresh", action="store_true", help="recompute the labels from the raw outputs")
    a = p.parse_args()
    if a.step == "tune":
        cmd_tune(a.model, a.fresh)
    elif a.step == "score":
        cmd_score(a.model, a.fresh)
    elif a.step == "choose":
        cmd_choose()
    else:
        cmd_sheets(a.model)


if __name__ == "__main__":
    main()
