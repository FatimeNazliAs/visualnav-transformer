"""Phase 14b-2 tables and checks (NoMaD side, numpy only; rules p14b2_rules.md; config p14b2_recog.yaml):

    python -m vint_train.mapmad.closed_loop.p14b2_tables photo --config <p14b2_recog.yaml>     # R2 choice + table
    python -m vint_train.mapmad.closed_loop.p14b2_tables r3 --config ...                        # R3 stop bench vs base
    python -m vint_train.mapmad.closed_loop.p14b2_tables r4 --config ...                        # R4 driving cells
    python -m vint_train.mapmad.closed_loop.p14b2_tables plain --config ...                     # the first table
    python -m vint_train.mapmad.closed_loop.p14b2_tables off --config ...                       # row 7: switch off = 14a
    python -m vint_train.mapmad.closed_loop.p14b2_tables gate --config ...                      # draft G14b2

photo: base YOLOE (Phase 9 photo_test/yoloe, read only) vs LP / FT (<out>/photo_test/yoloe_{lp,ft}); pick = the
highest half-A map F1 (ties -> base, then LP); a fine-tuned pick worse than base is impossible by construction (then
base is picked = the pre-set fallback "R3 / R4 use base YOLOE"). -> photo/choice.json, tables/photo.md.
r3: per detector (stopbench_yoloe_<v>) and rule family, rows base (14a marks) and f1med (F1-med 0.5), map scores
(+ F2 with the refitted T, report only), each picked on half A; vs the same row with base YOLOE (14b-1 stopbench);
keep rule as 14b-1 (half B, +0.02 success, pooled STOP precision not lower). Detection share on a target (box IoU >=
0.3 with a true box of the frame) from the evidence. -> tables/r3.{md,json}.
r4: cells base_off (14a W4 runs, read only), base_on (runs/am), pick_off (runs/det), pick_on (runs/detam) x M21b,
M21bh, N: per-run metrics from the logs (score_log at tau 0.45) -> means per half, paired diffs vs base_off (home CI),
pooled STOP precision change (home CI), map effect M21b - M21bh per cell. -> tables/r4.{md,json}.
"""

import argparse
import filecmp
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic
from mapmad_sim.stop_evidence import load as load_evidence
from mapmad_sim.stop_variants import Variant
from vint_train.mapmad.closed_loop import p14b_bench as pb
from vint_train.mapmad.closed_loop.exam_stats import paired

ARMS5 = ("M21b", "M21bh", "N", "F", "VF")
NET_ARMS = ("M21b", "M21bh", "N")
CELLS = {"base_off": ("p14a_finish", "runs/w4", "_w4"), "base_on": (None, "runs/am", "_w4_am"),
         "pick_off": (None, "runs/det", "_w4_d"), "pick_on": (None, "runs/detam", "_w4_d_am")}
CELL_NAME = {"base_off": "base YOLOE, A-mark off (14a W4 runs)", "base_on": "base YOLOE + A-mark",
             "pick_off": "picked detector, A-mark off", "pick_on": "picked detector + A-mark"}
VARIANTS = ("lp", "ft")
R3_ROWS = {"base": "base", "f1med": "med_d0.5"}
TAU = 0.45


def setup(config: Path) -> Tuple[Dict[str, Any], Path, Path]:
    lay = RunLayout.load(config)
    return lay.cfg, lay.base, lay.outputs


def boot(cfg: Dict[str, Any]) -> Tuple[int, int]:
    b = cfg["p14b1"]["bootstrap"]
    return int(b["draws"]), int(b["seed"])


def fmt(x: Optional[float], p: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"


def sg(x: Optional[float], ci: Optional[Sequence[float]] = None) -> str:
    if x is None:
        return "-"
    return f"{x:+.3f}" + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")


def keep(diff: float, p_new: Optional[float], p_old: Optional[float], gain: float) -> bool:
    """The keep rule: success +gain or more and pooled STOP precision not lower."""
    return diff >= gain - 1e-12 and (p_new or 0.0) >= (p_old or 0.0) - 1e-12


# --- photo ----------------------------------------------------------------------------------------------------------
PHOTO_KEYS = ("ap50", "precision", "recall", "fa_per_free", "H", "R", "F1")


def photo_rows(out: Path, outputs: Path) -> Dict[str, Dict[str, Any]]:
    rows = {"yoloe": json.loads((outputs / "p9_recognisers" / "photo_test" / "yoloe" / "scores.json").read_text())}
    for v in VARIANTS:
        f = out / "photo_test" / f"yoloe_{v}" / "scores.json"
        if f.exists():
            rows[f"yoloe_{v}"] = json.loads(f.read_text())
    return rows


def choose(rows: Dict[str, Dict[str, Any]]) -> str:
    """Highest half-A map F1; ties -> base YOLOE, then LP (the milder change)."""
    order = ["yoloe", "yoloe_lp", "yoloe_ft"]
    return max((k for k in order if k in rows), key=lambda k: (round(rows[k]["f1_half_a"], 12), -order.index(k)))


def cmd_photo(args) -> None:
    cfg, out, outputs = setup(args.config)
    rows = photo_rows(out, outputs)
    pick = choose(rows)
    ch = {"picked": pick, "threshold": rows[pick]["threshold"], "p99": rows[pick]["p99_kept_half_a"],
          "fine_tuned": pick != "yoloe", "f1_half_a": {k: r["f1_half_a"] for k, r in rows.items()},
          "fallback_base": pick == "yoloe"}
    write_text_atomic(out / "photo" / "choice.json", json.dumps(ch, indent=1) + "\n")
    lines = ["# R2 photo test (Phase 9 frame set and rules; pick on half A, report half B)", "",
             f"Pick: **{pick}** (half-A map F1, ties -> base, then LP); threshold {ch['threshold']}, p99 "
             f"{fmt(ch['p99'])}." + (" Fallback: no fine-tuned model beats base on half A -> R3 / R4 use base YOLOE."
                                     if ch["fallback_base"] else ""), "",
             "| model | t (half A) | F1 half A | AP50 B | precision B | recall B | false alarms / target-free item B | "
             "map hit rate H B | map recall R B | map F1 B |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        b = r["half_b"]
        lines.append(f"| {k} | {r['threshold']} | {fmt(r['f1_half_a'])} | " + " | ".join(fmt(b.get(m)) for m in PHOTO_KEYS) + " |")
    lines += ["", "Map F1 per category, half B:", "", "| model | " + " | ".join(r for r in rows["yoloe"]["per_category"]) + " |",
              "|---|" + "---|" * len(rows["yoloe"]["per_category"])]
    for k, r in rows.items():
        lines.append(f"| {k} | " + " | ".join(fmt(r["per_category"][c]["F1"]) for c in rows["yoloe"]["per_category"]) + " |")
    write_text_atomic(out / "tables" / "photo.md", "\n".join(lines) + "\n")
    print(json.dumps(ch), flush=True)


# --- r3 -------------------------------------------------------------------------------------------------------------
def on_target_share(ev: Dict[str, np.ndarray], iou_min: float = 0.3) -> Tuple[int, int]:
    """(detection rows, rows whose box overlaps a true target box of the same frame with IoU >= iou_min)."""
    true_by_t: Dict[int, List[np.ndarray]] = defaultdict(list)
    for t, b in zip(ev["true_t"].tolist(), ev["true_box"]):
        true_by_t[int(t)].append(b)
    on = 0
    for t, b in zip(ev["det_t"].tolist(), ev["det_box"]):
        if any(pb.iou(list(b), list(tb)) >= iou_min for tb in true_by_t.get(int(t), [])):
            on += 1
    return int(len(ev["det_t"])), on


def _share_job(path: str) -> Tuple[int, int]:
    return on_target_share(load_evidence(Path(path)))


def det_share(paths: Sequence[Path], workers: int) -> Dict[str, Any]:
    n = on = 0
    for a, b in pb.imap_safe(_share_job, [str(p) for p in paths], workers):
        n, on = n + a, on + b
    return {"rows": n, "on_target": on, "share": on / n if n else None}


def pick_row(bench: pb.Bench, fam: str, proj: str, score: str = "map") -> pb.Sel:
    """The half-A pick of a family's F3-off keys for one variant (14b-1's tie rule)."""
    msA = bench.mean_success("A")
    v = Variant("det", proj, score).name
    return pb.Sel(*pb.pick(lambda vv, k: float(msA[bench.vindex[vv], pb.KEY_INDEX[k]]),
                           [(v, k) for k in pb.family_keys(fam, False)])[:2])


def compare(new: pb.Bench, sel_new: pb.Sel, old: pb.Bench, sel_old: pb.Sel, home_of: Dict[str, str], draws: int,
            seed: int, gain: float) -> Dict[str, Any]:
    """Half-B effect of (new bench, row) vs (old bench, row): success diff + home CI, pooled precision diff + CI."""
    a, b = pb.summary_of(new, sel_new)["mean"], pb.summary_of(old, sel_old)["mean"]
    d = paired(pb.episode_means_b(new, sel_new), pb.episode_means_b(old, sel_old), home_of, draws, seed)
    pdiff, pci = pb.precision_diff_ci(pb.stop_counts_b(new, sel_new), pb.stop_counts_b(old, sel_old), home_of, draws, seed)
    diff = a["B"]["success"] - b["B"]["success"]
    return {"new": {"key": sel_new.key, **a}, "old": {"key": sel_old.key, **b}, "success_diff_B": diff,
            "success_ci_B": d.get("home_ci"), "precision_diff_B": pdiff, "precision_ci_B": pci,
            "keep": keep(diff, a["B"]["precision_pooled"], b["B"]["precision_pooled"], gain)}


def cmd_r3(args) -> None:
    cfg, out, outputs = setup(args.config)
    runs, _ = pb.layouts(args.config)
    half_of = pb.practice(runs)["episode_half"]
    home_of = {e: x["home"] for e, x in pb.practice_episodes(runs).items()}
    draws, seed = boot(cfg)
    gain = float(cfg["p14b1"]["keep_gain"])
    base = pb.Bench(outputs / "p14b1_marks", "stopbench", ARMS5, half_of)
    res: Dict[str, Any] = {"detectors": {}}
    sub = Path(cfg["r3"]["subdir"]).name
    res["det_share"] = {"yoloe": det_share(sorted((runs.root / "evidence").glob("*/*.npz")), args.workers)}
    for v in VARIANTS:
        key = f"yoloe_{v}"
        if not (out / f"stopbench_{key}" / "runs").exists():
            continue
        new = pb.Bench(out, f"stopbench_{key}", ARMS5, half_of)
        calib = json.loads((out / f"calib_{key}" / "calib.json").read_text())
        r: Dict[str, Any] = {"T": calib["T"], "ece_B": calib["ece"]["B"], "rows": {}}
        for fam in pb.FAMILIES:
            for row, proj in R3_ROWS.items():
                for score in ("map", "cal"):
                    c = compare(new, pick_row(new, fam, proj, score), base, pick_row(base, fam, proj, "map"), home_of,
                                draws, seed, gain)
                    c["report_only"] = score == "cal"
                    r["rows"][f"{fam}|{row}|{score}"] = c
        r["det_share"] = det_share(sorted((out / f"replay_{key}" / sub / "evidence").glob("*/*.npz")), args.workers)
        res["detectors"][key] = r
    write_text_atomic(out / "tables" / "r3.json", json.dumps(res, indent=1) + "\n")
    lines = ["# R3 stop bench with the fine-tuned detectors (14a baseline recordings, stop module only)", "",
             "Each row vs the same rule and marks with base YOLOE (14b-1 bench), keys picked on half A. Half B, mean "
             "over the 5 arms; STOP precision pooled; CIs home-clustered (10,000 draws). Keep = success +0.02 and "
             "precision not lower. `cal` rows (scores calibrated with the refitted T) are report only.", "",
             f"Detections on a target (box IoU >= 0.3 with a true box of the frame): base YOLOE "
             f"{fmt(res['det_share']['yoloe']['share'])} of {res['det_share']['yoloe']['rows']:,} rows" +
             "".join(f"; {k} {fmt(r['det_share']['share'])} of {r['det_share']['rows']:,}" for k, r in res["detectors"].items()) + ".", ""]
    for k, r in res["detectors"].items():
        lines += [f"## {k} (T = {r['T']:.2f}; ECE half B raw {r['ece_B']['raw']:.3f} -> calibrated "
                  f"{r['ece_B']['calibrated']:.3f})", "",
                  "| rule | marks | scores | key new / old | success B new / old | change [CI] | precision B new / old | "
                  "change [CI] | wrong-object stops B new / old | keep |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, c in r["rows"].items():
            fam, row, score = name.split("|")
            n, o = c["new"]["B"], c["old"]["B"]
            lines.append(f"| {pb.FAMILY_NAME[fam]} | {row} | {score} | `{c['new']['key']}` / `{c['old']['key']}` | "
                         f"{fmt(n['success'])} / {fmt(o['success'])} | {sg(c['success_diff_B'], c['success_ci_B'])} | "
                         f"{fmt(n['precision_pooled'])} / {fmt(o['precision_pooled'])} | "
                         f"{sg(c['precision_diff_B'], c['precision_ci_B'])} | {fmt(n['wrong_object'])} / "
                         f"{fmt(o['wrong_object'])} | {'report' if c['report_only'] else ('pass' if c['keep'] else 'fail')} |")
        lines.append("")
    write_text_atomic(out / "tables" / "r3.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:12]), flush=True)


# --- r4 -------------------------------------------------------------------------------------------------------------
def run_row(log_file: Path, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One run's R4 metrics (score_log at tau 0.45; A-mark counts from the result line)."""
    from mapmad_bridge import steplog
    from vint_train.mapmad.closed_loop.p12_score import score_log

    log = steplog.read(log_file)
    s = score_log([dict(log[0], instances=instances)] + log[1:], taus=(TAU,))
    r = s["rule"][f"{TAU:g}"]
    res = log[-1]
    ev = res.get("amark_events") or {}
    return {"episode_id": log[0]["episode_id"], "oracle": float(s["oracle_success"]), "spl": float(s["oracle_spl"]),
            "success": float(r["success"]), "rule_spl": float(r["spl"]), "stopped": float(r["stopped"]),
            "wrong": float(bool(r.get("wrong_object"))), "collisions": float(s["collisions"]),
            "stuck": float(s["stuck_streaks"] > 0), "amark_trigger": float(ev.get("trigger", 0)),
            "amark_confirmed": float(ev.get("confirmed", 0)), "amark_steps": float(res.get("amark_steps", 0) or 0),
            "amark_no_path": float(ev.get("no_path", 0) + ev.get("no_free", 0))}


def _row_job(job) -> Tuple[str, str, Dict[str, Any]]:
    cell, arm, log_file, insts = job
    return cell, arm, run_row(Path(log_file), insts)


def r4_data(cfg: Dict[str, Any], out: Path, outputs: Path, eps: Dict[str, Dict[str, Any]], workers: int,
            only: Optional[set] = None) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    jobs = []
    for cell, (src, sub, suf) in CELLS.items():
        root = (outputs / src if src else out) / sub
        for a in NET_ARMS:
            for f in sorted((root / "logs" / f"{a}{suf}").glob("*.jsonl")):
                e = parse_run_id(f.name[:-len(".jsonl")])[0]
                if only and e not in only:
                    continue
                jobs.append((cell, a, str(f), eps[e]["target"]["instances"]))
    data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {c: {a: {} for a in NET_ARMS} for c in CELLS}
    for cell, a, r in pb.imap_safe(_row_job, jobs, workers):
        data[cell][a][r["episode_id"]] = r
    return data


METRICS = ("oracle", "success", "spl", "rule_spl", "stopped", "wrong", "collisions", "stuck", "amark_trigger",
           "amark_confirmed", "amark_steps", "amark_no_path")


def cell_effect(data, cell: str, ref: str, half_of: Dict[str, str], home_of: Dict[str, str], draws: int, seed: int,
                arms: Sequence[str] = NET_ARMS, half: str = "B") -> Dict[str, Any]:
    """Mean over `arms` per episode (paired) of cell - ref on a half: success, oracle; pooled STOP precision change."""
    def ep_means(c: str, m: str) -> Dict[str, float]:
        acc: Dict[str, List[float]] = defaultdict(list)
        for a in arms:
            for e, r in data[c][a].items():
                if half == "all" or half_of[e] == half:
                    acc[e].append(r[m])
        return {e: float(np.mean(v)) for e, v in acc.items() if len(v) == len(arms)}

    def counts(c: str) -> Dict[str, List[float]]:
        acc: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
        for a in arms:
            for e, r in data[c][a].items():
                if half == "all" or half_of[e] == half:
                    acc[e][0] += r["success"]
                    acc[e][1] += r["stopped"]
        return dict(acc)

    out: Dict[str, Any] = {}
    for m in ("success", "oracle", "wrong", "collisions", "stuck"):
        a, b = ep_means(cell, m), ep_means(ref, m)
        d = paired(a, b, home_of, draws, seed)
        out[m] = {"new": float(np.mean(list(a.values()))) if a else None,
                  "old": float(np.mean(list(b.values()))) if b else None, "diff": d.get("diff"), "ci": d.get("home_ci"),
                  "n": d.get("n")}
    ca, cb = counts(cell), counts(ref)
    pd, pci = pb.precision_diff_ci(ca, cb, home_of, draws, seed)
    prec = lambda c: (sum(v[0] for v in c.values()) / sum(v[1] for v in c.values())) if sum(v[1] for v in c.values()) else None  # noqa: E731
    out["precision"] = {"new": prec(ca), "old": prec(cb), "diff": pd, "ci": pci}
    return out


def cmd_r4(args) -> None:
    cfg, out, outputs = setup(args.config)
    runs, _ = pb.layouts(args.config)
    eps = pb.practice_episodes(runs)
    half_of = pb.practice(runs)["episode_half"]
    home_of = {e: x["home"] for e, x in eps.items()}
    draws, seed = boot(cfg)
    gain = float(cfg["p14b1"]["keep_gain"])
    data = r4_data(cfg, out, outputs, eps, args.workers, pb.episodes_filter() or None)
    res: Dict[str, Any] = {"cells": {}, "effects": {}, "map_effect": {}}
    lines = ["# R4 driving test (practice set, seed 0, W4 on everywhere; rule stop = today's rule tau 0.45)", "",
             "| cell | arm | runs | oracle | rule-stop success | SPL (rule) | stops | wrong-object stops | collision steps | "
             "stuck runs | A-mark triggers | A-mark confirmed | A-mark no path | A-mark steps |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cell in CELLS:
        res["cells"][cell] = {}
        for a in NET_ARMS:
            d = data[cell][a]
            if not d:
                continue
            m = {k: float(np.mean([r[k] for r in d.values()])) for k in METRICS}
            stops = sum(r["stopped"] for r in d.values())
            m["wrong_share"] = (sum(r["wrong"] for r in d.values()) / stops) if stops else None
            res["cells"][cell][a] = {"runs": len(d), **m}
            lines.append(f"| {CELL_NAME[cell]} | {a} | {len(d)} | {fmt(m['oracle'])} | {fmt(m['success'])} | "
                         f"{fmt(m['rule_spl'])} | {int(stops)} | {fmt(m['wrong_share'])} | {m['collisions']:.1f} | "
                         f"{fmt(m['stuck'])} | {m['amark_trigger']:.2f} | {m['amark_confirmed']:.2f} | "
                         f"{m['amark_no_path']:.2f} | {m['amark_steps']:.1f} |")
    lines += ["", "## Change vs base YOLOE, A-mark off (half B, mean over M21b, M21bh, N; paired; home CI)", "",
              "| cell | correct stops new / old | change [CI] | STOP precision new / old | change [CI] | oracle change "
              "[CI] | collision steps change | keep |", "|---|---|---|---|---|---|---|---|"]
    for cell in ("base_on", "pick_off", "pick_on"):
        if not all(data[cell][a] for a in NET_ARMS):
            continue
        e = cell_effect(data, cell, "base_off", half_of, home_of, draws, seed)
        e["keep"] = keep(e["success"]["diff"] or 0.0, e["precision"]["new"], e["precision"]["old"], gain)
        res["effects"][cell] = e
        s, p, o = e["success"], e["precision"], e["oracle"]
        lines.append(f"| {CELL_NAME[cell]} | {fmt(s['new'])} / {fmt(s['old'])} | {sg(s['diff'], s['ci'])} | "
                     f"{fmt(p['new'])} / {fmt(p['old'])} | {sg(p['diff'], p['ci'])} | {sg(o['diff'], o['ci'])} | "
                     f"{sg(e['collisions']['diff'])} | {'pass' if e['keep'] else 'fail'} |")
    lines += ["", "## Map effect M21b - M21bh per cell (paired by episode, home CI)", "",
              "| cell | half | rule-stop success [CI] | oracle [CI] | n |", "|---|---|---|---|---|"]
    for cell in CELLS:
        if not (data[cell]["M21b"] and data[cell]["M21bh"]):
            continue
        res["map_effect"][cell] = {}
        for half in ("all", "B"):
            sel = lambda a, m: {e: r[m] for e, r in data[cell][a].items() if half == "all" or half_of[e] == half}  # noqa: E731
            ps = paired(sel("M21b", "success"), sel("M21bh", "success"), home_of, draws, seed)
            po = paired(sel("M21b", "oracle"), sel("M21bh", "oracle"), home_of, draws, seed)
            res["map_effect"][cell][half] = {"success": ps, "oracle": po}
            lines.append(f"| {CELL_NAME[cell]} | {half} | {sg(ps.get('diff'), ps.get('home_ci'))} | "
                         f"{sg(po.get('diff'), po.get('home_ci'))} | {ps.get('n')} |")
    write_text_atomic(out / "tables" / "r4.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(out / "tables" / "r4.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# --- plain ----------------------------------------------------------------------------------------------------------
def cmd_plain(args) -> None:
    cfg, out, _ = setup(args.config)
    r3 = json.loads((out / "tables" / "r3.json").read_text()) if (out / "tables" / "r3.json").exists() else {"detectors": {}}
    r4 = json.loads((out / "tables" / "r4.json").read_text()) if (out / "tables" / "r4.json").exists() else {"effects": {}}
    ch = json.loads((out / "photo" / "choice.json").read_text())
    lines = ["# Effect of each change on half B (correct stops = rule-stop success; STOP precision pooled; home CI)", "",
             "Detector rows: R3 stop bench on the 14a recordings (today's rule, 14a marks, key picked on half A; mean "
             "over 5 arms) vs base YOLOE. A-mark / driving rows: R4 driving test (today's rule tau 0.45, W4 on; mean "
             "over M21b, M21bh, N) vs base YOLOE with A-mark off (14a W4 runs). Keep = +0.02 and precision not lower.",
             f"Picked detector (photo test, half A): **{ch['picked']}**.", "",
             "| change | measured on | correct stops new / old | change [CI] | STOP precision new / old | change [CI] | keep |",
             "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        c = r3["detectors"].get(f"yoloe_{v}", {}).get("rows", {}).get("today|base|map")
        if c:
            n, o = c["new"]["B"], c["old"]["B"]
            lines.append(f"| detector {v.upper()} | R3 replay | {fmt(n['success'])} / {fmt(o['success'])} | "
                         f"{sg(c['success_diff_B'], c['success_ci_B'])} | {fmt(n['precision_pooled'])} / "
                         f"{fmt(o['precision_pooled'])} | {sg(c['precision_diff_B'], c['precision_ci_B'])} | "
                         f"{'pass' if c['keep'] else 'fail'} |")
    for cell, name in (("base_on", "A-mark"), ("pick_off", f"detector {ch['picked']} (driving)"),
                       ("pick_on", f"both ({ch['picked']} + A-mark)")):
        e = r4["effects"].get(cell)
        if e:
            s, p = e["success"], e["precision"]
            lines.append(f"| {name} | R4 driving | {fmt(s['new'])} / {fmt(s['old'])} | {sg(s['diff'], s['ci'])} | "
                         f"{fmt(p['new'])} / {fmt(p['old'])} | {sg(p['diff'], p['ci'])} | {'pass' if e['keep'] else 'fail'} |")
    write_text_atomic(out / "tables" / "plain.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# --- checks ---------------------------------------------------------------------------------------------------------
def same_gpu_pairs(out: Path, outputs: Path) -> Dict[str, Any]:
    """checks/off_g<G>/<kind>/<arm>/<run> vs p14a_finish/runs/w4/<kind>/<arm>/<run> for runs 14a drove on GPU G."""
    import re

    first: Dict[Tuple[str, str], set] = {}
    root14 = outputs / "p14a_finish" / "runs" / "w4"
    for f in root14.glob("progress.*.log"):
        gpu = int(re.search(r"shard(\d+)\.log$", f.name).group(1)) // 10
        for line in f.read_text().splitlines():
            m = re.match(r"\S+ \d+/\d+ (\S+) (\S+__s\d+): (.*)", line)
            if m and "steps" in m.group(3):
                first.setdefault((m.group(1), m.group(2)), set()).add(gpu)
    pairs, bad = 0, []
    for d in sorted(out.glob("checks/off_g*")):
        g = int(d.name.split("_g")[1])
        for log in sorted(d.glob("logs/*/*.jsonl")):
            arm, rid = log.parent.name, log.name[:-len(".jsonl")]
            if first.get((arm, rid)) != {g}:
                continue
            pairs += 1
            for kind, ext in (("logs", "jsonl"), ("dets", "jsonl")):
                fa, fb = d / kind / arm / f"{rid}.{ext}", root14 / kind / arm / f"{rid}.{ext}"
                if not (fa.exists() and fb.exists() and filecmp.cmp(fa, fb, shallow=False)):
                    bad.append(f"g{g} {arm} {rid} {kind}")
            ea, eb = (load_evidence(r / "evidence" / arm / f"{rid}.npz") for r in (d, root14))
            if sorted(ea) != sorted(eb) or any(not np.array_equal(ea[k], eb[k]) for k in ea):
                bad.append(f"g{g} {arm} {rid} evidence")
    return {"same_gpu_pairs": pairs, "differ": bad, "pass": pairs >= 2 and not bad}


def cmd_offcheck(args) -> None:
    cfg, out, outputs = setup(args.config)
    res = same_gpu_pairs(out, outputs)
    write_text_atomic(out / "checks" / "off.json", json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)
    if not res["pass"]:
        raise SystemExit(3)


def replay_check(out: Path, sub: str) -> Dict[str, Any]:
    """R3 replay check over every replay_yoloe_<v> run record: same poses (diff 0), true rows re-projected equal."""
    res = {}
    for v in VARIANTS:
        recs = [json.loads(f.read_text()) for f in sorted((out / f"replay_yoloe_{v}" / sub).glob("*/*.json"))]
        if not recs:
            continue
        res[f"yoloe_{v}"] = {"runs": len(recs), "pose_max_diff": max(r["pose_max_diff"] for r in recs),
                             "true_rows": sum(r["true_rows"] for r in recs),
                             "true_base_same": sum(r["true_base_same"] for r in recs),
                             "true_frames_count_mismatch": sum(r["true_frames_count_mismatch"] for r in recs),
                             "det_rows": sum(r["det_rows"] for r in recs), "det_rows_14a": sum(r["det_rows_14a"] for r in recs)}
        x = res[f"yoloe_{v}"]
        x["pass"] = x["pose_max_diff"] == 0.0 and x["true_base_same"] == x["true_rows"] and not x["true_frames_count_mismatch"]
    return res


def cmd_gate(args) -> None:
    """G14b2_table_draft.md (rows 1-5 reported, 6-8 must pass) from the outputs; -> checks/gate.json."""
    cfg, out, outputs = setup(args.config)
    rd = lambda f: json.loads((out / f).read_text()) if (out / f).exists() else None  # noqa: E731
    homes, checks, off = rd("trainset/homes.json"), rd("checks/checks.json") or {}, rd("checks/off.json")
    hb, ha = rd("checks/hashes_before.json"), rd("checks/hashes_after.json")
    rc = replay_check(out, Path(cfg["r3"]["subdir"]).name)
    ch, vid = rd("photo/choice.json"), rd("videos_detam/video_check.json")
    row6 = bool(homes and homes["contamination_free"])
    hashes_ok = bool(hb and ha and hb == ha)
    row7 = bool(checks.get("fairness", {}).get("pass")) and bool(off and off["pass"]) and hashes_ok
    row8 = bool(checks.get("determinism", {}).get("pass")) and bool(rc) and all(x["pass"] for x in rc.values())
    rows = [
        ("1", "Photo test half B: base vs LP vs FT", "reported", f"tables/photo.md; pick {ch['picked'] if ch else '-'}"),
        ("2", "R3 stop bench with the fine-tuned detectors, keep rule", "reported", "tables/r3.md"),
        ("3", "R4 driving cells, map effect per cell", "reported", "tables/r4.md"),
        ("4", "Detection share on targets, wrong-object stops, calibration", "reported", "tables/r3.md (share, T, ECE)"),
        ("5", "8 videos (box + pose check)", "reported", f"videos_detam/: {vid['summary'] if vid else 'missing'}"),
        ("6", "No contamination: training homes ∩ (photo test ∪ HM3D val ∪ MP3D) = ∅", "must pass",
         ("PASS" if row6 else "FAIL") + (f" (pool {len(homes['pool'])} homes, sha256 {homes['pool_sha256'][:12]}…; "
                                         f"overlap {homes['overlap']})" if homes else " (homes.json missing)")),
        ("7", "Fairness: every run on the frozen practice fingerprint; checkpoints sha256; switch off = 14a", "must pass",
         ("PASS" if row7 else "FAIL") + f" ({checks.get('fairness_summary', 'fairness missing')}; switch off: "
         f"{off if off else 'missing'}; 14a files unchanged: {hashes_ok})"),
        ("8", "Determinism: one episode twice on the same GPU; replay check", "must pass",
         ("PASS" if row8 else "FAIL") + f" ({checks.get('determinism_summary', 'determinism missing')}; replay: {rc})"),
    ]
    res = {"rows": {r[0]: {"check": r[1], "pass_rule": r[2], "result": r[3]} for r in rows},
           "must_pass": {"6": row6, "7": row7, "8": row8}, "all_must_pass": row6 and row7 and row8}
    write_text_atomic(out / "checks" / "gate.json", json.dumps(res, indent=1) + "\n")
    lines = ["# G14b2 (draft, written by the chain; finished in Part 3)", "", "| # | Check | Pass | Result |",
             "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    write_text_atomic(out / "G14b2_table_draft.md", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["photo", "r3", "r4", "plain", "off", "gate"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    {"photo": cmd_photo, "r3": cmd_r3, "r4": cmd_r4, "plain": cmd_plain, "off": cmd_offcheck, "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
