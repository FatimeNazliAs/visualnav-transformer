"""Phase 14b-1 bench (kickoff §3-§6, p14b1_rules.md): the stop module's marks rebuilt with F1 / F2 / F3 (+ c_cat) and
every variant scored on the Phase 14a practice runs (read only). numpy only (naz_mapmad; PYTHONPATH has mapmad/src).

    python -m vint_train.mapmad.closed_loop.p14b_bench calib  --config <p14b1_marks.yaml>             # 14b1.2: T, ECE
    python -m vint_train.mapmad.closed_loop.p14b_bench score  --config ... --subdir runs/baseline --workers 20
    python -m vint_train.mapmad.closed_loop.p14b_bench pick   --config ...                             # rows 1-7
    python -m vint_train.mapmad.closed_loop.p14b_bench score  --config ... --subdir runs/baseline --ccat   # row 8 runs
    python -m vint_train.mapmad.closed_loop.p14b_bench final  --config ...      # row 8, keep rule, recommended, tables
    python -m vint_train.mapmad.closed_loop.p14b_bench marks  --config ... --workers 20                # mark statistics
    python -m vint_train.mapmad.closed_loop.p14b_bench drive  --config ... --workers 20                # step 7 re-score
    python -m vint_train.mapmad.closed_loop.p14b_bench hashes --config ... --tag before|after          # gate row 6
    python -m vint_train.mapmad.closed_loop.p14b_bench gate   --config ...                             # G14b1 draft

Per run (score): every variant (stop_variants.Variant; det x 13 projections x {map, cal}; true x 13 projections) x
every rule key (stop_variants.all_keys(): F3 off and on) -> stop step, success, SPL, wrong object, near miss, stop cell
in view at the stop frame (p14_stopbench.judge, the 14a definitions) in <out>/runs/<arm>/<run>.npz; parity = the
14a bench's per-run stops of every 14a grid point equal det|base|map with F3 off.
Rows per rule family (today / Rule 1 / Rule 2), each picked on half A by the 14a selection rule (mean rule-stop success
over the 5 arms; ties -> the higher threshold, then fewer looks; between projection settings the milder one):
base · base (14a pick) · +F3 · +F2 · +F1-med · +F1-clu · +F1(best)+F2 · +F1(best)+F2+F3 · +F1(best)+F2+F3+c_cat.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_bridge import steplog
from mapmad_sim import stop_evidence
from mapmad_sim.mark_projection import ProjSpec
from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic
from mapmad_sim.stop_rules import parse_r1, parse_r2
from mapmad_sim.stop_variants import (F3_TAG, RunGeometry, Variant, all_keys, calibrated, cell_xy, family, first_stops,
                                      variant_evidence)
from mapmad_sim.video_events import iou
from vint_train.mapmad.closed_loop.p12_score import box_distance_2d, score_log
from vint_train.mapmad.closed_loop.p14_stopbench import judge, practice, practice_episodes

ARMS5 = ("M21b", "M21bh", "N", "F", "VF")
HALVES = ("A", "B", "all")
FAMILIES = ("today", "r1", "r2")
FAMILY_NAME = {"today": "today's rule", "r1": "Rule 1 (voting)", "r2": "Rule 2 (weighted voting)"}
PICK14A = {"today": "t_tau0.45", "r1": "r1_s0.10_n2_h0.5", "r2": "r2_v0.50_g1"}
ROWS = ("base", "base14a", "f3", "f2", "f1med", "f1clu", "f1f2", "f1f2f3", "f1f2f3c")
ROW_NAME = {"base": "base (14a marks, re-picked)", "base14a": "base (14a pick)", "f3": "+F3", "f2": "+F2",
            "f1med": "+F1-med", "f1clu": "+F1-clu", "f1f2": "+F1(best)+F2", "f1f2f3": "+F1(best)+F2+F3",
            "f1f2f3c": "+F1(best)+F2+F3+c_cat"}
# keep-rule comparisons: (row, the row without the fix, fix)
COMPARE = (("f3", "base", "F3"), ("f2", "base", "F2"), ("f1med", "base", "F1-med"), ("f1clu", "base", "F1-clu"),
           ("f1f2", "f2", "F1 on top of F2"), ("f1f2", "f1best", "F2 on top of F1"), ("f1f2f3", "f1f2", "F3 on top"),
           ("f1f2f3c", "f1f2f3", "c_cat on top"))
CHAIN = {"f3": ["f3"], "f2": ["f2"], "f1med": ["f1med"], "f1clu": ["f1clu"], "f1f2": ["f1f2", "f1f2b"],
         "f1f2f3": ["f1f2", "f1f2b", "f1f2f3"], "f1f2f3c": ["f1f2", "f1f2b", "f1f2f3", "f1f2f3c"]}
# CHAIN step -> its keep-rule comparison (keep_rule keys)
STEP_KEY = {"f3": "f3<base", "f2": "f2<base", "f1med": "f1med<base", "f1clu": "f1clu<base", "f1f2": "f1f2<f2",
            "f1f2b": "f1f2<f1best", "f1f2f3": "f1f2f3<f1f2", "f1f2f3c": "f1f2f3c<f1f2f3"}
# Part 3 fix-effects table: (fix, row (None = the best combination), the row it is compared with)
EFFECTS = (("F1-med", "f1med", "base"), ("F1-clu", "f1clu", "base"), ("F2", "f2", "base"), ("F3", "f3", "base"),
           ("c_cat", "f1f2f3c", "f1f2f3"), ("best combination", None, "base"))
COMBOS = ("f1f2", "f1f2f3", "f1f2f3c")
KEYS = all_keys()
KEY_INDEX = {k: i for i, k in enumerate(KEYS)}


# --- config -----------------------------------------------------------------------------------------------------------
def layouts(config: Path, subdir: str = "runs/baseline") -> Tuple[RunLayout, Path]:
    """(the 14a runs' layout, this phase's output folder)."""
    return RunLayout.load_runs(config, subdir=subdir), RunLayout.load(config).base


def proj_spec(cfg: Dict[str, Any]) -> ProjSpec:
    return ProjSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in (cfg.get("projection") or {}).items()})


def base_variants(spec: ProjSpec) -> List[Variant]:
    """Every variant of the main score stage (module docstring)."""
    return ([Variant("det", p, sc) for p in spec.names() for sc in ("map", "cal")]
            + [Variant("true", p, "map") for p in spec.names()])


def temperature(out: Path) -> float:
    return float(json.loads((out / "calib" / "calib.json").read_text())["T"])


# --- 14b1.2 calibration -----------------------------------------------------------------------------------------------
def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on [0, 1]: sum_b |b| / N |accuracy_b - confidence_b|."""
    p, y = np.asarray(p, np.float64), np.asarray(y, np.float64)
    if p.size == 0:
        return float("nan")
    b = np.minimum((p * bins).astype(int), bins - 1)
    out = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            out += m.sum() / p.size * abs(y[m].mean() - p[m].mean())
    return float(out)


def map_value(s: np.ndarray, t: float = 0.01, p99: float = 0.788) -> np.ndarray:
    """Today's map value of a kept YOLOE score (recogniser.map_score)."""
    return 0.4 + 0.6 * np.clip((np.asarray(s) - t) / (p99 - t), 0.0, 1.0)


def calib_rows(labels: List[Dict[str, Any]], halves: Dict[str, List[str]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """half -> (scores, TP flags) of the six-category YOLOE detections (p9q items; IGN rows left out)."""
    half_of = {h.split("-")[0]: name for name in ("A", "B") for h in halves[name]}
    out: Dict[str, List[Tuple[float, bool]]] = {"A": [], "B": []}
    for r in labels:
        if not r["item"].startswith("p9q-") or r["pix"] not in ("TP", "FP"):
            continue
        out[half_of[r["item"].split("-")[2]]].append((float(r["score"]), r["pix"] == "TP"))
    return {h: (np.array([s for s, _ in v]), np.array([y for _, y in v])) for h, v in out.items()}


def fit_temperature(s: np.ndarray, y: np.ndarray, grid: Sequence[float], bins: int) -> Tuple[float, Dict[float, float]]:
    """The grid T with the lowest ECE of sigmoid(logit(s) / T) (ties -> the T nearest 1)."""
    scores = {t: ece(calibrated(s, t), y, bins) for t in grid}
    best = min(grid, key=lambda t: (round(scores[t], 12), abs(t - 1.0)))
    return float(best), scores


def reliability(p: np.ndarray, y: np.ndarray, bins: int) -> List[Tuple[int, float, float]]:
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [(int((b == k).sum()), float(p[b == k].mean()) if (b == k).any() else float("nan"),
             float(y[b == k].mean()) if (b == k).any() else float("nan")) for k in range(bins)]


def cmd_calib(args) -> None:
    runs, out = layouts(args.config)
    c = runs.cfg["p14b1"]["calib"]
    labels = json.loads((runs.outputs / c["labels"]).read_text())
    halves = json.loads((runs.outputs / c["halves"]).read_text())
    lo, hi, step = c["t_grid"]
    grid = [round(lo + step * k, 2) for k in range(int(round((hi - lo) / step)) + 1)]
    bins = int(c["bins"])
    data = calib_rows(labels, halves)
    t, scores = fit_temperature(*data["A"], grid, bins)
    res = {"T": t, "grid": grid, "bins": bins, "fit_half": "A", "items": {h: int(len(v[0])) for h, v in data.items()},
           "tp_share": {h: float(v[1].mean()) for h, v in data.items()},
           "ece": {h: {"raw": ece(v[0], v[1], bins), "map_value": ece(map_value(v[0]), v[1], bins),
                       "calibrated": ece(calibrated(v[0], t), v[1], bins)} for h, v in data.items()},
           "ece_by_T_half_A": scores,
           "note": "logits recovered from the scores (logit = log(s / (1 - s))): the photo test kept no raw logits "
                   "(kickoff §3 fallback)"}
    (out / "calib").mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "calib" / "calib.json", json.dumps(res, indent=1) + "\n")
    sb, yb = data["B"]
    lines = ["# 14b1.2 temperature fit (F2; report)", "",
             f"YOLOE-11-L-seg on the Phase 9 photo test, six categories (p9q items; TP / FP at box IoU >= 0.5, IGN rows "
             f"left out). Fit on half A ({res['items']['A']} detections), checked on half B ({res['items']['B']}). "
             f"T grid {lo}-{hi} step {step}; ECE over {bins} equal bins. Logits recovered from scores (fallback).", "",
             f"**T = {t:.2f}**", "", "| half | detections | TP share | ECE raw | ECE today's map value | ECE calibrated |",
             "|---|---|---|---|---|---|"]
    for h in ("A", "B"):
        e = res["ece"][h]
        lines.append(f"| {h} | {res['items'][h]} | {res['tp_share'][h]:.3f} | {e['raw']:.4f} | {e['map_value']:.4f} | "
                     f"{e['calibrated']:.4f} |")
    lines += ["", "Reliability, half B (count / mean confidence / TP share per bin):", "",
              "| bin | raw n | raw conf | raw acc | cal n | cal conf | cal acc |", "|---|---|---|---|---|---|---|"]
    for k, (a, b) in enumerate(zip(reliability(sb, yb, bins), reliability(calibrated(sb, t), yb, bins))):
        lines.append(f"| {k / bins:.1f}-{(k + 1) / bins:.1f} | {a[0]} | {a[1]:.3f} | {a[2]:.3f} | {b[0]} | {b[1]:.3f} | "
                     f"{b[2]:.3f} |")
    write_text_atomic(out / "calib" / "calib.md", "\n".join(lines) + "\n")
    print("\n".join(lines[:12]), flush=True)


# --- one run: every variant x every key -------------------------------------------------------------------------------
def run_inputs(log_file: Path, episode: Dict[str, Any]) -> Dict[str, Any]:
    """What judge() needs from a run's log (as p14_stopbench.score_run) + the oracle."""
    log = steplog.read(log_file)
    head, steps = log[0], [x for x in log if x["kind"] == "step"]
    insts = episode["target"]["instances"]
    base = score_log([dict(head, instances=insts)] + log[1:], taus=[0.45])
    return {"head": head, "insts": insts, "vp": [head["start_vp_geodesic_m"]] + [s["vp_geodesic_m"] for s in steps],
            "path": np.concatenate([[0.0], np.cumsum([s["travelled_m"] for s in steps])]),
            "oracle_success": bool(base["oracle_success"]), "oracle_spl": float(base["oracle_spl"])}


def judge_all(ev: Dict[str, np.ndarray], stops: Dict[str, Optional[Tuple[int, int]]], ri: Dict[str, Any],
              views: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """One variant's stops judged on the run: arrays over KEYS."""
    n = len(KEYS)
    a = {"step": np.full(n, -1, np.int16), "success": np.zeros(n, bool), "spl": np.zeros(n, np.float32),
         "wrong": np.zeros(n, bool), "near": np.zeros(n, bool), "in_view": np.zeros(n, bool),
         "cell": np.full((n, 2), np.nan, np.float32)}
    cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for k, st in stops.items():
        if st is None:
            continue
        i = KEY_INDEX[k]
        if st not in cache:
            xy = cell_xy(ev, st[1])
            j = judge((st[0], xy), ri["vp"], ri["path"], ri["head"]["start_vp_geodesic_m"], ri["insts"])
            j["in_view"] = bool(np.isin(st[1], views[st[0]]))
            j["xy"] = xy
            cache[st] = j
        j = cache[st]
        a["step"][i], a["success"][i], a["spl"][i] = st[0], j["success"], j["spl"]
        a["wrong"][i], a["near"][i], a["in_view"][i], a["cell"][i] = j["wrong_object"], j["near_miss"], j["in_view"], j["xy"]
    return a


def score_run(log_file: Path, ev_file: Path, proj_file: Path, episode: Dict[str, Any], variants: Sequence[Variant],
              t_cal: float, bench14a: Optional[Path]) -> Dict[str, np.ndarray]:
    """Every variant x key of one run (module docstring) as stacked arrays."""
    ri = run_inputs(log_file, episode)
    ev = stop_evidence.load(ev_file)
    proj = stop_evidence.load(proj_file)
    geo = RunGeometry(ev)
    views = [ev["view"][ev["view_ptr"][t]:ev["view_ptr"][t + 1]] for t in range(len(ev["pose"]))]
    per = [judge_all(ev, first_stops(variant_evidence(ev, proj, v, t_cal), geo), ri, views) for v in variants]
    out = {f: np.stack([p[f] for p in per]) for f in per[0]}
    checked = same = 0
    if bench14a is not None and bench14a.exists():
        old = json.loads(bench14a.read_text())["rules"]
        vi = [v.name for v in variants].index(Variant().name)
        for k_old, r in old.items():
            k_new = "t_tau0.45" if k_old == "today" else k_old
            i = KEY_INDEX[k_new]
            checked += 1
            same += int((r["step"] if r["step"] is not None else -1) == int(out["step"][vi, i])
                        and bool(r["success"]) == bool(out["success"][vi, i]))
    h = ri["head"]
    out.update({"variants": np.array([v.name for v in variants]), "oracle_success": np.array(ri["oracle_success"]),
                "oracle_spl": np.array(ri["oracle_spl"]), "parity14a": np.array([checked, same]),
                "meta": np.array([h["arm"], h["episode_id"], h["home"], h["category"]])})
    return out


def _score_one(job) -> Tuple[str, Optional[str]]:
    log_file, ev_file, proj_file, out_file, episode, variants, t_cal, b14 = job
    try:
        arr = score_run(Path(log_file), Path(ev_file), Path(proj_file), episode, [Variant.parse(v) for v in variants],
                        t_cal, Path(b14) if b14 else None)
        save_npz(Path(out_file), arr)
        return out_file, None
    except Exception as exc:  # reported per run; the caller stops on any
        import traceback
        return out_file, f"{type(exc).__name__}: {exc} {traceback.format_exc()[-400:]}"


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    import io
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def episodes_filter() -> set:
    import os

    return set(os.environ.get("P14_EPISODES", "").split())


def score_jobs(runs: RunLayout, out: Path, arms: Sequence[str], variants: Sequence[Variant], dest: str, t_cal: float,
               force: bool, with_parity: bool) -> List[Tuple]:
    eps = practice_episodes(runs)
    only = episodes_filter()
    sub = Path(str(runs.root)).name
    jobs = []
    for arm in arms:
        for f in sorted((runs.root / "logs" / arm).glob("*.jsonl")):
            rid = f.name[:-len(".jsonl")]
            if only and parse_run_id(rid)[0] not in only:
                continue
            o = out / dest / "runs" / arm / f"{rid}.npz"
            if o.exists() and not force:
                continue
            proj = out / "replay" / sub / arm / f"{rid}.npz"
            if not proj.exists():
                raise SystemExit(f"missing replay file {proj}")
            b14 = runs.base / "stopbench" / "runs" / arm / f"{rid}.json" if with_parity else None
            jobs.append((str(f), str(runs.root / "evidence" / arm / f"{rid}.npz"), str(proj), str(o),
                         eps[parse_run_id(rid)[0]], [v.name for v in variants], t_cal, str(b14) if b14 else ""))
    return jobs


def imap_safe(fn: Callable[[Any], Any], jobs: Sequence[Any], workers: int) -> Iterator[Any]:
    """Run fn over jobs on `workers` processes and yield the results as they finish (any order).

    Unlike multiprocessing.Pool, a worker that dies (killed, segfault) raises BrokenProcessPool here instead of
    leaving the caller waiting forever for its job (Phase 14b-1 chain: one dead worker hung the bench 4 h).
    """
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fn, j) for j in jobs]):
            yield fut.result()


def run_pool(jobs: List[Tuple], workers: int) -> None:
    print(f"scoring {len(jobs)} runs on {workers} workers", flush=True)
    errors = []
    for k, (o, err) in enumerate(imap_safe(_score_one, jobs, workers)):
        if err:
            errors.append((o, err))
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(jobs)}", flush=True)
    if errors:
        raise SystemExit(f"{len(errors)} runs failed: {errors[:2]}")


def cmd_score(args) -> None:
    runs, out = layouts(args.config, args.subdir)
    t_cal = temperature(out)
    if args.ccat:
        pick = json.loads((out / "stopbench" / "PICK1.json").read_text())
        grid = runs.cfg["p14b1"]["ccat_grid"]
        projs = sorted({pick["f1best"][f] for f in FAMILIES})
        variants = [Variant("det", p, "cal", float(c)) for p in projs for c in grid]
        jobs = score_jobs(runs, out, args.arms, variants, "stopbench_ccat", t_cal, args.force, False)
    else:
        jobs = score_jobs(runs, out, args.arms, base_variants(proj_spec(runs.cfg)), "stopbench", t_cal, args.force, True)
    run_pool(jobs, args.workers)


# --- aggregation ------------------------------------------------------------------------------------------------------
class Bench:
    """Every scored run of the arms below <out>/<dest>/runs: arrays (runs, variants, keys) per arm."""

    FIELDS = ("step", "success", "spl", "wrong", "near", "in_view")

    def __init__(self, out: Path, dest: str, arms: Sequence[str], half_of: Dict[str, str]) -> None:
        self.arms, self.half_of = list(arms), half_of
        self.data: Dict[str, Dict[str, np.ndarray]] = {}
        self.variants: List[str] = []
        for a in arms:
            files = sorted((out / dest / "runs" / a).glob("*.npz"))
            if not files:
                raise SystemExit(f"no scored runs in {out / dest / 'runs' / a}")
            rows = [stop_evidence.load(f) for f in files]
            names = [str(x) for x in rows[0]["variants"]]
            if self.variants and names != self.variants:
                raise SystemExit("variant lists differ between arms")
            self.variants = names
            d = {f: np.stack([r[f] for r in rows]) for f in self.FIELDS}
            d["oracle"] = np.array([bool(r["oracle_success"]) for r in rows])
            d["episode"] = np.array([str(r["meta"][1]) for r in rows])
            d["home"] = np.array([str(r["meta"][2]) for r in rows])
            d["category"] = np.array([str(r["meta"][3]) for r in rows])
            d["half"] = np.array([half_of[e] for e in d["episode"]])
            d["parity"] = np.stack([r["parity14a"] for r in rows]) if "parity14a" in rows[0] else np.zeros((len(rows), 2))
            self.data[a] = d
        self.vindex = {v: i for i, v in enumerate(self.variants)}

    def mask(self, arm: str, half: str) -> np.ndarray:
        d = self.data[arm]
        return np.ones(len(d["episode"]), bool) if half == "all" else d["half"] == half

    def success(self, arm: str, half: str) -> np.ndarray:
        """(variants, keys) mean rule-stop success of the arm on the half."""
        return self.data[arm]["success"][self.mask(arm, half)].mean(axis=0)

    def mean_success(self, half: str) -> np.ndarray:
        return np.mean([self.success(a, half) for a in self.arms], axis=0)

    def select(self, arm: str, sel: "Sel", field: str, half: str = "all") -> np.ndarray:
        """Per run of the arm (in the half) the field of the selection (variant per run, key)."""
        d = self.data[arm]
        m = self.mask(arm, half)
        vi = np.array([self.vindex[sel.variant_of(c)] for c in d["category"][m]])
        return d[field][m][np.arange(m.sum()), vi, KEY_INDEX[sel.key]]


class Sel:
    """A picked (variant, key); c_cat rows pick the variant per category."""

    def __init__(self, variant: str, key: str, per_cat: Optional[Dict[str, str]] = None) -> None:
        self.variant, self.key, self.per_cat = variant, key, per_cat

    def variant_of(self, category: str) -> str:
        return self.per_cat[category] if self.per_cat else self.variant

    def to_json(self) -> Dict[str, Any]:
        return {"variant": self.variant, "key": self.key, "per_category": self.per_cat}


def key_order(key: str) -> Tuple:
    """14a tie rule (larger = preferred): the higher threshold, then fewer looks."""
    k = key.split("|")[0]
    if k.startswith("t_"):
        return (float(k[5:]),)
    if k.startswith("r1_"):
        s, n, h = parse_r1(k)
        return (h, s, -n)
    v, g = parse_r2(k)
    return (v, -g)


def family_keys(fam: str, f3: bool) -> List[str]:
    return [k for k in KEYS if family(k) == fam and (k.endswith(F3_TAG) == f3)]


def pick(score: Callable[[str, str], float], cands: Sequence[Tuple[str, str]]) -> Tuple[str, str, float]:
    """(variant, key, half-A score) with the highest score; ties -> key_order, then the earlier candidate variant."""
    order = {v: -i for i, v in enumerate(dict.fromkeys(v for v, _ in cands))}
    best = max(cands, key=lambda c: (round(score(*c), 12), key_order(c[1]), order[c[0]]))
    return best[0], best[1], score(*best)


def proj_candidates(spec: ProjSpec, kind: str) -> List[str]:
    """F1 settings in tie order (milder first): med larger d first; clu smaller k, then smaller min."""
    if kind == "med":
        return sorted(spec.med_names(), key=lambda n: -float(n[5:]))
    return spec.clu_names()


def metrics_of(bench: Bench, sel: Sel, arm: str, half: str) -> Dict[str, Any]:
    d = bench.data[arm]
    m = bench.mask(arm, half)
    succ = bench.select(arm, sel, "success", half)
    stops = bench.select(arm, sel, "step", half) >= 0
    orac = float(d["oracle"][m].mean()) if m.any() else float("nan")
    sm = float(succ.mean()) if m.any() else float("nan")
    share = (lambda f: float(bench.select(arm, sel, f, half)[stops].mean()) if stops.any() else None)
    return {"runs": int(m.sum()), "success": sm, "spl": float(bench.select(arm, sel, "spl", half).mean()),
            "oracle": orac, "capture": sm / orac if orac > 0 else None, "stops": int(stops.sum()),
            "precision": share("success"), "wrong_object": share("wrong"), "near_miss": share("near"),
            "not_in_view": (None if not stops.any() else 1.0 - float(bench.select(arm, sel, "in_view", half)[stops].mean()))}


def summary_of(bench: Bench, sel: Sel) -> Dict[str, Any]:
    """Per arm and half the metrics, plus the mean over arms and the pooled STOP precision."""
    per = {a: {h: metrics_of(bench, sel, a, h) for h in HALVES} for a in bench.arms}
    mean = {}
    for h in HALVES:
        stops = sum(per[a][h]["stops"] for a in bench.arms)
        good = sum(round((per[a][h]["precision"] or 0) * per[a][h]["stops"]) for a in bench.arms)
        nv = [per[a][h]["not_in_view"] for a in bench.arms if per[a][h]["not_in_view"] is not None]
        wr = [per[a][h]["wrong_object"] for a in bench.arms if per[a][h]["wrong_object"] is not None]
        mean[h] = {"success": float(np.mean([per[a][h]["success"] for a in bench.arms])),
                   "spl": float(np.mean([per[a][h]["spl"] for a in bench.arms])),
                   "oracle": float(np.mean([per[a][h]["oracle"] for a in bench.arms])),
                   "stops": stops, "precision_pooled": good / stops if stops else None,
                   "wrong_object": float(np.mean(wr)) if wr else None, "not_in_view": float(np.mean(nv)) if nv else None}
    return {**sel.to_json(), "arms": per, "mean": mean}


def episode_means(bench: Bench, sel: Sel) -> Dict[str, float]:
    """episode -> mean over the arms of rule-stop success (paired statistics)."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for a in bench.arms:
        d = bench.data[a]
        for e, s in zip(d["episode"], bench.select(a, sel, "success", "all")):
            acc[str(e)].append(float(s))
    return {e: float(np.mean(v)) for e, v in acc.items() if len(v) == len(bench.arms)}


def stage1_rows(bench: Bench, spec: ProjSpec) -> Tuple[Dict[str, Dict[str, Sel]], Dict[str, str]]:
    """Rows base ... +F1(best)+F2+F3 per family (module docstring), each picked on half A."""
    msA = bench.mean_success("A")
    sc = lambda v, k: float(msA[bench.vindex[v], KEY_INDEX[k]])  # noqa: E731
    rows: Dict[str, Dict[str, Sel]] = {}
    f1best: Dict[str, str] = {}
    for fam in FAMILIES:
        off, on = family_keys(fam, False), family_keys(fam, True)
        r: Dict[str, Sel] = {}
        base = Variant().name
        r["base"] = Sel(*pick(sc, [(base, k) for k in off])[:2])
        r["base14a"] = Sel(base, PICK14A[fam])
        r["f3"] = Sel(*pick(sc, [(base, k) for k in on])[:2])
        r["f2"] = Sel(*pick(sc, [(Variant(proj="base", score="cal").name, k) for k in off])[:2])
        med = pick(sc, [(Variant(proj=p).name, k) for p in proj_candidates(spec, "med") for k in off])
        clu = pick(sc, [(Variant(proj=p).name, k) for p in proj_candidates(spec, "clu") for k in off])
        r["f1med"], r["f1clu"] = Sel(*med[:2]), Sel(*clu[:2])
        best = Variant.parse(med[0] if round(med[2], 12) >= round(clu[2], 12) else clu[0]).proj
        f1best[fam] = best
        r["f1best"] = r["f1med"] if best == Variant.parse(med[0]).proj else r["f1clu"]
        cal = Variant(proj=best, score="cal").name
        r["f1f2"] = Sel(*pick(sc, [(cal, k) for k in off])[:2])
        r["f1f2f3"] = Sel(*pick(sc, [(cal, k) for k in on])[:2])
        rows[fam] = r
    return rows, f1best


def cmd_pick(args) -> None:
    runs, out = layouts(args.config)
    pr = practice(runs)
    bench = Bench(out, "stopbench", args.arms, pr["episode_half"])
    expected = len(episodes_filter()) or len(pr["episode_half"])
    missing = {a: expected - len(bench.data[a]["episode"]) for a in args.arms}
    if any(missing.values()):
        raise SystemExit(f"stop bench incomplete: missing runs {missing}")
    rows, f1best = stage1_rows(bench, proj_spec(runs.cfg))
    res = {"f1best": f1best, "rows": {f: {r: s.to_json() for r, s in rr.items()} for f, rr in rows.items()}}
    write_text_atomic(out / "stopbench" / "PICK1.json", json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1)[:2000], flush=True)


def ccat_pick(bench_c: Bench, sel7: Sel, grid: Sequence[float], categories: Sequence[str], fam: str) -> Sel:
    """Row 8: per category the c (grid) with the highest half-A mean success over the arms at row 7's key on that
    category's episodes (ties -> the higher c); then the key re-picked on half A with those c."""
    proj = Variant.parse(sel7.variant).proj
    per_cat = {}
    for cat in categories:
        def sc(c: float) -> float:
            v = Variant("det", proj, "cal", float(c)).name
            vals = []
            for a in bench_c.arms:
                d = bench_c.data[a]
                m = (d["half"] == "A") & (d["category"] == cat)
                vals.append(float(d["success"][m][:, bench_c.vindex[v], KEY_INDEX[sel7.key]].mean()) if m.any() else 0.0)
            return float(np.mean(vals))
        best = max(grid, key=lambda c: (round(sc(c), 12), c))
        per_cat[cat] = Variant("det", proj, "cal", float(best)).name
    keys = family_keys(fam, True)

    def sk(_: str, k: str) -> float:
        return float(np.mean([bench_c.select(a, Sel("", k, per_cat), "success", "A").mean() for a in bench_c.arms]))
    _, key, _ = pick(sk, [("c", k) for k in keys])
    return Sel(f"det|{proj}|cal|c_cat", key, per_cat)


class Merged:
    """Main bench + c_cat bench as one Bench-like view (select / data / arms)."""

    def __init__(self, a: Bench, b: Optional[Bench]) -> None:
        self.main, self.cc = a, b
        self.arms, self.data, self.half_of = a.arms, a.data, a.half_of

    def mask(self, arm: str, half: str) -> np.ndarray:
        return self.main.mask(arm, half)

    def select(self, arm: str, sel: Sel, field: str, half: str = "all") -> np.ndarray:
        return (self.cc if sel.per_cat else self.main).select(arm, sel, field, half)


def keep_rule(merged: Merged, rows: Dict[str, Dict[str, Sel]], summ: Dict[str, Dict[str, Any]], home_of: Dict[str, str],
              gain: float, boot: Dict[str, int]) -> Dict[str, Any]:
    """Every COMPARE pair per family: half-B mean success diff (+ home-clustered CI over episodes, all 5 arms averaged
    per episode), pooled STOP precision; helps = diff >= gain and precision not lower."""
    from vint_train.mapmad.closed_loop.exam_stats import paired

    out: Dict[str, Any] = {}
    for fam in FAMILIES:
        res = {}
        for row, prev, fix in COMPARE:
            a, b = summ[fam][row]["mean"]["B"], summ[fam][prev]["mean"]["B"]
            ea, eb = episode_means_b(merged, rows[fam][row]), episode_means_b(merged, rows[fam][prev])
            d = paired(ea, eb, home_of, boot["draws"], boot["seed"])
            pa, pb = a["precision_pooled"], b["precision_pooled"]
            diff = a["success"] - b["success"]
            helps = diff >= gain - 1e-12 and (pa if pa is not None else 0.0) >= (pb if pb is not None else 0.0) - 1e-12
            res[f"{row}<{prev}"] = {"row": row, "vs": prev, "fix": fix, "diff_B": diff, "ci_B": d.get("home_ci"),
                                    "precision_B": [pa, pb], "helps": bool(helps)}
        out[fam] = res
    return out


def episode_means_b(merged: Merged, sel: Sel) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for a in merged.arms:
        d = merged.data[a]
        m = merged.mask(a, "B")
        for e, s in zip(d["episode"][m], merged.select(a, sel, "success", "B")):
            acc[str(e)].append(float(s))
    return {e: float(np.mean(v)) for e, v in acc.items()}


def stop_counts_b(merged: Merged, sel: Sel) -> Dict[str, List[float]]:
    """episode -> [correct stops, stops] summed over the arms on half B (the parts of the pooled STOP precision)."""
    acc: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    for a in merged.arms:
        d = merged.data[a]
        m = merged.mask(a, "B")
        stops = merged.select(a, sel, "step", "B") >= 0
        good = merged.select(a, sel, "success", "B").astype(bool) & stops
        for e, s, g in zip(d["episode"][m], stops, good):
            acc[str(e)][0] += float(g)
            acc[str(e)][1] += float(s)
    return dict(acc)


def precision_diff_ci(ca: Dict[str, List[float]], cb: Dict[str, List[float]], home_of: Dict[str, str], draws: int,
                      seed: int) -> Tuple[Optional[float], Optional[List[float]]]:
    """Pooled STOP precision of a minus that of b (episodes both have) and its home-clustered bootstrap 95 % CI: the
    homes are drawn with replacement, the same draw for a and b, and each draw recomputes both pooled ratios."""
    ids = sorted(set(ca) & set(cb))
    homes = sorted({home_of[i] for i in ids})
    hk = {h: k for k, h in enumerate(homes)}
    pa, pb = np.zeros((len(homes), 2)), np.zeros((len(homes), 2))
    for i in ids:
        pa[hk[home_of[i]]] += ca[i]
        pb[hk[home_of[i]]] += cb[i]

    def diff(sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return sa[..., 0] / sa[..., 1] - sb[..., 0] / sb[..., 1]
    point = float(diff(pa.sum(0), pb.sum(0)))
    if not np.isfinite(point):
        return None, None
    pick_h = np.random.default_rng(seed).integers(0, len(homes), size=(draws, len(homes)))
    d = diff(pa[pick_h].sum(1), pb[pick_h].sum(1))
    d = d[np.isfinite(d)]
    return point, [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def passes_keep(keep_fam: Dict[str, Any], row: str) -> bool:
    """A row passes the keep rule when every fix it adds 'helps' (CHAIN); base passes."""
    return row == "base" or all(keep_fam[STEP_KEY[c]]["helps"] for c in CHAIN[row])


def fix_effects(merged: Merged, rows: Dict[str, Dict[str, Sel]], summ: Dict[str, Dict[str, Any]], keep: Dict[str, Any],
                home_of: Dict[str, str], boot: Dict[str, int]) -> Dict[str, Any]:
    """Per family and fix (EFFECTS): half-B change in mean rule-stop success and in pooled STOP precision vs the row
    without it, each with a home-clustered 95 % CI, and the keep-rule verdict. The best combination is the combination
    row (COMBOS) with the highest half-A mean success (ties -> fewer fixes), compared with base."""
    from vint_train.mapmad.closed_loop.exam_stats import paired

    out: Dict[str, Any] = {}
    for fam in FAMILIES:
        combo = max(COMBOS, key=lambda r: (round(summ[fam][r]["mean"]["A"]["success"], 12), -COMBOS.index(r)))
        res = []
        for fix, row, prev in EFFECTS:
            row = row or combo
            a, b = summ[fam][row]["mean"]["B"], summ[fam][prev]["mean"]["B"]
            d = paired(episode_means_b(merged, rows[fam][row]), episode_means_b(merged, rows[fam][prev]), home_of,
                       boot["draws"], boot["seed"])
            pd, pci = precision_diff_ci(stop_counts_b(merged, rows[fam][row]), stop_counts_b(merged, rows[fam][prev]),
                                        home_of, boot["draws"], boot["seed"])
            verdict = keep[fam][STEP_KEY["f1f2f3c"]]["helps"] if fix == "c_cat" else passes_keep(keep[fam], row)
            res.append({"fix": fix, "row": row, "vs": prev, "success_B": [a["success"], b["success"]],
                        "success_diff_B": a["success"] - b["success"], "success_ci_B": d.get("home_ci"),
                        "precision_B": [a["precision_pooled"], b["precision_pooled"]], "precision_diff_B": pd,
                        "precision_ci_B": pci, "stops_B": [a["stops"], b["stops"]], "passes_keep": bool(verdict)})
        out[fam] = {"best_combination": combo, "rows": res}
    return out


def fix_effects_md(eff: Dict[str, Any]) -> List[str]:
    """tables/fix_effects.md: one plain table, half B, mean over the 5 arms."""
    sg = lambda x: "-" if x is None else f"{x:+.3f}"  # noqa: E731
    ci = lambda c: "-" if not c else f"[{c[0]:+.3f}, {c[1]:+.3f}]"  # noqa: E731
    lines = ["# Effect of each fix on half B (mean over the 5 arms; picks on half A)", "",
             "Correct stops = rule-stop success (share of episodes where the stop module stops at the target). STOP "
             "precision = correct stops / all stops, pooled over the 5 arms. CIs: home-clustered bootstrap, 10,000 "
             "draws, seed 14401. Keep rule: success +0.02 or more and precision not lower (a combination needs every "
             "step). c_cat is compared with F1(best)+F2+F3; the others with base (14a marks, key re-picked).", "",
             "| rule | fix | row (vs) | correct stops B new / old | change [CI] | STOP precision B new / old | change [CI] "
             "| stops new / old | keep rule |", "|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        for r in eff[fam]["rows"]:
            fix = r["fix"] + (f" ({ROW_NAME[r['row']]})" if r["fix"] == "best combination" else "")
            lines.append(f"| {FAMILY_NAME[fam]} | {fix} | `{r['row']}` (vs `{r['vs']}`) | {fmt(r['success_B'][0])} / "
                         f"{fmt(r['success_B'][1])} | {sg(r['success_diff_B'])} {ci(r['success_ci_B'])} | "
                         f"{fmt(r['precision_B'][0])} / {fmt(r['precision_B'][1])} | {sg(r['precision_diff_B'])} "
                         f"{ci(r['precision_ci_B'])} | {r['stops_B'][0]} / {r['stops_B'][1]} | "
                         f"{'pass' if r['passes_keep'] else 'fail'} |")
    return lines


def recommend(keep: Dict[str, Any], summ: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per family: the row with the highest half-B mean success among rows whose every added fix 'helps' (CHAIN; the
    F1 step of a combination = F1 on top of F2 and F2 on top of F1); none -> base. Overall: the best family's."""
    out = {}
    for fam in FAMILIES:
        ok = {"base": True}
        for row in CHAIN:
            ok[row] = passes_keep(keep[fam], row)
        cand = [r for r in ok if ok[r]]
        best = max(cand, key=lambda r: (round(summ[fam][r]["mean"]["B"]["success"], 12), -ROWS.index(r) if r in ROWS else 0))
        out[fam] = {"row": best, "eligible": cand, "success_B": summ[fam][best]["mean"]["B"]["success"]}
    fam = max(FAMILIES, key=lambda f: round(out[f]["success_B"], 12))
    out["overall"] = {"family": fam, **out[fam]}
    return out


def fmt(x: Optional[float], p: int = 3) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"


def truth_ceiling(truth: Bench, rows: Dict[str, Dict[str, Sel]], spec: ProjSpec) -> Dict[str, Any]:
    """Per family and projection (base, the det rows' F1-med / F1-clu settings, every setting), F3 off / on: truth
    success (mean over arms) at the det row's key (no re-pick) and at its own half-A pick."""
    msA = truth.mean_success("A")
    out: Dict[str, Any] = {}
    for fam in FAMILIES:
        res = {}
        for proj in spec.names():
            v = Variant("true", proj, "map").name
            for f3 in (False, True):
                keys = family_keys(fam, f3)
                own = pick(lambda vv, k: float(msA[truth.vindex[vv], KEY_INDEX[k]]), [(v, k) for k in keys])
                row = rows[fam]["f3" if f3 else "base"].key
                res[f"{proj}{'|f3' if f3 else ''}"] = {
                    "own_pick": own[1], "own": {h: float(truth.mean_success(h)[truth.vindex[v], KEY_INDEX[own[1]]]) for h in HALVES},
                    "at_det_key": row, "det_key": {h: float(truth.mean_success(h)[truth.vindex[v], KEY_INDEX[row]]) for h in HALVES}}
        out[fam] = res
    return out


def cmd_final(args) -> None:
    runs, out = layouts(args.config)
    pr = practice(runs)
    half_of = pr["episode_half"]
    home_of = {e: x["home"] for e, x in practice_episodes(runs).items()}
    cfg = runs.cfg["p14b1"]
    spec = proj_spec(runs.cfg)
    bench = Bench(out, "stopbench", args.arms, half_of)
    rows, f1best = stage1_rows(bench, spec)
    cc_dir = out / "stopbench_ccat" / "runs"
    bench_c = Bench(out, "stopbench_ccat", args.arms, half_of) if cc_dir.exists() else None
    merged = Merged(bench, bench_c)
    cats = sorted({str(c) for a in args.arms for c in bench.data[a]["category"]})
    for fam in FAMILIES:
        if bench_c is not None:
            rows[fam]["f1f2f3c"] = ccat_pick(bench_c, rows[fam]["f1f2f3"], cfg["ccat_grid"], cats, fam)
        else:
            rows[fam]["f1f2f3c"] = rows[fam]["f1f2f3"]
    summ = {f: {r: summary_of(merged, s) for r, s in rr.items()} for f, rr in rows.items()}
    keep = keep_rule(merged, rows, summ, home_of, float(cfg["keep_gain"]), cfg["bootstrap"])
    rec = recommend(keep, summ)
    eff = fix_effects(merged, rows, summ, keep, home_of, cfg["bootstrap"])
    truth = truth_ceiling(bench, rows, spec) if any(v.startswith("true|") for v in bench.variants) else None
    par = {a: [int(bench.data[a]["parity"][:, 0].sum()), int(bench.data[a]["parity"][:, 1].sum())] for a in args.arms}
    res = {"temperature": temperature(out), "f1best": f1best, "ccat_done": bench_c is not None, "parity14a": par,
           "rows": {f: {r: summ[f][r] for r in rows[f]} for f in FAMILIES}, "keep": keep, "recommended": rec,
           "fix_effects": eff,
           "truth": truth, "arms": list(args.arms), "practice_fingerprint": pr["fingerprint"]}
    write_text_atomic(out / "stopbench" / "RESULTS.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(out / "stopbench" / "stopbench.md", "\n".join(results_md(res, merged, rows)) + "\n")
    write_text_atomic(out / "tables" / "fix_effects.md", "\n".join(fix_effects_md(eff)) + "\n")
    if truth:
        write_text_atomic(out / "stopbench_truth" / "truth.md", "\n".join(truth_md(res)) + "\n")
    print("\n".join(results_md(res, merged, rows)[:40]), flush=True)


def results_md(res: Dict[str, Any], merged: Merged, rows: Dict[str, Dict[str, Sel]]) -> List[str]:
    L = ["# 14b1.3 stop bench with fixed marks (practice set; report only)", "",
         f"Practice set `{res['practice_fingerprint'][:12]}…`, arms {', '.join(res['arms'])}, seed 0 (one seed). "
         f"T = {res['temperature']:.2f}. F1(best) per rule: {res['f1best']}. Parity with the 14a bench (det|base|map, "
         f"F3 off, every 14a grid point): {res['parity14a']} (checked, same).", "",
         "## Mean over the 5 arms (rule-stop success; STOP precision pooled; picks on half A)", "",
         "| rule | row | variant | key | A | B | all | precision B | wrong obj. B | not in view B | stops B |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        for r in ROWS:
            s = res["rows"][fam][r]
            m = s["mean"]
            L.append(f"| {FAMILY_NAME[fam]} | {ROW_NAME[r]} | `{s['variant']}` | `{s['key']}` | {fmt(m['A']['success'])} | "
                     f"{fmt(m['B']['success'])} | {fmt(m['all']['success'])} | {fmt(m['B']['precision_pooled'])} | "
                     f"{fmt(m['B']['wrong_object'])} | {fmt(m['B']['not_in_view'])} | {m['B']['stops']} |")
    L += ["", "## Keep rule (half B: mean success +0.02 and pooled STOP precision not lower)", "",
          "| rule | fix | row vs row | diff B [home CI] | precision B new / old | helps |", "|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        for k, x in res["keep"][fam].items():
            ci = x["ci_B"] or [float("nan")] * 2
            L.append(f"| {FAMILY_NAME[fam]} | {x['fix']} | {ROW_NAME.get(x['row'], x['row'])} vs "
                     f"{ROW_NAME.get(x['vs'], x['vs'])} | {x['diff_B']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{fmt(x['precision_B'][0])} / {fmt(x['precision_B'][1])} | {'yes' if x['helps'] else 'no'} |")
    rec = res["recommended"]
    L += ["", "## Recommended for 14b-2 (best 'helps' combination on half B; report only)", ""]
    for fam in FAMILIES:
        L.append(f"- {FAMILY_NAME[fam]}: **{ROW_NAME[rec[fam]['row']]}** (half B {rec[fam]['success_B']:.3f}; eligible "
                 f"{', '.join(rec[fam]['eligible'])})")
    L.append(f"- **Overall: {FAMILY_NAME[rec['overall']['family']]} · {ROW_NAME[rec['overall']['row']]}**")
    L += ["", "## Per arm (picked rows)", "", "| rule | row | arm | half | runs | success | SPL | oracle | capture | stops | "
          "precision | wrong obj. | near miss | not in view |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        for r in ROWS:
            for a in res["arms"]:
                for h in HALVES:
                    m = res["rows"][fam][r]["arms"][a][h]
                    L.append(f"| {fam} | {r} | {a} | {h} | {m['runs']} | {fmt(m['success'])} | {fmt(m['spl'])} | "
                             f"{fmt(m['oracle'])} | {fmt(m['capture'])} | {m['stops']} | {fmt(m['precision'])} | "
                             f"{fmt(m['wrong_object'])} | {fmt(m['near_miss'])} | {fmt(m['not_in_view'])} |")
    L += ["", "## Per category (all; base vs the recommended row of each rule)", "",
          "| rule | row | category | runs (5 arms) | success | oracle |", "|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        for r in sorted({"base", rec[fam]["row"]}):
            sel = rows[fam][r]
            acc: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
            for a in merged.arms:
                d = merged.data[a]
                for c, s, o in zip(d["category"], merged.select(a, sel, "success"), d["oracle"]):
                    acc[str(c)].append((bool(s), bool(o)))
            for c in sorted(acc):
                v = acc[c]
                L.append(f"| {fam} | {r} | {c} | {len(v)} | {np.mean([x[0] for x in v]):.3f} | {np.mean([x[1] for x in v]):.3f} |")
    return L


def truth_md(res: Dict[str, Any]) -> List[str]:
    L = ["# 14b1.3 true-mark ceiling (report only)", "",
         "Simulator-truth masks (every true instance in a frame = a look with score 1.0, MT-style) through the same "
         "projection variants, on the same logs. Columns: mean rule-stop success over the 5 arms at the YOLOE row's key "
         "(base / +F3 pick, no re-pick) and at the truth variant's own half-A pick.", "",
         "| rule | projection | F3 | key (det row) | A | B | all | own pick | A | B | all |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMILIES:
        for name, x in res["truth"][fam].items():
            proj, f3 = (name.split("|")[0], "on") if "|f3" in name else (name, "off")
            L.append(f"| {fam} | {proj} | {f3} | `{x['at_det_key']}` | {fmt(x['det_key']['A'])} | {fmt(x['det_key']['B'])} | "
                     f"{fmt(x['det_key']['all'])} | `{x['own_pick']}` | {fmt(x['own']['A'])} | {fmt(x['own']['B'])} | "
                     f"{fmt(x['own']['all'])} |")
    return L


# --- mark statistics --------------------------------------------------------------------------------------------------
def box_dist_cells(xy: np.ndarray, insts: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Per 2D point (N, 2) the horizontal distance to the nearest instance box (box_distance_2d, vectorised)."""
    best = np.full(len(xy), np.inf)
    for inst in insts:
        cx, _, cz = inst["centre"]
        sx, _, sz = inst["size"]
        x, z = -xy[:, 1], -xy[:, 0]
        d = np.hypot(np.maximum(np.abs(x - cx) - sx / 2.0, 0.0), np.maximum(np.abs(z - cz) - sz / 2.0, 0.0))
        best = np.minimum(best, d)
    return best


def marks_run(job) -> Tuple[str, Dict[str, Any]]:
    """Per projection: for YOLOE rows on a true target (IoU >= 0.3) the median distance of their cells to the nearest
    usable instance box and their area; area of rows on nothing true (IoU < 0.3); the same for true-instance rows."""
    ev_file, proj_file, episode, names, arm = job
    ev, proj = stop_evidence.load(Path(ev_file)), stop_evidence.load(Path(proj_file))
    insts = episode["target"]["instances"]
    assert box_distance_2d  # the definition vectorised in box_dist_cells
    shape1, res, origin = int(ev["shape"][1]), float(ev["res"]), ev["origin"]
    on_true = np.zeros(len(ev["det_t"]), bool)
    tb: Dict[int, List[np.ndarray]] = defaultdict(list)
    for k, t in enumerate(ev["true_t"].tolist()):
        tb[t].append(ev["true_box"][k])
    for k, t in enumerate(ev["det_t"].tolist()):
        b = ev["det_box"][k]
        on_true[k] = np.isfinite(b).all() and any(iou(list(b), list(x)) >= 0.3 for x in tb.get(t, []))
    out: Dict[str, Any] = {"arm": arm}
    for pre, rowsel in (("det_on", on_true), ("det_off", ~on_true), ("true", None)):
        src = "true" if pre == "true" else "det"
        for nm in names:
            cells, ptr = proj[f"{src}_cells_{nm}"], proj[f"{src}_ptr_{nm}"]
            idx = np.arange(len(ptr) - 1) if rowsel is None else np.flatnonzero(rowsel)
            dist, far90, smear, area, empty = [], [], [], [], 0
            for k in idx.tolist():
                c = cells[ptr[k]:ptr[k + 1]].astype(np.int64)
                if c.size == 0:
                    empty += 1
                    continue
                area.append(c.size * res * res)
                if pre != "det_off":
                    i, j = np.divmod(c, shape1)
                    xy = np.stack([origin[0] + (i + 0.5) * res, origin[1] + (j + 0.5) * res], axis=1)
                    dd = box_dist_cells(xy, insts)
                    dist.append(float(np.median(dd)))
                    far90.append(float(np.percentile(dd, 90)))
                    smear.append(float(np.mean(dd > 0.5)))
            out[f"{pre}|{nm}"] = {"rows": int(idx.size), "empty": empty, "dist": dist, "far90": far90, "smear": smear,
                                  "area": area}
    return proj_file, out


def cmd_marks(args) -> None:
    runs, out = layouts(args.config)
    eps = practice_episodes(runs)
    names = proj_spec(runs.cfg).names()
    only = episodes_filter()
    jobs = []
    for arm in args.arms:
        for f in sorted((out / "replay" / "baseline" / arm).glob("*.npz")):
            rid = f.name[:-4]
            if only and parse_run_id(rid)[0] not in only:
                continue
            jobs.append((str(runs.root / "evidence" / arm / f"{rid}.npz"), str(f), eps[parse_run_id(rid)[0]], names, arm))
    acc: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"dist": [], "far90": [], "smear": [], "area": [], "rows": 0,
                                                          "empty": 0})
    for _, r in imap_safe(marks_run, jobs, args.workers):
        for k, v in r.items():
            if k == "arm":
                continue
            a = acc[k]
            for f in ("dist", "far90", "smear", "area"):
                a[f] += v[f]
            a["rows"] += v["rows"]
            a["empty"] += v["empty"]
    q = lambda xs, p: float(np.percentile(xs, p)) if xs else float("nan")  # noqa: E731
    res = {k: {"rows": v["rows"], "empty": v["empty"], "dist_p50": q(v["dist"], 50), "dist_p90": q(v["dist"], 90),
               "far90_p50": q(v["far90"], 50), "far90_p90": q(v["far90"], 90),
               "smear_mean": float(np.mean(v["smear"])) if v["smear"] else float("nan"),
               "area_p50": q(v["area"], 50), "area_p90": q(v["area"], 90)} for k, v in acc.items()}
    L = ["# 14b1.3 mark statistics: old (base) vs new projections (report only)", "",
         f"All baseline runs of {', '.join(args.arms)}. Per row (one detection or true instance in one frame): its "
         "marked cells; distance = median over its cells of the horizontal distance to the nearest usable target "
         "instance's box (0 inside); far = the 90th percentile of its cells' distances (the smear tail); smear = the "
         "share of its cells > 0.5 m from every target box (mean over rows); area = cells x 0.01 m². 'on target' = "
         "YOLOE box IoU >= 0.3 with a true box of the frame; 'off target' = IoU < 0.3 with every true box; 'true' = "
         "simulator-truth instances. Over rows: p50 / p90.", "",
         "| rows | projection | rows | empty | median dist p50 / p90 (m) | far p50 / p90 (m) | smear share | "
         "area p50 / p90 (m²) |", "|---|---|---|---|---|---|---|---|"]
    for pre, lab in (("det_on", "YOLOE on target"), ("det_off", "YOLOE off target"), ("true", "true instances")):
        for nm in names:
            r = res[f"{pre}|{nm}"]
            L.append(f"| {lab} | {nm} | {r['rows']} | {r['empty']} | {fmt(r['dist_p50'], 2)} / {fmt(r['dist_p90'], 2)} | "
                     f"{fmt(r['far90_p50'], 2)} / {fmt(r['far90_p90'], 2)} | {fmt(r['smear_mean'])} | "
                     f"{fmt(r['area_p50'], 2)} / {fmt(r['area_p90'], 2)} |")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "tables" / "marks.json", json.dumps(res, indent=1) + "\n")
    write_text_atomic(out / "tables" / "marks.md", "\n".join(L) + "\n")
    print("\n".join(L[:12]), flush=True)


# --- step 7: driving-test re-score -----------------------------------------------------------------------------------
def cmd_drive(args) -> None:
    """The recommended variant (overall and per rule) + today's rule on the 14a driving-test runs (report only)."""
    from vint_train.mapmad.closed_loop.exam_stats import paired

    _, out = layouts(args.config)
    res = json.loads((out / "stopbench" / "RESULTS.json").read_text())
    rec = res["recommended"]
    sels = {f: res["rows"][f][rec[f]["row"]] for f in FAMILIES}
    names = sorted({Variant().name} | {s["variant"] for s in sels.values() if not s["per_category"]}
                   | {v for s in sels.values() if s["per_category"] for v in s["per_category"].values()})
    variants = [Variant.parse(n) for n in names]
    cfg = RunLayout.load(args.config).cfg["p14b1"]
    t_cal = temperature(out)
    jobs = []
    for sub, arms in cfg["drive_subdirs"].items():
        runs = RunLayout.load_runs(args.config, subdir=f"runs/{sub}")
        jobs += score_jobs(runs, out, arms, variants, f"rescore_driving/{sub}", t_cal, args.force, False)
    if jobs:
        run_pool(jobs, args.workers)
    runs0 = RunLayout.load_runs(args.config)
    pr = practice(runs0)
    home_of = {e: x["home"] for e, x in practice_episodes(runs0).items()}
    L = ["# Step 7: driving tests re-scored with the recommended marks (report only)", "",
         "14a driving-test runs (A1, W4, A1 + W4 x M21b, M21bh, N; all 219 practice episodes, seed 0), stop module "
         "re-scored: today's rule (14a, tau 0.45) vs each rule's recommended row. Paired by episode, home-clustered "
         "bootstrap 95 % CI.", "",
         "| switch | arm | runs | oracle | today tau 0.45 | rule | recommended row | success | diff [CI] |",
         "|---|---|---|---|---|---|---|---|---|"]
    out_res: Dict[str, Any] = {}
    for sub, arms in cfg["drive_subdirs"].items():
        bench = Bench(out, f"rescore_driving/{sub}", arms, pr["episode_half"])
        for a in arms:
            base = Sel(Variant().name, "t_tau0.45")
            sb = bench.select(a, base, "success")
            for fam in FAMILIES:
                s = sels[fam]
                sel = Sel(s["variant"], s["key"], s["per_category"])
                sn = bench.select(a, sel, "success")
                eps = bench.data[a]["episode"]
                d = paired(dict(zip(eps, sn.astype(float))), dict(zip(eps, sb.astype(float))), home_of,
                           cfg["bootstrap"]["draws"], cfg["bootstrap"]["seed"])
                out_res.setdefault(sub, {}).setdefault(a, {})[fam] = {"today": float(sb.mean()), "new": float(sn.mean()),
                                                                       "diff": d}
                L.append(f"| {sub} | {a} | {len(eps)} | {bench.data[a]['oracle'].mean():.3f} | {sb.mean():.3f} | {fam} | "
                         f"{ROW_NAME[rec[fam]['row']]} | {sn.mean():.3f} | {d['diff']:+.3f} [{d['home_ci'][0]:+.3f}, "
                         f"{d['home_ci'][1]:+.3f}] |")
    (out / "rescore_driving").mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "rescore_driving" / "rescore.json", json.dumps(out_res, indent=1) + "\n")
    write_text_atomic(out / "rescore_driving" / "rescore.md", "\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# --- gate: hashes, fairness, replay check -----------------------------------------------------------------------------
def cmd_hashes(args) -> None:
    """sha256 of every 14a log, evidence and dets file of the baseline + driving runs -> checks/hashes_<tag>.json."""
    runs, out = layouts(args.config)
    res = {}
    for sub in ["baseline"] + list(runs.cfg["p14b1"]["drive_subdirs"]):
        root = runs.base / "runs" / sub
        for kind in ("logs", "evidence", "dets"):
            for f in sorted((root / kind).glob("*/*")):
                res[str(f.relative_to(runs.base))] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out / "checks").mkdir(parents=True, exist_ok=True)
    write_text_atomic(out / "checks" / f"hashes_{args.tag}.json", json.dumps(res, indent=0, sort_keys=True) + "\n")
    print(f"{len(res)} files hashed -> checks/hashes_{args.tag}.json", flush=True)


def replay_summary(out: Path, sub: str, arms: Sequence[str]) -> Dict[str, Any]:
    recs = [json.loads(f.read_text()) for a in arms for f in sorted((out / "replay" / sub / a).glob("*.json"))]
    tot = lambda k: int(sum(r[k] for r in recs))  # noqa: E731
    return {"runs": len(recs), "frames": tot("frames"), "frames_rowwise": tot("frames_rowwise"),
            "max_raw_diff": max((r["max_raw_diff"] for r in recs), default=0.0), "det_rows": tot("det_rows"),
            "fallback_rows": tot("fallback_rows"), "det_base_same": tot("det_base_same"), "true_rows": tot("true_rows"),
            "true_base_same": tot("true_base_same"), "true_frames_count_mismatch": tot("true_frames_count_mismatch"),
            "pose_max_diff": max((r["pose_max_diff"] for r in recs), default=0.0),
            "same_gpu_runs": sum(1 for r in recs if r.get("orig_gpu", r["gpu"]) == r["gpu"]),
            "seconds": float(sum(r["seconds"] for r in recs))}


def cmd_gate(args) -> None:
    runs, out = layouts(args.config)
    cfg = runs.cfg["p14b1"]
    pr = practice(runs)
    n_eps = len(episodes_filter()) or len(pr["episode_half"])
    rows: List[Tuple[str, str, str]] = []
    res_f = out / "stopbench" / "RESULTS.json"
    res = json.loads(res_f.read_text()) if res_f.exists() else None
    calib = json.loads((out / "calib" / "calib.json").read_text()) if (out / "calib" / "calib.json").exists() else None
    rows.append(("1", "F1 / F2 / F3 one at a time and together, per rule and arm, halves A / B / all",
                 "reported (stopbench/stopbench.md)" if res else "MISSING"))
    rows.append(("2", "Keep rule applied; recommended variant for 14b-2",
                 f"reported: {FAMILY_NAME[res['recommended']['overall']['family']]} · "
                 f"{ROW_NAME[res['recommended']['overall']['row']]}" if res else "MISSING"))
    rows.append(("3", "True-mark ceiling with each projection",
                 "reported (stopbench_truth/truth.md)" if res and res.get("truth") else "MISSING"))
    rows.append(("4", "Calibration: T, ECE before / after (photo-test half B)",
                 f"reported: T {calib['T']:.2f}; ECE half B raw {calib['ece']['B']['raw']:.4f} -> "
                 f"{calib['ece']['B']['calibrated']:.4f}" if calib else "MISSING"))
    marks_ok = (out / "tables" / "marks.md").exists()
    vids = sorted((out / "videos_marks_fixed").glob("*.mp4")) if (out / "videos_marks_fixed").exists() else []
    rows.append(("5", "Mark statistics old vs new; 8 videos",
                 f"{'reported (tables/marks.md)' if marks_ok else 'MISSING marks'}; {len(vids)} videos"))
    # row 6: fairness
    problems = []
    if res:
        for a, (checked, same) in res["parity14a"].items():
            if checked != same:
                problems.append(f"{a}: 14a parity {same}/{checked}")
    for a in cfg["arms"]:
        n = len(list((out / "stopbench" / "runs" / a).glob("*.npz")))
        if n != n_eps:
            problems.append(f"{a}: {n} of {n_eps} runs re-scored")
    hb, ha = out / "checks" / "hashes_before.json", out / "checks" / "hashes_after.json"
    if hb.exists() and ha.exists():
        b, a = json.loads(hb.read_text()), json.loads(ha.read_text())
        if b != a:
            problems.append(f"log hashes changed: {len(set(b.items()) ^ set(a.items()))} entries differ")
    else:
        problems.append("hash files missing")
    rows.append(("6", "Fairness: every baseline run re-scored, none missing; trajectories untouched (log hashes)",
                 "PASS" if not problems else "FAIL: " + "; ".join(problems)))
    rs = replay_summary(out, "baseline", cfg["arms"])
    ok7 = rs["max_raw_diff"] <= 1e-4 and rs["fallback_rows"] == 0 and rs["det_base_same"] == rs["det_rows"] \
        and rs["true_base_same"] == rs["true_rows"] and rs["runs"] == n_eps * len(cfg["arms"])
    rows.append(("7", "Replay check: boxes / scores equal the logged ones within §3 limits",
                 ("PASS" if ok7 else "see numbers (F1-box rows where not)") + f": {json.dumps(rs)}"))
    L = ["# G14b1 (draft, written by the chain; finished in Part 3)", "", "| # | Check | Result |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    write_text_atomic(out / "G14b1_table_draft.md", "\n".join(L) + "\n")
    write_text_atomic(out / "checks" / "replay_summary.json", json.dumps(rs, indent=1) + "\n")
    print("\n".join(L), flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["calib", "score", "pick", "final", "marks", "drive", "hashes", "gate"])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="runs/baseline")
    p.add_argument("--arms", nargs="+", default=list(ARMS5))
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--force", action="store_true")
    p.add_argument("--ccat", action="store_true", help="score: the c_cat variants of PICK1.json's F1(best)")
    p.add_argument("--tag", default="before")
    args = p.parse_args(argv)
    {"calib": cmd_calib, "score": cmd_score, "pick": cmd_pick, "final": cmd_final, "marks": cmd_marks,
     "drive": cmd_drive, "hashes": cmd_hashes, "gate": cmd_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
