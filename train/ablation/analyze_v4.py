"""Phase 6: paired gaps for V4 against V1 / V2 / V2p / V3, and the plan's Verify table.

Joins the Phase 4 results.csv (V1, V2, V2p, V3; read only) with eval_v4.py's results_v4.csv
(V4_photo, V4_p, V4_word) on sample_idx, then reuses analyze_results.py's tiers, per-metric
case subsets and bootstrap (2000 resamples; case CI and observation-trajectory CI, '*' when
the trajectory CI excludes 0). Distance-head correlations come from eval_v4.py's
summary_table_v4.csv.

Gaps: V4_photo - V1, V4_photo - V2, V4_word - V4_p (the replicated modality-gap test),
V4_word - V3. Per word (headline, >= 20 cases): V4_p, V4_word and V4_word - V4_p.

Verify (plan Phase 6; harness units, object-word tier = headline):
    1 V4_photo gc_dist_loss        pass <= 26, strong <= 23, fail > 30
    2 V4_photo dist-head corr      pass >= 0.4, strong >= 0.6, fail < 0.2
    3 V4_photo - V1 gc_action      pass <= +0.20, strong CI contains 0, fail >= +0.35
    4 V4_photo - V2 gc_action      pass CI < 0
    5 V4_word - V4_p gc_action     pass CI contains 0 or <= +0.05; fail > +0.10 with CI > 0
    6 uc_action_loss               within +-0.05 of V2 and of V1
    7 V4_word dist loss / corr / progress / heading / success@tau2   reported
CIs are the trajectory CIs.

Outputs (in eval_v4.py's folder, never overwritten): gap_analysis_v4.txt, per_word_v4.csv.

Run inside the container, from /app/visualnav-transformer/train:

    python ablation/analyze_v4.py
"""
import argparse
import csv
import json
import os

import numpy as np

from analyze_results import METRICS, N_BOOT, SEED, SKIP_WORDS, boot_ci, paired_gap, rows_for, tier_masks
from summarize_clip_eval import MIN_WORD_CASES, load

PHASE4_DIR = "/outputs/nomad_clip_eval"
DEFAULT_V4_DIR = "/outputs/nomad_clip_v6/eval"
V4_CONFIGS = ["V4_photo", "V4_p", "V4_word"]
GAPS = [("V4_photo", "V1"), ("V4_photo", "V2"), ("V4_word", "V4_p"), ("V4_word", "V3")]
TIERS = ["headline", "all"]


def load_v4(path, reference, configs=V4_CONFIGS):
    """results_v4.csv (or results_v5.csv) -> {config: {column: array}}, in Phase 4 case order."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for name in configs:
        mine = sorted((r for r in rows if r["config"] == name), key=lambda r: int(r["sample_idx"]))
        cols = {}
        for key in mine[0]:
            raw = [r[key] for r in mine]
            try:
                cols[key] = np.array([float(v) if v != "" else np.nan for v in raw])
            except ValueError:
                cols[key] = np.array(raw)
        assert np.array_equal(cols["sample_idx"], reference["sample_idx"]), f"{name} case order differs"
        assert np.array_equal(cols["distance"], reference["distance"]), f"{name} distances differ"
        out[name] = cols
    return out


def fmt_gap(gap, pct, case_ci, traj_ci):
    star = "*" if traj_ci[0] > 0 or traj_ci[1] < 0 else " "
    return (f"{gap:+8.3f}{star} ({pct:+6.1f}%)  case [{case_ci[0]:+.3f}, {case_ci[1]:+.3f}]  "
            f"traj [{traj_ci[0]:+.3f}, {traj_ci[1]:+.3f}]")


def grade(value, passed, strong=None, failed=None):
    if strong is not None and strong(value):
        return "STRONG PASS"
    if passed(value):
        return "PASS"
    if failed is not None and not failed(value):
        return "BORDERLINE (neither pass nor fail)"
    return "FAIL"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--v4-dir", default=DEFAULT_V4_DIR)
    args = parser.parse_args()
    paths = {n: os.path.join(args.v4_dir, n) for n in ("gap_analysis_v4.txt", "per_word_v4.csv")}
    for path in paths.values():
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; refusing to overwrite")

    with open(os.path.join(PHASE4_DIR, "summary.meta.json")) as f:
        meta = json.load(f)
    taus = {"success@tau1": meta["tau1"], "success@tau2": meta["tau2"]}
    data = load(os.path.join(PHASE4_DIR, "results.csv"))
    data.update(load_v4(os.path.join(args.v4_dir, "results_v4.csv"), data["V1"]))
    tiers, positive, within = tier_masks(data)
    traj = data["V1"]["traj_id"]
    with open(os.path.join(args.v4_dir, "summary_table_v4.csv")) as f:
        corr = {(r["tier"], r["config"]): float(r["mean"]) for r in csv.DictReader(f)
                if r["metric"] == "dist_head_corr"}
    rng = np.random.default_rng(SEED)
    out = [f"Phase 6 V4 gap analysis; bootstrap {N_BOOT}, seed {SEED}; '*' = trajectory CI excludes 0",
           f"tau1 = {taus['success@tau1']}, tau2 = {taus['success@tau2']} units (1 unit = 0.12 m)"]

    gaps = {}
    for tier in TIERS:
        keep = tiers[tier]
        out += ["", f"== {tier}: {int(keep.sum()):,} cases =="]
        out.append(f"{'metric':<16}" + "".join(f"{c:>10}" for c in ["V1", "V2", "V2p", "V3"] + V4_CONFIGS))
        for metric in METRICS + ["gc_dist_loss"]:
            cells = []
            for c in ["V1", "V2", "V2p", "V3"] + V4_CONFIGS:
                rows = rows_for(metric, keep, positive, within)
                v = (data[c]["goal_step_dist"][rows] < taus[metric]).astype(float) if metric in taus \
                    else data[c][metric][rows]
                cells.append(f"{np.nanmean(v):>10.3f}")
            out.append(f"{metric:<16}" + "".join(cells))
        out.append(f"{'dist_head_corr':<16}" + "".join(
            f"{corr[(tier, c)]:>10.3f}" for c in ["V1", "V2", "V2p", "V3"] + V4_CONFIGS))
        for a, b in GAPS:
            out.append(f"-- {a} - {b}")
            for metric in METRICS + ["gc_dist_loss"]:
                result = paired_gap(data, a, b, metric, rows_for(metric, keep, positive, within), taus, traj, rng)
                gaps[(tier, a, b, metric)] = result
                out.append(f"   {metric:<16}{fmt_gap(*result)}")
        for ref in ("V1", "V2"):
            result = paired_gap(data, "V4_photo", ref, "uc_action_loss", keep, taus, traj, rng)
            gaps[(tier, "V4_photo", ref, "uc_action_loss")] = result

    # Per word, headline: V4_word - V4_p with trajectory CIs.
    headline = tiers["headline"]
    words = data["V4_word"]["word"]
    per_word, excluded = [], []
    for word in sorted(set(words[headline]) - SKIP_WORDS):
        rows = headline & (words == word)
        if rows.sum() < MIN_WORD_CASES:
            excluded.append(f"{word} ({int(rows.sum())})")
            continue
        diff = data["V4_word"]["gc_action_loss"][rows] - data["V4_p"]["gc_action_loss"][rows]
        ci = boot_ci(diff, traj[rows], rng)
        per_word.append((word, int(rows.sum()),
                         *(float(np.mean(data[c]["gc_action_loss"][rows])) for c in ("V1", "V2", "V2p", "V3", "V4_photo", "V4_p", "V4_word")),
                         float(diff.mean()), *ci))
    per_word.sort(key=lambda r: -r[9])
    out += ["", f"== per word (headline, n >= {MIN_WORD_CASES}), gc_action_loss, by V4_word - V4_p ==",
            f"{'word':<11}{'n':>6}" + "".join(f"{c:>9}" for c in ("V1", "V2", "V2p", "V3", "V4ph", "V4_p", "V4_w"))
            + f"{'gap':>9}  traj CI"]
    for word, n, *vals, gap, lo, hi in per_word:
        star = "*" if lo > 0 or hi < 0 else " "
        out.append(f"{word:<11}{n:>6}" + "".join(f"{v:>9.3f}" for v in vals) + f"{gap:>+8.3f}{star} [{lo:+.3f}, {hi:+.3f}]")
    out.append(f"excluded (< {MIN_WORD_CASES} cases): " + (", ".join(excluded) or "none"))

    # Verify, headline tier, trajectory CIs.
    def g(a, b, m):
        return gaps[("headline", a, b, m)]
    hl = tiers["headline"]
    dist_loss = float(np.nanmean(data["V4_photo"]["gc_dist_loss"][hl]))
    dist_corr = corr[("headline", "V4_photo")]
    v1_gap, _, _, v1_ci = g("V4_photo", "V1", "gc_action_loss")
    v2_gap, _, _, v2_ci = g("V4_photo", "V2", "gc_action_loss")
    w_gap, _, _, w_ci = g("V4_word", "V4_p", "gc_action_loss")
    uc = {ref: g("V4_photo", ref, "uc_action_loss")[0] for ref in ("V1", "V2")}
    contains0 = lambda ci: ci[0] <= 0 <= ci[1]
    if w_gap > 0.10 and w_ci[0] > 0:
        w_grade = "FAIL"
    elif contains0(w_ci) or w_gap <= 0.05:
        w_grade = "PASS"
    else:
        w_grade = "BORDERLINE (neither pass nor fail)"
    if contains0(v1_ci):
        v1_grade = "STRONG PASS"
    elif v1_gap <= 0.20:
        v1_grade = "PASS"
    elif v1_gap >= 0.35:
        v1_grade = "FAIL"
    else:
        v1_grade = "BORDERLINE (neither pass nor fail)"
    wv = lambda m, rows: float(np.nanmean(data["V4_word"][m][rows]))
    verify = [
        ("1 V4_photo gc_dist_loss", f"{dist_loss:.3f}",
         grade(dist_loss, lambda v: v <= 26, lambda v: v <= 23, lambda v: v > 30)),
        ("2 V4_photo dist-head corr", f"{dist_corr:.3f}",
         grade(dist_corr, lambda v: v >= 0.4, lambda v: v >= 0.6, lambda v: v < 0.2)),
        ("3 V4_photo - V1 gc_action_loss", f"{v1_gap:+.3f} traj CI [{v1_ci[0]:+.3f}, {v1_ci[1]:+.3f}]", v1_grade),
        ("4 V4_photo - V2 gc_action_loss", f"{v2_gap:+.3f} traj CI [{v2_ci[0]:+.3f}, {v2_ci[1]:+.3f}]",
         "PASS" if v2_ci[1] < 0 else "FAIL"),
        ("5 V4_word - V4_p gc_action_loss", f"{w_gap:+.3f} traj CI [{w_ci[0]:+.3f}, {w_ci[1]:+.3f}]", w_grade),
        ("6 uc_action_loss vs V2 / V1", f"{uc['V2']:+.3f} / {uc['V1']:+.3f}",
         "PASS" if all(abs(v) <= 0.05 for v in uc.values()) else "FAIL"),
        ("7 V4_word (report)",
         f"dist_loss {wv('gc_dist_loss', hl):.3f}, corr {corr[('headline', 'V4_word')]:.3f}, "
         f"progress {wv('progress', hl & positive):.3f}, heading {wv('heading_error', hl & positive):.2f} deg, "
         f"success@tau2 {float(np.mean(data['V4_word']['goal_step_dist'][hl & within] < taus['success@tau2'])):.3f}",
         "reported"),
    ]
    out += ["", "== Verify (plan Phase 6; headline / object-word tier; trajectory CIs) =="]
    out += [f"{name:<34} {value:<44} {result}" for name, value, result in verify]

    text = "\n".join(out)
    print(text)
    with open(paths["gap_analysis_v4.txt"], "w") as f:
        f.write(text + "\n")
    with open(paths["per_word_v4.csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "n", "V1", "V2", "V2p", "V3", "V4_photo", "V4_p", "V4_word",
                         "gap_V4word_minus_V4p", "traj_ci_lo", "traj_ci_hi"])
        writer.writerows(per_word)


if __name__ == "__main__":
    main()
