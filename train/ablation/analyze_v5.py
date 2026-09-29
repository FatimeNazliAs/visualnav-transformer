"""Phase 6b: paired gaps for V5 against V1-V4, the summary table and the Verify checks.

Joins the Phase 4 results.csv (V1, V2, V2p, V3), Phase 6's results_v4.csv (V4_*) and
eval_v5.py's results_v5.csv (V5_*) on sample_idx, all read only, then reuses
analyze_results.py's tiers, per-metric case subsets and bootstrap (2000 resamples; case CI
and observation-trajectory CI, '*' when the trajectory CI excludes 0). Distance-head
correlations come from eval_v5.py's summary_table_v5.csv (V1-V4 rows copied from Phase 6).

Gaps: V5_word - V3, V5_word - V5_p, V5_photo - V4_photo, V5_word - V4_word,
V5_photo - V1, V5_photo - V2. uc_action_loss: V5_photo - V2, V5_photo - V4_photo.
Per word (headline, >= 20 cases): V5_p, V5_word and V5_word - V5_p.

Verify (Phase 6b; object-word tier = headline; trajectory CIs):
    V5_word - V3 gc_action        pass CI < 0  -> V5 replaces V3 in the Phase 7 full run
    V5_word - V5_p gc_action      pass CI contains 0 or gap <= +0.05
    V5_photo - V4_photo gc_action pass gap <= +0.10
    V5_word dist-head corr        reported (target >= 0.3), next to V5_p's
    uc_action_loss                reported vs V2 and V4_photo

Outputs (in eval_v5.py's folder, never overwritten): gap_analysis_v5.txt, per_word_v5.csv,
summary_v5.csv (one row per tier x config, the report's columns).

Run inside the container, from /app/visualnav-transformer/train:

    python ablation/analyze_v5.py
"""
import argparse
import csv
import json
import os

import numpy as np

from analyze_results import METRICS, N_BOOT, SEED, SKIP_WORDS, boot_ci, paired_gap, rows_for, tier_masks
from analyze_v4 import V4_CONFIGS, fmt_gap, load_v4
from eval_v5 import DEFAULT_OUT_DIR, V4_DIR, V5_CONFIGS
from summarize_clip_eval import MIN_WORD_CASES, load

PHASE4_DIR = "/outputs/nomad_clip_eval"
OLD_CONFIGS = ["V1", "V2", "V2p", "V3"]
ALL_CONFIGS = OLD_CONFIGS + V4_CONFIGS + V5_CONFIGS
GAPS = [("V5_word", "V3"), ("V5_word", "V5_p"), ("V5_photo", "V4_photo"), ("V5_word", "V4_word"),
        ("V5_photo", "V1"), ("V5_photo", "V2")]
TIERS = ["headline", "all"]
# The report's columns, in order; gc_dist_loss for V1-V3 is the Phase 4 "[excluded]" row.
SUMMARY_METRICS = ["gc_action_loss", "uc_action_loss", "gc_dist_loss", "dist_head_corr",
                   "progress", "heading_error", "success@tau2"]


def load_all(v5_dir):
    data = load(os.path.join(PHASE4_DIR, "results.csv"))
    data.update(load_v4(os.path.join(V4_DIR, "results_v4.csv"), data["V1"]))
    data.update(load_v4(os.path.join(v5_dir, "results_v5.csv"), data["V1"], V5_CONFIGS))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--v5-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    paths = {n: os.path.join(args.v5_dir, n) for n in ("gap_analysis_v5.txt", "per_word_v5.csv", "summary_v5.csv")}
    for path in paths.values():
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; refusing to overwrite")

    with open(os.path.join(PHASE4_DIR, "summary.meta.json")) as f:
        meta = json.load(f)
    taus = {"success@tau1": meta["tau1"], "success@tau2": meta["tau2"]}
    data = load_all(args.v5_dir)
    tiers, positive, within = tier_masks(data)
    traj = data["V1"]["traj_id"]
    with open(os.path.join(args.v5_dir, "summary_table_v5.csv")) as f:
        corr = {(r["tier"], r["config"]): float(r["mean"]) for r in csv.DictReader(f)
                if r["metric"] == "dist_head_corr"}
    rng = np.random.default_rng(SEED)
    out = [f"Phase 6b V5 gap analysis; bootstrap {N_BOOT}, seed {SEED}; '*' = trajectory CI excludes 0",
           f"tau1 = {taus['success@tau1']}, tau2 = {taus['success@tau2']} units (1 unit = 0.12 m)"]

    summary, gaps = [], {}
    for tier in TIERS:
        keep = tiers[tier]
        out += ["", f"== {tier}: {int(keep.sum()):,} cases =="]
        out.append(f"{'metric':<16}" + "".join(f"{c:>10}" for c in ALL_CONFIGS))
        table = {c: {} for c in ALL_CONFIGS}
        for metric in SUMMARY_METRICS:
            for c in ALL_CONFIGS:
                if metric == "dist_head_corr":
                    table[c][metric] = corr[(tier, c)]
                else:
                    rows = rows_for(metric, keep, positive, within)
                    v = (data[c]["goal_step_dist"][rows] < taus[metric]).astype(float) if metric in taus \
                        else data[c][metric][rows]
                    table[c][metric] = float(np.nanmean(v))
            out.append(f"{metric:<16}" + "".join(f"{table[c][metric]:>10.3f}" for c in ALL_CONFIGS))
        summary += [[tier, c] + [table[c][m] for m in SUMMARY_METRICS] for c in ALL_CONFIGS]
        for a, b in GAPS:
            out.append(f"-- {a} - {b}")
            for metric in METRICS + ["gc_dist_loss"]:
                result = paired_gap(data, a, b, metric, rows_for(metric, keep, positive, within), taus, traj, rng)
                gaps[(tier, a, b, metric)] = result
                out.append(f"   {metric:<16}{fmt_gap(*result)}")
        out.append("-- uc_action_loss, V5_photo - reference (V5_p / V5_word share V5_photo's uc)")
        for ref in ("V1", "V2", "V4_photo"):
            result = paired_gap(data, "V5_photo", ref, "uc_action_loss", keep, taus, traj, rng)
            gaps[(tier, "V5_photo", ref, "uc_action_loss")] = result
            out.append(f"   vs {ref:<13}{fmt_gap(*result)}")

    # Per word, headline: V5_word - V5_p with trajectory CIs.
    headline = tiers["headline"]
    words = data["V5_word"]["word"]
    per_word, excluded = [], []
    shown = ("V3", "V4_p", "V4_word", "V5_p", "V5_word")
    for word in sorted(set(words[headline]) - SKIP_WORDS):
        rows = headline & (words == word)
        if rows.sum() < MIN_WORD_CASES:
            excluded.append(f"{word} ({int(rows.sum())})")
            continue
        diff = data["V5_word"]["gc_action_loss"][rows] - data["V5_p"]["gc_action_loss"][rows]
        ci = boot_ci(diff, traj[rows], rng)
        per_word.append((word, int(rows.sum()),
                         *(float(np.mean(data[c]["gc_action_loss"][rows])) for c in shown),
                         float(diff.mean()), *ci))
    per_word.sort(key=lambda r: -r[1])
    out += ["", f"== per word (headline, n >= {MIN_WORD_CASES}), gc_action_loss, by n ==",
            f"{'word':<11}{'n':>6}" + "".join(f"{c:>9}" for c in shown) + f"{'V5w-V5p':>9}  traj CI"]
    for word, n, *vals, gap, lo, hi in per_word:
        star = "*" if lo > 0 or hi < 0 else " "
        out.append(f"{word:<11}{n:>6}" + "".join(f"{v:>9.3f}" for v in vals) + f"{gap:>+8.3f}{star} [{lo:+.3f}, {hi:+.3f}]")
    out.append(f"excluded (< {MIN_WORD_CASES} cases): " + (", ".join(excluded) or "none"))

    # Verify, headline tier, trajectory CIs.
    def g(a, b, m="gc_action_loss"):
        return gaps[("headline", a, b, m)]
    contains0 = lambda ci: ci[0] <= 0 <= ci[1]
    w3, _, _, w3_ci = g("V5_word", "V3")
    wp, _, _, wp_ci = g("V5_word", "V5_p")
    ph, _, _, ph_ci = g("V5_photo", "V4_photo")
    uc = {ref: g("V5_photo", ref, "uc_action_loss")[0] for ref in ("V2", "V4_photo")}
    verify = [
        ("V5_word - V3 gc_action_loss", f"{w3:+.3f} traj CI [{w3_ci[0]:+.3f}, {w3_ci[1]:+.3f}]",
         "PASS -> V5 replaces V3 in the Phase 7 full run" if w3_ci[1] < 0 else "FAIL"),
        ("V5_word - V5_p gc_action_loss", f"{wp:+.3f} traj CI [{wp_ci[0]:+.3f}, {wp_ci[1]:+.3f}]",
         "PASS" if contains0(wp_ci) or wp <= 0.05 else "FAIL"),
        ("V5_photo - V4_photo gc_action_loss", f"{ph:+.3f} traj CI [{ph_ci[0]:+.3f}, {ph_ci[1]:+.3f}]",
         "PASS" if ph <= 0.10 else "FAIL"),
        ("V5_word dist-head corr", f"{corr[('headline', 'V5_word')]:.3f} (V5_p {corr[('headline', 'V5_p')]:.3f}, "
         f"V4_word {corr[('headline', 'V4_word')]:.3f})",
         "reported; target >= 0.3 " + ("met" if corr[("headline", "V5_word")] >= 0.3 else "not met")),
        ("uc_action_loss V5 - V2 / V5 - V4_photo", f"{uc['V2']:+.3f} / {uc['V4_photo']:+.3f}", "reported"),
    ]
    out += ["", "== Verify (Phase 6b; headline / object-word tier; trajectory CIs) =="]
    out += [f"{name:<38} {value:<52} {result}" for name, value, result in verify]

    text = "\n".join(out)
    print(text)
    with open(paths["gap_analysis_v5.txt"], "w") as f:
        f.write(text + "\n")
    with open(paths["per_word_v5.csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "n", *shown, "gap_V5word_minus_V5p", "traj_ci_lo", "traj_ci_hi"])
        writer.writerows(per_word)
    with open(paths["summary_v5.csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "config"] + SUMMARY_METRICS)
        writer.writerows(summary)


if __name__ == "__main__":
    main()
