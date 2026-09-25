"""Phase 5: paired gap analysis over eval_clip_harness.py's results.csv.

Reuses summarize_clip_eval.py's loader and tiers (same paired drops, same case alignment) and
reports, for the headline (primary) and all (secondary) tiers:
    a. per-config mean / std of every metric
    b. paired gaps V2 - V1, V2p - V2, V3 - V2p: absolute, relative (% of the second config)
       and a 95% bootstrap CI two ways -- resampling cases, and resampling observation
       trajectories. Consecutive frames of one trajectory are correlated, so the case CI is
       optimistic; '*' marks gaps whose trajectory CI excludes 0.
    c. per word (headline and full_positives, words with >= 20 cases): V1, V2, V2p, V3
       gc_action_loss and the gaps V2p - V2 and V3 - V2p with trajectory CIs, sorted by
       V3 - V2p descending. Words under 20 cases are listed as excluded.
    d. gc_dist_loss per config and its gaps, labelled excluded from the primary comparison.

Case subsets per metric follow summarize_clip_eval.py: action metrics over every case of the
tier, progress / heading_error over its non-negative cases, success@tau over within-horizon
cases (tau1 / tau2 read from summary.meta.json).

Outputs (next to results.csv, never overwritten):
    gap_analysis.txt                everything printed
    phase5_per_word_breakdown.csv   tier, word, n, per-config loss, gaps + trajectory CIs

Run inside the container, from /app/visualnav-transformer/train:

    python ablation/analyze_results.py
"""
import argparse
import csv
import json
import os

import numpy as np

from summarize_clip_eval import CONFIGS, DEFAULT_RESULTS, MIN_WORD_CASES, load

TIERS = ["headline", "all"]                # primary, secondary (Phase 5 decision)
WORD_TIERS = ["headline", "full_positives"]  # full_positives holds elevator + scene words
GAPS = [("V2", "V1"), ("V2p", "V2"), ("V3", "V2p")]
METRICS = ["gc_action_loss", "uc_action_loss", "cosine_sim", "progress", "heading_error",
           "success@tau1", "success@tau2"]
SKIP_WORDS = {"", "other"}
N_BOOT = 2000
SEED = 0


def tier_masks(data):
    """Same tiers and paired drops as summarize_clip_eval.py."""
    v1 = data["V1"]
    dropped = (data["V2p"]["skipped"] == 1) | (data["V3"]["skipped"] == 1)
    positive = v1["goal_is_negative"] == 0
    tiers = {
        "headline": (v1["is_headline"] == 1) & ~dropped,
        "full_positives": positive & ~dropped,
        "all": ~dropped,
    }
    return tiers, positive, v1["within_horizon"] == 1


def values(data, name, metric, taus):
    """Per-case values of one metric for one config (success@tau as 0/1 hits)."""
    cols = data[name]
    if metric in taus:
        return (cols["goal_step_dist"] < taus[metric]).astype(float)
    return cols[metric]


def rows_for(metric, keep, positive, within):
    if metric in ("progress", "heading_error"):
        return keep & positive
    if metric.startswith("success@"):
        return keep & within
    return keep


def boot_ci(diff, groups, rng):
    """95% percentile CI of mean(diff), resampling groups (cases or trajectories)."""
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=diff)
    counts = np.bincount(inverse).astype(float)
    means = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 200):
        picks = rng.integers(0, len(sums), size=(min(200, N_BOOT - start), len(sums)))
        means[start:start + len(picks)] = sums[picks].sum(1) / counts[picks].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_gap(data, a, b, metric, rows, taus, traj, rng):
    """a - b over rows -> gap, % of b, case CI (lo, hi), trajectory CI (lo, hi)."""
    va, vb = values(data, a, metric, taus)[rows], values(data, b, metric, taus)[rows]
    diff = va - vb
    case_ci = boot_ci(diff, np.arange(len(diff)), rng)
    traj_ci = boot_ci(diff, traj[rows], rng)
    return float(diff.mean()), float(100 * diff.mean() / vb.mean()), case_ci, traj_ci


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", help="default: next to --results")
    args = parser.parse_args()
    src_dir = os.path.dirname(args.results)
    out_dir = args.out_dir or src_dir
    paths = {name: os.path.join(out_dir, name) for name in
             ("gap_analysis.txt", "phase5_per_word_breakdown.csv")}
    for path in paths.values():
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; refusing to overwrite")

    with open(os.path.join(src_dir, "summary.meta.json")) as f:
        meta = json.load(f)
    taus = {"success@tau1": meta["tau1"], "success@tau2": meta["tau2"]}
    data = load(args.results)
    tiers, positive, within = tier_masks(data)
    traj = data["V1"]["traj_id"]
    rng = np.random.default_rng(SEED)
    out = []

    out.append(f"Phase 5 gap analysis -- {os.path.abspath(args.results)}")
    out.append(f"tau1 = {meta['tau1']} units, tau2 = {meta['tau2']} units (1 unit = 0.12 m)")
    out.append(f"Paired bootstrap: {N_BOOT} resamples, seed {SEED}, 95% percentile CI; "
               f"'*' = trajectory CI excludes 0")
    out.append("Primary tier: headline. Secondary: all.")

    # a. per-config mean / std
    for tier in TIERS:
        keep = tiers[tier]
        out.append(f"\n== a. per-config mean ± std -- {tier} ({keep.sum():,} cases, "
                   f"{(keep & within).sum():,} within horizon) ==")
        out.append(f"{'metric':<16}" + "".join(f"{name:>18}" for name in CONFIGS))
        for metric in METRICS:
            rows = rows_for(metric, keep, positive, within)
            cells = []
            for name in CONFIGS:
                v = values(data, name, metric, taus)[rows]
                cells.append(f"{v.mean():>9.3f} ±{v.std(ddof=1):>7.3f}")
            out.append(f"{metric:<16}" + "".join(cells))

    # b. paired gaps
    for tier in TIERS:
        keep = tiers[tier]
        for a, b in GAPS:
            out.append(f"\n== b. paired gap {a} - {b} -- {tier} ==")
            out.append(f"{'metric':<16}{'gap':>9}{'rel %':>9}{'95% CI cases':>22}"
                       f"{'95% CI trajectories':>24}")
            for metric in METRICS:
                rows = rows_for(metric, keep, positive, within)
                g, rel, (cl, ch), (tl, th) = paired_gap(data, a, b, metric, rows, taus, traj, rng)
                sig = "*" if tl > 0 or th < 0 else ""
                out.append(f"{metric:<16}{g:>+9.3f}{rel:>+8.1f}%"
                           f"{f'[{cl:+.3f}, {ch:+.3f}]':>22}{f'[{tl:+.3f}, {th:+.3f}]':>24} {sig}")

    # c. per word
    words = data["V3"]["word"]
    per_word = []
    for tier in WORD_TIERS:
        keep = tiers[tier]
        counts = {w: int((keep & (words == w)).sum()) for w in set(words[keep]) - SKIP_WORDS}
        kept = sorted(w for w, n in counts.items() if n >= MIN_WORD_CASES)
        excluded = sorted((w for w, n in counts.items() if n < MIN_WORD_CASES),
                          key=lambda w: -counts[w])
        rows_out = []
        for word in kept:
            rows = keep & (words == word)
            loss = {name: float(data[name]["gc_action_loss"][rows].mean()) for name in CONFIGS}
            g1, _, _, ci1 = paired_gap(data, "V2p", "V2", "gc_action_loss", rows, taus, traj, rng)
            g2, _, _, ci2 = paired_gap(data, "V3", "V2p", "gc_action_loss", rows, taus, traj, rng)
            rows_out.append((tier, word, counts[word], *(loss[n] for n in CONFIGS),
                             g1, *ci1, g2, *ci2))
        rows_out.sort(key=lambda r: -r[10])
        per_word.extend(rows_out)

        out.append(f"\n== c. per word -- {tier}, n >= {MIN_WORD_CASES}, gc_action_loss, "
                   f"by V3 - V2p descending (CIs: trajectories) ==")
        out.append(f"{'word':<12}{'n':>6}{'V1':>8}{'V2':>8}{'V2p':>8}{'V3':>8}"
                   f"{'V2p-V2':>9}{'CI':>18}{'V3-V2p':>9}{'CI':>18}")
        for r in rows_out:
            s1 = "*" if r[8] > 0 or r[9] < 0 else " "
            s2 = "*" if r[11] > 0 or r[12] < 0 else " "
            out.append(f"{r[1]:<12}{r[2]:>6}" + "".join(f"{v:>8.3f}" for v in r[3:7])
                       + f"{r[7]:>+9.3f}{f'[{r[8]:+.3f}, {r[9]:+.3f}]':>17}{s1}"
                       + f"{r[10]:>+9.3f}{f'[{r[11]:+.3f}, {r[12]:+.3f}]':>17}{s2}")
        out.append("excluded (< {} cases): {}".format(
            MIN_WORD_CASES, ", ".join(f"{w} ({counts[w]})" for w in excluded) or "none"))

    # d. dist_loss
    out.append("\n== d. gc_dist_loss -- EXCLUDED FROM PRIMARY COMPARISON "
               "(architectural: the CLIP goal token has no current+goal fusion; Phase 6) ==")
    for tier in TIERS:
        keep = tiers[tier]
        out.append(f"{tier}: " + "   ".join(
            f"{name} {data[name]['gc_dist_loss'][keep].mean():.3f}" for name in CONFIGS))
        for a, b in GAPS:
            g, rel, _, (tl, th) = paired_gap(data, a, b, "gc_dist_loss", keep, taus, traj, rng)
            out.append(f"  {a} - {b}: {g:+.3f} ({rel:+.1f}%)  CI traj [{tl:+.3f}, {th:+.3f}]")

    with open(paths["gap_analysis.txt"], "w") as f:
        f.write("\n".join(out) + "\n")
    with open(paths["phase5_per_word_breakdown.csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "word", "n", "V1", "V2", "V2p", "V3",
                         "gap_V2p_minus_V2", "ci_lo", "ci_hi",
                         "gap_V3_minus_V2p", "ci_lo", "ci_hi"])
        writer.writerows(per_word)
    print("\n".join(out))
    print(f"\nwrote {paths['gap_analysis.txt']}\nwrote {paths['phase5_per_word_breakdown.csv']}")


if __name__ == "__main__":
    main()
