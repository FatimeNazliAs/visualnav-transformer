"""Phase 4: summary tables from eval_clip_harness.py's results.csv.

Tiers (each case counted once per config, paired across configs)
    headline        non-negative, object word, not elevator (Phase 3 DECISIONS.md)
    full_positives  every non-negative case
    all             all 26,648 cases (matches ctx03_s0.npz)
A case skipped in V2p or V3 (no prototype / unknown word) is dropped from all four configs
of every tier, so every config's mean is over the same cases.

Metrics
    gc_action_loss, uc_action_loss, cosine_sim   over every case of the tier
    progress, heading_error                      over the tier's non-negative cases (goal_pos
                                                 is meaningless for a negative goal)
    success@tau                                  within_horizon cases only, on goal_step_dist
    gc_dist_loss                                 its own row: excluded from the primary
                                                 comparison (architectural -- the CLIP goal
                                                 token has no current+goal fusion)
tau: chosen once, from V1's within-horizon goal_step_dist on full_positives. tau1 (headline)
is V1's 25th percentile, so V1 succeeds ~25% of the time; tau2 is V1's median. Both are
rounded to 0.05 units (1 unit = 0.12 m).

Per-word (headline tier, words with >= 20 cases): gc_action_loss for V2, V2p, V3 and the
gap V3 - V2p, sorted by gap descending.

Outputs (next to results.csv, never overwritten):
    summary_table.csv       tier, config, metric, mean, std, n
    per_word_breakdown.csv  word, n, V2, V2p, V3 gc_action_loss, gap_V3_minus_V2p
    summary.meta.json       tau1/tau2, per-tier case counts, paired drops

Run inside the container, from /app/visualnav-transformer/train:

    python ablation/summarize_clip_eval.py
"""
import argparse
import csv
import json
import os

import numpy as np

DEFAULT_RESULTS = "/outputs/nomad_clip_eval/results.csv"
CONFIGS = ["V1", "V2", "V2p", "V3"]
PRIMARY = ["gc_action_loss", "uc_action_loss", "cosine_sim"]
HORIZON = ["progress", "heading_error"]
MIN_WORD_CASES = 20
FOOTNOTES = [
    "V2p word assignment: leave-one-goal-trajectory-out mean of centred CLIP image "
    "embeddings over LLaVA-labelled test goal frames (same labeller as V3). A CLIP-assigned "
    "training-frame variant agreed with LLaVA on 29.7% of test goals (7% for 'door') and "
    "was not used.",
    "uc_action_loss: same checkpoint, goal masked, identical by design for V2/V2p/V3.",
    "gc_dist_loss: excluded from primary comparison -- architectural (no current+goal fusion).",
    "door is 68.7% of the headline tier and ~44% strictly correct (Phase 3 spot-check).",
]


def load(path):
    """results.csv -> {config: {column: np.array}}, rows in sample_idx order per config."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_config = {}
    for name in CONFIGS:
        mine = [r for r in rows if r["config"] == name]
        mine.sort(key=lambda r: int(r["sample_idx"]))
        cols = {}
        for key in mine[0]:
            values = [r[key] for r in mine]
            try:
                cols[key] = np.array([float(v) if v != "" else np.nan for v in values])
            except ValueError:
                cols[key] = np.array(values)
        by_config[name] = cols
    n = {name: len(c["sample_idx"]) for name, c in by_config.items()}
    assert len(set(n.values())) == 1, f"case counts differ across configs: {n}"
    for name in CONFIGS[1:]:
        assert np.array_equal(by_config[name]["sample_idx"], by_config["V1"]["sample_idx"])
    return by_config


def rounded_quantile(values, q, step=0.05):
    return round(float(np.round(np.quantile(values, q) / step) * step), 2)


def stats(values):
    values = values[~np.isnan(values)]
    return float(values.mean()), float(values.std(ddof=1)), int(len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", help="default: next to --results")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.dirname(args.results)
    paths = {name: os.path.join(out_dir, name) for name in
             ("summary_table.csv", "per_word_breakdown.csv", "summary.meta.json")}
    for path in paths.values():
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; refusing to overwrite")

    data = load(args.results)
    v1 = data["V1"]
    dropped = (data["V2p"]["skipped"] == 1) | (data["V3"]["skipped"] == 1)
    positive = v1["goal_is_negative"] == 0
    within = v1["within_horizon"] == 1
    tiers = {
        "headline": (v1["is_headline"] == 1) & ~dropped,
        "full_positives": positive & ~dropped,
        "all": ~dropped,
    }

    tau_basis = v1["goal_step_dist"][tiers["full_positives"] & within]
    tau1, tau2 = rounded_quantile(tau_basis, 0.25), rounded_quantile(tau_basis, 0.5)

    table = []
    for tier, keep in tiers.items():
        for name in CONFIGS:
            cols = data[name]
            for metric in PRIMARY:
                table.append((tier, name, metric, *stats(cols[metric][keep])))
            for metric in HORIZON:
                table.append((tier, name, metric, *stats(cols[metric][keep & positive])))
            for label, tau in (("success@tau1", tau1), ("success@tau2", tau2)):
                hits = (cols["goal_step_dist"][keep & within] < tau).astype(float)
                table.append((tier, name, label, *stats(hits)))
            table.append((tier, name, "gc_dist_loss [excluded]", *stats(cols["gc_dist_loss"][keep])))

    words = data["V3"]["word"]
    headline = tiers["headline"]
    per_word = []
    for word in sorted(set(words[headline])):
        rows = headline & (words == word)
        if rows.sum() < MIN_WORD_CASES:
            continue
        losses = {name: float(np.nanmean(data[name]["gc_action_loss"][rows])) for name in CONFIGS}
        per_word.append((word, int(rows.sum()), losses["V1"], losses["V2"], losses["V2p"],
                         losses["V3"], losses["V3"] - losses["V2p"]))
    per_word.sort(key=lambda r: -r[-1])

    with open(paths["summary_table.csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tier", "config", "metric", "mean", "std", "n"])
        writer.writerows(table)
    with open(paths["per_word_breakdown.csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "n", "V1", "V2", "V2p", "V3", "gap_V3_minus_V2p"])
        writer.writerows(per_word)
    meta = {
        "results": os.path.abspath(args.results),
        "tau1": tau1, "tau2": tau2,
        "tau_basis": "V1 goal_step_dist, within_horizon, full_positives: q25 / q50, "
                     "rounded to 0.05 units (1 unit = 0.12 m)",
        "n_cases": {tier: int(keep.sum()) for tier, keep in tiers.items()},
        "n_within_horizon": {tier: int((keep & within).sum()) for tier, keep in tiers.items()},
        "paired_drops": int(dropped.sum()),
        "footnotes": FOOTNOTES,
    }
    with open(paths["summary.meta.json"], "w") as f:
        json.dump(meta, f, indent=2)

    metrics = PRIMARY + HORIZON + ["success@tau1", "success@tau2"]
    lookup = {(t, c, m): (mean, std, n) for t, c, m, mean, std, n in table}
    for tier in tiers:
        print(f"\n== {tier}: {meta['n_cases'][tier]:,} cases, "
              f"{meta['n_within_horizon'][tier]:,} within horizon ==")
        print(f"{'':<5}" + "".join(f"{m:>20}" for m in metrics))
        for name in CONFIGS:
            print(f"{name:<5}" + "".join(
                f"{lookup[(tier, name, m)][0]:>11.3f} ±{lookup[(tier, name, m)][1]:>7.3f}"
                for m in metrics))
        print(f"{'dist_loss (excluded from primary comparison -- architectural, no current+goal fusion)':}")
        print(f"{'':<5}" + "".join(
            f"{name}: {lookup[(tier, name, 'gc_dist_loss [excluded]')][0]:.3f}   " for name in CONFIGS))
    print(f"\ntau1 = {tau1} units ({tau1 * 0.12:.3f} m), tau2 = {tau2} units ({tau2 * 0.12:.3f} m)")
    print(f"\n== per word (headline, n >= {MIN_WORD_CASES}), gc_action_loss, by gap V3 - V2p ==")
    print(f"{'word':<11}{'n':>6}{'V1':>9}{'V2':>9}{'V2p':>9}{'V3':>9}{'gap':>9}")
    for word, n, *losses in per_word:
        print(f"{word:<11}{n:>6}" + "".join(f"{v:>9.3f}" for v in losses))
    print("\nNotes:\n" + "\n".join(f"- {note}" for note in FOOTNOTES))


if __name__ == "__main__":
    main()
