"""Phase 6b: V5 (V4 fusion + text mix-in) on the Phase 4 test cases, next to V1-V4.

The three V5 configs, one checkpoint (nomad_clip_v7.yaml, clip_text_mix_prob 0.3):
    V5_photo  goal_vec = (prep(img(current), mu_img), prep(img(goal), mu_img))
    V5_p      goal = V2p's leave-one-goal-trajectory-out word prototype
    V5_word   goal = prep(encode_text(template(word)), mu_txt), Phase 3 (LLaVA) words
exactly as eval_v4.py scores V4. goal_mask = 0 for gc_*, 1 for uc_* (model_output).

Stored scores are reused, never re-scored: V1-V3 rows come from the Phase 4
summary_table.csv via Phase 6's summary_table_v4.csv (which also holds V4's rows and every
config's dist_head_corr), copied verbatim. Two checks run first and stop the script on
any difference, because the dataset changed in Phase 6b (clip_text_mix_prob, default 0.0):
    V1 re-score        first --check-batches batches through eval_paired.score_checkpoint,
                       byte-identical to ctx03_s0.npz
    V4_photo re-score  same cases through the Phase 4 scorer: every metric identical to
                       results_v4.csv at its stored precision (%.6g), and the predicted
                       distance byte-identical to dist_preds.npz
V5 goes through the distance-only pass too; its squared error must reproduce the scorer's
gc_dist_loss (as in eval_v4.py).

Outputs (a new folder, never overwritten):
    rescore_check.json                    the two checks above
    results_v5.csv, results_v5.meta.json  per-case V5 rows, the harness's columns
    dist_preds_v5.npz                     predicted distance per V5 config, plus the labels
    summary_table_v5.csv                  summary_table_v4.csv verbatim + V5 rows + V5 corr

Run inside the container, from /app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 python ablation/eval_v5.py --checkpoint <run>/ema_29.pth
"""
# Must be set before torch creates its cuBLAS handle; see eval_paired.py.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json

import numpy as np
import torch

from analyze_v4 import load_v4
from eval_clip_harness import (
    BATCH_SIZE,
    DEFAULT_LABELS,
    SEED,
    V1_CHECKPOINT,
    V1_CONFIG,
    V1_REFERENCE_SCORES,
    build_loo_prototypes,
    build_text_goals,
    check_same_cases,
    read_labels,
    score_model,
)
from eval_paired import (
    PER_SAMPLE_ARRAYS,
    build_test_loader,
    checkpoint_digest,
    enforce_determinism,
    load_arm_config,
    score_checkpoint,
)
from eval_v4 import (
    DIST_REPRODUCTION_TOLERANCE,
    PHASE4_DIR,
    V4_CONFIG,
    distance_predictions,
    pearson,
    write_v4_results,
)
from label_test_goals import git_head
from summarize_clip_eval import HORIZON, PRIMARY, load, stats

V5_CONFIG = "config/nomad_clip_v7.yaml"
V4_CHECKPOINT = "/outputs/nomad_clip_v6/clip_vitb32_fusion_2026_09_28_12_53_39/ema_29.pth"
V4_DIR = "/outputs/nomad_clip_v6/eval"
DEFAULT_OUT_DIR = "/outputs/nomad_clip_v7/eval"
V5_CONFIGS = ["V5_photo", "V5_p", "V5_word"]
V5_WORD_CONFIGS = {"V5_p", "V5_word"}


def rescore_v1(device, batches):
    """V1 on the first batches, compared byte for byte with the stored Phase 4 scores."""
    config = load_arm_config(V1_CONFIG)
    dataset, loader = build_test_loader(config, BATCH_SIZE)
    try:
        scores = score_checkpoint(V1_CHECKPOINT, config, loader, device, SEED,
                                  max_batches=batches, desc="V1 re-score")
    finally:
        dataset.close()
    stored = np.load(V1_REFERENCE_SCORES)
    n = len(scores["gc_action_loss"])
    mismatched = [k for k in PER_SAMPLE_ARRAYS
                  if np.asarray(scores[k]).tobytes() != stored[k][:n].tobytes()]
    return {"n": n, "arrays": PER_SAMPLE_ARRAYS, "mismatched": mismatched, "pass": not mismatched}


def rescore_v4_photo(device, batches):
    """V4_photo on the same cases, against results_v4.csv and dist_preds.npz."""
    config = load_arm_config(V4_CONFIG)
    _, scores = score_model(config, V4_CHECKPOINT, {"V4_photo": "loader"}, device, batches)
    scores = scores["V4_photo"]
    n = len(scores["gc_action_loss"])
    phase4 = load(os.path.join(PHASE4_DIR, "results.csv"))
    stored = load_v4(os.path.join(V4_DIR, "results_v4.csv"), phase4["V1"])["V4_photo"]
    mismatched = []
    for metric, values in scores.items():
        if metric == "within_horizon":
            same = np.array_equal(np.asarray(values, dtype=int), stored[metric][:n].astype(int))
        else:
            # results_v4.csv holds f"{v:.6g}"; compare at exactly that precision.
            mine = [f"{float(v):.6g}" for v in values]
            theirs = [f"{float(v):.6g}" for v in stored[metric][:n]]
            same = mine == theirs
        if not same:
            mismatched.append(metric)
    preds = distance_predictions(config, V4_CHECKPOINT, {"V4_photo": "loader"}, device, batches)["V4_photo"]
    dist_same = preds.tobytes() == np.load(os.path.join(V4_DIR, "dist_preds.npz"))["V4_photo"][:n].tobytes()
    return {"n": n, "metrics_mismatched": mismatched, "dist_pred_byte_identical": bool(dist_same),
            "pass": not mismatched and dist_same}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="the V5 run's ema_29.pth")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--check-batches", type=int, default=16, help="re-score checks: 16 x 32 = 512 cases")
    parser.add_argument("--max-batches", type=int, help="score V5 on the first N batches only (pilot)")
    args = parser.parse_args()
    paths = {name: os.path.join(args.out_dir, name) for name in (
        "rescore_check.json", "results_v5.csv", "results_v5.meta.json",
        "dist_preds_v5.npz", "summary_table_v5.csv")}
    for path in paths.values():
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; refusing to overwrite")

    enforce_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 0. The dataset change must be inert for every stored score this reuses.
    checks = {"V1": rescore_v1(device, args.check_batches),
              "V4_photo": rescore_v4_photo(device, args.check_batches)}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(paths["rescore_check.json"], "w") as f:
        json.dump(checks, f, indent=2)
    print(json.dumps(checks, indent=2))
    if not all(c["pass"] for c in checks.values()):
        raise SystemExit("STOP: a re-score differs from the stored scores; V5 not scored")

    labels = read_labels(args.labels)
    config = load_arm_config(V5_CONFIG)
    assert config["clip_fusion"] != "none"
    prototypes, proto_skipped = build_loo_prototypes(labels, config)
    text_goals, text_skipped = build_text_goals(labels, config, device)

    # 1. V5 action + horizon metrics through the Phase 4 scorer.
    sources = {"V5_photo": "loader", "V5_p": prototypes, "V5_word": text_goals}
    case, scores = score_model(config, args.checkpoint, sources, device, args.max_batches)
    check_same_cases(case, case, labels, config["distance"]["max_dist_cat"])

    # 2. Distance predictions for V5; V1-V4 are in summary_table_v4.csv already.
    preds = distance_predictions(config, args.checkpoint, sources, device, args.max_batches)
    distance = case["distance"].astype(np.float64)
    n = len(distance)
    labels, proto_skipped, text_skipped = labels[:n], proto_skipped[:n], text_skipped[:n]
    reproduction = {}
    for name in V5_CONFIGS:
        stored = np.asarray(scores[name]["gc_dist_loss"], dtype=np.float64)
        ok = ~np.isnan(stored)
        rel = np.abs((preds[name] - distance) ** 2 - stored)[ok] / np.maximum(1.0, np.abs(stored[ok]))
        reproduction[name] = {"max_rel_diff": float(rel.max()),
                              "pass": bool(rel.max() <= DIST_REPRODUCTION_TOLERANCE)}
    assert all(r["pass"] for r in reproduction.values()), f"distance pass mismatch: {reproduction}"

    # 3. Summary: the Phase 4 tiers, drops and tau (as eval_v4.py), V5 rows, V5 corr rows.
    phase4 = {c: {k: v[:n] for k, v in cols.items()}
              for c, cols in load(os.path.join(PHASE4_DIR, "results.csv")).items()}
    v1 = phase4["V1"]
    dropped = (phase4["V2p"]["skipped"] == 1) | (phase4["V3"]["skipped"] == 1)
    assert np.array_equal(dropped, proto_skipped | text_skipped), "V5 skips differ from Phase 4's"
    positive = v1["goal_is_negative"] == 0
    within = v1["within_horizon"] == 1
    tiers = {
        "headline": (v1["is_headline"] == 1) & ~dropped,
        "full_positives": positive & ~dropped,
        "all": ~dropped,
    }
    with open(os.path.join(PHASE4_DIR, "summary.meta.json")) as f:
        phase4_meta = json.load(f)
    tau1, tau2 = phase4_meta["tau1"], phase4_meta["tau2"]
    with open(os.path.join(V4_DIR, "summary_table_v4.csv")) as f:
        old_rows = list(csv.reader(f))

    new_rows = []
    for tier, keep in tiers.items():
        for name in V5_CONFIGS:
            cols = {m: np.asarray(v, dtype=np.float64) for m, v in scores[name].items()}
            for metric in PRIMARY:
                new_rows.append((tier, name, metric, *stats(cols[metric][keep])))
            for metric in HORIZON:
                new_rows.append((tier, name, metric, *stats(cols[metric][keep & positive])))
            for label, tau in (("success@tau1", tau1), ("success@tau2", tau2)):
                hits = (cols["goal_step_dist"][keep & within] < tau).astype(float)
                new_rows.append((tier, name, label, *stats(hits)))
            new_rows.append((tier, name, "gc_dist_loss", *stats(cols["gc_dist_loss"][keep])))
            new_rows.append((tier, name, "dist_head_corr",
                             pearson(preds[name][keep], distance[keep]), float("nan"), int(keep.sum())))

    np.savez(paths["dist_preds_v5.npz"], distance=distance, **preds)
    skipped = {"V5_photo": np.zeros(n, dtype=bool), "V5_p": proto_skipped, "V5_word": text_skipped}
    write_v4_results(paths["results_v5.csv"], labels, case, scores, skipped, V5_CONFIGS, V5_WORD_CONFIGS)
    with open(paths["summary_table_v5.csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(old_rows)  # Phase 4 + V4 rows, header included, unchanged
        writer.writerows(new_rows)
    meta = {
        "v5": {"config": V5_CONFIG, "checkpoint": args.checkpoint,
               "sha256": checkpoint_digest(args.checkpoint)},
        "old_rows_from": os.path.join(V4_DIR, "summary_table_v4.csv"),
        "rescore_check": checks,
        "labels": args.labels, "seed": SEED, "batch_size": BATCH_SIZE, "max_batches": args.max_batches,
        "goal_mask": "gc = goal visible (input_goal_mask=0); uc = masked (1)",
        "tau1": tau1, "tau2": tau2,
        "n_cases": {tier: int(keep.sum()) for tier, keep in tiers.items()},
        "skipped": {c: int(s.sum()) for c, s in skipped.items()},
        "dist_reproduction": reproduction,
        "uc_action_loss_identical_V5": all(
            np.array_equal(scores["V5_photo"]["uc_action_loss"], scores[c]["uc_action_loss"])
            for c in ("V5_p", "V5_word")),
        "git_head": git_head(),
    }
    with open(paths["results_v5.meta.json"], "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in ("n_cases", "dist_reproduction", "uc_action_loss_identical_V5")},
                     indent=2))
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
