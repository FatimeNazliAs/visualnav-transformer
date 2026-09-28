"""Phase 6: V4 (current+goal fusion) on the Phase 4 test cases, next to V1 / V2 / V2p / V3.

The three V4 configs, one fused checkpoint (clip_fusion: interaction):
    V4_photo  goal_vec = (prep(img(current), mu_img), prep(img(goal), mu_img))
    V4_p      goal = V2p's leave-one-goal-trajectory-out word prototype
    V4_word   goal = prep(encode_text(template(word)), mu_txt)
The current frame o is always the cached embedding of the last context frame, so no goal
image is needed at test: V4_word sees only the camera frame and the word. V4_word - V4_p
is the Phase 6 replication of the V3 - V2p modality-gap test.

V1-V3 are never re-scored for their action metrics; their rows stay those of the Phase 4
results.csv, copied verbatim into the new summary. What is new for them is a distance-head
correlation: results.csv stores only the squared distance error, whose sign is lost, so a
distance-only pass (vision encoder, goal visible -> dist head; deterministic, no diffusion)
recomputes each config's predicted distance on the same seeded cases. Its squared error
must reproduce results.csv's gc_dist_loss; that checks the pass is the same one.

Outputs (a new folder, never overwritten):
    results_v4.csv, results_v4.meta.json   per-case V4 rows, the harness's columns
    dist_preds.npz                         predicted distance per config, plus the labels
    summary_table_v4.csv                   Phase 4 summary rows verbatim + V4 rows +
                                           dist_head_corr rows for every config

Run inside the container, from /app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 python ablation/eval_v4.py --checkpoint <run>/ema_29.pth
    ... --max-batches 5 --out-dir /outputs/nomad_clip_v6/eval_pilot   # smoke test
"""
# Must be set before torch creates its cuBLAS handle; see eval_paired.py.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json

import numpy as np
import torch
import tqdm

from eval_clip_harness import (
    BATCH_SIZE,
    CLIP_CHECKPOINT,
    CLIP_CONFIG,
    COLUMNS,
    DEFAULT_LABELS,
    SEED,
    V1_CHECKPOINT,
    V1_CONFIG,
    build_loo_prototypes,
    build_text_goals,
    check_same_cases,
    is_headline,
    read_labels,
    score_model,
)
from eval_paired import (
    IMAGENET_TRANSFORM,
    build_model,
    build_test_loader,
    checkpoint_digest,
    enforce_determinism,
    load_arm_config,
)
from label_test_goals import git_head
from summarize_clip_eval import HORIZON, PRIMARY, load, stats

V4_CONFIG = "config/nomad_clip_v6.yaml"
PHASE4_DIR = "/outputs/nomad_clip_eval"
DEFAULT_OUT_DIR = "/outputs/nomad_clip_v6/eval"
V4_CONFIGS = ["V4_photo", "V4_p", "V4_word"]
V4_WORD_CONFIGS = {"V4_p", "V4_word"}
OLD_CONFIGS = ["V1", "V2", "V2p", "V3"]
# The distance-only pass must reproduce the stored squared error to float32 print precision.
DIST_REPRODUCTION_TOLERANCE = 1e-3


@torch.no_grad()
def distance_predictions(config, checkpoint, goal_sources, device, max_batches=None):
    """Predicted distance per config, goal visible, over the aligned test split.

    goal_sources as in eval_clip_harness.score_model. Same loader, seed and model call as
    model_output's goal-visible pass; the distance head samples no noise.
    """
    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    dataset, loader = build_test_loader(config, BATCH_SIZE)
    preds = {name: [] for name in goal_sources}
    first = 0
    try:
        np.random.seed(SEED)
        for batch_index, data in enumerate(tqdm.tqdm(loader, dynamic_ncols=True)):
            if max_batches is not None and batch_index >= max_batches:
                break
            obs_image, goal_image, _, _, _, _, _, goal_vec = data
            batch_obs = torch.cat(
                [IMAGENET_TRANSFORM(obs) for obs in torch.split(obs_image, 3, dim=1)], dim=1
            ).to(device)
            batch_goal = IMAGENET_TRANSFORM(goal_image).to(device)
            size = batch_obs.shape[0]
            visible = torch.zeros(size, dtype=torch.long, device=device)
            for name, source in goal_sources.items():
                if source is None:
                    vec = None
                elif isinstance(source, str):
                    vec = goal_vec.to(device)
                else:
                    vec = source[first:first + size].to(device)
                    if goal_vec.dim() == 3:
                        vec = torch.stack([goal_vec[:, 0].to(device), vec], dim=1)
                cond = model("vision_encoder", obs_img=batch_obs, goal_img=batch_goal,
                             input_goal_mask=visible, goal_vec=vec)
                preds[name].append(model("dist_pred_net", obsgoal_cond=cond).squeeze(-1).cpu().numpy())
            first += size
    finally:
        dataset.close()
    return {name: np.concatenate(v) for name, v in preds.items()}


def write_v4_results(path, labels, case, scores, skipped):
    """results.csv's long format, for the V4 configs only."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(len(case["distance"])):
            r = labels[i]
            base = {
                "sample_idx": i, "traj_id": r["obs_traj_id"], "frame_idx": r["obs_frame_idx"],
                "goal_traj_id": r["goal_traj_id"], "goal_frame_idx": r["goal_frame_idx"],
                "goal_is_negative": r["goal_is_negative"], "distance": int(case["distance"][i]),
                "action_mask": int(case["action_mask"][i]), "is_headline": int(is_headline(r)),
            }
            for name in V4_CONFIGS:
                skip = bool(skipped[name][i])
                row = {**base, "config": name, "skipped": int(skip),
                       "word": r["word"] if name in V4_WORD_CONFIGS else ""}
                for metric, values in scores[name].items():
                    if metric == "within_horizon":
                        row[metric] = int(values[i])
                    else:
                        row[metric] = "" if skip else f"{float(values[i]):.6g}"
                writer.writerow(row)


def pearson(pred, true):
    return float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="the fused run's ema_29.pth")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-batches", type=int, help="score only the first N batches (pilot)")
    args = parser.parse_args()
    paths = {name: os.path.join(args.out_dir, name) for name in (
        "results_v4.csv", "results_v4.meta.json", "dist_preds.npz", "summary_table_v4.csv")}
    for path in paths.values():
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; refusing to overwrite")

    enforce_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = read_labels(args.labels)
    v4_config = load_arm_config(V4_CONFIG)
    clip_config = load_arm_config(CLIP_CONFIG)
    assert v4_config["clip_fusion"] != "none"
    prototypes, proto_skipped = build_loo_prototypes(labels, v4_config)
    text_goals, text_skipped = build_text_goals(labels, v4_config, device)

    # 1. V4 action + horizon metrics through the Phase 4 scorer.
    v4_sources = {"V4_photo": "loader", "V4_p": prototypes, "V4_word": text_goals}
    v4_case, v4_scores = score_model(v4_config, args.checkpoint, v4_sources, device, args.max_batches)
    check_same_cases(v4_case, v4_case, labels, v4_config["distance"]["max_dist_cat"])

    # 2. Distance predictions for every config, on the same cases.
    batches = args.max_batches
    preds = {
        **distance_predictions(load_arm_config(V1_CONFIG), V1_CHECKPOINT, {"V1": None}, device, batches),
        **distance_predictions(clip_config, CLIP_CHECKPOINT,
                               {"V2": "loader", "V2p": prototypes, "V3": text_goals}, device, batches),
        **distance_predictions(v4_config, args.checkpoint, v4_sources, device, batches),
    }
    distance = v4_case["distance"].astype(np.float64)
    # A pilot scores only the first n cases; cut every per-case array to them.
    n = len(distance)
    labels, proto_skipped, text_skipped = labels[:n], proto_skipped[:n], text_skipped[:n]

    # The distance-only pass must be the one the scorer ran: same squared error per case.
    phase4 = {c: {k: v[:n] for k, v in cols.items()}
              for c, cols in load(os.path.join(PHASE4_DIR, "results.csv")).items()}
    reproduction = {}
    for name in OLD_CONFIGS + V4_CONFIGS:
        stored = phase4[name]["gc_dist_loss"] if name in OLD_CONFIGS else v4_scores[name]["gc_dist_loss"]
        ok = ~np.isnan(stored)
        diff = np.abs((preds[name] - distance) ** 2 - stored)[ok]
        # Relative, since the stored values carry 6 significant digits.
        rel = diff / np.maximum(1.0, np.abs(stored[ok]))
        reproduction[name] = {"max_rel_diff": float(rel.max()),
                              "pass": bool(rel.max() <= DIST_REPRODUCTION_TOLERANCE)}
    assert all(r["pass"] for r in reproduction.values()), f"distance pass mismatch: {reproduction}"

    # 3. Summary: the Phase 4 tiers, paired drops and tau, V4 rows, then corr rows.
    v1 = phase4["V1"]
    dropped = (phase4["V2p"]["skipped"] == 1) | (phase4["V3"]["skipped"] == 1)
    assert np.array_equal(dropped, proto_skipped | text_skipped), "V4 skips differ from Phase 4's"
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
    with open(os.path.join(PHASE4_DIR, "summary_table.csv")) as f:
        old_rows = list(csv.reader(f))

    new_rows = []
    for tier, keep in tiers.items():
        for name in V4_CONFIGS:
            cols = {m: np.asarray(v, dtype=np.float64) for m, v in v4_scores[name].items()}
            for metric in PRIMARY:
                new_rows.append((tier, name, metric, *stats(cols[metric][keep])))
            for metric in HORIZON:
                new_rows.append((tier, name, metric, *stats(cols[metric][keep & positive])))
            for label, tau in (("success@tau1", tau1), ("success@tau2", tau2)):
                hits = (cols["goal_step_dist"][keep & within] < tau).astype(float)
                new_rows.append((tier, name, label, *stats(hits)))
            # Fusion gives the distance head a current-frame signal, so for V4 it is reported
            # as a primary metric; the label keeps the Phase 4 row name for V1-V3 lookups.
            new_rows.append((tier, name, "gc_dist_loss", *stats(cols["gc_dist_loss"][keep])))
        for name in OLD_CONFIGS + V4_CONFIGS:
            new_rows.append((tier, name, "dist_head_corr",
                             pearson(preds[name][keep], distance[keep]), float("nan"), int(keep.sum())))

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(paths["dist_preds.npz"], distance=distance, **preds)
    skipped = {"V4_photo": np.zeros(n, dtype=bool), "V4_p": proto_skipped, "V4_word": text_skipped}
    write_v4_results(paths["results_v4.csv"], labels, v4_case, v4_scores, skipped)
    with open(paths["summary_table_v4.csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(old_rows)  # Phase 4 rows, header included, unchanged
        writer.writerows(new_rows)
    meta = {
        "v4": {"config": V4_CONFIG, "checkpoint": args.checkpoint,
               "sha256": checkpoint_digest(args.checkpoint)},
        "phase4_rows_from": os.path.join(PHASE4_DIR, "summary_table.csv"),
        "labels": args.labels, "seed": SEED, "batch_size": BATCH_SIZE, "max_batches": args.max_batches,
        "goal_mask": "gc = goal visible (input_goal_mask=0); uc = masked (1)",
        "tau1": tau1, "tau2": tau2,
        "n_cases": {tier: int(keep.sum()) for tier, keep in tiers.items()},
        "skipped": {c: int(s.sum()) for c, s in skipped.items()},
        "dist_reproduction": reproduction,
        "uc_action_loss_identical_V4": all(
            np.array_equal(v4_scores["V4_photo"]["uc_action_loss"], v4_scores[c]["uc_action_loss"])
            for c in ("V4_p", "V4_word")),
        "git_head": git_head(),
    }
    with open(paths["results_v4.meta.json"], "w") as f:
        json.dump(meta, f, indent=2)

    lookup = {(t, c, m): float(mean) for t, c, m, mean, *_ in (
        [(r[0], r[1], r[2], r[3]) for r in old_rows[1:]] + [r[:4] for r in new_rows])}
    configs = OLD_CONFIGS + V4_CONFIGS
    metrics = PRIMARY + HORIZON + ["success@tau1", "success@tau2", "gc_dist_loss", "dist_head_corr"]
    for tier in tiers:
        print(f"\n== {tier}: {meta['n_cases'][tier]:,} cases ==")
        print(f"{'metric':<16}" + "".join(f"{c:>10}" for c in configs))
        for metric in metrics:
            cells = []
            for c in configs:
                key = "gc_dist_loss [excluded]" if metric == "gc_dist_loss" and c in OLD_CONFIGS else metric
                cells.append(f"{lookup[(tier, c, key)]:>10.3f}")
            print(f"{metric:<16}" + "".join(cells))
    print(json.dumps({k: meta[k] for k in ("dist_reproduction", "uc_action_loss_identical_V4")}, indent=2))
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
