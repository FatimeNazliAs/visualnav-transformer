"""Phase 6b G0b probe: does a text mix-in at training time teach P-int to read word goals?

No NoMaD training. Phase 6's G0 probe fit P-int = MLP([o, g, o*g, o-g]) on image-image
(current, goal) pairs only; it scored photo goals well (r 0.62) but word goals poorly
(r 0.12), because a text embedding never looks like a near-duplicate of the current frame.
This refits P-int on the same train triples, with each goal swapped for its word with
probability --mix-prob, exactly as the V5 dataset does:

    g = prep(img(goal), mu_img)                      with probability 1 - p
    g = word_goal_table[pseudo_label(g)]             with probability p

where pseudo_label is the CLIP zero-shot argmax over the 31 label words in the centred
space (Phase 3's fallback), and the table rows are prep(text(template(w)), mu_txt). The
test side is unchanged from G0 (probe_fusion.py): photo, prototype and word goals, with
the test words from test_words.csv (LLaVA labels). p = 0 is refit alongside, so the G0
baseline sits in the same table.

Gates (pre-registered, Phase 6b). The first run's absolute gate (word r >= 0.3 at p = 0.3)
failed with word r 0.164 while even the image-side prototype, which has no modality gap,
reached only 0.190: a category-level goal carries little distance signal, so an absolute
0.3 looks out of reach for any word or prototype goal. The gate was therefore replaced,
before the p = 1 ceiling run, by
    A  word r at p = 1.0 >= 0.30                     -> stop: a higher p may meet 0.3
    B  otherwise, at p = 0.3: word r >= prototype r - 0.05, word MSE <= 1.1 x constant,
       photo r (full test) >= 0.5                    -> pass: train V5 at p = 0.3
Word and prototype r are on the object-word tier. The summary prints both rules for
whichever mix probs were fitted.

Run inside the container, from /app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 python ablation/probe_v5.py                       # p = 0, 0.3
    CUDA_VISIBLE_DEVICES=1 python ablation/probe_v5.py --mix-prob 0.5 1.0 \
        --out-dir /outputs/nomad_clip_v7/probe_g0b_ceiling                   # ceiling curve
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from eval_clip_harness import (
    CLIP_CONFIG,
    DEFAULT_LABELS,
    build_loo_prototypes,
    build_text_goals,
    expected_distance,
    is_headline,
    read_labels,
)
from eval_paired import load_arm_config
from label_test_goals import WORDS
from probe_fusion import SEED, centred_images, fit, sample_train_triples, score
from vint_train.data.clip_goal_utils import LABEL_WORDS, load_mu, pseudo_label, word_goal_table

DEFAULT_OUT_DIR = "/outputs/nomad_clip_v7/probe_g0b"
CEILING_WORD_R = 0.30       # rule A, at p = 1.0
PROTOTYPE_MARGIN = 0.05     # rule B: word r >= prototype r - margin
WORD_MSE_RATIO = 1.1        # rule B: word MSE <= ratio x constant
PHOTO_R_GATE = 0.5          # rule B: photo r (full test)

assert LABEL_WORDS == WORDS, "the pseudo-label vocabulary must be Phase 3's word list"


def mix_in_words(train_g, word_table, mix_prob):
    """Swap each goal for its pseudo-label word with probability mix_prob (seeded)."""
    swap = np.random.default_rng(SEED).random(len(train_g)) < mix_prob
    mixed = train_g.clone()
    mixed[swap] = word_table[pseudo_label(train_g[swap], word_table)]
    return mixed, swap


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--mix-prob", type=float, nargs="+", default=[0.3],
                        help="one or more; p = 0 is always fitted too")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    summary_path = os.path.join(args.out_dir, "summary.txt")
    if os.path.exists(summary_path):
        raise SystemExit(f"FAIL: {summary_path} exists; refusing to overwrite")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_arm_config(CLIP_CONFIG)
    max_dist = config["distance"]["max_dist_cat"]
    word_table = word_goal_table(config["clip_model"], config["clip_text_template"],
                                 load_mu(config["clip_mu_txt"]), device)

    train_obs, train_goal, train_y = sample_train_triples(config)
    train_o = centred_images(config, train_obs)
    train_g = centred_images(config, train_goal)
    constant = float(train_y.mean())
    train_words = Counter(LABEL_WORDS[k] for k in pseudo_label(train_g, word_table).tolist())

    labels = read_labels(args.labels)
    test_y = np.array([expected_distance(r, max_dist) for r in labels], dtype=np.float32)
    test_o = centred_images(config, [(r["obs_traj_id"], int(r["obs_frame_idx"])) for r in labels])
    photo = centred_images(config, [(r["goal_traj_id"], int(r["goal_frame_idx"])) for r in labels])
    prototypes, proto_skipped = build_loo_prototypes(labels, config)
    words, word_skipped = build_text_goals(labels, config, device)
    headline = np.array([is_headline(r) for r in labels]) & ~proto_skipped & ~word_skipped
    full = np.ones(len(labels), dtype=bool)
    evaluations = [
        ("photo / full test", photo, full),
        ("photo / object words", photo, headline),
        ("prototype / object words", prototypes, headline),
        ("word / object words", words, headline),
    ]

    results = {}
    for mix_prob in sorted({0.0, *args.mix_prob}):
        mixed_g, swap = mix_in_words(train_g, word_table, mix_prob)
        model = fit("P-int", train_o, mixed_g, train_y, device, args.epochs, args.batch_size, args.lr)
        results[f"p={mix_prob}"] = {
            "swapped": int(swap.sum()),
            "train (mixed)": score(model, "P-int", train_o, mixed_g, train_y, constant, device),
            "train (photo only)": score(model, "P-int", train_o, train_g, train_y, constant, device),
            **{name: score(model, "P-int", test_o[mask], goals[mask], test_y[mask], constant, device)
               for name, goals, mask in evaluations},
        }

    gates = {}
    for mix, res in results.items():
        if mix == "p=0.0":
            continue
        word, proto = res["word / object words"], res["prototype / object words"]
        photo_r = res["photo / full test"]["pearson_r"]
        gates[mix] = [
            (f"A: word r {word['pearson_r']:.3f} >= {CEILING_WORD_R}", word["pearson_r"] >= CEILING_WORD_R),
            (f"B: word r {word['pearson_r']:.3f} >= prototype r {proto['pearson_r']:.3f} - {PROTOTYPE_MARGIN}",
             word["pearson_r"] >= proto["pearson_r"] - PROTOTYPE_MARGIN),
            (f"B: word MSE/const {word['mse'] / word['constant_mse']:.3f} <= {WORD_MSE_RATIO}",
             word["mse"] <= WORD_MSE_RATIO * word["constant_mse"]),
            (f"B: photo r (full test) {photo_r:.3f} >= {PHOTO_R_GATE}", photo_r >= PHOTO_R_GATE),
        ]

    lines = [
        f"train triples: {len(train_y)} (one sampler pass, seed {SEED}); constant = train mean {constant:.3f}",
        f"probe: P-int, {args.epochs} epochs, batch {args.batch_size}, AdamW lr {args.lr}",
        "train pseudo-labels (all goals): " + ", ".join(
            f"{w} {n / len(train_y):.1%}" for w, n in train_words.most_common(8)),
        "",
        f"{'evaluation':<26} {'mix':<7} {'n':>6} {'MSE':>8} {'const':>8} {'MSE/const':>9} {'r':>7}",
    ]
    for name in ["train (mixed)", "train (photo only)"] + [e[0] for e in evaluations]:
        for mix, res in results.items():
            s = res[name]
            lines.append(f"{name:<26} {mix:<7} {s['n']:>6} {s['mse']:>8.3f} {s['constant_mse']:>8.3f} "
                         f"{s['mse'] / s['constant_mse']:>9.3f} {s['pearson_r']:>7.3f}")
        lines.append("")
    lines.append(f"swapped train goals: " + ", ".join(f"{m} {r['swapped']}" for m, r in results.items()))
    lines.append("Pre-registered rules (A applies at p = 1.0, B at p = 0.3; see the docstring):")
    for mix, checks in gates.items():
        lines.append(f"  {mix}:")
        lines += [f"    [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]
    summary = "\n".join(lines)
    print(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"results": results, "gates": {m: dict(c) for m, c in gates.items()}, "train_pseudo_labels": dict(train_words),
                   "args": vars(args)}, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
