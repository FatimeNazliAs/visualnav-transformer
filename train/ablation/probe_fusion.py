"""Phase 6 G0 probe: can cached CLIP embeddings predict distance, with and without the current frame?

No NoMaD training. On (current frame, goal frame, distance) triples drawn by the dataset's
own sampler, fit three small regressors on the centred CLIP image embeddings
o = prep(img(current), mu_img) and g = prep(goal, mu):

    P-goal  MLP(g)                      the V2b situation: the goal token alone
    P-lin   Linear([o, g])              purely additive, W1 o + W2 g
    P-int   MLP([o, g, o*g, o-g])       the clip_fusion: interaction input

If P-goal is no better than a constant, the V2b distance head is constant because its
input carries no distance signal (the Phase 5 diagnosis). P-int beating P-lin says the
obs<->goal interaction terms are what carry it.

Train triples: one pass of ViNT_Dataset._sample_goal over the train index after
np.random.seed(0) (negatives included, distance = max_dist_cat). Test triples: the 26,648
Phase 4 cases from test_words.csv, so the tiers and words are the harness's own. Test goals
are fed three ways, as in the harness: photo (prep(img, mu_img)), prototype (V2p's
leave-one-goal-trajectory-out word prototype) and word (prep(text(template(word)), mu_txt)).
Photo is scored on the full test set and on the object-word (headline) tier; prototype and
word on the headline tier only.

Run inside the container, from /app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 python ablation/probe_fusion.py
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from build_word_prototypes import read_embeddings
from eval_clip_harness import (
    CLIP_CONFIG,
    DEFAULT_LABELS,
    build_loo_prototypes,
    build_text_goals,
    expected_distance,
    is_headline,
    read_labels,
)
from eval_paired import build_test_loader, load_arm_config
from vint_train.data.clip_goal_utils import fusion_features, load_mu, prep

DEFAULT_OUT_DIR = "/outputs/nomad_clip_v6/probe_g0"
SEED = 0


def sample_train_triples(config):
    """(obs frame, goal frame, distance) for every train sample, from the dataset's own sampler."""
    split_config = copy.deepcopy(config)
    data_config = split_config["datasets"]["go_stanford"]
    # build_test_loader reads the "test" split path; point it at train to reuse its arguments.
    data_config["test"] = data_config["train"]
    dataset, _ = build_test_loader(split_config, batch_size=1)
    obs, goal, distance = [], [], []
    try:
        np.random.seed(SEED)
        for f_curr, curr_time, max_goal_dist in dataset.index_to_data:
            f_goal, goal_time, negative = dataset._sample_goal(f_curr, curr_time, max_goal_dist)
            obs.append((f_curr, curr_time))
            goal.append((f_goal, goal_time))
            distance.append(dataset.max_dist_cat if negative
                            else (goal_time - curr_time) // dataset.waypoint_spacing)
    finally:
        dataset.close()
    return obs, goal, np.array(distance, dtype=np.float32)


def centred_images(config, frames):
    """prep(img, mu_img) for frames, reading each unique frame from the cache once."""
    unique = sorted(set(frames))
    row = {frame: k for k, frame in enumerate(unique)}
    table = prep(read_embeddings(config["clip_cache"], unique), load_mu(config["clip_mu_img"]))
    return table[[row[frame] for frame in frames]]


def probe_inputs(kind, o, g):
    if kind == "P-goal":
        return g
    if kind == "P-lin":
        return torch.cat([o, g], dim=-1)
    return fusion_features(o, g)


def make_probe(kind, in_dim):
    if kind == "P-lin":
        return nn.Linear(in_dim, 1)
    # The same widths as the clip_fusion: interaction adapter, plus a scalar head.
    return nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))


def fit(kind, o, g, y, device, epochs, batch_size, lr):
    torch.manual_seed(SEED)
    x = probe_inputs(kind, o, g).to(device)
    y = torch.as_tensor(y, device=device)
    model = make_probe(kind, x.shape[1]).to(device)
    # Start the output at the label mean, so every probe begins as the constant predictor.
    nn.init.constant_(model.bias if kind == "P-lin" else model[-1].bias, float(y.mean()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(SEED)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator, device=device)
        for start in range(0, len(x), batch_size):
            batch = order[start:start + batch_size]
            loss = F.mse_loss(model(x[batch]).squeeze(-1), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def score(model, kind, o, g, y, constant, device):
    pred = model(probe_inputs(kind, o, g).to(device)).squeeze(-1).cpu().numpy()
    return {
        "n": int(len(y)),
        "mse": float(((pred - y) ** 2).mean()),
        "constant_mse": float(((constant - y) ** 2).mean()),
        "pearson_r": float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
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

    train_obs, train_goal, train_y = sample_train_triples(config)
    train_o = centred_images(config, train_obs)
    train_g = centred_images(config, train_goal)
    constant = float(train_y.mean())

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
    for kind in ("P-goal", "P-lin", "P-int"):
        model = fit(kind, train_o, train_g, train_y, device, args.epochs, args.batch_size, args.lr)
        results[kind] = {
            "train": score(model, kind, train_o, train_g, train_y, constant, device),
            **{name: score(model, kind, test_o[mask], goals[mask], test_y[mask], constant, device)
               for name, goals, mask in evaluations},
        }

    photo_full = {k: results[k]["photo / full test"] for k in results}
    const_mse = photo_full["P-int"]["constant_mse"]
    gates = [
        ("P-goal photo MSE within 5% of constant",
         abs(photo_full["P-goal"]["mse"] / const_mse - 1) <= 0.05),
        ("P-int photo MSE <= 0.75 x constant", photo_full["P-int"]["mse"] <= 0.75 * const_mse),
        ("P-int photo Pearson r >= 0.4", photo_full["P-int"]["pearson_r"] >= 0.4),
        ("P-int beats P-lin (photo MSE)", photo_full["P-int"]["mse"] < photo_full["P-lin"]["mse"]),
    ]

    lines = [
        f"train triples: {len(train_y)} (one sampler pass, seed {SEED}); constant = train mean {constant:.3f}",
        f"probe fit: {args.epochs} epochs, batch {args.batch_size}, AdamW lr {args.lr}",
        "",
        f"{'evaluation':<26} {'probe':<7} {'n':>6} {'MSE':>8} {'const':>8} {'MSE/const':>9} {'r':>7}",
    ]
    for name in ["train"] + [e[0] for e in evaluations]:
        for kind in results:
            s = results[kind][name]
            lines.append(f"{name:<26} {kind:<7} {s['n']:>6} {s['mse']:>8.3f} {s['constant_mse']:>8.3f} "
                         f"{s['mse'] / s['constant_mse']:>9.3f} {s['pearson_r']:>7.3f}")
        lines.append("")
    lines.append("G0 gates (photo / full test):")
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in gates]
    summary = "\n".join(lines)
    print(summary)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"results": results, "gates": dict(gates), "args": vars(args)}, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
