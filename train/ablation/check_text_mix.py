"""Phase 6b unit check for the dataset text mix-in (clip_text_mix_prob), before any training.

1. Inert at 0.0: the test loader eval_paired builds (no clip_text_mix_prob -> default 0.0)
   must yield byte-identical samples to the pre-6b dataset code, on the first --n-test
   indices after np.random.seed(0) -- the paired eval's own order and seed. The pre-6b
   code is `git show HEAD:train/vint_train/data/vint_dataset.py`, passed as --reference.
2. Active at --mix-prob on train: over --n-train sampler draws, the fraction of goals that
   equal a word_goal_table row exactly (swapped), the word histogram of those, and that the
   current frame o is never swapped.

Run inside the container, from /app/visualnav-transformer/train:

    python ablation/check_text_mix.py --reference /outputs/nomad_clip_v7/unit_check/vint_dataset_head.py
"""
import argparse
import copy
import functools
import importlib.util
import os
from collections import Counter

import numpy as np
import torch

import eval_paired
from eval_paired import build_test_loader, load_arm_config
from vint_train.data.clip_goal_utils import LABEL_WORDS
from vint_train.data import vint_dataset
from vint_train.data.vint_dataset import ViNT_Dataset

CONFIG = "config/nomad_clip_v7.yaml"
DEFAULT_OUT_DIR = "/outputs/nomad_clip_v7/unit_check"
SEED = 0


def load_reference_class(path):
    spec = importlib.util.spec_from_file_location("vint_dataset_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # ViNT_Dataset reads data_config.yaml next to its own file; that is the package folder.
    module.__file__ = vint_dataset.__file__
    return module.ViNT_Dataset


def build_with(dataset_class, config):
    """build_test_loader's dataset, constructed through dataset_class instead."""
    original = eval_paired.ViNT_Dataset
    eval_paired.ViNT_Dataset = dataset_class
    try:
        dataset, _ = build_test_loader(config, batch_size=1)
    finally:
        eval_paired.ViNT_Dataset = original
    return dataset


def first_samples(dataset, n):
    np.random.seed(SEED)
    try:
        return [dataset[i] for i in range(n)]
    finally:
        dataset.close()


def check_inert(config, reference_path, n):
    new = first_samples(build_with(ViNT_Dataset, config), n)
    ref = first_samples(build_with(load_reference_class(reference_path), config), n)
    mismatches = [
        i for i, (a, b) in enumerate(zip(new, ref))
        if len(a) != len(b) or any(x.dtype != y.dtype or x.shape != y.shape
                                   or x.numpy().tobytes() != y.numpy().tobytes() for x, y in zip(a, b))
    ]
    return mismatches


def check_active(config, mix_prob, n):
    train_config = copy.deepcopy(config)
    data_config = train_config["datasets"]["go_stanford"]
    data_config["test"] = data_config["train"]  # build_test_loader reads "test"
    mixed_class = functools.partial(
        ViNT_Dataset, clip_text_mix_prob=mix_prob, clip_model=config["clip_model"],
        clip_text_template=config["clip_text_template"], clip_mu_txt=config["clip_mu_txt"],
    )
    dataset = build_with(mixed_class, train_config)
    table = dataset.clip_word_goals
    words, o_changed = Counter(), 0
    try:
        np.random.seed(SEED)
        picks = np.random.choice(len(dataset.index_to_data), size=n, replace=False)
        for i in picks:
            f_curr, curr_time, max_goal_dist = dataset.index_to_data[i]
            f_goal, goal_time, _ = dataset._sample_goal(f_curr, curr_time, max_goal_dist)
            o, g = dataset._load_goal_vec(f_goal, goal_time, f_curr, curr_time)
            o_changed += not torch.equal(o, dataset._load_clip_embedding(f_curr, curr_time))
            hits = (g == table).all(dim=1).nonzero().flatten()
            if len(hits):
                words[LABEL_WORDS[int(hits[0])]] += 1
    finally:
        dataset.close()
    return words, o_changed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", required=True, help="pre-6b vint_dataset.py (git show HEAD:...)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--mix-prob", type=float, default=0.3)
    args = parser.parse_args()
    summary_path = os.path.join(args.out_dir, "summary.txt")
    if os.path.exists(summary_path):
        raise SystemExit(f"FAIL: {summary_path} exists; refusing to overwrite")

    config = load_arm_config(CONFIG)
    mismatches = check_inert(config, args.reference, args.n_test)
    words, o_changed = check_active(config, args.mix_prob, args.n_train)
    swapped = sum(words.values())

    lines = [
        f"1. test loader at clip_text_mix_prob 0.0 vs pre-6b code, first {args.n_test} indices, seed {SEED}:",
        f"   [{'PASS' if not mismatches else 'FAIL'}] byte-identical samples "
        f"({len(mismatches)} mismatches{': ' + str(mismatches[:10]) if mismatches else ''})",
        f"2. train at clip_text_mix_prob {args.mix_prob}, {args.n_train} draws, seed {SEED}:",
        f"   swapped goals: {swapped} ({swapped / args.n_train:.1%}; expected {args.mix_prob:.0%})",
        f"   [{'PASS' if o_changed == 0 else 'FAIL'}] current frame o never swapped ({o_changed} changed)",
        "   word histogram of swapped goals: " + ", ".join(
            f"{w} {n / swapped:.1%}" for w, n in words.most_common()),
    ]
    summary = "\n".join(lines)
    print(summary)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
