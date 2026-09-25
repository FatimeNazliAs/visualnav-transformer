"""Phase 4 diagnostic: CLIP-assigned image-space prototypes per goal word.

Diagnostic only -- not used in the primary comparison. CLIP zero-shot agreed with the
LLaVA test words on just 29.7% of test goal frames (7% for "door"), so prototypes built
from CLIP-assigned training frames would make a V2p -> V3 gap partly labeller noise. V2p
instead uses eval_clip_harness.build_loo_prototypes: leave-one-goal-trajectory-out means
over the LLaVA-labelled test goal frames. This file is kept for inspection.

V2p feeds the CLIP-goal model a *word's worth* of information on the *image* side of
CLIP: for each word, the mean centred CLIP image embedding of every training frame that
word is assigned to. V2 -> V2p is then the cost of a word category instead of a specific
place; V2p -> V3 the cost of the modality gap.

Which training frames a word gets
---------------------------------
Training frames have no VLM labels, so each is assigned by CLIP zero-shot -- Phase 3's own
`clip_zero_shot` (label_test_goals.py), over the same WORDS, so the rule cannot drift:
argmax prep(img, mu_img) . prep(template(word), mu_txt). Caveat for the table: the word
assignment uses the same CLIP as the encoder, so prototype quality is not independent of
CLIP's own geometry. Test words are LLaVA's; the per-word CLIP/LLaVA agreement on the
test goal frames is recorded in the meta file to quantify that mismatch.

What a prototype is
-------------------
    proto[w] = l2_normalize(mean_{frames f assigned w} prep(img_f, mu_img))
Re-normalised because the adapter was only ever fed unit-norm prep() outputs; the raw
mean's norm (a measure of how tight the category is) is kept in the meta file.

Outputs (in --out-dir, never overwritten):
    word_prototypes.npz        word -> float32 [512], the vector V2p feeds
    word_prototypes.meta.json  per-word frame counts, raw-mean norms, CLIP/LLaVA agreement

Run inside the container, from /app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 python ablation/build_word_prototypes.py
"""
# Must be set before torch creates its cuBLAS handle; see eval_paired.py.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import collections
import csv
import json
import sys

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from label_test_goals import WORDS, clip_zero_shot, git_head
from vint_train.data.clip_goal_utils import cache_key, load_mu, prep

# precompute_clip_embeddings.py lives one level up, in train/; its split_frames is the
# frame set the cache (and mu_img) was built from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from precompute_clip_embeddings import split_frames  # noqa: E402

DEFAULT_CONFIG = "config/nomad_clip.yaml"
DEFAULT_LABELS = "/outputs/nomad_clip_labels/test_words.csv"
DEFAULT_OUT_DIR = "/outputs/nomad_clip_eval"
# 158,420 in the Phase 1 cache meta.yaml; a mismatch means a different split or cache.
EXPECTED_TRAIN_FRAMES = 158420


def read_embeddings(cache_path, frames):
    """Raw cached CLIP image embeddings for frames, [N, 512] float32."""
    env = lmdb.open(cache_path, readonly=True, lock=False)
    try:
        with env.begin() as txn:
            return torch.from_numpy(np.stack([
                np.frombuffer(txn.get(cache_key(t, f)), dtype=np.float32) for t, f in frames
            ]))
    finally:
        env.close()


def build_prototypes(frames, words, config, device):
    """Centred mean image embedding per word, plus per-word counts and raw-mean norms."""
    centred = prep(read_embeddings(config["clip_cache"], frames).to(device),
                   load_mu(config["clip_mu_img"]).to(device))
    assigned = np.array(words)
    prototypes, stats = {}, {}
    for word in WORDS:
        rows = np.flatnonzero(assigned == word)
        if len(rows) == 0:
            stats[word] = {"n_frames": 0, "raw_mean_norm": None}
            continue
        mean = centred[torch.from_numpy(rows).to(device)].mean(dim=0)
        prototypes[word] = F.normalize(mean, dim=0).cpu().numpy().astype(np.float32)
        stats[word] = {"n_frames": int(len(rows)), "raw_mean_norm": float(mean.norm())}
    return prototypes, stats


def test_agreement(labels_path, config, device):
    """Per-word agreement between LLaVA's test-goal word and CLIP zero-shot on that frame."""
    with open(labels_path) as f:
        frame_word = {(r["goal_traj_id"], int(r["goal_frame_idx"])): r["word"]
                      for r in csv.DictReader(f) if r["source"] == "vlm"}
    frames = sorted(frame_word)
    clip_words = clip_zero_shot(frames, config, device)
    per_word = collections.defaultdict(lambda: [0, 0])
    for frame, clip_word in zip(frames, clip_words):
        per_word[frame_word[frame]][0] += clip_word == frame_word[frame]
        per_word[frame_word[frame]][1] += 1
    return {
        "n_frames": len(frames),
        "overall": sum(a for a, _ in per_word.values()) / len(frames),
        "per_llava_word": {w: {"agree": a, "n": n, "rate": a / n}
                           for w, (a, n) in sorted(per_word.items(), key=lambda kv: -kv[1][1])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    npz_path = os.path.join(args.out_dir, "word_prototypes.npz")
    meta_path = os.path.join(args.out_dir, "word_prototypes.meta.json")
    for path in (npz_path, meta_path):
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; refusing to overwrite")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_config = config["datasets"]["go_stanford"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = split_frames(data_config["data_folder"], data_config["train"])
    assert len(frames) == EXPECTED_TRAIN_FRAMES, (
        f"{len(frames)} train frames, expected {EXPECTED_TRAIN_FRAMES}")
    train_words = clip_zero_shot(frames, config, device)
    prototypes, stats = build_prototypes(frames, train_words, config, device)
    agreement = test_agreement(args.labels, config, device)

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(npz_path, **prototypes)
    meta = {
        "config": os.path.abspath(args.config),
        "clip_cache": config["clip_cache"],
        "clip_mu_img": config["clip_mu_img"],
        "clip_mu_txt": config["clip_mu_txt"],
        "template": config["clip_text_template"],
        "assignment": "label_test_goals.clip_zero_shot: "
                      "argmax prep(img, mu_img) . prep(template(word), mu_txt)",
        "prototype": "l2_normalize(mean(prep(img, mu_img)))",
        "n_train_frames": len(frames),
        "words": stats,
        "test_clip_vs_llava_agreement": agreement,
        "git_head": git_head(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{len(prototypes)} prototypes -> {npz_path}")
    print(f"{'word':<12} {'train frames':>12} {'raw norm':>9} {'test agree':>16}")
    for word, s in sorted(stats.items(), key=lambda kv: -kv[1]["n_frames"]):
        raw_norm = round(s["raw_mean_norm"], 3) if s["raw_mean_norm"] is not None else None
        a = agreement["per_llava_word"].get(word)
        agree_str = "" if a is None else f"{a['rate']:.0%} (n={a['n']})"
        print(f"{word:<12} {s['n_frames']:>12,} {raw_norm!s:>9} {agree_str:>16}")
    print(f"CLIP/LLaVA agreement on test goal frames: {agreement['overall']:.1%}")


if __name__ == "__main__":
    main()
