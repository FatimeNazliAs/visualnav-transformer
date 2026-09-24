"""Phase 3: one goal word per eval_paired.py test case, for the language-goal (V3) arm.

Which goal each test case has
-----------------------------
eval_paired.py scores the whole go_stanford test split with shuffle=False and seeds numpy
once (np.random.seed(seed)) before iterating; ViNT_Dataset then draws each sample's goal
with that RNG inside __getitem__ (_sample_goal), and nothing else there touches numpy's
RNG. Replaying _sample_goal over the samples in order, after the same seed, therefore
reproduces exactly the goal frame eval_paired.py fed to sample i -- without loading an
image. The replay uses the dataset's own index and _sample_goal, built by eval_paired's
own build_test_loader, so it cannot drift from what gets scored.

How a goal frame gets its word
------------------------------
Each unique goal frame is labelled once and the word spread to every sample using it.
  1. A VLM, greedy decoding, asked for the main object: BLIP-2 (open answer) or LLaVA-1.5
     (closed list: one of WORDS or "none"). With the closed list a low "other" rate only
     means a word was picked, not that it is right -- the spot-check is the quality gate.
  2. The answer is normalised onto WORDS (GOAL_WORDS with corridor merged into hallway)
     through SYNONYMS; anything that does not map is "other".
  3. "other" falls back to CLIP zero-shot over WORDS, scored in the centred space the
     model is fed: prep(image, mu_img) . prep(template(word), mu_txt), with the image
     embedding read from the Phase 1 cache (mildly circular, since the model uses CLIP).
Human spot-check fixes (--apply-fixes) overwrite a frame's word with source "human".

Outputs, next to each other in --out-dir (never overwritten):
    <name>.csv          one row per sample, in eval order; row i == row i of the .npz
    <name>.meta.json    VLM name + revision, prompt, decoding, seeds, word list, synonyms
    word_counts.txt     per-word counts (full run only)

Run inside the container, from /app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 HF_HOME=/outputs/nomad_clip_labels/hf_cache TOKENIZERS_PARALLELISM=false \\
        python ablation/label_test_goals.py --vlm llava --batch-size 8 --pilot 50
"""
# Must be set before torch creates its cuBLAS handle; see eval_paired.py.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import collections
import csv
import hashlib
import json
import re
import subprocess

import lmdb
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from eval_paired import ALIGNED_INDEX_CONTEXT_SIZE, build_test_loader, load_arm_config
from vint_train.data.clip_goal_utils import (
    GOAL_WORDS,
    cache_key,
    encode_words,
    load_clip,
    load_mu,
    prep,
)
from vint_train.data.data_utils import get_data_path

DEFAULT_CONFIG = "config/nomad_clip.yaml"
DEFAULT_OUT_DIR = "/outputs/nomad_clip_labels"
# BLIP-2's OPT decoder was tuned on "Question: ... Answer:" prompts. The pilot showed it
# ignores "one word" (echoing the question instead) and reads "landmark" as "building",
# so neither is asked for; normalize_answer picks the noun out of a short phrase.
BLIP2_PROMPT = "Question: What is the main object in this photo? Answer:"
MAX_NEW_TOKENS = 20
# temperature=0 is greedy decoding; transformers expresses it as do_sample=False.
GENERATION = {"do_sample": False, "num_beams": 1}

# eval_paired.py's n_samples for every arm scored on the aligned index (ctx03_s0.json).
EXPECTED_N_SAMPLES = 26648

# The label vocabulary. corridor and hallway are the same thing in GoStanford, so corridor
# is merged into hallway here; GOAL_WORDS itself stays as is because mu_txt was built on it.
WORDS = [word for word in GOAL_WORDS if word != "corridor"]
# Words that name an area rather than a place-identifying object.
SCENE_WORDS = {"hallway", "wall", "lobby", "office", "sidewalk", "building"}

# LLaVA follows instructions, so it chooses from the list itself. "none" maps to OTHER and
# goes to the CLIP fallback, so a frame with no recognisable object is not forced onto a word.
# One flat list. Asking for objects first (pilot 4) was less accurate: LLaVA then forced an
# object onto lobbies and corridors (mostly "door"). The flat list leans on "hallway", which
# is usually true of GoStanford goals; the object-only decision is made on the word counts.
NONE = "none"
PROMPT_WORDS = list(WORDS)
# The CLIP fallback scores WORDS; the two lists must be one space, or "none" -> CLIP would
# label on a different vocabulary than the VLM path.
assert PROMPT_WORDS == WORDS, "LLaVA prompt list and CLIP fallback candidates differ"
LLAVA_PROMPT = (
    "USER: <image>\nWhich word best names the main object or place in this photo? "
    f"Choose one from: {', '.join(PROMPT_WORDS + [NONE])}. "
    "Answer with the word only. ASSISTANT:"
)

VLMS = {
    "blip2": {"model": "Salesforce/blip2-opt-2.7b", "prompt": BLIP2_PROMPT},
    "llava": {"model": "llava-hf/llava-1.5-7b-hf", "prompt": LLAVA_PROMPT},
}

# VLM answer -> canonical word in WORDS. Plurals of WORDS are handled by strip_plural.
SYNONYMS = {
    "corridor": "hallway", "hall": "hallway", "passage": "hallway",
    "passageway": "hallway", "aisle": "hallway",
    "staircase": "stairs", "stair": "stairs", "stairway": "stairs",
    "stairwell": "stairs", "steps": "stairs", "step": "stairs",
    "couch": "sofa", "lift": "elevator",
    "entrance": "doorway", "entryway": "doorway", "gate": "door",
    "seat": "chair", "stool": "chair", "armchair": "chair",
    "trash": "trash can", "trashcan": "trash can", "bin": "trash can",
    "garbage": "trash can", "wastebasket": "trash can", "dumpster": "trash can",
    "bike": "bicycle",
    "pillar": "column", "post": "column",
    "rail": "railing", "handrail": "railing", "banister": "railing",
    "monitor": "screen", "tv": "screen", "television": "screen",
    "display": "screen", "computer": "screen",
    "bookcase": "bookshelf", "shelf": "bookshelf", "shelves": "bookshelf",
    "cupboard": "cabinet", "locker": "cabinet", "drawer": "cabinet",
    "light": "lamp",
    "bush": "plant", "shrub": "plant", "flower": "plant", "flowers": "plant",
    "vehicle": "car", "truck": "car",
    "blackboard": "whiteboard", "chalkboard": "whiteboard",
    "picture": "poster", "painting": "poster",
    "signage": "sign",
    "walkway": "sidewalk", "pavement": "sidewalk", "path": "sidewalk",
    "house": "building",
}
assert set(SYNONYMS.values()) <= set(WORDS), set(SYNONYMS.values()) - set(WORDS)

OTHER = "other"
CSV_COLUMNS = [
    "sample_idx", "obs_traj_id", "obs_frame_idx", "goal_traj_id", "goal_frame_idx",
    "goal_is_negative", "word", "word_kind", "source", "raw_vlm",
]


def replay_test_cases(config, seed):
    """(sample_idx, obs_traj, obs_time, goal_traj, goal_time, is_negative), as eval sees them."""
    dataset, _ = build_test_loader(config, batch_size=1)
    try:
        np.random.seed(seed)
        cases = []
        for i, (f_curr, curr_time, max_goal_dist) in enumerate(dataset.index_to_data):
            f_goal, goal_time, is_negative = dataset._sample_goal(f_curr, curr_time, max_goal_dist)
            cases.append((i, f_curr, curr_time, f_goal, goal_time, bool(is_negative)))
        return cases, dataset.data_folder
    finally:
        dataset.close()


def strip_plural(token):
    if token in WORDS or token in SYNONYMS:
        return token
    for suffix, repl in (("ies", "y"), ("es", ""), ("s", "")):
        if token.endswith(suffix) and token[: -len(suffix)] + repl in set(WORDS) | set(SYNONYMS):
            return token[: -len(suffix)] + repl
    return token


def canonical(term):
    term = strip_plural(term)
    if term in WORDS:
        return term
    return SYNONYMS.get(term)


def normalize_answer(raw):
    """VLM answer -> a word in WORDS, or OTHER.

    Whole answer first, then multi-word entries ("trash can") as substrings, then single
    tokens right to left -- the head noun of an English phrase is usually last, so
    "exit sign" -> sign and "glass door" -> door.
    """
    text = re.sub(r"[^a-z ]", " ", raw.lower())
    tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
    if not tokens or tokens == [NONE]:
        return OTHER
    whole = canonical(" ".join(tokens))
    if whole:
        return whole
    joined = " " + " ".join(tokens) + " "
    for entry in [w for w in list(WORDS) + list(SYNONYMS) if " " in w]:
        if f" {entry} " in joined:
            return canonical(entry)
    for token in reversed(tokens):
        word = canonical(token)
        if word:
            return word
    return OTHER


class GoalFrames(Dataset):
    def __init__(self, data_folder, frames):
        self.data_folder = data_folder
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        traj_name, time = self.frames[i]
        return Image.open(get_data_path(self.data_folder, traj_name, time)).convert("RGB")


def load_vlm(vlm, revision, device):
    if vlm == "blip2":
        from transformers import Blip2ForConditionalGeneration as Model, Blip2Processor as Processor
    else:
        from transformers import LlavaForConditionalGeneration as Model, LlavaProcessor as Processor
    name = VLMS[vlm]["model"]
    processor = Processor.from_pretrained(name, revision=revision)
    model = Model.from_pretrained(name, revision=revision, torch_dtype=torch.float16)
    return processor, model.to(device).eval()


def vlm_answers(vlm, frames, data_folder, device, batch_size, revision, prompt, max_new_tokens):
    """Raw VLM answer per frame, plus the resolved model revision."""
    processor, model = load_vlm(vlm, revision, device)
    loader = DataLoader(
        GoalFrames(data_folder, frames),
        batch_size=batch_size,
        num_workers=4,
        collate_fn=list,
    )
    answers = []
    with torch.no_grad():
        for images in tqdm.tqdm(loader, desc=vlm, dynamic_ncols=True):
            inputs = processor(
                images=images, text=[prompt] * len(images), return_tensors="pt"
            ).to(device, torch.float16)
            output = model.generate(**inputs, **GENERATION, max_new_tokens=max_new_tokens)
            if vlm == "llava":
                # A decoder-only model returns the prompt too; keep the new tokens only.
                output = output[:, inputs["input_ids"].shape[1]:]
            for text in processor.batch_decode(output, skip_special_tokens=True):
                # Some transformers versions echo the prompt; keep only the answer.
                answers.append(text.split("Answer:")[-1].strip())
    return answers, model.config._commit_hash


def clip_zero_shot(frames, config, device):
    """Best WORDS entry per frame, from the Phase 1 image cache in the centred space."""
    model, _, tokenizer = load_clip(config["clip_model"], device)
    text = prep(
        encode_words(model, tokenizer, WORDS, config["clip_text_template"], device),
        load_mu(config["clip_mu_txt"]).to(device),
    )
    mu_img = load_mu(config["clip_mu_img"]).to(device)
    env = lmdb.open(config["clip_cache"], readonly=True, lock=False)
    try:
        with env.begin() as txn:
            images = torch.stack([
                torch.from_numpy(np.frombuffer(txn.get(cache_key(t, f)), dtype=np.float32).copy())
                for t, f in frames
            ]).to(device)
    finally:
        env.close()
    scores = prep(images, mu_img) @ text.T
    return [WORDS[k] for k in scores.argmax(dim=1).tolist()]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_head():
    try:
        # The repo is a root-mounted volume, so git in the container calls it dubious.
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo}", "-C", repo, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def write_word_counts(path, rows):
    """Per-word counts over samples, descending; sparse and scene words flagged."""
    counts = collections.Counter(row["word"] for row in rows)
    positives = collections.Counter(row["word"] for row in rows if str(row["goal_is_negative"]) == "0")
    lines = [
        f"per-word counts over {len(rows):,} samples ({sum(positives.values()):,} non-negative)",
        f"{'word':<12} {'kind':<7} {'all':>7} {'non-neg':>8}  flag",
    ]
    for word, n in counts.most_common():
        kind = "scene" if word in SCENE_WORDS else ("object" if word != OTHER else "-")
        flag = "SPARSE(<20 non-neg)" if positives[word] < 20 else ""
        lines.append(f"{word:<12} {kind:<7} {n:>7} {positives[word]:>8}  {flag}")
    scene = sum(n for w, n in counts.items() if w in SCENE_WORDS)
    lines.append(f"scene words: {scene:,} ({scene / len(rows):.1%}), "
                 f"top: {', '.join(w for w, _ in counts.most_common() if w in SCENE_WORDS)}")
    objects = sum(n for w, n in counts.items() if w not in SCENE_WORDS and w != OTHER)
    lines.append(f"object words: {objects:,} ({objects / len(rows):.1%})")
    text = "\n".join(lines)
    print(text)
    with open(path, "w") as f:
        f.write(text + "\n")


def apply_fixes(csv_path, fixes_path):
    """Overwrite words per goal frame from a fixes CSV (goal_traj_id, goal_frame_idx, word)."""
    with open(fixes_path, newline="") as f:
        fixes = {
            (r["goal_traj_id"], r["goal_frame_idx"]): r["word"]
            for r in csv.DictReader(line for line in f if not line.startswith("#"))
        }
    for word in fixes.values():
        assert word in WORDS or word == OTHER, f"fix word {word!r} not in WORDS"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    changed = 0
    for row in rows:
        word = fixes.get((row["goal_traj_id"], row["goal_frame_idx"]))
        if word is not None:
            row.update(word=word, source="human",
                       word_kind="scene" if word in SCENE_WORDS else "object" if word != OTHER else "-")
            changed += 1
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"applied {len(fixes)} frame fixes to {changed} samples in {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--name", help="output basename (default test_words, or test_words_pilot)")
    parser.add_argument("--seed", type=int, default=0,
                        help="the EVAL seed of eval_paired.py; fixes which goal each sample has")
    parser.add_argument("--pilot", type=int, help="label only N samples, drawn with --pilot-seed")
    parser.add_argument("--pilot-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vlm", choices=sorted(VLMS), default="llava")
    parser.add_argument("--revision", help="pin the VLM revision (commit hash)")
    parser.add_argument("--prompt", help="override the VLM's default prompt")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--apply-fixes", metavar="FIXES_CSV",
                        help="only apply human fixes to the existing <name>.csv, then recount")
    args = parser.parse_args()

    name = args.name or ("test_words_pilot" if args.pilot else "test_words")
    prompt = args.prompt or VLMS[args.vlm]["prompt"]
    csv_path = os.path.join(args.out_dir, f"{name}.csv")
    meta_path = os.path.join(args.out_dir, f"{name}.meta.json")

    if args.apply_fixes:
        rows = apply_fixes(csv_path, args.apply_fixes)
        write_word_counts(os.path.join(args.out_dir, "word_counts.txt"), rows)
        return

    for path in (csv_path, meta_path):
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; pass a new --name, never overwrite")
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_arm_config(args.config, ALIGNED_INDEX_CONTEXT_SIZE)
    cases, data_folder = replay_test_cases(config, args.seed)
    if len(cases) != EXPECTED_N_SAMPLES:
        raise SystemExit(f"FAIL: replayed {len(cases)} samples, eval_paired scores {EXPECTED_N_SAMPLES}")
    if args.pilot:
        picked = np.random.RandomState(args.pilot_seed).choice(len(cases), args.pilot, replace=False)
        cases = [cases[i] for i in sorted(picked)]

    frames = sorted({(c[3], c[4]) for c in cases})
    print(f"{len(cases):,} samples, {len(frames):,} unique goal frames")

    raw, revision = vlm_answers(
        args.vlm, frames, data_folder, device, args.batch_size, args.revision, prompt,
        args.max_new_tokens,
    )
    words = [normalize_answer(r) for r in raw]
    sources = ["vlm" if w != OTHER else "clip_fallback" for w in words]
    fallback = [i for i, w in enumerate(words) if w == OTHER]
    if fallback:
        for i, word in zip(fallback, clip_zero_shot([frames[i] for i in fallback], config, device)):
            words[i] = word
    label = {frame: (words[i], sources[i], raw[i]) for i, frame in enumerate(frames)}

    rows = []
    for i, obs_traj, obs_time, goal_traj, goal_time, is_negative in cases:
        word, source, answer = label[(goal_traj, goal_time)]
        rows.append({
            "sample_idx": i, "obs_traj_id": obs_traj, "obs_frame_idx": obs_time,
            "goal_traj_id": goal_traj, "goal_frame_idx": goal_time,
            "goal_is_negative": int(is_negative), "word": word,
            "word_kind": "scene" if word in SCENE_WORDS else "object",
            "source": source, "raw_vlm": answer,
        })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    index_pkl = os.path.join(
        config["datasets"]["go_stanford"]["test"],
        f"dataset_dist_{config['distance']['min_dist_cat']}_to_{config['distance']['max_dist_cat']}"
        f"_context_{config['context_type']}_n{ALIGNED_INDEX_CONTEXT_SIZE}_slack_"
        f"{config['datasets']['go_stanford']['end_slack']}.pkl",
    )
    import transformers
    frame_fallback = len(fallback) / len(frames)
    sample_fallback = sum(r["source"] == "clip_fallback" for r in rows) / len(rows)
    meta = {
        "vlm": args.vlm, "vlm_model": VLMS[args.vlm]["model"], "vlm_revision": revision,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "dtype": "float16", "prompt": prompt, "prompt_words": PROMPT_WORDS,
        "generation": {**GENERATION, "max_new_tokens": args.max_new_tokens},
        "batch_size": args.batch_size, "eval_seed": args.seed,
        "pilot": args.pilot, "pilot_seed": args.pilot_seed if args.pilot else None,
        "config": os.path.abspath(args.config), "index_pkl": index_pkl,
        "index_pkl_sha256": sha256(index_pkl), "git_head": git_head(),
        "n_samples": len(rows), "n_unique_goal_frames": len(frames),
        "other_rate_frames": frame_fallback, "other_rate_samples": sample_fallback,
        "clip_fallback": {
            "clip_model": config["clip_model"], "template": config["clip_text_template"],
            "scoring": "argmax prep(img, mu_img) . prep(txt, mu_txt)",
            "clip_cache": config["clip_cache"], "clip_mu_img": config["clip_mu_img"],
            "clip_mu_txt": config["clip_mu_txt"],
        },
        "words": WORDS, "scene_words": sorted(SCENE_WORDS), "synonyms": SYNONYMS,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {csv_path}\n      {meta_path}")
    print(f"VLM 'other' rate = CLIP fallback rate: {frame_fallback:.1%} of frames, "
          f"{sample_fallback:.1%} of samples")
    if args.pilot:
        for row in rows:
            print(f"{row['sample_idx']:>6} {row['goal_traj_id']:>16}/{row['goal_frame_idx']:<4} "
                  f"{row['word']:<11} {row['source']:<13} {row['raw_vlm']!r}")
    else:
        write_word_counts(os.path.join(args.out_dir, "word_counts.txt"), rows)


if __name__ == "__main__":
    main()
