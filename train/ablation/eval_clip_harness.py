"""Phase 4: V1 / V2 / V2p / V3 on the same seeded test cases, goal always visible.

The four configs
----------------
    V1   vanilla NoMaD (ctx03 ema_29), photo goal through its own goal encoder
    V2   CLIP-goal NoMaD (V2b ema_29), goal_vec = prep(cached image embedding, mu_img)
    V2p  same model, goal_vec = leave-one-trajectory-out word prototype (image side)
    V3   same model, goal_vec = prep(encode_text(template(word)), mu_txt)
V2 -> V2p is the cost of a word category instead of a specific place; V2p -> V3 the cost
of the modality gap. V2p and V3 use the same (LLaVA) test word, so a V2p -> V3 gap is not
labeller noise.

Same cases, same noise
----------------------
Both models are scored through eval_paired's own build_test_loader (index_context_size 20,
shuffle=False, num_workers=0, batch 32) with np.random.seed(0) immediately before
iterating, so every sample draws the same goal as ctx03_s0.npz and test_words.csv. This is
checked, not assumed: each sample's distance label must match the one implied by the csv's
goal frame, and distance / action_mask / goal_pos must match between the V1 and CLIP passes.
The sampler noise is re-seeded (_seed_sampler) before every config's model_output, so V2,
V2p and V3 share the same noise and their uc_action_loss is identical by design.

Goal mask
---------
gc_* comes from model_output's goal-visible pass, which hard-codes input_goal_mask = 0;
uc_* from its masked pass (= 1). Neither reads goal_mask_prob from a training yaml.

Horizon-aware metrics (goal_pos: local frame at curr_time, normalised waypoint steps,
1 unit = 0.12 m; the same frame and units as the predicted waypoints)
    progress            ||g|| - ||g - p_final||        (gt_progress: same, for the GT path)
    heading_error       angle(p_final, g), degrees
    within_horizon      action_mask == 1 and distance <= len_traj_pred (distances 4..8)
    final_waypoint_dist ||g - p_final||
    goal_step_dist      ||g - p[distance - 1]||, the waypoint at the goal's own timestep
success@tau is left to the summary step, which picks tau from V1's within-horizon
distribution. Horizon metrics are meaningless for negatives (goal_pos is another
trajectory's position read in this one's frame); filter on within_horizon or action_mask.

Run inside the container, from /app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 python ablation/eval_clip_harness.py --max-batches 1 --tag pilot
    CUDA_VISIBLE_DEVICES=1 python ablation/eval_clip_harness.py
"""
# Must be set before torch creates its cuBLAS handle; see eval_paired.py.
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from build_word_prototypes import read_embeddings
from eval_paired import (
    IMAGENET_TRANSFORM,
    _seed_sampler,
    build_model,
    build_test_loader,
    checkpoint_digest,
    enforce_determinism,
    load_arm_config,
    per_sample_metrics,
)
from label_test_goals import EXPECTED_N_SAMPLES, WORDS, git_head
from vint_train.data.clip_goal_utils import encode_words, load_clip, load_mu, prep
from vint_train.training.train_utils import model_output

V1_CONFIG = "config/nomad_ctx03.yaml"
V1_CHECKPOINT = "/outputs/nomad_ctx_ablation/ctx03_2026_08_31_17_47_18/ema_29.pth"
V1_REFERENCE_SCORES = "/outputs/nomad_capstone/scores/ctx03_s0.npz"
CLIP_CONFIG = "config/nomad_clip.yaml"
CLIP_CHECKPOINT = "/outputs/nomad_clip_v2b/clip_vitb32_ln_2026_09_24_04_30_33/ema_29.pth"
DEFAULT_LABELS = "/outputs/nomad_clip_labels/test_words.csv"
DEFAULT_OUT_DIR = "/outputs/nomad_clip_eval"
# The eval seed and batch size ctx03_s0.npz was scored with; the sampler noise depends on both.
SEED = 0
BATCH_SIZE = 32

CONFIGS = ["V1", "V2", "V2p", "V3"]
WORD_CONFIGS = {"V2p", "V3"}
COLUMNS = [
    "sample_idx", "traj_id", "frame_idx", "goal_traj_id", "goal_frame_idx",
    "goal_is_negative", "distance", "action_mask", "is_headline", "config",
    "gc_action_loss", "uc_action_loss", "cosine_sim", "gc_dist_loss",
    "progress", "gt_progress", "heading_error", "within_horizon",
    "final_waypoint_dist", "goal_step_dist", "word", "skipped",
]


def read_labels(path):
    """test_words.csv rows, in eval order (row i is sample i)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == EXPECTED_N_SAMPLES, f"{len(rows)} label rows, expected {EXPECTED_N_SAMPLES}"
    assert all(int(r["sample_idx"]) == i for i, r in enumerate(rows)), "labels out of eval order"
    return rows


def is_headline(row):
    """Phase 3's headline V3 set: non-negative, object word, not elevator (DECISIONS.md)."""
    return row["goal_is_negative"] == "0" and row["word_kind"] == "object" and row["word"] != "elevator"


def expected_distance(row, max_dist_cat):
    """The distance label the dataset must produce if it drew the csv's goal for this sample."""
    if row["goal_is_negative"] == "1":
        return max_dist_cat
    return int(row["goal_frame_idx"]) - int(row["obs_frame_idx"])


def build_loo_prototypes(labels, config):
    """V2p goal vectors: per case, the word prototype built without the case's own trajectory.

    Pool: every unique goal frame of a non-negative, LLaVA-labelled (source == vlm) test
    case, with its centred image embedding prep(img, mu_img). For a case with word w and
    goal trajectory t, the prototype is l2_normalize(mean of the pool frames labelled w
    whose trajectory is not t). The exclusion is on goal_traj_id, not obs_traj_id: it is
    the goal frame's trajectory whose frames would leak the goal place into the
    prototype. For positives the two are the same; for negatives they differ.

    Returns ([N, 512] prototypes, [N] bool: no frame left -> the case is skipped for V2p).
    """
    pool = {}
    for r in labels:
        if r["source"] == "vlm" and r["goal_is_negative"] == "0":
            pool[(r["goal_traj_id"], int(r["goal_frame_idx"]))] = r["word"]
    frames = sorted(pool)
    centred = prep(read_embeddings(config["clip_cache"], frames), load_mu(config["clip_mu_img"]))

    word_sum, word_count, traj_sum, traj_count = {}, {}, {}, {}
    for (traj, _), word, vec in zip(frames, (pool[f] for f in frames), centred):
        word_sum[word] = word_sum.get(word, 0) + vec
        word_count[word] = word_count.get(word, 0) + 1
        traj_sum[(word, traj)] = traj_sum.get((word, traj), 0) + vec
        traj_count[(word, traj)] = traj_count.get((word, traj), 0) + 1

    prototypes = torch.zeros(len(labels), centred.shape[1])
    skipped = np.zeros(len(labels), dtype=bool)
    for i, r in enumerate(labels):
        word, traj = r["word"], r["goal_traj_id"]
        left = word_count.get(word, 0) - traj_count.get((word, traj), 0)
        if left == 0:
            skipped[i] = True
            continue
        # The mean's 1/left factor is removed by the normalisation.
        prototypes[i] = F.normalize(word_sum[word] - traj_sum.get((word, traj), 0), dim=0)
    return prototypes, skipped


def build_text_goals(labels, config, device):
    """V3 goal vectors: prep(encode_text(template(word)), mu_txt) per case; unknown word -> skipped."""
    model, _, tokenizer = load_clip(config["clip_model"], device)
    text = prep(
        encode_words(model, tokenizer, WORDS, config["clip_text_template"], device),
        load_mu(config["clip_mu_txt"]).to(device),
    ).cpu()
    del model
    index = {word: k for k, word in enumerate(WORDS)}
    skipped = np.array([r["word"] not in index for r in labels])
    vectors = torch.stack([text[index.get(r["word"], 0)] for r in labels])
    return vectors, skipped


def horizon_metrics(gc_actions, labels, goal_pos, distance, action_mask, len_traj_pred):
    """progress, gt_progress, heading_error, within_horizon, final_waypoint_dist, goal_step_dist."""
    pred, true = gc_actions[:, :, :2], labels[:, :, :2]
    goal_norm = goal_pos.norm(dim=-1)
    final_dist = (goal_pos - pred[:, -1]).norm(dim=-1)
    cross = pred[:, -1, 0] * goal_pos[:, 1] - pred[:, -1, 1] * goal_pos[:, 0]
    dot = (pred[:, -1] * goal_pos).sum(dim=-1)
    # pred[distance - 1] is the waypoint at the goal's timestep only because distance is
    # the raw temporal gap in frames (waypoint_spacing 1, checked in score_model).
    assert not distance.is_floating_point(), "distance must be an integer frame gap"
    goal_step = (distance - 1).clamp(0, len_traj_pred - 1)
    rows = torch.arange(len(pred), device=pred.device)
    return {
        "progress": goal_norm - final_dist,
        "gt_progress": goal_norm - (goal_pos - true[:, -1]).norm(dim=-1),
        "heading_error": torch.rad2deg(torch.atan2(cross.abs(), dot)),
        "within_horizon": (action_mask == 1) & (distance <= len_traj_pred),
        "final_waypoint_dist": final_dist,
        "goal_step_dist": (goal_pos - pred[rows, goal_step]).norm(dim=-1),
    }


def score_model(config, checkpoint, goal_sources, device, max_batches):
    """One pass over the aligned test split, scoring each config in goal_sources.

    goal_sources: config name -> None (photo goal, V1), "loader" (the dataset's goal_vec,
    V2), or an [N, 512] tensor of per-case goal vectors (V2p, V3).
    Returns (per-sample case fields, {config: per-sample metric arrays}).
    """
    assert config["datasets"]["go_stanford"].get("waypoint_spacing", 1) == 1, (
        "goal_step_dist indexes pred[distance - 1], which assumes waypoint_spacing 1")
    model = build_model(config, device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    dataset, loader = build_test_loader(config, BATCH_SIZE)
    case = {"distance": [], "action_mask": [], "goal_pos": []}
    scores = {name: {} for name in goal_sources}
    try:
        # As in eval_paired.score_checkpoint: pinned immediately before iterating.
        np.random.seed(SEED)
        first = 0
        with torch.no_grad():
            for batch_index, data in enumerate(tqdm.tqdm(
                    loader, total=max_batches or len(loader), dynamic_ncols=True)):
                if max_batches is not None and batch_index >= max_batches:
                    break
                obs_image, goal_image, actions, distance, goal_pos, _, action_mask, goal_vec = data
                batch_obs = torch.cat(
                    [IMAGENET_TRANSFORM(obs) for obs in torch.split(obs_image, 3, dim=1)], dim=1
                ).to(device)
                batch_goal = IMAGENET_TRANSFORM(goal_image).to(device)
                labels = actions.to(device)
                size = labels.shape[0]
                distance, goal_pos, action_mask = (
                    distance.to(device), goal_pos.to(device), action_mask.to(device))

                for name, source in goal_sources.items():
                    if source is None:
                        vec = None
                    elif isinstance(source, str):
                        vec = goal_vec.to(device)
                    else:
                        vec = source[first:first + size].to(device)
                    # Same sampler noise for every config on this batch.
                    _seed_sampler(SEED, batch_index)
                    outputs = model_output(
                        model, noise_scheduler, batch_obs, batch_goal,
                        pred_horizon=labels.shape[1], action_dim=labels.shape[2],
                        num_samples=1, device=device, batch_goal_vec=vec,
                    )
                    gc_loss, gc_cos = per_sample_metrics(outputs["gc_actions"], labels)
                    uc_loss, _ = per_sample_metrics(outputs["uc_actions"], labels)
                    batch = {
                        "gc_action_loss": gc_loss,
                        "uc_action_loss": uc_loss,
                        "cosine_sim": gc_cos,
                        "gc_dist_loss": (outputs["gc_distance"].squeeze(-1) - distance.float()) ** 2,
                        **horizon_metrics(outputs["gc_actions"], labels, goal_pos, distance,
                                          action_mask, config["len_traj_pred"]),
                    }
                    for metric, values in batch.items():
                        scores[name].setdefault(metric, []).append(values.cpu().numpy())

                case["distance"].append(distance.cpu().numpy())
                case["action_mask"].append(action_mask.cpu().numpy())
                case["goal_pos"].append(goal_pos.cpu().numpy())
                first += size
    finally:
        # LMDB reader slots are per-process; see eval_paired.main.
        dataset.close()
    case = {k: np.concatenate(v) for k, v in case.items()}
    scores = {name: {m: np.concatenate(v) for m, v in s.items()} for name, s in scores.items()}
    return case, scores


def check_same_cases(v1_case, clip_case, labels, max_dist_cat):
    """Fail unless both passes saw the csv's goals: same distance, action_mask and goal_pos."""
    n = len(v1_case["distance"])
    expected = np.array([expected_distance(r, max_dist_cat) for r in labels[:n]])
    for name, case in (("V1", v1_case), ("CLIP", clip_case)):
        bad = np.flatnonzero(case["distance"] != expected)
        assert len(bad) == 0, f"{name} pass drew different goals than test_words.csv at samples {bad[:10]}"
    assert np.array_equal(v1_case["action_mask"], clip_case["action_mask"]), "action_mask differs"
    assert np.allclose(v1_case["goal_pos"], clip_case["goal_pos"]), "goal_pos differs"


# Harness column -> ctx03_s0.npz column validated against it. *_diffusion_loss is not
# computed here, so it is not part of the check.
REPRODUCTION_COLUMNS = {"gc_action_loss": "gc_action_loss", "cosine_sim": "gc_action_waypts_cos_sim"}
# Pass if the means agree within this; same code path and seeds, so expected bit-exact.
REPRODUCTION_TOLERANCE = 0.05


def v1_reproduction(v1_scores, reference_path):
    """V1 against ctx03_s0.npz on the rows scored here, per REPRODUCTION_COLUMNS."""
    reference_scores = np.load(reference_path)
    result = {"reference": reference_path, "tolerance_on_mean": REPRODUCTION_TOLERANCE}
    for ours_name, reference_name in REPRODUCTION_COLUMNS.items():
        ours = v1_scores[ours_name]
        reference = reference_scores[reference_name][:len(ours)]
        diff = ours - reference
        result[f"{ours_name} vs {reference_name}"] = {
            "n": int(len(ours)),
            "mean_ours": float(ours.mean()),
            "mean_reference": float(reference.mean()),
            "mean_abs_diff": float(np.abs(diff).mean()),
            "max_abs_diff": float(np.abs(diff).max()),
            "pass": bool(abs(ours.mean() - reference.mean()) <= REPRODUCTION_TOLERANCE),
        }
    return result


def write_results(path, labels, case, scores, skipped):
    """Long format: one row per case x config, V1 V2 V2p V3 per case."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(len(case["distance"])):
            r = labels[i]
            base = {
                "sample_idx": i,
                "traj_id": r["obs_traj_id"],
                "frame_idx": r["obs_frame_idx"],
                "goal_traj_id": r["goal_traj_id"],
                "goal_frame_idx": r["goal_frame_idx"],
                "goal_is_negative": r["goal_is_negative"],
                "distance": int(case["distance"][i]),
                "action_mask": int(case["action_mask"][i]),
                "is_headline": int(is_headline(r)),
            }
            for name in CONFIGS:
                skip = bool(skipped[name][i])
                row = {**base, "config": name, "skipped": int(skip),
                       "word": r["word"] if name in WORD_CONFIGS else ""}
                for metric, values in scores[name].items():
                    value = values[i]
                    if metric == "within_horizon":
                        row[metric] = int(value)
                    else:
                        row[metric] = "" if skip else f"{float(value):.6g}"
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="", help="suffix for the output names, e.g. pilot")
    parser.add_argument("--max-batches", type=int, help="score only the first N batches (pilot)")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    results_path = os.path.join(args.out_dir, f"results{suffix}.csv")
    meta_path = os.path.join(args.out_dir, f"results{suffix}.meta.json")
    for path in (results_path, meta_path):
        if os.path.exists(path):
            raise SystemExit(f"FAIL: {path} exists; refusing to overwrite")

    enforce_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = read_labels(args.labels)
    v1_config = load_arm_config(V1_CONFIG)
    clip_config = load_arm_config(CLIP_CONFIG)
    assert clip_config["goal_type"] == "clip" and v1_config["goal_type"] == "image"

    prototypes, proto_skipped = build_loo_prototypes(labels, clip_config)
    text_goals, text_skipped = build_text_goals(labels, clip_config, device)

    v1_case, v1_scores = score_model(v1_config, V1_CHECKPOINT, {"V1": None}, device, args.max_batches)
    clip_case, clip_scores = score_model(
        clip_config, CLIP_CHECKPOINT,
        {"V2": "loader", "V2p": prototypes, "V3": text_goals}, device, args.max_batches)
    check_same_cases(v1_case, clip_case, labels, v1_config["distance"]["max_dist_cat"])

    scores = {**v1_scores, **clip_scores}
    n = len(v1_case["distance"])
    none = np.zeros(n, dtype=bool)
    skipped = {"V1": none, "V2": none, "V2p": proto_skipped[:n], "V3": text_skipped[:n]}

    os.makedirs(args.out_dir, exist_ok=True)
    reproduction = v1_reproduction(scores["V1"], V1_REFERENCE_SCORES)
    uc_identical = all(
        np.array_equal(scores["V2"]["uc_action_loss"], scores[c]["uc_action_loss"])
        for c in ("V2p", "V3"))
    meta = {
        "configs": {
            "V1": {"config": V1_CONFIG, "checkpoint": V1_CHECKPOINT,
                   "sha256": checkpoint_digest(V1_CHECKPOINT)},
            "V2/V2p/V3": {"config": CLIP_CONFIG, "checkpoint": CLIP_CHECKPOINT,
                          "sha256": checkpoint_digest(CLIP_CHECKPOINT)},
        },
        "labels": args.labels,
        "seed": SEED, "batch_size": BATCH_SIZE, "max_batches": args.max_batches,
        "n_cases": n, "n_rows": n * len(CONFIGS),
        "goal_mask": "gc = model_output goal-visible pass (input_goal_mask=0); uc = masked pass (1)",
        "v2p_prototypes": "leave-one-goal-trajectory-out mean of prep(img, mu_img) over "
                          "non-negative source==vlm test goal frames with the same word",
        "diffusion_loss": "not computed; V1 reproduction validated on "
                          + ", ".join(REPRODUCTION_COLUMNS.values()),
        "skipped": {c: int(s.sum()) for c, s in skipped.items()},
        # A case skipped in V2p or V3 is dropped from all four configs in the summary.
        "paired_drop_per_word": {
            word: int(count) for word, count in zip(*np.unique(
                [labels[i]["word"] for i in np.flatnonzero(skipped["V2p"] | skipped["V3"])],
                return_counts=True))
        },
        "n_within_horizon": int(scores["V1"]["within_horizon"].sum()),
        "v1_reproduction": reproduction,
        "uc_action_loss_identical_V2_V2p_V3": uc_identical,
        "any_nan": {c: bool(any(np.isnan(v[~skipped[c]]).any() for m, v in s.items()
                                if m != "within_horizon"))
                    for c, s in scores.items()},
        "git_head": git_head(),
    }
    # Written only after every check has run, so a failed run leaves no partial output.
    write_results(results_path, labels, v1_case, scores, skipped)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: meta[k] for k in (
        "n_cases", "skipped", "n_within_horizon", "v1_reproduction",
        "uc_action_loss_identical_V2_V2p_V3", "any_nan")}, indent=2))
    print(f"-> {results_path}")
    shown = ["gc_action_loss", "uc_action_loss", "cosine_sim", "progress", "heading_error",
             "goal_step_dist"]
    print(f"{'case':>4} {'cfg':<4} {'word':<9} {'dist':>4} {'hor':>3} "
          + " ".join(f"{m[:14]:>14}" for m in shown))
    for i in range(min(10, n)):
        for name in CONFIGS:
            word = labels[i]["word"] if name in WORD_CONFIGS else ""
            print(f"{i:>4} {name:<4} {word:<9} {int(v1_case['distance'][i]):>4} "
                  f"{int(scores[name]['within_horizon'][i]):>3} "
                  + " ".join(f"{scores[name][m][i]:>14.4f}" for m in shown))


if __name__ == "__main__":
    main()
