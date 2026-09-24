"""Phase 2 modality-gap sanity check: does a word goal steer NoMaD like its photo does?

The CLIP-goal model was trained only on goal *images* (CLIP image embeddings centred on
mu_img); at test time it is meant to take a *word* (the CLIP text embedding of the
template prompt, centred on mu_txt). Before the full Phase 4 evaluation exists, this
checks on ~20 hand-labelled goals that word-fed actions are not degenerate:

  - not collapsed: near-zero spread across the 8 waypoints, or a near-zero reach, and
  - not just the goal-masked behaviour: the word must actually condition the policy.

Each row of the CSV (traj_id, frame_idx, word) names a goal frame of a test trajectory.
The observation context is the real context_size + 1 frames ending --goal-offset steps
before it on the same trajectory. The goal is then fed three ways, with that identical
observation and identical diffusion noise, so every difference between the outputs comes
from the goal token alone:

  img  CLIP image embedding of the goal frame, prep(., mu_img)       goal visible
  txt  CLIP text embedding of the templated word, prep(., mu_txt)    goal visible
  uc   goal masked -- the unconditional policy

Waypoints are in the dataset's normalised units (metres / metric_waypoint_spacing), which
is what the model predicts.

Run inside the container, from /app/visualnav-transformer/train:

    CUDA_VISIBLE_DEVICES=1 python ablation/sanity_clip_gap.py
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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from eval_paired import (
    IMAGENET_TRANSFORM,
    build_model,
    build_test_loader,
    enforce_determinism,
    load_arm_config,
)
from vint_train.data.clip_goal_utils import (
    GOAL_WORDS,
    encode_words,
    l2_normalize,
    load_clip,
    load_mu,
    prep,
)
from vint_train.training.train_utils import ACTION_STATS, get_action

DEFAULT_CSV = "ablation/sanity_words.csv"
DEFAULT_CONFIG = "config/nomad_clip.yaml"
DEFAULT_CHECKPOINT = "/outputs/nomad_clip_v2/clip_vitb32_2026_09_23_17_58_18/ema_29.pth"
DEFAULT_OUTPUT_DIR = "/outputs/nomad_clip_v2/sanity"

# FAIL thresholds on the word-goal output (normalised units, 1 unit = 0.12 m on GoStanford).
MIN_VARIANCE = 1e-4
MIN_L2_NORM = 0.1
# WARN when more than this fraction of word goals land nearer the masked-goal output than
# the photo-goal output -- the word is then barely conditioning the policy.
MAX_UC_COLLAPSE_FRACTION = 0.2

CSV_COLUMNS = ["traj_id", "frame_idx", "word"]


def read_cases(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        missing = set(CSV_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"FAIL: {path} is missing columns {sorted(missing)}")
        cases = [
            {"traj_id": row["traj_id"].strip(), "frame_idx": int(row["frame_idx"]),
             "word": row["word"].strip()}
            for row in reader
        ]
    if not cases:
        raise SystemExit(f"FAIL: {path} has no rows")
    return cases


def load_case_inputs(dataset, case, goal_offset, device):
    """Observation context, goal image, CLIP image goal and ground truth for one row.

    Uses the dataset's own loaders, so frames, resizing, context spacing, centring and
    action labels are exactly what training and eval see.
    """
    traj_id, goal_time = case["traj_id"], case["frame_idx"]
    if traj_id not in dataset.traj_names:
        raise SystemExit(f"FAIL: {traj_id} is not a test-split trajectory")
    traj_data = dataset._get_trajectory(traj_id)
    curr_time = goal_time - goal_offset
    context_times = dataset._context_times(curr_time)
    if context_times[0] < 0 or goal_time >= len(traj_data["position"]):
        raise SystemExit(
            f"FAIL: {traj_id}/{goal_time}: needs frames {context_times[0]}..{goal_time}, "
            f"trajectory has 0..{len(traj_data['position']) - 1}"
        )

    obs_image = torch.cat([dataset._load_image(traj_id, t) for t in context_times])
    obs_image = torch.cat(
        [IMAGENET_TRANSFORM(obs) for obs in torch.split(obs_image, 3, dim=0)]
    ).unsqueeze(0).to(device)
    goal_image = IMAGENET_TRANSFORM(dataset._load_image(traj_id, goal_time)).unsqueeze(0).to(device)
    img_goal_vec = dataset._load_goal_vec(traj_id, goal_time).unsqueeze(0).to(device)
    gt_actions, _ = dataset._compute_actions(traj_data, curr_time, goal_time)
    return obs_image, goal_image, img_goal_vec, np.asarray(gt_actions[:, :2], dtype=np.float64)


def sample_waypoints(model, noise_scheduler, cond, pred_horizon, seed, device):
    """Full reverse diffusion from a seeded generator, so each goal sees identical noise."""
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn((cond.shape[0], pred_horizon, 2), generator=generator, device=device)
    for k in noise_scheduler.timesteps:
        noise_pred = model(
            "noise_pred_net",
            sample=sample,
            timestep=k.unsqueeze(-1).repeat(sample.shape[0]).to(device),
            global_cond=cond,
        )
        sample = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=sample, generator=generator
        ).prev_sample
    return get_action(sample, ACTION_STATS)[0].cpu().numpy().astype(np.float64)


def spread(waypoints):
    """Variance of the 8 waypoints about their mean, averaged over x and y."""
    return float(waypoints.var(axis=0).mean())


def mean_l2(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--goal-offset", type=int, default=10,
                        help="steps between the current frame and the goal frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_path = os.path.join(args.output_dir, "sanity_results.csv")
    summary_path = os.path.join(args.output_dir, "sanity_summary.txt")
    if os.path.exists(results_path) and not args.overwrite:
        raise SystemExit(f"FAIL: {results_path} exists (--overwrite to redo)")
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"FAIL: no checkpoint at {args.checkpoint}")

    enforce_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_arm_config(args.config)
    if config["goal_type"] != "clip":
        raise SystemExit(f"FAIL: {args.config} is not a goal_type: clip config")
    cases = read_cases(args.csv)
    action = config["action"]
    if not action["min_dist_cat"] < args.goal_offset < action["max_dist_cat"]:
        raise SystemExit(f"FAIL: --goal-offset must lie strictly inside the action range {action}")

    model = build_model(config, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    dataset, _ = build_test_loader(config, batch_size=1)

    # Text goals: encoded once, always through the template, centred on mu_txt -- the
    # text-side mirror of what the dataset does to image goals with mu_img.
    clip_model, _, tokenizer = load_clip(config["clip_model"], device)
    raw_text = encode_words(clip_model, tokenizer, [c["word"] for c in cases],
                            config["clip_text_template"], device)
    if config["clip_center"]:
        text_goal_vecs = prep(raw_text, load_mu(config["clip_mu_txt"]).to(device))
    else:
        text_goal_vecs = l2_normalize(raw_text)
    del clip_model

    visible = torch.zeros(1, dtype=torch.long, device=device)
    masked = torch.ones(1, dtype=torch.long, device=device)
    rows = []
    try:
        with torch.no_grad():
            for i, case in enumerate(cases):
                obs, goal_image, img_vec, gt = load_case_inputs(dataset, case, args.goal_offset, device)
                txt_vec = text_goal_vecs[i:i + 1]
                seed = args.seed + i

                conds = {
                    "img": model("vision_encoder", obs_img=obs, goal_img=goal_image, input_goal_mask=visible, goal_vec=img_vec),
                    "txt": model("vision_encoder", obs_img=obs, goal_img=goal_image, input_goal_mask=visible, goal_vec=txt_vec),
                    "uc": model("vision_encoder", obs_img=obs, goal_img=goal_image, input_goal_mask=masked, goal_vec=img_vec),
                }
                wp = {k: sample_waypoints(model, noise_scheduler, c, config["len_traj_pred"], seed, device)
                      for k, c in conds.items()}
                dist = {k: float(model("dist_pred_net", obsgoal_cond=conds[k]).item()) for k in ("img", "txt")}

                rows.append({
                    **case,
                    "in_word_list": case["word"] in GOAL_WORDS,
                    "img_l2_norm": float(np.linalg.norm(wp["img"][-1])),
                    "txt_l2_norm": float(np.linalg.norm(wp["txt"][-1])),
                    "uc_l2_norm": float(np.linalg.norm(wp["uc"][-1])),
                    "img_variance": spread(wp["img"]),
                    "txt_variance": spread(wp["txt"]),
                    "cosine_sim": float(F.cosine_similarity(
                        torch.from_numpy(wp["img"].ravel()), torch.from_numpy(wp["txt"].ravel()), dim=0)),
                    "txt_to_img": mean_l2(wp["txt"], wp["img"]),
                    "txt_to_uc": mean_l2(wp["txt"], wp["uc"]),
                    "img_gt_err": float(np.linalg.norm(wp["img"][-1] - gt[-1])),
                    "txt_gt_err": float(np.linalg.norm(wp["txt"][-1] - gt[-1])),
                    "img_dist_pred": dist["img"],
                    "txt_dist_pred": dist["txt"],
                    "img_waypoints": json.dumps(np.round(wp["img"], 5).tolist()),
                    "txt_waypoints": json.dumps(np.round(wp["txt"], 5).tolist()),
                    "uc_waypoints": json.dumps(np.round(wp["uc"], 5).tolist()),
                    "gt_waypoints": json.dumps(np.round(gt, 5).tolist()),
                })
    finally:
        dataset.close()

    summary = summarize(rows, args)
    print(summary)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    print(f"\nWrote {results_path}\nWrote {summary_path}")


def summarize(rows, args):
    header = (f"{'traj_id':<22} {'frame':>5} {'word':<12} {'img_norm':>8} {'txt_norm':>8} "
              f"{'txt_var':>8} {'cos_sim':>7} {'txt>img':>7} {'txt>uc':>7}")
    lines = [
        f"checkpoint : {args.checkpoint}",
        f"cases      : {len(rows)} from {args.csv}, goal offset {args.goal_offset}, seed {args.seed}",
        "units      : normalised waypoints (1 = 0.12 m); txt>img / txt>uc = mean waypoint L2",
        "",
        header,
        "-" * len(header),
    ]
    for r in rows:
        lines.append(
            f"{r['traj_id']:<22} {r['frame_idx']:>5} {r['word']:<12} {r['img_l2_norm']:>8.3f} "
            f"{r['txt_l2_norm']:>8.3f} {r['txt_variance']:>8.4f} {r['cosine_sim']:>7.3f} "
            f"{r['txt_to_img']:>7.3f} {r['txt_to_uc']:>7.3f}"
        )

    collapsed = [r for r in rows if r["txt_variance"] < MIN_VARIANCE or r["txt_l2_norm"] < MIN_L2_NORM]
    uc_like = [r for r in rows if r["txt_to_uc"] < r["txt_to_img"]]
    uc_fraction = len(uc_like) / len(rows)
    img_variances = np.array([r["img_variance"] for r in rows])
    lines += [
        "",
        f"mean cosine_sim img vs txt      : {np.mean([r['cosine_sim'] for r in rows]):.3f}",
        f"mean final-wp error vs GT       : img {np.mean([r['img_gt_err'] for r in rows]):.3f}, "
        f"txt {np.mean([r['txt_gt_err'] for r in rows]):.3f}",
        f"img_variance range (reference)  : {img_variances.min():.4f} .. {img_variances.max():.4f}",
        f"collapsed word goals            : {len(collapsed)}/{len(rows)} "
        f"(txt_variance < {MIN_VARIANCE:g} or txt_l2_norm < {MIN_L2_NORM:g})",
        f"word goals nearer uc than img   : {len(uc_like)}/{len(rows)} ({uc_fraction:.0%})",
        f"words outside GOAL_WORDS        : {sum(not r['in_word_list'] for r in rows)}",
        "",
    ]
    if uc_fraction > MAX_UC_COLLAPSE_FRACTION:
        lines.append(f"WARN: more than {MAX_UC_COLLAPSE_FRACTION:.0%} of word goals behave like the "
                     "masked goal -- the word is barely conditioning the policy")
    if collapsed:
        lines.append("VERDICT: FAIL -- collapsed word-goal output in: "
                     + ", ".join(f"{r['traj_id']}/{r['frame_idx']}" for r in collapsed))
    else:
        lines.append("VERDICT: PASS -- no collapsed word-goal output")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
