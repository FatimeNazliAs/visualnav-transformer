"""Language-goal G7.1b: is G7.1's action gate measuring the model or the metric?

Action computation only — no episode is run. One frame (the start of the pilot
set's first chair task, repeated as the context, as G7.1 uses it), the same
five seeds, the same metric as G7.1:

    paired shift     mean over seeds, samples and steps of ||a_i - b_i||, where
                     a and b are the 8 action samples under goals A and B drawn
                     with the SAME seed (same diffusion noise)
    noise-only shift the same, between goal A under seed s and goal A under
                     seed s + 50000
    ratio            paired / noise-only (G7.1 wanted >= 2)

Controls:
  (a) V1 (ctx03): photo of the target vs masked — V1 uses its goal strongly
      offline, so this calibrates the ratio threshold.
  (b) V3 (clip_v2b): chair vs sofa, chair vs photo (chair_3's goal photo
      through CLIP), and masked vs masked with two different goal vectors
      (zero vs a random unit vector) — exactly 0 if masking drops the token.
      Also for V1: masked with the photo vs masked with the current frame.

Also reported, not gated (G7.1 option c): the mean-trajectory shift over
noise. Each goal's 40 samples (5 seeds x 8) give a mean trajectory; per
step t, z_t = ||mean_A - mean_B|| / sqrt(se_x^2 + se_y^2), where se^2 =
s_A^2 / 40 + s_B^2 / 40 per coordinate (the standard error of a difference
of two means). Reported as the mean of z_t over the 8 steps (and the max).

    ./sim_eval/run_lg7.sh lg7_2b_controls.py --output-dir sim_eval/outputs/p7_1b_controls
"""

import argparse
import json
from pathlib import Path

import numpy as np

import checkpoints
import episode_runner
import gpu
import object_tasks
import run_eval

SIM_EVAL_DIR = Path(__file__).resolve().parent
CONFIG = SIM_EVAL_DIR / "configs" / "word_goal.yaml"
WORD = "chair"
SEEDS = 5
NOISE_SEED_SHIFT = 50000


def samples(policy, frames, goal, seed):
    episode_runner.seed_episode(seed)
    return np.asarray(policy.act_goal(frames, goal).samples, dtype=np.float64)


def draws(policy, frames, goal, base_seed, shift=0):
    return [samples(policy, frames, goal, base_seed + k + shift) for k in range(SEEDS)]


def paired_shift(a, b):
    """Per seed, mean ||a_i - b_i|| over samples and steps."""
    return [float(np.linalg.norm(x - y, axis=-1).mean()) for x, y in zip(a, b)]


def mean_trajectory_z(a, b):
    """z_t of the difference of the two goals' mean trajectories, per step."""
    a, b = np.concatenate(a), np.concatenate(b)  # (40, steps, 2)
    n = a.shape[0]
    diff = np.linalg.norm(a.mean(0) - b.mean(0), axis=-1)
    se = np.sqrt((a.var(0, ddof=1) / n + b.var(0, ddof=1) / n).sum(-1))
    return diff / se


def compare(policy, frames, goal_a, goal_b, base_seed):
    a = draws(policy, frames, goal_a, base_seed)
    b = draws(policy, frames, goal_b, base_seed)
    a_other = draws(policy, frames, goal_a, base_seed, NOISE_SEED_SHIFT)
    paired = paired_shift(a, b)
    noise = paired_shift(a, a_other)
    z = mean_trajectory_z(a, b)
    return {
        "paired_shift_per_seed": [round(v, 6) for v in paired],
        "paired_shift": round(float(np.mean(paired)), 6),
        "noise_only_shift": round(float(np.mean(noise)), 6),
        "ratio": round(float(np.mean(paired) / np.mean(noise)), 4),
        "max_abs_difference": float(max(np.abs(x - y).max() for x, y in zip(a, b))),
        "mean_traj_z_mean": round(float(z.mean()), 3),
        "mean_traj_z_max": round(float(z.max()), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("FAILED: {} exists; controls go to a new folder".format(args.output_dir))
    selected_gpu = gpu.select_gpu(args.gpu)

    import torch

    import bridge
    import goals
    from nomad_policy import NomadPolicy

    config = run_eval.ObjectEvalConfig.from_dict(run_eval.read_layered_yaml(CONFIG))
    manifest, tasks = object_tasks.load(config.task_directory)
    if manifest["fingerprint"] != config.tasks.fingerprint():
        raise SystemExit("FAILED: {} is not the config's task set".format(config.task_directory))
    task = next(t for t in tasks if t.word == WORD)
    photo = task.photo()
    device = torch.device("cuda")

    body = bridge.SimBody(config_path=task.world_config, floor=config.tasks.floor)
    try:
        body.verify_gpu(selected_gpu)
        body.reset()
        body.place(*task.start_pose(body.scene.floor_height))
        frame = body.observe()
    finally:
        body.close()

    report = {"task": task.task_id, "seeds": [task.seed + k for k in range(SEEDS)],
              "task_set_fingerprint": manifest["fingerprint"]}

    v1_spec = checkpoints.load_goal_arm("ctx03")
    v1 = NomadPolicy(v1_spec, device, config.driver)
    frames = [frame] * (v1_spec.context_size + 1)
    v1_photo = goals.build_goal("photo", v1_spec, None, word=WORD, photo=photo)
    v1_masked = goals.build_goal("masked", v1_spec, None, word=WORD)
    v1_masked_photo = goals.Goal("masked", "masked, photo as the goal image",
                                 image=photo, masked=True)
    report["a_V1_photo_vs_masked"] = compare(v1, frames, v1_photo, v1_masked, task.seed)
    report["V1_masked_photo_vs_masked_frame"] = compare(v1, frames, v1_masked_photo,
                                                        v1_masked, task.seed)
    del v1
    torch.cuda.empty_cache()

    v3_spec = checkpoints.load_goal_arm("clip_v2b")
    v3 = NomadPolicy(v3_spec, device, config.driver)
    frames = [frame] * (v3_spec.context_size + 1)
    chair = goals.build_goal("word", v3_spec, v3.clip, word="chair")
    sofa = goals.build_goal("word", v3_spec, v3.clip, word="sofa")
    v3_photo = goals.build_goal("photo", v3_spec, v3.clip, word=WORD, photo=photo)
    masked_zero = goals.build_goal("masked", v3_spec, v3.clip, word=WORD)
    generator = torch.Generator().manual_seed(0)
    random_vec = torch.randn(masked_zero.vec.shape[0], generator=generator)
    masked_random = goals.Goal("masked", "masked, random goal vector",
                               vec=(random_vec / random_vec.norm()).to(device), masked=True)
    report["V3_word_vs_masked"] = compare(v3, frames, chair, masked_zero, task.seed)
    report["b_V3_chair_vs_sofa"] = compare(v3, frames, chair, sofa, task.seed)
    report["b_V3_chair_vs_photo"] = compare(v3, frames, chair, v3_photo, task.seed)
    report["b_V3_masked_zero_vs_masked_random"] = compare(v3, frames, masked_zero,
                                                          masked_random, task.seed)

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "controls.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
