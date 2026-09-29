"""Language-goal G7.1: the word reaches the policy, and the policy drives.

One episode, V3 (clip_v2b + word) on the frozen pilot set's first chair
task, filmed, and four gated checks before any arm is scored:

  1. config     goal_type, clip_fusion, context_size, context_stride and
                image_size were read from the checkpoint (or established from
                its weights), and are the V3 recipe's.
  2. actions: word != masked   from the start frame, under the same diffusion
                noise, the word goal and the masked goal give different
                action samples: the mean waypoint shift between them (over the
                8 samples and every step, averaged over ACTION_SEEDS noise
                draws) is at least NOISE_FACTOR x the shift between two noise
                draws under the same word goal. Gated on actions, not tokens:
                a masked goal token never reaches the policy (input_goal_mask
                drops it from attention), so its cosine to the word token says
                nothing about whether the word steers.
  3. directional waypoints  the chosen waypoint moves forward on most ticks,
                its heading varies, and the eight samples are not collapsed.
  4. >= 10 ticks, no crash.

Reported, not gated: the goal-token cosine (word vs masked), the
conditioning cosine, the token norms (word, photo), and how far iGibson
frames sit from GoStanford's in CLIP image space — cos(frame, mu_img). V3
runs no image CLIP, so that last one matters only for photo-fed CLIP arms
and fused (V4/V5) arms in the full run.

    ./sim_eval/run_lg7.sh lg7_2_plumbing.py [--output-dir ...] [--task-set ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import checkpoints
import episode_runner
import gpu
import metrics
import object_tasks
import objects
import recorder
import run_eval

SIM_EVAL_DIR = Path(__file__).resolve().parent
CONFIG = SIM_EVAL_DIR / "configs" / "word_goal.yaml"
OUTPUT_DIR = SIM_EVAL_DIR / "outputs" / "p7_1_plumbing"
ARM = "clip_v2b+word"
WORD = "chair"
ACTION_SEEDS = 5
NOISE_FACTOR = 2.0
# A second noise draw, far from every task seed and its replay offsets.
NOISE_SEED_SHIFT = 50000
EXPECTED = {"goal_type": "clip", "clip_fusion": "none", "context_size": 3,
            "context_stride": 1, "image_size": [96, 96]}
GOSTANFORD_SAMPLE = 2000


def gostanford_mu_cosines(cache_path, mu_img, count=GOSTANFORD_SAMPLE):
    """cos(l2(v), mu_img) for an evenly spaced sample of the cached training frames."""
    import lmdb

    env = lmdb.open(str(cache_path), readonly=True, lock=False)
    try:
        with env.begin() as txn:
            total = txn.stat()["entries"]
            step = max(total // count, 1)
            vectors = [np.frombuffer(value, dtype=np.float32)
                       for index, (_key, value) in enumerate(txn.cursor())
                       if index % step == 0]
    finally:
        env.close()
    vectors = np.stack(vectors)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors @ (mu_img / np.linalg.norm(mu_img))


def mean_shift(a, b):
    """Mean L2 distance between two sets of action samples, per sample and step."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1).mean())


def action_samples(policy, frames, goal, seed):
    episode_runner.seed_episode(seed)
    return np.asarray(policy.act_goal(frames, goal).samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="a new folder (default: %(default)s)")
    parser.add_argument("--task-set", type=Path, default=None,
                        help="the frozen pilot task set (default: the config's "
                             "object_tasks.directory)")
    parser.add_argument("--no-film", action="store_true",
                        help="score the episode without filming it")
    args = parser.parse_args()
    selected_gpu = gpu.select_gpu(args.gpu)

    import torch

    import bridge
    import goals
    from nomad_policy import NomadPolicy

    output_dir = args.output_dir
    if output_dir.exists():
        raise SystemExit("FAILED: {} exists; G7.1 writes to a new folder".format(output_dir))
    config = run_eval.ObjectEvalConfig.from_dict(run_eval.read_layered_yaml(CONFIG))
    if args.task_set is None:
        args.task_set = config.task_directory
    manifest, tasks = object_tasks.load(args.task_set)
    if manifest["fingerprint"] != config.tasks.fingerprint():
        raise SystemExit("FAILED: {} was not built from {} (fingerprint {} vs {})".format(
            args.task_set, CONFIG, manifest["fingerprint"], config.tasks.fingerprint()))
    task = next(t for t in tasks if t.word == WORD)
    output_dir.mkdir(parents=True)
    csv_path = output_dir / "{}.csv".format(ARM)
    checks = {"task": {"task_set": str(args.task_set), "fingerprint": manifest["fingerprint"],
                       "task_id": task.task_id, "seed": task.seed}}

    # 1. config
    checkpoint_name, kind = run_eval.parse_arm(ARM)
    spec = checkpoints.load_goal_arm(checkpoint_name)
    print("checkpoint: {}".format(spec.summary()))
    got = {key: spec.model_params[key] for key in EXPECTED}
    checks["config"] = {"pass": got == EXPECTED, "read": got,
                        "provenance": spec.goal_provenance}

    policy = NomadPolicy(spec, torch.device("cuda"), config.driver)
    word = goals.build_goal("word", spec, policy.clip, word=task.word)
    masked = goals.build_goal("masked", spec, policy.clip, word=task.word)
    photo = goals.build_goal("photo", spec, policy.clip, word=task.word, photo=task.photo())

    body = bridge.SimBody(config_path=task.world_config, floor=config.tasks.floor)
    try:
        body.verify_gpu(selected_gpu)
        house = objects.SceneObjects.for_scene(body.scene)
        body.reset()
        body.place(*task.start_pose(body.scene.floor_height))
        frames = [body.observe()] * (spec.context_size + 1)

        # Reported: the goal tokens and the conditioning they produce.
        tokens = {name: policy.goal_token(frames, goal)
                  for name, goal in (("word", word), ("masked", masked), ("photo", photo))}
        norms = {name: float(np.linalg.norm(token)) for name, token in tokens.items()}
        cond = {name: policy.condition(frames, goal).cpu().numpy().ravel()
                for name, goal in (("word", word), ("masked", masked))}
        checks["goal_token_report"] = {
            "cos_token_word_vs_masked": round(goals.cosine(tokens["word"], tokens["masked"]), 4),
            "cos_conditioning_word_vs_masked": round(
                goals.cosine(cond["word"], cond["masked"]), 4),
            "token_norms": {k: round(v, 3) for k, v in norms.items()}}
        checks["word_vs_photo_goal_vec"] = round(goals.cosine(
            word.vec.cpu().numpy(), photo.vec.cpu().numpy()), 4)
        # The word's goal vector against every other word's, for scale.
        others = [w for w in ("sofa", "table", "door", "hallway") if w != task.word]
        checks["photo_vec_vs_words"] = {
            w: round(goals.cosine(photo.vec.cpu().numpy(),
                                  policy.clip.word(w).cpu().numpy()), 4)
            for w in [task.word] + others}

        # 2. Same frame, same noise: do word and masked steer differently, by
        # more than the noise alone moves the samples?
        pairs = []
        for k in range(ACTION_SEEDS):
            seed = task.seed + k
            word_samples = action_samples(policy, frames, word, seed)
            masked_samples = action_samples(policy, frames, masked, seed)
            other_noise = action_samples(policy, frames, word, seed + NOISE_SEED_SHIFT)
            pairs.append({"seed": seed,
                          "word_vs_masked": mean_shift(word_samples, masked_samples),
                          "noise_only": mean_shift(word_samples, other_noise)})
            if k == 0:
                first_word_samples = word_samples
        goal_shift = float(np.mean([p["word_vs_masked"] for p in pairs]))
        noise_shift = float(np.mean([p["noise_only"] for p in pairs]))
        checks["actions_word_vs_masked"] = {
            "pass": goal_shift >= NOISE_FACTOR * noise_shift,
            "mean_shift_word_vs_masked": round(goal_shift, 4),
            "mean_shift_noise_only": round(noise_shift, 4),
            "ratio": round(goal_shift / noise_shift, 3) if noise_shift > 0 else None,
            "required_ratio": NOISE_FACTOR,
            "units": "normalized waypoint units (x {:.3f} m)".format(
                bridge.waypoint_scale_m(policy.model_params, body.limits)),
            "per_seed": [{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in p.items()} for p in pairs]}

        # 5. the episode itself, filmed
        config.recording.enabled = not args.no_film
        config.recording.directory = output_dir / "videos"
        goal_for = lambda _task: word  # noqa: E731
        film = recorder.disabled() if args.no_film else recorder.ObjectGoalRecorder(
            config.recording, waypoint_scale_m=bridge.waypoint_scale_m(
                policy.model_params, body.limits),
            success_radius_m=config.rules.success_radius_m, objects=house,
            goal_for=goal_for)
        runner = episode_runner.ObjectGoalRunner(policy, body, house, goal_for,
                                                 rules=config.rules)
        table = metrics.MetricsTable(csv_path).open(resume=False)
        crashed = False
        sim_frames = []
        try:
            episode = runner.episode(task, ARM)
            with film.episode(task, ARM, body.scene) as video:
                for record in episode:
                    video.capture(record)
                    if record.index % 10 == 0:
                        sim_frames.append(record.frame)
                result = episode.result()
                if hasattr(video, "finish"):
                    video.finish(result.metrics)
            table.append(result.metrics, trace=result.trace())
            print(result.metrics.summary())
        except Exception:  # noqa: BLE001 — a crash is the finding here
            import traceback
            traceback.print_exc()
            crashed = True
    finally:
        body.close()

    ticks_log = [] if crashed else result.trace()["ticks_log"]
    checks["ticks_no_crash"] = {"pass": (not crashed) and len(ticks_log) >= 10,
                                "ticks": len(ticks_log), "crashed": crashed}

    # 3. directional waypoints
    if ticks_log:
        waypoints = np.asarray([tick["waypoint_m"] for tick in ticks_log])
        forward = float(np.mean(waypoints[:, 0] > 0.02))
        headings = np.degrees(np.arctan2(waypoints[:, 1], waypoints[:, 0]))
        spread = float(np.linalg.norm(first_word_samples - first_word_samples.mean(0),
                                      axis=-1).mean())
        checks["directional"] = {
            "pass": forward >= 0.5 and spread > 1e-3 and float(np.std(headings)) > 1.0,
            "forward_fraction": round(forward, 3),
            "heading_std_deg": round(float(np.std(headings)), 2),
            "sample_spread": round(spread, 4)}
    else:
        checks["directional"] = {"pass": False}

    # Domain shift (reported): sim frames vs GoStanford frames against mu_img.
    mu_img = policy.clip.mu_img.cpu().numpy()
    raw = np.stack([policy.clip.raw_image(frame).cpu().numpy() for frame in sim_frames]
                   + [policy.clip.raw_image(task.photo()).cpu().numpy()])
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    sim_cos = raw @ (mu_img / np.linalg.norm(mu_img))
    train_cos = gostanford_mu_cosines(
        Path(spec.model_params["clip_mu_img"]).parent / "embeddings.lmdb", mu_img)
    checks["domain_shift"] = {
        "sim_cos_mu_img_mean": round(float(sim_cos.mean()), 4),
        "sim_cos_mu_img_min": round(float(sim_cos.min()), 4),
        "gostanford_cos_mu_img_mean": round(float(train_cos.mean()), 4),
        "gostanford_cos_mu_img_p05": round(float(np.percentile(train_cos, 5)), 4),
        "sim_frames": int(len(sim_cos)), "gostanford_frames": int(len(train_cos))}

    gated = ("config", "actions_word_vs_masked", "directional", "ticks_no_crash")
    passed = all(checks[name]["pass"] for name in gated)
    (output_dir / "g7_1.json").write_text(json.dumps(checks, indent=2, default=str))
    print(json.dumps(checks, indent=2, default=str))
    print("G7.1 {}".format("PASSED" if passed else "FAILED: " + ", ".join(
        name for name in gated if not checks[name]["pass"])))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
