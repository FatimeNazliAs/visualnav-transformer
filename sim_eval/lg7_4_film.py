"""Language-goal Phase 7: film the pilot's clip, and prove it is the scored episode.

Scoring runs film nothing. This picks the clip afterwards, by a fixed rule —
an episode (task, seed) where V3+word succeeds and V3+masked fails, if there
is one, the best of them by highest word SPL, then fewest ticks; otherwise
the best V3+word success by the same order — and re-runs exactly that
(task, seed) under every arm with recording on. The rule, what it found and
`--label` are written to selection.json and captioned on every frame. Episodes are
deterministic in their seed, so each film should *be* the scored episode;
that is checked, not assumed: every re-film's whole metrics row, column for
column, must equal the row the scoring run logged.

Only if every arm matches are the three films stacked into one synchronized
clip — the same house, the same start, the model given the word, the same
model given no goal, vanilla NoMaD given a photo — each held on its last
frame until the longest ends. On any mismatch there is no stacked clip, the
mismatched film is renamed `*.MISMATCH.mp4` (it is not the scored episode),
and the script exits 1.

The task set, seed offsets, driver and rules are the run's own, read from its
`run_manifest.json`; `clips/` must not exist yet.

    ./sim_eval/run_lg7.sh lg7_4_film.py --run-dir sim_eval/outputs/p7_pilot \
        --output-dir sim_eval/outputs/p7_videos [--label "G7.2 not passed"]

Writes <output-dir>/ (a new folder): <arm>/<task>[+<offset>].mp4,
<task>+<offset>_3arms.mp4, refilm_<arm>.csv/.jsonl, match.csv, selection.json.
"""

import argparse
import csv
from pathlib import Path

import checkpoints
import episode_runner
import gpu
import metrics
import object_tasks
import objects
import recorder
import run_eval

SIM_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SIM_EVAL_DIR / "configs" / "word_goal.yaml"
HEADLINE_ARM = "clip_v2b+word"
MASKED_ARM = "clip_v2b+masked"
# The whole row: a re-film is the scored episode only if nothing differs.
MATCHED_COLUMNS = metrics.CSV_COLUMNS


def pick_clip(run_dir, arm=HEADLINE_ARM, masked_arm=MASKED_ARM):
    """(task_id, seed, selection) by the fixed rule in the module docstring."""
    rows = [r for r in metrics.read_table(run_dir / "{}.csv".format(arm)) if int(r["success"])]
    if not rows:
        raise SystemExit("FAILED: {} has no successful episode to film".format(arm))
    masked = {(r["task_id"], int(r["seed"])): int(r["success"])
              for r in metrics.read_table(run_dir / "{}.csv".format(masked_arm))}
    contrast = [r for r in rows if masked.get((r["task_id"], int(r["seed"]))) == 0]
    pool = contrast or rows
    best = max(pool, key=lambda r: (float(r["spl"]), -int(r["ticks"])))
    if contrast:
        text = ("selected example: word succeeds, masked fails ({} such episodes of {} "
                "word successes; chosen by highest SPL, then fewest ticks)".format(
                    len(contrast), len(rows)))
    else:
        text = ("selected example: best word success (no episode where word succeeds "
                "and masked fails; chosen by highest SPL, then fewest ticks)")
    selection = {"rule": "word succeeds and masked fails if any, else best word success; "
                         "highest SPL, then fewest ticks",
                 "word_arm": arm, "masked_arm": masked_arm,
                 "word_successes": len(rows), "word_succeeds_masked_fails": len(contrast),
                 "candidates": [[r["task_id"], int(r["seed"])] for r in contrast],
                 "chosen": [best["task_id"], int(best["seed"])], "caption": text}
    return best["task_id"], int(best["seed"]), selection


def film_arm(config, arm, task, seed, directory, selected_gpu, caption=None):
    import torch

    import bridge
    from nomad_policy import NomadPolicy

    checkpoint_name, kind = run_eval.parse_arm(arm)
    spec = checkpoints.load_goal_arm(checkpoint_name)
    policy = NomadPolicy(spec, torch.device("cuda"), config.driver)
    goal_for = run_eval.goal_provider(kind, spec, policy.clip)
    config.recording.enabled = True
    config.recording.directory = directory
    body = bridge.SimBody(config_path=task.world_config, floor=config.tasks.floor)
    try:
        body.verify_gpu(selected_gpu)
        house = objects.SceneObjects.for_scene(body.scene)
        film = recorder.ObjectGoalRecorder(
            config.recording, waypoint_scale_m=bridge.waypoint_scale_m(
                policy.model_params, body.limits),
            success_radius_m=config.rules.success_radius_m, objects=house,
            goal_for=goal_for, caption=caption)
        runner = episode_runner.ObjectGoalRunner(
            policy, body, house, goal_for, rules=config.rules,
            seed_offset=seed - task.seed)
        table = metrics.MetricsTable(directory / "refilm_{}.csv".format(arm)).open(resume=False)
        run_eval.score_task(task, 0, runner, arm, table, film, verbose=False)
    finally:
        body.close()
    video = film.video_path(arm, task.task_id, seed - task.seed)
    return metrics.read_table(directory / "refilm_{}.csv".format(arm))[-1], video


def rows_match(logged, refilmed):
    """Is a re-filmed row the logged one, column for column?"""
    return all(str(logged[c]) == str(refilmed[c]) for c in MATCHED_COLUMNS)


def quarantine_mismatches(matches):
    """Rename every mismatched film to `*.MISMATCH.mp4`; True if none mismatched.

    Not deleted: a mismatched film is evidence. Renamed, so it cannot pass for
    the scored episode, and the caller stacks nothing unless this is True.
    """
    for match in matches:
        if not match["match"]:
            video = Path(match["video"])
            renamed = video.with_suffix(".MISMATCH.mp4")
            video.rename(renamed)
            match["video"] = str(renamed)
    return all(match["match"] for match in matches)


def stack_videos(paths, out_path):
    """The films one above the other, the shorter ones held on their last frame."""
    import imageio
    import numpy as np

    readers = [imageio.get_reader(str(p)) for p in paths]
    frames = [[frame for frame in reader] for reader in readers]
    fps = readers[0].get_meta_data()["fps"]
    length = max(len(f) for f in frames)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=7,
                                macro_block_size=16, pixelformat="yuv420p")
    try:
        for index in range(length):
            writer.append_data(np.concatenate(
                [f[min(index, len(f) - 1)] for f in frames], axis=0))
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    return length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="a new folder for the films")
    parser.add_argument("--label", default="", help="prefixed to the caption")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", default=None, help="task id; default: best V3+word success")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    selected_gpu = gpu.select_gpu(args.gpu)

    config = run_eval.ObjectEvalConfig.from_dict(run_eval.read_layered_yaml(args.config))
    run = run_eval.read_run_manifest(args.run_dir)
    if config.driver.label() != run["driver"] or config.rules.as_dict() != run["rules"]:
        raise SystemExit("FAILED: {}'s driver or rules differ from the run's".format(
            args.config))
    task_directory = Path(run["task_set"]["directory"])
    manifest, tasks = object_tasks.load(task_directory)
    if manifest["fingerprint"] != run["task_set"]["fingerprint"]:
        raise SystemExit("FAILED: the task set at {} is not the one the run scored "
                         "(fingerprint {} vs {})".format(task_directory,
                                                         manifest["fingerprint"],
                                                         run["task_set"]["fingerprint"]))
    config.task_directory = task_directory
    config.arms = [arm for arm in config.arms if arm in run["arms"]]
    if args.task:
        task_id, seed = args.task, args.seed
        selection = {"rule": "chosen by hand (--task/--seed)", "chosen": [task_id, seed],
                     "caption": "selected example: chosen by hand"}
    else:
        task_id, seed, selection = pick_clip(args.run_dir)
    caption = " · ".join(part for part in (args.label, selection["caption"]) if part)
    selection["caption"] = caption
    task = next(t for t in tasks if t.task_id == task_id)
    clips = args.output_dir
    if clips.exists():
        raise SystemExit("FAILED: {} exists; films are written to a fresh folder".format(clips))
    clips.mkdir(parents=True)
    import json
    (clips / "selection.json").write_text(json.dumps(dict(selection, run_dir=str(args.run_dir)),
                                                     indent=2))
    print("selection:  {}".format(caption))
    print("clip:       {} seed {} (offset {})".format(task_id, seed, seed - task.seed))

    # Film order is the stacking order: the word, no goal, then the photo reference.
    order = [HEADLINE_ARM, "clip_v2b+masked", "ctx03+photo"]
    matches, videos = [], []
    for arm in [a for a in order if a in config.arms]:
        logged = next(r for r in metrics.read_table(args.run_dir / "{}.csv".format(arm))
                      if r["task_id"] == task_id and int(r["seed"]) == seed)
        refilmed, video = film_arm(config, arm, task, seed, clips, selected_gpu, caption)
        same = rows_match(logged, refilmed)
        matches.append({"arm": arm, "task_id": task_id, "seed": seed, "match": int(same),
                        **{"logged_" + c: logged[c] for c in MATCHED_COLUMNS},
                        **{"refilmed_" + c: refilmed[c] for c in MATCHED_COLUMNS},
                        "video": str(video)})
        videos.append(video)
        differ = [c for c in MATCHED_COLUMNS if str(logged[c]) != str(refilmed[c])]
        print("{:<16} {} -> {}".format(arm, "all {} columns equal".format(len(MATCHED_COLUMNS))
                                       if same else "differs in {}".format(differ),
                                       "MATCH" if same else "MISMATCH"))

    passed = quarantine_mismatches(matches)
    matched = sum(m["match"] for m in matches)
    with open(clips / "match.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matches[0]))
        writer.writeheader()
        writer.writerows(matches)
    print("re-film match-check: {}/{} {}".format(
        matched, len(matches), "PASSED" if passed else "FAILED"))
    if not passed:
        raise SystemExit("FAILED: no stacked clip; mismatched films renamed *.MISMATCH.mp4")
    stacked = clips / "{}+{}_3arms.mp4".format(task_id, seed - task.seed)
    frames = stack_videos(videos, stacked)
    print("stacked:    {} ({} frames)".format(stacked, frames))


if __name__ == "__main__":
    main()
