"""Re-film exact scored episodes — only those, and prove they are the same.

A scoring run films nothing by default (P4: filming costs ~0.09 s a tick), so
the P7 clips are made afterwards: each listed (arm, task, seed) episode is run
again under the arm's own scoring config — its task set, rules, driver and
seed — with recording on. Nothing else is run. The episode is deterministic in
its seed, so the film is the scored episode; this checks that rather than
assuming it. Every re-film's row is compared with the row the scoring run
logged, on success, ticks and path length, and any difference is reported as
a mismatch — the clip stays on disk, but it is named as not the scored episode.

Each frame is captioned with the task's success across all its seeds, per arm,
read from the scored tables ("Denmark_14 — task success over all seeds:
best_combined 0/3, clean_stock 3/3"), so one episode is never read as the task.

    ./sim_eval/run_p7_4_film.sh [--config sim_eval/configs/p7_4_clips.yaml]

Writes under the clips config's `directory`:

    <arm>/<task>[+<offset>].mp4   one film per (arm, task, seed), named as P4 names them
    refilm_<arm>.csv / .jsonl     the re-filmed episodes' rows and traces
    p7_4_match.csv                each clip: logged vs re-filmed, and whether they match
"""

import argparse
from pathlib import Path

import yaml

import checkpoints
import comparison
import episode_runner
import gpu
import metrics
import run_eval
import task_set
import task_stats
from recorder import RecordingConfig

DEFAULT_CONFIG = run_eval.SIM_EVAL_DIR / "configs" / "p7_4_clips.yaml"
MATCH_CSV = "p7_4_match.csv"

# What a re-film must reproduce to be the scored episode.
MATCHED_COLUMNS = ("success", "ticks", "path_length_m")


class Clip:
    """One task under one seed offset, and the arms that film it."""

    def __init__(self, task, seed_offset, arms):
        self.task_id = task
        self.seed_offset = int(seed_offset)
        # {arm: path of the arm's scoring config}, in caption order.
        self.arms = {arm: run_eval.resolve_path(Path("configs") / config)
                     for arm, config in arms.items()}


def read_clips(path):
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    return (run_eval.resolve_path(data["directory"]), int(data.get("fps", 10)),
            [Clip(**clip) for clip in data["clips"]])


def task_success(eval_config, arm, task_id):
    """'2/3' — the arm's successes on the task over every seed it was scored on."""
    rows = [row for row in metrics.read_table(eval_config.csv_path(arm))
            if row["task_id"] == task_id]
    seeds = task_stats.seed_counts(rows)[task_id]
    return "{}/{}".format(sum(int(row["success"]) for row in rows), seeds)


def caption(clip, configs):
    return "{} — task success over all seeds: {}".format(
        clip.task_id, ", ".join("{} {}".format(arm, task_success(configs[arm], arm,
                                                                  clip.task_id))
                                for arm in clip.arms))


def logged_row(eval_config, arm, task, seed):
    for row in metrics.read_table(eval_config.csv_path(arm)):
        if row["task_id"] == task.task_id and int(row["seed"]) == seed:
            return row
    raise LookupError("{} has no scored row for {} seed {} in {}".format(
        arm, task.task_id, seed, eval_config.csv_path(arm)))


def find_task(eval_config, task_id):
    _manifest, tasks = task_set.load(eval_config.task_directory)
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise LookupError("no task {} in {}".format(task_id, eval_config.task_directory))


def film_arm(arm, episodes, directory, fps, selected_gpu):
    """Film every (clip, config) of one arm with one loaded policy.

    Returns the re-filmed rows, keyed by (task_id, seed)."""
    import bridge
    import torch

    from nomad_policy import NomadPolicy

    first_config = episodes[0][1]
    spec = checkpoints.load(arm)
    print("\n=== {} ===".format(spec.summary()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = NomadPolicy(spec, device, first_config.driver)
    table = metrics.MetricsTable(directory / "refilm_{}.csv".format(arm)).open()

    captions = {}
    for clip, eval_config, text in episodes:
        captions[(clip.task_id, clip.seed_offset)] = text
        # Every config an arm films under must steer it the same way, or the
        # one policy loaded here would drive some clips wrongly.
        if eval_config.driver.summary() != first_config.driver.summary():
            raise ValueError("{}'s clip configs disagree on the driver".format(arm))

    for index, (clip, eval_config, _text) in enumerate(episodes):
        task = find_task(eval_config, clip.task_id)
        eval_config.recording = RecordingConfig(enabled=True, fps=fps,
                                                directory=directory)
        film = run_eval.build_recorder(
            eval_config, policy,
            caption_for=lambda task, _arm, offset: captions[(task.task_id, offset)])
        body = bridge.SimBody(config_path=task.world_config, floor=eval_config.floor)
        try:
            body.verify_gpu(selected_gpu)
            runner = episode_runner.EpisodeRunner(
                policy, body, rules=eval_config.rules, seed_offset=clip.seed_offset)
            run_eval.score_task(task, index, runner, arm, table, film, verbose=False)
        finally:
            body.close()
    return {(row["task_id"], int(row["seed"])): row
            for row in metrics.read_table(table.csv_path)}


def match_rows(episodes_by_arm, refilmed):
    """One row per filmed episode: the logged and re-filmed values, and a verdict."""
    rows = []
    for arm, episodes in episodes_by_arm.items():
        for clip, eval_config, _text in episodes:
            task = find_task(eval_config, clip.task_id)
            seed = episode_runner.episode_seed(task, clip.seed_offset)
            logged = logged_row(eval_config, arm, task, seed)
            again = refilmed[arm][(task.task_id, seed)]
            row = {"arm": arm, "task_id": task.task_id, "seed": seed}
            for column in MATCHED_COLUMNS:
                row["logged_" + column] = logged[column]
                row["refilm_" + column] = again[column]
            row["match"] = int(all(logged[column] == again[column]
                                   for column in MATCHED_COLUMNS))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to use; never guesses")
    args = parser.parse_args()

    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))
    directory, fps, clips = read_clips(args.config)

    episodes_by_arm = {}
    for clip in clips:
        configs = {arm: run_eval.EvalConfig.from_yaml(path)
                   for arm, path in clip.arms.items()}
        text = caption(clip, configs)
        for arm, eval_config in configs.items():
            episodes_by_arm.setdefault(arm, []).append((clip, eval_config, text))

    refilmed = {arm: film_arm(arm, episodes, directory, fps, selected_gpu)
                for arm, episodes in episodes_by_arm.items()}

    rows = match_rows(episodes_by_arm, refilmed)
    columns = (("arm", "task_id", "seed")
               + tuple(prefix + column for column in MATCHED_COLUMNS
                       for prefix in ("logged_", "refilm_"))
               + ("match",))
    comparison.write_csv(directory / MATCH_CSV, rows, columns)
    print("\n{} of {} re-films match their logged episode".format(
        sum(row["match"] for row in rows), len(rows)))
    for row in rows:
        if not row["match"]:
            print("MISMATCH:   {arm} {task_id} seed {seed}".format(**row))
    print("wrote:      {}".format(directory / MATCH_CSV))


if __name__ == "__main__":
    try:
        main()
    except (gpu.GpuSelectionError, checkpoints.CheckpointError,
            task_set.TaskSetError, LookupError, ValueError) as error:
        raise SystemExit("FAILED: {}".format(error))
