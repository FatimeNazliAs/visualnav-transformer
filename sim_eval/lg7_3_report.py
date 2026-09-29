"""Language-goal Phase 7: the fairness gate, then the success counts.

Reads the arms' tables from one run folder and refuses to report anything
until `comparison.fairness_problems` passes: every arm ran exactly the word
task set under the config's seed offsets, no episode twice, the same driver
and success rule everywhere. Only then does it count.

    ./sim_eval/run_lg7.sh lg7_3_report.py --run-dir /outputs/nomad_clip_sim/pilot_<ts>

The task set, seed offsets and arms are the run's own, from its
`run_manifest.json` (run_eval.record_arm), and the task set on disk must
still carry the fingerprint the run was scored on.

Writes into the run folder: fairness.txt, success_counts.csv / .md,
paired_word_minus_masked.csv / .md. The counts are a pilot's: rough, few
tasks, labelled as such. Task-level paired statistics are the full run's
(plan, Phase 7 Verify).

The readouts are the ones pre-registered before G7.2
(sim_eval/outputs/p7_1b_controls/G7_1_VERDICT.md) and no others:

  1. success count per arm and word;
  2. mean minimum distance reached (trace `min_geodesic_distance_m`);
  3. mean ticks to success (successful episodes only);
  4. per-task paired counts, word - masked: each task's word-arm successes
     minus its masked-arm successes over the seed offsets.

`--label` (e.g. "G7.2 not passed") heads every file written.
"""

import argparse
import csv
import json
from pathlib import Path

import comparison
import metrics
import object_tasks
import run_eval

SIM_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SIM_EVAL_DIR / "configs" / "word_goal.yaml"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def rounded(value, places=3):
    return None if value is None else round(value, places)


def min_distances(run_dir, arm):
    """{(task_id, seed): closest the episode got to the word}, from the arm's traces."""
    path = run_dir / "{}.jsonl".format(arm)
    found = {}
    with open(path) as handle:
        for line in handle:
            trace = json.loads(line)
            found[(trace["task_id"], int(trace["seed"]))] = trace.get("min_geodesic_distance_m")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--arms", default=None, help="comma-separated; default: the config's")
    parser.add_argument("--word-arm", default="clip_v2b+word")
    parser.add_argument("--masked-arm", default="clip_v2b+masked")
    parser.add_argument("--label", default="", help="heads every file, e.g. 'G7.2 not passed'")
    args = parser.parse_args()
    head = ["LABEL: {}".format(args.label)] if args.label else []

    config = run_eval.ObjectEvalConfig.from_dict(run_eval.read_layered_yaml(args.config))
    run = run_eval.read_run_manifest(args.run_dir)
    arms = args.arms.split(",") if args.arms else config.arms
    unknown = [arm for arm in arms if arm not in run["arms"]]
    if unknown:
        raise SystemExit("FAILED: {} did not run {}".format(args.run_dir, ", ".join(unknown)))
    task_directory = Path(run["task_set"]["directory"])
    _manifest, tasks = object_tasks.load(task_directory)
    if _manifest["fingerprint"] != run["task_set"]["fingerprint"]:
        raise SystemExit("FAILED: the task set at {} is not the one the run scored "
                         "(fingerprint {} vs {})".format(task_directory, _manifest["fingerprint"],
                                                         run["task_set"]["fingerprint"]))
    config.task_directory = task_directory
    config.seed_offsets = run["seed_offsets"]
    expected = comparison.expected_episodes(tasks, config.seed_offsets)
    tables = {arm: metrics.read_table(args.run_dir / "{}.csv".format(arm)) for arm in arms}

    problems = comparison.fairness_problems(tables, expected)
    lines = head + ["fairness gate: {}".format("PASSED" if not problems else "FAILED"),
             "arms: {}".format(", ".join(arms)),
             "task set: {} ({} tasks, fingerprint {})".format(
                 config.task_directory, len(tasks), _manifest["fingerprint"]),
             "seed offsets: {} -> {} episodes per arm".format(config.seed_offsets,
                                                               len(expected))]
    lines += ["  " + problem for problem in problems]
    (args.run_dir / "fairness.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if problems:
        raise SystemExit("FAILED: not a fair comparison; no counts reported")

    closest = {arm: min_distances(args.run_dir, arm) for arm in arms}
    words = list(dict.fromkeys(task.word for task in tasks))
    word_of = {task.task_id: task.word for task in tasks}
    rows = []
    for arm in arms:
        for word in words + ["all"]:
            episodes = [r for r in tables[arm] if word == "all" or word_of[r["task_id"]] == word]
            successes = sum(int(r["success"]) for r in episodes)
            rows.append({
                "arm": arm, "word": word, "episodes": len(episodes),
                "successes": successes,
                "success_rate": round(successes / len(episodes), 3),
                "mean_min_distance_m": rounded(mean([
                    closest[arm].get((r["task_id"], int(r["seed"]))) for r in episodes])),
                "mean_ticks_to_success": rounded(mean([
                    float(r["ticks"]) for r in episodes if int(r["success"])]), 1),
            })
    write_table(args.run_dir, "success_counts", rows, head,
                "Phase 7 pilot — success counts (pilot: few tasks, not a paired test)")

    # 4. per task, word minus masked, over the seed offsets.
    if args.word_arm in tables and args.masked_arm in tables:
        paired = []
        for task in tasks:
            count = {arm: sum(int(r["success"]) for r in tables[arm]
                              if r["task_id"] == task.task_id)
                     for arm in (args.word_arm, args.masked_arm)}
            paired.append({"task_id": task.task_id, "word": task.word,
                           "view": "in" if task.entry.get("target_in_view") else "out",
                           "word_successes": count[args.word_arm],
                           "masked_successes": count[args.masked_arm],
                           "word_minus_masked": count[args.word_arm] - count[args.masked_arm],
                           "of": len(config.seed_offsets)})
        write_table(args.run_dir, "paired_word_minus_masked", paired, head,
                    "Phase 7 pilot — per task, {} minus {} successes (pilot)".format(
                        args.word_arm, args.masked_arm))


def write_table(directory, name, rows, head, title):
    columns = list(rows[0])
    with open(directory / "{}.csv".format(name), "w", newline="") as handle:
        for line in head:
            handle.write("# {}\n".format(line))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    md = ["# " + title, ""] + ["**{}**".format(line) for line in head] + ([""] if head else [])
    md += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    md += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    (directory / "{}.md".format(name)).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
