"""Language-goal G7.0: the harness still scores what it scored on `sim-eval`.

`sim_eval/` was brought onto `clip-language-goal` from 506141a, a branch whose
model code is older than this one. Before any new mode is added, one episode
the old branch scored is run again here, through `run_eval`'s own code path,
and its row is compared with the stored one byte for byte.

The episode: ctx03 (the language task's V1), task Rs_00 of the P6 task set,
seed offset 0, the P7 E1 config it was scored under. The stored row is the
first Rs_00 row of `legacy/sim_eval/outputs/p7_2_e1_metrics/ctx03.csv`.

    ./sim_eval/run_lg7_0_port_check.sh
"""

import argparse
import csv
from pathlib import Path

import gpu
import metrics
import run_eval
import task_set

SIM_EVAL_DIR = Path(__file__).resolve().parent
LEGACY_CSV = Path("/outputs/legacy/sim_eval/outputs/p7_2_e1_metrics/ctx03.csv")
DEFAULT_CONFIG = SIM_EVAL_DIR / "configs" / "p7_2_e1.yaml"
DEFAULT_OUTPUT = SIM_EVAL_DIR / "outputs" / "lg7_0_port_check" / "ctx03.csv"
CHECKPOINT = "ctx03"
TASK_ID = "Rs_00"
SEED_OFFSET = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-set", type=Path,
                        default=SIM_EVAL_DIR / "outputs" / "p6_0_task_set",
                        help="a copy of the P6 task set (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def raw_rows(path):
    """The CSV's header and data lines as written, so the comparison is on bytes."""
    lines = Path(path).read_text().splitlines()
    return lines[0], lines[1:]


def stored_row(seed):
    header, rows = raw_rows(LEGACY_CSV)
    for line in rows:
        fields = next(csv.reader([line]))
        if fields[0] == CHECKPOINT and fields[2] == TASK_ID and int(fields[3]) == seed:
            return header, line
    raise SystemExit("FAILED: no {} {} seed {} row in {}".format(
        CHECKPOINT, TASK_ID, seed, LEGACY_CSV))


def main():
    args = parse_args()
    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {}".format(selected_gpu))

    config = run_eval.EvalConfig.from_yaml(DEFAULT_CONFIG)
    config.seed_offsets = [SEED_OFFSET]
    config.recording.enabled = False
    _manifest, tasks = task_set.ensure(config.tasks, args.task_set, open_body=None)
    tasks = [task for task in tasks if task.task_id == TASK_ID]

    if args.output.exists():
        raise SystemExit("FAILED: {} exists; G7.0 writes a fresh table".format(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    import checkpoints
    from nomad_policy import NomadPolicy

    spec = checkpoints.load(CHECKPOINT)
    print("checkpoint: {}".format(spec.summary()))
    policy = NomadPolicy(spec, torch.device("cuda"), config.driver)
    table = metrics.MetricsTable(args.output).open(resume=False)
    crashed = run_eval.score_checkpoint(
        config, tasks, policy, CHECKPOINT, table, selected_gpu,
        run_eval.build_recorder(config, policy), verbose=False)
    if crashed:
        raise SystemExit("FAILED: episode crashed: {}".format(crashed))

    header, rows = raw_rows(args.output)
    assert len(rows) == 1, rows
    stored_header, stored = stored_row(tasks[0].seed + SEED_OFFSET)
    print("\nstored: {}\nnow:    {}".format(stored, rows[0]))
    if header != stored_header or rows[0] != stored:
        raise SystemExit("G7.0 FAILED: the row differs from the stored one")
    print("G7.0 PASSED: byte-identical row")


if __name__ == "__main__":
    main()
