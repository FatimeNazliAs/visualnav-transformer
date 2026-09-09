"""Prove that `eval_paired.py` scores a checkpoint identically every time.

The capstone's headline claim is a *paired* difference between two runs, often only a few
hundredths of a loss unit. That claim is only readable if the scorer itself contributes no
run-to-run noise: any wobble in the eval would be indistinguishable from a real gap. So
before anything is scored for the record, the scorer is checked against itself.

The two passes run as **separate processes**, not twice in one. An in-process repeat would
share a warmed-up cuDNN algorithm cache and a single CUDA context, and would therefore pass
even if algorithm selection were still being made at runtime -- exactly the failure
`enforce_determinism` exists to rule out.

Run inside the container, from `/app/visualnav-transformer/train`:
    CUDA_VISIBLE_DEVICES=0 python ablation/check_eval_determinism.py \
        --config config/nomad.yaml \
        --checkpoint /outputs/nomad/nomad_2026_06_13_18_04_23/ema_99.pth
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from eval_paired import METRICS

EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_paired.py")


def score_once(out_dir, config, checkpoint, batches, batch_size, seed):
    subprocess.run(
        [
            sys.executable,
            EVAL_SCRIPT,
            "--arm", f"repeat={config}={checkpoint}",
            "--out", out_dir,
            "--max-batches", str(batches),
            "--batch-size", str(batch_size),
            "--seed", str(seed),
        ],
        check=True,
    )
    return dict(np.load(os.path.join(out_dir, "repeat.npz")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the run's training config")
    parser.add_argument("--checkpoint", required=True, help="the .pth to score twice")
    parser.add_argument(
        "--batches",
        type=int,
        default=20,
        help="batches scored per pass; enough to exercise every code path, not the whole split",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    workspace = tempfile.mkdtemp(prefix="eval_determinism_")
    try:
        first = score_once(
            os.path.join(workspace, "pass1"), args.config, args.checkpoint,
            args.batches, args.batch_size, args.seed,
        )
        second = score_once(
            os.path.join(workspace, "pass2"), args.config, args.checkpoint,
            args.batches, args.batch_size, args.seed,
        )
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    n_samples = len(first["gc_action_loss"])
    print()
    print(f"checkpoint: {args.checkpoint}")
    print(f"config:     {args.config}")
    print(f"samples compared per pass: {n_samples:,}")
    print()
    header = f"{'metric':<28}{'pass 1 mean':>16}{'pass 2 mean':>16}{'max |per-sample d|':>22}"
    print(header)
    print("-" * len(header))

    worst_metric, worst_error = None, 0.0
    exact = True
    for metric in METRICS:
        difference = np.abs(first[metric] - second[metric])
        error = float(difference.max())
        exact = exact and error == 0.0
        if error > worst_error:
            worst_metric, worst_error = metric, error
        print(
            f"{metric:<28}{first[metric].mean():>16.9f}"
            f"{second[metric].mean():>16.9f}{error:>22.3e}"
        )

    print()
    if worst_error > args.tolerance:
        raise SystemExit(
            f"FAIL: {worst_metric} differs by {worst_error:.3e} between two passes, "
            f"above the {args.tolerance:.0e} tolerance."
        )
    if exact:
        print(f"PASS: both passes are bit-identical on all {len(METRICS)} metrics.")
    else:
        print(
            f"PASS: worst per-sample difference {worst_error:.3e} "
            f"(at {worst_metric}) is within {args.tolerance:.0e}."
        )


if __name__ == "__main__":
    main()
