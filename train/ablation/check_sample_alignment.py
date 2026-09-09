"""Prove that runs with different input recipes are scored on the same samples.

`eval_paired.py` compares runs *pairwise*, per sample. That is only meaningful if two
runs scored side by side actually see the same samples: the same trajectory, the same
current timestep, the same sampled goal, the same negatives. Each run is fed at its own
`image_size`, `context_size` and `context_stride`, so the pixels legitimately differ --
but everything that defines *which* sample it is must not.

Holding `index_context_size` fixed is what is supposed to guarantee that. This check
tests the guarantee rather than assuming it, by comparing the parts of a batch that carry
sample identity and the prediction target:

    actions, distance, goal_pos, action_mask

If any of those differ between two configs, the paired deltas in the results table are
comparing different questions and are not interpretable.

Run inside the container, from `/app/visualnav-transformer/train`:
    python ablation/check_sample_alignment.py \
        --config vanilla=config/nomad.yaml \
        --config stride3=config/nomad_stride3.yaml
"""

import argparse

import numpy as np

from eval_paired import ALIGNED_INDEX_CONTEXT_SIZE, build_test_loader, load_arm_config

# Everything a batch carries except the images, which are meant to differ.
IDENTITY_FIELDS = {"actions": 2, "distance": 3, "goal_pos": 4, "action_mask": 6}


def collect_identity(config, batch_size, batches, seed):
    """The identity fields of the first `batches` batches, under this config's recipe."""
    dataset, loader = build_test_loader(config, batch_size)
    try:
        # The dataset draws goals and negatives from numpy at __getitem__ time, so the
        # seed has to be pinned here, exactly as the scorer does it.
        np.random.seed(seed)
        collected = {name: [] for name in IDENTITY_FIELDS}
        for batch_index, data in enumerate(loader):
            if batch_index >= batches:
                break
            for name, position in IDENTITY_FIELDS.items():
                collected[name].append(data[position].numpy())
        return len(dataset), {
            name: np.concatenate(values) for name, values in collected.items()
        }
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="NAME=CONFIG_PATH",
        help="Repeatable; the first one is the reference the rest are compared against",
    )
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index-context-size", type=int, default=ALIGNED_INDEX_CONTEXT_SIZE)
    args = parser.parse_args()

    arms = [entry.split("=", 1) for entry in args.config]
    reference_name = arms[0][0]

    sizes, identities, recipes = {}, {}, {}
    for name, config_path in arms:
        config = load_arm_config(config_path, args.index_context_size)
        recipes[name] = (
            f"context_size={config['context_size']}, "
            f"context_stride={config['context_stride']}, "
            f"image_size={config['image_size'][0]}x{config['image_size'][1]}"
        )
        sizes[name], identities[name] = collect_identity(
            config, args.batch_size, args.batches, args.seed
        )

    print()
    for name, _ in arms:
        print(f"{name:<16} {sizes[name]:>8,} samples   {recipes[name]}")
    print()

    failures = []
    if len(set(sizes.values())) != 1:
        failures.append(f"sample counts differ across configs: {sizes}")

    header = f"{'arm':<16}" + "".join(f"{field:>16}" for field in IDENTITY_FIELDS)
    print(f"max |difference| vs `{reference_name}` over "
          f"{len(identities[reference_name]['actions']):,} samples")
    print(header)
    print("-" * len(header))
    for name, _ in arms[1:]:
        cells = []
        for field in IDENTITY_FIELDS:
            error = float(
                np.abs(identities[name][field] - identities[reference_name][field]).max()
            )
            cells.append(f"{error:>16.3e}")
            if error != 0.0:
                failures.append(f"{name}: {field} differs from {reference_name}")
        print(f"{name:<16}" + "".join(cells))

    print()
    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print(
        f"PASS: every config sees the same {sizes[reference_name]:,} samples, with "
        "identical goals, negatives and prediction targets."
    )


if __name__ == "__main__":
    main()
