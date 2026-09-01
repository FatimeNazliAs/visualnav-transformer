"""Proof that `context_stride` moves the context window and nothing else.

Ablation A2 rests on one claim: widening the stride changes *which frames the model
looks at* while leaving *what it is asked to predict* bit-identical. That claim is easy
to state and easy to get wrong, because in the stock dataloader a single variable
(`waypoint_spacing`) drives both. This script demonstrates it on real samples instead of
asserting it in a comment.

For a handful of fixed sample indices it builds the dataset at each stride and reports:

  * stride 1 vs the pre-A2 code   -> must select exactly the frames it used to
  * the context frame timesteps   -> must spread as stride grows
  * the action target timesteps   -> must not move at all
  * the returned action tensor    -> must be bit-identical across strides
  * the sample index itself       -> must be the same (trajectory, curr_time) set

`np.random` is reseeded before every `__getitem__` so goal sampling is identical across
strides; without that the sampled goal, and therefore the action tensor, would differ for
reasons that have nothing to do with stride.

Run inside the container, from the `train` directory:
    python ablation/probe_context_stride.py
"""

import argparse
import os

import numpy as np
import yaml

from vint_train.data.vint_dataset import ViNT_Dataset

GOAL_SAMPLING_SEED = 0


def load_config(config_path):
    config_dir = os.path.dirname(config_path)
    with open(os.path.join(config_dir, "defaults.yaml")) as f:
        config = yaml.safe_load(f)
    with open(config_path) as f:
        config.update(yaml.safe_load(f))
    return config


def build_dataset(config, dataset_name, split, context_stride):
    data_config = config["datasets"][dataset_name]
    return ViNT_Dataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config[split],
        dataset_name=dataset_name,
        image_size=config["image_size"],
        waypoint_spacing=data_config.get("waypoint_spacing", 1),
        min_dist_cat=config["distance"]["min_dist_cat"],
        max_dist_cat=config["distance"]["max_dist_cat"],
        min_action_distance=config["action"]["min_dist_cat"],
        max_action_distance=config["action"]["max_dist_cat"],
        negative_mining=data_config["negative_mining"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        context_size=config["context_size"],
        context_type=config["context_type"],
        context_stride=context_stride,
        index_context_size=config["index_context_size"],
        end_slack=data_config["end_slack"],
        goals_per_obs=data_config["goals_per_obs"],
        normalize=config["normalize"],
        goal_type=config["goal_type"],
    )


def stock_context_times(curr_time, context_size, waypoint_spacing):
    """Context frame selection exactly as it stood before context_stride existed.

    Frozen here verbatim from the pre-A2 dataloader, so that the stride-1 default can be
    *shown* to reproduce the baseline rather than merely claimed to.
    """
    return list(
        range(
            curr_time + -context_size * waypoint_spacing,
            curr_time + 1,
            waypoint_spacing,
        )
    )


def report_stride_one_matches_stock(dataset):
    """At stride 1 the new code must pick the same frames the old code picked.

    Checked over every sample in the split, not a spot check: this is what makes A1's
    ctx03 run a legitimate stride-1 baseline instead of a near-enough one.
    """
    mismatches = [
        curr_time
        for _, curr_time, _ in dataset.index_to_data
        if dataset._context_times(curr_time)
        != stock_context_times(curr_time, dataset.context_size, dataset.waypoint_spacing)
    ]
    print(
        f"stride 1 vs pre-A2 formula: {len(dataset.index_to_data)} samples checked, "
        f"{len(mismatches)} mismatches"
    )
    assert not mismatches, f"stride 1 diverged from the baseline at {mismatches[:5]}"


def actions_for(dataset, sample_index):
    """The action tensor for one sample, with goal sampling pinned to a fixed seed."""
    np.random.seed(GOAL_SAMPLING_SEED + sample_index)
    return dataset[sample_index][2].numpy()


def target_times(dataset, curr_time):
    target_slice = dataset._action_target_slice(curr_time)
    return list(range(target_slice.start, target_slice.stop, target_slice.step))


def report_sample(datasets, sample_index):
    reference_stride = min(datasets)
    traj_name, curr_time, _ = datasets[reference_stride].index_to_data[sample_index]
    reference_actions = actions_for(datasets[reference_stride], sample_index)

    print(f"\nsample {sample_index}  traj={traj_name}  curr_time={curr_time}")
    for stride, dataset in sorted(datasets.items()):
        context = dataset._context_times(curr_time)
        window = dataset.context_size * stride
        print(
            f"  stride {stride}:  context frames {context}"
            f"   window {window} waypoints  ({len(context)} frames)"
        )

    print(f"  action target frames, every stride: {target_times(datasets[reference_stride], curr_time)}")
    for stride, dataset in sorted(datasets.items()):
        deviation = np.abs(actions_for(dataset, sample_index) - reference_actions).max()
        print(f"  stride {stride}:  max |action - stride{reference_stride} action| = {deviation:.1e}")
        assert deviation == 0.0, f"stride {stride} moved the action targets"


def report_shared_sample_index(datasets):
    reference_stride = min(datasets)
    reference_index = datasets[reference_stride].index_to_data
    print(f"sample index: {len(reference_index)} samples")
    for stride, dataset in sorted(datasets.items()):
        identical = dataset.index_to_data == reference_index
        print(f"  stride {stride}:  same (trajectory, curr_time) set as stride {reference_stride}: {identical}")
        assert identical, f"stride {stride} changed the sample index"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="config/nomad_stride2.yaml")
    parser.add_argument("--dataset", default="go_stanford")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 5000, 12345])
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = {
        stride: build_dataset(config, args.dataset, args.split, stride)
        for stride in args.strides
    }

    print(
        f"\n=== context_stride probe: {args.dataset} {args.split} split, "
        f"context_size={config['context_size']}, "
        f"index_context_size={config['index_context_size']} ===\n"
    )
    report_shared_sample_index(datasets)
    if 1 in datasets:
        report_stride_one_matches_stock(datasets[1])
    for sample_index in args.samples:
        report_sample(datasets, sample_index)

    print(
        "\nStride 1 reproduces the pre-A2 baseline. Context frames spread with stride,"
        "\nwhile action targets and the sample index do not move: the stride knob is"
        "\nindependent of the prediction targets.\n"
    )


if __name__ == "__main__":
    main()
