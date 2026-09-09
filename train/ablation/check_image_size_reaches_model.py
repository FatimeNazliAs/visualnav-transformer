"""Prove that a config's `image_size` reaches the model, not just the dataset.

`image_size` is a dataset-side knob: nothing in `NoMaD_ViNT.__init__` takes it, and both
EfficientNet encoders end in `AdaptiveAvgPool2d(1)`, so every resolution collapses to the
same 1280 features and produces identically-shaped weights. That is exactly what makes a
silent failure possible -- a resize that never happened, or a transform that squashed the
frames back to 96x96, would leave the parameter count, the conditioning width and the loss
curve all looking completely normal. A 19-hour run would finish and mean nothing.

So this does not check shapes on paper. It pulls a real batch from the real train split,
runs a real forward pass, and hooks the input of each encoder's average-pooling layer --
the last point at which resolution still exists. If the pixels are genuinely bigger, the
pre-pool feature grid is bigger. If two configs at different resolutions produce the same
grid, the resolution is not reaching the convolutions and the run is worthless.

Run inside the container, from `/app/visualnav-transformer/train`:
    CUDA_VISIBLE_DEVICES=0 python ablation/check_image_size_reaches_model.py \
        --config stock=config/nomad_ctx03.yaml \
        --config best_combined=config/nomad_best_combined_100ep.yaml
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_paired import IMAGENET_TRANSFORM, build_model, load_arm_config
from vint_train.data.vint_dataset import ViNT_Dataset


def build_train_loader(config, batch_size):
    """The train split, exactly as `train.py` builds it -- this is what the run reads."""
    data_config = config["datasets"]["go_stanford"]
    dataset = ViNT_Dataset(
        data_folder=data_config["data_folder"],
        data_split_folder=data_config["train"],
        dataset_name="go_stanford",
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
        context_stride=config["context_stride"],
        index_context_size=config["index_context_size"],
        end_slack=data_config["end_slack"],
        goals_per_obs=data_config["goals_per_obs"],
        normalize=config["normalize"],
        goal_type=config["goal_type"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataset, loader


def trace_forward(config, batch_size, device):
    """Real batch, real forward pass; returns the shapes resolution actually controls."""
    dataset, loader = build_train_loader(config, batch_size)
    try:
        np.random.seed(0)
        obs_image, goal_image, actions, _, _, _, _ = next(iter(loader))

        model = build_model(config, device)
        grids = {}

        def record(name):
            # The input of _avg_pooling is the last tensor that still carries spatial
            # extent; everything after it is resolution-blind by construction.
            def hook(_module, inputs, _output):
                grids[name] = tuple(inputs[0].shape[-2:])
            return hook

        handles = [
            model.vision_encoder.obs_encoder._avg_pooling.register_forward_hook(
                record("obs_encoder")
            ),
            model.vision_encoder.goal_encoder._avg_pooling.register_forward_hook(
                record("goal_encoder")
            ),
        ]
        try:
            batch_obs = torch.cat(
                [IMAGENET_TRANSFORM(part) for part in torch.split(obs_image, 3, dim=1)],
                dim=1,
            ).to(device)
            batch_goal = IMAGENET_TRANSFORM(goal_image).to(device)
            with torch.no_grad():
                cond = model(
                    "vision_encoder",
                    obs_img=batch_obs,
                    goal_img=batch_goal,
                    input_goal_mask=torch.zeros(
                        batch_obs.shape[0], dtype=torch.long, device=device
                    ),
                )
        finally:
            for handle in handles:
                handle.remove()

        return {
            "configured_image_size": tuple(config["image_size"]),
            "obs_batch": tuple(batch_obs.shape),
            "goal_batch": tuple(batch_goal.shape),
            "obs_grid": grids["obs_encoder"],
            "goal_grid": grids["goal_encoder"],
            "conditioning": tuple(cond.shape),
            "train_samples": len(dataset),
            "action_horizon": tuple(actions.shape),
        }
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", action="append", required=True, metavar="NAME=CONFIG_PATH"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms = [entry.split("=", 1) for entry in args.config]

    traces, failures = {}, []
    for name, config_path in arms:
        config = load_arm_config(config_path)
        trace = trace_forward(config, args.batch_size, device)
        traces[name] = trace

        width, height = trace["configured_image_size"]
        tensor_height, tensor_width = trace["obs_batch"][-2:]
        if (tensor_width, tensor_height) != (width, height):
            failures.append(
                f"{name}: config says {width}x{height} (WxH) but the tensor entering the "
                f"model is {tensor_width}x{tensor_height}"
            )
        expected_channels = 3 * (config["context_size"] + 1)
        if trace["obs_batch"][1] != expected_channels:
            failures.append(
                f"{name}: obs tensor has {trace['obs_batch'][1]} channels, expected "
                f"{expected_channels} for context_size {config['context_size']}"
            )

        print()
        print(f"=== {name}  ({config_path}) ===")
        print(f"  configured image_size (WxH)     {width}x{height}")
        print(f"  obs tensor into model  (B,C,H,W) {trace['obs_batch']}")
        print(f"  goal tensor into model (B,C,H,W) {trace['goal_batch']}")
        print(f"  obs encoder pre-pool grid  (HxW) {trace['obs_grid'][0]}x{trace['obs_grid'][1]}")
        print(f"  goal encoder pre-pool grid (HxW) {trace['goal_grid'][0]}x{trace['goal_grid'][1]}")
        print(f"  conditioning out of encoder      {trace['conditioning']}")
        print(f"  action target                    {trace['action_horizon']}")
        print(f"  train samples indexed            {trace['train_samples']:,}")

    grids = {name: traces[name]["obs_grid"] for name, _ in arms}
    sizes = {name: traces[name]["configured_image_size"] for name, _ in arms}
    if len(set(sizes.values())) > 1 and len(set(grids.values())) == 1:
        failures.append(
            f"configs differ in image_size {sizes} but produce the same pre-pool grid "
            f"{grids}; the resolution is not reaching the convolutions"
        )

    print()
    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print(
        "PASS: every config's pixels reach the convolutions at the configured "
        "resolution, and differing resolutions produce differing feature grids."
    )


if __name__ == "__main__":
    main()
