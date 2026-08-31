"""Confirm the ablation edits do not move the baseline.

Two claims are checked:

1. Model. `context_size` only changes the transformer's token count, so parameter count
   and the conditioning dimension must be identical at 3 / 10 / 20. The positional
   encoding is a sinusoidal buffer, not a parameter.

2. Sample index. The new `index_context_size` argument must (a) reproduce the stock
   index byte-for-byte when left at its default, and (b) produce the *same* index for
   every context_size when pinned, which is what makes the sweep a controlled comparison.

Run inside the container:
    CUDA_VISIBLE_DEVICES=1 python ablation/check_baseline_equivalence.py
"""

import argparse
import os
import pickle

import torch
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

CONTEXT_SIZES = [3, 10, 20]
PINNED_INDEX_CONTEXT_SIZE = 20
ENCODING_SIZE = 256


def build_model(context_size):
    vision_encoder = replace_bn_with_gn(
        NoMaD_ViNT(
            obs_encoding_size=ENCODING_SIZE,
            context_size=context_size,
            mha_num_attention_heads=4,
            mha_num_attention_layers=4,
            mha_ff_dim_factor=4,
        )
    )
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=ENCODING_SIZE,
        down_dims=[64, 128, 256],
        cond_predict_scale=False,
    )
    return NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=DenseNetwork(embedding_dim=ENCODING_SIZE),
    )


def check_model_invariance():
    print("== model invariance across context_size ==")
    reference = None
    for context_size in CONTEXT_SIZES:
        model = build_model(context_size)
        num_params = sum(p.numel() for p in model.parameters())
        obs = torch.zeros(2, 3 * (context_size + 1), 96, 96)
        goal = torch.zeros(2, 3, 96, 96)
        with torch.no_grad():
            cond = model(
                "vision_encoder",
                obs_img=obs,
                goal_img=goal,
                input_goal_mask=torch.zeros(2, dtype=torch.long),
            )
        seq_len = model.vision_encoder.positional_encoding.pos_enc.shape[1]
        print(
            f"  context_size={context_size:2d}  seq_len={seq_len:2d}  "
            f"params={num_params:,}  cond_dim={tuple(cond.shape)}"
        )
        if reference is None:
            reference = (num_params, cond.shape)
        elif (num_params, cond.shape) != reference:
            raise SystemExit("FAIL: model is not invariant to context_size")
        assert seq_len == context_size + 2, seq_len
    print("  PASS: parameter count and conditioning dim identical\n")


def build_dataset(data_folder, split_folder, context_size, index_context_size):
    return ViNT_Dataset(
        data_folder=data_folder,
        data_split_folder=split_folder,
        dataset_name="go_stanford",
        image_size=[96, 96],
        waypoint_spacing=1,
        min_dist_cat=0,
        max_dist_cat=20,
        min_action_distance=3,
        max_action_distance=20,
        negative_mining=True,
        len_traj_pred=8,
        learn_angle=False,
        context_size=context_size,
        context_type="temporal",
        index_context_size=index_context_size,
        end_slack=0,
        goals_per_obs=2,
        normalize=True,
        goal_type="image",
    )


def check_index_equivalence(data_folder, split_folder, stock_index_path):
    print(f"== sample index: {split_folder} ==")

    with open(stock_index_path, "rb") as f:
        stock_index, _ = pickle.load(f)

    # (a) default index_context_size reproduces the stock index exactly
    dataset = build_dataset(data_folder, split_folder, context_size=3, index_context_size=None)
    rebuilt_index, _ = dataset._build_index()
    if rebuilt_index != stock_index:
        raise SystemExit("FAIL: default index_context_size no longer reproduces the stock index")
    print(f"  PASS: default reproduces stock index ({len(stock_index):,} samples)")

    # (b) pinning index_context_size makes the index independent of context_size
    reference_index = None
    for context_size in CONTEXT_SIZES:
        dataset = build_dataset(
            data_folder, split_folder, context_size, PINNED_INDEX_CONTEXT_SIZE
        )
        index, _ = dataset._build_index()
        print(f"  context_size={context_size:2d} -> {len(index):,} samples")
        if reference_index is None:
            reference_index = index
        elif index != reference_index:
            raise SystemExit("FAIL: pinned index still varies with context_size")
    shrinkage = 1 - len(reference_index) / len(stock_index)
    print(
        f"  PASS: identical index for all context_size "
        f"({len(reference_index):,} samples, {shrinkage:.1%} smaller than stock n=3)\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default="/data/raw/go_stanford/go_stanford")
    parser.add_argument("--splits-root", default="/data/splits/go_stanford")
    args = parser.parse_args()

    check_model_invariance()
    for split in ["train", "test"]:
        split_folder = os.path.join(args.splits_root, split)
        check_index_equivalence(
            args.data_folder,
            split_folder,
            os.path.join(split_folder, "dataset_dist_0_to_20_context_temporal_n3_slack_0.pkl"),
        )
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
