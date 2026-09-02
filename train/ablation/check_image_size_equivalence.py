"""Confirm the Ablation B edits do not move the baseline.

Ablation B changes only `image_size`. Three claims are checked:

1. Model. `image_size` never reaches the model: `NoMaD_ViNT.__init__` does not take it,
   and both EfficientNet encoders end in `AdaptiveAvgPool2d(1)`, so any input size
   collapses to 1280 features before the `Linear(1280, 256)` compression. Parameter count
   and conditioning dimension must therefore be identical at every arm size. The encoder's
   pre-pool feature grid *does* change, which is the mechanism under test, so it is
   reported alongside.

2. Image cache. `_build_caches` now opens the LMDB with `lock=False` to stop the
   reader-slot crash when several arms are scored in one process. That changes locking,
   not data: the bytes read back for a given key must be identical either way.

3. Sample index. `image_size` does not appear in the index path or the index contents --
   only `index_context_size` does. So every arm indexes the identical (trajectory,
   curr_time) pairs, and the `img096` arm is genuinely the A1 `ctx03` run.

Run inside the container:
    CUDA_VISIBLE_DEVICES=1 python ablation/check_image_size_equivalence.py
"""

import argparse
import os
import pickle

import lmdb
import torch
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

# (arm, image_size as configured: [width, height])
ARMS = [
    ("img096", [96, 96]),
    ("img112", [112, 112]),
    ("img128x96", [128, 96]),
    ("img160x120", [160, 120]),
]
BASELINE_ARM = "img096"
CONTEXT_SIZE = 3
INDEX_CONTEXT_SIZE = 20
ENCODING_SIZE = 256


def build_model():
    vision_encoder = replace_bn_with_gn(
        NoMaD_ViNT(
            obs_encoding_size=ENCODING_SIZE,
            context_size=CONTEXT_SIZE,
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
    """Parameter count and conditioning dim must not move with image_size."""
    print("== model invariance across image_size ==")
    model = build_model().eval()
    num_params = sum(p.numel() for p in model.parameters())
    reference = None

    for arm, (width, height) in ARMS:
        obs = torch.zeros(2, 3 * (CONTEXT_SIZE + 1), height, width)
        goal = torch.zeros(2, 3, height, width)
        with torch.no_grad():
            grid = model.vision_encoder.obs_encoder.extract_features(obs[:, :3])
            cond = model(
                "vision_encoder",
                obs_img=obs,
                goal_img=goal,
                input_goal_mask=torch.zeros(2, dtype=torch.long),
            )
        print(
            "  %-11s image_size=%-11s tensor=%-14s featgrid=%-6s params=%s cond=%s"
            % (
                arm,
                str([width, height]),
                str(tuple(obs.shape[1:])),
                str(tuple(grid.shape[2:])),
                "{:,}".format(num_params),
                tuple(cond.shape),
            )
        )
        if reference is None:
            reference = (num_params, cond.shape)
        elif (num_params, cond.shape) != reference:
            raise SystemExit("FAIL: model is not invariant to image_size")

    print("  PASS: parameter count and conditioning dim identical at every size\n")


def check_cache_lock_equivalence(split_folder, num_keys):
    """`lock=False` must change locking only -- never the bytes handed to the decoder."""
    print("== LMDB lock=False reads identical bytes ==")
    cache_filename = os.path.join(split_folder, "dataset_go_stanford.lmdb")
    if not os.path.exists(cache_filename):
        raise SystemExit(f"FAIL: no image cache at {cache_filename}")

    def read_keys(lock):
        env = lmdb.open(cache_filename, readonly=True, lock=lock)
        try:
            with env.begin() as txn:
                keys = [key for key, _ in zip(txn.cursor().iternext(values=False), range(num_keys))]
                return {key: txn.get(key) for key in keys}
        finally:
            env.close()

    locked = read_keys(lock=True)
    unlocked = read_keys(lock=False)
    if locked != unlocked:
        raise SystemExit("FAIL: lock=False changed the bytes read from the cache")
    print(f"  PASS: {len(locked):,} keys byte-identical with and without the reader lock\n")


def build_dataset(data_folder, split_folder, image_size):
    return ViNT_Dataset(
        data_folder=data_folder,
        data_split_folder=split_folder,
        dataset_name="go_stanford",
        image_size=image_size,
        waypoint_spacing=1,
        min_dist_cat=0,
        max_dist_cat=20,
        min_action_distance=3,
        max_action_distance=20,
        negative_mining=True,
        len_traj_pred=8,
        learn_angle=False,
        context_size=CONTEXT_SIZE,
        context_type="temporal",
        index_context_size=INDEX_CONTEXT_SIZE,
        end_slack=0,
        goals_per_obs=2,
        normalize=True,
        goal_type="image",
    )


def check_index_invariance(data_folder, split_folder):
    """Every arm must index the identical samples, so img096 really is ctx03."""
    print(f"== sample index invariance: {split_folder} ==")
    index_path = os.path.join(
        split_folder,
        f"dataset_dist_0_to_20_context_temporal_n{INDEX_CONTEXT_SIZE}_slack_0.pkl",
    )
    with open(index_path, "rb") as f:
        pinned_index, _ = pickle.load(f)
    print(f"  index file: {os.path.basename(index_path)} ({len(pinned_index):,} samples)")

    reference = None
    for arm, image_size in ARMS:
        dataset = build_dataset(data_folder, split_folder, image_size)
        if dataset.index_to_data != pinned_index:
            raise SystemExit(f"FAIL: {arm} does not index the pinned sample set")
        if reference is None:
            reference = len(dataset)
        elif len(dataset) != reference:
            raise SystemExit(f"FAIL: {arm} has {len(dataset)} samples, expected {reference}")
    print(f"  PASS: all {len(ARMS)} arms index the identical {reference:,} samples\n")


def check_sample_shapes(data_folder, split_folder):
    """Loaded tensors must carry exactly the configured resolution, obs and goal alike."""
    print("== loaded sample shapes follow image_size ==")
    for arm, (width, height) in ARMS:
        dataset = build_dataset(data_folder, split_folder, [width, height])
        obs_image, goal_image = dataset[0][0], dataset[0][1]
        expected_obs = (3 * (CONTEXT_SIZE + 1), height, width)
        expected_goal = (3, height, width)
        if tuple(obs_image.shape) != expected_obs or tuple(goal_image.shape) != expected_goal:
            raise SystemExit(
                f"FAIL: {arm} produced obs {tuple(obs_image.shape)} goal "
                f"{tuple(goal_image.shape)}, expected {expected_obs} / {expected_goal}"
            )
        print(f"  {arm:<11} obs={expected_obs}  goal={expected_goal}")
    print("  PASS: obs and goal both follow the configured [width, height]\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", default="/data/raw/go_stanford/go_stanford")
    parser.add_argument("--split-folder", default="/data/splits/go_stanford/test/")
    parser.add_argument("--cache-keys", type=int, default=2000)
    args = parser.parse_args()

    check_model_invariance()
    check_cache_lock_equivalence(args.split_folder, args.cache_keys)
    check_index_invariance(args.data_folder, args.split_folder)
    check_sample_shapes(args.data_folder, args.split_folder)
    print("All Ablation B baseline-equivalence checks passed.")


if __name__ == "__main__":
    main()
