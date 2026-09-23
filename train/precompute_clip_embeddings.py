"""Precompute the frozen-CLIP image embedding of every frame a CLIP-goal run can use.

One pass over the train and test split frames of each dataset in the config, writing:
    clip_cache   LMDB, key "<traj_name>/<frame>" -> raw float32 [512] image embedding
    clip_mu_img  mean normalised image embedding over the TRAIN frames only
    clip_mu_txt  mean normalised text embedding over GOAL_WORDS x template + SCENE_CAPTIONS
    meta.yaml    what produced the three files above

Embeddings are stored raw; `prep` centres them at load time. CLIP sees the original
frames through its own 224px preprocessing, not the 96px training resize.

Refuses to run if the output folder already exists, so no cache is ever overwritten.

Run inside the container, from `/app/visualnav-transformer/train`:
    python precompute_clip_embeddings.py --config config/nomad_clip.yaml
"""
import argparse
import os
import pickle
from typing import List, Tuple

import lmdb
import numpy as np
import torch
import tqdm
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vint_train.data.clip_goal_utils import (
    GOAL_WORDS,
    SCENE_CAPTIONS,
    apply_template,
    cache_key,
    encode_captions,
    encode_images,
    encode_words,
    l2_normalize,
    load_clip,
    modality_mean,
)
from vint_train.data.data_utils import get_data_path

class FrameDataset(Dataset):
    """Every (trajectory, frame) of a list of trajectories, as CLIP-preprocessed images."""

    def __init__(self, data_folder: str, frames: List[Tuple[str, int]], preprocess):
        self.data_folder = data_folder
        self.frames = frames
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        traj_name, time = self.frames[i]
        image = Image.open(get_data_path(self.data_folder, traj_name, time))
        return self.preprocess(image), i


def split_frames(data_folder: str, split_folder: str) -> List[Tuple[str, int]]:
    """All frames of the split's trajectories -- the same set ViNT_Dataset draws goals from."""
    with open(os.path.join(split_folder, "traj_names.txt")) as f:
        traj_names = [name for name in f.read().split("\n") if name]
    frames = []
    for traj_name in traj_names:
        with open(os.path.join(data_folder, traj_name, "traj_data.pkl"), "rb") as f:
            traj_len = len(pickle.load(f)["position"])
        frames.extend((traj_name, time) for time in range(traj_len))
    return frames


def embed_frames(model, preprocess, data_folder, frames, txn, device, batch_size, num_workers):
    """Write each frame's embedding to the LMDB transaction; return their normalised sum."""
    loader = DataLoader(
        FrameDataset(data_folder, frames, preprocess),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    normalised_sum = torch.zeros(512, device=device)
    for images, indices in tqdm.tqdm(loader, dynamic_ncols=True):
        embeddings = encode_images(model, images.to(device))
        normalised_sum += l2_normalize(embeddings).sum(dim=0)
        for index, embedding in zip(indices.tolist(), embeddings.cpu().numpy()):
            txn.put(cache_key(*frames[index]), embedding.astype(np.float32).tobytes())
    return normalised_sum


def main(config: dict, batch_size: int, num_workers: int) -> None:
    cache_dir = os.path.dirname(config["clip_cache"])
    assert not os.path.exists(cache_dir), f"{cache_dir} already exists; refusing to overwrite"
    for path in (config["clip_mu_img"], config["clip_mu_txt"]):
        assert os.path.dirname(path) == cache_dir, f"{path} must live next to the cache"
    os.makedirs(cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_clip(config["clip_model"], device)

    train_sum, train_count, frame_counts = torch.zeros(512, device=device), 0, {}
    with lmdb.open(config["clip_cache"], map_size=2**36) as env:
        for dataset_name, data_config in config["datasets"].items():
            for split in ("train", "test"):
                if split not in data_config:
                    continue
                frames = split_frames(data_config["data_folder"], data_config[split])
                print(f"{dataset_name} {split}: {len(frames)} frames")
                with env.begin(write=True) as txn:
                    split_sum = embed_frames(
                        model, preprocess, data_config["data_folder"], frames,
                        txn, device, batch_size, num_workers,
                    )
                frame_counts[f"{dataset_name}_{split}"] = len(frames)
                if split == "train":
                    train_sum += split_sum
                    train_count += len(frames)

    mu_img = (train_sum / train_count).cpu().numpy()
    word_prompts = apply_template(GOAL_WORDS, config["clip_text_template"])
    text_embeddings = torch.cat([
        encode_words(model, tokenizer, GOAL_WORDS, config["clip_text_template"], device),
        encode_captions(model, tokenizer, SCENE_CAPTIONS, device),
    ])
    mu_txt = modality_mean(text_embeddings).cpu().numpy()
    np.save(config["clip_mu_img"], mu_img)
    np.save(config["clip_mu_txt"], mu_txt)

    with open(os.path.join(cache_dir, "meta.yaml"), "w") as f:
        yaml.safe_dump({
            "clip_model": config["clip_model"],
            "clip_text_template": config["clip_text_template"],
            "frames": frame_counts,
            "mu_img_frames": train_count,
            "mu_txt_prompts": word_prompts + SCENE_CAPTIONS,
        }, f, sort_keys=False)
    print(f"Wrote {config['clip_cache']}, mu_img, mu_txt and meta.yaml to {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    with open(args.config) as f:
        main(yaml.safe_load(f), args.batch_size, args.num_workers)
