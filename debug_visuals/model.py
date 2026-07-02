# debug_visuals/model.py
"""
NoMaD checkpoint loading for the debug_visuals pipeline.

Kept in one place so every stage (2, 4, 5) loads the model identically.
The construction mirrors train.py's checkpoint-loading logic: the same
NoMaD_ViNT + ConditionalUnet1D + DenseNetwork wiring, the BatchNorm→GroupNorm
swap, and a strict=False state-dict load. All hyperparameters come from
config.py and must stay in sync with the checkpoint's nomad.yaml.

Repo root / train are put on sys.path by debug_visuals/__init__.py, so the
vint_train / diffusion_policy imports below resolve without vint_train being
pip-installed.
"""

from __future__ import annotations

import torch

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from debug_visuals.config import (
    CHECKPOINT,
    CONTEXT_SIZE,
    ENCODING_SIZE,
    DOWN_DIMS,
    DEVICE,
)


def load_model():
    print(f"  Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    # Checkpoint may store state_dict directly or nested under a key
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=ENCODING_SIZE,
        context_size=CONTEXT_SIZE,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=ENCODING_SIZE,
        down_dims=DOWN_DIMS,
        cond_predict_scale=False,
    )
    dist_pred_net = DenseNetwork(embedding_dim=ENCODING_SIZE)

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING — missing keys  : {missing[:3]} ...")
    if unexpected:
        print(f"  WARNING — unexpected keys: {unexpected[:3]} ...")

    model = model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters : {n_params:.1f}M")
    print(f"  Device     : {DEVICE}")
    return model
