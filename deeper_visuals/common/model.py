# deeper_visuals/common/model.py
"""
NoMaD checkpoint loading for the deeper_visuals phases.

Adapted from debug_visuals/model.py; the only structural change is that the
checkpoint path arrives via PhaseConfig instead of a module-level constant, so
one loader serves every phase and both weight variants.

The construction mirrors train.py's checkpoint-loading logic: the same
NoMaD_ViNT + ConditionalUnet1D + DenseNetwork wiring, the BatchNorm->GroupNorm
swap, and a strict=False state-dict load. Every hyperparameter comes from
common/settings.py and must stay in sync with the checkpoint's nomad.yaml.

Repo root / train are put on sys.path by deeper_visuals/__init__.py, so the
vint_train / diffusion_policy imports resolve without vint_train being
pip-installed.
"""

from __future__ import annotations

import torch

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from deeper_visuals.common import settings
from deeper_visuals.common.config import PhaseConfig


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model() -> NoMaD:
    """Wire up the NoMaD architecture with the training-time hyperparameters."""
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=settings.ENCODING_SIZE,
        context_size=settings.CONTEXT_SIZE,
        mha_num_attention_heads=settings.MHA_NUM_ATTENTION_HEADS,
        mha_num_attention_layers=settings.MHA_NUM_ATTENTION_LAYERS,
        mha_ff_dim_factor=settings.MHA_FF_DIM_FACTOR,
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=settings.ACTION_DIM,
        global_cond_dim=settings.ENCODING_SIZE,
        down_dims=settings.DOWN_DIMS,
        cond_predict_scale=False,
    )
    dist_pred_net = DenseNetwork(embedding_dim=settings.ENCODING_SIZE)

    return NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )


def load_model(cfg: PhaseConfig, *, device: str | None = None, verbose: bool = True):
    """
    Build NoMaD and load cfg's weights onto it.

    Returns (model, info) where info is a dict of load facts worth recording in
    facts.json — the weight file used, parameter count, and any key mismatch.
    A silently-partial load is the classic way these figures go subtly wrong,
    so the missing/unexpected key counts are surfaced rather than just printed.
    """
    device = device or get_device()
    if verbose:
        print(f"  Loading checkpoint: {cfg.checkpoint}")

    ckpt = torch.load(cfg.checkpoint, map_location=device)
    # Checkpoint may store the state_dict directly or nested under a key.
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = build_model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if verbose:
        if missing:
            print(f"  WARNING — missing keys   : {len(missing)} e.g. {missing[:3]}")
        if unexpected:
            print(f"  WARNING — unexpected keys: {len(unexpected)} e.g. {unexpected[:3]}")

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    info = {
        "checkpoint_tag":  cfg.checkpoint_tag,
        "checkpoint_file": cfg.checkpoint.name,
        "run":             cfg.run_dir.name,
        "device":          device,
        "n_params":        n_params,
        "n_params_m":      round(n_params / 1e6, 1),
        "missing_keys":    len(missing),
        "unexpected_keys": len(unexpected),
    }
    if verbose:
        print(f"  Parameters : {info['n_params_m']}M")
        print(f"  Device     : {device}")
    return model, info
