# debug/stage2_obs_enc.py
import torch
from debug.config import DEVICE, CONTEXT_SIZE, ENCODING_SIZE


def run_obs_encoder(model, obs_input, verbose=True):
    """
    Runs each of the P=3 observation frames independently through
    EfficientNet-B0, then projects to ENCODING_SIZE=256.

    Why independent per frame: the CNN handles spatial features.
    Temporal relationships across frames are the Transformer's job.

    (B, P, C, H, W) → flatten to (B*P, C, H, W) → encode → (B, P, 256)
    """
    backbone = _get_backbone(model)
    B, P, C, H, W = obs_input.shape

    obs_flat = obs_input.view(B * P, C, H, W)           # (B*P, C, H, W)

    with torch.no_grad():
        # Don't call backbone.obs_encoder(obs_flat) directly — that runs
        # EfficientNet's own forward(), which ends in its built-in 1000-way
        # ImageNet classifier (_fc), not the 256-dim NoMaD embedding.
        # Instead, replicate NoMaD_ViNT.forward()'s obs-encoding path:
        # stop after pooling, then apply the model's own compression layer.
        obs_encoded = backbone.obs_encoder.extract_features(obs_flat)
        obs_encoded = backbone.obs_encoder._avg_pooling(obs_encoded)
        obs_encoded = obs_encoded.flatten(start_dim=1)      # (B*P, 1280)
        obs_encoded = backbone.obs_encoder._dropout(obs_encoded)
        obs_encoded = backbone.compress_obs_enc(obs_encoded)  # (B*P, 256)

    obs_tokens = obs_encoded.view(B, P, ENCODING_SIZE)  # (B, P, 256)

    if verbose:
        print(f"  Input  obs_input  : {tuple(obs_input.shape)}")
        print(f"  Flat   obs_flat   : {tuple(obs_flat.shape)}  (batch*context, C, H, W)")
        print(f"  Encoded           : {tuple(obs_encoded.shape)}")
        print(f"  Output obs_tokens : {tuple(obs_tokens.shape)}  (batch, context, 256)")
        print(f"  Value range       : [{obs_tokens.min():.3f}, {obs_tokens.max():.3f}]")

    return obs_tokens


def _get_backbone(model):
    """
    The backbone attribute name varies slightly across repo versions.
    This tries the known names in order.
    """
    for attr in ("backbone", "vint", "vision_encoder"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        f"Cannot find backbone. Top-level model attributes: {list(vars(model).keys())}"
    )