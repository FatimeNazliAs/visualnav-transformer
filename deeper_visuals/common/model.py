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


def encode_tokens(model, obs_batch, goal_batch, device: str):
    """
    Run the real vision encoder over a batch and read its tokens back out.

    obs_batch  (B, 3*N_OBS_FRAMES, H, W)  ->  obs_tokens  (B, N_OBS_FRAMES, 256)
    goal_batch (B, 3, H, W)               ->  goal_tokens (B, 256)

    Captured with forward hooks on the two compression layers rather than by
    re-running the encoder pipeline by hand. Those layers are the seam worth
    hooking: everything before them is EfficientNet, everything after is the
    transformer, and their output is exactly the token sequence NoMaD_ViNT
    assembles. The frozen debug_visuals/visualize_stage2.py reimplemented that
    pipeline instead — a second copy of upstream code, free to drift from it.
    A hook has no copy to drift.

    The row ordering is this function's one real piece of knowledge, and the
    reason it lives here rather than at each call site. NoMaD_ViNT splits the
    channel-stacked observations into frames along the *batch* dimension, so
    compress_obs_enc emits (N_OBS_FRAMES * B, 256) frame-major. Unpacking that
    is easy to get subtly wrong and impossible to notice when B is 1 — so it is
    done once, here, the same way the model does it.

    The mask is the navigation one (goal visible). Masking hides the goal token
    from attention, not from the encoder, so it cannot change either token; P3
    is where the mask has something to show.
    """
    captured = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output.detach().cpu().numpy()
        return hook

    encoder = model.vision_encoder
    handles = [
        encoder.compress_obs_enc.register_forward_hook(capture("obs")),
        encoder.compress_goal_enc.register_forward_hook(capture("goal")),
    ]
    try:
        with torch.no_grad():
            model(
                "vision_encoder",
                obs_img=torch.as_tensor(obs_batch).to(device),
                goal_img=torch.as_tensor(goal_batch).to(device),
                input_goal_mask=torch.zeros(len(obs_batch), dtype=torch.long,
                                            device=device),
            )
    finally:
        for handle in handles:
            handle.remove()

    batch_size = len(obs_batch)
    obs_tokens = captured["obs"].reshape(
        settings.N_OBS_FRAMES, batch_size, settings.ENCODING_SIZE)
    return obs_tokens.transpose(1, 0, 2), captured["goal"]


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

    # Which weights, from the config; then what only loading them can tell us.
    info = {
        **cfg.weights_provenance(),
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
