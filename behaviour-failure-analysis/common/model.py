# behaviour-failure-analysis/common/model.py
"""
NoMaD checkpoint loading.

Copied unchanged from deeper_visuals/common/model.py (branch
feature/deeper-visualization) apart from the import paths and the type of the
config it takes: an ExperimentConfig rather than a PhaseConfig. Both expose
`.checkpoint` and `.weights_provenance()`, which is the entire interface this
module needs, so the body is untouched.

The construction mirrors train.py's checkpoint-loading logic: the same
NoMaD_ViNT + ConditionalUnet1D + DenseNetwork wiring, the BatchNorm->GroupNorm
swap, and a strict=False state-dict load. Every hyperparameter comes from
common/settings.py and must stay in sync with the checkpoint's nomad.yaml.

Repo root / train are put on sys.path by common/__init__.py, so the
vint_train / diffusion_policy imports resolve without vint_train being
pip-installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from common import settings
from common.config import ExperimentConfig


# NoMaD_ViNT's goal-mask convention (nomad_vint.py: 0 = no mask, 1 = mask), named
# so call sites read as behaviours rather than as magic integers. The mask selects
# a row of the model's own `all_masks`, which is a src_key_padding_mask: in
# exploration the goal token is hidden from attention, NOT zeroed at the input.
# It is still encoded — the other four tokens simply cannot see it.
GOAL_VISIBLE = 0   # navigation  — the goal token takes part in attention
GOAL_HIDDEN  = 1   # exploration — the goal token is masked out of attention


@dataclass(frozen=True)
class VisionEncoding:
    """
    One pass of the vision encoder: the tokens that went in, and the c_t out.

    Kept together because they are one forward pass. The tokens are what P2
    draws and what P3 feeds to the transformer; `context` is what P4's diffusion
    head is conditioned on. Splitting them across two calls would mean running
    the two EfficientNets twice to see both halves of the same computation.
    """

    obs_tokens: np.ndarray   # (B, N_OBS_FRAMES, ENCODING_SIZE)
    goal_token: np.ndarray   # (B, ENCODING_SIZE)
    context:    np.ndarray   # (B, ENCODING_SIZE) — c_t, the mean-pooled output

    @property
    def tokens(self) -> np.ndarray:
        """The full (B, N_TOKENS, ENCODING_SIZE) sequence the transformer sees."""
        return np.concatenate([self.obs_tokens, self.goal_token[:, None, :]], axis=1)


def encode_tokens(model, obs_batch, goal_batch, device: str, *,
                  goal_mask: int = GOAL_VISIBLE) -> VisionEncoding:
    """
    Run the real vision encoder over a batch and read its tokens back out.

    obs_batch  (B, 3*N_OBS_FRAMES, H, W)  ->  obs_tokens  (B, N_OBS_FRAMES, 256)
    goal_batch (B, 3, H, W)               ->  goal_token  (B, 256)
                                              context     (B, 256)

    The tokens are captured with forward hooks on the two compression layers
    rather than by re-running the encoder pipeline by hand. Those layers are the
    seam worth hooking: everything before them is EfficientNet, everything after
    is the transformer, and their output is exactly the token sequence
    NoMaD_ViNT assembles. The frozen debug_visuals/visualize_stage2.py
    reimplemented that pipeline instead — a second copy of upstream code, free to
    drift from it. A hook has no copy to drift.

    The row ordering is this function's one real piece of knowledge, and the
    reason it lives here rather than at each call site. NoMaD_ViNT splits the
    channel-stacked observations into frames along the *batch* dimension, so
    compress_obs_enc emits (N_OBS_FRAMES * B, 256) frame-major. Unpacking that
    is easy to get subtly wrong and impossible to notice when B is 1 — so it is
    done once, here, the same way the model does it.

    `goal_mask` (GOAL_VISIBLE / GOAL_HIDDEN) changes `context` and nothing else.
    Masking hides the goal token from attention, not from the encoder, so both
    token arrays come back identical either way — which is precisely the fact P3
    is built to show, and the reason the mask belongs on this call rather than
    on a separate one.
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
            context = model(
                "vision_encoder",
                obs_img=torch.as_tensor(obs_batch).to(device),
                goal_img=torch.as_tensor(goal_batch).to(device),
                input_goal_mask=torch.full((len(obs_batch),), goal_mask,
                                           dtype=torch.long, device=device),
            )
    finally:
        for handle in handles:
            handle.remove()

    batch_size = len(obs_batch)
    obs_tokens = captured["obs"].reshape(
        settings.N_OBS_FRAMES, batch_size, settings.ENCODING_SIZE)
    return VisionEncoding(
        obs_tokens=obs_tokens.transpose(1, 0, 2),
        goal_token=captured["goal"],
        context=context.detach().cpu().numpy(),
    )


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


def load_model(cfg: ExperimentConfig, *, device: str | None = None, verbose: bool = True):
    """
    Build NoMaD and load cfg's weights onto it.

    Returns (model, info) where info is a dict of load facts worth recording in
    provenance.json — the weight file used, parameter count, and any key
    mismatch.
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
