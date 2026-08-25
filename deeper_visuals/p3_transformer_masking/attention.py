# deeper_visuals/p3_transformer_masking/attention.py
"""
Reading attention weights out of NoMaD's transformer.

nn.TransformerEncoder throws its attention weights away — the figure this phase
exists to draw is not in the model's output. Two obstacles stand in the way of
getting it, both re-verified against torch 2.4.1 on this host before this module
was written:

1. In eval mode under no_grad, TransformerEncoderLayer.forward takes a fused
   fast path (torch._transformer_encoder_layer_fwd) that bypasses
   self_attn.forward ENTIRELY, so a hook on self_attn never fires. It is
   disabled for the duration of the pass via
   torch.backends.mha.set_fastpath_enabled and restored afterwards.
2. Once on the plain Python path, TransformerEncoderLayer._sa_block hardcodes
   need_weights=False, so a forward hook would still only ever see None. A
   forward_pre_hook with with_kwargs=True rewrites the call to
   need_weights=True / average_attn_weights=False, and the forward hook then
   captures the per-head (B, n_heads, N_TOKENS, N_TOKENS) weights.

Disabling the fused kernel changes the arithmetic slightly — same computation,
different order of operations. Measured on the hero scene it moves c_t by at
most 1.6e-3, against values reaching 10.3 and a navigation-vs-exploration
difference of 0.46 mean / 1.90 max: roughly three orders of magnitude of
headroom. run_model.py measures that agreement on every run and records it in
facts.json rather than trusting this paragraph, and the c_t the page plots is
the model's own fused-path output, not this one.

Adapted from the frozen debug_visuals/visualize_stage4.py, which solved the same
two problems. What changed: the masks are taken from the model's own `all_masks`
and `avg_pool_mask` instead of being rebuilt here (a rebuilt mask is a second
copy of upstream's convention, free to drift from it), all four layers are kept
rather than only the plotted one, and the result is a value object instead of a
dict of tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from deeper_visuals.common import measure


@dataclass(frozen=True)
class TransformerPass:
    """One pass of the self-attention stack, with the weights it discarded."""

    attention: np.ndarray      # (n_layers, n_heads, N_TOKENS, N_TOKENS)
    output_tokens: np.ndarray  # (N_TOKENS, ENCODING_SIZE) — after the stack
    context: np.ndarray        # (ENCODING_SIZE,) — c_t, mean-pooled

    @property
    def n_layers(self) -> int:
        return self.attention.shape[0]

    @property
    def n_heads(self) -> int:
        return self.attention.shape[1]

    def layer(self, index: int) -> np.ndarray:
        """One layer's weights, (n_heads, N_TOKENS, N_TOKENS)."""
        return self.attention[index]


def _capture_attention(sa_encoder):
    """
    Hook every layer's self_attn so it reports its per-head weights.

    Returns (captured, handles); the caller removes the handles. The pre-hook
    rewrites the call kwargs, the post-hook takes output[1] — the weights that
    _sa_block would otherwise drop on the floor.
    """
    captured: dict[int, np.ndarray] = {}

    def force_weights(_module, args, kwargs):
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return args, kwargs

    def keep(layer_index: int):
        def hook(_module, _args, output):
            captured[layer_index] = output[1].detach().cpu().numpy()
        return hook

    handles = []
    for index, layer in enumerate(sa_encoder.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(
            force_weights, with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(keep(index)))
    return captured, handles


def run(model, tokens: np.ndarray, device: str, *, goal_mask: int) -> TransformerPass:
    """
    Push one scene's tokens through the transformer, keeping the attention.

    `tokens` is (N_TOKENS, ENCODING_SIZE) — the four observation tokens plus the
    goal token, exactly as the encoders produced them. `goal_mask` is
    common.model.GOAL_VISIBLE or GOAL_HIDDEN.

    This mirrors the second half of NoMaD_ViNT.forward: positional encoding, the
    self-attention stack under a src_key_padding_mask, then the rescaled mean
    pool. Every piece of the mask arithmetic is the model's own — `all_masks`
    selects which token is hidden, `avg_pool_mask` carries the rescale that
    spreads the pool over the four surviving tokens rather than averaging a
    dropped one in as a zero. Rebuilding either here would be a second copy of a
    convention this phase's whole point is to report accurately.
    """
    encoder = model.vision_encoder
    sa_encoder = encoder.sa_encoder

    batch = torch.from_numpy(tokens).unsqueeze(0).to(device)
    selector = torch.tensor([goal_mask], dtype=torch.long, device=device)
    src_key_padding_mask = torch.index_select(
        encoder.all_masks.to(device), 0, selector)
    pool_weights = torch.index_select(
        encoder.avg_pool_mask.to(device), 0, selector).unsqueeze(-1)

    captured, handles = _capture_attention(sa_encoder)
    was_enabled = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            positioned = encoder.positional_encoding(batch)
            output = sa_encoder(positioned,
                                src_key_padding_mask=src_key_padding_mask)
            context = (output * pool_weights).mean(dim=1)
    finally:
        torch.backends.mha.set_fastpath_enabled(was_enabled)
        for handle in handles:
            handle.remove()

    if len(captured) != len(sa_encoder.layers):
        raise RuntimeError(
            f"attention hooks fired for {len(captured)} of "
            f"{len(sa_encoder.layers)} layers. The fast-path bypass or the "
            f"MultiheadAttention call signature has changed in this torch "
            f"version ({torch.__version__}); see this module's docstring."
        )

    return TransformerPass(
        attention=np.stack([captured[i][0] for i in sorted(captured)]),
        output_tokens=output.detach().cpu().numpy()[0],
        context=context.detach().cpu().numpy()[0],
    )


def geometry(model) -> dict:
    """The transformer's shape, measured off the loaded model rather than asserted."""
    sa_encoder = model.vision_encoder.sa_encoder
    self_attn = sa_encoder.layers[0].self_attn
    return {
        "n_layers":     len(sa_encoder.layers),
        "n_heads":      self_attn.num_heads,
        "d_model":      self_attn.embed_dim,
        "d_head":       self_attn.embed_dim // self_attn.num_heads,
        # A number, not "3.16M". The key says _m and measure.millions exists;
        # formatting it here put the rounding and the unit behind a GPU run.
        "n_params_m":   measure.millions(
                            sum(p.numel() for p in sa_encoder.parameters())),
    }
