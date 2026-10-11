"""The map token's attention share (one definition for the training log, the in-training checks and the offline
harness): the TransformerEncoder re-run layer by layer with `need_weights` to read the attention weights."""

from typing import Tuple

import torch


def attention_share(encoder, tokens: torch.Tensor, padding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(encoder output, per-sample map attention share): the TransformerEncoder (norm_first, eval) re-run layer by
    layer to read the attention weights; share = mean over layers, heads and visible non-map queries of the
    weight on the last (map) token."""
    x, shares = tokens, []
    for layer in encoder.layers:
        h = layer.norm1(x)
        a, w = layer.self_attn(h, h, h, key_padding_mask=padding, need_weights=True, average_attn_weights=False)
        x = x + layer.dropout1(a)
        x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
        shares.append(w[..., -1])  # (B, heads, queries)
    if encoder.norm is not None:
        x = encoder.norm(x)
    q = (~padding).float()
    q[:, -1] = 0.0  # the map token's own query does not count
    per_layer_head = torch.stack(shares, 1)  # (B, layers, heads, queries)
    share = (per_layer_head * q[:, None, None, :]).sum(-1) / q.sum(-1)[:, None, None]
    return x, share.mean(dim=(1, 2))
