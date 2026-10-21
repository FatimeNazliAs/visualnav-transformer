"""The map tokens' attention share (one definition for the training log, the in-training checks and the offline
harness): the TransformerEncoder re-run layer by layer with `need_weights` to read the attention weights."""

from typing import Tuple

import torch


def attention_shares(encoder, tokens: torch.Tensor, padding: torch.Tensor,
                     n_map: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(encoder output, per-sample attention share of each of the last n_map (map) tokens, (B, n_map)): the
    TransformerEncoder (norm_first, eval) re-run layer by layer to read the attention weights; share of map token k =
    mean over layers, heads and visible non-map queries of the weight on token k. v2: n_map = 2 (near, far)."""
    x, shares = tokens, []
    for layer in encoder.layers:
        h = layer.norm1(x)
        a, w = layer.self_attn(h, h, h, key_padding_mask=padding, need_weights=True, average_attn_weights=False)
        x = x + layer.dropout1(a)
        x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))
        shares.append(w[..., -n_map:])  # (B, heads, queries, n_map)
    if encoder.norm is not None:
        x = encoder.norm(x)
    q = (~padding).float()
    q[:, -n_map:] = 0.0  # the map tokens' own queries do not count
    per_layer_head = torch.stack(shares, 1)  # (B, layers, heads, queries, n_map)
    share = (per_layer_head * q[:, None, None, :, None]).sum(-2) / q.sum(-1)[:, None, None, None]
    return x, share.mean(dim=(1, 2))


def attention_share(encoder, tokens: torch.Tensor, padding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """v1 (one map token, last): (encoder output, per-sample map attention share (B,))."""
    x, share = attention_shares(encoder, tokens, padding, 1)
    return x, share[:, 0]
