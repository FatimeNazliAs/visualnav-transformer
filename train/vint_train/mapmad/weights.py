"""Load the official nomad.pth into MapMaD (Phase 3 confirmation item 6, gate G3 row 2).

- Every checkpoint key must exist in the model with the same shape (no checkpoint key may be dropped).
- The only model keys the checkpoint lacks must be `vision_encoder.map_encoder.*` (kept at their init: random
  convs, zero last Linear).
- The positional encoding is a computed sine table (a buffer, not learned). It may grow from context_size + 2 to
  context_size + 3 rows ONLY if the model's first rows equal the checkpoint's exactly; the extra row stays the
  model's own sine row.
Then `load_state_dict(strict=True)` on the merged dict, so nothing is skipped silently.

Our own checkpoints (train_loop saves them unwrapped) load with `load_checkpoint`: strict, every key.
"""

from typing import Any, Dict, Tuple

import torch

POS_KEY = "vision_encoder.positional_encoding.pos_enc"
NEW_PREFIX = "vision_encoder.map_encoder."


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove a DataParallel `module.` prefix from every key."""
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}


def load_official_into(model: torch.nn.Module, weights_path: str) -> Dict[str, Any]:
    """Load `weights_path` into `model` by the rules above; returns a report (new keys, positional-row check)."""
    ckpt = strip_module_prefix(torch.load(weights_path, map_location="cpu", weights_only=True))
    own = model.state_dict()
    unexpected = sorted(set(ckpt) - set(own))
    if unexpected:
        raise RuntimeError(f"checkpoint keys missing from the model: {unexpected[:10]} ({len(unexpected)})")
    new_keys = sorted(set(own) - set(ckpt))
    bad_new = [k for k in new_keys if not k.startswith(NEW_PREFIX)]
    if bad_new:
        raise RuntimeError(f"model keys not in the checkpoint outside {NEW_PREFIX}*: {bad_new}")

    merged, pos_report = dict(ckpt), {"grown": False}
    for k, v in ckpt.items():
        if own[k].shape == v.shape:
            continue
        if k != POS_KEY:
            raise RuntimeError(f"shape mismatch for {k}: checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)}")
        rows = v.shape[1]
        if not torch.equal(own[k][:, :rows].cpu(), v):
            raise RuntimeError(f"{POS_KEY}: the model's first {rows} rows differ from the checkpoint's")
        merged[k] = own[k].detach().clone()
        pos_report = {"grown": True, "checkpoint_rows": rows, "model_rows": int(own[k].shape[1]),
                      "first_rows_equal": True}
    for k in new_keys:
        merged[k] = own[k]
    model.load_state_dict(merged, strict=True)
    return {"weights": weights_path, "checkpoint_keys": len(ckpt), "loaded_from_checkpoint": len(ckpt),
            "new_keys": new_keys,
            "new_parameters": int(sum(own[k].numel() for k in new_keys)), "positional_encoding": pos_report}


def build_with_official(cfg: Dict[str, Any], weights_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """MapMaD (map_input true) with the official nomad.pth loaded by `load_official_into`, in eval mode."""
    from vint_train.mapmad.closed_loop.nomad_policy import build_nomad

    model = build_nomad(dict(cfg, map_input=True))
    report = load_official_into(model, weights_path)
    return model.to(device).eval(), report


def load_checkpoint(cfg: Dict[str, Any], path: str, device: torch.device, map_input: bool) -> torch.nn.Module:
    """A NoMaD/MapMaD model with a plain (unwrapped or `module.`-prefixed) state dict loaded strictly, in eval mode."""
    from vint_train.mapmad.closed_loop.nomad_policy import build_nomad

    model = build_nomad(dict(cfg, map_input=map_input))
    model.load_state_dict(strip_module_prefix(torch.load(path, map_location="cpu", weights_only=True)), strict=True)
    return model.to(device).eval()
