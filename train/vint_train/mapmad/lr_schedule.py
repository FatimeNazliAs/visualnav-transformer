"""Phase 4 learning-rate schedule and parameter groups (Phase 4 confirmation items 3 and 4).

Used by train.py / the MapMaD loop when a config sets `lr_schedule: warmup_cosine` (Phase 3 configs do not, so
they keep NoMaD's CosineAnnealingLR stepped once per epoch).

Two AdamW parameter groups, in every run:
- `map_encoder`: the 18 `vision_encoder.map_encoder.*` tensors (new, randomly initialised), base lr `lr`;
- `pretrained`: every other weight (from nomad.pth), base lr `pretrained_lr` (default `lr`).
Run A: both 1e-4 (same as one group). Run B ("careful"): pretrained 1e-5.

The lr of a group = its base lr x factor(epoch, step), set before EVERY optimizer step (epochs counted from 0):
- epoch 0 (the first epoch, steps 0..S-1): linear warmup, factor = r + (1 - r) * step / S, with S the steps of an
  epoch (782 at 200,000 samples / batch 256) and r = `warmup_start_ratio` (0.01: 1e-4 -> 1e-6 start);
- epoch e >= 1: factor = 0.5 * (1 + cos(pi * (e - 1) / (epochs - 1))), constant within the epoch
  (1-based epoch n = e + 1: 1e-4 * 0.5 * (1 + cos(pi * (n - 2) / 19)) for 20 epochs).
So step S (the first step of epoch 1) is exactly the peak, and the last epoch is about 0.007 x peak.
NoMaD's GradualWarmupScheduler (epoch 0 at lr 0) is not used.
"""

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn

MAP_ENCODER_PREFIX = "vision_encoder.map_encoder."


def lr_factor(epoch: int, step: int, steps_per_epoch: int, epochs: int, warmup_start_ratio: float) -> float:
    """Multiplier of the base lr at (epoch, step in that epoch), both counted from 0 (rules in the module doc)."""
    if epoch == 0:
        return warmup_start_ratio + (1.0 - warmup_start_ratio) * step / steps_per_epoch
    if epochs <= 1:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / (epochs - 1)))


def param_groups(model: nn.Module, lr: float, pretrained_lr: float) -> List[Dict[str, Any]]:
    """[map_encoder group, pretrained group]; each carries its `name` and `base_lr`. Order of names is kept."""
    named = list(model.named_parameters())
    new = [p for n, p in named if n.startswith(MAP_ENCODER_PREFIX)]
    old = [p for n, p in named if not n.startswith(MAP_ENCODER_PREFIX)]
    if not new:
        raise ValueError(f"no {MAP_ENCODER_PREFIX}* parameters: is map_input on?")
    return [{"params": new, "lr": lr, "base_lr": lr, "name": "map_encoder"},
            {"params": old, "lr": pretrained_lr, "base_lr": pretrained_lr, "name": "pretrained"}]


def group_param_names(model: nn.Module) -> Dict[str, List[str]]:
    """Parameter names per group (for the tests and run_info.json)."""
    out: Dict[str, List[str]] = {"map_encoder": [], "pretrained": []}
    for n, _ in model.named_parameters():
        out["map_encoder" if n.startswith(MAP_ENCODER_PREFIX) else "pretrained"].append(n)
    return out


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW (train.py's call: torch defaults besides lr) over the two groups."""
    if config["optimizer"].lower() != "adamw":
        raise ValueError("lr_schedule warmup_cosine is implemented for adamw only")
    lr = float(config["lr"])
    return torch.optim.AdamW(param_groups(model, lr, float(config.get("pretrained_lr", lr))), lr=lr)


class WarmupCosine:
    """Sets every group's lr from its base_lr before each optimizer step; no state besides the config."""

    def __init__(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int, epochs: int,
                 warmup_start_ratio: float) -> None:
        self.optimizer, self.steps_per_epoch, self.epochs = optimizer, int(steps_per_epoch), int(epochs)
        self.warmup_start_ratio = float(warmup_start_ratio)

    def apply(self, epoch: int, step: int) -> Dict[str, float]:
        """Set and return {group name: lr} for (epoch, step)."""
        f = lr_factor(epoch, step, self.steps_per_epoch, self.epochs, self.warmup_start_ratio)
        out = {}
        for g in self.optimizer.param_groups:
            g["lr"] = g["base_lr"] * f
            out[g.get("name", "group")] = g["lr"]
        return out

    def lrs(self) -> Dict[str, float]:
        """The lr each group has right now."""
        return {g.get("name", "group"): g["lr"] for g in self.optimizer.param_groups}
