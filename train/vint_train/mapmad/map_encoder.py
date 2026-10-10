"""Map encoder (Phase 3 confirmation item 4): the [3, 64, 64] local map -> one 256-number map token.

4 conv layers (3x3, stride 2, padding 1; channels 3 -> 32 -> 64 -> 128 -> 128; 64 -> 32 -> 16 -> 8 -> 4 cells),
each followed by GroupNorm(8 groups) + ReLU; then FLATTEN 4 x 4 x 128 = 2,048 -> Linear -> 256.
- Flatten, not global pooling: the token must still say WHERE the heat is (left edge vs right edge).
- GroupNorm, not BatchNorm: GoStanford's all-zero maps would skew batch statistics, and DataParallel computes
  BatchNorm statistics per GPU.
- The last Linear starts at zero (item 6): at step 0 the map token is its positional row only, so the new random
  encoder does not disturb the pretrained transformer; gradients reach the Linear from step 1 (the conv
  layers get gradients once the Linear is non-zero).
About 0.24 M conv + 0.52 M linear weights.
"""

from typing import Sequence

import torch
import torch.nn as nn


class MapEncoder(nn.Module):
    """[B, in_channels, 64, 64] float map -> [B, out_dim] token."""

    def __init__(self, in_channels: int = 3, channels: Sequence[int] = (32, 64, 128, 128), map_size: int = 64,
                 out_dim: int = 256, groups: int = 8) -> None:
        super().__init__()
        layers, c_in = [], in_channels
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1), nn.GroupNorm(groups, c_out),
                       nn.ReLU(inplace=True)]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        side = map_size // 2 ** len(channels)
        self.proj = nn.Linear(c_in * side * side, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, map_img: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.flatten(self.conv(map_img), start_dim=1))
