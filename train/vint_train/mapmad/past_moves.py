"""Past moves as an input token + past-move prediction (Phase 14b-3 item B1; Past-Token Prediction, arXiv 2505.09561).

Plain words: the robot is told how it moved in the last ~10 s (8 moves, one per 5 control steps = the same spacing as
the predicted waypoints) and must also say them back from what it sees. A robot that remembers it has just turned
left three times is less likely to flip right, then left again.

- **Displacements** (`past_displacements`): for k = 8 .. 1 (oldest first) the move from the pose at step t - 5k to the
  pose at step t - 5(k - 1), written in the frame of its start pose: (dx forward, dy left) / 0.25 m and dtheta (rad,
  wrapped to [-pi, pi]). Steps before 0 repeat pose 0 (the robot stood there), so early steps get zero moves -- the
  same at training and at test time, no special case.
- **Token** (`PastEncoder`): the 24 numbers -> Linear 24 -> 256, GELU, Linear 256 -> 256 (zero init, like the map
  encoder's last Linear: at step 0 the token is its positional row only). Placed LAST, after the two map windows, so
  every existing token keeps its positional row. It has its own mask bit (1 = hidden); a hidden token is excluded from
  attention and pooling; with both maps hidden AND the token hidden (or absent) the network runs NoMaD's own
  computation exactly (G10 row 2).
- **Auxiliary head** (`PastHead`): the pooled context (the condition the diffusion head gets) -> 256 -> 24, trained with
  an MSE on the true displacements, weight lambda_past (0.1). In training the token is hidden with probability
  `drop_p`, so the head must read the past from the context pictures in those samples (PTP's point).
- Parameters live under `vision_encoder.map_encoder.past*`: the official loader accepts them as new keys and the
  map-encoder lr group (1e-4) trains them.
"""

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

N_DIMS = 3  # dx, dy, dtheta


def wrap(a: float) -> float:
    """Angle into [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def past_displacements(xy: Sequence[Sequence[float]], yaw: Sequence[float], t: int, n: int = 8, spacing: int = 5,
                       scale_m: float = 0.25) -> np.ndarray:
    """(n, 3) float32: the last n moves before step t (oldest first), each in its own start frame (module docstring).
    xy (T, 2) metres and yaw (T,) radians in NoMaD's 2D frame (traj_data.pkl / mapmad_sim.frames.to_2d)."""
    xy = np.asarray(xy, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    out = np.zeros((n, N_DIMS), dtype=np.float32)
    for j, k in enumerate(range(n, 0, -1)):
        a, b = max(t - spacing * k, 0), max(t - spacing * (k - 1), 0)
        d = xy[b] - xy[a]
        c, s = math.cos(yaw[a]), math.sin(yaw[a])
        out[j] = (( d[0] * c + d[1] * s) / scale_m, (-d[0] * s + d[1] * c) / scale_m, wrap(float(yaw[b] - yaw[a])))
    return out


class PastEncoder(nn.Module):
    """[B, n, 3] past moves -> [B, out_dim] token; last Linear zero-initialised (module docstring)."""

    def __init__(self, n: int = 8, out_dim: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n * N_DIMS, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        return self.net(torch.flatten(past, start_dim=1))


class PastHead(nn.Module):
    """[B, in_dim] pooled context -> [B, n, 3] predicted past moves."""

    def __init__(self, n: int = 8, in_dim: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.n = n
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, n * N_DIMS))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond).view(-1, self.n, N_DIMS)


def past_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the samples with a true past (valid 1; GoStanford 0); 0 when there is none."""
    per = ((pred - target.to(pred.device)) ** 2).mean(dim=(1, 2))
    v = valid.to(pred.device).float()
    return (per * v).sum() / v.sum().clamp(min=1.0)
