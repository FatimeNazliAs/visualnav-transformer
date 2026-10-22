"""MapMaD v2 as a closed-loop Policy (Phase 11 item 11.1; Phase 12 reuses it): the v2 checkpoint through
inference_v2.MapMaDV2Inference with the Phase 5 MapMaD arm settings.

Per control step (0.25 s):
1. context: the pictures of control steps t-15, t-10, t-5, t (spacing 5, as trained); at the episode start the
   missing older pictures repeat the first one (mapmad_policy.context_indices), so the robot drives from step 0;
2. goal: photo hidden (the current picture as placeholder, masked); maps: the server's near + far windows
   (obs.extras["maps"], [2, 4, 64, 64]) shown, or both hidden (`maps=False`: then the network uses the first 5 tokens
   only = NoMaD's own computation, Phase 10 G10 row 2);
3. vision encoder -> condition x 8 samples; 10-step DDPM from Gaussian noise, all noise from one torch.Generator
   seeded at reset (NomadPolicy), so the same seed gives the same drive;
4. sample 0, waypoint `waypoint_index` (1) x step_m (0.25 m), then NoMaD's pd_controller within 0.2 m/s, 0.4 rad/s.
The checkpoint's sha256 is checked when the policy is built; a mismatch stops the run.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from vint_train.mapmad.closed_loop.mapmad_policy import context_indices
from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy
from vint_train.mapmad.closed_loop.policy import Observation
from vint_train.mapmad.inference_v2 import MapMaDV2Inference


class MapMaDV2Policy(NomadPolicy):
    """MapMaD v2, maps shown or both hidden; several policies may share one network (`net`)."""

    def __init__(self, checkpoint: str, checkpoint_sha256: str, cfg: Dict[str, Any], device: torch.device, maps: bool,
                 num_samples: int = 8, sample_index: int = 0, waypoint_index: int = 1, context_spacing: int = 5,
                 step_m: float = 0.25, max_v: float = 0.2, max_w: float = 0.4, rate_hz: float = 4.0,
                 net: Optional[MapMaDV2Inference] = None) -> None:
        self.net = net if net is not None else MapMaDV2Inference(checkpoint, cfg, device, checkpoint_sha256,
                                                                 num_samples=num_samples, step_m=step_m)
        self.checkpoint_sha256 = checkpoint_sha256
        self._setup(self.net.model, cfg, device, num_samples, sample_index, waypoint_index, max_v, max_w, rate_hz)
        self.maps, self.spacing, self.step_m = bool(maps), int(context_spacing), float(step_m)
        self.context_frames = cfg["context_size"] * self.spacing + 1
        self.name = "mapmad_v2_maps" if self.maps else "mapmad_v2_hidden"

    def ready(self, obs: Observation) -> bool:
        return len(obs.frames) >= 1  # missing past pictures repeat the first one

    @property
    def metres_per_unit(self) -> float:
        return self.step_m

    def encoder_inputs(self, obs: Observation) -> Dict[str, torch.Tensor]:
        frames = [obs.frames[i] for i in context_indices(len(obs.frames), self.cfg["context_size"], self.spacing)]
        maps = obs.extras.get("maps") if self.maps else None
        if self.maps and maps is None:
            raise RuntimeError("maps shown but the simulator sent no map windows")
        near, far = (maps[0], maps[1]) if maps is not None else (np.zeros((4, 64, 64), np.float32),) * 2
        return self.net.encoder_inputs(frames, near, far, self.maps, self.maps, None)
