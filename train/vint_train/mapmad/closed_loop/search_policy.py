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
        # Phase 14b-3 (v3): a checkpoint trained with the past-move token gets the robot's last n moves (the runner
        # passes its pose history in extras["pose_history"]); v2 / v2.1b checkpoints: no token, as before
        self.past_cfg = (cfg.get("map") or {}).get("past_moves") or None
        self.wants_poses = self.past_cfg is not None

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
        return self.net.encoder_inputs(frames, near, far, self.maps, self.maps, None, self.past_moves(obs))

    def past_moves(self, obs: Observation) -> Optional[np.ndarray]:
        """(n, 3) past moves at this step (past_moves.past_displacements over the runner's pose history), or None."""
        if self.past_cfg is None:
            return None
        from vint_train.mapmad.past_moves import past_displacements

        hist = obs.extras.get("pose_history")
        if not hist:
            raise RuntimeError("a past-move checkpoint needs the runner's pose history (extras['pose_history'])")
        c = self.past_cfg
        h = np.asarray(hist, dtype=np.float64)
        return past_displacements(h[:, :2], h[:, 2], len(h) - 1, int(c["n"]), int(c["spacing"]), float(c["scale_m"]))
