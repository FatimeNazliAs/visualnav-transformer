"""MapMaD as a closed-loop Policy (Phase 5 block item 4): the exam checkpoint with the deployment pipeline of the
official arms, plus the map input and the spacing the model was trained with.

Per control step (0.25 s):
1. context: the pictures of control steps t-15, t-10, t-5 and t (spacing 5 = 1.25 s apart, as in training,
   `vint_dataset._context_times`); at the episode start the missing older pictures repeat the first picture
   (block item 4), so the robot drives from step 0;
2. goal: arm C `map` -- photo hidden (the current picture as its placeholder, as in training), map shown with heat;
   arm D `masked` -- photo hidden, map hidden; arm F `photo` -- goal photo shown, map hidden. A hidden map is all
   zeros (as in training); hidden tokens never reach attention or pooling;
3. vision encoder -> condition, repeated for 8 samples; 10-step DDPM from Gaussian noise (one torch.Generator per
   episode, seeded at reset, as NomadPolicy);
4. get_action -> path 0, waypoint `waypoint_index`, times step_m = 0.25 m (metric_waypoint_spacing 0.05 x
   waypoint_spacing 5: the training normalisation), then NoMaD's pd_controller and velocity limits.
The checkpoint's sha256 is checked when the policy is built; a mismatch stops the run.
"""

import hashlib
from typing import Any, Dict, List

import numpy as np
import torch

from vint_train.mapmad.closed_loop.nomad_policy import NomadPolicy
from vint_train.mapmad.closed_loop.policy import Observation

GOALS = ("map", "masked", "photo")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sha256(path: str, expected: str) -> str:
    """sha256 of the file; raises if it is not `expected`."""
    got = file_sha256(path)
    if got != expected:
        raise RuntimeError(f"{path}: sha256 {got} != expected {expected} -- not the frozen checkpoint")
    return got


def context_indices(n_frames: int, context_size: int, spacing: int) -> List[int]:
    """Indices into the kept pictures (oldest first, at most context_size * spacing + 1) of the context frames
    t - context_size * spacing, ..., t - spacing, t; an index before the first kept picture becomes 0 (the episode's
    first picture: the buffer has dropped nothing yet while it is short)."""
    last = n_frames - 1
    return [max(last - k * spacing, 0) for k in range(context_size, -1, -1)]


class MapMaDPolicy(NomadPolicy):
    """MapMaD with one goal setting (GOALS); several instances may share one model (`model`)."""

    def __init__(self, checkpoint: str, checkpoint_sha256: str, cfg: Dict[str, Any], device: torch.device, goal: str,
                 num_samples: int = 8, sample_index: int = 0, waypoint_index: int = 2, context_spacing: int = 5,
                 step_m: float = 0.25, max_v: float = 0.2, max_w: float = 0.4, rate_hz: float = 4.0,
                 model: torch.nn.Module = None) -> None:
        from vint_train.mapmad.weights import load_checkpoint

        if goal not in GOALS:
            raise ValueError(f"goal {goal!r} not in {GOALS}")
        self.checkpoint_sha256 = check_sha256(checkpoint, checkpoint_sha256)
        if model is None:
            model = load_checkpoint(cfg, checkpoint, device, map_input=True)
        self._setup(model, cfg, device, num_samples, sample_index, waypoint_index, max_v, max_w, rate_hz)
        self.goal, self.spacing, self.step_m = goal, int(context_spacing), float(step_m)
        self.context_frames = cfg["context_size"] * self.spacing + 1
        self.map_size = int(cfg["map"]["size"])
        self.name = f"mapmad_{goal}"

    def ready(self, obs: Observation) -> bool:
        return len(obs.frames) >= 1  # missing past pictures repeat the first one

    @property
    def metres_per_unit(self) -> float:
        return self.step_m

    def encoder_inputs(self, obs: Observation) -> Dict[str, torch.Tensor]:
        frames = [obs.frames[i] for i in context_indices(len(obs.frames), self.cfg["context_size"], self.spacing)]
        out = {"obs_img": self._pictures(frames)}
        if self.goal == "photo":
            out["goal_img"], gmask = self._pictures([obs.goal_image]), 0
        else:
            out["goal_img"], gmask = self._pictures([obs.frames[-1]]), 1
        if self.goal == "map":
            out["map_img"], mmask = torch.from_numpy(np.array(obs.extras["map"], dtype=np.float32))[None], 0
        else:
            out["map_img"], mmask = torch.zeros((1, 3, self.map_size, self.map_size)), 1
        out["map_img"] = out["map_img"].to(self.device)
        out["input_goal_mask"] = torch.full((1,), gmask, dtype=torch.long, device=self.device)
        out["input_map_mask"] = torch.full((1,), mmask, dtype=torch.long, device=self.device)
        return out
