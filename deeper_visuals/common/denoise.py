# deeper_visuals/common/denoise.py
"""
The diffusion half of NoMaD: c_t in, an action sequence out.

common/model.py deliberately stops at c_t — VisionEncoding names `context` as
"what P4's diffusion head is conditioned on" and goes no further. This module is
the other side of that boundary, and it exists before P4 does on purpose: the
K-step loop was already written twice (common/smoke_test.py, and the frozen
debug_visuals/visualize_stage5.py) with two more callers coming. A third and
fourth copy is exactly the drift that hooking the real forward pass was meant to
avoid everywhere else in this library.

The scheduler settings are not tunable. They must match the ones NoMaD trains
and evaluates under (train.py builds the same DDPMScheduler), so they live here
beside settings.K_DENOISING rather than at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from deeper_visuals.common import settings


@dataclass(frozen=True)
class Denoised:
    """
    One denoising run: the action sequence, and every step that led to it.

    `trajectory` is the whole descent — (K+1, NUM_ACTIONS, ACTION_DIM), pure
    noise first and the clean sequence last. It is kept because it *is* P4's
    figure; a caller that only wants the answer reads `actions`.
    """

    actions: np.ndarray      # (NUM_ACTIONS, ACTION_DIM) — the clean sequence
    trajectory: np.ndarray   # (K+1, NUM_ACTIONS, ACTION_DIM) — noise -> clean
    seed: int | None

    @property
    def n_steps(self) -> int:
        """K, the number of denoising steps taken."""
        return len(self.trajectory) - 1

    @property
    def path(self) -> np.ndarray:
        """The actions accumulated into a path, (NUM_ACTIONS, ACTION_DIM)."""
        return self.actions.cumsum(axis=0)


def build_scheduler() -> DDPMScheduler:
    """The scheduler NoMaD trains under. Not a knob — it must match training."""
    scheduler = DDPMScheduler(
        num_train_timesteps=settings.K_DENOISING,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(settings.K_DENOISING)
    return scheduler


def denoise(model, context, device: str, *, seed: int | None = None) -> Denoised:
    """
    Run the K-step reverse diffusion conditioned on one context vector.

    `context` is c_t — a (1, ENCODING_SIZE) tensor or array, straight off
    common.model.encode_tokens. `seed` fixes the starting noise, which is what
    makes two runs comparable: P3 needs the same noise under two goal masks, and
    P5's multimodality is the same c_t under many seeds. Left None, the ambient
    torch RNG decides and the run is not reproducible.
    """
    if seed is not None:
        torch.manual_seed(seed)

    scheduler = build_scheduler()
    conditioning = torch.as_tensor(context).to(device)
    actions = torch.randn(
        (1, settings.NUM_ACTIONS, settings.ACTION_DIM), device=device)

    steps = [actions.cpu().numpy()[0]]
    with torch.no_grad():
        for step in scheduler.timesteps:
            noise = model("noise_pred_net", sample=actions,
                          timestep=step.unsqueeze(0).to(device),
                          global_cond=conditioning)
            actions = scheduler.step(
                model_output=noise, timestep=step, sample=actions).prev_sample
            steps.append(actions.cpu().numpy()[0])

    return Denoised(actions=steps[-1], trajectory=np.stack(steps), seed=seed)


def distance_to_goal(model, context, device: str) -> float:
    """
    The distance head's read on c_t, in steps.

    Lives here rather than in model.py because it answers the same question the
    diffusion head does — what the policy makes of this context — and callers
    that want one usually want the other.
    """
    conditioning = torch.as_tensor(context).to(device)
    with torch.no_grad():
        return float(model("dist_pred_net", obsgoal_cond=conditioning).item())
