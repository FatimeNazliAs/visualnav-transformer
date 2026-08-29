# behaviour-failure-analysis/common/denoise.py
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

from common import actions as action_space
from common import settings

# The action-space arithmetic lives in common/actions.py, which imports numpy and
# nothing else, so it can be exercised without a GPU. Re-exported here because
# `denoise.to_waypoints` is the name three phases and the Notion pages already
# use, and because a caller holding a Denoised wants it right there.
to_waypoints = action_space.to_waypoints


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
        """Where the clean sequence goes, in metres — (NUM_ACTIONS, ACTION_DIM)."""
        return to_waypoints(self.actions)

    @property
    def paths(self) -> np.ndarray:
        """Every step of the descent as a path — (K+1, NUM_ACTIONS, ACTION_DIM)."""
        return to_waypoints(self.trajectory)


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


def corrupt(actions: np.ndarray, *, seed: int | None = None) -> np.ndarray:
    """
    The forward process: a real action sequence, buried under noise step by step.

    Returns (K+1, NUM_ACTIONS, ACTION_DIM), index k holding `actions` with k
    steps' worth of noise on it — so index 0 is the untouched sequence and
    index K is indistinguishable from a random draw.

    This is the half of diffusion that training does and inference never
    touches, and it is the half that explains the other. `denoise` above is
    trained to undo exactly this: pick a random k, corrupt a real sequence to
    that level, and ask the network which part of the result was the noise.
    Every step of the descent is that same question asked again.

    Uses the scheduler's own add_noise, so the amount added at each k matches
    the schedule the checkpoint was trained under rather than an approximation
    of it. One noise draw is shared across all K+1 levels: the ladder is meant
    to show one sequence disappearing, and re-drawing per level would make each
    rung a different picture that happens to be noisier.
    """
    if seed is not None:
        torch.manual_seed(seed)

    scheduler = build_scheduler()
    clean = torch.as_tensor(np.asarray(actions), dtype=torch.float32)[None]
    noise = torch.randn(clean.shape)

    rungs = [clean[0].numpy()]
    for step in reversed(scheduler.timesteps):
        rungs.append(scheduler.add_noise(clean, noise, step.unsqueeze(0))[0].numpy())
    return np.stack(rungs)


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
