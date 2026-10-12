"""Reverse diffusion with noise fixed PER ITEM (Phase 5 code list item 1): an item's waypoints depend only on its
own id, never on the batch it lands in or on its position there.

NoMaD's sampler draws the scheduler noise of every reverse step for the whole batch from one generator, so item 7
of batch 3 gets different noise than the same item scored in another batch (Phase 4 lesson: the 1,000-item
training-check subset and the full harness gave different numbers for the same items). Here every item owns one
CPU generator seeded with `seed + id` that draws, in this order, its starting noise (len_traj_pred, 2) -- the very
tensor `offline_eval` used before -- and then one (len_traj_pred, 2) noise per reverse step.

`ddpm_step` is diffusers 0.11.1 `DDPMScheduler.step` (epsilon prediction, clip_sample, fixed_small variance -- the
NoMaD settings) with the noise passed in instead of drawn; `tests/test_diffusion.py` checks it equals
`scheduler.step` bit for bit when both get the same noise.
"""

from typing import Sequence, Tuple

import torch


def item_noise(ids: Sequence[int], horizon: int, n_steps: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(starting noise (B, horizon, 2), step noise (B, n_steps, horizon, 2)) from one generator per item id."""
    start, steps = [], []
    for i in ids:
        g = torch.Generator().manual_seed(seed + int(i))
        start.append(torch.randn((horizon, 2), generator=g))
        steps.append(torch.randn((n_steps, horizon, 2), generator=g))
    return torch.stack(start), torch.stack(steps)


def ddpm_step(scheduler, model_output: torch.Tensor, timestep, sample: torch.Tensor,
              noise: torch.Tensor) -> torch.Tensor:
    """One reverse step x_t -> x_{t-1} of NoMaD's DDPMScheduler with the given noise (added only for t > 0)."""
    cfg = scheduler.config
    if cfg.prediction_type != "epsilon" or cfg.variance_type != "fixed_small":
        raise ValueError("ddpm_step mirrors NoMaD's scheduler only (epsilon, fixed_small)")
    t = timestep
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    if cfg.clip_sample:
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
    current_sample_coeff = scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
    if t > 0:
        variance = (scheduler._get_variance(t) ** 0.5) * noise.to(device=model_output.device, dtype=model_output.dtype)
        pred_prev_sample = pred_prev_sample + variance
    return pred_prev_sample


@torch.no_grad()
def reverse_diffusion(model, cond: torch.Tensor, ids: Sequence[int], scheduler, horizon: int,
                      seed: int) -> torch.Tensor:
    """Full reverse diffusion (scheduler.timesteps already set) -> normalised action deltas (B, horizon, 2),
    each item with its own noise (item_noise)."""
    device = cond.device
    start, steps = item_noise(ids, horizon, len(scheduler.timesteps), seed)
    naction = start.to(device)
    for n, k in enumerate(scheduler.timesteps[:]):
        pred = model("noise_pred_net", sample=naction, timestep=k.unsqueeze(-1).repeat(naction.shape[0]).to(device),
                     global_cond=cond)
        naction = ddpm_step(scheduler, pred, k, naction, steps[:, n])
    return naction
