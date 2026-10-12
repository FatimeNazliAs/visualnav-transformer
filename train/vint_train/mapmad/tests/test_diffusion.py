"""Per-item diffusion noise (vint_train/mapmad/diffusion.py, Phase 5 code list item 1)."""

import torch

from vint_train.mapmad.diffusion import ddpm_step, item_noise, reverse_diffusion
from vint_train.mapmad.reference_forward import make_noise_scheduler

CFG = {"num_diffusion_iters": 10}


def scheduler():
    s = make_noise_scheduler(CFG)
    s.set_timesteps(CFG["num_diffusion_iters"])
    return s


def test_ddpm_step_equals_scheduler_step_with_the_same_noise():
    s = scheduler()
    g = torch.Generator().manual_seed(3)
    sample, out = torch.randn((5, 8, 2), generator=g), torch.randn((5, 8, 2), generator=g)
    for k in s.timesteps:
        ref = s.step(model_output=out, timestep=k, sample=sample, generator=torch.Generator().manual_seed(11)).prev_sample
        noise = torch.randn((5, 8, 2), generator=torch.Generator().manual_seed(11))
        assert torch.equal(ddpm_step(s, out, k, sample, noise), ref)


class FakeModel:
    """noise_pred_net = a fixed function of (sample, cond); enough to follow the noise through the loop."""

    def __call__(self, name, sample, timestep, global_cond):
        return 0.3 * sample + global_cond[:, None, :2]


def test_item_result_does_not_depend_on_batch():
    s = scheduler()
    cond = torch.randn((6, 4), generator=torch.Generator().manual_seed(0))
    ids = [10, 11, 12, 13, 14, 15]
    whole = reverse_diffusion(FakeModel(), cond, ids, s, 8, seed=0)
    parts = torch.cat([reverse_diffusion(FakeModel(), cond[i:i + 2], ids[i:i + 2], s, 8, seed=0) for i in (4, 0, 2)])
    reordered = torch.cat([whole[4:6], whole[0:2], whole[2:4]])
    assert torch.equal(parts, reordered)


def test_starting_noise_is_the_old_per_id_noise():
    """The first draw of an item's generator is the starting noise offline_eval used before Phase 5."""
    start, steps = item_noise([7, 8], 8, 10, seed=0)
    old = torch.stack([torch.randn((8, 2), generator=torch.Generator().manual_seed(i)) for i in (7, 8)])
    assert torch.equal(start, old) and steps.shape == (2, 10, 8, 2)
