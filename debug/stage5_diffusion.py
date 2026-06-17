# debug/stage5_diffusion.py
import torch
from diffusers import DDPMScheduler
from debug.config import DEVICE, K_DENOISING, NUM_ACTIONS, ACTION_DIM


def run_diffusion(model, ct, verbose=True):
    """
    Reverse diffusion: pure noise → clean action sequence.

    Starts from a^K ~ N(0, I) and runs K=10 denoising steps.
    At each step, the U-Net εθ predicts the noise in a^k,
    and the scheduler subtracts it to produce a^(k-1).

    After K steps: a^0 is the clean predicted action sequence.

    Shape through the loop:
        a^K  : (1, 8, 2)  ← pure Gaussian noise
        εθ   : (1, 8, 2)  ← predicted noise at each step
        a^0  : (1, 8, 2)  ← clean actions (linear_vel, angular_vel)

    Running this twice with the same ct gives DIFFERENT a^0 —
    that is the multimodal property of diffusion policies.
    """
    scheduler = _get_scheduler(model)

    # Sample fresh noise — do NOT reuse across runs
    a_k = torch.randn(1, NUM_ACTIONS, ACTION_DIM, device=DEVICE)

    if verbose:
        print(f"  Starting from noise a^K : {tuple(a_k.shape)}")
        print(f"  {'Step':>6}  {'||a^k||':>10}  {'||eps_pred||':>14}")
        print(f"  {'──────':>6}  {'──────':>10}  {'──────────────':>14}")

    noise_pred_net = getattr(model, 'noise_pred_net', None)

    with torch.no_grad():
        for t_step in scheduler.timesteps:

            if noise_pred_net is not None:
                # ConditionalUnet1D takes sample as (B, T, action_dim) directly —
                # it permutes to channels-first internally for its own conv1d
                # stack and permutes back before returning. Pre-permuting here
                # double-flips the layout: (1,8,2) -> our permute -> (1,2,8)
                # -> its internal permute -> (1,8,2) again, but now treated as
                # (B, C=8, T=2), which is why the conv saw "8 channels".
                eps_pred = noise_pred_net(
                    sample=a_k,
                    timestep=t_step,
                    global_cond=ct,
                )
            else:
                print("  WARNING: noise_pred_net not found — using zero noise (dummy run)")
                eps_pred = torch.zeros_like(a_k)

            # Scheduler step: a^k → a^(k-1) — DDPMScheduler is shape-agnostic
            # elementwise arithmetic, so no permuting needed here either.
            result = scheduler.step(eps_pred, t_step, a_k)
            a_k = result.prev_sample

            if verbose:
                print(f"  {int(t_step):>6}  {a_k.norm().item():>10.4f}  {eps_pred.norm().item():>14.4f}")

    if verbose:
        print(f"\n  Output a^0 : {tuple(a_k.shape)}  (batch, H=8 steps, 2 velocities)")

    return a_k


def _get_scheduler(model):
    if hasattr(model, 'noise_scheduler'):
        return model.noise_scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=K_DENOISING,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(K_DENOISING)
    return scheduler