"""Prove that gradient accumulation is arithmetically equivalent to one large batch.

`train_nomad` accumulates `gradient_accumulation_steps` microbatches, scaling each
microbatch loss by 1/steps. That is exact only when the loss is linear in the batch.
NoMaD's `action_reduce` is not:

    (loss * action_mask).mean() / (action_mask.mean() + 1e-2)

so on real data an accumulated run legitimately differs from a full-batch run, and a
loss-curve comparison cannot distinguish that expected difference from an accumulation
bug. This check removes the nonlinearity by forcing `action_mask` to all ones, which
makes `action_reduce` an exact scalar multiple of the mean. Under that condition the
accumulated gradient MUST equal the full-batch gradient.

Run inside the container:
    CUDA_VISIBLE_DEVICES=1 python ablation/check_grad_accum.py
"""

import argparse

import torch
import torch.nn.functional as F
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

ALPHA = 1e-4
ENCODING_SIZE = 256
LEN_TRAJ_PRED = 8
ACTION_DIM = 2


def build_model(context_size, device):
    torch.manual_seed(0)
    vision_encoder = replace_bn_with_gn(
        NoMaD_ViNT(
            obs_encoding_size=ENCODING_SIZE,
            context_size=context_size,
            mha_num_attention_heads=4,
            mha_num_attention_layers=4,
            mha_ff_dim_factor=4,
        )
    )
    noise_pred_net = ConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=ENCODING_SIZE,
        down_dims=[64, 128, 256],
        cond_predict_scale=False,
    )
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=DenseNetwork(embedding_dim=ENCODING_SIZE),
    )
    # eval() disables the EfficientNet dropout. Dropout draws different masks for the
    # full-batch pass and the accumulated passes, which would swamp the comparison with
    # sampling noise unrelated to the accumulation arithmetic.
    return model.to(device).eval()


def make_batch(batch_size, context_size, image_size, device):
    """A fixed synthetic batch with an all-ones action mask."""
    generator = torch.Generator(device="cpu").manual_seed(1)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    return {
        "obs_image": randn(batch_size, 3 * (context_size + 1), *image_size),
        "goal_image": randn(batch_size, 3, *image_size),
        "noisy_action": randn(batch_size, LEN_TRAJ_PRED, ACTION_DIM),
        "noise": randn(batch_size, LEN_TRAJ_PRED, ACTION_DIM),
        "distance": randn(batch_size, 1),
        "timesteps": torch.zeros(batch_size, dtype=torch.long, device=device),
        "goal_mask": torch.zeros(batch_size, dtype=torch.long, device=device),
        "action_mask": torch.ones(batch_size, device=device),
    }


def compute_loss(model, batch, lo, hi):
    """The loss of train_nomad, over batch[lo:hi]."""
    sl = slice(lo, hi)
    action_mask = batch["action_mask"][sl]

    cond = model(
        "vision_encoder",
        obs_img=batch["obs_image"][sl],
        goal_img=batch["goal_image"][sl],
        input_goal_mask=batch["goal_mask"][sl],
    )
    dist_pred = model("dist_pred_net", obsgoal_cond=cond.flatten(start_dim=1))
    dist_loss = F.mse_loss(dist_pred, batch["distance"][sl]) / (action_mask.mean() + 1e-2)

    noise_pred = model(
        "noise_pred_net",
        sample=batch["noisy_action"][sl],
        timestep=batch["timesteps"][sl],
        global_cond=cond,
    )

    def action_reduce(unreduced_loss):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    diffusion_loss = action_reduce(
        F.mse_loss(noise_pred, batch["noise"][sl], reduction="none")
    )
    return ALPHA * dist_loss + (1 - ALPHA) * diffusion_loss


def gradients(model):
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-size", type=int, default=3)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    assert args.effective_batch % args.accumulation_steps == 0
    microbatch = args.effective_batch // args.accumulation_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # float64 keeps the comparison about the accumulation logic rather than float32 noise.
    torch.set_default_dtype(torch.float64)

    model = build_model(args.context_size, device)
    batch = make_batch(args.effective_batch, args.context_size, (96, 96), device)

    model.zero_grad(set_to_none=True)
    compute_loss(model, batch, 0, args.effective_batch).backward()
    full_batch_grads = gradients(model)

    model.zero_grad(set_to_none=True)
    for step in range(args.accumulation_steps):
        lo = step * microbatch
        loss = compute_loss(model, batch, lo, lo + microbatch)
        (loss / args.accumulation_steps).backward()
    accumulated_grads = gradients(model)

    worst_name, worst_error = None, 0.0
    for name, full in full_batch_grads.items():
        scale = full.abs().max().item()
        if scale == 0.0:
            continue
        error = (accumulated_grads[name] - full).abs().max().item() / scale
        if error > worst_error:
            worst_name, worst_error = name, error

    print(
        f"context_size={args.context_size} effective_batch={args.effective_batch} "
        f"microbatch={microbatch} accumulation_steps={args.accumulation_steps}"
    )
    print(f"tensors compared: {len(full_batch_grads)}")
    print(f"worst relative gradient error: {worst_error:.3e} (at {worst_name})")

    if worst_error > args.tolerance:
        raise SystemExit(
            f"FAIL: gradient mismatch {worst_error:.3e} exceeds tolerance {args.tolerance:.0e}"
        )
    print(f"PASS: accumulation matches full batch within {args.tolerance:.0e}")


if __name__ == "__main__":
    main()
