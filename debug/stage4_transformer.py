# debug/stage4_transformer.py
import torch
from debug.stage2_obs_enc import _get_backbone


def run_transformer(model, obs_input, goal_input, mask: bool, verbose=True):
    """
    Runs the full Transformer forward pass with or without the goal token.

    Token sequence entering the Transformer:
        [obs_t-3,  obs_t-2,  obs_t-1,  obs_t,  goal]   ← 5 tokens of 256-D each
         ↑ 4 observation tokens ↑                        ↑ 1 goal token

    mask=False → NAVIGATION mode: goal token participates in attention
    mask=True  → EXPLORATION mode: goal token is blocked (input_goal_mask=1)

    All 5 output tokens are average-pooled → context vector ct (B, 256).

    dist_pred is NOT produced by NoMaD_ViNT — it only returns ct. Distance
    comes from the separate dist_pred_net head, conditioned on ct, exactly
    like NoMaD.forward(func_name="dist_pred_net", ...) does for training.
    """
    backbone = _get_backbone(model)   # = model.vision_encoder (NoMaD_ViNT)

    # NoMaD_ViNT expects obs frames channel-concatenated: (B, P*C, H, W),
    # not the stacked (B, P, C, H, W) shape used in stages 1-3.
    B, P, C, H, W = obs_input.shape
    obs_img = obs_input.view(B, P * C, H, W)

    # input_goal_mask is a per-sample LongTensor (0 = goal visible, 1 = goal
    # masked) — NOT a plain bool. It's used as an index into a lookup table
    # of attention masks inside NoMaD_ViNT.forward().
    input_goal_mask = torch.full((B,), int(mask), dtype=torch.long, device=obs_input.device)

    with torch.no_grad():
        ct = backbone(obs_img, goal_input, input_goal_mask=input_goal_mask)
        dist_pred = model.dist_pred_net(ct)

    if verbose:
        mode = "EXPLORATION (m=1, goal masked)" if mask else "NAVIGATION (m=0, goal active)"
        print(f"  Mode      : {mode}")
        print(f"  ct        : {tuple(ct.shape)}  (batch, 256)")
        print(f"  dist_pred : {tuple(dist_pred.shape)}  value = {dist_pred[0, 0].item():.2f} steps")
        print(f"  ct range  : [{ct.min():.3f}, {ct.max():.3f}]")

    return ct, dist_pred


def compare_masked_vs_unmasked(model, obs_input, goal_input, verbose=True):
    """
    Runs both modes and checks that masking genuinely changes ct.
    A mean absolute difference near zero means masking is broken.
    """
    ct_nav,     _ = run_transformer(model, obs_input, goal_input, mask=False, verbose=False)
    ct_explore, _ = run_transformer(model, obs_input, goal_input, mask=True,  verbose=False)

    diff = (ct_nav - ct_explore).abs().mean().item()

    if verbose:
        print(f"  ct_navigation vs ct_exploration")
        print(f"  Mean absolute difference: {diff:.4f}")
        if diff > 0.01:
            print(f"  ✓ Masking is working — ct changes when goal is masked")
        else:
            print(f"  ✗ WARNING: masking has no effect — investigate backbone.forward()")

    return ct_nav, ct_explore