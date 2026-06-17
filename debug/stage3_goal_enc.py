# debug/stage3_goal_enc.py
import torch
from debug.stage2_obs_enc import _get_backbone


def run_goal_encoder(model, obs_input, goal_input, verbose=True):
    """
    Encodes the goal image into a single 256-D token.

    The goal encoder (φ) is NOT a goal-only encoder: it was built with
    in_channels=6 (see NoMaD_ViNT.__init__) because it jointly encodes
    [most-recent observation frame ++ goal image] concatenated channel-wise
    (3+3=6). This mirrors NoMaD_ViNT.forward()'s own obsgoal_img step.

    This is the token that gets MASKED when mask=True (exploration mode).

    (B, P, C, H, W), (B, C, H, W) → (B, 256)
    """
    backbone = _get_backbone(model)

    last_obs_frame = obs_input[:, -1]                       # (B, 3, H, W) — most recent frame
    obsgoal_img = torch.cat([last_obs_frame, goal_input], dim=1)  # (B, 6, H, W)

    with torch.no_grad():
        # Same trap as stage 2: calling backbone.goal_encoder(x) directly
        # would run EfficientNet's full forward() → its 1000-way classifier.
        # Stop after pooling and use the model's own compression layer instead.
        goal_token = backbone.goal_encoder.extract_features(obsgoal_img)
        goal_token = backbone.goal_encoder._avg_pooling(goal_token)
        goal_token = goal_token.flatten(start_dim=1)          # (B, 1280)
        goal_token = backbone.goal_encoder._dropout(goal_token)
        goal_token = backbone.compress_goal_enc(goal_token)   # (B, 256)

    if verbose:
        print(f"  Input  goal_input : {tuple(goal_input.shape)}")
        print(f"  Output goal_token : {tuple(goal_token.shape)}  (batch, 256)")
        print(f"  Value range       : [{goal_token.min():.3f}, {goal_token.max():.3f}]")

    return goal_token