"""Unchanged NoMaD as a closed-loop Policy, doing exactly what deployment/src/{navigate,explore}.py do per tick:

1. wait until context_size + 1 = 4 pictures are queued (deployment publishes no waypoint before: the robot stands);
2. transform_images -> 4 x 96 x 96, oldest first; goal photo the same way (photo arm, mask 0) or a random
   picture with the goal masked (explore arm, mask 1);
3. vision encoder -> condition, repeated for num_samples = 8; DDPM with num_diffusion_iters = 10 steps from
   Gaussian noise (squaredcos_cap_v2, clip_sample, epsilon);
4. get_action (un-normalise deltas with action_stats, cumulative sum) -> path 0, waypoint 2, times MAX_V / RATE;
5. pd_controller -> (v, w).
All random draws (noise, fake goal, scheduler) come from one torch.Generator seeded at reset.
Frame queue (deployment `callback_obs`): append the newest picture, drop the oldest beyond context_size + 1; no
padding: until 4 pictures exist, deployment publishes no waypoint and the robot stands (here: (0, 0)).
"""

import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from vint_train.mapmad.closed_loop.deployment import pd_controller, transform_images
from vint_train.mapmad.closed_loop.policy import Command, Observation, Policy


def build_nomad(cfg: Dict[str, Any]) -> torch.nn.Module:
    """NoMaD model as deployment/src/utils.py load_model builds it (model_type nomad, vision_encoder nomad_vint)."""
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
    from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

    if cfg["model_type"] != "nomad" or cfg["vision_encoder"] != "nomad_vint":
        raise ValueError("only model_type nomad with vision_encoder nomad_vint is supported")
    vision_encoder = NoMaD_ViNT(obs_encoding_size=cfg["encoding_size"], context_size=cfg["context_size"],
                                mha_num_attention_heads=cfg["mha_num_attention_heads"],
                                mha_num_attention_layers=cfg["mha_num_attention_layers"],
                                mha_ff_dim_factor=cfg["mha_ff_dim_factor"])
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(input_dim=2, global_cond_dim=cfg["encoding_size"], down_dims=cfg["down_dims"],
                                       cond_predict_scale=cfg["cond_predict_scale"])
    dist_pred_net = DenseNetwork(embedding_dim=cfg["encoding_size"])
    return NoMaD(vision_encoder=vision_encoder, noise_pred_net=noise_pred_net, dist_pred_net=dist_pred_net)


def load_nomad(weights: str, cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Build and load the official weights. Strict (deployment uses strict=False): every key must match.
    A DataParallel `module.` prefix is removed first."""
    state = torch.load(weights, map_location=device, weights_only=True)  # a plain state dict: no pickled code
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model = build_nomad(cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class NomadPolicy(Policy):
    """NoMaD with the deployment pipeline; goal photo given -> navigate, goal None -> explore."""

    name = "nomad"

    def __init__(self, weights: str, cfg: Dict[str, Any], device: torch.device, num_samples: int = 8,
                 sample_index: int = 0, waypoint_index: int = 2, max_v: float = 0.2, max_w: float = 0.4,
                 rate_hz: float = 4.0) -> None:
        from vint_train.training.train_utils import get_action

        self.cfg, self.device = cfg, device
        self.model = load_nomad(weights, cfg, device)
        self.get_action = get_action
        self.scheduler = DDPMScheduler(num_train_timesteps=cfg["num_diffusion_iters"], beta_schedule="squaredcos_cap_v2",
                                       clip_sample=True, prediction_type="epsilon")
        self.context_frames = cfg["context_size"] + 1
        self.image_size = list(cfg["image_size"])
        self.num_samples, self.sample_index, self.waypoint_index = num_samples, sample_index, waypoint_index
        self.max_v, self.max_w, self.rate_hz = max_v, max_w, rate_hz
        self.generator: Optional[torch.Generator] = None

    def reset(self, meta: Dict[str, Any]) -> None:
        """Seed Python, NumPy and torch with meta["seed"], plus a dedicated generator for the diffusion noise."""
        seed = int(meta["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def _pictures(self, arrays) -> torch.Tensor:
        return transform_images([PILImage.fromarray(a) for a in arrays], self.image_size, center_crop=False).to(self.device)

    @torch.no_grad()
    def sample_paths(self, obs: Observation) -> np.ndarray:
        """(num_samples, len_traj_pred, 2) waypoints in normalised units (as get_action returns them)."""
        obs_images = self._pictures(obs.frames)
        if obs.goal_image is None:  # explore.py: random goal picture, masked
            goal = torch.randn((1, 3, *self.image_size), generator=self.generator, device=self.device)
            mask = torch.ones(1).long().to(self.device)
        else:  # navigate.py: goal picture, not masked
            goal = self._pictures([obs.goal_image])
            mask = torch.zeros(1).long().to(self.device)
        obs_cond = self.model("vision_encoder", obs_img=obs_images, goal_img=goal, input_goal_mask=mask)
        obs_cond = obs_cond.repeat(self.num_samples, 1) if obs_cond.dim() == 2 else obs_cond.repeat(self.num_samples, 1, 1)
        naction = torch.randn((self.num_samples, self.cfg["len_traj_pred"], 2), generator=self.generator,
                              device=self.device)
        self.scheduler.set_timesteps(self.cfg["num_diffusion_iters"])
        for k in self.scheduler.timesteps[:]:
            noise_pred = self.model("noise_pred_net", sample=naction, timestep=k, global_cond=obs_cond)
            naction = self.scheduler.step(model_output=noise_pred, timestep=k, sample=naction,
                                          generator=self.generator).prev_sample
        return self.get_action(naction).cpu().numpy()

    def act(self, obs: Observation) -> Command:
        if len(obs.frames) < self.context_frames:  # deployment: no waypoint yet -> zero velocity
            return Command(0.0, 0.0, {"waiting_for_frames": len(obs.frames)})
        paths = self.sample_paths(obs)
        waypoint = paths[self.sample_index][self.waypoint_index].astype(np.float64)
        if self.cfg["normalize"]:
            waypoint = waypoint * (self.max_v / self.rate_hz)
        v, w = pd_controller(waypoint, self.max_v, self.max_w, 1.0 / self.rate_hz)
        return Command(v, w, {"waypoint_m": [float(waypoint[0]), float(waypoint[1])],
                              "path_m": (paths[self.sample_index] * (self.max_v / self.rate_hz)).round(4).tolist()})
