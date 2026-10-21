"""MapMaD v2 inference entry point (Phase 10 item 10.5; used by Phases 11-12): the last 4 camera frames + the near and
far map windows + masks -> the 8 waypoints NoMaD's diffusion head samples, and the distance head.

    net = MapMaDV2Inference(checkpoint, cfg, device, checkpoint_sha256="...")
    out = net.predict(frames, near, far, seed=episode_seed)        # maps shown, photo hidden (search)
    out = net.predict(frames, near, far, near_shown=False, far_shown=False, seed=s)   # = NoMaD explore

- frames: 4 RGB uint8 pictures (H, W, 3), oldest first, at the training spacing (control steps t-15, t-10, t-5, t;
  closed_loop.mapmad_policy.context_indices picks them from a picture buffer);
- near, far: float32 [4, 64, 64] (obstacle, explored, frontier, target; robot-centred, forward = up), as the loader
  builds them (map_sample_v2); a hidden window is replaced by zeros and masked;
- goal_image: None (photo hidden, the current picture as placeholder, as in training) or a goal photo (photo shown).
Pictures go through the deployment transform (closed_loop.deployment.transform_images), as the Phase 5 policies.
Output: `waypoints` (num_samples, 8, 2) in normalised units (get_action), `waypoints_m` (x step_m = 0.25 m: 0.05 m x
waypoint spacing 5), `distance` (num_samples,) from the distance head (meaningful with a photo goal only). All noise
comes from one torch.Generator seeded with `seed`, so a call is reproducible.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

from vint_train.mapmad.closed_loop.deployment import transform_images
from vint_train.mapmad.closed_loop.mapmad_policy import check_sha256


class MapMaDV2Inference:
    """A v2 checkpoint ready to predict (module docstring)."""

    def __init__(self, checkpoint: str, cfg: Dict[str, Any], device: torch.device,
                 checkpoint_sha256: Optional[str] = None, num_samples: int = 8, step_m: float = 0.25,
                 model: Optional[torch.nn.Module] = None) -> None:
        from vint_train.mapmad.weights import load_checkpoint

        if int(cfg["map"].get("tokens", 1)) != 2:
            raise ValueError("MapMaDV2Inference needs a v2 config (map.tokens: 2)")
        if checkpoint_sha256 is not None:
            check_sha256(checkpoint, checkpoint_sha256)
        self.model = model if model is not None else load_checkpoint(cfg, checkpoint, device, map_input=True)
        self.cfg, self.device = cfg, device
        self.num_samples, self.step_m = int(num_samples), float(step_m)
        self.image_size = list(cfg["image_size"])
        self.scheduler = DDPMScheduler(num_train_timesteps=cfg["num_diffusion_iters"], beta_schedule="squaredcos_cap_v2",
                                       clip_sample=True, prediction_type="epsilon")

    def _pictures(self, arrays: Sequence[np.ndarray]) -> torch.Tensor:
        return transform_images([PILImage.fromarray(a) for a in arrays], self.image_size, center_crop=False).to(self.device)

    def encoder_inputs(self, frames: Sequence[np.ndarray], near: np.ndarray, far: np.ndarray, near_shown: bool,
                       far_shown: bool, goal_image: Optional[np.ndarray]) -> Dict[str, torch.Tensor]:
        """The vision encoder's keyword inputs for one step (batch of 1)."""
        if len(frames) != self.cfg["context_size"] + 1:
            raise ValueError(f"need {self.cfg['context_size'] + 1} frames, got {len(frames)}")
        maps = np.zeros((2, 4, 64, 64), dtype=np.float32)
        if near_shown:
            maps[0] = near
        if far_shown:
            maps[1] = far
        photo = goal_image is not None
        return {"obs_img": self._pictures(frames),
                "goal_img": self._pictures([goal_image if photo else frames[-1]]),
                "input_goal_mask": torch.full((1,), 0 if photo else 1, dtype=torch.long, device=self.device),
                "map_img": torch.from_numpy(maps)[None].to(self.device),
                "input_map_mask": torch.tensor([[0 if near_shown else 1, 0 if far_shown else 1]], dtype=torch.long,
                                               device=self.device)}

    @torch.no_grad()
    def predict(self, frames: Sequence[np.ndarray], near: np.ndarray, far: np.ndarray, near_shown: bool = True,
                far_shown: bool = True, goal_image: Optional[np.ndarray] = None, seed: int = 0) -> Dict[str, np.ndarray]:
        """Waypoints (normalised and metres) and distances of num_samples diffusion samples (module docstring)."""
        from vint_train.training.train_utils import get_action

        g = torch.Generator(device=self.device).manual_seed(int(seed))
        cond = self.model("vision_encoder", **self.encoder_inputs(frames, near, far, near_shown, far_shown, goal_image))
        cond = cond.repeat(self.num_samples, 1)
        naction = torch.randn((self.num_samples, self.cfg["len_traj_pred"], 2), generator=g, device=self.device)
        self.scheduler.set_timesteps(self.cfg["num_diffusion_iters"])
        for k in self.scheduler.timesteps[:]:
            pred = self.model("noise_pred_net", sample=naction, timestep=k, global_cond=cond)
            naction = self.scheduler.step(model_output=pred, timestep=k, sample=naction, generator=g).prev_sample
        waypoints = get_action(naction).cpu().numpy()
        distance = self.model("dist_pred_net", obsgoal_cond=cond).reshape(-1).cpu().numpy()
        return {"waypoints": waypoints, "waypoints_m": waypoints * self.step_m, "distance": distance}
