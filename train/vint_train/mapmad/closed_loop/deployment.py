"""NoMaD's deployment rules, copied without ROS so they run in the simulator loop.

Sources (deployment/src, which import rospy and so cannot be imported here):
- transform_images: utils.py — PIL resize to image_size (no crop), ToTensor, ImageNet normalisation,
  pictures concatenated along the channel axis, oldest first;
- pd_controller + clip_angle: pd_controller.py — waypoint (dx, dy) in metres -> (v, w) with DT = 1 / frame_rate;
- the waypoint scale: navigate.py / explore.py multiply the chosen normalised waypoint by MAX_V / RATE.
tests/test_deployment_rules.py checks these against the original files (ROS stubbed out).
"""

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from torchvision import transforms

from vint_train.data.data_utils import IMAGE_ASPECT_RATIO

EPS = 1e-8


def transform_images(pil_imgs: Union[PILImage.Image, List[PILImage.Image]], image_size: Sequence[int],
                     center_crop: bool = False) -> torch.Tensor:
    """deployment/src/utils.py transform_images: list of PIL pictures -> (1, 3 * n, h, w) tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


def clip_angle(theta: float) -> float:
    """deployment/src/pd_controller.py clip_angle: angle into [-pi, pi]."""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi


def pd_controller(waypoint: np.ndarray, max_v: float, max_w: float, dt: float) -> Tuple[float, float]:
    """deployment/src/pd_controller.py pd_controller, with its module constants passed in."""
    assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    # this controller only uses the predicted heading if dx and dy near zero
    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0
        w = clip_angle(np.arctan2(hy, hx)) / dt
    elif np.abs(dx) < EPS:
        v = 0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / dx) / dt
    v = np.clip(v, 0, max_v)
    w = np.clip(w, -max_w, max_w)
    return float(v), float(w)
