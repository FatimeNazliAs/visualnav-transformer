"""NoMaD's brain: load a checkpoint, and turn one observation into one waypoint.

This is the policy half of `deployment/src/navigate.py`, ported verbatim. Only
the ends change (plan §4): the observation arrives as a list of PIL frames
instead of off a ROS topic, and the waypoint is returned instead of published.
Everything between — the transform, the goal mask, the localization window, the
diffusion loop, `get_action`, waypoint #2 — is the deployment code, in the
deployment order, with the deployment defaults.

Ported alongside it, from `deployment/src/utils.py`, are `transform_images`,
`to_numpy` and the NoMaD branch of `load_model`. They are copied rather than
imported because that module does `from sensor_msgs.msg import Image` at the
top, and ROS message packages are not installed in the sim container (nor
should they be: there is no ROS graph here).

The defaults below are `navigate.py`'s own argparse defaults, which is what the
real robot runs with. Plan decision E says mirror them and tune nothing.
"""

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision import transforms

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.training.train_utils import get_action

from driver import DriverConfig

# ImageNet statistics, as in deployment/src/utils.py.
IMAGENET_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(pil_imgs, image_size):
    """Transforms a list of PIL image to a torch tensor.

    `deployment/src/utils.py`, minus the `center_crop` branch: navigate.py
    calls it with `center_crop=False` for NoMaD, and the sim camera is already
    rendered at the aspect ratio the checkpoint wants.
    """
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        pil_img = pil_img.resize(image_size)
        transf_img = IMAGENET_NORMALIZE(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)


def build_model(model_params):
    """Instantiate NoMaD from a training config — `load_model`'s nomad branch."""
    if model_params["vision_encoder"] != "nomad_vint":
        raise ValueError(
            "vision encoder {!r} is not supported here. Every checkpoint in "
            "configs/checkpoints.yaml is nomad_vint; wiring up another encoder "
            "without one to test against would be untested code."
            .format(model_params["vision_encoder"]))

    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=model_params["encoding_size"],
        context_size=model_params["context_size"],
        mha_num_attention_heads=model_params["mha_num_attention_heads"],
        mha_num_attention_layers=model_params["mha_num_attention_layers"],
        mha_ff_dim_factor=model_params["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=model_params["encoding_size"],
        down_dims=model_params["down_dims"],
        cond_predict_scale=model_params["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=model_params["encoding_size"])

    return NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )


def load_model(weights_path, model_params, device):
    """Load NoMaD weights — `load_model`'s nomad path, including strict=False.

    `strict=False` is the deployment behaviour and is deliberate: the training
    checkpoint carries keys the inference model has no place for.
    """
    model = build_model(model_params)
    checkpoint = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    return model.to(device)


def localization_window(closest_node, goal_node, radius):
    """The slice of the trail the distance head scores, as (start, end) inclusive.

    navigate.py's two clamps, verbatim. `start` floors at the first node and
    `end` ceilings at the goal, so the window shrinks rather than running off
    either end of the trail.
    """
    start = max(closest_node - radius, 0)
    end = min(closest_node + radius + 1, goal_node)
    return start, end


def window_size(start, end):
    """How many nodes `localization_window` selected — the slice is [start:end+1]."""
    return end + 1 - start


def localize(distances, start, num_window_nodes, close_threshold):
    """Pick the current node and the subgoal from the distance head's scores.

    Returns `(closest_node, subgoal_offset)`, and the two are deliberately
    different kinds of index:

      * `closest_node` is an **absolute** trail index — it is what the next tick
        centres its window on, and what "reached the goal" is tested against.
      * `subgoal_offset` is **relative to the window** — it indexes the encoder
        output for this window, which has `num_window_nodes` rows.

    Confusing the two is not hypothetical. Upstream shipped exactly that bug and
    fixed it in commit 7b5b24c ("Fix closest node update for topomap
    localization"), which replaced `closest_node = np.argmin(distances)` with
    `start + min_dist_idx`. An off-by-one here does not crash: it steers the
    robot confidently at the wrong node, which looks fine in a replay and
    quietly invalidates every metric downstream. Hence this function exists
    apart from the model, where `tests/test_localization.py` can pin it without
    torch, a checkpoint or a GPU.

    The arithmetic itself is navigate.py's, unchanged.
    """
    min_idx = int(np.argmin(distances))
    closest_node = min_idx + start
    subgoal_offset = min(min_idx + int(distances[min_idx] < close_threshold),
                         num_window_nodes - 1)
    return closest_node, subgoal_offset


class PolicyStep:
    """What one policy call decided, and what it saw when deciding."""

    def __init__(self, waypoint, closest_node, subgoal_node, distances, samples):
        # Normalized units, straight out of get_action. The caller scales it
        # into metres — see bridge.NomadBridge._waypoint_to_metres.
        self.waypoint = waypoint
        self.closest_node = closest_node
        self.subgoal_node = subgoal_node
        # Predicted temporal distance to each node in the localization window.
        self.distances = distances
        # All num_samples action sequences, for inspection and overlays.
        self.samples = samples

    def window_start(self):
        """Which absolute trail index `distances[0]` scored.

        The step names its two nodes by *absolute* trail index but carries the
        distance head's scores by window offset, so anything reading a score
        back out has to recover the window. `localize` chose `closest_node` as
        the window entry the head scored lowest, which fixes the origin.

        Returns None when there are no scores to index.
        """
        distances = np.asarray(self.distances, dtype=float)
        if distances.size == 0:
            return None
        return int(self.closest_node) - int(np.argmin(distances))

    def _score(self, node):
        """The head's reading for one absolute trail index, or None.

        None rather than a guess: a mislabelled temporal distance is worse than
        a blank, whether it is printed under a picture or logged in a trace.
        """
        start = self.window_start()
        if start is None:
            return None
        distances = np.asarray(self.distances, dtype=float)
        offset = int(node) - start
        if not 0 <= offset < distances.size:
            return None
        return float(distances[offset])

    def closest_distance(self):
        """How far the head thinks the node it localized onto is.

        The minimum of the window by construction, and the number that decides
        whether the subgoal advances (`close_threshold`). Logged per tick from
        P5 on, because a run whose readings sit far above that threshold is
        one where the trail never advances — a failure mode no metric names.
        """
        return self._score(self.closest_node)

    def subgoal_distance(self):
        """How far the head thinks the node being steered at is."""
        return self._score(self.subgoal_node)


class NomadPolicy:
    """A loaded NoMaD checkpoint that answers "where next" for one observation."""

    def __init__(self, spec, device, driver=None):
        self.spec = spec
        self.device = device
        # The four steering knobs, as one config object — see DriverConfig for
        # why they are not four keyword arguments any more.
        self.driver = driver or DriverConfig()

        self.model_params = spec.model_params
        self.model = load_model(spec.weights_path, self.model_params, device)
        self.model.eval()

        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def encode_topomap(self, topomap):
        """Pre-transform the topomap once; it does not change during an episode."""
        return [transform_images(node, self.spec.image_size).to(self.device)
                for node in topomap]

    def act(self, context_frames, encoded_topomap, closest_node, goal_node):
        """One policy step: localize in the topomap, then predict a waypoint.

        `context_frames` is oldest-first and must hold exactly context_size+1
        frames — the same thing navigate.py waits for its queue to accumulate.
        """
        obs_images = transform_images(context_frames, self.spec.image_size)
        # A faithful no-op from navigate.py: split into per-frame 3-channel
        # chunks and concatenate them back along the same axis. Kept so this
        # pipeline is diff-able against the deployment one.
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1)
        obs_images = obs_images.to(self.device)

        # Goal masking off: the goal token stays visible, which is NoMaD's
        # goal-directed navigation behaviour. Exploration would set this to 1.
        mask = torch.zeros(1).long().to(self.device)

        start, end = localization_window(closest_node, goal_node, self.driver.radius)
        goal_image = torch.concat(encoded_topomap[start:end + 1], dim=0)

        obsgoal_cond = self.model(
            "vision_encoder",
            obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
            goal_img=goal_image,
            input_goal_mask=mask.repeat(len(goal_image)),
        )
        dists = to_numpy(self.model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten())
        closest_node, subgoal_offset = localize(
            dists, start, len(obsgoal_cond), self.driver.close_threshold)
        obs_cond = obsgoal_cond[subgoal_offset].unsqueeze(0)

        naction = self._denoise(obs_cond)
        return PolicyStep(
            waypoint=naction[0][self.driver.waypoint],
            closest_node=closest_node,
            subgoal_node=start + subgoal_offset,
            distances=dists,
            samples=naction,
        )

    def _denoise(self, obs_cond):
        """Reverse diffusion, verbatim from navigate.py's inner loop."""
        with torch.no_grad():
            # encoder vision features
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.driver.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.driver.num_samples, 1, 1)

            # initialize action from Gaussian noise
            noisy_action = torch.randn(
                (self.driver.num_samples, self.model_params["len_traj_pred"], 2),
                device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps[:]:
                # predict noise
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )
                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction,
                ).prev_sample

        return to_numpy(get_action(naction))
