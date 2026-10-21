"""Scoring of the offline harness: one checkpoint, per-sample `gc_action_loss` (MSE between the expert's 8 waypoints
and the waypoints from the full 10-step reverse diffusion, normalised units, NoMaD's `_compute_losses_nomad`
formula) and the map attention share, on the frozen Habitat list (`ArmSamples`, one arm = one input mode) or on the
fixed GoStanford photo-goal items (`GoStanfordPhotoSamples`). Every sample's noise comes from its own id
(`diffusion.item_noise`), so a sample scores the same in every arm, batch and subset.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset

from vint_train.mapmad import modes as M
from vint_train.mapmad.attention import attention_shares
from vint_train.mapmad.config import loader_timeout
from vint_train.mapmad.diffusion import reverse_diffusion
from vint_train.mapmad.offline_list import LIST_SEED
from vint_train.mapmad.reference_forward import IMAGENET, make_noise_scheduler
from vint_train.mapmad.train_loop import to_batch

NOISE_SEED = 0
PHOTO_K_MIN, PHOTO_K_MAX = 4, 19  # strictly inside NoMaD's action range (3, 20)
ARMS = ("map_goal", "map_hidden", "wrong_heat", "photo", "photo_map", "explore_map")
ARM_MODE = {"map_goal": "map_goal", "map_hidden": "explore", "wrong_heat": "map_goal", "photo": "photo",
            "photo_map": "photo_map", "explore_map": "explore_map"}

# v2 (Phase 10 block item 5; p10_rules.md v2): arm -> (mode, map options). On the search list: maps (primary, noisy
# layer) vs zeroed (G10 row 3: both map tokens shown, all four layers zeroed); maps vs hidden and the other variants
# are report only. On the exploration list: explore_maps vs hidden.
SHIFT_M = 1.0
ROW3_CONTROL = "zeroed"
ARMS_V2 = ("maps", ROW3_CONTROL, "hidden", "clean", "near_only", "far_only", "target_zeroed", "frontier_zeroed",
           "target_shifted")
ARMS_V2_EXPLORE = ("explore_maps", "hidden")
ARM_MODE_V2 = {"maps": "search_map", ROW3_CONTROL: "search_map", "hidden": "explore", "clean": "search_map",
               "near_only": "search_map", "far_only": "search_map", "target_zeroed": "search_map",
               "frontier_zeroed": "search_map", "target_shifted": "search_map", "explore_maps": "explore_map",
               "photo": "photo"}


def with_row3_control(arms: List[str]) -> List[str]:
    """v2 arms to score: `arms` plus the row-3 control arm right after `maps` when `maps` is scored without it (the
    chain's pre-training report asks for maps,hidden only; rules v2 need maps - zeroed on the step-0 model too)."""
    if "maps" not in arms or ROW3_CONTROL in arms:
        return list(arms)
    k = arms.index("maps") + 1
    return list(arms[:k]) + [ROW3_CONTROL] + list(arms[k:])


def shift_of(sample_id: int, metres: float = SHIFT_M) -> Tuple[float, float]:
    """The target_shifted arm's world shift of one sample: `metres` in a direction seeded by the sample id."""
    a = float(np.random.default_rng([LIST_SEED, int(sample_id), 3]).uniform(0.0, 2.0 * np.pi))
    return metres * float(np.cos(a)), metres * float(np.sin(a))


def arm_options(arm: str, sample_id: int):
    """MapOptions of a v2 arm for one sample."""
    from vint_train.mapmad.map_sample_v2 import MapOptions
    from vint_train.mapmad.map_v2 import LAYERS_V2

    return {"clean": MapOptions(target_key="clean"), "near_only": MapOptions(show=(True, False)),
            "far_only": MapOptions(show=(False, True)), ROW3_CONTROL: MapOptions(zero_layers=LAYERS_V2),
            "target_zeroed": MapOptions(zero_layers=("target",)),
            "frontier_zeroed": MapOptions(zero_layers=("frontier",)),
            "target_shifted": MapOptions(target_shift_m=shift_of(sample_id))}.get(arm, MapOptions())


def is_v2(ds) -> bool:
    return hasattr(ds, "pool_index")


# --- models ------------------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_actions(model, cond: torch.Tensor, ids: Sequence[int], scheduler, horizon: int) -> torch.Tensor:
    """Full reverse diffusion -> waypoints (B, 8, 2) in normalised units (get_action); every sample's noise (start
    and reverse steps) comes from its own id (diffusion.item_noise), independent of batching and arm."""
    from vint_train.training.train_utils import get_action

    return get_action(reverse_diffusion(model, cond, ids, scheduler, horizon, NOISE_SEED))


def per_sample_loss(actions: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """gc_action_loss and waypoint cosine similarity per sample (the _compute_losses_nomad formulas)."""
    loss = F.mse_loss(actions, labels, reduction="none").mean(dim=(1, 2))
    cos = F.cosine_similarity(actions[:, :, :2], labels[:, :, :2], dim=-1).mean(dim=-1)
    return loss, cos


def coordinate_losses(actions: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The forward (x) and sideways (y) parts of gc_action_loss per sample: loss = (forward + sideways) / 2."""
    sq = F.mse_loss(actions, labels, reduction="none").mean(dim=1)
    return sq[:, 0], sq[:, 1]


def hidden_map_inputs(model, b: int, device) -> Dict[str, torch.Tensor]:
    """All-zero, hidden map inputs for a batch of b (v1: [b, 3, 64, 64] + mask (b,); v2: [b, 2, 4, 64, 64] + (b, 2))."""
    if model.vision_encoder.n_map_tokens == 2:
        return {"map_img": torch.zeros((b, 2, 4, 64, 64), device=device),
                "input_map_mask": torch.ones((b, 2), dtype=torch.long, device=device)}
    return {"map_img": torch.zeros((b, 3, 64, 64), device=device),
            "input_map_mask": torch.ones(b, dtype=torch.long, device=device)}


# --- Habitat arms ------------------------------------------------------------------------------------------------
class ArmSamples(Dataset):
    """The listed samples built in one arm (deterministic: photo goal offset from the sample id)."""

    def __init__(self, ds, samples: List[Dict[str, Any]], arm: str) -> None:
        self.ds, self.samples, self.arm = ds, samples, arm

    def __len__(self) -> int:
        return len(self.samples)

    def photo_offset(self, s: Dict[str, Any]) -> int:
        k_max = min(PHOTO_K_MAX, self.ds.max_goal_offset(s["drive"], s["t"]))
        return int(np.random.default_rng([LIST_SEED, s["id"]]).integers(PHOTO_K_MIN, k_max + 1))

    def __getitem__(self, j: int):
        s = self.samples[j]
        rng = np.random.default_rng([LIST_SEED, s["id"]])
        if is_v2(self.ds):
            mode = M.BY_NAME[ARM_MODE_V2[self.arm]]
            out = self.ds.habitat_sample(s["drive"], s["t"], mode, rng,
                                         goal_offset=self.photo_offset(s) if mode.goal_mask == 0 else None,
                                         options=arm_options(self.arm, s["id"]))
            return out + (torch.tensor(s["id"]),)
        mode = M.BY_NAME[ARM_MODE[self.arm]]
        out = self.ds.habitat_sample(s["drive"], s["t"], mode, rng,
                                     goal_offset=self.photo_offset(s) if mode.goal_mask == 0 else None,
                                     wrong_heat=self.arm == "wrong_heat", perturbation=None)
        return out + (torch.tensor(s["id"]),)


@torch.no_grad()
def score_arm(model, ds, samples, arm: str, cfg, device, batch_size: int, workers: int) -> Dict[str, np.ndarray]:
    """Per sample: gc_action_loss, cos_sim, its forward / sideways parts, map attention share (v1: (N,); v2:
    (N, 2) near/far; NaN where hidden), action mask."""
    scheduler = make_noise_scheduler(cfg)
    scheduler.set_timesteps(cfg["num_diffusion_iters"])
    loader = DataLoader(ArmSamples(ds, samples, arm), batch_size=batch_size, shuffle=False, num_workers=workers,
                        timeout=loader_timeout(workers))
    enc = model.vision_encoder
    n_map = enc.n_map_tokens
    res = {"gc_action_loss": [], "cos_sim": [], "attn_share": [], "action_mask": [], "loss_forward": [],
           "loss_sideways": []}
    for bi, data in enumerate(tqdm.tqdm(loader, desc=arm, dynamic_ncols=True, leave=False)):
        b, ids = to_batch(data[:-1], IMAGENET, device), data[-1]  # the training input recipe
        obs, goal, labels, maps, gm, mm = (b["obs"], b["goal"], b["actions"].to(device), b["map"], b["goal_mask"],
                                           b["map_mask"])
        am = b["action_mask"].cpu()
        tokens, padding = enc.map_tokens(enc._encode_images(obs, goal), gm, maps, mm)
        out_manual, share = attention_shares(enc.sa_encoder, tokens, padding, n_map)
        if n_map == 1:
            share = share[:, 0]
        cond = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=gm, map_img=maps, input_map_mask=mm)
        if bi == 0:  # the manual layer pass must reproduce nn.TransformerEncoder (up to float32 kernel noise:
            # on the GPU the encoder runs PyTorch's fused fast path, about 1e-4 of the output scale apart)
            ref = enc.sa_encoder(tokens, src_key_padding_mask=padding)
            visible = ~padding
            rel = float((out_manual - ref)[visible].abs().max() / ref[visible].abs().max())
            assert rel < 1e-3, f"manual attention pass differs from the encoder: relative {rel:.2e}"
        actions = sample_actions(model, cond, ids.tolist(), scheduler, cfg["len_traj_pred"])
        loss, cos = per_sample_loss(actions, labels)
        fwd, side = coordinate_losses(actions, labels)
        res["gc_action_loss"].append(loss.cpu())
        res["cos_sim"].append(cos.cpu())
        res["loss_forward"].append(fwd.cpu())
        res["loss_sideways"].append(side.cpu())
        res["attn_share"].append(torch.where(mm.bool(), torch.full_like(share, float("nan")), share).cpu())
        res["action_mask"].append(am.float())
    return {k: torch.cat(v).numpy().astype(np.float64) for k, v in res.items()}


# --- GoStanford photo goals --------------------------------------------------------------------------------------
class GoStanfordPhotoSamples(Dataset):
    """Fixed GoStanford test samples with a positive photo goal k waypoints ahead (k from the sample's seed)."""

    def __init__(self, ds, n: int, seed: int = LIST_SEED) -> None:
        self.ds = ds
        rng = np.random.default_rng([seed, 1])
        idx = np.arange(len(ds)) if n >= len(ds) else np.sort(rng.choice(len(ds), n, replace=False))
        self.items = []
        for j, i in enumerate(idx):
            drive, t, max_goal = ds.index_to_data[int(i)]
            k_max = min(PHOTO_K_MAX, int(max_goal) // ds.waypoint_spacing)
            if k_max < PHOTO_K_MIN:
                continue
            k = int(np.random.default_rng([seed, 2, j]).integers(PHOTO_K_MIN, k_max + 1))
            self.items.append((drive, int(t), k))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, j: int):
        drive, t, k = self.items[j]
        ds = self.ds
        obs = torch.cat([ds._load_image(drive, ct) for ct in ds._context_times(t)])
        goal = ds._load_image(drive, t + k * ds.waypoint_spacing)
        actions, _ = ds._compute_actions(ds._get_trajectory(drive), t, t + k * ds.waypoint_spacing)
        return obs, goal, torch.as_tensor(actions, dtype=torch.float32), torch.tensor(j)


@torch.no_grad()
def score_gostanford(model, loader, cfg, device, map_input: bool) -> np.ndarray:
    scheduler = make_noise_scheduler(cfg)
    scheduler.set_timesteps(cfg["num_diffusion_iters"])
    losses = []
    for obs_image, goal_image, labels, ids in tqdm.tqdm(loader, dynamic_ncols=True, leave=False):
        obs = torch.cat([IMAGENET(o) for o in torch.split(obs_image, 3, dim=1)], dim=1).to(device)
        goal = IMAGENET(goal_image).to(device)
        b = obs.shape[0]
        kwargs = dict(obs_img=obs, goal_img=goal, input_goal_mask=torch.zeros(b, dtype=torch.long, device=device))
        if map_input:
            kwargs.update(hidden_map_inputs(model, b, device))
        cond = model("vision_encoder", **kwargs)
        actions = sample_actions(model, cond, ids.tolist(), scheduler, cfg["len_traj_pred"])
        losses.append(per_sample_loss(actions, labels.to(device))[0].cpu())
    return torch.cat(losses).numpy().astype(np.float64)
