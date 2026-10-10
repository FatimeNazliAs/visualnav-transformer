"""MapMaD's training loop (Phase 3; used by train.py when `map_input: true`). It follows `train_nomad` /
`train_eval_loop_nomad` (same diffusion objective, action mask, EMA, AdamW) with these differences:

- the batch carries map, goal_mask, map_mask and mode per sample (drawn by the dataset, not per batch);
- the distance loss counts only samples with the photo shown, per sample (NoMaD's code multiplies an already
  averaged scalar by the mask, so it averages over all samples);
- the diffusion loss is also logged per mode (5 Habitat modes + GoStanford photo / explore);
- the lr scheduler is stepped ONCE per epoch (`train_eval_loop_nomad` steps it twice), so cosine over `epochs`
  gives lr 1e-4 x (1 + cos(pi e / epochs)) / 2 at epoch e; the lr of every epoch is logged;
- checkpoints hold `model.module` weights (no DataParallel `module.` prefix), EMA too;
- in-loop eval: per-mode diffusion loss of the EMA model on a fixed seeded subset of each test set, with fixed
  noise and timesteps (cheap; the offline harness `offline_eval.py` gives the real numbers -- never mix the two);
- per step: data-wait and compute seconds (timing test), `max_steps_per_epoch` to cut an epoch short;
- `run_info.json` (config, seed, git commit) and `metrics.jsonl` next to the checkpoints.
"""

import json
import math
import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from diffusers.training_utils import EMAModel

from vint_train.mapmad.config import git_commit
from vint_train.mapmad.modes import MODES
from vint_train.training.train_utils import ACTION_STATS, get_delta, normalize_data
from vint_train.visualizing.visualize_utils import from_numpy

MODE_NAMES = {m.id: m.name for m in MODES}


def unwrap(model: nn.Module) -> nn.Module:
    """The model without a DataParallel wrapper."""
    return model.module if isinstance(model, nn.DataParallel) else model


def normalized_deltas(actions: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Action waypoints -> NoMaD's normalised deltas (as train_nomad does)."""
    return from_numpy(normalize_data(get_delta(actions), ACTION_STATS)).to(device)


def encode(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model("vision_encoder", obs_img=batch["obs"], goal_img=batch["goal"], input_goal_mask=batch["goal_mask"],
                 map_img=batch["map"], input_map_mask=batch["map_mask"])


def to_batch(data, transform, device: torch.device) -> Dict[str, torch.Tensor]:
    """MapMaDDataset tuple -> dict of device tensors; images ImageNet-normalised like train_nomad."""
    obs_image, goal_image, actions, distance, _goal_pos, _ds, action_mask, map_img, goal_mask, map_mask, mode = data
    obs = torch.cat([transform(o) for o in torch.split(obs_image, 3, dim=1)], dim=1).to(device)
    return {"obs": obs, "goal": transform(goal_image).to(device), "actions": actions,
            "distance": distance.float().to(device), "action_mask": action_mask.to(device), "map": map_img.to(device),
            "goal_mask": goal_mask.long().to(device), "map_mask": map_mask.long().to(device), "mode": mode.to(device)}


class ModeMeter:
    """Running action-mask-weighted mean of a per-sample loss, per mode."""

    def __init__(self) -> None:
        self.num, self.den = defaultdict(float), defaultdict(float)

    def add(self, per_sample: torch.Tensor, weight: torch.Tensor, mode: torch.Tensor) -> None:
        for m in torch.unique(mode).tolist():
            sel = mode == m
            self.num[m] += float((per_sample[sel] * weight[sel]).sum())
            self.den[m] += float(weight[sel].sum())

    def means(self) -> Dict[str, float]:
        return {MODE_NAMES[m]: self.num[m] / self.den[m] for m in sorted(self.num) if self.den[m] > 0}


def train_mapmad_epoch(model, ema_model, optimizer, loader, transform, device, noise_scheduler, epoch: int,
                       alpha: float, print_log_freq: int, wandb_log_freq: int, use_wandb: bool,
                       max_steps: Optional[int] = None) -> Dict[str, Any]:
    """One epoch; returns mean losses, per-mode diffusion losses and step timing."""
    model.train()
    meter, totals = ModeMeter(), defaultdict(float)
    wait_s, step_s, steps = [], [], 0
    t_ready = time.perf_counter()
    for data in tqdm.tqdm(loader, desc=f"MapMaD epoch {epoch}", leave=False, dynamic_ncols=True):
        t_data = time.perf_counter()
        b = to_batch(data, transform, device)
        cond = encode(model, b)
        dist_pred = model("dist_pred_net", obsgoal_cond=cond).squeeze(-1)
        shown = 1.0 - b["goal_mask"].float()
        dist_loss = ((dist_pred - b["distance"]) ** 2 * shown).sum() / (shown.sum() + 1e-2)

        naction = normalized_deltas(b["actions"], device)
        noise = torch.randn(naction.shape, device=device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (naction.shape[0],), device=device).long()
        noise_pred = model("noise_pred_net", sample=noise_scheduler.add_noise(naction, noise, timesteps),
                           timestep=timesteps, global_cond=cond)
        per_sample = F.mse_loss(noise_pred, noise, reduction="none").mean(dim=(1, 2))
        am = b["action_mask"]
        diffusion_loss = (per_sample * am).mean() / (am.mean() + 1e-2)
        loss = alpha * dist_loss + (1 - alpha) * diffusion_loss
        loss_cpu = loss.item()
        if not math.isfinite(loss_cpu):
            raise RuntimeError(f"Training diverged: non-finite loss ({loss_cpu}) at epoch {epoch}, step {steps}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_model.step(model)

        meter.add(per_sample.detach(), am, b["mode"])
        totals["diffusion_loss"] += float(diffusion_loss)
        totals["dist_loss"] += float(dist_loss)
        totals["total_loss"] += loss_cpu
        steps += 1
        t_done = time.perf_counter()
        wait_s.append(t_data - t_ready)
        step_s.append(t_done - t_data)
        t_ready = t_done
        if use_wandb and wandb_log_freq and steps % wandb_log_freq == 0:
            wandb.log({"total_loss": loss_cpu, "diffusion_loss": float(diffusion_loss), "dist_loss": float(dist_loss),
                       **{f"train_mode/{k}": v for k, v in meter.means().items()}})
        if print_log_freq and steps % print_log_freq == 0:
            modes = " ".join(f"{k}={v:.4f}" for k, v in meter.means().items())
            print(f"(epoch {epoch}) (step {steps}/{len(loader)}) loss {totals['total_loss'] / steps:.4f} | {modes}",
                  flush=True)
        if max_steps is not None and steps >= max_steps:
            break
    skip = min(10, max(len(wait_s) - 1, 0))  # first steps include worker start-up
    return {"epoch": epoch, "steps": steps, **{k: v / max(steps, 1) for k, v in totals.items()},
            "train_mode_diffusion_loss": meter.means(),
            "timing": {"data_wait_s": float(np.mean(wait_s[skip:])) if wait_s else None,
                       "compute_s": float(np.mean(step_s[skip:])) if step_s else None,
                       "first_steps_skipped": skip}}


@torch.no_grad()
def evaluate_mapmad(ema_net: nn.Module, loader, transform, device, noise_scheduler, seed: int) -> Dict[str, float]:
    """Per-mode diffusion (noise-prediction) loss of the EMA model, fixed noise + timesteps per batch."""
    ema_net.eval()
    meter = ModeMeter()
    for bi, data in enumerate(loader):
        b = to_batch(data, transform, device)
        g = torch.Generator(device="cpu").manual_seed(seed + bi)
        naction = normalized_deltas(b["actions"], device)
        noise = torch.randn(naction.shape, generator=g).to(device)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (naction.shape[0],), generator=g).to(device)
        cond = encode(ema_net, b)
        pred = ema_net("noise_pred_net", sample=noise_scheduler.add_noise(naction, noise, timesteps), timestep=timesteps,
                       global_cond=cond)
        meter.add(F.mse_loss(pred, noise, reduction="none").mean(dim=(1, 2)), b["action_mask"], b["mode"])
    return meter.means()


def save_checkpoints(project_folder: str, epoch: int, model, ema_model, optimizer, lr_scheduler) -> None:
    """ema_{epoch}.pth / {epoch}.pth / latest.pth hold unwrapped (no `module.`) weights."""
    torch.save(unwrap(ema_model.averaged_model).state_dict(), os.path.join(project_folder, f"ema_{epoch}.pth"))
    torch.save(unwrap(ema_model.averaged_model).state_dict(), os.path.join(project_folder, "ema_latest.pth"))
    state = unwrap(model).state_dict()
    torch.save(state, os.path.join(project_folder, f"{epoch}.pth"))
    torch.save(state, os.path.join(project_folder, "latest.pth"))
    torch.save(optimizer.state_dict(), os.path.join(project_folder, "optimizer_latest.pth"))
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), os.path.join(project_folder, "scheduler_latest.pth"))


def train_eval_loop_mapmad(model, optimizer, lr_scheduler, noise_scheduler, train_loader, sampler,
                           test_loaders: Dict[str, Any], transform, config: Dict[str, Any], device,
                           current_epoch: int = 0, load_report: Optional[Dict[str, Any]] = None) -> None:
    """Epoch loop: train, save, in-loop eval, step the lr scheduler once; metrics to metrics.jsonl (+ wandb)."""
    folder = config["project_folder"]
    with open(os.path.join(folder, "run_info.json"), "w") as f:
        json.dump({"config": config, "seed": config["seed"], "git_commit": git_commit(),
                   "load_report": load_report, "torch": torch.__version__,
                   "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, f, indent=1,
                  default=str)
    ema_model = EMAModel(model=model, power=0.75)
    use_wandb = config["use_wandb"]
    for epoch in range(current_epoch, current_epoch + config["epochs"]):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Start MapMaD epoch {epoch} (lr {lr:.3e})", flush=True)
        sampler.set_epoch(epoch)
        t0 = time.time()
        record = train_mapmad_epoch(model, ema_model, optimizer, train_loader, transform, device, noise_scheduler, epoch,
                                    float(config["alpha"]), config["print_log_freq"], config["wandb_log_freq"],
                                    use_wandb, config.get("max_steps_per_epoch"))
        record.update({"lr": lr, "train_seconds": time.time() - t0})
        save_checkpoints(folder, epoch, model, ema_model, optimizer, lr_scheduler)
        if config.get("eval_freq", 1) and (epoch + 1) % config.get("eval_freq", 1) == 0:
            t1 = time.time()
            record["eval_mode_diffusion_loss"] = {
                name: evaluate_mapmad(ema_model.averaged_model, loader, transform, device, noise_scheduler, config["seed"])
                for name, loader in test_loaders.items()}
            record["eval_seconds"] = time.time() - t1
        with open(os.path.join(folder, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if use_wandb:
            log = {"lr": lr, "epoch": epoch}
            for name, modes in record.get("eval_mode_diffusion_loss", {}).items():
                log.update({f"eval_{name}/{k}": v for k, v in modes.items()})
            wandb.log(log)
        if lr_scheduler is not None:
            lr_scheduler.step()
