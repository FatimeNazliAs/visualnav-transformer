"""A MapMaD run's state on disk (moved out of train_loop.py in the Phase 4 clean-up, unchanged):
- `checkpoints: phase3` (default): ema_{epoch}.pth, ema_latest.pth, {epoch}.pth, latest.pth (plain weights),
  optimizer_latest.pth, scheduler_latest.pth;
- `checkpoints: ema_and_resume` (Phase 4): ema_{epoch}.pth + ONE latest.pth holding model, EMA weights + step,
  optimizer, RNG states and the epoch -- `load_resume_state` restores it (train.py --resume);
- run_info.json (run_info_resume_<epoch>.json after a resume): config, seed, git commit, parameter groups, GPUs.
All weights are saved unwrapped (no DataParallel `module.` prefix).
"""

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from vint_train.mapmad.config import git_commit

CHECKPOINT_STYLES = ("phase3", "ema_and_resume")


def unwrap(model: nn.Module) -> nn.Module:
    """The model without a DataParallel wrapper."""
    return model.module if isinstance(model, nn.DataParallel) else model


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


def atomic_save(obj: Any, path: str) -> None:
    """torch.save to a temporary file, then rename (a crash never leaves a half-written checkpoint)."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_ema_and_resume(project_folder: str, epoch: int, model, ema_model, optimizer) -> None:
    """`checkpoints: ema_and_resume`: ema_{epoch}.pth (unwrapped EMA weights) + latest.pth (resume state)."""
    atomic_save(unwrap(ema_model.averaged_model).state_dict(), os.path.join(project_folder, f"ema_{epoch}.pth"))
    atomic_save({"epoch": epoch, "model": unwrap(model).state_dict(),
                 "ema_model": unwrap(ema_model.averaged_model).state_dict(),
                 "ema_optimization_step": ema_model.optimization_step, "ema_decay": ema_model.decay,
                 "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "numpy_rng": np.random.get_state()}, os.path.join(project_folder, "latest.pth"))


def load_resume_state(path: str, model, ema_model, optimizer) -> int:
    """Restore a `save_ema_and_resume` latest.pth; returns the next epoch (0-based)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(state["model"], strict=True)
    unwrap(ema_model.averaged_model).load_state_dict(state["ema_model"], strict=True)
    ema_model.optimization_step, ema_model.decay = state["ema_optimization_step"], state["ema_decay"]
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    return int(state["epoch"]) + 1


def write_run_info(folder: str, config: Dict[str, Any], optimizer, load_report, resumed_at: Optional[int]) -> None:
    """run_info.json (run_info_resume_<epoch>.json after a resume): config, seed, git commit, groups, GPUs."""
    name = "run_info.json" if resumed_at is None else f"run_info_resume_{resumed_at}.json"
    groups = [{"name": g.get("name", f"group{i}"), "tensors": len(g["params"]),
               "parameters": int(sum(p.numel() for p in g["params"])), "base_lr": g.get("base_lr", g["lr"])}
              for i, g in enumerate(optimizer.param_groups)]
    with open(os.path.join(folder, name), "w") as f:
        json.dump({"config": config, "seed": config["seed"], "git_commit": git_commit(),
                   "load_report": load_report, "resumed_at_epoch": resumed_at, "param_groups": groups,
                   "torch": torch.__version__,
                   "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, f, indent=1,
                  default=str)


def checkpoint_style(config: Dict[str, Any]) -> str:
    """The run's `checkpoints` style (phase3 if unset); unknown names are refused."""
    style = config.get("checkpoints", "phase3")
    if style not in CHECKPOINT_STYLES:
        raise ValueError(f"unknown checkpoints style {style}")
    return style


def save_epoch(style: str, project_folder: str, epoch: int, model, ema_model, optimizer, lr_scheduler) -> None:
    """Save the end of `epoch` in the run's checkpoint style."""
    if style == "ema_and_resume":
        save_ema_and_resume(project_folder, epoch, model, ema_model, optimizer)
    else:
        save_checkpoints(project_folder, epoch, model, ema_model, optimizer, lr_scheduler)
