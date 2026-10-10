"""Gate G3 row 1 reference: the unmodified NoMaD's outputs on one fixed batch, saved before any NoMaD file changes.

The batch: 8 GoStanford-test + 8 habitat_mapmad-test samples (evenly spaced sample indices, goals drawn with
numpy seeded per sample), each run with the photo goal shown (mask 0) and hidden (mask 1). Saved per mask:
- `cond`       vision-encoder output (16, 256);
- `noise_pred` one noise-prediction call on fixed noisy actions at fixed timesteps (16, 8, 2);
- `naction`    the full 10-step reverse diffusion from fixed noise (scheduler noise from a seeded generator);
- `dist`       distance head output (16, 1).
`run_forward` is the one function used both here and in the later checks (`checks_g3.py`), so "byte-identical"
means: same inputs, same call sequence, same device, same deterministic kernels, `torch.equal` on every output.

Run inside naz_mapmad from /app/visualnav-transformer/train:
    CUDA_VISIBLE_DEVICES=0 python -m vint_train.mapmad.reference_forward \
        --model-config /outputs/mapmad/weights/official/nomad.yaml \
        --weights /outputs/mapmad/weights/official/nomad.pth \
        --data-config config/mapmad.yaml --out /outputs/mapmad/p3_model/reference_forward.pt
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch touches CUDA (deterministic cuBLAS)

import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision import transforms

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.mapmad.config import git_commit

SEED = 0
SAMPLES_PER_DATASET = 8
DATASETS = ("go_stanford", "habitat_mapmad")
IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def enforce_determinism() -> None:
    """Same kernels on every run: no cuDNN autotuning, deterministic algorithms only."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def make_noise_scheduler(cfg: Dict[str, Any]) -> DDPMScheduler:
    """NoMaD's DDPM scheduler (train.py / deployment settings)."""
    return DDPMScheduler(num_train_timesteps=cfg["num_diffusion_iters"], beta_schedule="squaredcos_cap_v2",
                         clip_sample=True, prediction_type="epsilon")


def test_dataset(name: str, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> ViNT_Dataset:
    """The unchanged ViNT_Dataset on a test split, built with the arguments train.py passes (spacing 1)."""
    d = data_cfg["datasets"][name]
    return ViNT_Dataset(
        data_folder=d["data_folder"], data_split_folder=d["test"], dataset_name=name,
        image_size=model_cfg["image_size"], waypoint_spacing=1,
        min_dist_cat=model_cfg["distance"]["min_dist_cat"], max_dist_cat=model_cfg["distance"]["max_dist_cat"],
        min_action_distance=model_cfg["action"]["min_dist_cat"], max_action_distance=model_cfg["action"]["max_dist_cat"],
        negative_mining=True, len_traj_pred=model_cfg["len_traj_pred"], learn_angle=model_cfg["learn_angle"],
        context_size=model_cfg["context_size"], context_type="temporal", end_slack=0,
        goals_per_obs=1, normalize=True, goal_type="image",
    )


def build_inputs(model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed batch: images ImageNet-normalised exactly as train_nomad does; one numpy seed per sample."""
    obs, goal, names = [], [], []
    for name in DATASETS:
        ds = test_dataset(name, model_cfg, data_cfg)
        for k, i in enumerate(np.linspace(0, len(ds) - 1, SAMPLES_PER_DATASET).astype(int)):
            np.random.seed(SEED + k)
            o, g = ds[int(i)][:2]
            obs.append(torch.cat([IMAGENET(x) for x in torch.split(o, 3, dim=0)], dim=0))
            goal.append(IMAGENET(g))
            names.append(f"{name}:{ds.index_to_data[int(i)][0]}:{ds.index_to_data[int(i)][1]}")
        ds.close()
    obs_t, goal_t = torch.stack(obs), torch.stack(goal)
    g = torch.Generator().manual_seed(SEED)
    b, horizon = obs_t.shape[0], model_cfg["len_traj_pred"]
    return {
        "obs_img": obs_t, "goal_img": goal_t, "samples": names,
        "noisy_action": torch.randn((b, horizon, 2), generator=g),
        "timesteps": torch.randint(0, model_cfg["num_diffusion_iters"], (b,), generator=g),
        "init_noise": torch.randn((b, horizon, 2), generator=g),
    }


@torch.no_grad()
def run_forward(model: torch.nn.Module, inputs: Dict[str, Any], cfg: Dict[str, Any], device: torch.device,
                extra_encoder_kwargs: Dict[str, torch.Tensor] = None) -> Dict[str, Dict[str, torch.Tensor]]:
    """NoMaD's four calls for goal mask 0 and 1 on the fixed batch; outputs on the CPU.

    `extra_encoder_kwargs` (e.g. map + map mask) are passed to the vision encoder only; None = the old call.
    """
    model.eval()
    scheduler = make_noise_scheduler(cfg)
    scheduler.set_timesteps(cfg["num_diffusion_iters"])
    obs, goal = inputs["obs_img"].to(device), inputs["goal_img"].to(device)
    out = {}
    for mask_value in (0, 1):
        mask = torch.full((obs.shape[0],), mask_value, dtype=torch.long, device=device)
        kwargs = dict(obs_img=obs, goal_img=goal, input_goal_mask=mask)
        kwargs.update({k: v.to(device) for k, v in (extra_encoder_kwargs or {}).items()})
        cond = model("vision_encoder", **kwargs)
        noise_pred = model("noise_pred_net", sample=inputs["noisy_action"].to(device),
                           timestep=inputs["timesteps"].to(device), global_cond=cond)
        generator = torch.Generator(device=device).manual_seed(SEED)
        naction = inputs["init_noise"].to(device)
        for k in scheduler.timesteps[:]:
            pred = model("noise_pred_net", sample=naction, timestep=k.unsqueeze(-1).repeat(naction.shape[0]).to(device),
                         global_cond=cond)
            naction = scheduler.step(model_output=pred, timestep=k, sample=naction, generator=generator).prev_sample
        dist = model("dist_pred_net", obsgoal_cond=cond)
        out[f"mask{mask_value}"] = {"cond": cond.cpu(), "noise_pred": noise_pred.cpu(), "naction": naction.cpu(),
                                    "dist": dist.cpu()}
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-config", required=True, help="released nomad.yaml")
    parser.add_argument("--weights", required=True, help="official nomad.pth")
    parser.add_argument("--data-config", required=True, help="train config whose `datasets` gives the test splits")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from vint_train.mapmad.closed_loop.nomad_policy import load_nomad

    enforce_determinism()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg, data_cfg = load_yaml(args.model_config), load_yaml(args.data_config)
    inputs = build_inputs(model_cfg, data_cfg)
    model = load_nomad(args.weights, model_cfg, device)
    outputs = run_forward(model, inputs, model_cfg, device)
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; refusing to overwrite the G3 reference")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"inputs": inputs, "outputs": outputs, "seed": SEED, "git_commit": git_commit(),
                "weights": args.weights, "model_config": model_cfg, "device": str(device),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
               args.out)
    for m, o in outputs.items():
        print(m, {k: (tuple(v.shape), float(v.float().abs().mean())) for k, v in o.items()})
    print("saved", args.out)


if __name__ == "__main__":
    main()
