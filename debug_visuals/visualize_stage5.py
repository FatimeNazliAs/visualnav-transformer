# debug_visuals/visualize_stage5.py
"""
Stage 5 visualisation — "What does the diffusion denoising loop actually do?"

Runs the full pipeline up to ct (reusing visualize_stage1/2/4's functions
exactly — same model load, same data load, same ct computation), then
manually drives the DDPM reverse-diffusion loop step by step: starting
from pure Gaussian noise a^K, applying noise_pred_net + the scheduler's
update rule K times, down to the clean action sequence a^0.

Note on "DDIM" vs "DDPM": this checkpoint and the rest of this codebase
(debug/stage5_diffusion.py, train_utils.py) use diffusers' DDPMScheduler,
not DDIM — confirmed by reading both before writing this script. Using a
different scheduler type would not reproduce how this model was actually
trained/sampled, so this script uses DDPMScheduler to match.

Note on hooks: there is no high-level "sample()" convenience method
anywhere in this codebase that wraps the denoising loop — model.noise_pred_net
is called directly, once per step, inside a loop we write ourselves (exactly
like debug/stage5_diffusion.py). That already gives us a^k at every step
without needing a forward hook to "intercept" anything hidden — there's
nothing hidden to intercept.

Saves 3 PNGs to debug_visuals/outputs/<traj>_f<idx>/, prefixed stage5_:

  1. stage5_1_denoising_strip.png   — 11 small 2D paths, noise -> clean
  2. stage5_2_final_trajectory.png  — real kinematic path overlaid on the
                                       raw current camera frame + velocity
                                       commands over time
  3. stage5_3_multimodal_runs.png   — same ct, 10 seeds -> 10 different
                                       kinematic paths (the multimodal
                                       property), plus their mean path

Run from the repo root inside the container:

    conda activate vint_train
    cd /app/visualnav-transformer
    python debug_visuals/visualize_stage5.py

To visualise a different sample, edit TRAJ_NAME and FRAME_IDX in
debug_visuals/config.py.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed inside the container
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DDPMScheduler

# ── Make the repo root importable when running as `python debug_visuals/…` ────
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "train"))

from debug_visuals.config import (
    TRAJ_NAME,
    FRAME_IDX,
    ENCODING_SIZE,
    NUM_ACTIONS,
    ACTION_DIM,
    K_DENOISING,
    DEVICE,
    OUTPUTS_DIR,
)
from debug_visuals import visualize_stage1
from debug_visuals import visualize_stage2
from debug_visuals import visualize_stage4

SEP = "─" * 60


# ══════════════════════════════════════════════════════════════════════════════
# Diffusion scheduler + manual denoising loop
# ══════════════════════════════════════════════════════════════════════════════

def _build_scheduler() -> DDPMScheduler:
    # Matches debug/stage5_diffusion.py / train.py exactly: NoMaD trains with
    # num_train_timesteps == num_inference_steps == K_DENOISING (no step
    # skipping), squared-cosine beta schedule, epsilon (noise) prediction.
    scheduler = DDPMScheduler(
        num_train_timesteps=K_DENOISING,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(K_DENOISING)
    return scheduler


def _print_step_stats(k_label: int, a_k: torch.Tensor) -> None:
    flat = a_k.detach()
    print(
        f"  Step k={k_label:>2}: a^k mean={flat.mean().item():+.4f}  "
        f"std={flat.std().item():.4f}  min={flat.min().item():+.4f}  "
        f"max={flat.max().item():+.4f}  norm={flat.norm().item():.4f}"
    )


def run_denoising_loop(model, ct: torch.Tensor, seed: int, capture_all_steps: bool = False) -> dict:
    """
    Runs the reverse diffusion process from pure noise a^K to clean a^0.

    seed pins torch's RNG right before sampling the initial noise, so calls
    with different seeds but the same ct produce different a^0 — that
    divergence is exactly the multimodal property stage5_3_multimodal_runs.png
    visualises.

    capture_all_steps=True additionally records (and prints) every one of
    the 11 intermediate states (a^10 .. a^0), needed for PART 1's console
    trace and stage5_1_denoising_strip.png. The multimodal comparison runs
    don't need this — only their final a^0 matters — so it's skipped there
    to avoid wasted work.

    Returns {"final": a^0 tensor (1, NUM_ACTIONS, ACTION_DIM),
             "steps": [(k_label, a_k tensor), ...] in noise->clean order}.
    """
    torch.manual_seed(seed)   # reseed so this run's noise is reproducible and seed-dependent
    scheduler = _build_scheduler()

    # a^K: the diffusion starting point, pure standard-normal noise.
    a_k = torch.randn(1, NUM_ACTIONS, ACTION_DIM, device=DEVICE)

    steps = []
    if capture_all_steps:
        # Record/print the initial noise state itself as k=K_DENOISING,
        # before any denoising step has been applied.
        steps.append((K_DENOISING, a_k.detach().clone()))
        _print_step_stats(K_DENOISING, a_k)

    try:
        with torch.no_grad():
            # scheduler.timesteps counts down K_DENOISING-1, ..., 1, 0 — each
            # iteration turns a^{t+1} into a^{t}.
            for t_step in scheduler.timesteps:
                # Predict the noise component currently present in a_k, given
                # the navigation context vector ct as conditioning.
                eps_pred = model.noise_pred_net(
                    sample=a_k,
                    timestep=t_step,
                    global_cond=ct,
                )
                # Scheduler subtracts the predicted noise (scaled by the
                # diffusion schedule) to produce the next, slightly cleaner
                # sample a^{t_step}.
                a_k = scheduler.step(eps_pred, t_step, a_k).prev_sample

                k_label = int(t_step)
                if capture_all_steps:
                    steps.append((k_label, a_k.detach().clone()))
                    _print_step_stats(k_label, a_k)
    except Exception:
        print(f"  ERROR during denoising loop (seed={seed}):")
        traceback.print_exc()
        raise

    return {"final": a_k, "steps": steps}


def print_action_table(a_0: torch.Tensor) -> None:
    """Prints the final 8-step action sequence as a simple text table."""
    actions = a_0[0].detach().cpu().numpy()   # (NUM_ACTIONS, ACTION_DIM)
    print("\n  Timestep | linear_vel | angular_vel")
    print("  ---------|------------|------------")
    for t in range(actions.shape[0]):
        lin, ang = actions[t]
        print(f"  {t:>8} | {lin:>10.2f} | {ang:>11.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Shared plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _grey_to_blue(frac: float) -> tuple:
    """frac=0 -> grey (still noisy), frac=1 -> blue (fully denoised)."""
    grey = np.array(mcolors.to_rgb("#999999"))
    blue = np.array(mcolors.to_rgb("#2980b9"))
    return tuple(grey + (blue - grey) * frac)


def _cumsum_path(a_tensor: torch.Tensor):
    """
    Turns an (1, NUM_ACTIONS, 2) action tensor into a 2D path by treating
    linear_vel as a forward step and angular_vel as a lateral step, each
    accumulated over time. This is a simplified visual proxy for the robot's
    path, not a real unicycle/kinematic integration — good enough to see
    "is this a coherent curve or random noise," which is the point here.
    """
    actions = a_tensor[0].detach().cpu().numpy()   # (NUM_ACTIONS, 2)
    lin, ang = actions[:, 0], actions[:, 1]
    x = np.concatenate([[0.0], np.cumsum(lin)])    # prepend origin as waypoint 0
    y = np.concatenate([[0.0], np.cumsum(ang)])
    return x, y


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Denoising strip: 11 small paths, noise -> clean
# ══════════════════════════════════════════════════════════════════════════════

def plot_denoising_strip(steps: list, save_path: Path) -> None:
    n = len(steps)   # 11: k=10 .. k=0
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.4))

    for i, (k_label, a_k) in enumerate(steps):
        frac = i / (n - 1)   # 0 at the first (noisiest) panel, 1 at the last (cleanest)
        color = _grey_to_blue(frac)
        x, y = _cumsum_path(a_k)

        ax = axes[i]
        ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=2.5)
        if k_label == K_DENOISING:
            title = f"k={k_label} (noise)"
        elif k_label == 0:
            title = f"k={k_label} (clean)"
        else:
            title = f"k={k_label}"
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle("Diffusion Denoising: Random Noise → Coherent Robot Trajectory", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.94, "Each panel shows the predicted path at one denoising step (K=10 → 0)",
             ha="center", fontsize=10, style="italic", color="#555555")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Final path overlaid on the camera frame + velocity commands
# ══════════════════════════════════════════════════════════════════════════════

def _kinematic_path(a_tensor: torch.Tensor):
    """
    Proper unicycle kinematic integration (not cumulative sums): heading
    theta accumulates angular_vel, and each step's forward motion is
    rotated by the CURRENT heading before being added to x/y. This is what
    actually makes the path curve, unlike _cumsum_path's straight-line
    accumulation in the (linear_vel, angular_vel) plane.

    x = forward axis, y = lateral axis, theta = heading (radians).
    Returns (x, y, theta), each of length NUM_ACTIONS+1 (waypoint 0 = origin).
    """
    actions = a_tensor[0].detach().cpu().numpy()   # (NUM_ACTIONS, 2)
    lin, ang = actions[:, 0], actions[:, 1]
    n = len(lin)
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    theta = np.zeros(n + 1)
    for t in range(n):
        x[t + 1] = x[t] + lin[t] * np.cos(theta[t])
        y[t + 1] = y[t] + lin[t] * np.sin(theta[t])
        theta[t + 1] = theta[t] + ang[t]
    return x, y, theta


def _path_to_pixels(x: np.ndarray, y: np.ndarray, img_h: int, img_w: int):
    """
    Maps kinematic (x=forward, y=lateral) coordinates to image pixel
    coordinates. Auto-scales to the path's own extent (35% of the image's
    width/height) so the path always lands on-screen regardless of its
    absolute magnitude, anchored at the bottom-center of the image.
    """
    # Find the actual range of x and y values to auto-scale
    x_range = max(abs(x).max(), 0.001)
    y_range = max(abs(y).max(), 0.001)

    # Use 35% of image width/height as the max path extent
    scale_x = (img_w * 0.35) / x_range
    scale_y = (img_h * 0.35) / y_range
    scale = min(scale_x, scale_y)

    # Start point: bottom center of image
    origin_col = img_w / 2.0
    origin_row = img_h * 0.85  # 85% down from top

    # x (forward) maps to columns: forward = up on image = decreasing row
    # y (lateral) maps to rows: left = right on image = increasing col
    col = origin_col + y * scale
    row = origin_row - x * scale  # subtract because forward = up = smaller row

    return col, row


def _draw_path_on_frame(ax, obs_raw_last: np.ndarray, x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> None:
    """Overlays the kinematic path on the raw (un-normalised) current frame."""
    h, w = obs_raw_last.shape[:2]
    ax.imshow(obs_raw_last)   # raw uint8 RGB frame — looks like a normal photo, not the green/black normalised version

    col, row = _path_to_pixels(x, y, h, w)

    ax.plot(col, row, color="cyan", linewidth=4, zorder=2)
    ax.scatter(col, row, color="white", s=60, zorder=3, edgecolors="cyan", linewidths=1.0)

    # Lock the view to the image's own pixel extent — without this, matplotlib
    # would auto-zoom-out to fit any path points that land outside the frame,
    # shrinking the photo instead of showing the path drawn over it.
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    ax.set_title("Planned Path on Current Camera Frame", fontsize=12)
    ax.axis("off")
    ax.text(
        0.02, 0.98,
        "linear_vel = forward (+) / backward (−)\nangular_vel = left (+) / right (−)",
        transform=ax.transAxes, fontsize=10, color="white", va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
    )


def plot_final_trajectory(a_0: torch.Tensor, obs_raw_last: np.ndarray, save_path: Path) -> None:
    x, y, theta = _kinematic_path(a_0)
    actions = a_0[0].detach().cpu().numpy()
    lin, ang = actions[:, 0], actions[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: real kinematic path drawn on top of the raw current camera frame.
    _draw_path_on_frame(axes[0], obs_raw_last, x, y, theta)

    # Panel 2: the same kinematic path as a clean top-down 2D plot (no
    # camera image underneath) — easier to read the path's actual shape.
    # Axis orientation matches Panel 1 exactly: there, col = origin + y*scale
    # (y/lateral -> screen-right) and row = origin - x*scale (x/forward ->
    # screen-up). So here too: y on the horizontal axis, x on the vertical
    # axis, so "forward" reads as "up" in both panels.
    ax_top = axes[1]
    ax_top.plot(y, x, color="#2980b9", linewidth=1.5, marker="o", markersize=5,
                markerfacecolor="white", markeredgecolor="#2980b9", zorder=2)
    ax_top.scatter(y[0], x[0], color="#27ae60", s=90, zorder=3, label="start")
    ax_top.scatter(y[-1], x[-1], color="#e74c3c", s=90, zorder=3, label="end")
    ax_top.set_aspect("equal", adjustable="datalim")
    ax_top.set_xlabel("Lateral: left(+) / right(−)")
    ax_top.set_ylabel("Forward (m)")
    ax_top.set_title("Integrated 2D Path (top-down view)")
    ax_top.annotate(
        "forward = up", xy=(0.5, 0.97), xycoords="axes fraction",
        ha="center", va="top", fontsize=8, color="#555555",
    )
    ax_top.legend(fontsize=8)
    ax_top.grid(alpha=0.3)

    # Panel 3: linear_vel and angular_vel as two line plots over the 8 timesteps.
    ax_vel = axes[2]
    t = range(len(lin))
    ax_vel.plot(t, lin, marker="o", color="#27ae60", label="linear_vel (forward + / backward −)")
    ax_vel.plot(t, ang, marker="o", color="#e67e22", label="angular_vel (left + / right −)")
    ax_vel.set_title("Velocity Commands Over Time")
    ax_vel.set_xlabel("timestep")
    ax_vel.set_ylabel("value")
    ax_vel.legend(fontsize=8)

    fig.suptitle("Final Denoised Action Sequence (a⁰) — Robot Motion Commands", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.94, "Left: path overlaid on camera view  |  Center: top-down 2D path  |  Right: velocity commands",
             ha="center", fontsize=10, style="italic", color="#555555")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Multiple runs, same ct, different seeds (multimodality)
# ══════════════════════════════════════════════════════════════════════════════

MULTI_RUN_SEEDS = [0, 7, 13, 21, 42, 55, 63, 77, 88, 99]


def plot_multimodal_runs(model, ct: torch.Tensor, save_path: Path) -> dict:
    """
    Runs inference once per seed in MULTI_RUN_SEEDS, all from the same ct,
    and plots each run's real kinematic path. Same conditioning, different
    sampled noise -> different plausible trajectories: that spread is the
    multimodal property of a diffusion policy.
    """
    cmap = plt.colormaps.get_cmap("coolwarm")
    paths = []   # (x, y) per run, all length NUM_ACTIONS+1

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, seed in enumerate(MULTI_RUN_SEEDS):
        run = run_denoising_loop(model, ct, seed=seed, capture_all_steps=False)
        x, y, _ = _kinematic_path(run["final"])
        paths.append((x, y))
        color = cmap(i / (len(MULTI_RUN_SEEDS) - 1))
        ax.plot(x, y, color=color, marker="o", markersize=3, linewidth=1.3,
                label=f"seed={seed}")

    # Mean trajectory across all runs, waypoint-by-waypoint.
    xs = np.stack([p[0] for p in paths])   # (n_runs, NUM_ACTIONS+1)
    ys = np.stack([p[1] for p in paths])
    mean_x, mean_y = xs.mean(axis=0), ys.mean(axis=0)
    ax.plot(mean_x, mean_y, color="black", linewidth=3.0, linestyle="--",
            label="mean trajectory", zorder=5)

    ax.set_title("Same Input, Different Random Seeds → Multiple Valid Paths", fontsize=13, fontweight="bold")
    ax.set_xlabel("x position (forward=positive, backward=negative)")
    ax.set_ylabel("y position (left=positive, right=negative)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.text(
        0.98, 0.02,
        "Same observation, same goal, same model weights.\n"
        "Only the initial random noise differs → each run\n"
        "produces a different plausible trajectory.\n"
        "This multimodality is the key advantage of\n"
        "diffusion policies over direct regression.",
        transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved : {save_path}")

    return {"paths": paths, "mean": (mean_x, mean_y)}


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration — shared by main() (standalone run) and run_pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

def run_stage5(model, ct: torch.Tensor, obs_raw_last: np.ndarray, save_dir: Path) -> dict:
    """
    Runs PART 1 (denoising trace + action table, seed=0) and PART 2 (all 3
    PNGs, including the 10-seed multimodal comparison) given an
    already-computed ct, and saves into save_dir.

    obs_raw_last is the raw (un-normalised) current observation frame —
    used as the background image for the path-overlay plot.
    """
    assert ct.shape == (1, ENCODING_SIZE), f"ct shape mismatch: {ct.shape}"

    print("  PART 1 — Running denoising loop (seed=0), printing every step …")
    run1 = run_denoising_loop(model, ct, seed=0, capture_all_steps=True)
    print_action_table(run1["final"])

    print(f"\n  PART 2 — Saving 3 PNG files to {save_dir} …")
    plot_denoising_strip(run1["steps"], save_dir / "stage5_1_denoising_strip.png")
    plot_final_trajectory(run1["final"], obs_raw_last, save_dir / "stage5_2_final_trajectory.png")
    plot_multimodal_runs(model, ct, save_dir / "stage5_3_multimodal_runs.png")

    return {"run1": run1}


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{SEP}")
    print("  NoMaD — Stage 5 Visualisation (diffusion denoising)")
    print(SEP)

    print("\n[1/4] Loading model …")
    model = visualize_stage2.load_model()

    print("\n[2/4] Loading frames from disk and computing ct (navigation mode) …")
    sample = visualize_stage1.load_sample_frames()
    obs_tokens_np = visualize_stage2.extract_obs_tokens(model, sample["obs_raw"])
    goal_token_np = visualize_stage2.extract_goal_token(model, sample["obs_raw"], sample["goal_raw"])

    # Same tokens_in construction as visualize_stage4.py: 4 obs tokens + 1 goal token.
    obs_tokens = torch.from_numpy(obs_tokens_np).unsqueeze(0).to(DEVICE)
    goal_token = torch.from_numpy(goal_token_np).unsqueeze(0).to(DEVICE)
    tokens_in = torch.cat([obs_tokens, goal_token.unsqueeze(1)], dim=1)

    # Navigation mode (goal active) Transformer pass, then pool to ct — both
    # reused directly from visualize_stage4.py rather than reimplemented.
    output_nav, _ = visualize_stage4.run_transformer_pass(model, tokens_in, mask_goal=False)
    ct = visualize_stage4.pool_ct(output_nav, mask_goal=False)
    print(f"  ct shape : {tuple(ct.shape)}")

    obs_raw_last = sample["obs_raw"][-1]   # current frame, raw uint8 RGB (no normalisation)

    run_dir = OUTPUTS_DIR / f"{TRAJ_NAME}_f{FRAME_IDX}"
    print(f"\n[3/4] Running Stage 5 (denoising trace + 3 PNGs) …")
    run_stage5(model, ct, obs_raw_last, save_dir=run_dir)

    print(f"\n{SEP}")
    print("Stage 5 complete. 3 files in debug_visuals/outputs/")
    print(f"(actual path: {run_dir})")
    print(SEP)


if __name__ == "__main__":
    main()
