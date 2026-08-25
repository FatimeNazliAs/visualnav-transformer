#!/usr/bin/env python3
# deeper_visuals/p1_inputs/run_model.py
"""
P1 — Inputs. What the model is handed before it thinks about anything.

Writes out/p1/<tag>/:
    stage1_frame_strip.png   4 observation frames [t-3 … t] + the goal frame t+8
    facts.json               the frame indices and the real tensor shapes

The strip is adapted from the frozen debug_visuals/visualize_stage1.py. Two
things changed. The scene arrives on a PhaseConfig instead of module globals, so
one figure serves any sample; and the heavy matplotlib suptitle is gone, because
the advisor page supplies its own heading and caption and stacking a second
title on top made the figure read like a slide.

This phase deliberately does not run a forward pass. Its subject is the input
itself, and every number below is measured off the arrays actually built for the
model rather than copied out of settings.py — so a wrong hyperparameter shows up
as a wrong figure instead of a matching pair of wrong constants.

Run (inside the container, from the repo root):
    python3 deeper_visuals/p1_inputs/run_model.py
or, the usual way:
    ./deeper_visuals/p1_inputs/update.sh
"""

from __future__ import annotations

from pathlib import Path

from deeper_visuals.common import facts, figures, settings, viz
from deeper_visuals.common.config import load_config
from deeper_visuals.common.data import load_sample
from deeper_visuals.common.facts import write_facts
from deeper_visuals.common.viz import plt

PHASE_DIR = Path(__file__).resolve().parent
FIGURE_NAME = "stage1_frame_strip.png"

# Width of the gap column that separates the observation group from the goal,
# as a fraction of one frame panel. Purely visual — it holds the divider rule.
GAP_RATIO = 0.34


# ══════════════════════════════════════════════════════════════════════════════
# The figure
# ══════════════════════════════════════════════════════════════════════════════

def _group_label(fig, axes: list, text: str, *, color: str) -> None:
    """Caption a run of panels, centred just under them."""
    left, right, bottom, _ = viz.band(axes)
    fig.text(
        (left + right) / 2, bottom - 0.13, text,
        ha="center", va="center", fontsize=10.5, color=color, fontweight="bold",
    )


def _divider(fig, left_axes: list, right_ax, *, color: str) -> None:
    """The dashed rule between the two encoders' inputs, sized to the frames."""
    x = (viz.band(left_axes)[1] + viz.band([right_ax])[0]) / 2
    _, _, bottom, top = viz.band(left_axes)
    fig.add_artist(plt.Line2D(
        [x, x], [bottom, top],
        transform=fig.transFigure, color=color, linewidth=1.0, linestyle=(0, (4, 4)),
    ))


def plot_frame_strip(sample: dict, save_path):
    """
    The hero figure: [t-3][t-2][t-1][t = now] │ [goal t+8].

    Orange border = the current frame. Purple = the goal. The gap column carries
    a vertical rule, because the two groups go to two different encoders and a
    reader who misses that reads the strip as five frames of one video.
    """
    obs_raw, goal_raw = sample.obs_raw, sample.goal_raw
    obs_idxs, goal_idx = sample.obs_idxs, sample.goal_idx
    n_obs = len(obs_raw)

    widths = [1.0] * n_obs + [GAP_RATIO, 1.0]
    fig, axes = plt.subplots(
        1, len(widths),
        figsize=(1.85 * sum(widths), 2.4),
        gridspec_kw={"width_ratios": widths, "wspace": 0.10},
    )
    obs_axes, gap_ax, goal_ax = list(axes[:n_obs]), axes[n_obs], axes[-1]
    gap_ax.axis("off")

    # ── Observation context -> encoder psi ────────────────────────────────────
    # The frame names come from the scene rather than from this loop, so the
    # words on this strip and the words on P3's attention axes are the same
    # strings and not two lists that agree by inspection.
    labels = sample.token_labels
    for offset, (ax, frame, fidx) in enumerate(zip(obs_axes, obs_raw, obs_idxs)):
        is_current = offset == n_obs - 1
        heading = "now  (t)" if is_current else labels[offset]
        figures.scene_frame(ax, frame, kind="now" if is_current else "past",
                            title=f"{heading}\nframe {fidx}")

    # ── Goal -> encoder phi ───────────────────────────────────────────────────
    figures.scene_frame(
        goal_ax, goal_raw, kind="goal",
        title=f"goal  (t + {settings.NUM_ACTIONS})\nframe {goal_idx}")

    # Room at the bottom for the group labels, which sit outside every axes and
    # so are invisible to the layout engine.
    fig.subplots_adjust(bottom=0.22)

    viz.settle(fig)
    _divider(fig, obs_axes, goal_ax, color=viz.COLOR_MUTED)
    _group_label(fig, obs_axes, "what it has just seen", color=viz.COLOR_MUTED)
    _group_label(fig, [goal_ax], "where to go", color=viz.COLOR_GOAL)

    return viz.save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# The numbers
# ══════════════════════════════════════════════════════════════════════════════

def measure_input(sample: dict) -> dict:
    """
    Shapes taken off the arrays the model would actually receive.

    obs_tensor is the four frames channel-concatenated (12 x 96 x 96); the
    encoder splits them back apart internally. goal_tensor is the goal frame
    alone (3 x 96 x 96) — NoMaD_ViNT stacks it with the current frame inside
    forward(), which is why encoder phi is a 6-channel network while only three
    channels are handed in.
    """
    obs_tensor  = sample.obs_input
    goal_tensor = sample.goal_input
    height, width = obs_tensor.shape[1:]

    n_obs = len(sample.obs_raw)
    channels_per_frame = obs_tensor.shape[0] // n_obs        # 12 / 4 = 3, i.e. RGB
    phi_channels = channels_per_frame + goal_tensor.shape[0]  # current frame + goal

    # Shapes are the one thing still written as strings — see facts.shape_str,
    # which is where the exemption is argued and where every phase now spells it.
    return {
        "n_obs_frames":     n_obs,
        "n_frames_in":      n_obs + 1,
        "image_size":       facts.shape_str(height, width),
        "obs_tensor":       facts.shape_str(*obs_tensor.shape),
        "goal_tensor":      facts.shape_str(*goal_tensor.shape),
        "phi_input_tensor": facts.shape_str(phi_channels, height, width),
        "goal_lead_steps":  settings.NUM_ACTIONS,
        "n_input_values":   int(obs_tensor.size + goal_tensor.size),
        "value_min":        float(min(obs_tensor.min(), goal_tensor.min())),
        "value_max":        float(max(obs_tensor.max(), goal_tensor.max())),
    }


def main() -> None:
    cfg = load_config(PHASE_DIR)
    print(f"\n  P1 — Inputs\n  {cfg.summary()}\n  {cfg.description}\n")

    sample = load_sample(cfg)
    strip = plot_frame_strip(sample, cfg.out_dir / FIGURE_NAME)
    write_facts(
        cfg,
        {"model": cfg.weights_provenance(), "input": measure_input(sample)},
        figures=[strip],
    )


if __name__ == "__main__":
    main()
