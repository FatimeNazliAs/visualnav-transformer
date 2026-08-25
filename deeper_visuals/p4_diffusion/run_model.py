#!/usr/bin/env python3
# deeper_visuals/p4_diffusion/run_model.py
"""
P4 — The diffusion policy. A page of noise is rubbed out until a path is left.

Everything before this phase produced one thing: c_t, 256 numbers summarising
what the robot can see and where it is going. This phase is what happens to it.
c_t is not decoded into a path. It is used as the *condition* on a denoiser that
starts from a random draw and is run K times:

    a_K ~ N(0, I)                       8 x 2 numbers, pure noise
    for k = K-1 … 0:
        eps = noise_pred_net(a_k, k, global_cond=c_t)
        a_k = scheduler.step(eps, k, a_k)
    a_0                                 the action sequence

c_t enters as `global_cond` on every one of those K calls, not once at the
start — the same context steers each step. Two things follow that are worth
seeing rather than being told: the network never outputs a path directly, and
the path that appears is the one c_t makes probable, not the one the noise
happened to start near.

Writes out/p4/<tag>/:
    stage4_diffusion_pair.png    training buries a real route; running it digs one out
    stage4_outputs.png           the two things c_t is turned into
    facts.json                   K, the tensor's shape, how the descent shrinks,
                                 where the clean path lands, the distance head

Adapted from the frozen debug_visuals/visualize_stage5.py (its denoising part).
What changed, and why:

1. One scale across all eleven panels. The old strip let each panel autoscale,
   which rendered a 4.7 m flail and a 1.3 m path at the same apparent size — so
   eleven panels of similar-looking squiggle, with the one quantity that
   actually falls monotonically scaled out of the picture. The shrink IS the
   figure; it only exists on a shared axis.
2. The actions are decoded properly. The old figure ran them through a unicycle
   integrator, reading the two channels as (linear velocity, angular velocity).
   NoMaD's are neither: with learn_angle=False they are normalised *position
   deltas*, and the inverse the model was fitted against is
   train_utils.get_action — unnormalise, then accumulate. That now lives in
   common/denoise.to_waypoints, so P5 inherits the corrected version.
3. Every panel carries a ghost of the final path. Eleven differently-shaped
   lines on a shared axis still read as eleven unrelated pictures without a
   fixed reference to converge on.
4. A second view of the same run, as values. The strip shows a path because a
   path is what a reader can hold; the tensor is 16 numbers, and the phase is
   about numbers being cleaned up. One figure would have had to imply the other.

Run (inside the container, from the repo root):
    python3 deeper_visuals/p4_diffusion/run_model.py
or, the usual way:
    ./deeper_visuals/p4_diffusion/update.sh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from deeper_visuals.common import actions as action_space
from deeper_visuals.common import denoise, figures, measure, settings, viz
from deeper_visuals.common.config import load_config, read_raw
from deeper_visuals.common.data import Scene, load_ground_truth, load_sample
from deeper_visuals.common import facts
from deeper_visuals.common.facts import write_facts
from deeper_visuals.common import model as model_lib
from deeper_visuals.common.model import load_model
from deeper_visuals.common.viz import plt

PHASE_DIR = Path(__file__).resolve().parent
PAIR_FIGURE_NAME = "stage4_diffusion_pair.png"
OUTPUTS_FIGURE_NAME = "stage4_outputs.png"

DEFAULT_SEED = 0


# ══════════════════════════════════════════════════════════════════════════════
# The forward pass
# ══════════════════════════════════════════════════════════════════════════════

def run_descent(model, sample: Scene, device: str,
                seed: int) -> tuple[denoise.Denoised, model_lib.VisionEncoding]:
    """
    Encode the scene, then run the K-step descent once from a pinned noise draw.

    The goal is visible: P4 is about the denoiser, and P3 already owns the
    difference the mask makes. Running the navigation mode keeps this phase's
    figure comparable with the c_t that P3's page ends on.
    """
    obs_batch, goal_batch = sample.obs_batch, sample.goal_batch
    encoding = model_lib.encode_tokens(
        model, obs_batch, goal_batch, device, goal_mask=model_lib.GOAL_VISIBLE)
    return denoise.denoise(model, encoding.context, device, seed=seed), encoding


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — the descent as geometry
# ══════════════════════════════════════════════════════════════════════════════

def plot_diffusion_pair(result: denoise.Denoised, ladder: np.ndarray, save_path):
    """
    The hero: the same ruin, done deliberately on top and undone below.

    Diffusion is two processes and only one of them happens at run time. Showing
    the descent alone — which the first version of this figure did — describes
    the network as an eraser working on nothing in particular, and leaves the
    obvious question unasked: erasing toward WHAT? The answer is the top row.
    Training takes a route the robot really drove and buries it under noise one
    step at a time; the network's whole job is to undo one of those steps. The
    bottom row is that job run ten times from a random start.

    Columns are noise level, not time, and that is what makes the rows
    comparable: both run from fully buried on the left to clean on the right, so
    any column shows the real route and the model's estimate at the same amount
    of corruption. The arrows disagree because the processes do — training moves
    left, inference moves right.

    The rightmost column is the payoff and the reason for the whole arrangement:
    the route actually driven, directly above what the model produced having
    never seen it.
    """
    forward = ladder[::-1]                      # noisiest first, real route last
    reverse = result.paths                      # noise first, prediction last
    n_panels = len(reverse)
    box = figures.shared_box(
        np.concatenate([denoise.to_waypoints(forward), reverse]))

    fig = plt.figure(figsize=(11.4, 4.1))
    grid = fig.add_gridspec(2, n_panels, wspace=0.16, hspace=0.52)

    rows = (
        figures.Row("training", "buries the real route",
                    denoise.to_waypoints(forward), viz.COLOR_TRUTH,
                    "the real route", "←"),
        figures.Row("running it", "digs one back out",
                    reverse, viz.COLOR_NAV, "what the model made", "→"),
    )

    drawn = []
    for r, row in enumerate(rows):
        axes = [fig.add_subplot(grid[r, c]) for c in range(n_panels)]
        figures.draw_row(axes, row, box)
        drawn.append(axes)

    fig.subplots_adjust(left=0.135, right=0.99, top=0.88, bottom=0.16)
    viz.settle(fig)

    for row, axes in zip(rows, drawn):
        figures.label_row(fig, axes, row)
    figures.number_panels(fig, drawn[1])
    return viz.save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — what the stage actually hands on
# ══════════════════════════════════════════════════════════════════════════════

def plot_outputs(result: denoise.Denoised, truth: np.ndarray, distance: dict, save_path):
    """
    The two things c_t is turned into, side by side.

    They are NOT two outputs of the diffusion process, and drawing them as a
    pair is the only honest way to say so. c_t feeds two separate heads:

        noise_pred_net  ConditionalUnet1D, denoised K times -> the action
                        sequence, NUM_ACTIONS x ACTION_DIM
        dist_pred_net   DenseNetwork, a 256 -> 64 -> 16 -> 1 MLP, one forward
                        pass, no diffusion anywhere -> temporal distance

    An earlier version of this figure showed the descent a second time, as a
    grid of its raw values. It was the same fact twice and it read as neither.
    What was missing from the page was never a second view of the descent — it
    was the descent's *product*, and the second head that never touches it.

    The distance panel is drawn against the goal frame's real offset because
    the estimate means nothing alone. Here the head reads the scene as further
    off than it is, which is the sort of thing a figure should show rather than
    a page claim.
    """
    fig, (path_ax, dist_ax) = plt.subplots(
        1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})

    _draw_action_sequence(path_ax, result.path, truth)
    _draw_distance(dist_ax, distance)

    fig.subplots_adjust(left=0.06, right=0.965, top=0.86, bottom=0.17, wspace=0.30)
    return viz.save_fig(fig, save_path)


def _draw_action_sequence(ax, path: np.ndarray, truth: np.ndarray) -> None:
    """
    The clean action sequence, with its waypoints marked.

    Drawn on its own scale rather than the descent's: this panel is no longer
    making a comparison with the noise, it is showing what the robot was told
    to do, and on the descent's box that is a thumbnail in the middle.
    """
    # The real route goes down thick and the prediction thin on top of it: the
    # two agree to within a few centimetres, so drawn at equal weight the lower
    # one simply disappears and the panel looks like it plots one path.
    figures.draw_path(ax, truth, viz.COLOR_TRUTH, width=5.0, alpha=0.45)
    figures.draw_path(ax, path, viz.COLOR_NAV, width=2.0, dot=6.0)
    ax.plot(0, 0, marker="o", markersize=8.0, color=viz.COLOR_MUTED, zorder=5)
    ax.legend(handles=[
        Line2D([], [], color=viz.COLOR_NAV, linewidth=2.0, label="what the model made"),
        Line2D([], [], color=viz.COLOR_TRUTH, linewidth=5.0, alpha=0.45,
               label="where the robot really went"),
    ], loc="lower left", frameon=False, fontsize=viz.LABEL_SIZE,
        labelcolor=viz.COLOR_MUTED, handlelength=1.5, borderpad=0.1)
    ax.annotate("the robot, now", xy=(0, 0), xytext=(6, -14),
                textcoords="offset points", fontsize=viz.LABEL_SIZE,
                color=viz.COLOR_MUTED)
    # The 8th waypoint IS the goal. load_ground_truth walks NUM_ACTIONS frames
    # on from the current one, which is exactly the frame the goal picture was
    # taken at — so the far end of the real route is the goal, and the panel
    # was already showing it. Readers asked where the goal was, which means
    # "already on screen, unnamed" was indistinguishable from "not drawn".
    # Named in the series' goal purple, so it reads as the same thing the right
    # panel is talking about.
    ax.plot(truth[-1, 1], truth[-1, 0], marker="o", markersize=10.0,
            markerfacecolor="none", markeredgecolor=viz.COLOR_GOAL,
            markeredgewidth=1.8, zorder=6)
    # Above the marker, not beside it: the route's own top segment occupies the
    # space to its right, and a label set there crosses the path it describes.
    ax.annotate(f"step {settings.NUM_ACTIONS} — where the\ngoal picture was taken",
                xy=(truth[-1, 1], truth[-1, 0]), xytext=(0, 14),
                textcoords="offset points", fontsize=viz.LABEL_SIZE,
                color=viz.COLOR_GOAL, ha="center")

    both = np.concatenate([path, truth])
    span = float(np.abs(both).max()) * 1.22
    centre = both.mean(axis=0)
    ax.set_xlim(centre[1] - span, centre[1] + span)
    ax.set_ylim(min(centre[0] - span, -0.15), centre[0] + span)
    ax.set_aspect("equal")
    viz.plate(ax, edge=viz.COLOR_NAV, width=1.4,
              title="the actions — one dot per step", bold=True)
    ax.set_ylabel("ahead ↑", fontsize=viz.LABEL_SIZE, color=viz.COLOR_MUTED,
                  labelpad=4)


def _draw_distance(ax, distance: dict) -> None:
    """
    The other head's answer, against the offset it was asked about.

    Two bars, not one. A predicted "12" is unreadable on its own — the reader
    has no idea whether that is close — and the comparison is the whole content
    of the panel.
    """
    values = [distance["actual"], distance["predicted"]]
    # Formatted from the measurements, never written out. These read "8" and
    # "12" on the hero scene, and both move with the scene — spelling them into
    # the string is how a figure ends up captioning a run it did not come from.
    # The parenthetical that used to hang off the first label is gone: the left
    # panel now marks that spot on the route itself, and the long tick text
    # overran into the neighbouring panel.
    labels = [f"really {distance['actual']:.0f} steps away",
              f"the model guesses {distance['predicted']:.0f}"]
    colours = [viz.COLOR_MUTED, viz.COLOR_GOAL]

    bars = ax.barh([1, 0], values, height=0.52, color=colours)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(labels, fontsize=viz.LABEL_SIZE)
    for label, colour in zip(ax.get_yticklabels(), colours):
        label.set_color(colour)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("steps away", fontsize=viz.LABEL_SIZE, color=viz.COLOR_MUTED,
                  labelpad=6)
    ax.tick_params(axis="x", labelsize=8.5, colors=viz.COLOR_MUTED, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_edgecolor(viz.COLOR_MUTED)
    ax.set_title("the distance — steps until it reaches the goal picture",
                 fontsize=viz.TITLE_SIZE,
                 pad=viz.TITLE_PAD, color=viz.COLOR_GOAL, fontweight="bold")
    ax.bar_label(bars, fmt=" %.0f", fontsize=viz.LABEL_SIZE,
                 color=viz.COLOR_MUTED, padding=1)


# ══════════════════════════════════════════════════════════════════════════════
# The numbers
# ══════════════════════════════════════════════════════════════════════════════

def measure_descent(result: denoise.Denoised) -> dict:
    """
    What falls as the descent proceeds, and where it ends up.

    `travelled_per_step` is the honest series and the reason it is recorded in
    full rather than summarised: it does NOT fall monotonically. On the hero
    scene it goes 4.7 -> 3.7 -> 4.1 m, an overshoot at k=8, before settling.
    Quoting only its two ends would let the page imply a smooth creep toward the
    answer, which is not what the figure shows and not what diffusion does.

    Distances are metres. common/denoise.to_waypoints has already undone the
    training normalisation and applied the dataset's waypoint spacing, so these
    are on the floor rather than in the model's units.
    """
    paths = result.paths
    final = paths[-1]
    per_step = [action_space.travelled(p) for p in paths]
    spread = [float(np.abs(step).max()) for step in result.trajectory]

    return {
        "n_steps":            result.n_steps,
        "n_states":           len(paths),
        "seed":               result.seed,
        "shape":              facts.shape_str(*result.trajectory.shape),
        "n_numbers":          int(result.actions.size),
        "travelled_noise":    per_step[0],
        "travelled_final":    per_step[-1],
        "travelled_per_step": per_step,
        # The widest single value in the tensor. Falls from well outside the
        # trained range to comfortably inside it. Recorded rather than drawn:
        # the raw-value grid this used to describe was cut from plot_outputs.
        "spread_noise":       spread[0],
        "spread_final":       spread[-1],
        "n_rises":            _n_rises(per_step),
        "ahead":              float(final[-1, 0]),
        # The signed value, in the model's own frame: +y is to the robot's left
        # (see settings.ACTION_MIN). The page cannot use it directly — "-0.61 m
        # to its right" — so the magnitude and the direction are recorded
        # separately, and the direction as a word rather than as a sign, so the
        # sentence stays true on a scene that turns the other way.
        "left":               float(final[-1, 1]),
        "sideways":           float(abs(final[-1, 1])),
        "side":               "right" if final[-1, 1] < 0 else "left",
    }


def _n_rises(per_step: list[float]) -> int:
    """
    How many of the K passes left the path LONGER than they found it.

    The page's claim is that the descent does not creep toward the answer, and a
    claim like that needs a number or it is an impression. On the hero scene it
    is 2 of 10 — the path grows at the third pass and again at the fifth before
    the run settles.

    Counted rather than thresholded. An earlier version asked when the length
    came within 10% of its final value and stayed there, which returned "after
    the tenth of ten passes" — technically true, and useless: a measure whose
    answer is always "at the end" is measuring the end, not the settling.
    """
    return int(sum(later > earlier
                   for earlier, later in zip(per_step, per_step[1:])))


def measure_against_truth(result: denoise.Denoised, truth: np.ndarray) -> dict:
    """
    How close the uncovered path is to the route the robot actually drove.

    The comparison is free — the scene comes from a recorded trajectory, so the
    future is on disk — and without it the page can say the descent produced a
    smooth right turn but not whether it produced the RIGHT smooth right turn.

    `endpoint` is the headline because it is the error a reader can picture; the
    mean over all NUM_ACTIONS waypoints is kept beside it so a single lucky
    endpoint cannot flatter the result.

    The distance itself comes from common/measure.error_per_step. P5 asks the
    same question of six paths at once, and the two phases had each written a
    `measure_against_truth` of their own — one definition each of a number both
    of their pages quote.
    """
    per_step = measure.error_per_step(result.path, truth)
    # Both units are stored. Everything else in this phase is metres, but these
    # are the only distances small enough that metres round to two noisy
    # decimals, and "7 cm" is the reading the page wants.
    return {
        "endpoint":     float(per_step[-1]),
        "endpoint_cm":  float(per_step[-1]) * 100,
        "mean":         float(per_step.mean()),
        "mean_cm":      float(per_step.mean()) * 100,
        "worst":        float(per_step.max()),
    }


def conditioning_facts(model) -> dict:
    """
    How c_t reaches the denoiser: as `global_cond`, on every one of the K calls.

    Recorded rather than asserted in prose because it is the single sentence of
    this phase most likely to be got wrong — the obvious reading is that c_t is
    decoded into a path, or fed in once at the start.
    """
    unet = model.noise_pred_net
    head = model.dist_pred_net
    return {
        "context_dim":  settings.ENCODING_SIZE,
        # A list, not "64 → 128 → 256". build_page.fill joins a list per
        # element with " → " already, and is tested for it.
        "down_dims":    list(settings.DOWN_DIMS),
        "unet_params":  float(sum(p.numel() for p in unet.parameters())) / 1e6,
        "head_params":  float(sum(p.numel() for p in head.parameters())) / 1e6,
        "enters_as":    "global_cond",
        "times_used":   settings.K_DENOISING,
        # FiLM, bias-only: nomad.yaml sets cond_predict_scale: False, so each
        # residual block adds a per-channel bias projected from [c_t ; step
        # embedding] rather than a scale-and-bias pair.
        "conditioning": "FiLM bias per residual block",
    }


def main() -> None:
    cfg = load_config(PHASE_DIR)
    seed = int(read_raw(PHASE_DIR).get("seed", DEFAULT_SEED))
    print(f"\n  P4 — Diffusion policy\n  {cfg.summary()}  seed={seed}\n  {cfg.description}\n")

    sample = load_sample(cfg)
    truth = load_ground_truth(cfg)
    model, info = load_model(cfg)

    result, encoding = run_descent(model, sample, info["device"], seed)
    descent = measure_descent(result)
    distance = {
        "predicted": denoise.distance_to_goal(model, encoding.context, info["device"]),
        # The goal frame is taken NUM_ACTIONS frames on, and go_stanford is
        # sampled at waypoint_spacing 1, so its true temporal distance is that
        # many steps. This is the number the head was trained to reproduce.
        "actual":    float(settings.NUM_ACTIONS),
    }
    distance["over_by"] = distance["predicted"] / distance["actual"] - 1
    accuracy = measure_against_truth(result, truth["waypoints"])

    # One line. Every value is in facts.json a moment later; this is here to
    # confirm the descent happened and ended somewhere sane.
    print(f"  descent {result.trajectory.shape}  "
          f"{descent['travelled_noise']:.2f} m of noise → "
          f"{descent['travelled_final']:.2f} m of path  ·  "
          f"ends {descent['ahead']:.2f} m ahead, {descent['sideways']:.2f} m "
          f"{descent['side']}  ·  {accuracy['endpoint'] * 100:.1f} cm off the "
          f"real route  ·  distance head "
          f"{distance['predicted']:.1f} vs {distance['actual']:.0f} steps")

    ladder = denoise.corrupt(truth["deltas"], seed=seed)
    pair_fig = plot_diffusion_pair(result, ladder, cfg.out_dir / PAIR_FIGURE_NAME)
    outputs_fig = plot_outputs(result, truth["waypoints"], distance,
                               cfg.out_dir / OUTPUTS_FIGURE_NAME)

    write_facts(
        cfg,
        {
            "model":        info,
            "descent":      descent,
            "accuracy":     accuracy,
            "conditioning": conditioning_facts(model),
            "distance": distance,
            "goal": {
                "frames_ahead": settings.NUM_ACTIONS,
                # Frames, not seconds. See settings.METRIC_WAYPOINT_SPACING.
                "metres_ahead": settings.NUM_ACTIONS * settings.METRIC_WAYPOINT_SPACING,
            },
        },
        figures=[pair_fig, outputs_fig],
    )


if __name__ == "__main__":
    main()
