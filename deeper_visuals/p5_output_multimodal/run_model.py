#!/usr/bin/env python3
# deeper_visuals/p5_output_multimodal/run_model.py
"""
P5 — The output, sampled. One scene, run several times, is not one answer.

P4 pinned the noise draw so that its eleven pictures described a single descent.
That pin is this phase's subject: unpin it, run the SAME context vector
num_seeds times, and the model answers differently every time. Nothing about the
scene changed — c_t is computed once and reused — so every difference between
the runs came from randomness, and there are two sources of it:

    a_K ~ N(0, I)                       a different starting scribble per run
    scheduler.step(...)                 fresh noise re-injected at each reverse
                                        step but the last (ancestral sampling)

which is why this is sampling rather than a wobble in a deterministic function.

What the measurement actually shows, and what this page therefore claims
─────────────────────────────────────────────────────────────────────────
Not two routes. The runs do not fork: on every one of the ten curated scenes,
in both goal-masked and goal-visible mode, the widest gap between sampled
endpoints is 8-23% of the path's own length, and the hero scene sits at 16%.
There is no left-or-right bimodality anywhere in this range to photograph.

What IS there, and is large, is the shape of the disagreement. The runs leave
the robot together and separate as they go: on the hero scene about a centimetre
apart at the first waypoint and tens of centimetres by the eighth. That is a
real and useful fact — it is why the policy is re-run every step and only its
near waypoints are ever driven — and it is what these figures are built to show.

Writes out/p5/<tag>/:
    stage5_seed_spread.png     the runs together, and where they come apart
    stage5_in_and_out.png      each run's random start, and what it became
    facts.json                 how many runs, how far apart at each waypoint,
                               and how they sit against the recorded route

Deliberately NOT here, though the frozen debug_visuals/visualize_stage5.py drew
all three:

1. No mean trajectory. Averaging samples is the thing a diffusion policy exists
   in order not to do, and a bold dashed mean demotes the six runs to error bars
   around an answer the model never produced. The reference line is the route
   the robot actually drove.
2. No velocity-against-time panel. The channels are normalised position deltas
   (learn_angle: False), not velocities, and the repo records no frame rate — so
   there is no time axis to plot against. Distance and waypoint index only.
3. No path drawn over the camera frame. data_config.yaml carries camera_metrics
   for `recon` alone; go_stanford has none, so there is no projection to do. The
   old overlay auto-scaled the path to 35% of the image, which looks like a
   projection and is not one.
4. No distance head. It is P4's, explained there once (CLAUDE.md decision 11).

Run (inside the container, from the repo root):
    python3 deeper_visuals/p5_output_multimodal/run_model.py
or, the usual way:
    ./deeper_visuals/p5_output_multimodal/update.sh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from deeper_visuals.common import denoise, figures, measure, settings, viz
from deeper_visuals.common.config import load_config, read_raw
from deeper_visuals.common.data import Scene, load_ground_truth, load_sample
from deeper_visuals.common.facts import write_facts
from deeper_visuals.common import model as model_lib
from deeper_visuals.common.model import load_model
from deeper_visuals.common.viz import plt

PHASE_DIR = Path(__file__).resolve().parent
SPREAD_FIGURE_NAME = "stage5_seed_spread.png"
IN_OUT_FIGURE_NAME = "stage5_in_and_out.png"

DEFAULT_NUM_SEEDS = 6


# ══════════════════════════════════════════════════════════════════════════════
# The forward pass
# ══════════════════════════════════════════════════════════════════════════════

def run_seeds(model, sample: Scene, device: str, num_seeds: int) -> list:
    """
    Encode the scene ONCE, then sample the diffusion head num_seeds times from it.

    Encoding once is not an optimisation, it is the experiment: every run shares
    one context vector, so nothing about what the robot sees or where it is
    going can account for a difference between them. Seeds are 0 … num_seeds-1
    so that config.yaml's one number fully determines the figure.

    The goal is visible throughout. P3 owns the difference the mask makes, and
    the ten-scene sweep behind this phase found the spread essentially unchanged
    with the goal hidden — so a masked run here would add a second subject and
    show nothing.
    """
    encoding = model_lib.encode_tokens(
        model, sample.obs_batch, sample.goal_batch, device,
        goal_mask=model_lib.GOAL_VISIBLE)
    return [denoise.denoise(model, encoding.context, device, seed=seed)
            for seed in range(num_seeds)]


def sampled_paths(runs: list) -> np.ndarray:
    """Every run's finished path — (n_runs, NUM_ACTIONS, ACTION_DIM), metres."""
    return np.stack([run.path for run in runs])


def starting_paths(runs: list) -> np.ndarray:
    """Every run's opening scribble, decoded the same way its answer is."""
    return np.stack([run.paths[0] for run in runs])


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — the runs together, and where they come apart
# ══════════════════════════════════════════════════════════════════════════════

def plot_seed_spread(paths: np.ndarray, truth: np.ndarray, gaps: np.ndarray,
                     save_path):
    """
    The hero: several runs of one scene, and the gap between them as it opens.

    Two panels making one argument, because either alone is misreadable. The
    left panel shows runs that a reader can see are all the same turn — and at
    this scale that is nearly all they can see, since the runs differ by
    centimetres on a path of over a metre. Left there, the page would be
    claiming a spread the picture appears to deny. The right panel is that same
    spread measured, where centimetres are the axis rather than a rounding
    error, and it is monotone: the disagreement is not noise on top of the
    answer, it is a fan that opens with distance.

    Every run is drawn in one colour rather than a ramp of six. A ramp encodes
    an order, and these runs have none — they are exchangeable draws, and
    colouring them 1..6 invites a reader to look for a progression from the
    first to the last that does not exist.
    """
    fig, (path_ax, gap_ax) = plt.subplots(
        1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})

    _draw_runs(path_ax, paths, truth)
    _draw_gap_curve(gap_ax, gaps)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.17, wspace=0.28)
    return viz.save_fig(fig, save_path)


def _draw_runs(ax, paths: np.ndarray, truth: np.ndarray) -> None:
    """
    Every run's path from the robot outward, over the route it really drove.

    The real route goes down thick and faint and the runs thin on top, the same
    arrangement P4's outputs panel uses — the runs agree with it to within
    centimetres, so at equal weight the lower line simply vanishes and the panel
    looks like it is plotting one thing.

    The gap is annotated at the first waypoint and at the last, between the two
    runs that actually achieve it there. The pair comes from
    measure.widest_pair, which is also what facts.json records, so the arrow
    cannot end up spanning one pair while the page quotes another.
    """
    figures.draw_path(ax, truth, viz.COLOR_TRUTH, width=5.0, alpha=0.45)
    for path in paths:
        figures.draw_path(ax, path, viz.COLOR_NAV, width=1.6, alpha=0.85, dot=3.2)
    ax.plot(0, 0, marker="o", markersize=8.0, color=viz.COLOR_MUTED, zorder=5)

    # The runs sweep from the bottom right of the box to the top left, leaving
    # two empty corners for three things that must not collide: the legend takes
    # the bottom left, the far gap's label the top right, and the near gap's
    # label the space between them. Placed in fractions of the panel rather than
    # as offsets from the arrows, because an offset from a point inside a dense
    # fan lands inside the fan — the whole difficulty here is that the arrows
    # are exactly where there is no room.
    _annotate_gap(ax, paths[:, 0], "first step", text_xy=(0.74, 0.29))
    _annotate_gap(ax, paths[:, -1], f"step {settings.NUM_ACTIONS}",
                  text_xy=(0.99, 0.88))

    ax.legend(handles=[
        Line2D([], [], color=viz.COLOR_NAV, linewidth=1.6,
               label=f"{len(paths)} runs of the same scene"),
        Line2D([], [], color=viz.COLOR_TRUTH, linewidth=5.0, alpha=0.45,
               label="where the robot really went"),
    ], loc="lower left", frameon=False, fontsize=viz.LABEL_SIZE,
        labelcolor=viz.COLOR_MUTED, handlelength=1.5, borderpad=0.1)

    # figures.shared_box rather than a span about the origin. Squaring a box
    # around the robot to hold a path that only ever goes forward-left spends
    # three of its quadrants on floor nothing visits, and shrinks the runs to a
    # thumbnail in the middle — which is fatal here, where the whole question is
    # whether you can see them come apart.
    figures.frame(ax, figures.shared_box(
        np.concatenate([paths, truth[None]])))
    viz.plate(ax, edge=viz.COLOR_NAV, width=1.4,
              title="same scene, sampled several times", bold=True)
    ax.set_ylabel("ahead ↑", fontsize=viz.LABEL_SIZE, color=viz.COLOR_MUTED,
                  labelpad=4)


def _annotate_gap(ax, points: np.ndarray, where: str, *,
                  text_xy: tuple) -> None:
    """
    Mark how far apart the runs are at one waypoint — an arrow, and the reading.

    The arrow spans the two runs `measure.widest_pair` picks out, and the
    centimetres are measured between those same two. Nothing is written out: a
    number typed beside a figure is a caption for whichever run produced it, and
    it outlives that run.

    `text_xy` is a position in the panel, right-aligned there, chosen by the
    caller that knows where this scene's runs left room.
    """
    first, second = measure.widest_pair(points)
    start, end = points[first], points[second]
    gap = float(np.linalg.norm(start - end))
    ax.annotate("", xy=(end[1], end[0]), xytext=(start[1], start[0]),
                arrowprops=dict(arrowstyle="<->", color=viz.COLOR_CURRENT,
                                linewidth=1.3, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(text_xy[0], text_xy[1], f"{where}: {gap * 100:.1f} cm apart",
            transform=ax.transAxes, fontsize=viz.LABEL_SIZE,
            color=viz.COLOR_CURRENT, ha="right", va="center", zorder=6)


def _draw_gap_curve(ax, gaps: np.ndarray) -> None:
    """
    The same disagreement, measured — centimetres against waypoint number.

    Centimetres because the quantity spans about one to a few tens of them and
    metres would render the interesting half as leading zeroes. Waypoint number
    because the horizontal axis is emphatically NOT time: the two channels are
    position deltas and the repo records no frame rate, so there is no rate at
    which these steps are taken and nothing here may be stated in seconds.

    The area under the line is filled for one reason: it makes the first
    waypoint's value read as almost-nothing rather than as a point that happens
    to sit low, which is the comparison the panel is for.
    """
    steps = np.arange(1, len(gaps) + 1)
    centimetres = gaps * 100

    ax.fill_between(steps, 0, centimetres, color=viz.COLOR_NAV, alpha=0.13)
    ax.plot(steps, centimetres, color=viz.COLOR_NAV, linewidth=2.0,
            marker="o", markersize=5.0, markeredgecolor="white",
            markeredgewidth=0.8, zorder=3)

    for index in (0, len(gaps) - 1):
        ax.annotate(f"{centimetres[index]:.1f} cm",
                    xy=(steps[index], centimetres[index]),
                    xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=viz.LABEL_SIZE, color=viz.COLOR_NAV,
                    fontweight="bold")

    ax.set_xlim(0.6, len(gaps) + 0.6)
    ax.set_ylim(0, float(centimetres.max()) * 1.30)
    ax.set_xticks(steps)
    # Counted off the data, never written out. A reader meeting "step 8" for
    # the first time cannot tell whether eight is the model's or ours, and the
    # axis is exactly where they ask.
    ax.set_xlabel(f"step of the plan — the model always plans {len(gaps)}",
                  fontsize=viz.LABEL_SIZE, color=viz.COLOR_MUTED, labelpad=6)
    ax.set_ylabel("how far apart the runs are (cm)", fontsize=viz.LABEL_SIZE,
                  color=viz.COLOR_MUTED, labelpad=6)
    ax.tick_params(labelsize=8.5, colors=viz.COLOR_MUTED, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_edgecolor(viz.COLOR_MUTED)
    ax.set_title("they agree nearby and part company further out",
                 fontsize=viz.TITLE_SIZE, pad=viz.TITLE_PAD,
                 color=viz.COLOR_NAV, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — what each run started from, and what it became
# ══════════════════════════════════════════════════════════════════════════════

def plot_in_and_out(starts: np.ndarray, paths: np.ndarray, save_path):
    """
    Each run's opening scribble above the plan it turned into.

    This is the figure that says where the differences come from, and it says it
    by being drawn on ONE box across both rows. The starts sprawl several metres
    and disagree completely; the answers are the same turn six times over, small
    in the same frame. Read together that is the phase's real finding: almost
    all of the randomness is washed out, and the fan the hero figure zooms into
    is the little of it that survives. Two boxes, one per row, would have shown
    the same two things and hidden the relation between them.

    Columns are runs, not steps. They have no order, so no arrow is drawn under
    them and no panel is the destination — which is why figures.Row leaves both
    of those optional rather than this file supplying a plausible-looking value.
    """
    n_runs = len(paths)
    box = figures.shared_box(np.concatenate([starts, paths]))

    fig = plt.figure(figsize=(1.75 * n_runs + 1.8, 4.3))
    grid = fig.add_gridspec(2, n_runs, wspace=0.16, hspace=0.30)

    rows = (
        figures.Row("what it starts from", "pure random scribble",
                    starts, viz.COLOR_MUTED),
        figures.Row("what it produces", "one plan for the next steps",
                    paths, viz.COLOR_NAV),
    )

    drawn = []
    for r, row in enumerate(rows):
        axes = [fig.add_subplot(grid[r, c]) for c in range(n_runs)]
        for ax, path in zip(axes, row.paths):
            figures.draw_path(ax, path, row.colour, width=1.5)
            figures.frame(ax, box)
            viz.plate(ax, edge=row.colour, width=0.9)
        drawn.append(axes)

    fig.subplots_adjust(left=0.155, right=0.99, top=0.94, bottom=0.13)
    viz.settle(fig)

    for row, axes in zip(rows, drawn):
        figures.label_row(fig, axes, row)
    figures.number_panels(fig, drawn[1],
                          labels=[f"run {i + 1}" for i in range(n_runs)])
    return viz.save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# The numbers
# ══════════════════════════════════════════════════════════════════════════════

def measure_spread(runs: list, paths: np.ndarray, starts: np.ndarray) -> dict:
    """
    How far apart the runs are, where, and how much of the noise survived.

    The seeds themselves are not recorded. They are 0 … n_runs-1 by
    construction — see run_seeds and config.yaml — so a list of them is a
    restatement of n_runs that renders on a page as "0 → 1 → 2 → 3 → 4 → 5".

    `gap_per_step` is the series the hero's right-hand panel plots and the
    reason it is recorded whole rather than as its two ends: the claim is that
    the disagreement GROWS along the plan, and a pair of endpoints is consistent
    with any shape in between.

    Every gap is a widest-pair distance, never a standard deviation. A std is a
    spread about a mean, and the mean of several sampled paths is precisely the
    averaged-away answer the diffusion head exists in order not to produce —
    quoting one would smuggle it onto the page as the thing the runs vary about.
    """
    gaps = measure.gap_per_step(paths)

    # Centimetres, not metres, and only centimetres. These gaps run from under
    # one to a few tens of them, so metres would put the whole interesting half
    # of the range behind leading zeroes — and holding both units invites the
    # page to quote one while a figure annotates the other. gap_per_step keeps
    # the underlying metres for the panel that plots them.
    #
    # There is deliberately no "growth" multiplier here. It would be a third
    # rounded number derived from two others already on the page, and a reader
    # who divides them would not get it back.
    return {
        "n_runs":         len(runs),
        "n_steps":        settings.NUM_ACTIONS,
        "gap_per_step":   gaps.tolist(),
        "gap_first_cm":   float(gaps[0]) * 100,
        "gap_last_cm":    float(gaps[-1]) * 100,
        # Whether the fan only ever opens. The page says it never narrows, and
        # on a scene where it did, that sentence would be false and nothing in
        # the picture would look wrong.
        "n_narrowings":   int((np.diff(gaps) < 0).sum()),
        # The same measurement on the scribbles the runs began from, so the
        # amount of randomness that did NOT survive can be stated rather than
        # implied by two rows of a figure.
        "start_gap":      float(measure.widest_gap(starts[:, -1])),
    }


def measure_against_truth(paths: np.ndarray, truth: np.ndarray) -> dict:
    """
    How the runs sit against the route the robot actually drove.

    This is what licenses the word "valid". A page can show several different
    paths from one scene without it, but not that all of them are reasonable —
    and a fan of plausible-looking wrong answers looks exactly the same.

    The WORST run is what the page quotes, not the best or the mean. "Each of
    them ends within this far of the real route" is a claim about the whole set
    that one lucky sample cannot flatter; "the closest one got within 2 cm"
    would be true of a set that also contained a run driving into a wall.
    """
    endpoint = measure.error_per_step(paths, truth)[:, -1]
    return {
        "endpoint_worst_cm": float(endpoint.max()) * 100,
        "endpoint_mean_cm":  float(endpoint.mean()) * 100,
    }


def sampling_facts() -> dict:
    """
    Where the randomness enters, counted.

    The reverse process re-injects noise at every step but the last — DDPM
    ancestral sampling, DDPMScheduler.step's own variance term — so a run's
    starting draw is one of K sources of randomness rather than the only one.
    That is the mechanism the page rests on, and it is the sentence most likely
    to be got wrong (the obvious reading is that only the first draw differs),
    so it is recorded as a number instead of asserted in prose.
    """
    return {
        "context_dim":      settings.ENCODING_SIZE,
        "k_steps":          settings.K_DENOISING,
        "noise_injections": settings.K_DENOISING - 1,
    }


def main() -> None:
    cfg = load_config(PHASE_DIR)
    num_seeds = int(read_raw(PHASE_DIR).get("num_seeds", DEFAULT_NUM_SEEDS))
    print(f"\n  P5 — Output & multimodality\n  {cfg.summary()}  "
          f"num_seeds={num_seeds}\n  {cfg.description}\n")

    sample = load_sample(cfg)
    truth = load_ground_truth(cfg)
    model, info = load_model(cfg)

    runs = run_seeds(model, sample, info["device"], num_seeds)
    paths = sampled_paths(runs)
    starts = starting_paths(runs)

    spread = measure_spread(runs, paths, starts)
    accuracy = measure_against_truth(paths, truth["waypoints"])

    # One line. Every value is in facts.json a moment later; this is here to
    # confirm the runs happened and came apart by a sane amount.
    print(f"  {spread['n_runs']} runs of one context vector  ·  "
          f"{spread['gap_first_cm']:.1f} cm apart at step 1 → "
          f"{spread['gap_last_cm']:.1f} cm at step {spread['n_steps']}  ·  "
          f"{spread['n_narrowings']} step(s) where the gap narrowed  ·  starts "
          f"were {spread['start_gap']:.1f} m apart  ·  "
          f"{accuracy['endpoint_mean_cm']:.1f} cm off the real route on average")

    spread_fig = plot_seed_spread(paths, truth["waypoints"],
                                  np.asarray(spread["gap_per_step"]),
                                  cfg.out_dir / SPREAD_FIGURE_NAME)
    in_out_fig = plot_in_and_out(starts, paths, cfg.out_dir / IN_OUT_FIGURE_NAME)

    write_facts(
        cfg,
        {
            "model":    info,
            "spread":   spread,
            "accuracy": accuracy,
            "sampling": sampling_facts(),
        },
        figures=[spread_fig, in_out_fig],
    )


if __name__ == "__main__":
    main()
