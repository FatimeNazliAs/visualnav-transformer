#!/usr/bin/env python3
# deeper_visuals/p3_transformer_masking/run_model.py
"""
P3 — Transformer & goal masking. Five tokens go in together; one vector comes out.

The same five tokens are pushed through the same weights twice, and the only
thing that differs is one boolean:

    navigation  — the goal token takes part in attention
    exploration — the goal token is hidden from attention by a
                  src_key_padding_mask. It is still encoded; the other four
                  tokens simply cannot see it, and the mean-pool is rescaled
                  over the four that remain (NoMaD_ViNT's avg_pool_mask).

That distinction is the phase. Zeroing the goal *image* instead would be a
different and wrong experiment: the encoder would happily embed a black frame
and the transformer would attend to that embedding.

Writes out/p3/<tag>/:
    stage3_attention.png       who looks at what, all four heads, both modes
    stage3_context_vector.png  the vector handed to the next stage, both modes
    facts.json                 transformer geometry, attention to the goal per
                               head, how far c_t moves, and the fidelity check

Adapted from the frozen debug_visuals/visualize_stage4.py. What changed:

1. The layer to plot is chosen from measurement, not inherited. The goal token
   is barely consulted before the last layer — the strongest head's attention to
   it runs 0.015 / 0.120 / 0.067 across layers 1-3 and then 0.482 at layer 4 —
   so layer 4 is the only one where masking has anything visible to remove.
   run_model records every layer's figure in facts.json so the claim is checkable.
2. The two modes sit in one figure, one above the other, rather than behind a
   tab. The content of this phase is a difference between two pictures; a widget
   that shows one at a time optimises for space at the cost of the comparison.
3. The cells carry no printed numbers. Eight panels of 25 cells is 200 numbers;
   the series' rule is that figures carry words and facts.json carries numbers.
   The goal column is outlined in the series' goal purple instead, so under the
   mask the reader sees a labelled box that has gone empty.
4. c_t is plotted from the model's own forward pass, not from the instrumented
   one — see attention.py on why they differ in the last decimal place.

Run (inside the container, from the repo root):
    python3 deeper_visuals/p3_transformer_masking/run_model.py
or, the usual way:
    ./deeper_visuals/p3_transformer_masking/update.sh
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Rectangle

from deeper_visuals.common import denoise, figures, measure, settings, viz
from deeper_visuals.common.config import load_config
from deeper_visuals.common.data import load_sample
from deeper_visuals.common.facts import write_facts
from deeper_visuals.common import model as model_lib
from deeper_visuals.common.model import load_model
from deeper_visuals.common.viz import plt
from deeper_visuals.p3_transformer_masking import attention

PHASE_DIR = Path(__file__).resolve().parent
ATTENTION_FIGURE_NAME = "stage3_attention.png"
CONTEXT_FIGURE_NAME = "stage3_context_vector.png"

# The last layer. Chosen by measurement (see the module docstring), not by
# convention: it is the only layer where the goal token is meaningfully attended
# to, and so the only one where hiding it changes the picture.
LAYER_SHOWN = -1

# The five tokens' names now come from data.Scene.token_labels, derived from
# settings.CONTEXT_SIZE. They used to be a literal list here with a comment
# saying they were deliberately P1's and P2's words — which asserted the
# agreement instead of arranging it, and baked in a context size of 3. Raising
# CONTEXT_SIZE would have left this figure mislabelled and nothing would fail.
GOAL_COLUMN = settings.N_TOKENS - 1

# Each mode: the label the panel carries, the sub-label that says what the
# picture literally shows, and the series colour.
MODES = (
    ("navigating", "goal in view",  model_lib.GOAL_VISIBLE, viz.COLOR_NAV),
    ("exploring",  "goal hidden",   model_lib.GOAL_HIDDEN,  viz.COLOR_EXPLORE),
)


# ══════════════════════════════════════════════════════════════════════════════
# The forward passes
# ══════════════════════════════════════════════════════════════════════════════

def run_both_modes(model, sample: dict, device: str) -> tuple[dict, dict]:
    """
    Encode the scene once, then run the transformer over it under both masks.

    Returns (passes, encodings) keyed by mode name. The encoders run once per
    mode as well, but only so that each mode's *true* c_t — the model's own
    fused-path output — is on hand to plot and to check the instrumented pass
    against. The tokens themselves are identical across modes by construction,
    and that is asserted rather than assumed.
    """
    obs_batch, goal_batch = sample.obs_batch, sample.goal_batch

    passes, encodings = {}, {}
    for name, _sublabel, goal_mask, _colour in MODES:
        encoding = model_lib.encode_tokens(
            model, obs_batch, goal_batch, device, goal_mask=goal_mask)
        encodings[name] = encoding
        passes[name] = attention.run(
            model, encoding.tokens[0], device, goal_mask=goal_mask)

    first, second = (encodings[name] for name, *_ in MODES)
    if not np.array_equal(first.tokens, second.tokens):
        raise RuntimeError(
            "the goal mask changed the encoder tokens, which it cannot do — "
            "masking hides the goal token from attention, it does not change "
            "how anything is encoded. Check common.model.encode_tokens."
        )
    return passes, encodings


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — who looks at what
# ══════════════════════════════════════════════════════════════════════════════

def _draw_head(ax, weights: np.ndarray, vmax: float, *, colour: str,
               labels: list[str], title: str | None, show_y: bool,
               show_x: bool):
    """
    One head's N x N attention grid.

    Row r is what token r looked at, and each row sums to 1 — the grid is five
    budgets, not twenty-five independent readings. Both axes are labelled with
    the frame names rather than thumbnails: the names are what the rest of the
    page and every other phase call these frames, and a picture on the axis made
    the reader map picture -> name anyway before they could read the caption.

    The goal column is outlined in the goal purple in every panel, including the
    ones where it is empty: that is what makes exploration read as "the thing
    that was there is gone" rather than as a differently-shaped picture.
    """
    # aspect="auto" so the grid fills its cell rather than shrinking to a square
    # inside it and drifting out of line with its neighbours.
    image = ax.imshow(weights, cmap="Purples", vmin=0, vmax=vmax, aspect="auto")
    viz.plate(ax, edge=colour, width=1.2, title=title)

    ax.add_patch(Rectangle(
        (GOAL_COLUMN - 0.5, -0.5), 1, settings.N_TOKENS,
        fill=False, edgecolor=viz.COLOR_GOAL, linewidth=1.8, clip_on=False,
    ))

    if show_y:
        ax.set_yticks(range(settings.N_TOKENS))
        ax.set_yticklabels(labels, fontsize=viz.LABEL_SIZE)
        _colour_frame_labels(ax.get_yticklabels())
        ax.set_ylabel("this picture …", fontsize=viz.LABEL_SIZE,
                      color=viz.COLOR_MUTED, labelpad=8)
    if show_x:
        ax.set_xticks(range(settings.N_TOKENS))
        ax.set_xticklabels(labels, fontsize=viz.LABEL_SIZE,
                           rotation=45, ha="right")
        _colour_frame_labels(ax.get_xticklabels())
    return image


def _colour_frame_labels(labels) -> None:
    """Tick labels in the series colours — 'now' is orange, 'goal' is purple."""
    for label in labels:
        text = label.get_text()
        if text == "now":
            label.set_color(viz.COLOR_CURRENT)
            label.set_fontweight("bold")
        elif text == "goal":
            label.set_color(viz.COLOR_GOAL)
            label.set_fontweight("bold")
        else:
            label.set_color(viz.COLOR_MUTED)


def plot_attention(passes: dict, labels: list[str], save_path):
    """
    The hero: four ways of looking across, two modes down.

    Each panel is one head — one whole way of looking, not a column of the grid
    beside it. They are titled "way N of 4" for exactly that reason: "way 1"
    alone reads as a column heading when it sits above a five-column grid.

    All four are shown, not just the two that use the goal. Ways 3 and 4 barely
    look at it even in navigation, and that is the honest content — masking does
    not switch the network off, it removes one input from the ways that were
    using it. Cropping to the interesting ones would delete the evidence for the
    more accurate story.

    One colour scale across all eight panels, so a shade means the same thing
    everywhere and the two rows are comparable at a glance.
    """
    n_heads = passes[MODES[0][0]].n_heads
    layers = {name: p.layer(LAYER_SHOWN) for name, p in passes.items()}
    vmax = max(float(w.max()) for w in layers.values())

    fig = plt.figure(figsize=(2.5 * n_heads + 1.7, 7.0))
    grid = gridspec.GridSpec(
        len(MODES), n_heads, figure=fig, wspace=0.22, hspace=0.30)

    panels, first_column = [], []
    for row, (name, sublabel, _mask, colour) in enumerate(MODES):
        weights = layers[name]
        is_last_row = row == len(MODES) - 1
        for head in range(n_heads):
            ax = fig.add_subplot(grid[row, head])
            image = _draw_head(
                ax, weights[head], vmax, colour=colour, labels=labels,
                title=f"way {head + 1} of {n_heads}" if row == 0 else None,
                show_y=head == 0, show_x=is_last_row,
            )
            panels.append(ax)
            if head == 0:
                first_column.append(ax)

    fig.subplots_adjust(left=0.135, right=0.985, top=0.91, bottom=0.25)
    viz.settle(fig)

    for ax, (name, sublabel, _mask, colour) in zip(first_column, MODES):
        figures.label_row(
            fig, [ax], figures.Row(name, sublabel, None, colour),
            rotate=90, fontsize=11.5, bold=True, gap=20.0)

    # Says which way to read the grid, once, in the figure itself — the caption
    # should not have to carry "rows, not columns".
    last = panels[-1].get_position()
    first = panels[-n_heads].get_position()
    fig.text((first.x0 + last.x1) / 2, 0.115,
             "… spent this much of its looking on each of these",
             ha="center", va="center", fontsize=viz.LABEL_SIZE,
             color=viz.COLOR_MUTED)

    viz.worded_key(fig, image, panels[-n_heads:],
                   ["barely looks", "looks hard"], drop=0.20)
    return viz.save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — the vector the next stage gets
# ══════════════════════════════════════════════════════════════════════════════

def plot_context_vector(encodings: dict, save_path):
    """
    c_t under both modes, and the gap between them.

    The first two strips are all but indistinguishable, and that is the finding
    rather than a flaw in the drawing: the two vectors sit at cosine 0.99. The
    figure would be dishonest if it hid that — so it shows both strips on one
    shared scale, and puts the gap on a third strip of its own where it can
    actually be seen. What the reader should take away is that masking does not
    replace c_t with something else, it nudges most of its numbers; whether that
    nudge matters is a question only the trajectories in P5 can answer.

    The third strip is the absolute gap, because the question it answers is how
    much each number moved, not in which direction. Plotted 1 x 256 rather than
    tiled into a square for the same reason P2's tokens are: the dimensions have
    no order and no neighbours, and a square would invent a structure to look for.
    """
    vectors = {name: encodings[name].context[0] for name, *_ in MODES}
    (nav_name, _, _, nav_colour), (exp_name, _, _, exp_colour) = MODES
    nav, explore = vectors[nav_name], vectors[exp_name]
    gap = np.abs(nav - explore)
    vabs = float(max(np.abs(nav).max(), np.abs(explore).max()))

    # Nested, not a flat 3 x 1: the first two strips are a pair on one scale and
    # belong close together, while the gap strip needs room beneath the pair's
    # colour key. A uniform hspace cannot say that — it either crowds the key or
    # leaves a hole between two strips meant to be read as one thing.
    fig = plt.figure(figsize=(12.6, 6.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=1.05)
    pair = outer[0].subgridspec(2, 1, hspace=0.55)
    cells = [pair[0], pair[1], outer[1]]

    strips = [
        (nav,     f"{nav_name} — goal in view",  "RdBu_r", -vabs, vabs, nav_colour),
        (explore, f"{exp_name} — goal hidden",   "RdBu_r", -vabs, vabs, exp_colour),
        (gap,     "how far each number moved",   "Purples", 0.0, float(gap.max()),
         viz.COLOR_GOAL),
    ]

    axes, images = [], []
    for values, title, cmap, vmin, vmax, colour in strips:
        ax = fig.add_subplot(cells[len(axes)])
        images.append(viz.vector_strip(ax, values, cmap=cmap,
                                       vmin=vmin, vmax=vmax))
        viz.plate(ax, edge=colour, width=1.2, title=title)
        axes.append(ax)

    fig.subplots_adjust(left=0.035, right=0.985, top=0.93, bottom=0.15)
    viz.settle(fig)

    # Two scales, so two keys. The first two strips share one diverging scale —
    # its key hangs under the second of them, since it describes both. Leaving
    # the top pair unkeyed (the first draft did) meant the only colours a reader
    # met first were the ones nothing explained.
    viz.worded_key(fig, images[1], [axes[1]], ["lower", "0", "higher"],
                   caption=f"the value of each of the {nav.size} numbers",
                   drop=0.075)
    viz.worded_key(fig, images[2], [axes[2]], ["unchanged", "moved most"],
                   caption="how far it moved when the goal was hidden",
                   drop=0.075)
    return viz.save_fig(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# The numbers
# ══════════════════════════════════════════════════════════════════════════════

def measure_attention(passes: dict, labels: list[str]) -> dict:
    """
    How much attention the goal token receives, per head and per layer.

    The headline is the share the four observation tokens give the goal in the
    plotted layer: it is what the figure shows, what the page quotes, and what
    goes to exactly zero under the mask. The per-layer figures are recorded so
    that "the goal is barely consulted before the last layer" — the reason this
    phase plots layer 4 — is a checkable claim rather than a comment.

    Everything here is a fraction, not a percentage string. The page decides it
    reads as "48%" with a {…:.0%} spec, which is what lets the same number be
    diffed against another run or asserted with a tolerance.
    """
    facts = {}
    for name, *_ in MODES:
        pass_ = passes[name]
        shown = pass_.layer(LAYER_SHOWN)
        # Column GOAL_COLUMN, rows 0..3: what each observation token spent on
        # the goal. The goal's own row is left out — a token attending to itself
        # is not the model consulting its destination.
        per_head = shown[:, :GOAL_COLUMN, GOAL_COLUMN].mean(axis=1)
        per_layer = [
            float(pass_.layer(i)[:, :GOAL_COLUMN, GOAL_COLUMN].mean())
            for i in range(pass_.n_layers)
        ]
        # Reading DOWN a column: how much each picture was looked at, totalled
        # over the four observation rows. Rows are the normalised unit (each
        # sums to 1), but the column is the more useful read for this figure —
        # it is what "spends 48% of its looking on the goal" measures, and it is
        # how the heads' specialisation becomes visible.
        columns = shown[:, :GOAL_COLUMN, :].mean(axis=1)
        facts[name] = {
            "goal_share":           per_layer[LAYER_SHOWN],
            "goal_share_per_head":  [float(v) for v in per_head],
            "strongest_head":       int(per_head.argmax()) + 1,
            "strongest_head_share": float(per_head.max()),
            "heads_using_goal":     int((per_head >= 0.10).sum()),
            "per_layer_goal_share": per_layer,
            "favourites": ", ".join(
                f"way {i + 1} → {labels[int(col.argmax())]}"
                for i, col in enumerate(columns)
            ),
        }

    # An invariant, not a fact. Each row of an attention matrix is a softmax, so
    # it sums to 1 by construction; if one does not, the hooks captured
    # something that is not attention and every number above is meaningless.
    # This was previously written to facts.json and checked by nobody.
    for name, pass_ in passes.items():
        if not np.allclose(pass_.attention.sum(axis=-1), 1.0, atol=1e-4):
            raise RuntimeError(
                f"{name}: attention rows do not sum to 1. The captured tensor "
                f"is not a softmax — check the hooks in attention.py."
            )

    facts["layer_shown"] = passes[MODES[0][0]].n_layers + LAYER_SHOWN + 1
    return facts


def measure_distance(model, encodings: dict, device: str) -> dict:
    """
    What the distance head makes of each mode's c_t.

    Included because the c_t figure provokes exactly one question — the two
    vectors are 99% alike, so does the goal matter at all? — and the page should
    answer it with a measurement rather than a promise about a later phase.
    """
    steps = {name: denoise.distance_to_goal(model, encodings[name].context, device)
             for name, *_ in MODES}
    (nav_name, *_), (exp_name, *_) = MODES
    return {"steps_nav": steps[nav_name], "steps_explore": steps[exp_name]}


def measure_context(encodings: dict, passes: dict) -> dict:
    """
    How far masking moves c_t, and whether the instrumented pass can be trusted.

    `agreement` is the fidelity check: the attention figure comes from a pass
    with torch's fused attention kernel disabled, which is the only way to see
    the weights at all, and that changes the arithmetic in the last decimal
    place. Comparing its c_t against the model's own leaves a number on record
    every run instead of a promise in a docstring.
    """
    (nav_name, *_), (exp_name, *_) = MODES
    nav = encodings[nav_name].context[0]
    explore = encodings[exp_name].context[0]
    gap = nav - explore
    agreement = max(
        float(np.abs(passes[name].context - encodings[name].context[0]).max())
        for name, *_ in MODES
    )

    return {
        "dim":                 int(nav.size),
        "pooled_from":         settings.N_TOKENS,
        "pooled_from_explore": settings.N_TOKENS - 1,
        "mean_abs_diff":       float(np.abs(gap).mean()),
        "max_abs_diff":        float(np.abs(gap).max()),
        "cosine":              measure.cosine(nav, explore),
        # Cosine alone reads as "basically identical" — it only measures
        # direction. The relative length of the gap is the honest companion to
        # it, and the two together are what the page quotes.
        "relative_change":     float(np.linalg.norm(gap) / np.linalg.norm(nav)),
        "value_min":           float(min(nav.min(), explore.min())),
        "value_max":           float(max(nav.max(), explore.max())),
        "dims_moved_1pct":     int((np.abs(gap) > 0.01 * np.abs(nav).max()).sum()),
        "agreement":           agreement,
    }


def main() -> None:
    cfg = load_config(PHASE_DIR)
    print(f"\n  P3 — Transformer & goal masking\n  {cfg.summary()}\n  {cfg.description}\n")

    sample = load_sample(cfg)
    model, info = load_model(cfg)

    passes, encodings = run_both_modes(model, sample, info["device"])
    attention_facts = measure_attention(passes, sample.token_labels)
    context_facts = measure_context(encodings, passes)
    distance_facts = measure_distance(model, encodings, info["device"])

    # One line, not twenty. Every value printed here is in facts.json a moment
    # later; the banner exists to confirm the run happened and that masking did
    # what it should, not to restate the record.
    print(f"  attention {passes[MODES[0][0]].attention.shape}  "
          f"goal share {attention_facts[MODES[0][0]]['goal_share']:.0%} -> "
          f"{attention_facts[MODES[1][0]]['goal_share']:.0%} masked  ·  "
          f"c_t moves {context_facts['relative_change']:.0%}")

    attention_fig = plot_attention(passes, sample.token_labels,
                                   cfg.out_dir / ATTENTION_FIGURE_NAME)
    context_fig = plot_context_vector(encodings, cfg.out_dir / CONTEXT_FIGURE_NAME)

    write_facts(
        cfg,
        {
            "model":       info,
            "transformer": attention.geometry(model),
            "goal":        {
                "frames_ahead": settings.NUM_ACTIONS,
                # Frames, not seconds. See settings.METRIC_WAYPOINT_SPACING.
                "metres_ahead": settings.NUM_ACTIONS * settings.METRIC_WAYPOINT_SPACING,
            },
            "attention":   attention_facts,
            "context":     context_facts,
            "distance":    distance_facts,
        },
        figures=[attention_fig, context_fig],
    )


if __name__ == "__main__":
    main()
