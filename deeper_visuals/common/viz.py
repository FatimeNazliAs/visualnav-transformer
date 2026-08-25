# deeper_visuals/common/viz.py
"""
Shared plotting helpers for the deeper_visuals phases.

Adapted from debug_visuals/viz_utils.py. Matplotlib is forced to the Agg
backend here — the phases run headless inside the container, and importing
pyplot without this raises before any figure is drawn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")   # headless — no display inside the container
import matplotlib.pyplot as plt   # noqa: E402  — must follow matplotlib.use

# One consistent accent palette across all six phases, so a colour means the
# same thing on every page: orange = the current frame, purple = the goal.
COLOR_CURRENT = "#f39c12"
COLOR_GOAL    = "#8e44ad"
COLOR_NAV     = "#2980b9"
COLOR_EXPLORE = "#16a085"
COLOR_MUTED   = "#7f8c8d"
# What the robot actually did, where a figure has it to compare against. Magenta
# because that is what NoMaD's own training visualisations use for the ground
# truth (train_utils.visualize_diffusion_action_distribution), and a reader who
# has seen those should not have to learn a second convention.
COLOR_TRUTH   = "#c2247e"

DPI = 150

# The house style for a framed panel. Stated once here rather than at each call
# site, which is what stops fontsize 9 / 10 / 11 appearing in three phases that
# are meant to look like one series.
TITLE_SIZE = 10
TITLE_PAD = 6
LABEL_SIZE = 9

# How heavy a plate's border is. "The border is the label", says plate() below —
# and the weight is half of that label: a heavy border means this panel is the
# subject, a light one means it is context. Three phases picked 2.6 for the
# emphasis and one picked 2.2, which is the sort of difference nobody notices in
# one figure and everybody feels across five.
PLATE_EMPHASIS = 2.6
PLATE_PLAIN = 1.0


def plate(ax, *, edge: str = COLOR_MUTED, width: float = 1.0,
          title: str | None = None, subtitle: str | None = None,
          bold: bool = False) -> None:
    """
    Strip an axes to its content and frame it in one of the phase colours.

    The caller draws whatever it likes into `ax` — a photograph, a token strip,
    a photograph with a heat map over it — and this decides what a panel of the
    deeper_visuals series looks like around it: no ticks, a coloured border, a
    title in the border's colour, an optional grey line underneath.

    The border is the label. Orange means the current frame, purple means the
    goal, grey means neither, in every figure of every phase — which only holds
    while one place decides it.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(edge)
        spine.set_linewidth(width)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, pad=TITLE_PAD, color=edge,
                     fontweight="bold" if bold else "normal")
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=LABEL_SIZE, color=COLOR_MUTED, labelpad=6)


def worded_key(fig, mappable, axes: list, labels: list[str], *,
               caption: str | None = None, drop: float = 0.06) -> None:
    """
    A slim horizontal colour key beneath `axes`, labelled in words not numbers.

    The caller says what the ends of the scale *mean*; the geometry and the tick
    positions are this function's problem. Ticks come from the mappable's own
    limits, so they cannot disagree with the colours actually drawn: two labels
    put them at the extremes, three put the middle one at zero — which is the
    only interior tick a diverging scale has any business showing.

    Words rather than numbers because every scale in this series is either
    per-encoder or a cosine distance, and neither reads as a quantity to the
    audience these pages are for. The numbers live in facts.json.

    Call it after the layout has settled; it reads laid-out axes positions.
    """
    low, high = mappable.get_clim()
    ticks = [low, high] if len(labels) == 2 else [low, 0, high]
    if len(ticks) != len(labels):
        raise ValueError(f"worded_key takes 2 or 3 labels, got {len(labels)}")

    first, last = axes[0].get_position(), axes[-1].get_position()
    cax = fig.add_axes([first.x0, first.y0 - drop, last.x1 - first.x0, 0.02])
    bar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    bar.ax.set_xticklabels(labels)
    bar.ax.tick_params(labelsize=8.5, colors=COLOR_MUTED, length=0)
    bar.outline.set_edgecolor(COLOR_MUTED)
    bar.outline.set_linewidth(0.6)
    if caption:
        bar.set_label(caption, fontsize=LABEL_SIZE, color=COLOR_MUTED)


def settle(fig) -> None:
    """
    Lay the figure out, so laid-out positions can be read from it.

    Call this once before any annotation that reads `get_position()`. imshow
    pins each axes to the image aspect, so until the canvas has been laid out
    get_position() still reports the cell the gridspec allotted, not the smaller
    box the image was fitted into — annotating from that box leaves labels and
    rules floating in the slack.

    One line, but it earns a name: the rationale above was written out five
    times across four phase files before this existed, once per call site, and
    a hazard explained five times is a module that has not been created yet.
    """
    fig.canvas.draw()


def band(axes: list) -> tuple[float, float, float, float]:
    """
    The figure-fraction box a run of panels actually occupies.

    Returns (left, right, bottom, top). Call after `settle`.

    Every figure in the series annotates groups of panels — a divider between
    two runs, a caption under one, a colour key spanning several — and each
    needs the same rectangle. Four call sites computed it independently before
    this was promoted out of P1.
    """
    boxes = [ax.get_position() for ax in axes]
    return (min(b.x0 for b in boxes), max(b.x1 for b in boxes),
            min(b.y0 for b in boxes), max(b.y1 for b in boxes))


def occupied(fig, axes: list) -> tuple[float, float, float, float]:
    """
    The figure-fraction box a run of panels occupies once DRAWN — ticks, tick
    labels and axis labels included. Returns (left, right, bottom, top).

    `band` above measures the axes rectangles. That is the right answer when the
    panels have had their ticks stripped, which is most of this series — and the
    wrong one for a row that keeps them. A caption placed a fixed distance from
    the axes box lands clear of one row and on top of another's tick labels,
    which is precisely what happened when P3's rows joined the shared row
    labeller. Both measurements are worth having; they differ only for panels
    with decorations, and it is exactly those panels that need this one.

    Call after `settle` — it reads the renderer.
    """
    renderer = fig.canvas.get_renderer()
    to_fraction = fig.transFigure.inverted()
    boxes = [to_fraction.transform_bbox(ax.get_tightbbox(renderer))
             for ax in axes]
    return (min(b.x0 for b in boxes), max(b.x1 for b in boxes),
            min(b.y0 for b in boxes), max(b.y1 for b in boxes))


def vector_strip(ax, values, *, vabs: float | None = None, cmap: str = "RdBu_r",
                 vmin: float | None = None, vmax: float | None = None):
    """
    One vector as a single-row heatmap, and the reason it is drawn that way.

    Rendered as 1 x N rather than reshaped into a square: the dimensions of
    these vectors have no order and no neighbours, so a 16 x 16 tile would
    invent a structure the vector does not have and invite the reader to look
    for patches in it. P2 draws tokens this way, P3 draws c_t this way, and P4's
    noise vectors and P5's action sequences will want the same thing.

    `vabs` is the common case — a symmetric diverging scale centred on zero.
    Pass vmin/vmax instead for a sequential one.
    """
    if vabs is not None:
        vmin, vmax = -vabs, vabs
    return ax.imshow(np.asarray(values).reshape(1, -1), aspect="auto",
                     cmap=cmap, vmin=vmin, vmax=vmax)


def save_fig(fig, save_path: Path, *, facecolor: str | None = "white") -> Path:
    """
    Write `fig` to `save_path` (creating parent dirs), close it, and log.

    A white facecolor is the default rather than an opt-in: these PNGs get
    embedded in a page that renders in both light and dark themes, and a
    transparent figure turns into unreadable dark-on-dark in the dark one.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": DPI, "bbox_inches": "tight"}
    if facecolor is not None:
        kwargs["facecolor"] = facecolor
    fig.savefig(save_path, **kwargs)
    plt.close(fig)
    print(f"  Saved : {save_path}")
    return save_path
