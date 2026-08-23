# deeper_visuals/common/viz.py
"""
Shared plotting helpers for the deeper_visuals phases.

Adapted from debug_visuals/viz_utils.py. Matplotlib is forced to the Agg
backend here — the phases run headless inside the container, and importing
pyplot without this raises before any figure is drawn.
"""

from __future__ import annotations

from pathlib import Path

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

DPI = 150

# The house style for a framed panel. Stated once here rather than at each call
# site, which is what stops fontsize 9 / 10 / 11 appearing in three phases that
# are meant to look like one series.
TITLE_SIZE = 10
TITLE_PAD = 6
LABEL_SIZE = 9


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
