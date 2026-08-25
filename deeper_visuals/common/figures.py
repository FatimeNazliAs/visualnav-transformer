# deeper_visuals/common/figures.py
"""
The series' figure language for paths and panel rows.

`viz.py` holds the *look* — the palette, the plate, where a laid-out axes ended
up. It has no home for anything path-shaped or row-shaped, which is the majority
of what every phase actually draws, so each phase re-derived it and the copies
drifted: three variants of "draw a rule between two axes", two row-labellers with
different maths placing their labels 22px apart, and two bounding-box routines
inside one file. `viz.vector_strip` even predicted this module in writing —
"P4's noise vectors and P5's action sequences will want the same thing" — and
then shipped only the 1-D case.

This is that module, promoted out of P4 the same way `viz.band` was promoted out
of P1 after four call sites computed it independently. Three ideas:

    path     one route from the robot outward, forward-up
    box      one square data-space box shared by every panel, so a shape's size
             means the same thing everywhere
    row      a strip of panels across a ramp, named on the left, numbered below

P5 is specified as predicted paths, several sampled trajectories and a top-down
view — the same three ideas, which is why they are here rather than in P4.

Nothing in this module knows what a descent is. It takes arrays of metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from deeper_visuals.common import viz


def ramp(fraction: float, colour: str | None = None) -> tuple:
    """Muted grey at `fraction` 0, the given colour at 1."""
    start = np.array(mcolors.to_rgb(viz.COLOR_MUTED))
    end = np.array(mcolors.to_rgb(colour or viz.COLOR_NAV))
    return tuple(start + (end - start) * fraction)


def shared_box(paths: np.ndarray) -> tuple[float, float, float, float]:
    """
    The one box every panel of a figure is drawn in — (x0, x1, y0, y1).

    Fitted to all the paths at once, so a shape's size and position mean the
    same thing in every panel. Two details earn their arithmetic:

    * Square, because the panels are drawn at equal aspect. A box fitted per
      axis leaves the geometry undistorted but the empty space around it
      misleading — the same path would sit differently in a wide panel than in
      a tall one.
    * Centred on the paths rather than on the robot. A set of states can run
      well behind the robot as well as ahead of it, and squaring a box about
      the origin to hold that spends much of its area on floor nothing visits.

    Note this is a **data-space** box, unlike `viz.band`, which returns a
    figure-fraction one. Both are 4-tuples in the same order, so they are easy
    to confuse and impossible to catch by type — hence the different name.
    """
    flat = paths.reshape(-1, paths.shape[-1])
    low = np.minimum(flat.min(axis=0), 0.0)
    high = np.maximum(flat.max(axis=0), 0.0)
    centre = (low + high) / 2
    half = float((high - low).max()) / 2 * 1.10
    return (centre[1] - half, centre[1] + half,
            centre[0] - half, centre[0] + half)


def frame(ax, box: tuple) -> None:
    """Put an axes in a shared box at equal aspect."""
    x0, x1, y0, y1 = box
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")


def draw_path(ax, path: np.ndarray, colour, *, width: float, alpha: float = 1.0,
              dot: float = 2.6):
    """
    One path, from the robot outward.

    Plotted with the lateral axis horizontal and the forward axis vertical, so
    "ahead" is up — the orientation P5's top-down view uses as well. The (0, 0)
    the robot occupies is prepended, because the first waypoint is a step away
    from it and a path that starts at its first waypoint is missing its first
    step.

    `path` is metres in the robot's frame: column 0 ahead, column 1 left.
    """
    forward = np.concatenate([[0.0], path[:, 0]])
    left = np.concatenate([[0.0], path[:, 1]])
    return ax.plot(left, forward, color=colour, linewidth=width, alpha=alpha,
                   marker="o", markersize=dot, solid_capstyle="round",
                   markeredgecolor="white", markeredgewidth=0.6 if dot > 4 else 0,
                   zorder=3 if alpha == 1.0 else 2)


# What a frame of the scene looks like, by what it IS. Before this, five call
# sites across three files each decided the colour and the border weight from an
# `is_current` flag, and one of them chose 2.2 where the others chose 2.6.
# Naming the roles rather than the widths is what keeps a sixth call site from
# inventing a seventh convention.
FRAME_STYLES = {
    "now":  (viz.COLOR_CURRENT, viz.PLATE_EMPHASIS),
    "past": (viz.COLOR_MUTED,   viz.PLATE_PLAIN),
    "goal": (viz.COLOR_GOAL,    viz.PLATE_EMPHASIS),
}


def scene_frame(ax, image, *, kind: str, title: str | None = None,
                subtitle: str | None = None, bold: bool = False):
    """
    Draw one frame of the scene, framed the way this series frames that kind.

    `kind` is what the frame is — "now", "past" or "goal" — not what it should
    look like. That is the whole point: the caller knows which frame it is
    holding and nothing else, so the colour and the weight cannot drift between
    a phase that shows the frame strip and a phase that shows it again beside a
    heat map.
    """
    if kind not in FRAME_STYLES:
        raise ValueError(
            f"unknown frame kind {kind!r} — one of {', '.join(FRAME_STYLES)}")
    edge, width = FRAME_STYLES[kind]
    ax.imshow(image)
    viz.plate(ax, edge=edge, width=width, title=title, subtitle=subtitle,
              bold=bold)
    return ax


@dataclass(frozen=True, eq=False)
class Row:
    """One strip of panels: its name, its states, its colour, which way it runs."""

    title: str
    subtitle: str
    paths: np.ndarray
    colour: str
    # The last panel's title, where the row builds toward one. None when the
    # columns are not a progression and no panel is the destination.
    end_title: str | None = None
    # "→" or "←", or None when the columns are not a process. P4's columns are
    # noise levels, so they run in a direction and the arrow says which. P5's
    # are independent samples — they have no order at all, and an arrow drawn
    # under them would assert one. A row that is not going anywhere says so by
    # leaving this off rather than by picking a harmless-looking direction.
    arrow: str | None = None


def draw_row(axes: list, row: Row, box: tuple) -> None:
    """
    One process across the noise levels, cleanest at the right.

    The ramp runs grey at the buried end to the row's own colour at the clean
    end, so two rows stay distinguishable at a glance while agreeing on what
    grey means. Every panel is otherwise identical — same box, same line, same
    markers — so each difference is the data's.
    """
    for index, (ax, path) in enumerate(zip(axes, row.paths)):
        fraction = index / (len(row.paths) - 1)
        is_clean = index == len(row.paths) - 1
        draw_path(ax, path, ramp(fraction, row.colour),
                  width=2.4 if is_clean else 1.5)
        frame(ax, box)
        viz.plate(ax, edge=row.colour if is_clean else viz.COLOR_MUTED,
                  width=1.6 if is_clean else 0.7,
                  title=row.end_title if is_clean else None, bold=True)


def label_row(fig, axes: list, row: Row, *, gap: float = 11.0,
              rotate: float = 0.0, fontsize: float | None = None,
              bold: bool = False, arrow_drop: float = 0.105) -> None:
    """
    The row's name beside it, and an arrow showing which way it runs.

    The label is placed against the row it labels — `gap` points clear of
    everything that row actually occupies — rather than at a figure fraction the
    caller supplies.

    "Occupies" rather than "the panels" on purpose: P3's rows carry tick labels
    outside their axes and P4's do not, so measuring the axes box alone puts the
    same gap in two different places. viz.occupied measures what was rendered.

    That is the fix for a coupling this function used to document and then hand
    back: the old `x` had to agree with whatever `subplots_adjust(left=…)` the
    caller had chosen, and nothing checked that it did. P4 paired left=0.135
    with x=0.008 and P5 paired left=0.155 with x=0.010 — neither wrong, neither
    connected, and the only detector a human comparing two PNGs. A parameter
    that must be kept in sync with another parameter is better deleted than
    documented, so the band decides and `x` is gone.

    The caller still chooses its own left margin; it just no longer has to tell
    this function about it. Too small a margin now clips the label instead of
    silently misplacing it, which is a failure you can see.
    """
    # Two measurements, because the label and the arrow answer to different
    # things. The label must clear whatever the row actually rendered, tick
    # labels included; the arrow says how the PANELS run and so is drawn to the
    # panels' own edges — measuring it against the decorations would stretch it
    # past the strip it describes.
    outer_left, _, _, _ = viz.occupied(fig, axes)
    left, right, bottom, top = viz.band(axes)
    fig.text(outer_left, (bottom + top) / 2, f"{row.title}\n{row.subtitle}",
             transform=offset_copy(fig.transFigure, fig=fig, x=-gap, y=0,
                                   units="points"),
             ha="center" if rotate else "right", va="center",
             rotation=rotate, fontsize=fontsize or viz.LABEL_SIZE,
             fontweight="bold" if bold else "normal",
             color=row.colour, linespacing=1.9 if bold else 1.8)

    if row.arrow is None:
        return

    y = bottom - arrow_drop
    ends = (right, left) if row.arrow == "←" else (left, right)
    fig.add_artist(Line2D(
        [ends[0], ends[1]], [y, y], color=viz.COLOR_MUTED, linewidth=0.9,
        marker=">" if row.arrow == "→" else "<", markevery=[1], markersize=6,
        transform=fig.transFigure,
    ))


def number_panels(fig, axes: list, *, first: str = "start",
                  labels: list[str] | None = None, drop: float = 0.045) -> None:
    """
    Caption a row's panels, so they can be counted off the figure.

    A count in a caption is a claim; a reader who can point at the sixth panel
    has checked it.

    Two captioning rules, because the series has two kinds of row. By default
    the columns are steps of one process, so they are numbered and the first
    panel is named by `first` rather than numbered — numbering a starting state
    would make the last panel one higher than the number of steps taken. Pass
    `labels` instead when the columns are not steps at all (P5's are separate
    runs) and the caption is the caller's word rather than an index.
    """
    if labels is not None and len(labels) != len(axes):
        raise ValueError(
            f"{len(labels)} labels for {len(axes)} panels — one each, in order")

    for index, ax in enumerate(axes):
        position = ax.get_position()
        text = labels[index] if labels is not None else (
            first if index == 0 else str(index))
        fig.text((position.x0 + position.x1) / 2, position.y0 - drop, text,
                 ha="center", va="center", fontsize=8.5,
                 color=viz.COLOR_MUTED)
