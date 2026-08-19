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
