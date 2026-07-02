# debug_visuals/viz_utils.py
"""
Small shared plotting helpers for the debug_visuals stages.

Kept separate from config.py (which stays matplotlib-free) and from the
stage modules (which focus on the figures themselves). For now this is
just save_fig() — the identical mkdir + savefig + close + log block that
every stage repeated at the end of each plot function.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_fig(fig, save_path: Path, *, facecolor: str | None = None) -> None:
    """
    Write `fig` to `save_path` (creating parent dirs), close it, and log.

    All stages save at dpi=150 with bbox_inches="tight"; only the Stage 0
    architecture diagram additionally forces a white facecolor, so that's
    the single optional knob.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": 150, "bbox_inches": "tight"}
    if facecolor is not None:
        save_kwargs["facecolor"] = facecolor
    fig.savefig(save_path, **save_kwargs)
    plt.close(fig)
    print(f"  Saved : {save_path}")
