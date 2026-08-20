# deeper_visuals/common/facts.py
"""
facts.json — the handoff between the two halves of a phase.

run_model.py (slow, needs the GPU) writes facts.json + the PNGs.
build_page.py (instant, never loads the model) reads only those two things.

Keeping that boundary strict is what makes `update.sh --page-only` possible:
wording and layout can be iterated all day without touching a GPU, and every
number on the published page provably came from a real forward pass rather than
from something hand-typed into the HTML.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FACTS_NAME = "facts.json"


def _jsonable(obj: Any) -> Any:
    """
    Make numpy scalars/arrays and Paths survive json.dump.

    numpy is imported inside the function, not at module scope, because only
    the writing half of this module needs it. read_facts is what build_page.py
    calls, and build_page.py is the half that is supposed to run anywhere —
    a module-level `import numpy` put the whole scientific stack behind
    `update.sh --page-only` for no reason.
    """
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def write_facts(cfg, payload: dict, figures: list[str]) -> Path:
    """
    Write cfg.out_dir/facts.json.

    The stored record is always the phase-specific `payload` plus a provenance
    block, so any page can state which scene and which weight file produced it
    without the phase having to remember to include that itself.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "phase":     cfg.phase,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scene": {
            "sample":      cfg.sample_key,
            "traj":        cfg.traj,
            "frame":       cfg.frame,
            "description": cfg.description,
            "obs_idxs":    cfg.obs_idxs,
            "goal_idx":    cfg.goal_idx,
        },
        "figures": figures,
        **payload,
    }
    path = cfg.out_dir / FACTS_NAME
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=_jsonable)
    print(f"  Saved : {path}")
    return path


def read_facts(out_dir: Path) -> dict:
    path = Path(out_dir) / FACTS_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no {FACTS_NAME} in {out_dir} — run update.sh without --page-only "
            f"at least once to generate it."
        )
    with open(path) as fh:
        return json.load(fh)
