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

# The keys build_page.render_provenance reads with a bare subscript, checked
# here so a phase that forgets one fails while writing rather than while
# publishing. P1 writes cfg.weights_provenance() (3 keys) and P2/P3 write
# load_model's info (8); this is the subset every phase must agree on, and the
# reason the two shapes can coexist without the footer quietly printing "n/a".
REQUIRED_MODEL_KEYS = ("checkpoint_file", "run")


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


def build_record(cfg, payload: dict, figures) -> dict:
    """
    Assemble the record without writing it.

    Split from write_facts so the seam's *shape* can be asserted without
    producing a file — previously there was no way to obtain the record at all
    except by writing one, which put the only structural guarantee in the
    pipeline out of reach of any test. `deeper_visuals/tests/` now does.

    `figures` accepts the paths the plot functions return as well as bare
    names, and stores the names. Each figure's filename used to be written out
    three times per phase — a constant, this argument, and the literal in
    page.yaml — so passing the path that was actually saved removes one of the
    two ways they could disagree.
    """
    missing = [k for k in REQUIRED_MODEL_KEYS
               if k not in payload.get("model", {})]
    if missing:
        raise KeyError(
            f"facts payload is missing model.{', model.'.join(missing)} — "
            f"the page footer reads these to say which weights produced it. "
            f"Pass cfg.weights_provenance() or load_model's info as 'model'."
        )
    return {
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
        "figures": [Path(f).name for f in figures],
        **payload,
    }


def write_facts(cfg, payload: dict, figures) -> Path:
    """
    Write cfg.out_dir/facts.json.

    The stored record is always the phase-specific `payload` plus a provenance
    block, so any page can state which scene and which weight file produced it
    without the phase having to remember to include that itself.

    Numbers go in as numbers. How they read on the page is decided at the page
    seam by build_page.fill's format specs, so facts.json keeps what was
    measured rather than a rendering of it.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    record = build_record(cfg, payload, figures)
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
