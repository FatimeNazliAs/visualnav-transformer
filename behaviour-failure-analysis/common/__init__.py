# behaviour-failure-analysis/common/__init__.py
"""
Package init for the shared behaviour-analysis library.

Puts the repo root and train/ onto sys.path so that `from vint_train… import …`
and `from diffusion_policy… import …` resolve. The container's image installs
vint_train editable against /app/visualnav-transformer, so this is belt and
braces — but it also makes the package work from a plain checkout, which is what
lets the pure-numpy modules (actions, measure) be exercised without the image.

Doing it here, once at package import, means every entry point gets the paths
set before any submodule body runs.

Note on importing this package: `behaviour-failure-analysis` and
`01-goal-conditioned-behaviour` contain hyphens, so they are not importable
Python identifiers. That is fine — neither is ever imported. Entry-point scripts
put `behaviour-failure-analysis/` itself on sys.path and import `common`
directly:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import config, data, model

This folder is a copy-and-adapt of deeper_visuals/common/ on branch
feature/deeper-visualization. It deliberately does NOT import from it — that
branch backs a delivered presentation and is not even checked out in this
worktree.
"""

import sys
from pathlib import Path

# common/ -> behaviour-failure-analysis/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
