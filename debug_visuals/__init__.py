# debug_visuals/__init__.py
"""
Package init for debug_visuals.

Puts the repo root and train/ onto sys.path so that, from anywhere:
  * `from debug_visuals... import ...` resolves, and
  * `from vint_train... import ...` / `from diffusion_policy... import ...`
    resolve without vint_train being pip-installed.

Doing it here — once, at package import — means every entry point (whether
`python -m debug_visuals.run_pipeline` or `python -m debug_visuals.visualize_stageN`)
gets the paths set before any submodule body runs. The individual stage
modules therefore no longer each repeat this bootstrap.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
