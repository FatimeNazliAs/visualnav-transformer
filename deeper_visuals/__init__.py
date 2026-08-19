# deeper_visuals/__init__.py
"""
Package init for deeper_visuals.

Puts the repo root and train/ onto sys.path so that, from anywhere:
  * `from deeper_visuals... import ...` resolves, and
  * `from vint_train... import ...` / `from diffusion_policy... import ...`
    resolve without vint_train being pip-installed.

Doing it here — once, at package import — means every entry point (a phase's
run_model.py, common/build_page.py, common/smoke_test.py) gets the paths set
before any submodule body runs.

This folder is a self-contained copy-and-adapt of the frozen debug_visuals/
package. It deliberately does NOT import from debug_visuals — that folder
backs an already-delivered presentation and must keep working untouched.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
