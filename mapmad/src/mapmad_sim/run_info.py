"""Save what a run needs to be repeated (config, seed, git commit) next to its outputs (plan §0.8)."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[3]


def git_state() -> Dict[str, Any]:
    """Current commit of the MapMaD worktree and whether tracked files have uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": "unknown", "error": str(exc)}


def save_run_info(out_dir: Path, config: Dict[str, Any], seed: int, name: str = "run_info.json") -> Path:
    """Write config + seed + git state + command line + time to <out_dir>/<name>; returns the file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "python": platform.python_version(),
        "seed": seed,
        "git": git_state(),
        "config": config,
    }
    path = out_dir / name
    path.write_text(json.dumps(info, indent=2, default=str) + "\n")
    return path
