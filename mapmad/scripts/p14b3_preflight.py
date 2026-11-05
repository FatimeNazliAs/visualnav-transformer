"""Phase 14b-3 pre-flight (chain step 0; p14b3_rules.md §9). Habitat side (or anywhere with numpy):

    python mapmad/scripts/p14b3_preflight.py --config <p14b3_v3.yaml> --rules <p14b3_rules.md>

Every checkpoint sha256 and set fingerprint in the rules file's ```json preflight``` block (paths below the outputs
root) is recomputed; any mismatch -> exit 3 (automatic stop: checkpoint sha256 mismatch). The round-2 rollout set
and exam B must have unique episode ids (exit 3 otherwise).
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.run_layout import RunLayout  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight_block(rules: Path) -> dict:
    m = re.search(r"```json preflight\n(.*?)\n```", rules.read_text(), re.S)
    if not m:
        raise SystemExit(f"{rules}: no ```json preflight``` block")
    return json.loads(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--rules", type=Path, required=True)
    a = p.parse_args()
    outputs = RunLayout.load(a.config).outputs
    block = preflight_block(a.rules)
    bad = []
    for rel, want in block["checkpoints"].items():
        got = sha256(outputs / rel)
        if got != want:
            bad.append(f"checkpoint {rel}: {got[:12]} != {want[:12]}")
    ids_ok = {}
    for rel, want in block["sets"].items():
        eps, fp = load_episodes(outputs / rel)
        if fp != want:
            bad.append(f"set {rel}: fingerprint {fp[:12]} != {want[:12]}")
        ids = [e["episode_id"] for e in eps]
        ids_ok[rel] = len(ids) == len(set(ids))
        if not ids_ok[rel]:
            bad.append(f"set {rel}: duplicate episode ids")
    print(json.dumps({"checkpoints": len(block["checkpoints"]), "sets": len(block["sets"]), "unique_ids": ids_ok,
                      "problems": bad}))
    if bad:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
