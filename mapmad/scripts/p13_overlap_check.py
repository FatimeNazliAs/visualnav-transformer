"""Phase 13 (13.0): do MP3D and HM3D share a home? Compares scene ids and sha256 of the mesh files (MP3D *.glb vs
every HM3D v0.2 *.glb of all splits). Host or container, CPU only, read-only on both datasets.

    python3 mapmad/scripts/p13_overlap_check.py <mp3d scenes> <hm3d scenes> <out.json>
"""

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    mp3d, hm3d, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    mp = {p.parent.name: p for p in sorted(mp3d.glob("*/*.glb"))}
    hm = sorted(hm3d.glob("*/*/*.glb"))
    hm_ids = {p.parent.name.split("-", 1)[-1] for p in hm}
    mp_hash = {sha256(p): k for k, p in mp.items()}
    same_hash = [{"mp3d": mp_hash[h], "hm3d": str(p)} for p in hm if (h := sha256(p)) in mp_hash]
    rec = {"mp3d_homes": len(mp), "hm3d_glb_files": len(hm), "shared_ids": sorted(set(mp) & hm_ids),
           "shared_mesh_sha256": same_hash}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in rec.items()}))


if __name__ == "__main__":
    main()
