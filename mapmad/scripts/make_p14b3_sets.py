"""Phase 14b-3 sets (kickoff §3 item 1, gate row 6; p14b3_rules.md §2). Habitat side or host, pure Python:

    python mapmad/scripts/make_p14b3_sets.py rollout --outputs <mapmad outputs>      # freeze the round-2 start set
    python mapmad/scripts/make_p14b3_sets.py contamination --outputs <mapmad outputs> [--drives <folder or traj_names.txt> ...]

`rollout`: Phase 11's rollout start set (fingerprint 8e3428c7..., 1,448 episodes) WITHOUT the episodes in the 10
photo-test homes (= the set's 10 dev / test-split homes; gate row 6 forbids any round-2 rollout there) ->
<outputs>/p14b3_v3/rollouts_r2/set/episodes.json (+ .sha256, own fingerprint, Phase 11 ids unchanged, so the drives
pair with round 1 by id) and checks/contamination.json. The exclusion lists are the frozen 14b-2 lists
(p14b2_recog/trainset/homes.json: photo test 10, HM3D val 100, MP3D 90 homes), their sha256 re-checked.
`contamination`: rollout homes and the homes of every drive in the given folders (round-2 drives, all training data
folders) against the three lists, by full id and by scene id; exit 3 on any overlap (automatic stop).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROLL_SET = "p11_dagger/rollouts/set/episodes.json"
ROLL_FP = "8e3428c74454f7a6f0a6021694c0e0eb8565c3c646d3ec5780b83930e187aa40"
HOMES = "p14b2_recog/trainset/homes.json"
OUT = "p14b3_v3/rollouts_r2/set"
CHECKS = "p14b3_v3/checks"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mapmad_sim.episodes import fingerprint, load_episodes  # noqa: E402
from mapmad_sim.home_splits import list_sha256  # noqa: E402


def scene_of(home: str) -> str:
    """'00081-5biL7VEkByM' -> '5biL7VEkByM'; an MP3D scene id stays itself."""
    return home.split("-", 1)[1] if "-" in home else home


def exclusions(outputs: Path) -> Dict[str, List[str]]:
    """The frozen 14b-2 lists, sha256 re-checked."""
    doc = json.loads((outputs / HOMES).read_text())
    out = {}
    for k, v in doc["exclusions"].items():
        if list_sha256(v["homes"]) != v["sha256"]:
            raise SystemExit(f"{HOMES}: list {k} changed (sha256)")
        out[k] = list(v["homes"])
    if len(out.get("photo_test", [])) != 10 or not out.get("hm3d_val") or not out.get("mp3d"):
        raise SystemExit(f"exclusion lists incomplete: { {k: len(v) for k, v in out.items()} }")
    return out


def overlap(homes: Sequence[str], excl: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """homes ∩ each list, by full id and by scene id (14b-2 rule)."""
    hs, scenes = set(homes), {scene_of(h) for h in homes}
    return {k: sorted(hs & set(v) | {x for x in v if scene_of(x) in scenes}) for k, v in excl.items()}


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, path)


def cmd_rollout(outputs: Path) -> None:
    eps, fp = load_episodes(outputs / ROLL_SET)
    if fp != ROLL_FP:
        raise SystemExit(f"{ROLL_SET}: fingerprint {fp} != {ROLL_FP}")
    excl = exclusions(outputs)
    photo = set(excl["photo_test"])
    keep = [e for e in eps if e["home"] not in photo]
    dropped = sorted({e["home"] for e in eps if e["home"] in photo})
    homes = sorted({e["home"] for e in keep})
    ov = overlap(homes, excl)
    out = outputs / OUT
    doc = {"fingerprint": fingerprint(keep), "episodes": keep,
           "source": {"file": ROLL_SET, "fingerprint": fp, "episodes": len(eps)},
           "rule": "Phase 11 rollout set minus the 10 photo-test homes (gate row 6)"}
    write_json(out / "episodes.json", doc)
    (out / "episodes.sha256").write_text(hashlib.sha256((out / "episodes.json").read_bytes()).hexdigest() + "  episodes.json\n")
    rep = {"rollout_set": {"episodes": len(keep), "homes": len(homes), "fingerprint": doc["fingerprint"],
                           "dropped_photo_test_homes": dropped, "dropped_episodes": len(eps) - len(keep),
                           "split_roles": sorted({e["split_role"] for e in keep})},
           "overlap": ov, "contamination_free": not any(ov.values()),
           "lists": {k: {"homes": len(v), "sha256": list_sha256(v)} for k, v in excl.items()}}
    write_json(outputs / CHECKS / "contamination_rollout_set.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "lists"}, indent=1))
    if not rep["contamination_free"]:
        raise SystemExit(3)


def drive_homes(folder: Path) -> List[str]:
    """Homes of the drive folders below `folder` ('p11r-<home5>-...' / 'p8d-<home5>-<scene>-...' ids -> the
    meta's home when present, else the id's second field), or of the drives a split list (traj_names.txt) names."""
    homes = set()
    if folder.is_file():  # a split list (traj_names.txt): the drives a training run reads
        for name in folder.read_text().split():
            parts = name.split("-")
            if len(parts) > 2:
                homes.add(f"{parts[1]}-{parts[2]}" if parts[0] == "p8d" else parts[1])
        return sorted(homes)
    for d in folder.iterdir():
        meta = d / "mapmad_meta.json"
        if meta.exists():
            h = json.loads(meta.read_text()).get("home")
            if h:
                homes.add(h)
                continue
        parts = d.name.split("-")
        if len(parts) > 2:
            homes.add(f"{parts[1]}-{parts[2]}" if parts[0] == "p8d" else parts[1])
    return sorted(homes)


def cmd_contamination(outputs: Path, folders: Sequence[Path], out_dir: Path) -> None:
    excl = exclusions(outputs)
    eps, fp = load_episodes(outputs / OUT / "episodes.json")
    rows = {"rollout_set": {"fingerprint": fp, "homes": sorted({e["home"] for e in eps})}}
    for f in folders:
        rows[str(f)] = {"homes": drive_homes(Path(f))}
    rep = {}
    for k, v in rows.items():
        ov = overlap(v["homes"], excl)
        # drive ids carry only the 5-digit home prefix: compare those too
        pref = {h.split("-")[0] for h in v["homes"]}
        ov_pref = {name: sorted(x for x in lst if x.split("-")[0] in pref and "-" in x) for name, lst in excl.items()
                   if name != "mp3d"}
        rep[k] = {"homes": len(v["homes"]), "overlap": ov, "overlap_by_prefix": ov_pref,
                  "clean": not any(ov.values()) and not any(ov_pref.values())}
    doc = {"rows": rep, "contamination_free": all(r["clean"] for r in rep.values())}
    write_json(Path(out_dir) / "checks" / "contamination.json", doc)
    print(json.dumps(doc, indent=1)[:3000])
    if not doc["contamination_free"]:
        raise SystemExit(3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["rollout", "contamination"])
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--drives", type=Path, nargs="*", default=[])
    p.add_argument("--out-dir", type=Path, default=None, help="contamination: the phase folder (default <outputs>/p14b3_v3)")
    a = p.parse_args()
    if a.cmd == "rollout":
        cmd_rollout(a.outputs)
    else:
        cmd_contamination(a.outputs, a.drives, a.out_dir or a.outputs / "p14b3_v3")


if __name__ == "__main__":
    main()
