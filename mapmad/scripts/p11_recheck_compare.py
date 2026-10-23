"""Phase 11 §0.8 re-checks: compare the re-run outputs with the originals (run_p11_rechecks.sh, Habitat container).

    python mapmad/scripts/p11_recheck_compare.py --p11 /outputs/mapmad/p11_dagger --episode <dev id> \
        --drives /mapmad_data/habitat_mapmad_v2_dagger --redrives /mapmad_data/p11_recheck_dagger \
        --out /outputs/mapmad/p11_dagger/checks/recheck/rechecks.json <rollout id> ...

Checks (each {"pass": bool, ...}):
- g10_rows_1_2: rows 1-2 pass again and every pass/equal flag equals the original offline/v21_ema9 file (the v1
  map-hidden comparison is report only);
- rescore: every array of checks/recheck/search/scores.npz equal to offline/v21_ema9/search/scores.npz;
- determinism: the two re-runs of the dev episode (arm v21, seed 0, one GPU) byte-identical; the dev check's own log of
  that run (maybe another GPU) is compared too, report only;
- rollouts: per drive the same files; every .jpg byte-identical; every .npz array and traj_data.pkl equal (NaN = NaN);
  mapmad_meta.json equal except timing fields.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

TIMING = ("time", "seconds", "elapsed", "date", "host", "created")


def same_array(x: np.ndarray, y: np.ndarray) -> bool:
    """Equal shape and values; NaN equals NaN in float arrays (e.g. no goal yet, attention of a hidden token)."""
    x, y = np.asarray(x), np.asarray(y)
    return np.array_equal(x, y, equal_nan=x.dtype.kind == "f" and y.dtype.kind == "f")


def same_npz(a: Path, b: Path) -> List[str]:
    """Names of the arrays that differ (or exist in only one file)."""
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    keys = set(x.files) | set(y.files)
    return sorted(k for k in keys if k not in x.files or k not in y.files or not same_array(x[k], y[k]))


def same_obj(a: Any, b: Any) -> bool:
    """Deep equality for pickled drive data (dicts, lists, arrays, numbers)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_obj(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same_obj(u, v) for u, v in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return same_array(a, b)
    return a == b


def strip_timing(d: Any) -> Any:
    """A JSON value without keys that name a time, duration or host."""
    if isinstance(d, dict):
        return {k: strip_timing(v) for k, v in d.items() if not any(t in k.lower() for t in TIMING)}
    if isinstance(d, list):
        return [strip_timing(v) for v in d]
    return d


def g10_flags(d: Any, path: str = "") -> Dict[str, Any]:
    """Every "pass" and "equal" field of a checks_g10 result, by its path."""
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if k in ("pass", "equal"):
                out[p] = v
            else:
                out.update(g10_flags(v, p))
    return out


def check_g10(p11: Path) -> Dict[str, Any]:
    """Rows 1-2 pass again and every pass/equal flag equals the original run's. The v1 map-hidden comparison is
    report only (on CPU it differs by ~4e-6 > 1e-6 in every Phase 10/11 file; 0.0 on GPU)."""
    orig = json.loads((p11 / "offline/v21_ema9/g10_rows_1_2_cpu.json").read_text())
    new = json.loads((p11 / "checks/recheck/g10_rows_1_2_cpu.json").read_text())
    same = g10_flags(orig) == g10_flags(new)
    rows = new["row1"]["pass"] is True and new["row2"]["pass"] is True
    return {"pass": same and rows, "rows_1_2_pass": rows, "same_flags_as_original": same,
            "v1_map_hidden_report_only": new.get("v1_map_hidden_report")}


def check_rescore(p11: Path) -> Dict[str, Any]:
    diff = same_npz(p11 / "offline/v21_ema9/search/scores.npz", p11 / "checks/recheck/search/scores.npz")
    return {"pass": not diff, "differing_arrays": diff}


def check_determinism(p11: Path, episode: str) -> Dict[str, Any]:
    name = f"logs/v21/{episode}__s0.jsonl"
    a, b = (p11 / f"checks/recheck/determinism_{r}" / name for r in "ab")
    dev = p11 / "dev_v21" / name
    return {"pass": a.read_bytes() == b.read_bytes(), "episode": episode,
            "same_as_dev_run_report_only": a.read_bytes() == dev.read_bytes()}


def check_drive(orig: Path, new: Path) -> Dict[str, Any]:
    files_a = sorted(p.name for p in orig.iterdir())
    files_b = sorted(p.name for p in new.iterdir())
    bad: List[str] = []
    for f in set(files_a) & set(files_b):
        a, b = orig / f, new / f
        if f.endswith(".jpg") and a.read_bytes() != b.read_bytes():
            bad.append(f)
        elif f.endswith(".npz"):
            bad += [f"{f}:{k}" for k in same_npz(a, b)]
        elif f.endswith(".pkl") and not same_obj(pickle.loads(a.read_bytes()), pickle.loads(b.read_bytes())):
            bad.append(f)
        elif f.endswith(".json") and strip_timing(json.loads(a.read_text())) != strip_timing(json.loads(b.read_text())):
            bad.append(f)
    return {"pass": files_a == files_b and not bad, "files": len(files_a), "same_file_list": files_a == files_b,
            "jpg": sum(f.endswith(".jpg") for f in files_a), "differing": bad[:20]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--p11", type=Path, required=True)
    p.add_argument("--episode", required=True, help="the determinism episode (dev set)")
    p.add_argument("--drives", type=Path, required=True)
    p.add_argument("--redrives", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("rollouts", nargs="+")
    a = p.parse_args()
    drives = {e: check_drive(a.drives / e, a.redrives / e) for e in a.rollouts}
    res = {"g10_rows_1_2": check_g10(a.p11), "rescore": check_rescore(a.p11),
           "determinism": check_determinism(a.p11, a.episode),
           "rollouts": {"pass": all(d["pass"] for d in drives.values()), "drives": drives}}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
