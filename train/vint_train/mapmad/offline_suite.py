"""Offline check suites (Phase 12 sub-step 12.0, architecture candidate 2): "the offline checks of a checkpoint" as data
(train/config/mapmad_offline_suites.yaml) behind one command, with one result file.

    python -m vint_train.mapmad.offline_eval run-suite --suite recheck --checkpoint <ema.pth> --config config/mapmad_v2.yaml \
        --out <dir> [--suites config/mapmad_offline_suites.yaml]
    python -m vint_train.mapmad.offline_eval suite-compare --a <dir> --b <reference dir> --out <dir>/compare.json

run-suite runs every row of the suite as the existing command (unchanged code, its own process: the G10 rows on the
CPU, scoring on the visible GPU) into <out>/<row name>/, then writes <out>/suite.json = {row: {kind, out, numbers}}
(numbers: g10 -> row1 / row2 pass; score -> mean loss per arm; compare -> the compare.json headline; seen -> the
seen-report; gostanford -> its json). suite-compare is the §0.8 re-check of two runs of a suite (or of a suite run and
an older output folder laid out the same way): every scores.npz array identical (NaN = NaN), every G10 "pass"/"equal"
flag identical, and the gostanford / compare numbers identical. Exit 0 = all identical, 3 = something differs.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

SUITES = "config/mapmad_offline_suites.yaml"  # relative to train/


def _module(*args: str) -> List[str]:
    return [sys.executable, "-m", *args]


def row_commands(row: Dict[str, Any], checkpoint: str, config: str, out: Path) -> List[List[str]]:
    """The command lines of one row (module docstring)."""
    o = str(out / row["name"])
    kind = row["kind"]
    if kind == "g10":
        return [_module("vint_train.mapmad.checks_g10", "--config", row["config"], "--v1-config", row["v1_config"],
                        "--reference", row["reference"], "--out", f"{o}/g10_rows_1_2_cpu.json")]
    if kind in ("score", "seen"):
        cmd = _module("vint_train.mapmad.offline_eval", "score", "--config", config, "--checkpoint", checkpoint,
                      "--list", row["list"], "--out", o)
        if row.get("arms"):
            cmd += ["--arms", row["arms"]]
        out = [cmd]
        if row.get("compare"):
            out.append(_module("vint_train.mapmad.offline_eval", "compare", "--scores", f"{o}/scores.npz", "--out", o))
        if kind == "seen":
            out.append(_module("vint_train.mapmad.target_groups", "seen-report", "--scores", f"{o}/scores.npz",
                               "--out", o))
        return out
    if kind == "gostanford":
        return [_module("vint_train.mapmad.offline_eval", "gostanford", "--config", config, "--checkpoint", checkpoint,
                        "--official", row["official"], "--out", o)]
    if kind == "guard":
        return [_module("vint_train.mapmad.target_groups", "guard", "--scores", f"{out}/{row['scores']}/scores.npz",
                        "--reference", row["reference"], "--out", f"{o}.json")]
    raise ValueError(f"unknown row kind {kind!r}")


def row_numbers(row: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """The numbers suite.json keeps for one finished row."""
    o = out / row["name"]
    kind = row["kind"]
    if kind == "g10":
        d = json.loads((o / "g10_rows_1_2_cpu.json").read_text())
        return {"row1_pass": bool(d["row1"]["pass"]), "row2_pass": bool(d["row2"]["pass"])}
    if kind == "gostanford":
        d = json.loads((o / "gostanford.json").read_text())
        return {k: d[k] for k in ("official", "mapmad", "diff", "pass")}
    if kind == "guard":
        return json.loads(Path(f"{o}.json").read_text())
    with np.load(o / "scores.npz") as z:
        nums: Dict[str, Any] = {k.split(".")[0]: float(np.mean(z[k])) for k in z.files if k.endswith(".gc_action_loss")}
    if (o / "compare.json").exists():
        c = json.loads((o / "compare.json").read_text())
        for key in ("row3", "maps_vs_hidden"):
            if key in c:
                nums[key] = {k: c[key][k] for k in ("diff", "diff_ci_drive") if k in c[key]}
    return {"mean_loss": nums}


def run_suite(suite: str, checkpoint: str, config: str, out: Path, suites_file: str = SUITES) -> Dict[str, Any]:
    rows = yaml.safe_load(Path(suites_file).read_text())["suites"][suite]
    out.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {"suite": suite, "checkpoint": checkpoint, "config": config, "rows": {}}
    for row in rows:
        env = dict(os.environ)
        if row["kind"] == "g10":
            env["CUDA_VISIBLE_DEVICES"] = ""  # rows 1-2 are checked on the CPU (Phase 10 reference)
        for cmd in row_commands(row, checkpoint, config, out):
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, env=env)
        result["rows"][row["name"]] = {"kind": row["kind"], "out": str(out / row["name"]),
                                       "numbers": row_numbers(row, out)}
    tmp = out / ".suite.json.tmp"
    tmp.write_text(json.dumps(result, indent=1) + "\n")
    os.replace(tmp, out / "suite.json")
    return result


def same_array(x: np.ndarray, y: np.ndarray) -> bool:
    """Equal shape and values; NaN equals NaN in float arrays."""
    x, y = np.asarray(x), np.asarray(y)
    return np.array_equal(x, y, equal_nan=x.dtype.kind == "f" and y.dtype.kind == "f")


def g10_flags(d: Any, path: str = "") -> Dict[str, Any]:
    """Every "pass" and "equal" field of a checks_g10 result, by its path."""
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if k in ("pass", "equal") and not isinstance(v, dict):
                out[p] = v
            else:
                out.update(g10_flags(v, p))
    return out


def compare_dirs(a: Path, b: Path) -> Dict[str, Any]:
    """{file: {"pass", ...}} for every scores.npz / g10 json / gostanford.json found below a (module docstring)."""
    res: Dict[str, Any] = {}
    for fa in sorted(a.rglob("scores.npz")):
        fb = b / fa.relative_to(a)
        if not fb.exists():
            res[str(fa.relative_to(a))] = {"pass": False, "why": f"{fb} missing"}
            continue
        x, y = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
        keys = set(x.files) | set(y.files)
        diff = sorted(k for k in keys if k not in x.files or k not in y.files or not same_array(x[k], y[k]))
        res[str(fa.relative_to(a))] = {"pass": not diff, "arrays": len(keys), "differ": diff}
    for fa in sorted(a.rglob("g10_rows_1_2_cpu.json")):
        fb = b / fa.relative_to(a)
        if not fb.exists():
            res[str(fa.relative_to(a))] = {"pass": False, "why": f"{fb} missing"}
            continue
        fx, fy = g10_flags(json.loads(fa.read_text())), g10_flags(json.loads(fb.read_text()))
        report_only = [k for k in fx if k.startswith("report") and "v1" in k]  # CPU rounding, report only (Phase 11)
        diff = sorted(k for k in set(fx) | set(fy) if k not in report_only and fx.get(k) != fy.get(k))
        res[str(fa.relative_to(a))] = {"pass": not diff and bool(fx.get("row1.pass")) and bool(fx.get("row2.pass")),
                                       "flags": len(fx), "differ": diff}
    for fa in sorted(a.rglob("gostanford.json")):
        fb = b / fa.relative_to(a)
        nums = ("official", "mapmad", "diff")
        same = fb.exists() and all(json.loads(fa.read_text())[k] == json.loads(fb.read_text())[k] for k in nums)
        res[str(fa.relative_to(a))] = {"pass": bool(same)}
    return res


def suite_compare(a: Path, b: Path, out: Path) -> int:
    res = compare_dirs(a, b)
    ok = bool(res) and all(r["pass"] for r in res.values())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"a": str(a), "b": str(b), "pass": ok, "files": res}, indent=1) + "\n")
    print(json.dumps({"pass": ok, "files": {k: v["pass"] for k, v in res.items()}}), flush=True)
    return 0 if ok else 3
