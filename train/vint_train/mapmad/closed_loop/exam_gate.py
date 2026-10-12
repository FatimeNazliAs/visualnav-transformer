"""Gate G5 table (block item 12) from the exam outputs (inside naz_mapmad, after exam_tables.py):

    python -m vint_train.mapmad.closed_loop.exam_gate --config /app/visualnav-transformer/mapmad/configs/p5_exam.yaml \
        --subdir runs --tables tables --determinism checks/determinism_exam/determinism.json --videos videos

Rows (pass/fail decided here only where the rule is mechanical; the rest is marked for Naz's review):
1. fairness: every arm has every (episode, seed index) of the frozen set (planner: index 0), minus the episodes
   removed for crashes in all arms (block item 13b: > 2% removed = a bug -> fail); every run's weights sha256 equals
   the frozen hash and its episode-set fingerprint the frozen set's;
2. C - B primary success > 0 with home-clustered one-sided p < 0.05 (the only primary p-value);
3. C - D primary success > 0 with the home-clustered 95% CI lower bound > 0;
4. planner comparison reported per distance bin (C vs E: success, SPL, collisions, stuck streaks);
5. videos rendered (count); legibility is Naz's check;
6. determinism: the pre-registered C episode re-run gives a byte-identical log.
Writes <out_dir>/<tables>/gate.json and gate.md.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mapmad_sim.episodes import load_episodes
from mapmad_sim.exam_arms import arms_of
from mapmad_sim.run_layout import RunLayout

MAX_REMOVED_SHARE = 0.02


def fairness(cfg: Dict[str, Any], episodes: List[Dict[str, Any]], fp: str, rows: List[Dict[str, str]],
             crashed: List[str]) -> Dict[str, Any]:
    """Row 1 facts: expected vs present runs per arm, hashes, removed episodes."""
    seeds = range(int(cfg["run"]["seeds"]))
    sha = {"nomad": cfg["nomad"]["weights_sha256"], "mapmad": cfg["mapmad"]["checkpoint_sha256"]}
    kept = {e["episode_id"] for e in episodes} - set(crashed)
    have = Counter((r["arm"], r["episode_id"], r["seed_index"]) for r in rows)
    per_arm, problems = {}, []
    for a in arms_of(cfg):
        arm = a.name
        want = {(arm, e, str(s)) for e in kept for s in a.seed_indices(seeds)}
        got = {key for key in have if key[0] == arm}
        mine = [r for r in rows if r["arm"] == arm]
        bad_sha = sum((r["weights_sha256"] or "") != sha.get(a.weights_key, "") for r in mine)
        bad_fp = sum(r["episodes_fingerprint"] != fp for r in mine)
        per_arm[arm] = {"expected": len(want), "present": len(want & got), "missing": len(want - got),
                        "extra": len(got - want), "duplicates": sum(have[x] > 1 for x in got),
                        "wrong_weights_sha256": bad_sha, "wrong_fingerprint": bad_fp}
        if want - got or got - want or bad_sha or bad_fp:
            problems.append(arm)
    share = len(crashed) / len(episodes)
    return {"per_arm": per_arm, "removed_episodes": crashed, "removed_share": share,
            "pass": not problems and share <= MAX_REMOVED_SHARE, "arms_with_problems": problems}


def planner_by_bin(episodes_csv: Path, bins: List[float]) -> Dict[str, Any]:
    """C and E per distance bin: success, SPL, collisions, stuck streaks (episode means)."""
    out: Dict[str, Any] = {}
    with open(episodes_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["arm"] in ("C", "E")]
    for b in range(len(bins) - 1):
        name = f"{bins[b]:g}-{bins[b + 1]:g} m"
        out[name] = {arm: {m: float(np.mean([float(r[m]) for r in rows if r["arm"] == arm and int(r["distance_bin"]) == b]))
                           for m in ("success", "spl", "collisions", "stuck_streaks")}
                     for arm in ("C", "E")}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="runs")
    p.add_argument("--tables", default="tables")
    p.add_argument("--determinism", default="checks/determinism_exam/determinism.json")
    p.add_argument("--videos", default="videos")
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    cfg = layout.cfg
    episodes, fp = load_episodes(layout.episodes_file)
    tables = layout.base / args.tables
    with open(tables / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    doc = json.loads((tables / "paired.json").read_text())
    paired = doc["paired"]
    fair = fairness(cfg, episodes, fp, rows, doc["crashed"])
    cb, cd = paired["C-B"]["success"], paired["C-D"]["success"]
    det_path = layout.base / args.determinism
    det = json.loads(det_path.read_text()) if det_path.exists() else {"pass": False, "note": "not run"}
    videos = sorted(p.name for p in (layout.base / args.videos).glob("*.mp4"))
    rows_out = [
        {"row": 1, "name": "fairness", "pass": fair["pass"],
         "result": f"{sum(a['present'] for a in fair['per_arm'].values())} runs present of "
                   f"{sum(a['expected'] for a in fair['per_arm'].values())} expected; problems: "
                   f"{fair['arms_with_problems'] or 'none'}; removed episodes {len(fair['removed_episodes'])} "
                   f"({fair['removed_share']:.1%}, limit {MAX_REMOVED_SHARE:.0%})"},
        {"row": 2, "name": "C - B > 0, home p < 0.05", "pass": cb["diff"] > 0 and cb["home_p_one_sided"] < 0.05,
         "result": f"{cb['diff']:+.3f} [{cb['home_ci'][0]:+.3f}, {cb['home_ci'][1]:+.3f}], p = {cb['home_p_one_sided']:.4f}"},
        {"row": 3, "name": "C - D > 0, home CI lower bound > 0", "pass": cd["diff"] > 0 and cd["home_ci"][0] > 0,
         "result": f"{cd['diff']:+.3f} [{cd['home_ci'][0]:+.3f}, {cd['home_ci'][1]:+.3f}]"},
        {"row": 4, "name": "planner comparison per bin", "pass": None, "result": "reported (tables.md, gate.json)",
         "per_bin": planner_by_bin(tables / "episodes.csv", cfg["set_rules"]["bins_m"])},
        {"row": 5, "name": "videos", "pass": None, "result": f"{len(videos)} videos rendered; legibility: Naz",
         "files": videos},
        {"row": 6, "name": "determinism", "pass": bool(det.get("pass")), "result": det.get("note", ""), **det},
    ]
    out = {"fingerprint": fp, "rows": rows_out, "fairness": fair}
    (tables / "gate.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    mark = {True: "pass", False: "FAIL", None: "for review"}
    lines = ["# Gate G5", "", f"Exam set `{fp}`. Mechanical rows decided by `exam_gate.py`; rows 4-5 need Naz's review.",
             "", "| row | check | result | status |", "|---|---|---|---|"]
    lines += [f"| {r['row']} | {r['name']} | {r['result']} | {mark[r['pass']]} |" for r in rows_out]
    lines += ["", "## Row 4: C vs E per distance bin (episode means)", "",
              "| bin | C success | E success | C SPL | E SPL | C collisions | E collisions | C stuck | E stuck |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, v in rows_out[3]["per_bin"].items():
        c, e = v["C"], v["E"]
        lines.append(f"| {name} | {c['success']:.2f} | {e['success']:.2f} | {c['spl']:.2f} | {e['spl']:.2f} | "
                     f"{c['collisions']:.0f} | {e['collisions']:.0f} | {c['stuck_streaks']:.2f} | {e['stuck_streaks']:.2f} |")
    (tables / "gate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
