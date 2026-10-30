"""Build and freeze the Phase 13 MP3D episode sets (configs/p13_claim.yaml `sets.examb` + `sets.dev`; kickoff 13.2):
exam B (MP3D val + test homes) and the new dev set (MP3D train homes), with Phase 12's builder rules
(mapmad_sim/search_set.DevSetBuilder). Inside naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/make_p13_sets.py vp-calibrate --gpu 0                        # once, before the candidates
    python mapmad/scripts/make_p13_sets.py examb-candidates --gpu 0 --worker 0 --workers 3
    python mapmad/scripts/make_p13_sets.py dev-candidates --gpu 1 --worker 0 --workers 3
    python mapmad/scripts/make_p13_sets.py freeze                                       # once, after all

Homes = <p13>/mp3d_homes.json (13.0 integrity: frozen train / val / test lists). Every home runs with split "mp3d",
LIMO's floor map rebuilt, the pin025 camera, mpcat40 labels.
- val homes: official ObjectNav MP3D val episodes and goals of our 6 categories (Phase 12 exactly).
- test homes (no official episodes): goals and view points from mapmad_sim/mp3d_goals.py (ObjectNav's recipe,
  radius calibrated on train homes), and `test_candidates` (60) seeded starts per home: a random point of LIMO's floor
  map (pathfinder seeded with test_seed + home index), a uniform random heading, the category round-robin over the
  home's categories with a usable view point (seeded order); ids <prefix>-<home>-<category>-g<nnn>.
- dev (train homes): official train episodes of our 6 categories, `candidates_per_home` (30) seeded per home.
Rules (first failing one counted): start_snap, no_reachable_instance / other_storey, instance_visible (any pixel,
8 headings), distance (3-15 m to the nearest usable view point on the start storey), not_seeable (Phase 8 start
filter). `freeze`: exam_set.draw (seeded, <= per_home per home, aim 100 per bin, neighbour-bin fill, round-robin over
the categories within a bin); a bin below report_min_per_bin (50) is reported, never a stop (kickoff §3); duplicate
ids refuse the set (exit 3, automatic stop). Writes <folder>/episodes.json (+ fingerprint), build_report.md and
<p13>/sets_mp3d.json. `vp-calibrate` -> <p13>/checks/vp_calibration.json (official vs generated usable view points
on seeded train homes, per radius; the chosen radius goes into the config before any exam-B candidate).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

from mapmad_sim import config, objectnav  # noqa: E402
from mapmad_sim.categories import CATEGORIES  # noqa: E402
from mapmad_sim.datagen_v2 import robot_spec  # noqa: E402
from mapmad_sim.exam_set import HomeSetBuilder, SetRules, draw, freeze  # noqa: E402
from mapmad_sim.mp3d_goals import VpRule, box_distance_xz, generated_goals  # noqa: E402
from mapmad_sim.robot import LimoSim  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, read_config, write_text_atomic  # noqa: E402
from mapmad_sim.search_set import DEV_RULES, DevSetBuilder, search_set_ok  # noqa: E402
from mapmad_sim.target_layer import NoiseSpec  # noqa: E402

CONFIG = config.CONFIG_DIR / "p13_claim.yaml"
STOP_EXIT = 3
SPLIT = config.MP3D_SPLIT


def home_lists(layout: RunLayout) -> Dict[str, List[str]]:
    return json.loads((layout.base / "mp3d_homes.json").read_text())["lists"]


def builder(robot: LimoSim, goals, layout: RunLayout, dg: Dict[str, Any], **kw) -> DevSetBuilder:
    return DevSetBuilder(robot, goals, SetRules.from_config(layout.cfg["set_rules"]),
                         NoiseSpec.from_config(dg["target_layer"]), **kw)


def official(split: str, home: str) -> List[objectnav.Goal]:
    """Official goals of our 6 categories."""
    return [g for g in objectnav.home_goals(split, home, config.paths()["mp3d_objectnav"]) if g.category in CATEGORIES]


def official_episodes(split: str, home: str) -> List[objectnav.OfficialEpisode]:
    eps = objectnav.home_episodes(split, home, config.paths()["mp3d_objectnav"])
    return [e for e in eps if e.category in CATEGORIES]


def generated_starts(robot: LimoSim, b: DevSetBuilder, home: str, n: int, seed: int) -> List[objectnav.OfficialEpisode]:
    """n seeded starts of a test home (module docstring); none if no category has a usable view point."""
    cats = [c for c in CATEGORIES if any(i.view_points for i in b.base.category_instances(c))]
    if not cats:
        return []
    rng = np.random.default_rng(seed)
    cats = [cats[int(k)] for k in rng.permutation(len(cats))]
    robot.pathfinder.seed(int(seed))
    out = []
    for k in range(n):
        p = np.array(robot.pathfinder.get_random_navigable_point(), dtype=np.float64)
        yaw = float(rng.uniform(-math.pi, math.pi))
        out.append(objectnav.OfficialEpisode(f"g{k:03d}", home, [float(v) for v in p], yaw, cats[k % len(cats)]))
    return out


def evaluate_home(layout: RunLayout, dg: Dict[str, Any], home: str, kind: str, gpu: int, idx: int) -> Dict[str, Any]:
    """Rule outcomes of every candidate episode of one home (kind: val | test | dev)."""
    s = layout.cfg["sets"]["dev" if kind == "dev" else "examb"]
    with LimoSim(SPLIT, home, robot_spec(dg), semantic=True, depth=True, gpu=gpu) as robot:
        if kind == "test":
            goals = generated_goals(robot, VpRule.from_config(layout.cfg["mp3d_view_points"]))
        else:
            goals = official("train" if kind == "dev" else "val", home)
        b = builder(robot, goals, layout, dg, id_prefix=s["id_prefix"], set_type=f"{'dev' if kind == 'dev' else 'examb'}_{kind}")
        if kind == "test":
            eps = generated_starts(robot, b, home, int(s["test_candidates"]), int(s["test_seed"]) + idx)
        else:
            eps = official_episodes("train" if kind == "dev" else "val", home)
            n = s.get("candidates_per_home")
            if n and n < len(eps):
                pick = np.random.default_rng([int(s["candidate_seed"]), idx]).choice(len(eps), int(n), replace=False)
                eps = [eps[i] for i in sorted(pick)]
        results = [b.evaluate(e, SPLIT) for e in eps]
        vps = {c: sum(len(i.view_points) for i in b.base.category_instances(c)) for c in CATEGORIES}
        return {"home": home, "kind": kind, "navmesh_sha256": robot.navmesh_sha256, "examined": len(eps),
                "goals": len(goals), "usable_view_points": vps, "seeable": b._seeable, "results": results}


def cmd_candidates(args, layout: RunLayout, dg: Dict[str, Any], which: str) -> None:
    lists = home_lists(layout)
    todo = [(h, "dev") for h in lists["train"]] if which == "dev" else \
        [(h, "val") for h in lists["val"]] + [(h, "test") for h in lists["test"]]
    s = layout.cfg["sets"]["dev" if which == "dev" else "examb"]
    folder = layout.base / s["folder"]
    mine = [(i, h, k) for i, (h, k) in enumerate(todo)][args.worker::args.workers]
    save_run_info(folder / "_runs", {**vars(args), "set": s, "homes": [h for _, h, _ in mine]}, int(s["draw_seed"]),
                  name=f"run_info_candidates_w{args.worker}of{args.workers}.json")
    for idx, home, kind in mine:
        path = folder / "candidates" / f"{home}.json"
        if path.exists():
            continue
        t0 = time.time()
        out = evaluate_home(layout, dg, home, kind, args.gpu, idx)
        out["seconds"] = round(time.time() - t0, 1)
        write_text_atomic(path, json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({"home": home, "kind": kind, "examined": out["examined"], "seconds": out["seconds"],
                          "kept": sum(r.get("episode") is not None for r in out["results"])}), flush=True)


# --- view-point calibration (train homes only) ---------------------------------------------------------------------
def near_share(a: List[np.ndarray], b: List[np.ndarray], tol: float = 0.2) -> float:
    """Share of points of a with a point of b within tol horizontally and 0.5 m vertically (1.0 if a is empty)."""
    if not a:
        return 1.0
    if not b:
        return 0.0
    bb = np.asarray(b)
    hits = 0
    for p in a:
        d = np.hypot(bb[:, 0] - p[0], bb[:, 2] - p[2])
        hits += bool(((d <= tol) & (np.abs(bb[:, 1] - p[1]) <= 0.5)).any())
    return hits / len(a)


def cmd_vp_calibrate(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    """Official vs generated usable view points (largest radius once, smaller radii as subsets) on seeded train homes."""
    c = layout.cfg["mp3d_view_points"]
    radii = [float(r) for r in c["calibrate_radii"]]
    train = home_lists(layout)["train"]
    homes = sorted(train[i] for i in np.random.default_rng(int(c["calibrate_seed"])).choice(len(train), int(c["calibrate_homes"]), replace=False))
    rules = SetRules.from_config(layout.cfg["set_rules"])
    rows: Dict[float, List[Dict[str, float]]] = {r: [] for r in radii}
    for home in homes:
        with LimoSim(SPLIT, home, robot_spec(dg), semantic=True, depth=True, gpu=args.gpu) as robot:
            off = HomeSetBuilder(robot, official("train", home), rules, np.random.default_rng(0))
            gen_goals = generated_goals(robot, VpRule.from_config(dict(c, radius_m=max(radii))))
            gen = HomeSetBuilder(robot, gen_goals, rules, np.random.default_rng(0))
            by_id = {g.object_id: g for g in gen_goals}
            for g in off.goals:
                oi = off.instance(g)
                if oi is None or g.object_id not in by_id:
                    continue
                gi = gen.instance(by_id[g.object_id])
                for r in radii:
                    gv = [p for p in gi.view_points if box_distance_xz(p, gi.box_center, gi.box_size) <= r]
                    rows[r].append({"recall": near_share(oi.view_points, gv), "precision": near_share(gv, oi.view_points),
                                    "official_any": float(bool(oi.view_points)), "generated_any": float(bool(gv))})
        print(json.dumps({"home": home, "goals": len(rows[radii[0]])}), flush=True)
    out = {"homes": homes, "per_radius": {}}
    for r in radii:
        rec = float(np.mean([x["recall"] for x in rows[r]]))
        pre = float(np.mean([x["precision"] for x in rows[r]]))
        out["per_radius"][str(r)] = {"goals": len(rows[r]), "recall": round(rec, 4), "precision": round(pre, 4),
                                     "f1": round(2 * rec * pre / max(rec + pre, 1e-9), 4),
                                     "official_has_vp": round(float(np.mean([x["official_any"] for x in rows[r]])), 4),
                                     "generated_has_vp": round(float(np.mean([x["generated_any"] for x in rows[r]])), 4)}
    best = max(radii, key=lambda r: (out["per_radius"][str(r)]["f1"], -r))
    out["chosen_radius_m"] = best
    write_text_atomic(layout.base / "checks" / "vp_calibration.json", json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# --- freeze -----------------------------------------------------------------------------------------------------------
def drop_table(results: List[Dict[str, Any]]) -> List[str]:
    cols = list(DEV_RULES) + ["kept"]
    first: Dict[str, Counter] = {}
    for r in results:
        for k in ("all", r["category"]):
            first.setdefault(k, Counter())[r["first_fail"] or "kept"] += 1
    lines = ["| | examined | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k in ["all"] + sorted(x for x in first if x != "all"):
        cnt = first[k]
        lines.append(f"| {k} | {sum(cnt.values())} | " + " | ".join(str(cnt.get(x, 0)) for x in cols) + " |")
    return lines


def freeze_one(layout: RunLayout, name: str, homes: List[str]) -> str:
    s = layout.cfg["sets"][name]
    bins = layout.cfg["set_rules"]["bins_m"]
    n_bins = len(bins) - 1
    folder = layout.base / s["folder"]
    files = [folder / "candidates" / f"{h}.json" for h in homes]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{name} candidates missing for {len(missing)} homes: {missing[:5]}")
    docs = [json.loads(f.read_text()) for f in files]
    results = [dict(r, home=d["home"], kind=d["kind"]) for d in docs for r in d["results"]]
    kept = [r["episode"] for r in results if r.get("episode") is not None]
    chosen, rep = draw(kept, int(s["total"]), int(s["per_home"]), n_bins, int(s["draw_seed"]), "round_robin_category")
    bad = search_set_ok(chosen)
    if bad:
        write_text_atomic(layout.base / "STOP_sets.txt", f"{name} set refused: {bad}\n")
        raise SystemExit(STOP_EXIT)
    kind_of = {d["home"]: d["kind"] for d in docs}
    fp = freeze(folder / "episodes.json", chosen, {"set": name, "rules": layout.cfg["set_rules"], "set_config": s,
                                                   "navmesh_sha256": {d["home"]: d["navmesh_sha256"] for d in docs},
                                                   "home_kind": kind_of, "draw": rep})
    short = [f"{bins[b]:g}-{bins[b + 1]:g} m" for b in range(n_bins) if rep["chosen_per_bin"][b] < int(s.get("report_min_per_bin", 0))]
    lines = [f"# Phase 13 {name} set: build report", "",
             f"Fingerprint `{fp}`; {len(chosen)} episodes (asked {s['total']}, <= {s['per_home']} per home, draw seed "
             f"{s['draw_seed']}, round-robin over the categories within a bin); candidates examined: {len(results)} in "
             f"{len(docs)} homes.", "",
             f"Bins below {s.get('report_min_per_bin', 0)} after the fill (reported, no stop): {short or 'none'}.", "",
             "## Drops (first failing rule)", ""] + drop_table(results)
    for kind in sorted(set(kind_of.values())):
        sub = [r for r in results if r["kind"] == kind]
        lines += ["", f"### {kind} homes", ""] + drop_table(sub)
    lines += ["", "## Distance bins", "", "| bin | kept | chosen |", "|---|---|---|"]
    lines += [f"| {bins[b]:g}-{bins[b + 1]:g} m | {rep['kept_per_bin'][b]} | {rep['chosen_per_bin'][b]} |" for b in range(n_bins)]
    lines += ["", f"Moved quota: {rep['moved'] or 'none'}; missing: {rep['missing']}.", "",
              "| category | kept | chosen |", "|---|---|---|"]
    kc, cc = Counter(e["target"]["category"] for e in kept), Counter(e["target"]["category"] for e in chosen)
    lines += [f"| {c} | {kc[c]} | {cc[c]} |" for c in sorted(kc)]
    lines += ["", "| home | kind | kept | chosen |", "|---|---|---|---|"]
    kh = Counter(e["home"] for e in kept)
    lines += [f"| {h} | {kind_of[h]} | {kh[h]} | {rep['chosen_per_home'].get(h, 0)} |" for h in homes]
    write_text_atomic(folder / "build_report.md", "\n".join(lines) + "\n")
    return fp


def cmd_freeze(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    out = layout.base / "sets_mp3d.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the MP3D sets are frozen (move it away to rebuild)")
    lists = home_lists(layout)
    fps = {"examb": freeze_one(layout, "examb", lists["val"] + lists["test"]), "dev": freeze_one(layout, "dev", lists["train"])}
    write_text_atomic(out, json.dumps({"fingerprints": fps, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1) + "\n")
    save_run_info(layout.base / layout.cfg["sets"]["examb"]["folder"] / "_runs", {**vars(args), "fingerprints": fps},
                  int(layout.cfg["sets"]["examb"]["draw_seed"]), name="run_info_freeze.json")
    print(json.dumps(fps, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["vp-calibrate", "examb-candidates", "dev-candidates", "freeze"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    dg = read_config(config.CONFIG_DIR / layout.cfg["datagen"])
    if args.cmd == "vp-calibrate":
        cmd_vp_calibrate(args, layout, dg)
    elif args.cmd == "freeze":
        cmd_freeze(args, layout, dg)
    else:
        cmd_candidates(args, layout, dg, "dev" if args.cmd == "dev-candidates" else "examb")


if __name__ == "__main__":
    main()
