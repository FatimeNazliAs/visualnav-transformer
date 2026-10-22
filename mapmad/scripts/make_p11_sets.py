"""Build the Phase 11 episode sets (configs/p11_dagger.yaml `sets`; rules in mapmad_sim/search_set.py). Inside
naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/make_p11_sets.py dev-candidates --gpu 0 --worker 0 --workers 2    # per worker
    python mapmad/scripts/make_p11_sets.py dev-freeze                                       # once, after all
    python mapmad/scripts/make_p11_sets.py rollout-candidates --gpu 1 --worker 0 --workers 6
    python mapmad/scripts/make_p11_sets.py rollout-freeze

`dev-candidates`: a seeded sample of each dev home's official episodes -> rule outcomes in
<out>/dev_set/candidates/<home>.json (resumable: a home with a file is skipped). `dev-freeze`: exam_set.draw over the
kept ones -> dev_set/episodes.json (+ episodes.sha256, fingerprint), build_report.md / .json.
`rollout-candidates`: per labelled train home (worker k of n takes homes k, k+n, ...), per_home draws with Phase 8's
search-start rules and seed 1101 -> rollouts/set/candidates/<home>.json. `rollout-freeze`: all of them ->
rollouts/set/episodes.json (+ sha256), build_report.md. Both freezes refuse duplicate ids and an existing frozen set.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

from mapmad_sim import config, home_splits, objectnav  # noqa: E402
from mapmad_sim.datagen_v2 import HomeGeneratorV2, robot_spec  # noqa: E402
from mapmad_sim.exam_set import SetRules, draw, freeze  # noqa: E402
from mapmad_sim.robot import LimoSim  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, read_config, write_text_atomic  # noqa: E402
from mapmad_sim.search_set import DEV_RULES, DevSetBuilder, rollout_episodes, search_set_ok  # noqa: E402
from mapmad_sim.target_layer import NoiseSpec  # noqa: E402

CONFIG = config.CONFIG_DIR / "p11_dagger.yaml"


def datagen_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return read_config(config.CONFIG_DIR / cfg["datagen"])


def dev_homes(s: Dict[str, Any]) -> List[str]:
    return sorted(home_splits.by_prefix(home_splits.train_homes(s["split"]), s["homes"]))


def labelled_homes(dg: Dict[str, Any]) -> List[str]:
    """The 145 labelled train homes (generate_drives_v2.labelled_homes)."""
    lab = set(config.labelled_homes(dg["split"]))
    return [h for h in home_splits.train_homes(dg["split"]) if h in lab and objectnav.has_goals(dg["objectnav_split"], h)]


def cmd_dev_candidates(args, layout: RunLayout) -> None:
    s = layout.cfg["sets"]["dev"]
    dg = datagen_config(layout.cfg)
    rules = SetRules.from_config(layout.cfg["set_rules"])
    folder = layout.base / s["folder"]
    homes = dev_homes(s)
    mine = homes[args.worker::args.workers]
    save_run_info(folder / "_runs", {**vars(args), "set": s, "set_rules": layout.cfg["set_rules"], "homes": mine},
                  int(s["candidate_seed"]), name=f"run_info_candidates_w{args.worker}of{args.workers}.json")
    for home in mine:
        path = folder / "candidates" / f"{home}.json"
        if path.exists():
            continue
        t0 = time.time()
        episodes = objectnav.home_episodes(s["objectnav_split"], home)
        n = s["candidates_per_home"]
        if n and n < len(episodes):
            pick = np.random.default_rng([s["candidate_seed"], homes.index(home)]).choice(len(episodes), n, replace=False)
            episodes = [episodes[i] for i in sorted(pick)]
        with LimoSim(s["split"], home, robot_spec(dg), semantic=True, depth=True, gpu=args.gpu) as robot:
            b = DevSetBuilder(robot, objectnav.home_goals(s["objectnav_split"], home), rules,
                              NoiseSpec.from_config(dg["target_layer"]))
            results = [b.evaluate(e, s["split"]) for e in episodes]
            navmesh = robot.navmesh_sha256
        out = {"home": home, "navmesh_sha256": navmesh, "examined": len(episodes), "seeable": b._seeable,
               "seconds": round(time.time() - t0, 1), "results": results}
        write_text_atomic(path, json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({"home": home, "examined": len(episodes), "seconds": out["seconds"],
                          "kept": sum(r.get("episode") is not None for r in results)}), flush=True)


def drop_table(results: List[Dict[str, Any]], rules: List[str]) -> List[str]:
    cols = rules + ["kept"]
    first: Dict[str, Counter] = defaultdict(Counter)
    for r in results:
        first["all"][r["first_fail"] or "kept"] += 1
        first[r["category"]][r["first_fail"] or "kept"] += 1
    lines = ["| | examined | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k in ["all"] + sorted(x for x in first if x != "all"):
        c = first[k]
        lines.append(f"| {k} | {sum(c.values())} | " + " | ".join(str(c.get(x, 0)) for x in cols) + " |")
    return lines


def cmd_dev_freeze(args, layout: RunLayout) -> None:
    s = layout.cfg["sets"]["dev"]
    folder = layout.base / s["folder"]
    homes = dev_homes(s)
    files = [folder / "candidates" / f"{h}.json" for h in homes]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"candidates missing for {len(missing)} homes: {missing}")
    results, navmesh = [], {}
    for f in files:
        doc = json.loads(f.read_text())
        navmesh[doc["home"]] = doc["navmesh_sha256"]
        results += [dict(r, home=doc["home"]) for r in doc["results"]]
    kept = [r["episode"] for r in results if r.get("episode") is not None]
    bins = layout.cfg["set_rules"]["bins_m"]
    chosen, rep = draw(kept, int(s["total"]), int(s["per_home"]), len(bins) - 1, int(s["draw_seed"]))
    bad = search_set_ok(chosen)
    if bad:
        raise SystemExit(f"dev set refused: {bad}")
    fp = freeze(folder / "episodes.json", chosen, {"set": "dev", "rules": layout.cfg["set_rules"], "set_config": s,
                                                    "navmesh_sha256": navmesh, "draw": rep})
    lines = [f"# Phase 11 dev set: build report", "", f"Fingerprint `{fp}`; {len(chosen)} episodes (asked {s['total']}, "
             f"<= {s['per_home']} per home, draw seed {s['draw_seed']}; {s['candidates_per_home']} official ObjectNav "
             f"v2 train episodes examined per home, seed {s['candidate_seed']}).", "",
             "## Drops (first failing rule)", ""] + drop_table(results, list(DEV_RULES))
    lines += ["", "## Distance bins", "", "| bin | kept | chosen |", "|---|---|---|"]
    for b in range(len(bins) - 1):
        lines.append(f"| {bins[b]:g}-{bins[b + 1]:g} m | {rep['kept_per_bin'][b]} | {rep['chosen_per_bin'][b]} |")
    lines += ["", f"Moved quota: {rep['moved'] or 'none'}; missing: {rep['missing']}.", "",
              "| category | chosen |", "|---|---|"]
    for cat, n in sorted(Counter(e["target"]["category"] for e in chosen).items()):
        lines.append(f"| {cat} | {n} |")
    lines += ["", "| home | chosen |", "|---|---|"] + [f"| {h} | {n} |" for h, n in sorted(rep["chosen_per_home"].items())]
    (folder / "build_report.md").write_text("\n".join(lines) + "\n")
    (folder / "build_report.json").write_text(json.dumps({"fingerprint": fp, "draw": rep, "navmesh_sha256": navmesh},
                                                         indent=1) + "\n")
    save_run_info(folder / "_runs", {**vars(args), "set": s, "fingerprint": fp}, int(s["draw_seed"]),
                  name="run_info_freeze.json")
    print("\n".join(lines))


def cmd_rollout_candidates(args, layout: RunLayout) -> None:
    s = layout.cfg["sets"]["rollout"]
    dg = dict(datagen_config(layout.cfg), seed=int(s["seed"]))
    folder = layout.base / s["folder"]
    homes = labelled_homes(dg)
    mine = homes[args.worker::args.workers]
    save_run_info(folder / "_runs", {**vars(args), "set": s, "homes": mine, "datagen": dg}, int(s["seed"]),
                  name=f"run_info_candidates_w{args.worker}of{args.workers}.json")
    for home in mine:
        path = folder / "candidates" / f"{home}.json"
        if path.exists():
            continue
        t0 = time.time()
        gen = HomeGeneratorV2(dg, home, labelled=True, gpu=args.gpu)
        try:
            eps, notes = rollout_episodes(gen, home, int(s["per_home"]), layout.cfg["set_rules"]["bins_m"])
            out = {"home": home, "navmesh_sha256": gen.robot.navmesh_sha256, "categories": gen.tasks.summary(),
                   "notes": notes, "episodes": eps, "seconds": round(time.time() - t0, 1)}
        finally:
            gen.close()
        write_text_atomic(path, json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({"home": home, "episodes": len(eps), "seconds": out["seconds"]}), flush=True)


def cmd_rollout_freeze(args, layout: RunLayout) -> None:
    s = layout.cfg["sets"]["rollout"]
    dg = datagen_config(layout.cfg)
    folder = layout.base / s["folder"]
    homes = labelled_homes(dg)
    files = [folder / "candidates" / f"{h}.json" for h in homes]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"candidates missing for {len(missing)} homes: {missing[:5]}")
    eps, no_sample, navmesh = [], 0, {}
    for f in files:
        doc = json.loads(f.read_text())
        eps += doc["episodes"]
        navmesh[doc["home"]] = doc["navmesh_sha256"]
        no_sample += sum(n["outcome"] == "no_sample" for n in doc["notes"])
    bad = search_set_ok(eps)
    if bad:
        raise SystemExit(f"rollout set refused: {bad}")
    test = set(dev_homes(layout.cfg["sets"]["dev"])) if s.get("test_homes") == "dev" else set()
    for e in eps:
        e["split_role"] = "test" if e["home"] in test else "train"
    fp = freeze(folder / "episodes.json", eps, {"set": "rollout", "set_config": s, "navmesh_sha256": navmesh,
                                                "no_sample": no_sample})
    cats = Counter(e["target"]["category"] for e in eps)
    lines = ["# Phase 11 rollout set", "", f"Fingerprint `{fp}`; {len(eps)} episodes in {len(homes)} homes "
             f"({s['per_home']} draws per home, seed {s['seed']}; {no_sample} draws found no start). "
             f"DAgger test split (dev homes): {sum(e['split_role'] == 'test' for e in eps)} episodes.", "",
             "| category | episodes |", "|---|---|"] + [f"| {c} | {n} |" for c, n in sorted(cats.items())]
    geo = np.array([e["start_geodesic_m"] for e in eps])
    lines += ["", f"Start -> nearest view point: mean {geo.mean():.2f} m, median {np.median(geo):.2f} m; wall starts "
              f"{sum(e['wall_start'] for e in eps)}."]
    (folder / "build_report.md").write_text("\n".join(lines) + "\n")
    save_run_info(folder / "_runs", {**vars(args), "set": s, "fingerprint": fp}, int(s["seed"]), name="run_info_freeze.json")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["dev-candidates", "dev-freeze", "rollout-candidates", "rollout-freeze"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    {"dev-candidates": cmd_dev_candidates, "dev-freeze": cmd_dev_freeze, "rollout-candidates": cmd_rollout_candidates,
     "rollout-freeze": cmd_rollout_freeze}[args.cmd](args, layout)


if __name__ == "__main__":
    main()
