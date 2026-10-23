"""Build and freeze the Phase 12 episode sets (configs/p12_exam.yaml `sets`; block items 2, 5-7; rules in
mapmad_sim/search_set.py, the plan 8.7 builder that made Phase 11's dev set). Inside naz_mapmad_habitat, from
/app/visualnav-transformer:

    python mapmad/scripts/make_p12_sets.py exam-candidates --gpu 0 --worker 0 --workers 3     # per worker
    python mapmad/scripts/make_p12_sets.py ovon-candidates --gpu 1 --worker 0 --workers 3
    python mapmad/scripts/make_p12_sets.py freeze                                           # once, after all

Candidates (resumable: a home with a file is skipped): every official ObjectNav HM3D v2 val episode of each of the
36 homes (exam) / every HM3D-OVON val episode of each home and split (ovon), with the pin025 camera and the LIMO
floor map -> rule outcomes in <out>/<set folder>/candidates/<home>.json (OVON: <split>/<home>.json). Rules, first
failing one counted: start_snap, no_reachable_instance / other_storey, instance_visible (any pixel, 8 headings),
distance (3-15 m to the nearest usable view point on the start storey), not_seeable (Phase 8 start filter). An
episode failing ONLY not_seeable becomes an unseeable candidate (exam only).
OVON goals: an OVON goal (string id, object centre, view points) = the HM3D semantic object whose box centre is within
match_max_m (0.1 m) of the goal's centre (unmatched goals are dropped and counted); success = OVON's own view points;
an episode's `children_object_categories` count as the target too (category key "<category>|<child>..."); the
recogniser is asked with the episode's category name (`target.prompt`).

`freeze` (refuses duplicate ids and an existing frozen set; exit 3 = automatic stop):
- exam: exam_set.draw over the kept episodes (seeded, <= 10 per home, aim 100 per bin, neighbour-bin fill, within a
  bin round-robin over the categories); a bin below min_per_bin (40) after the fill -> exit 3 (STOP_sets.txt);
- unseeable: the episodes dropped ONLY by the start filter, seeded, <= 3 per home; drop counts per category;
- ovon: per split, seeded, <= 5 per home, 100 per split; ids p12o-<split>-<home5>-<category>-<id>;
- explore: the exam episodes as p12x-... (same start pose; the target is never shown to an exploration arm);
- qwen: 60 exam episodes, seeded, stratified by distance bin (proportional, largest remainder), ids unchanged.
Each set -> <folder>/episodes.json (+ episodes.sha256, fingerprint) and build_report.md; sets.json lists every
fingerprint (the chain checks them before any exam run).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

from mapmad_sim import config, objectnav, ovon  # noqa: E402
from mapmad_sim.datagen_v2 import robot_spec  # noqa: E402
from mapmad_sim.exam_set import SetRules, draw, freeze  # noqa: E402
from mapmad_sim.objects import object_box  # noqa: E402
from mapmad_sim.robot import LimoSim  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, read_config, write_text_atomic  # noqa: E402
from mapmad_sim.search_set import DEV_RULES, DevSetBuilder, search_set_ok  # noqa: E402
from mapmad_sim.target_layer import NoiseSpec  # noqa: E402

CONFIG = config.CONFIG_DIR / "p12_exam.yaml"
STOP_EXIT = 3


def val_homes() -> List[str]:
    """The 36 ObjectNav v2 val homes (exam homes) as HM3D folder names."""
    scenes = {p.name[: -len(".json.gz")] for p in (config.paths()["objectnav"] / "val" / "content").glob("*.json.gz")}
    return sorted(p.name for p in (config.paths()["hm3d_scenes"] / "val").glob("*-*") if p.name.split("-", 1)[1] in scenes)


def exam_homes(s: Dict[str, Any]) -> List[str]:
    """`homes: all` = the 36 val homes; a list of 5-digit prefixes (DRY runs: dev homes of the train split)."""
    if s["homes"] == "all":
        return val_homes()
    from mapmad_sim import home_splits

    return sorted(home_splits.by_prefix(home_splits.train_homes(s["split"]), s["homes"]))


def candidate_episodes(s: Dict[str, Any], home: str, homes: List[str]):
    """The official episodes examined in a home: all (candidates_per_home null) or a seeded sample (DRY runs)."""
    episodes = objectnav.home_episodes(s["objectnav_split"], home)
    n = s.get("candidates_per_home")
    if n and n < len(episodes):
        pick = np.random.default_rng([s["candidate_seed"], homes.index(home)]).choice(len(episodes), n, replace=False)
        episodes = [episodes[i] for i in sorted(pick)]
    return episodes


def builder(robot: LimoSim, goals, layout: RunLayout, dg: Dict[str, Any], **kw) -> DevSetBuilder:
    return DevSetBuilder(robot, goals, SetRules.from_config(layout.cfg["set_rules"]),
                         NoiseSpec.from_config(dg["target_layer"]), **kw)


def cmd_exam_candidates(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    s = layout.cfg["sets"]["exam"]
    folder = layout.base / s["folder"]
    homes = exam_homes(s)
    mine = homes[args.worker::args.workers]
    save_run_info(folder / "_runs", {**vars(args), "set": s, "homes": mine}, int(s["draw_seed"]),
                  name=f"run_info_candidates_w{args.worker}of{args.workers}.json")
    for home in mine:
        path = folder / "candidates" / f"{home}.json"
        if path.exists():
            continue
        t0 = time.time()
        with LimoSim(s["split"], home, robot_spec(dg), semantic=True, depth=True, gpu=args.gpu) as robot:
            b = builder(robot, objectnav.home_goals(s["objectnav_split"], home), layout, dg, id_prefix=s["id_prefix"],
                        set_type="exam", keep_unseeable=True)
            episodes = candidate_episodes(s, home, homes)
            results = [b.evaluate(e, s["split"]) for e in episodes]
            navmesh = robot.navmesh_sha256
        out = {"home": home, "navmesh_sha256": navmesh, "examined": len(episodes), "seeable": b._seeable,
               "seconds": round(time.time() - t0, 1), "results": results}
        write_text_atomic(path, json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({"home": home, "examined": len(episodes), "seconds": out["seconds"],
                          "kept": sum(r.get("episode") is not None for r in results),
                          "unseeable": sum(r.get("unseeable_episode") is not None for r in results)}), flush=True)


# --- OVON ---------------------------------------------------------------------------------------------------------
def ovon_goals(split: str, home: str, root: Path, robot: LimoSim, match_max_m: float
               ) -> Tuple[List[objectnav.Goal], Dict[str, Any]]:
    """The home's OVON goals with HM3D semantic ids (module docstring) and a match report."""
    goals = objectnav.home_goals(split, home, root)
    objs = [o for o in robot.sim.semantic_scene.objects if o is not None]
    ids = np.array([o.semantic_id for o in objs])
    centres = np.array([object_box(o)[0] for o in objs]) if objs else np.zeros((0, 3))
    out, unmatched, dist = [], [], []
    for g in goals:
        d = np.linalg.norm(centres - np.asarray(g.position)[None], axis=1) if len(centres) else np.array([np.inf])
        k = int(np.argmin(d))
        if d[k] <= match_max_m:
            out.append(replace(g, object_id=int(ids[k])))
            dist.append(float(d[k]))
        else:
            unmatched.append({"category": g.category, "file_id": g.object_id, "nearest_m": round(float(d[k]), 3)})
    return out, {"goals": len(goals), "matched": len(out), "unmatched": unmatched,
                 "max_match_m": round(max(dist), 4) if dist else None}


def with_children(goals: List[objectnav.Goal], episodes) -> Tuple[List[objectnav.Goal], Dict[str, str], List[Any]]:
    """Episodes with children categories get a combined category key "<cat>|<child>..." whose goals are the goals of
    all those categories; returns (goals incl. the combined copies, prompt per category key, episodes re-keyed)."""
    prompt = {g.category: g.category for g in goals}
    out_eps = []
    extra: Dict[str, List[objectnav.Goal]] = {}
    for e, children in episodes:
        if children:
            key = "|".join([e.category] + sorted(children))
            if key not in extra:
                cats = {e.category, *children}
                extra[key] = [replace(g, category=key) for g in goals if g.category in cats]
                prompt[key] = e.category
            e = replace(e, category=key)
        out_eps.append(e)
    return goals + [g for gs in extra.values() for g in gs], prompt, out_eps


def ovon_episodes(split: str, home: str, root: Path):
    """(OfficialEpisode, children categories) of one home's OVON split, in file order."""
    import gzip

    with gzip.open(objectnav.content_file(split, home, root), "rt") as f:
        data = json.load(f)
    eps = objectnav.home_episodes(split, home, root)
    return [(e, list(d.get("children_object_categories") or [])) for e, d in zip(eps, data["episodes"])]


def cmd_ovon_candidates(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    s = layout.cfg["sets"]["ovon"]
    root = config.paths()["mapmad_data"] / s["root"]
    folder = layout.base / s["folder"]
    homes = [] if s.get("dry_from_exam") else val_homes()
    mine = homes[args.worker::args.workers]
    save_run_info(folder / "_runs", {**vars(args), "set": s, "homes": mine}, int(s["draw_seed"]),
                  name=f"run_info_candidates_w{args.worker}of{args.workers}.json")
    for home in mine:
        todo = [sp for sp in s["splits"] if not (folder / "candidates" / sp / f"{home}.json").exists()
                and objectnav.has_goals(sp, home, root)]
        if not todo:
            continue
        with LimoSim("val", home, robot_spec(dg), semantic=True, depth=True, gpu=args.gpu) as robot:
            for sp in todo:
                t0 = time.time()
                goals, match = ovon_goals(sp, home, root, robot, float(s["match_max_m"]))
                goals, prompt, eps = with_children(goals, ovon_episodes(sp, home, root))
                b = builder(robot, goals, layout, dg, id_prefix=f"{s['id_prefix']}-{sp}", set_type=f"ovon_{sp}",
                            prompt_of=prompt)
                results = [b.evaluate(e, "val") for e in eps]
                out = {"home": home, "split": sp, "navmesh_sha256": robot.navmesh_sha256, "examined": len(eps),
                       "match": match, "seeable": b._seeable, "seconds": round(time.time() - t0, 1),
                       "results": results}
                write_text_atomic(folder / "candidates" / sp / f"{home}.json",
                                  json.dumps(out, indent=1, default=float) + "\n")
                print(json.dumps({"home": home, "split": sp, "examined": len(eps), "seconds": out["seconds"],
                                  "matched": f"{match['matched']}/{match['goals']}",
                                  "kept": sum(r.get("episode") is not None for r in results)}), flush=True)


# --- freeze -------------------------------------------------------------------------------------------------------
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


def capped_draw(eps: List[Dict[str, Any]], total: Optional[int], per_home: int, seed: int) -> List[Dict[str, Any]]:
    """Seeded order, <= per_home per home, at most `total` (None = all that the cap allows); sorted by id."""
    rng = np.random.default_rng(seed)
    homes: Counter = Counter()
    out = []
    for i in rng.permutation(len(eps)):
        e = eps[int(i)]
        if homes[e["home"]] < per_home and (total is None or len(out) < total):
            out.append(e)
            homes[e["home"]] += 1
    return sorted(out, key=lambda e: e["episode_id"])


def stratified(eps: List[Dict[str, Any]], total: int, seed: int, n_bins: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    """`total` episodes, seeded, per distance bin proportional to the bin's size (largest remainder; ties -> lower bin)."""
    sizes = [sum(e["distance_bin"] == b for e in eps) for b in range(n_bins)]
    exact = [total * n / max(sum(sizes), 1) for n in sizes]
    quota = [int(x) for x in exact]
    for b in sorted(range(n_bins), key=lambda b: (-(exact[b] - quota[b]), b))[:total - sum(quota)]:
        quota[b] += 1
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_bins):
        pool = sorted((e for e in eps if e["distance_bin"] == b), key=lambda e: e["episode_id"])
        out += [pool[int(i)] for i in rng.permutation(len(pool))[:quota[b]]]
    return sorted(out, key=lambda e: e["episode_id"]), quota


def read_candidates(files: List[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    results, navmesh = [], {}
    for f in files:
        doc = json.loads(f.read_text())
        navmesh[doc["home"]] = doc["navmesh_sha256"]
        results += [dict(r, home=doc["home"]) for r in doc["results"]]
    return results, navmesh


def write_report(folder: Path, lines: List[str]) -> None:
    (folder / "build_report.md").write_text("\n".join(lines) + "\n")


def cmd_freeze(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    cfg, base = layout.cfg, layout.base
    sets = cfg["sets"]
    bins = cfg["set_rules"]["bins_m"]
    n_bins = len(bins) - 1
    fps: Dict[str, str] = {}
    # --- exam (+ unseeable) ---
    s = sets["exam"]
    folder = base / s["folder"]
    files = [folder / "candidates" / f"{h}.json" for h in exam_homes(s)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"exam candidates missing for {len(missing)} homes: {missing[:5]}")
    results, navmesh = read_candidates(files)
    kept = [r["episode"] for r in results if r.get("episode") is not None]
    chosen, rep = draw(kept, int(s["total"]), int(s["per_home"]), n_bins, int(s["draw_seed"]), s["draw_order"])
    short = [b for b in range(n_bins) if rep["chosen_per_bin"][b] < int(s["min_per_bin"])]
    if short:
        reason = (f"exam set: bin(s) {[f'{bins[b]:g}-{bins[b + 1]:g} m' for b in short]} below {s['min_per_bin']} after "
                  f"the fill (chosen per bin {rep['chosen_per_bin']}, kept per bin {rep['kept_per_bin']})")
        write_text_atomic(base / "STOP_sets.txt", reason + "\n")
        print(reason, flush=True)
        raise SystemExit(STOP_EXIT)
    for name, eps in (("exam", chosen),):
        bad = search_set_ok(eps)
        if bad:
            write_text_atomic(base / "STOP_sets.txt", f"{name} set refused: {bad}\n")
            raise SystemExit(STOP_EXIT)
    fps["exam"] = freeze(folder / "episodes.json", chosen, {"set": "exam", "rules": cfg["set_rules"], "set_config": s,
                                                           "navmesh_sha256": navmesh, "draw": rep})
    lines = ["# Phase 12 main exam set: build report", "",
             f"Fingerprint `{fps['exam']}`; {len(chosen)} episodes (asked {s['total']}, <= {s['per_home']} per home, "
             f"draw seed {s['draw_seed']}, order {s['draw_order']}); official ObjectNav v2 {s['objectnav_split']} episodes "
             f"examined: {len(results)}.", "", "## Drops (first failing rule)", ""] + drop_table(results, list(DEV_RULES))
    lines += ["", "## Distance bins", "", "| bin | kept | chosen |", "|---|---|---|"]
    for b in range(n_bins):
        lines.append(f"| {bins[b]:g}-{bins[b + 1]:g} m | {rep['kept_per_bin'][b]} | {rep['chosen_per_bin'][b]} |")
    short_bins = [b for b in range(n_bins) if rep["kept_per_bin"][b] < rep["first_quota"][b]]
    lines += ["", f"Moved quota: {rep['moved'] or 'none'}; missing: {rep['missing']}; bins that took all their eligible "
              f"episodes (bin fallback, logged): {[f'{bins[b]:g}-{bins[b + 1]:g} m' for b in short_bins] or 'none'}.",
              "", "| category | kept | chosen |", "|---|---|---|"]
    kc, cc = Counter(e["target"]["category"] for e in kept), Counter(e["target"]["category"] for e in chosen)
    lines += [f"| {c} | {kc[c]} | {cc[c]} |" for c in sorted(kc)]
    lines += ["", "| home | chosen |", "|---|---|"] + [f"| {h} | {n} |" for h, n in sorted(rep["chosen_per_home"].items())]
    write_report(folder, lines)
    # unseeable
    u = sets["unseeable"]
    ufolder = base / u["folder"]
    useeable = [dict(r["unseeable_episode"], episode_id=r["unseeable_episode"]["episode_id"].replace("p12e-", "p12u-", 1),
                     type="unseeable") for r in results if r.get("unseeable_episode") is not None]
    uchosen = capped_draw(useeable, None, int(u["per_home"]), int(u["draw_seed"]))
    bad = search_set_ok(uchosen)
    if bad:
        write_text_atomic(base / "STOP_sets.txt", f"unseeable set refused: {bad}\n")
        raise SystemExit(STOP_EXIT)
    fps["unseeable"] = freeze(ufolder / "episodes.json", uchosen, {"set": "unseeable", "set_config": u,
                                                                  "navmesh_sha256": navmesh})
    dropped = Counter(e["target"]["category"] for e in useeable)
    write_report(ufolder, ["# Phase 12 unseeable set (report only)", "",
                           f"Fingerprint `{fps['unseeable']}`; {len(uchosen)} episodes (<= {u['per_home']} per home, seed "
                           f"{u['draw_seed']}) of {len(useeable)} eligible episodes dropped ONLY by the start filter.", "",
                           "| category | dropped by the start filter only | chosen |", "|---|---|---|"]
                 + [f"| {c} | {dropped[c]} | {sum(e['target']['category'] == c for e in uchosen)} |" for c in sorted(dropped)])
    # explore
    x = sets["explore"]
    xeps = [dict(e, episode_id=e["episode_id"].replace("p12e-", f"{x['id_prefix']}-", 1), type="explore") for e in chosen]
    fps["explore"] = freeze(base / x["folder"] / "episodes.json", xeps, {"set": "explore", "from": "exam",
                                                                         "exam_fingerprint": fps["exam"]})
    # qwen subset
    q = sets["qwen"]
    qeps, quota = stratified(chosen, int(q["total"]), int(q["draw_seed"]), n_bins)
    fps["qwen"] = freeze(base / q["folder"] / "episodes.json", qeps, {"set": "qwen", "from": "exam", "per_bin": quota,
                                                                      "exam_fingerprint": fps["exam"]})
    # --- ovon ---
    o = sets["ovon"]
    ofolder = base / o["folder"]
    olines = ["# Phase 12 OVON set: build report", ""]
    ovon_all: List[Dict[str, Any]] = []
    if o.get("dry_from_exam"):  # DRY runs only: OVON exists for the val homes alone; stand-ins from the DRY exam set
        ovon_all = [dict(e, episode_id=e["episode_id"].replace("p12e-", f"{o['id_prefix']}-dry-", 1), type="ovon_dry",
                         target=dict(e["target"], prompt=e["target"]["category"]))
                    for e in chosen[:int(o["dry_from_exam"])]]
    for sp in ([] if o.get("dry_from_exam") else o["splits"]):
        files = sorted((ofolder / "candidates" / sp).glob("*.json"))
        res, nav = read_candidates(files)
        okept = [r["episode"] for r in res if r.get("episode") is not None]
        och = capped_draw(okept, int(o["per_split"]), int(o["per_home"]), int(o["draw_seed"]))
        ovon_all += och
        unmatched = sum(len(json.loads(f.read_text())["match"]["unmatched"]) for f in files)
        olines += [f"## {sp}", "", f"{len(files)} homes, {len(res)} episodes examined, {len(okept)} kept, {len(och)} "
                   f"chosen (<= {o['per_home']} per home); OVON goals without an HM3D object within {o['match_max_m']} m: "
                   f"{unmatched}.", ""] + drop_table(res, list(DEV_RULES)) + [""]
    bad = search_set_ok(ovon_all)
    if bad:
        write_text_atomic(base / "STOP_sets.txt", f"ovon set refused: {bad}\n")
        raise SystemExit(STOP_EXIT)
    fps["ovon"] = freeze(ofolder / "episodes.json", ovon_all, {"set": "ovon", "set_config": o, "rules": cfg["set_rules"]})
    write_report(ofolder, [f"Fingerprint `{fps['ovon']}`; {len(ovon_all)} episodes.", ""] + olines)
    write_text_atomic(base / "sets.json", json.dumps({"fingerprints": fps, "time": time.strftime("%Y-%m-%d %H:%M:%S")},
                                                     indent=1) + "\n")
    save_run_info(base / "exam_set" / "_runs", {**vars(args), "fingerprints": fps}, int(s["draw_seed"]),
                  name="run_info_freeze.json")
    print(json.dumps(fps, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["exam-candidates", "ovon-candidates", "freeze"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    dg = read_config(config.CONFIG_DIR / layout.cfg["datagen"])
    {"exam-candidates": cmd_exam_candidates, "ovon-candidates": cmd_ovon_candidates,
     "freeze": cmd_freeze}[args.cmd](args, layout, dg)


if __name__ == "__main__":
    main()
