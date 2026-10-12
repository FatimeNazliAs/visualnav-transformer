"""Build the Phase 5 episode sets (configs/p5_exam.yaml `sets`; rules in mapmad_sim/exam_set.py). Inside
naz_mapmad_habitat, from /app/visualnav-transformer:

    python mapmad/scripts/make_p5_sets.py candidates --set dev --gpu 0 --worker 0 --workers 2   # per worker
    python mapmad/scripts/make_p5_sets.py freeze --set dev                                       # once, after all
    python mapmad/scripts/make_p5_sets.py rephoto --set dev --out dev/set_1a                     # new photo rule

`candidates`: every examined official episode of this worker's homes (worker k of n takes homes k, k+n, ... of
the sorted list) gets its rule outcomes in <set folder>/candidates/<home>.json; kept episodes also their goal photo
<set folder>/goal_photos/<episode_id>.png. Resumable: a home with a candidates file is skipped.
`freeze`: the seeded draw over all kept episodes -> <set folder>/episodes.json (+ episodes.sha256, fingerprint),
build_report.json / build_report.md (drops per rule, per category, per home; kept and chosen per bin), and
contact sheets of 20 random goal photos (seeded). Refuses to overwrite a frozen set.
`rephoto`: the frozen set's episodes with only the goal photo re-chosen by the config's current rule
(set_rules.goal_photo_box_m; Naz's 1a at S-dev) -> a new frozen set in <outputs>/<out_dir>/<--out> (+ photos, sheet,
rephoto_report.md). Used for the dev runs of arms B and F with the 1a photos (report only).
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mapmad_sim import config, objectnav  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.exam_set import RULES, HomeSetBuilder, Instance, SetRules, draw, freeze, rules_dict  # noqa: E402
from mapmad_sim.objectnav import Goal  # noqa: E402
from mapmad_sim.robot import LimoSim, RobotSpec  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout  # noqa: E402

CONFIG = config.CONFIG_DIR / "p5_exam.yaml"
PHOTO_SHEET = 20
PHOTO_SHEET_SEED = 7


def set_homes(s: Dict[str, Any]) -> List[str]:
    """The set's HM3D home folders, sorted."""
    homes = config.labelled_homes(s["split"])
    if s["homes"] == "all":
        return [h for h in homes if objectnav.has_goals(s["objectnav_split"], h)]
    return sorted(h for h in homes if h[:5] in set(s["homes"]))


def candidates_for_home(s: Dict[str, Any], rules: SetRules, home: str, home_index: int, folder: Path, gpu: int,
                        set_name: str) -> Dict[str, Any]:
    """Evaluate the home's official episodes (all, or a seeded sample) and save outcomes + goal photos."""
    episodes = objectnav.home_episodes(s["objectnav_split"], home)
    n = s["candidates_per_home"]
    if n and n < len(episodes):
        pick = np.random.default_rng([s["candidate_seed"], home_index]).choice(len(episodes), n, replace=False)
        episodes = [episodes[i] for i in sorted(pick)]
    t0 = time.time()
    rng = np.random.default_rng([s["candidate_seed"], home_index, 1])
    with LimoSim(s["split"], home, RobotSpec.from_config(config.robot()), semantic=True, gpu=gpu) as robot:
        builder = HomeSetBuilder(robot, objectnav.home_goals(s["objectnav_split"], home), rules, rng)
        results = [builder.evaluate(e, set_name, s["split"]) for e in episodes]
        # goal photos of the kept episodes (re-rendered at the stored pose, exactly what the runner shows)
        for r in results:
            if r.get("episode") is not None:
                save_photo(robot, r["episode"], folder)
        navmesh = robot.navmesh_sha256
    out = {"home": home, "home_index": home_index, "navmesh_sha256": navmesh, "examined": len(episodes),
           "seconds": round(time.time() - t0, 1), "results": results}
    path = folder / "candidates" / f"{home}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def cmd_candidates(args, layout: RunLayout) -> None:
    s = layout.cfg["sets"][args.set]
    rules = SetRules.from_config(layout.cfg["set_rules"])
    folder = layout.base / s["folder"]
    homes = set_homes(s)
    mine = homes[args.worker::args.workers]
    save_run_info(folder / "_runs", {**vars(args), "set": s, "set_rules": rules_dict(rules), "homes": mine},
                  int(s["candidate_seed"]), name=f"run_info_candidates_w{args.worker}of{args.workers}.json")
    for home in mine:
        if (folder / "candidates" / f"{home}.json").exists():
            continue
        r = candidates_for_home(s, rules, home, homes.index(home), folder, args.gpu, args.set)
        kept = sum(x.get("episode") is not None for x in r["results"])
        print(json.dumps({"home": home, "examined": r["examined"], "kept": kept, "seconds": r["seconds"]}), flush=True)


def drop_tables(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """First failing rule (exclusive) and every failing rule (not exclusive), overall / per category / per home."""
    first, every = Counter(), Counter()
    by_cat: Dict[str, Counter] = defaultdict(Counter)
    by_home: Dict[str, Counter] = defaultdict(Counter)
    for r in results:
        key = r["first_fail"] or "kept"
        first[key] += 1
        by_cat[r["category"]][key] += 1
        by_home[r["home"]][key] += 1
        every.update(r["fails"])
    return {"examined": len(results), "first_fail": dict(first), "any_fail": dict(every),
            "per_category": {k: dict(v) for k, v in sorted(by_cat.items())},
            "per_home": {k: dict(v) for k, v in sorted(by_home.items())}}


def report_md(name: str, drops: Dict[str, Any], draw_report: Dict[str, Any], chosen: List[Dict[str, Any]],
              fp: str, bins: List[float]) -> str:
    cols = list(RULES) + ["kept"]
    lines = [f"# Phase 5 {name} set: build report", "", f"Fingerprint `{fp}`; {len(chosen)} episodes "
             f"(asked {draw_report['total_asked']}, <= {draw_report['per_home_cap']} per home, draw seed "
             f"{draw_report['seed']}).", "", "## Drops (first failing rule, in rule order)", "",
             "| | examined | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols),
             f"| all | {drops['examined']} | " + " | ".join(str(drops["first_fail"].get(c, 0)) for c in cols) + " |"]
    for cat, c in drops["per_category"].items():
        lines.append(f"| {cat} | {sum(c.values())} | " + " | ".join(str(c.get(k, 0)) for k in cols) + " |")
    lines += ["", "Every rule an episode fails (not exclusive): " +
              ", ".join(f"{k} {drops['any_fail'].get(k, 0)}" for k in RULES) +
              f". Other instance visible (block item 6, separately): {drops['any_fail'].get('other_instance_visible', 0)}.",
              "", "## Distance bins", "", "| bin | kept | first quota | final quota | chosen |", "|---|---|---|---|---|"]
    for b in range(len(bins) - 1):
        lines.append(f"| {bins[b]:g}-{bins[b + 1]:g} m | {draw_report['kept_per_bin'][b]} | {draw_report['first_quota'][b]} | "
                     f"{draw_report['final_quota'][b]} | {draw_report['chosen_per_bin'][b]} |")
    lines += ["", f"Moved quota: {draw_report['moved'] or 'none'}; missing: {draw_report['missing']}.", "",
              "## Chosen per category and home", "",
              "| category | episodes |", "|---|---|"]
    for cat, n in sorted(Counter(e["target"]["category"] for e in chosen).items()):
        lines.append(f"| {cat} | {n} |")
    lines += ["", "| home | examined | kept | chosen |", "|---|---|---|---|"]
    per_home_chosen = Counter(e["home"] for e in chosen)
    for home, c in drops["per_home"].items():
        lines.append(f"| {home} | {sum(c.values())} | {c.get('kept', 0)} | {per_home_chosen.get(home, 0)} |")
    geo = np.array([e["start_geodesic_m"] for e in chosen])
    lines += ["", f"Walking distance start -> nearest view point: mean {geo.mean():.2f} m, median {np.median(geo):.2f} m, "
              f"min {geo.min():.2f}, max {geo.max():.2f}. Goal photos: target share mean "
              f"{np.mean([e['goal_photo_share'] for e in chosen]):.3f}; photo from the nearest view point in "
              f"{sum(e['goal_photo_rank'] == 0 for e in chosen)} of {len(chosen)}. Target-point fallback (no Phase 1 point "
              f"reachable): {sum(e['target']['point_fallback'] for e in chosen)}."]
    return "\n".join(lines) + "\n"


def photo_sheet(folder: Path, chosen: List[Dict[str, Any]], out: Path) -> None:
    """20 random goal photos (seeded) of the chosen episodes, 5 x 4, captioned with category and episode id."""
    rng = np.random.default_rng(PHOTO_SHEET_SEED)
    pick = [chosen[i] for i in sorted(rng.choice(len(chosen), min(PHOTO_SHEET, len(chosen)), replace=False))]
    tiles = []
    for e in pick:
        img = cv2.imread(str(folder / "goal_photos" / f"{e['episode_id']}.png"))
        cv2.putText(img, f"{e['target']['category']} {e['episode_id']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{e['target']['category']} {e['episode_id']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"share {e['goal_photo_share']:.2f}", (4, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255),
                    1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % 5:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    cv2.imwrite(str(out), np.vstack(rows))


def cmd_freeze(args, layout: RunLayout) -> None:
    s = layout.cfg["sets"][args.set]
    folder = layout.base / s["folder"]
    homes = set_homes(s)
    files = [folder / "candidates" / f"{h}.json" for h in homes]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"candidates missing for {len(missing)} homes: {missing[:5]}")
    results, navmesh = [], {}
    for f in files:
        doc = json.loads(f.read_text())
        navmesh[doc["home"]] = doc["navmesh_sha256"]
        results += [dict(r, home=doc["home"]) for r in doc["results"]]
    kept = [r["episode"] for r in results if r.get("episode") is not None]
    chosen, rep = draw(kept, int(s["total"]), int(s["per_home"]), len(layout.cfg["set_rules"]["bins_m"]) - 1,
                       int(s["draw_seed"]))
    drops = drop_tables(results)
    fp = freeze(folder / "episodes.json", chosen,
                {"set": args.set, "rules": layout.cfg["set_rules"], "set_config": s, "navmesh_sha256": navmesh,
                 "draw": rep})
    (folder / "build_report.json").write_text(json.dumps({"fingerprint": fp, "drops": drops, "draw": rep,
                                                           "navmesh_sha256": navmesh}, indent=1) + "\n")
    (folder / "build_report.md").write_text(report_md(args.set, drops, rep, chosen, fp, layout.cfg["set_rules"]["bins_m"]))
    photo_sheet(folder, chosen, folder / "goal_photos_sheet.png")
    save_run_info(folder / "_runs", {**vars(args), "set": s, "fingerprint": fp}, int(s["draw_seed"]),
                  name="run_info_freeze.json")
    print((folder / "build_report.md").read_text())


def save_photo(robot: LimoSim, e: Dict[str, Any], folder: Path) -> None:
    """Render the episode's goal photo at its stored pose (what the runner shows) into <folder>/goal_photos."""
    (folder / "goal_photos").mkdir(parents=True, exist_ok=True)
    robot.place(np.array(e["goal_photo_position"]), e["goal_photo_yaw"])
    cv2.imwrite(str(folder / "goal_photos" / f"{e['episode_id']}.png"), cv2.cvtColor(robot.observe()["rgb"], cv2.COLOR_RGB2BGR))


def cmd_rephoto(args, layout: RunLayout) -> None:
    s = layout.cfg["sets"][args.set]
    rules = SetRules.from_config(layout.cfg["set_rules"])
    src = layout.base / s["folder"] / "episodes.json"
    dst = layout.base / args.out
    episodes, fp = load_episodes(src)
    new: Dict[str, Dict[str, Any]] = {}
    for home in sorted({e["home"] for e in episodes}):
        mine = [e for e in episodes if e["home"] == home]
        with LimoSim(s["split"], home, RobotSpec.from_config(config.robot()), semantic=True, gpu=args.gpu) as robot:
            builder = HomeSetBuilder(robot, [], rules, np.random.default_rng(0))
            for e in mine:
                t = e["target"]
                vps = [np.array(v, dtype=np.float64) for v in t["view_points"]]
                inst = Instance(Goal(t["object_id"], t["category"], t["box_center"], t["view_points"]),
                                np.array(t["box_center"]), np.array(t["box_size"]), vps, t["view_point_snap_m"])
                photo = builder.goal_photo(inst)
                if photo is None:  # the old photo's view point is among them, so this cannot happen
                    raise RuntimeError(f"{e['episode_id']}: no goal photo under the new rule")
                new[e["episode_id"]] = dict(e, goal_photo_position=photo["position"], goal_photo_yaw=photo["yaw"],
                                            goal_photo_share=photo["share"], goal_photo_rank=photo["rank"],
                                            goal_photo_black_share=photo["black_share"],
                                            goal_photo_box_distance_m=photo["box_distance_m"])
                save_photo(robot, new[e["episode_id"]], dst)
        print(json.dumps({"home": home, "episodes": len(mine)}), flush=True)
    out = [new[e["episode_id"]] for e in episodes]
    nfp = freeze(dst / "episodes.json", out, {"set": f"{args.set}_rephoto", "rules": layout.cfg["set_rules"],
                                              "derived_from": {"file": str(src), "fingerprint": fp},
                                              "change": "goal photo fields re-chosen with set_rules.goal_photo_box_m"})

    def stats(eps: List[Dict[str, Any]], key: str) -> str:
        v = np.array([e[key] for e in eps], dtype=float)
        return f"mean {v.mean():.3f}, median {np.median(v):.3f}, min {v.min():.3f}, max {v.max():.3f}"

    lines = [f"# {args.set} set, goal photos re-chosen", "",
             f"From `{src}` (fingerprint `{fp}`) -> `{dst / 'episodes.json'}` (fingerprint `{nfp}`); only the goal-photo "
             f"fields changed. Rule: view point box distance closest to {rules.goal_photo_box_m} m first, >= "
             f"{rules.goal_photo_min_share:.0%} target pixels.", "", "| | before | after |", "|---|---|---|",
             f"| target share | {stats(episodes, 'goal_photo_share')} | {stats(out, 'goal_photo_share')} |",
             f"| view point -> box (m) | {stats(episodes, 'goal_photo_box_distance_m')} | "
             f"{stats(out, 'goal_photo_box_distance_m')} |",
             f"| first view point in the order used | {sum(e['goal_photo_rank'] == 0 for e in episodes)} of "
             f"{len(episodes)} | {sum(e['goal_photo_rank'] == 0 for e in out)} of {len(out)} |",
             f"| photo changed | | {sum(a['goal_photo_position'] != b['goal_photo_position'] for a, b in zip(episodes, out))} |"]
    (dst / "rephoto_report.md").write_text("\n".join(lines) + "\n")
    photo_sheet(dst, out, dst / "goal_photos_sheet.png")
    save_run_info(dst / "_runs", {**vars(args), "set": s, "fingerprint": nfp, "derived_from": fp}, 0,
                  name="run_info_rephoto.json")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["candidates", "freeze", "rephoto"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--set", choices=["dev", "exam"], required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", help="rephoto: the new set folder below <outputs>/<out_dir>")
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    {"candidates": cmd_candidates, "freeze": cmd_freeze, "rephoto": cmd_rephoto}[args.cmd](args, layout)


if __name__ == "__main__":
    main()
