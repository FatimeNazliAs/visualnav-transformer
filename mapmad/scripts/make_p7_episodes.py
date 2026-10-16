"""Phase 7.5: the frozen episode set of the camera sweep (block items 5-6).

    python mapmad/scripts/make_p7_episodes.py new        # 60 new in-view episodes (Phase 1 rules, seed 1001, p7x-iv-*)
    python mapmad/scripts/make_p7_episodes.py assemble   # + Phase 1's 60 in-view and 30 out-of-view -> exam format
    python mapmad/scripts/make_p7_episodes.py freeze     # drop episodes invalid on the tallest camera's floor map, sha256
    python mapmad/scripts/make_p7_episodes.py shares     # report: target pixel share at the start / in the goal photo per camera

All in configs/p7_sweep.yaml `episodes`; outputs in <outputs>/p7_camera/episodes/. Dev homes only.

new: Phase 1's EpisodeBuilder (in view: >= 1% target pixels at LIMO's pinhole 0.18 m camera, 3-8 m walking distance,
  <= 2 per category per home) with seed 1001. Each home is asked for per_home + 1; up to per_home are kept per home and
  a shortfall (a home with too few categories) is filled with the extra episodes of the other homes in home order.
  A new episode whose (home, object, start) repeats an earlier one is dropped.
assemble: every episode gets (Phase 5 rules, LIMO pinhole 0.18 m, LIMO floor map): the target's usable ObjectNav view
  points (snapped <= 0.2 m, real floor; primary metric), the goal-photo pose (view point closest to 1.0 m from the box,
  first one with >= 5% target pixels), and the floor-map hash of every floor-map height the sweep uses
  (`navmesh_sha256_by_floor_map`, "0.55" = LIMO's, "1.00" = for the 0.95 m camera).
freeze: an episode is invalid on a floor map if its start is not navigable on it, no usable view point is navigable
  and reachable from the start, or the target point is unreachable; episodes invalid on the tallest floor map are
  dropped from every setting (block item 5). Episodes whose start is already a primary success (walking distance to a
  view point <= drop_start_within_m, Naz 4b) are dropped. Ids must be unique; frozen with its sha256 fingerprint.
"""

import argparse
import dataclasses
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mapmad_sim import config, episodes, objectnav
from mapmad_sim.camera_setting import CameraSetting
from mapmad_sim.exam_set import HomeSetBuilder, SetRules
from mapmad_sim.robot import LimoSim, RobotSpec
from mapmad_sim.run_info import save_run_info
from mapmad_sim.run_layout import RunLayout

CONFIG = config.CONFIG_DIR / "p7_sweep.yaml"


def layout_of(path: Path) -> RunLayout:
    return RunLayout.load(path)


def folder(layout: RunLayout) -> Path:
    return layout.episodes_dir


def limo_spec() -> RobotSpec:
    return RobotSpec.from_config(config.robot())


def floor_map_spec(height_m: float) -> RobotSpec:
    """LIMO with its floor map built for this agent height (the camera stays LIMO's)."""
    spec = limo_spec()
    nav = dict(spec.navmesh)
    nav["agent_height_m"] = height_m
    return dataclasses.replace(spec, navmesh=tuple(sorted(nav.items())))


# --- new -----------------------------------------------------------------------------------------------------------

def pair_key(e: Dict[str, Any]) -> Tuple[str, int, Tuple[float, ...]]:
    """(home, target object, start position to 1 cm): two episodes with the same key repeat a start/target pair."""
    return e["home"], int(e["target"]["object_id"]), tuple(round(float(v), 2) for v in e["start_position"])


def cmd_new(args: argparse.Namespace) -> None:
    layout = layout_of(args.config)
    cfg = layout.cfg["episodes"]["new"]
    out = folder(layout) / "new_iv.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; --force to rebuild")
    p1cfg = RunLayout.load(config.CONFIG_DIR / cfg["rules_from"]).cfg["episodes"]
    p1cfg = {**p1cfg, "id_prefix": cfg["id_prefix"], "seed": cfg["seed"]}
    rng = np.random.default_rng(cfg["seed"])
    per_home, made = int(cfg["per_home"]), {}
    for home in cfg["homes"]:
        goals = objectnav.home_goals(p1cfg["objectnav_split"], home)
        with LimoSim(p1cfg["split"], home, limo_spec(), semantic=True, depth=True, gpu=args.gpu, seed=cfg["seed"]) as robot:
            builder = episodes.EpisodeBuilder(robot, p1cfg, rng)
            made[home] = [episodes.episode_dict(e) for e in builder.episodes("in_view", per_home + 1, goals, p1cfg["split"])]
        print(f"{home}: {len(made[home])} in-view", flush=True)
    seen = {pair_key(e) for e in phase1_episodes(layout)}
    for home in cfg["homes"]:
        fresh = []
        for e in made[home]:
            if pair_key(e) not in seen:
                fresh.append(e)
                seen.add(pair_key(e))
        made[home] = fresh
    chosen =[e for home in cfg["homes"] for e in made[home][:per_home]]
    extras = [e for home in cfg["homes"] for e in made[home][per_home:]]
    while len(chosen) < cfg["total"] and extras:
        chosen.append(extras.pop(0))
    if len(chosen) < cfg["total"]:
        raise SystemExit(f"only {len(chosen)} new in-view episodes (need {cfg['total']})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"created": now(), "config": cfg, "per_home": dict(Counter(e["home"] for e in chosen)),
                               "episodes": chosen}, indent=1) + "\n")
    print(f"{len(chosen)} new in-view episodes -> {out}")


# --- assemble ------------------------------------------------------------------------------------------------------

def phase1_episodes(layout: RunLayout) -> List[Dict[str, Any]]:
    """The Phase 1 episodes the sweep reuses (frozen files, fingerprints checked)."""
    out = []
    for src in layout.cfg["episodes"]["reuse"]:
        eps, fp = episodes.load_episodes(layout.resolve(src["file"]))
        if fp != src["fingerprint"]:
            raise SystemExit(f"{src['file']}: fingerprint {fp} != {src['fingerprint']}")
        out += [e for e in eps if e["type"] in src["types"]]
    return out


def convert(e: Dict[str, Any], builder: HomeSetBuilder, goals: Dict[int, Any], robot: LimoSim,
            hashes: Dict[str, str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """One Phase 1 episode in the exam format (view points, Phase 5 goal photo, floor-map hashes), or (None, why)."""
    goal = goals.get(e["target"]["object_id"])
    if goal is None:
        return None, "object not in ObjectNav goals"
    inst = builder.instance(goal)
    if inst is None or not inst.view_points:
        return None, "no usable view point"
    photo = builder.goal_photo(inst)
    if photo is None:
        return None, "no goal photo >= 5%"
    vp = robot.geodesic_to_any(e["start_position"], inst.view_points)
    t = dict(e["target"])
    t["view_points"] = [episodes.rounded(p) for p in inst.view_points]
    out = {k: e[k] for k in ("episode_id", "type", "split", "home", "start_position", "start_yaw", "start_geodesic_m",
                             "start_target_share")}
    out.update({"target": t, "goal_photo_position": photo["position"], "goal_photo_yaw": photo["yaw"],
                "goal_photo_share": photo["share"], "goal_photo_rank": photo["rank"],
                "goal_photo_box_distance_m": photo["box_distance_m"], "phase1_goal_photo_position": e["goal_photo_position"],
                "phase1_goal_photo_yaw": e["goal_photo_yaw"], "start_vp_geodesic_m": round(vp, 4) if math.isfinite(vp) else None,
                "distance_bin": None, "navmesh_sha256": hashes["0.55"], "navmesh_sha256_by_floor_map": dict(hashes)})
    if e.get("navmesh_sha256") != hashes["0.55"]:
        return None, "Phase 1 floor-map hash differs from today's LIMO floor map"
    return out, ""


def floor_heights(layout: RunLayout) -> List[float]:
    """Agent heights of every floor map the sweep's cameras use (LIMO's first)."""
    limo = float(config.robot()["navmesh"]["agent_height_m"])
    cams = [CameraSetting.from_config(n, c) for n, c in layout.cfg["episodes"]["floor_map_cameras"].items()]
    return sorted({limo, *(c.floor_map_height(limo) for c in cams)})


def cmd_assemble(args: argparse.Namespace) -> None:
    layout = layout_of(args.config)
    ecfg = layout.cfg["episodes"]
    src = phase1_episodes(layout) + json.loads((folder(layout) / "new_iv.json").read_text())["episodes"]
    rules = SetRules.from_config(RunLayout.load(config.CONFIG_DIR / ecfg["set_rules_from"]).cfg["set_rules"])
    heights = floor_heights(layout)
    kept, dropped = [], {}
    for home in sorted({e["home"] for e in src}):
        hashes = {}
        for h in heights:
            with LimoSim("train", home, floor_map_spec(h), gpu=args.gpu) as r:
                hashes[f"{h:.2f}"] = r.navmesh_sha256
        goals = {g.object_id: g for g in objectnav.home_goals("train", home)}
        with LimoSim("train", home, limo_spec(), semantic=True, gpu=args.gpu, seed=ecfg["new"]["seed"]) as robot:
            builder = HomeSetBuilder(robot, list(goals.values()), rules, np.random.default_rng(0))
            for e in (e for e in src if e["home"] == home):
                out, why = convert(e, builder, goals, robot, hashes)
                if out is None:
                    dropped[e["episode_id"]] = why
                else:
                    kept.append(out)
        print(f"{home}: {sum(e['home'] == home for e in kept)} kept, floor maps {list(hashes)}", flush=True)
    kept.sort(key=lambda e: (e["type"] != "in_view", e["home"], e["episode_id"]))
    (folder(layout) / "assembled.json").write_text(json.dumps({"created": now(), "dropped": dropped, "episodes": kept},
                                                              indent=1) + "\n")
    print(f"{len(kept)} assembled, {len(dropped)} dropped: {dropped}")


# --- freeze --------------------------------------------------------------------------------------------------------

def invalid_reason(robot: LimoSim, e: Dict[str, Any]) -> str:
    """Why the episode cannot run on this robot's floor map ('' = valid)."""
    pf = robot.pathfinder
    if not pf.is_navigable(np.asarray(e["start_position"], np.float32)):
        return "start not navigable"
    vps = [v for v in e["target"]["view_points"] if pf.is_navigable(np.asarray(v, np.float32))]
    if not vps or not math.isfinite(robot.geodesic_to_any(e["start_position"], vps)):
        return "no navigable, reachable view point"
    if not math.isfinite(robot.geodesic(e["start_position"], e["target"]["point"])):
        return "target point unreachable"
    return ""


def cmd_freeze(args: argparse.Namespace) -> None:
    layout = layout_of(args.config)
    out = folder(layout)
    if (out / "episodes.json").exists() and not args.force:
        raise SystemExit(f"{out / 'episodes.json'} exists (frozen); --force to replace it")
    eps = json.loads((out / "assembled.json").read_text())["episodes"]
    ids = [e["episode_id"] for e in eps]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate episode ids: {[i for i, n in Counter(ids).items() if n > 1]}")
    tallest = max(floor_heights(layout))
    invalid: Dict[str, Dict[str, str]] = {}
    for h in sorted(set(floor_heights(layout))):
        for home in sorted({e["home"] for e in eps}):
            with LimoSim("train", home, floor_map_spec(h), gpu=args.gpu) as robot:
                for e in (e for e in eps if e["home"] == home):
                    assert robot.navmesh_sha256 == e["navmesh_sha256_by_floor_map"][f"{h:.2f}"], "floor map changed"
                    why = invalid_reason(robot, e)
                    if why:
                        invalid.setdefault(f"{h:.2f}", {})[e["episode_id"]] = why
    drop = set(invalid.get(f"{tallest:.2f}", {}))
    near = float(layout.cfg["episodes"].get("drop_start_within_m", -1.0))
    at_start = sorted(e["episode_id"] for e in eps if e["episode_id"] not in drop and e["start_vp_geodesic_m"] is not None
                      and e["start_vp_geodesic_m"] <= near)
    final = [e for e in eps if e["episode_id"] not in drop and e["episode_id"] not in set(at_start)]
    fp = episodes.fingerprint(final)
    doc = {"created": now(), "config": layout.cfg["episodes"], "floor_map_heights": floor_heights(layout),
           "invalid_by_floor_map": invalid, "dropped_tallest": sorted(drop), "dropped_start_success": at_start,
           "fingerprint": fp, "episodes": final}
    (out / "episodes.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "episodes.sha256").write_text(f"{fp}  episodes (canonical JSON of the 'episodes' list)\n")
    (out / "summary.md").write_text(summary(final, doc))
    save_run_info(out, {**vars(args), "config": str(args.config), "episodes": layout.cfg["episodes"]}, 0)
    print(summary(final, doc))


def summary(final: List[Dict[str, Any]], doc: Dict[str, Any]) -> str:
    lines = ["# Phase 7 sweep episodes", "", f"fingerprint (sha256): `{doc['fingerprint']}`", "",
             f"floor maps (agent height m): {doc['floor_map_heights']}", "",
             f"invalid per floor map: { {k: len(v) for k, v in doc['invalid_by_floor_map'].items()} }", "",
             f"dropped (invalid on the tallest floor map, from every setting): {len(doc['dropped_tallest'])} "
             f"{doc['dropped_tallest']}", "",
             f"dropped (start already within {near_m(doc)} m of a view point): {len(doc['dropped_start_success'])} "
             f"{doc['dropped_start_success']}", ""]
    for kind in ("in_view", "out_of_view"):
        sel = [e for e in final if e["type"] == kind]
        geo = np.array([e["start_vp_geodesic_m"] for e in sel], dtype=float)
        lines += [f"## {kind}: {len(sel)} episodes", "", f"- per home: {dict(Counter(e['home'] for e in sel))}",
                  f"- per category: {dict(Counter(e['target']['category'] for e in sel))}",
                  f"- start walking distance to nearest view point m: min {np.nanmin(geo):.2f}, median "
                  f"{np.nanmedian(geo):.2f}, max {np.nanmax(geo):.2f}",
                  f"- goal photo share (pinhole 0.18): min {min(e['goal_photo_share'] for e in sel):.3f}", ""]
    return "\n".join(lines)


def near_m(doc: Dict[str, Any]) -> float:
    return doc["config"].get("drop_start_within_m")


def cmd_shares(args: argparse.Namespace) -> None:
    """Target pixel share at the start (start heading) and in the goal photo, for every camera of the config (block
    item 6: reported per setting; the frozen set is not changed)."""
    from mapmad_sim.camera_setting import cameras_of

    layout = layout_of(args.config)
    eps, fp = episodes.load_episodes(folder(layout) / "episodes.json")
    rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, cam in cameras_of(layout.cfg).items():
        rows[name] = {}
        for home in sorted({e["home"] for e in eps}):
            with LimoSim("train", home, cam.apply(limo_spec()), semantic=True, gpu=args.gpu) as robot:
                for e in (e for e in eps if e["home"] == home):
                    oid = int(e["target"]["object_id"])
                    share = {}
                    for key, pos, yaw in (("start", e["start_position"], e["start_yaw"]),
                                          ("goal_photo", e["goal_photo_position"], e["goal_photo_yaw"])):
                        robot.place(pos, yaw)
                        share[key] = round(float((robot.observe()["semantic"] == oid).mean()), 5)
                    rows[name][e["episode_id"]] = share
        print(name, flush=True)
    lines = ["# Target pixel share per camera (report only)", "", f"episodes fingerprint `{fp}`", "",
             "| camera | in view: start share (median) | in view: start share 0 | goal photo share (median) | goal photo < 5% |",
             "|---|---|---|---|---|"]
    for name, r in rows.items():
        iv = [r[e["episode_id"]] for e in eps if e["type"] == "in_view"]
        allg = [v["goal_photo"] for v in r.values()]
        lines.append(f"| {name} | {np.median([v['start'] for v in iv]):.4f} | {sum(v['start'] == 0 for v in iv)} | "
                     f"{np.median(allg):.3f} | {sum(g < 0.05 for g in allg)} |")
    (folder(layout) / "shares.json").write_text(json.dumps({"fingerprint": fp, "shares": rows}, indent=1) + "\n")
    (folder(layout) / "shares.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["new", "assemble", "freeze", "shares"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    {"new": cmd_new, "assemble": cmd_assemble, "freeze": cmd_freeze, "shares": cmd_shares}[args.cmd](args)


if __name__ == "__main__":
    main()
