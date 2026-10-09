"""Build and freeze the Phase 1 episodes (configs/p1_baseline.yaml `episodes`).

    python mapmad/scripts/make_p1_episodes.py            # inside naz_mapmad_habitat, a few minutes
    python mapmad/scripts/make_p1_episodes.py --force    # replace an existing episode file
    python mapmad/scripts/make_p1_episodes.py --out /outputs/mapmad/p1_baseline/checks/episodes_rebuild   # elsewhere

Homes: the labelled HM3D train homes with ObjectNav v2 train goals, shuffled with home_seed; a home is used if
it gives per_home episodes of every type, until n_homes are used (skipped homes are logged). Writes
<outputs>/p1_baseline/episodes/ (or --out): episodes.json (episodes + sha256 fingerprint), episodes.sha256, summary.md,
goal_photos/<episode>.png and start_views/<episode>.png (LIMO camera), run_info.json.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from mapmad_sim import config, episodes, objectnav
from mapmad_sim.robot import LimoSim, RobotSpec
from mapmad_sim.run_info import save_run_info
from mapmad_sim.run_layout import P1_CONFIG, RunLayout


def save_view(robot: LimoSim, position: List[float], yaw: float, path: Path) -> None:
    robot.place(position, yaw)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(robot.observe()["rgb"], cv2.COLOR_RGB2BGR))


def candidate_homes(cfg: Dict[str, Any]) -> List[str]:
    """Labelled train homes that have an ObjectNav file, in the seeded order (or the config's `homes` list)."""
    if cfg.get("homes"):
        return list(cfg["homes"])
    homes = [h for h in config.labelled_homes(cfg["split"]) if objectnav.has_goals(cfg["objectnav_split"], h)]
    return [homes[i] for i in np.random.default_rng(cfg["home_seed"]).permutation(len(homes))]


def summary_markdown(eps: List[Dict[str, Any]], fp: str, homes: List[str], skipped: Dict[str, str]) -> str:
    lines = ["# Phase 1 episodes", "", f"fingerprint (sha256): `{fp}`", "",
             f"homes ({len(homes)}; Phase 2: hold these out as its val-drive split): {', '.join(homes)}", "",
             f"homes tried and skipped: {skipped or 'none'}", ""]
    for kind in ("out_of_view", "in_view"):
        sel = [e for e in eps if e["type"] == kind]
        if not sel:  # a config with one type only
            continue
        geo = np.array([e["start_geodesic_m"] for e in sel])
        lines += [f"## {kind}: {len(sel)} episodes", "",
                  f"- per category: {dict(Counter(e['target']['category'] for e in sel))}",
                  f"- distinct target objects: {len({(e['home'], e['target']['object_id']) for e in sel})}",
                  f"- start geodesic m: min {geo.min():.2f}, median {np.median(geo):.2f}, max {geo.max():.2f}",
                  f"- start target share: min {min(e['start_target_share'] for e in sel):.4f}, "
                  f"max {max(e['start_target_share'] for e in sel):.4f}",
                  f"- goal photo share: min {min(e['goal_photo_share'] for e in sel):.3f}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=P1_CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--force", action="store_true", help="replace an existing episodes.json")
    p.add_argument("--out", type=Path, help="write here instead of the run's episodes folder")
    args = p.parse_args()

    layout = RunLayout.load(args.config)
    cfg = layout.cfg["episodes"]
    out = args.out or layout.episodes_dir
    if (out / "episodes.json").exists() and not args.force:
        raise SystemExit(f"{out / 'episodes.json'} exists (frozen); pass --force to replace it")
    spec = RobotSpec.from_config(config.robot(), hfov_deg=cfg["visibility_hfov_deg"])
    rng = np.random.default_rng(cfg["seed"])
    all_eps: List[Dict[str, Any]] = []
    used: List[str] = []
    navmeshes: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    for home in candidate_homes(cfg):
        if len(used) == cfg["n_homes"]:
            break
        goals = objectnav.home_goals(cfg["objectnav_split"], home)
        with LimoSim(cfg["split"], home, spec, semantic=True, depth=True, gpu=args.gpu, seed=cfg["seed"]) as robot:
            builder = episodes.EpisodeBuilder(robot, cfg, rng)
            made = {kind: builder.episodes(kind, cfg["per_home"], goals, cfg["split"]) for kind in cfg["types"]}
            counts = {k: len(v) for k, v in made.items()}
            print(f"{home}: {counts} from {len(goals)} goal objects, navmesh {robot.navmesh_sha256[:12]}", flush=True)
            if any(n < cfg["per_home"] for n in counts.values()):
                skipped[home] = f"only {counts}"
                continue
            used.append(home)
            navmeshes[home] = robot.navmesh_sha256
            for e in (e for kind in cfg["types"] for e in made[kind]):
                save_view(robot, e.goal_photo_position, e.goal_photo_yaw, out / "goal_photos" / f"{e.episode_id}.png")
                save_view(robot, e.start_position, e.start_yaw, out / "start_views" / f"{e.episode_id}.png")
                all_eps.append(episodes.episode_dict(e))
    if len(used) < cfg["n_homes"]:
        raise SystemExit(f"only {len(used)} homes could give {cfg['per_home']} episodes per type")
    all_eps.sort(key=lambda e: (e["type"] != "out_of_view", e["home"], e["episode_id"]))
    fp = episodes.fingerprint(all_eps)
    doc = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config": cfg, "homes": used,
           "skipped_homes": skipped, "navmesh_sha256": navmeshes, "fingerprint": fp, "episodes": all_eps}
    (out / "episodes.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "episodes.sha256").write_text(f"{fp}  episodes (canonical JSON of the 'episodes' list)\n")
    (out / "summary.md").write_text(summary_markdown(all_eps, fp, used, skipped))
    save_run_info(out, {**vars(args), "config": str(args.config), "episodes": cfg, "robot": config.robot()}, cfg["seed"])
    print(f"{len(all_eps)} episodes in {len(used)} homes, fingerprint {fp}\nwrote {out}")


if __name__ == "__main__":
    main()
