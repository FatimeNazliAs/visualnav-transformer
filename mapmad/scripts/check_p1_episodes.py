"""Check a frozen Phase 1 episode file against the current floor map (robot_limo.yaml `navmesh`), e.g. after a
robot-size change. Nothing is rebuilt or rewritten.

    python mapmad/scripts/check_p1_episodes.py                       # inside naz_mapmad_habitat; p1_baseline episodes
    python mapmad/scripts/check_p1_episodes.py --out /outputs/mapmad/p1_baseline_spec/checks   # report elsewhere

Per episode, on the LIMO-sized navmesh rebuilt now:
- start: navigable and on the real floor (floor / carpet / rug / mat);
- target point: navigable and on the real floor;
- same island: finite geodesic start -> target point; the geodesic is still inside the type's range;
- info only: goal-photo pose navigable, on the real floor, reaches the target point.
Writes floor_map_check.json (per episode + the new navmesh sha256 per home) next to the episode file (or into
--out) and prints a summary; exit code 1 if any episode fails. The runner refuses an episode whose stored floor-map
hash differs from the live one unless this report, next to the episode file, says it passed on that floor map.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mapmad_sim import config, episodes
from mapmad_sim.robot import LimoSim, RobotSpec
from mapmad_sim.run_layout import P1_CONFIG, RunLayout


def check_episode(builder: episodes.EpisodeBuilder, ep: Dict[str, Any], geodesic_range: List[float]) -> Dict[str, Any]:
    """The rule results of one episode on the builder's floor map (see the module docstring)."""
    robot, pf = builder.robot, builder.pf
    start, point, photo = (np.asarray(p, np.float64) for p in (ep["start_position"], ep["target"]["point"],
                                                               ep["goal_photo_position"]))
    geo = robot.geodesic(start, point)
    photo_geo = robot.geodesic(photo, point)
    out = {
        "home": ep["home"],
        "start_navigable": bool(pf.is_navigable(start.astype(np.float32))),
        "start_real_floor": builder.on_real_floor(start),
        "target_navigable": bool(pf.is_navigable(point.astype(np.float32))),
        "target_real_floor": builder.on_real_floor(point),
        "same_island": math.isfinite(geo),
        "geodesic_in_range": math.isfinite(geo) and geodesic_range[0] <= geo <= geodesic_range[1],
        "start_geodesic_m": {"stored": ep["start_geodesic_m"], "now": round(geo, 3) if math.isfinite(geo) else None},
        "goal_photo_ok": bool(pf.is_navigable(photo.astype(np.float32))) and builder.on_real_floor(photo)
                         and math.isfinite(photo_geo),
    }
    out["pass"] = all(out[k] for k in ("start_navigable", "start_real_floor", "target_navigable",
                                       "target_real_floor", "same_island", "geodesic_in_range"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=P1_CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", type=Path, help="write the report here instead of next to the episode file")
    args = p.parse_args()

    layout = RunLayout.load(args.config)
    cfg = layout.cfg["episodes"]
    eps, fp = episodes.load_episodes(layout.episodes_file)
    spec = RobotSpec.from_config(config.robot(), hfov_deg=cfg["visibility_hfov_deg"])
    rng = np.random.default_rng(0)  # unused by the checks (no sampling); EpisodeBuilder needs one
    results: Dict[str, Any] = {}
    navmeshes: Dict[str, Dict[str, str]] = {}
    for home in sorted({e["home"] for e in eps}):
        with LimoSim(cfg["split"], home, spec, semantic=True, depth=True, gpu=args.gpu, seed=cfg["seed"]) as robot:
            builder = episodes.EpisodeBuilder(robot, cfg, rng)
            old = {e["navmesh_sha256"] for e in eps if e["home"] == home}
            navmeshes[home] = {"stored": ",".join(sorted(old)), "now": robot.navmesh_sha256}
            for e in (e for e in eps if e["home"] == home):
                results[e["episode_id"]] = check_episode(builder, e, cfg["types"][e["type"]]["geodesic_m"])
    failed = sorted(k for k, r in results.items() if not r["pass"])
    photo_bad = sorted(k for k, r in results.items() if not r["goal_photo_ok"])
    changes = [abs(r["start_geodesic_m"]["now"] - r["start_geodesic_m"]["stored"])
               for r in results.values() if r["start_geodesic_m"]["now"] is not None]
    out = (args.out or layout.episodes_dir) / episodes.FLOOR_MAP_CHECK
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes_file": str(layout.episodes_file), "fingerprint": fp,
                               "navmesh": dict(config.robot()["navmesh"]), "navmesh_sha256": navmeshes,
                               "failed": failed, "goal_photo_not_ok": photo_bad, "episodes": results}, indent=1) + "\n")
    print(f"{len(results)} episodes, fingerprint {fp}: {len(results) - len(failed)} pass, {len(failed)} fail {failed}")
    print(f"goal-photo poses not ok (info only): {len(photo_bad)} {photo_bad}")
    if changes:
        print(f"start geodesic change: max {max(changes):.3f} m, mean {np.mean(changes):.4f} m")
    print(f"wrote {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
