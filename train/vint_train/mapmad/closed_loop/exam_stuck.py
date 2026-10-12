"""Where was the heat when a run got stuck? Report only (Naz, G5 review 2026-10-12), from the step logs (inside naz_mapmad):

    python -m vint_train.mapmad.closed_loop.exam_stuck --config /app/visualnav-transformer/mapmad/configs/p5_exam.yaml \
        --subdir runs --arms C --out tables/stuck_heat_bearing

For every run with a stuck streak (>= run.stuck_steps consecutive collision steps), the robot's pose when its first
streak starts, and the bearing from its heading to the nearest point of the target box (the heat's footprint; the
centre if the robot is inside it): ahead = |bearing| <= 45 deg, side = 45-135 deg (left / right), behind = > 135 deg.
The same sectors for the median bearing over that whole streak (the robot keeps turning while it pushes), and the
straight-line distance to the box at onset. Writes <out>.md and <out>.json (one row per stuck run).
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from mapmad_bridge import steplog
from mapmad_sim.episodes import load_episodes
from mapmad_sim.run_layout import RunLayout, parse_run_id

AHEAD_DEG, BEHIND_DEG = 45.0, 135.0


def nearest_box_point(p: Sequence[float], center: Sequence[float], size: Sequence[float]) -> Tuple[float, float]:
    """(x, z) of the box footprint point nearest to p (p itself if inside)."""
    x = min(max(p[0], center[0] - size[0] / 2), center[0] + size[0] / 2)
    z = min(max(p[2], center[2] - size[2] / 2), center[2] + size[2] / 2)
    return x, z


def bearing_deg(position: Sequence[float], yaw: float, target_xz: Tuple[float, float]) -> float:
    """Signed angle from the robot's heading to the target (Habitat: forward = (-sin yaw, -cos yaw) in x, z);
    positive = to the left."""
    vx, vz = target_xz[0] - position[0], target_xz[1] - position[2]
    fwd = (-math.sin(yaw), -math.cos(yaw))
    left = (-math.cos(yaw), math.sin(yaw))
    return math.degrees(math.atan2(vx * left[0] + vz * left[1], vx * fwd[0] + vz * fwd[1]))


def sector(b: float) -> str:
    a = abs(b)
    if a <= AHEAD_DEG:
        return "ahead"
    if a <= BEHIND_DEG:
        return "side-left" if b > 0 else "side-right"
    return "behind-left" if b > 0 else "behind-right"


def stuck_onset(log_path: Path, episode: Dict[str, Any], stuck_steps: int) -> Dict[str, Any]:
    """The facts at the first stuck streak of one run, or {} if it has none."""
    log = steplog.read(log_path)
    streaks = steplog.stuck_streaks(log, stuck_steps)
    if not streaks:
        return {}
    step, length = streaks[0]
    t = episode["target"]

    def heat_bearing(position: Sequence[float], yaw: float) -> float:
        if inside_box(position, t["box_center"], t["box_size"]):
            return bearing_deg(position, yaw, (t["box_center"][0], t["box_center"][2]))
        return bearing_deg(position, yaw, nearest_box_point(position, t["box_center"], t["box_size"]))

    poses = steplog.poses(log)[step - 1:step - 1 + length]  # from the pose before the streak's first collision step
    position, yaw = poses[0]
    b = heat_bearing(position, yaw)
    target = nearest_box_point(position, t["box_center"], t["box_size"])
    med = float(np.median([heat_bearing(p, y) for p, y in poses]))
    return {"episode_id": episode["episode_id"], "seed_index": steplog.header(log)["seed_index"], "onset_step": step,
            "streak_steps": length, "bearing_deg": round(b, 1), "sector": sector(b),
            "median_bearing_deg": round(med, 1), "median_sector": sector(med),
            "box_distance_m": round(math.hypot(target[0] - position[0], target[1] - position[2]), 2),
            "success": bool(steplog.result(log)["success"]), "distance_bin": episode["distance_bin"]}


def inside_box(p: Sequence[float], c: Sequence[float], s: Sequence[float]) -> bool:
    return abs(p[0] - c[0]) <= s[0] / 2 and abs(p[2] - c[2]) <= s[2] / 2


def table(rows: List[Dict[str, Any]], arm: str, key: str = "sector", what: str = "at stuck onset") -> List[str]:
    n = len(rows)
    c = Counter(r[key] for r in rows)
    ahead = c["ahead"]
    side = c["side-left"] + c["side-right"]
    behind = c["behind-left"] + c["behind-right"]
    d = np.array([r["box_distance_m"] for r in rows]) if rows else np.zeros(1)
    lines = [f"## Arm {arm}: {n} stuck runs, heat {what}", "",
             f"| heat {what} | runs | share |", "|---|---|---|",
             f"| ahead (|bearing| <= 45 deg) | {ahead} | {ahead / max(n, 1):.2f} |",
             f"| side (45-135 deg): left / right | {side} ({c['side-left']} / {c['side-right']}) | {side / max(n, 1):.2f} |",
             f"| behind (> 135 deg): left / right | {behind} ({c['behind-left']} / {c['behind-right']}) | {behind / max(n, 1):.2f} |",
             "", f"Distance to the box at stuck onset: median {np.median(d):.2f} m (quartiles {np.quantile(d, 0.25):.2f} - "
             f"{np.quantile(d, 0.75):.2f} m).", "", "| bin | stuck runs | ahead | side | behind |", "|---|---|---|---|---|"]
    for b in sorted({r["distance_bin"] for r in rows}):
        rb = [r for r in rows if r["distance_bin"] == b]
        cb = Counter(r[key].split("-")[0] for r in rb)
        lines.append(f"| {b} | {len(rb)} | {cb['ahead']} | {cb['side']} | {cb['behind']} |")
    return lines + [""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="runs")
    p.add_argument("--arms", nargs="+", default=["C"])
    p.add_argument("--out", required=True, help="below <outputs>/<out_dir>, without extension")
    p.add_argument("--show", nargs="*", default=[], help="also print these runs (<arm>/<episode>__s<k>)")
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    episodes, fp = load_episodes(layout.episodes_file)
    by_id = {e["episode_id"]: e for e in episodes}
    stuck_steps = int(layout.cfg["run"]["stuck_steps"])
    out: Dict[str, Any] = {"fingerprint": fp, "subdir": args.subdir, "arms": {}}
    lines = ["# Heat bearing at the onset of the first stuck streak (report only)", "",
             f"Runs `{layout.root}`; stuck = >= {stuck_steps} consecutive collision steps; bearing from the robot's heading "
             f"to the nearest point of the target box (the heat footprint): ahead <= {AHEAD_DEG:g} deg, side "
             f"{AHEAD_DEG:g}-{BEHIND_DEG:g} deg, behind > {BEHIND_DEG:g} deg. Distance bins 0/1/2 = 3-6 / 6-10 / 10-15 m.", ""]
    for arm in args.arms:
        rows = []
        for f in layout.logs(arm):
            r = stuck_onset(f, by_id[parse_run_id(f.name)[0]], stuck_steps)
            if r:
                rows.append(r)
        out["arms"][arm] = rows
        lines += table(rows, arm) + table(rows, arm, "median_sector", "over the streak (median bearing)")
    for key in args.show:
        arm, run = key.split("/")
        r = stuck_onset(layout.log(arm, run), by_id[parse_run_id(run)[0]], stuck_steps)
        lines.append(f"- {key}: " + (f"onset step {r['onset_step']} ({r['streak_steps']} steps), bearing at onset "
                                     f"{r['bearing_deg']:+.0f} deg ({r['sector']}), median over the streak "
                                     f"{r['median_bearing_deg']:+.0f} deg ({r['median_sector']}), {r['box_distance_m']} m "
                                     f"from the box" if r else "no stuck streak"))
    path = layout.base / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
