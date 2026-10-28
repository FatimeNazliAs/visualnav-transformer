"""Phase 12 video pose check (Naz, 10-25, item f): the re-rendered videos follow the logged poses exactly, and the
floor-height fill (robot.move_to keeps the last good floor height) happens only on the hole frames.

    python mapmad/scripts/p12_video_pose_check.py --gpu 1      # inside naz_mapmad_habitat, after p12_videos.py render

For every video of videos/video_list.json and every arm in it, the arm's log (seed 0) is replayed pose by pose exactly
as p12_videos.py does (robot.move_to on the logged position and yaw), and at every frame:
- the robot's pose equals the logged pose (position bit for bit, yaw after wrapping);
- hole frame = the floor probe found no floor (robot.place raises there; move_to keeps the last good floor height);
- every other frame: the floor height equals the one robot.place gives (no fill used).
The log's sha256 must equal the one the render recorded in video_list.json (the video was made from this log).
Writes checks/video_pose_check.json (--which review / per_arm: checks/video_pose_check_<which>.json); exit 3 if a check
fails.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))

from mapmad_bridge import steplog  # noqa: E402
from mapmad_sim import config  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.robot import LimoSim, wrap_angle  # noqa: E402  (LimoSim: type of the robot)
from mapmad_sim.run_layout import RunLayout, write_text_atomic  # noqa: E402
from mapmad_sim.search_wire import camera_wire  # noqa: E402
from mapmad_sim.sidecar_server import limo_factory  # noqa: E402


def check_log(robot: LimoSim, log_path: Path) -> dict:
    log = steplog.read(log_path)
    poses = [(log[0]["start"]["position"], log[0]["start"]["yaw"])] + [(s["position"], s["yaw"]) for s in steplog.steps(log)]
    holes, pose_bad, floor_bad = [], 0, 0
    robot.place(np.asarray(poses[0][0], np.float64), float(poses[0][1]))  # the start is never a hole (the set checks it)
    for t, (p, yaw) in enumerate(poses[1:], start=1):
        before = robot.floor_probe_failures
        robot.move_to(p, float(yaw))
        hole = robot.floor_probe_failures > before
        pose_bad += int(not np.array_equal(robot.position, np.asarray(p, np.float64))
                        or abs(wrap_angle(robot.yaw - float(yaw))) > 1e-12)
        floor_fill = robot.floor_below_feet
        try:
            robot.place(np.asarray(p, np.float64), float(yaw))
            if hole or robot.floor_below_feet != floor_fill:
                floor_bad += 1
        except ValueError:  # place() leaves the floor height unknown (NaN): restore the replay's own state
            if not hole:
                floor_bad += 1
            holes.append(t)
            robot.floor_below_feet = floor_fill
            robot.move_to(p, float(yaw))
    return {"frames": len(poses), "pose_mismatches": pose_bad, "floor_mismatches": floor_bad, "hole_frames": holes,
            "pass": pose_bad == 0 and floor_bad == 0}


def logs_of(pick: dict, which: str) -> dict:
    """arm -> log sha256 the video was rendered from (the overview, the review copy or the per-arm videos)."""
    return (pick.get("logs") if which == "overview" else (pick.get(which) or {}).get("logs")) or {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--config", type=Path, default=config.CONFIG_DIR / "p12_exam.yaml")
    p.add_argument("--which", choices=["overview", "review", "per_arm"], default="overview",
                   help="which rendered videos: overview (picks[].logs), review or per_arm (picks[].<which>.logs)")
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    lst = json.loads((layout.base / "videos" / "video_list.json").read_text())
    eps = {e["episode_id"]: e for e in load_episodes(layout.base / layout.cfg["sets"]["exam"]["folder"] / "episodes.json")[0]}
    rows, ok = [], True
    picks = [pk for pk in lst["picks"] if args.which != "per_arm" or pk.get("per_arm")]  # per-arm: 3 picks only
    for pick in picks:
        e = eps[pick["episode"]]
        camera = camera_wire(layout.cfg["camera"])  # the robot exactly as the exam's server builds it
        with limo_factory(args.gpu)(e["split"], e["home"], camera["hfov_deg"], depth=True, camera=camera,
                                    semantic=True) as robot:
            for arm, sha in sorted(logs_of(pick, args.which).items()):
                f = layout.base / "runs/exam/logs" / arm / f"{e['episode_id']}__s0.jsonl"
                r = check_log(robot, f)
                r.update(video=pick["name"], arm=arm, log_sha256_matches=hashlib.sha256(f.read_bytes()).hexdigest() == sha)
                r["pass"] = r["pass"] and r["log_sha256_matches"]
                ok &= r["pass"]
                rows.append(r)
                print(json.dumps({k: (v if k != "hole_frames" else len(v)) for k, v in r.items()}), flush=True)
        if not logs_of(pick, args.which):
            ok = False
            rows.append({"video": pick["name"], "pass": False, "why": "no logs recorded (not rendered)"})
    res = {"pass": ok and bool(rows), "videos": len(picks), "which": args.which, "arm_logs": len(rows),
           "hole_frames_total": sum(len(r.get("hole_frames", [])) for r in rows), "rows": rows}
    out = "video_pose_check.json" if args.which == "overview" else f"video_pose_check_{args.which}.json"
    write_text_atomic(layout.base / "checks" / out, json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("pass", "videos", "arm_logs", "hole_frames_total")}), flush=True)
    raise SystemExit(0 if res["pass"] else 3)


if __name__ == "__main__":
    main()
