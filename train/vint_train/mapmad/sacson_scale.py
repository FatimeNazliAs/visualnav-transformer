"""Phase 7.4: metric scale of SACSoN's /odometry from its AR-tag detections (naz_mapmad: needs rosbags).

SACSoN's /odometry ("global odometry using long-term visual anchors") has no reliable metric scale: its pose speed is
0.45 m/s in one building and 0.76 m/s in another (a Roomba drives at most 0.5 m/s). The bags also hold /ar_marker:
ALVAR tag poses in the fisheye camera's frame (`/usb_cam`, OpenCV axes), metric because ALVAR knows the tag size
and the camera calibration. While the robot drives straight, a fixed tag's horizontal position in the camera frame
(forward = z, left = -x) moves by exactly the robot's metric displacement, so

    scale = |tag shift between t1 and t2| / |odometry displacement between t1 and t2|

for pairs 0.5-1.5 s apart with < 3 deg turn and an odometry displacement > 0.15 m; per bag the median over pairs.

    cd /app/visualnav-transformer/train && python -m vint_train.mapmad.sacson_scale \
        --bags /data/sacson_raw --out /outputs/mapmad/p7_camera/height/sacson_odom_scale.json
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def typestore_for(conn):
    """ROS1 Noetic types + the bag's own ar_track_alvar_msgs definitions."""
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    ts = get_typestore(Stores.ROS1_NOETIC)
    md = conn.msgdef if isinstance(conn.msgdef, str) else conn.msgdef.data
    parts = re.split(r"\n=+\nMSG: ", md)
    types = dict(get_types_from_msg(parts[0], conn.msgtype))
    for p in parts[1:]:
        name, body = p.split("\n", 1)
        name = name.strip()
        if name.startswith("ar_track_alvar_msgs/"):
            types.update(get_types_from_msg(body, name.replace("/", "/msg/", 1)))
    ts.register(types)
    return ts


def read_bag(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[Tuple[float, np.ndarray]]]]:
    """Odometry times (N,), poses (N, 3: x, y, yaw) and tag detections {id: [(time, xyz in camera frame)]}."""
    from rosbags.rosbag1 import Reader

    times, poses, tags = [], [], {}
    with Reader(path) as r:
        odo = [c for c in r.connections if c.topic == "/odometry"]
        ar = [c for c in r.connections if c.topic == "/ar_marker"]
        if not odo or not ar:
            return np.zeros(0), np.zeros((0, 3)), {}
        ts = typestore_for(ar[0])
        for con, t, raw in r.messages(connections=odo + ar):
            m = ts.deserialize_ros1(raw, con.msgtype)
            if con.topic == "/odometry":
                q = m.pose.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                times.append(t / 1e9)
                poses.append([m.pose.pose.position.x, m.pose.pose.position.y, yaw])
            else:
                for mk in m.markers:
                    p = mk.pose.pose.position
                    tags.setdefault(int(mk.id), []).append((t / 1e9, np.array([p.x, p.y, p.z])))
    return np.asarray(times), np.asarray(poses), tags


def bag_scale(times: np.ndarray, poses: np.ndarray, tags: Dict[int, List[Tuple[float, np.ndarray]]],
              gap_s: Tuple[float, float] = (0.5, 1.5), max_turn_deg: float = 3.0, min_move_m: float = 0.15,
              max_tag_m: float = 4.0) -> List[float]:
    """Per-pair scale factors (metric / odometry) of one bag."""
    out: List[float] = []
    if len(times) < 2:
        return out
    for dets in tags.values():
        for a in range(len(dets)):
            ta, pa = dets[a]
            for b in range(a + 1, len(dets)):
                tb, pb = dets[b]
                if tb - ta > gap_s[1]:
                    break
                if tb - ta < gap_s[0] or np.linalg.norm(pa) > max_tag_m or np.linalg.norm(pb) > max_tag_m:
                    continue
                ia, ib = np.searchsorted(times, ta), np.searchsorted(times, tb)
                if ib >= len(times) or ia >= len(times):
                    continue
                turn = math.degrees(abs(math.atan2(math.sin(poses[ib, 2] - poses[ia, 2]), math.cos(poses[ib, 2] - poses[ia, 2]))))
                odom = float(np.hypot(*(poses[ib, :2] - poses[ia, :2])))
                if turn > max_turn_deg or odom < min_move_m:
                    continue
                shift = float(np.hypot(pb[2] - pa[2], pb[0] - pa[0]))  # horizontal: forward z, sideways x
                out.append(shift / odom)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bags", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-pairs", type=int, default=20, help="bags with fewer pairs use their folder's median")
    args = p.parse_args()
    per_bag, per_folder = {}, {}
    for bag in sorted(args.bags.glob("*/*.bag")):
        s = bag_scale(*read_bag(bag))
        name = f"{bag.parent.name}_{bag.stem}"
        per_bag[name] = {"pairs": len(s), "scale": float(np.median(s)) if s else None,
                         "p25_p75": [float(v) for v in np.percentile(s, [25, 75])] if s else None}
        per_folder.setdefault(bag.parent.name, []).extend(s)
        print(name, per_bag[name], flush=True)
    folders = {f: {"pairs": len(s), "scale": float(np.median(s)) if s else None,
                   "p25_p75": [float(v) for v in np.percentile(s, [25, 75])] if s else None} for f, s in per_folder.items()}
    use = {b: (v["scale"] if v["pairs"] >= args.min_pairs else folders[b.rsplit("_", 1)[0]]["scale"]) for b, v in per_bag.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"method": __doc__.split("\n\n")[1], "min_pairs": args.min_pairs, "per_folder": folders,
                                    "per_bag": per_bag, "scale_used": use}, indent=1) + "\n")
    print(json.dumps(folders, indent=1))


if __name__ == "__main__":
    main()
