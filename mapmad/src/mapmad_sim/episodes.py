"""Phase 1 episodes: a start pose, one target object and a goal photo, frozen in a JSON file with a sha256.

For one HM3D train home (configs/p1_baseline.yaml `episodes`; floor map = LIMO-sized navmesh, robot.py):
- targets: ObjectNav v2 train goals of the home (`goals_by_category`: object id, category, view points), object
  boxes from the OBB (objects.py); at most max_per_category episodes per category per type;
- starts: random navigable points on the real floor (the surface under the robot is floor / carpet / rug / mat,
  not a bed or stairs), on the target's floor, not facing something up close (at most half of the picture
  nearer than 0.5 m); target point = the real-floor navigable point horizontally nearest to the object's box
  (at most target_point_max_m away) on the start's island; geodesic start -> target point inside the type's range;
  out_of_view: 0 target pixels in all 8 headings (every 45 deg) at the start, LIMO's 66.5 deg camera; random
  start heading; in_view: heading within half the FOV of the object, target >= 1% of the start picture;
- goal photo: one of the object's ObjectNav view points 0.8-1.2 m (horizontally) from its box, navigable on our
  floor map, on the real floor, facing the object's centre, object >= 5% of the pixels, at most 5% of the
  picture near-black (mesh holes); closest to 1 m wins.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.objectnav import Goal
from mapmad_sim.objects import name_matches, object_box
from mapmad_sim.robot import LimoSim, horizontal_distance, yaw_towards

TYPE_CODES = {"out_of_view": "oov", "in_view": "iv"}
FLOOR_MAP_CHECK = "floor_map_check.json"  # written next to episodes.json by mapmad/scripts/check_p1_episodes.py


@dataclass
class Target:
    object_id: int
    category: str
    name: str
    box_center: List[float]
    box_size: List[float]
    point: List[float]  # navigable target point; success = geodesic to it <= success_m


@dataclass
class Episode:
    episode_id: str
    type: str
    split: str
    home: str
    target: Target
    start_position: List[float]
    start_yaw: float
    start_geodesic_m: float
    start_target_share: float  # share of the start picture showing the target (LIMO camera)
    goal_photo_position: List[float]
    goal_photo_yaw: float
    goal_photo_share: float
    navmesh_sha256: str
    extra: Dict[str, Any] = field(default_factory=dict)


def box_distance_xz(point: Sequence[float], center: Sequence[float], size: Sequence[float]) -> float:
    """Horizontal distance from a point to an axis-aligned box (0 inside it)."""
    dx = max(abs(point[0] - center[0]) - size[0] / 2.0, 0.0)
    dz = max(abs(point[2] - center[2]) - size[2] / 2.0, 0.0)
    return math.hypot(dx, dz)


def rounded(values: Sequence[float], digits: int = 4) -> List[float]:
    return [round(float(v), digits) for v in values]


def fingerprint(episodes: Sequence[Dict[str, Any]]) -> str:
    """sha256 of the canonical JSON of the episode list (key order and spacing fixed)."""
    blob = json.dumps(list(episodes), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load_episodes(path: Path, check: bool = True) -> Tuple[List[Dict[str, Any]], str]:
    """Episodes of a frozen file and their fingerprint; raises if the stored fingerprint doesn't match."""
    doc = json.loads(Path(path).read_text())
    fp = fingerprint(doc["episodes"])
    if check and fp != doc["fingerprint"]:
        raise ValueError(f"{path}: fingerprint {fp} != stored {doc['fingerprint']} (file changed)")
    return doc["episodes"], fp


def checked_floor_maps(episodes_dir: Path, fingerprint: str) -> Dict[str, str]:
    """episode id -> floor-map sha256 the episode passed check_p1_episodes.py on; empty if this episode file
    (same fingerprint) was never checked."""
    path = Path(episodes_dir) / FLOOR_MAP_CHECK
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    if doc["fingerprint"] != fingerprint:
        return {}
    now = {home: shas["now"] for home, shas in doc["navmesh_sha256"].items()}
    return {eid: now[r["home"]] for eid, r in doc["episodes"].items() if r["pass"]}


def floor_map_ok(episode: Dict[str, Any], live_sha256: Optional[str], checked: Dict[str, str]) -> bool:
    """True if the episode was built on this floor map, or passed check_p1_episodes.py on it."""
    return episode.get("navmesh_sha256") == live_sha256 or checked.get(episode["episode_id"]) == live_sha256


class EpisodeBuilder:
    """Samples Phase 1 episodes in one home (LimoSim with LIMO's camera, semantic=True, depth=True)."""

    def __init__(self, robot: LimoSim, cfg: Dict[str, Any], rng: np.random.Generator) -> None:
        self.robot, self.cfg, self.rng = robot, cfg, rng
        self.pf = robot.pathfinder
        objects = [o for o in robot.sim.semantic_scene.objects if o is not None]
        self.objects = {o.semantic_id: o for o in objects}
        self.names = {o.semantic_id: (o.category.name().lower() if o.category is not None else "") for o in objects}
        self.floor_ids = {i for i, n in self.names.items() if name_matches(n, cfg["floor_names"])}

    # --- helpers ---------------------------------------------------------------------------------------------

    def random_point(self) -> np.ndarray:
        """A random navmesh point, drawn from our own rng (reproducible regardless of habitat's state)."""
        while True:  # habitat returns NaN now and then
            self.pf.seed(int(self.rng.integers(2**31 - 1)))
            p = np.array(self.pf.get_random_navigable_point(), dtype=np.float64)
            if np.isfinite(p).all():
                return p

    def points_near(self, center: Sequence[float], radius: float, n: int) -> Iterator[np.ndarray]:
        for _ in range(n):
            self.pf.seed(int(self.rng.integers(2**31 - 1)))
            p = np.array(self.pf.get_random_navigable_point_near(np.asarray(center, np.float32), radius, 100))
            if np.isfinite(p).all():
                yield p.astype(np.float64)

    def on_real_floor(self, position: np.ndarray) -> bool:
        """Robot placed here stands on floor / carpet / rug / mat (not a bed, stairs or a hole)."""
        try:
            self.robot.place(position, 0.0)
        except ValueError:
            return False
        return self.robot.surface_id() in self.floor_ids

    def target_share(self, position: np.ndarray, yaw: float, object_id: int) -> float:
        self.robot.place(position, yaw)
        return float((self.robot.observe()["semantic"] == object_id).mean())

    def black_share(self) -> float:
        """Share of near-black pixels in the current picture: HM3D meshes have holes (rendered black), e.g.
        where a toilet should be, while the labels around them still say 'toilet'."""
        return float((self.robot.observe()["rgb"].max(axis=2) < 16).mean())

    def view_blocked(self) -> bool:
        """More than max_near_share of the current picture is closer than near_m (facing a wall or furniture
        up close). Needs depth=True and a placed robot."""
        rule = self.cfg["start_view"]
        return float((self.robot.observe()["depth"] < rule["near_m"]).mean()) > rule["max_near_share"]

    # --- targets, goal photos, starts --------------------------------------------------------------------------

    def target(self, goal: Goal) -> Optional[Tuple[Target, List[np.ndarray]]]:
        """The object and its candidate target points (real floor, <= target_point_max_m from the box, nearest
        first); None if there are none."""
        obj = self.objects.get(goal.object_id)
        if obj is None:
            return None
        center, size = object_box(obj)
        bottom = center[1] - size[1] / 2.0
        near = []
        for p in self.points_near(center, 2.0, 300):
            # the object's storey (a wall-mounted tv hangs up to 2 m above its floor), never the floor above
            if -2.0 <= p[1] - bottom <= 0.5:
                d = box_distance_xz(p, center, size)
                if d <= self.cfg["target_point_max_m"]:
                    near.append((d, tuple(p)))
        points = [np.array(p) for _, p in sorted(set(near)) if self.on_real_floor(np.array(p))]
        if not points:
            return None
        target = Target(goal.object_id, goal.category, self.names.get(goal.object_id, ""), rounded(center),
                        rounded(size), rounded(points[0]))
        return target, points

    def goal_photo(self, goal: Goal, target: Target) -> Optional[Tuple[np.ndarray, float, float]]:
        """(position, yaw, target share) of the goal photo from the object's ObjectNav view points, or None."""
        g = self.cfg["goal_photo"]
        lo, hi = g["distance_m"]
        found = []
        for vp in goal.view_points:
            d = box_distance_xz(vp, target.box_center, target.box_size)
            if not lo <= d <= hi:
                continue
            p = np.array(self.pf.snap_point(np.asarray(vp, np.float32)), dtype=np.float64)
            if not np.isfinite(p).all() or horizontal_distance(p, vp) > 0.1 or abs(p[1] - target.point[1]) > self.cfg["same_floor_max_dy_m"]:
                continue  # not walkable for LIMO's floor map
            if not math.isfinite(self.robot.geodesic(target.point, p)) or not self.on_real_floor(p):
                continue
            yaw = yaw_towards(p, target.box_center)
            share = self.target_share(p, yaw, target.object_id)
            if share >= g["min_target_share"] and self.black_share() <= g["max_black_share"]:
                found.append((abs(box_distance_xz(p, target.box_center, target.box_size) - g["target_distance_m"]), p, yaw, share))
        if not found:
            return None
        _, p, yaw, share = min(found, key=lambda f: f[0])
        return p, yaw, share

    def hidden_all_round(self, position: np.ndarray, object_id: int) -> bool:
        """0 target pixels in all 8 headings (every 45 deg) at this position."""
        return all(self.target_share(position, k * math.pi / 4.0, object_id) == 0.0 for k in range(8))

    def start(self, target: Target, points: List[np.ndarray], kind: str
              ) -> Optional[Tuple[np.ndarray, float, float, float, np.ndarray]]:
        """(position, yaw, geodesic, target share, target point) of a start of this type, or None."""
        rule = self.cfg["types"][kind]
        lo, hi = rule["geodesic_m"]
        half_fov = math.radians(self.cfg["visibility_hfov_deg"]) / 2.0
        for _ in range(self.cfg["tries_per_target"]):
            p = self.random_point()
            if abs(p[1] - target.point[1]) > self.cfg["same_floor_max_dy_m"]:
                continue
            # target point: the nearest candidate on the start's island
            point, geo = next(((c, g) for c in points for g in [self.robot.geodesic(p, c)] if math.isfinite(g)),
                              (None, math.inf))
            if point is None or not lo <= geo <= hi or not self.on_real_floor(p):
                continue
            if kind == "in_view":
                yaw = yaw_towards(p, target.box_center) + float(self.rng.uniform(-half_fov, half_fov))
            else:
                yaw = float(self.rng.uniform(-math.pi, math.pi))
                if not self.hidden_all_round(p, target.object_id):
                    continue
            share = self.target_share(p, yaw, target.object_id)
            if self.view_blocked() or (kind == "in_view" and share < rule["min_target_share"]):
                continue
            return p, yaw, geo, share, point
        return None

    def episodes(self, kind: str, count: int, goals: Sequence[Goal], split: str) -> List[Episode]:
        """Up to `count` episodes of one type, cycling through the categories, at most max_per_category each."""
        by_cat: Dict[str, List[Goal]] = {}
        for goal in goals:
            by_cat.setdefault(goal.category, []).append(goal)
        queues = {c: [gs[i] for i in self.rng.permutation(len(gs))] for c, gs in sorted(by_cat.items())}
        order = [sorted(queues)[i] for i in self.rng.permutation(len(queues))]
        used = {c: 0 for c in queues}
        out: List[Episode] = []
        while len(out) < count and any(queues[c] and used[c] < self.cfg["max_per_category"] for c in queues):
            for cat in order:
                if len(out) >= count or not queues[cat] or used[cat] >= self.cfg["max_per_category"]:
                    continue
                goal = queues[cat].pop(0)
                episode = self.episode(goal, kind, split, len(out))
                if episode is not None:  # objects without a valid episode drop out; the others get more turns
                    out.append(episode)
                    used[cat] += 1
                    queues[cat].append(goal)
        return out

    def episode(self, goal: Goal, kind: str, split: str, index: int) -> Optional[Episode]:
        found = self.target(goal)
        if found is None:
            return None
        target, points = found
        photo = self.goal_photo(goal, target)
        if photo is None:
            return None
        start = self.start(target, points, kind)
        if start is None:
            return None
        p, yaw, geo, share, point = start
        target.point = rounded(point)
        home = self.robot.home
        prefix = self.cfg.get("id_prefix", "p1")
        return Episode(episode_id=f"{prefix}-{TYPE_CODES[kind]}-{home[:5]}-{index:03d}", type=kind, split=split, home=home,
                       target=target, start_position=rounded(p), start_yaw=round(yaw, 4),
                       start_geodesic_m=round(geo, 3), start_target_share=round(share, 4),
                       goal_photo_position=rounded(photo[0]), goal_photo_yaw=round(photo[1], 4),
                       goal_photo_share=round(photo[2], 4), navmesh_sha256=self.robot.navmesh_sha256)


def episode_dict(e: Episode) -> Dict[str, Any]:
    return asdict(e)
