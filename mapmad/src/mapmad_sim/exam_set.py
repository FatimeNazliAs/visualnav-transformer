"""Phase 5 episode sets (block items 3, 6, 8): official ObjectNav HM3D v2 starts, one specific target instance, hidden
from the start; frozen with a fingerprint and tied to the LIMO floor maps.

For each official episode of a home (configs/p5_exam.yaml `sets.<name>`), with LIMO's camera and floor map:
1. start = the official start position and rotation; the position snapped onto the LIMO floor map must lie
   <= start_snap_max_m (0.2 m) away horizontally and on the real floor (scene_labels.SceneLabels.on_real_floor);
2. target = the NEAREST instance of the episode's category: the ObjectNav instance with the shortest walking
   distance from the start to one of its usable view points. A usable view point is snapped onto the LIMO floor
   map <= view_point_max_snap_m (0.2 m), on the real floor (as Phase 2), and on the start's storey (navmesh
   heights <= same_storey_dy_m (0.5 m) apart). No instance reachable at all -> `no_reachable_instance`; reachable
   only through view points of another storey -> `other_storey`;
3. `target_visible`: the target has > 0 pixels in one of 8 headings at the start (start yaw + k x 45 deg);
4. `other_instance_visible`: another instance of the category has > 0 pixels in one of those 8 pictures;
5. `distance`: walking distance to the target's nearest usable view point outside distance_m (3-15 m);
6. `no_goal_photo`: the goal photo (arms B, F) is rendered at the target's usable view points facing the box
   centre, in order of horizontal box distance closest to goal_photo_box_m (1.0 m: Naz's correction 1a of block
   item 8 at S-dev = Phase 1's rule; null = nearest the box first, as the dev set was built); the first with >= 5%
   target pixels is used; none -> dropped.
An episode is kept if all hold; the drop table counts the FIRST failing rule in this order (and, separately, every
rule an episode fails). Kept episodes also store the Phase 1 style target point (real-floor point <= 1 m from the
box, nearest first, reachable from the start; secondary metric "target point <= 1.0 m"), the other instances'
boxes (failure type "wrong instance") and the floor-map sha256.

Draw (`draw`): from the kept episodes, seeded, at most per_home per home, `total` spread equally over the distance
bins (3-6 / 6-10 / 10-15 m; the remainder goes to the first bins). Pass 1 fills every bin up to its quota from a
seeded random order of its episodes, the bins with the fewest kept episodes first (they get the home quota first).
A bin left short passes its shortfall to its neighbouring bin(s) (two neighbours: split in half, the odd one to the
neighbour with more episodes left); pass 2 fills the raised quotas the same way. What is still missing is
reported, never filled from other homes' caps.
"""

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.episodes import EpisodeBuilder, box_distance_xz, fingerprint, rounded
from mapmad_sim.objectnav import Goal, OfficialEpisode
from mapmad_sim.robot import LimoSim, horizontal_distance, yaw_towards

RULES = ("start_snap", "no_reachable_instance", "other_storey", "target_visible", "other_instance_visible",
         "distance", "no_goal_photo")
HEADINGS = 8


@dataclass(frozen=True)
class SetRules:
    """The set-building numbers (configs/p5_exam.yaml `set_rules`)."""

    start_snap_max_m: float
    view_point_max_snap_m: float
    same_storey_dy_m: float
    distance_m: Tuple[float, float]
    bins_m: Tuple[float, ...]
    goal_photo_min_share: float
    target_point_max_m: float
    floor_names: Tuple[str, ...]
    goal_photo_box_m: Optional[float] = None  # view point order for the goal photo (None: nearest the box first)

    @classmethod
    def from_config(cls, c: Dict[str, Any]) -> "SetRules":
        return cls(float(c["start_snap_max_m"]), float(c["view_point_max_snap_m"]), float(c["same_storey_dy_m"]),
                   tuple(c["distance_m"]), tuple(c["bins_m"]), float(c["goal_photo_min_share"]),
                   float(c["target_point_max_m"]), tuple(c["floor_names"]),
                   None if c.get("goal_photo_box_m") is None else float(c["goal_photo_box_m"]))


def distance_bin(d: float, bins: Sequence[float]) -> int:
    """Index of the bin [bins[i], bins[i + 1]) holding d (the last bin includes its upper edge)."""
    for i in range(len(bins) - 1):
        if bins[i] <= d < bins[i + 1] or (i == len(bins) - 2 and d == bins[-1]):
            return i
    raise ValueError(f"{d} outside the bins {bins}")


@dataclass
class Instance:
    """One ObjectNav instance with what the rules need (cached per home)."""

    goal: Goal
    box_center: np.ndarray
    box_size: np.ndarray
    view_points: List[np.ndarray]  # usable: snapped, on the real floor (all storeys)
    view_point_snap_m: List[float]


class HomeSetBuilder:
    """Applies the rules to the official episodes of one home (LimoSim with semantic=True)."""

    def __init__(self, robot: LimoSim, goals: Sequence[Goal], rules: SetRules, rng: np.random.Generator) -> None:
        self.robot, self.rules, self.rng = robot, rules, rng
        self.pf = robot.pathfinder
        self.p1 = EpisodeBuilder(robot, {"floor_names": list(rules.floor_names),
                                         "target_point_max_m": rules.target_point_max_m}, rng)
        self.labels = self.p1.labels
        self.goals = list(goals)
        self._instances: Dict[int, Optional[Instance]] = {}
        self._target_points: Dict[int, List[np.ndarray]] = {}

    # --- instances ----------------------------------------------------------------------------------------------
    def instance(self, goal: Goal) -> Optional[Instance]:
        """The instance's box and usable view points; None if the home has no such object."""
        if goal.object_id not in self._instances:
            box = self.labels.box(goal.object_id)
            if box is None:
                self._instances[goal.object_id] = None
            else:
                vps, snaps = [], []
                for vp in goal.view_points:
                    vp = np.asarray(vp, dtype=np.float64)
                    p = np.array(self.pf.snap_point(vp.astype(np.float32)), dtype=np.float64)
                    if not np.isfinite(p).all():
                        continue
                    d = horizontal_distance(p, vp)
                    if d <= self.rules.view_point_max_snap_m and abs(p[1] - vp[1]) <= 0.5 and self.labels.on_real_floor(p):
                        vps.append(p)
                        snaps.append(d)
                self._instances[goal.object_id] = Instance(goal, box[0], box[1], vps, snaps)
        return self._instances[goal.object_id]

    def category_instances(self, category: str) -> List[Instance]:
        return [i for g in self.goals if g.category == category for i in [self.instance(g)] if i is not None]

    def target_points(self, inst: Instance) -> List[np.ndarray]:
        """Phase 1 target-point candidates of an instance (real floor, <= target_point_max_m from the box)."""
        oid = inst.goal.object_id
        if oid not in self._target_points:
            found = self.p1.target(inst.goal)
            self._target_points[oid] = found[1] if found is not None else []
        return self._target_points[oid]

    # --- rules --------------------------------------------------------------------------------------------------
    def snap_start(self, position: Sequence[float]) -> Tuple[Optional[np.ndarray], float]:
        """(snapped start on the real floor or None, horizontal snap distance)."""
        raw = np.asarray(position, dtype=np.float64)
        p = np.array(self.pf.snap_point(raw.astype(np.float32)), dtype=np.float64)
        if not np.isfinite(p).all():
            return None, math.inf
        d = horizontal_distance(p, raw)
        if d > self.rules.start_snap_max_m or abs(p[1] - raw[1]) > 0.5 or not self.labels.on_real_floor(p):
            return None, d
        return p, d

    def nearest_instance(self, start: np.ndarray, insts: List[Instance]) -> Tuple[Optional[Instance], float, str]:
        """(instance, walking distance to its nearest same-storey usable view point, failure rule or '')."""
        best, best_d, any_reachable = None, math.inf, False
        for inst in insts:
            if not inst.view_points:
                continue
            if math.isfinite(self.robot.geodesic_to_any(start, inst.view_points)):
                any_reachable = True
            same = [p for p in inst.view_points if abs(p[1] - start[1]) <= self.rules.same_storey_dy_m]
            d = self.robot.geodesic_to_any(start, same) if same else math.inf
            if d < best_d:
                best, best_d = inst, d
        if best is None:
            return None, math.inf, "other_storey" if any_reachable else "no_reachable_instance"
        return best, best_d, ""

    def heading_shares(self, start: np.ndarray, yaw: float, ids: Sequence[int]) -> np.ndarray:
        """(8, len(ids)) pixel shares of each object in the 8 start headings (yaw + k x 45 deg)."""
        out = np.zeros((HEADINGS, len(ids)))
        for k in range(HEADINGS):
            self.robot.place(start, yaw + 2.0 * math.pi * k / HEADINGS)
            sem = self.robot.observe()["semantic"]
            for j, oid in enumerate(ids):
                out[k, j] = float((sem == oid).mean())
        return out

    def goal_photo(self, inst: Instance) -> Optional[Dict[str, Any]]:
        """The first usable view point (photo_order) whose picture facing the box centre shows >= min share."""
        order = photo_order(inst.view_points, inst.box_center, inst.box_size, self.rules.goal_photo_box_m)
        for rank, i in enumerate(order):
            p = inst.view_points[i]
            yaw = yaw_towards(p, inst.box_center)
            share = self.labels.target_share(p, yaw, inst.goal.object_id)
            if share >= self.rules.goal_photo_min_share:
                obs = self.robot.observe()
                return {"position": rounded(p), "yaw": round(yaw, 4), "share": round(share, 4), "rank": rank,
                        "black_share": round(float((obs["rgb"].max(axis=2) < 16).mean()), 4),
                        "box_distance_m": round(box_distance_xz(p, inst.box_center, inst.box_size), 3)}
        return None

    def evaluate(self, official: OfficialEpisode, set_name: str, split: str) -> Dict[str, Any]:
        """All rule outcomes of one official episode, and the kept episode (with its goal photo) if it passes."""
        fails: List[str] = []
        res: Dict[str, Any] = {"objectnav_episode_id": official.episode_id, "category": official.category}
        start, snap = self.snap_start(official.start_position)
        res["start_snap_m"] = round(snap, 4) if math.isfinite(snap) else None
        if start is None:
            return dict(res, fails=["start_snap"], first_fail="start_snap")
        insts = self.category_instances(official.category)
        target, dist, why = self.nearest_instance(start, insts)
        if target is None:
            return dict(res, fails=[why], first_fail=why)
        ids = [i.goal.object_id for i in insts]
        shares = self.heading_shares(start, official.start_yaw, ids)
        t = ids.index(target.goal.object_id)
        if shares[:, t].max() > 0:
            fails.append("target_visible")
        others = [j for j in range(len(ids)) if j != t]
        if others and shares[:, others].max() > 0:
            fails.append("other_instance_visible")
        lo, hi = self.rules.distance_m
        if not lo <= dist <= hi:
            fails.append("distance")
        photo = self.goal_photo(target)
        if photo is None:
            fails.append("no_goal_photo")
        res.update(walking_m=round(dist, 3), target_object_id=target.goal.object_id,
                   max_target_share=round(float(shares[:, t].max()), 5),
                   max_other_share=round(float(shares[:, others].max()), 5) if others else 0.0)
        if fails:
            return dict(res, fails=fails, first_fail=fails[0])
        return dict(res, fails=[], first_fail=None,
                    episode=self.episode(official, set_name, split, start, snap, target, dist, insts, photo))

    def episode(self, official: OfficialEpisode, set_name: str, split: str, start: np.ndarray, snap: float,
                target: Instance, dist: float, insts: List[Instance], photo: Dict[str, Any]) -> Dict[str, Any]:
        """The kept episode in the closed-loop runner's format (episodes.Episode fields + Phase 5 fields)."""
        point, point_geo, fallback = None, math.inf, False
        for c in self.target_points(target):
            g = self.robot.geodesic(start, c)
            if math.isfinite(g):
                point, point_geo = c, g
                break
        if point is None:  # no reachable Phase 1 target point: the nearest view point stands in
            fallback = True
            point = min(target.view_points, key=lambda p: self.robot.geodesic(start, p))
            point_geo = self.robot.geodesic(start, point)
        home = self.robot.home
        return {
            "episode_id": episode_id(set_name, home, target.goal.category, official.episode_id), "type": set_name,
            "split": split,
            "home": home, "objectnav_episode_id": official.episode_id,
            "target": {"object_id": target.goal.object_id, "category": target.goal.category,
                       "name": self.labels.names.get(target.goal.object_id, ""), "box_center": rounded(target.box_center),
                       "box_size": rounded(target.box_size), "point": rounded(point), "point_fallback": fallback,
                       "view_points": [rounded(p) for p in target.view_points],
                       "view_point_snap_m": [round(d, 4) for d in target.view_point_snap_m]},
            "start_position": rounded(start), "start_yaw": round(official.start_yaw, 4),
            "official_start_position": rounded(official.start_position), "start_snap_m": round(snap, 4),
            "start_geodesic_m": round(dist, 3), "start_target_point_geodesic_m": round(point_geo, 3),
            "distance_bin": distance_bin(dist, self.rules.bins_m),
            "goal_photo_position": photo["position"], "goal_photo_yaw": photo["yaw"], "goal_photo_share": photo["share"],
            "goal_photo_rank": photo["rank"], "goal_photo_black_share": photo["black_share"],
            "goal_photo_box_distance_m": photo["box_distance_m"],
            "other_instances": [{"object_id": i.goal.object_id, "box_center": rounded(i.box_center),
                                 "box_size": rounded(i.box_size)} for i in insts if i is not target],
            "navmesh_sha256": self.robot.navmesh_sha256,
        }


def photo_order(view_points: Sequence[Sequence[float]], box_center: Sequence[float], box_size: Sequence[float],
                target_m: Optional[float]) -> List[int]:
    """View point indices in goal-photo order: horizontal box distance closest to target_m first (None: nearest the
    box first); ties by index."""
    def key(i: int) -> Tuple[float, int]:
        d = box_distance_xz(view_points[i], box_center, box_size)
        return (d if target_m is None else abs(d - target_m), i)

    return sorted(range(len(view_points)), key=key)


def episode_id(set_name: str, home: str, category: str, objectnav_id: str) -> str:
    """p5<d|e>-<home5>-<category>-<ObjectNav id>: ObjectNav (val) reuses ids within a home, once per category, so the
    category is part of the id (bug found 2026-10-12: without it 446 eligible exam episodes had 181 distinct ids)."""
    return f"p5{set_name[0]}-{home[:5]}-{category}-{objectnav_id}"


# --- the draw ------------------------------------------------------------------------------------------------------
def bin_quotas(total: int, n_bins: int) -> List[int]:
    q = [total // n_bins] * n_bins
    for i in range(total - sum(q)):
        q[i] += 1
    return q


def draw(kept: List[Dict[str, Any]], total: int, per_home: int, n_bins: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The seeded draw described in the module docstring -> (chosen episodes sorted by id, report)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(kept))
    by_bin: List[List[Dict[str, Any]]] = [[] for _ in range(n_bins)]
    for i in order:
        by_bin[kept[i]["distance_bin"]].append(kept[i])
    quota = bin_quotas(total, n_bins)
    homes: Counter = Counter()
    chosen: List[List[Dict[str, Any]]] = [[] for _ in range(n_bins)]
    used = set()

    def fill(b: int) -> None:
        for e in by_bin[b]:
            if len(chosen[b]) >= quota[b]:
                return
            if e["episode_id"] not in used and homes[e["home"]] < per_home:
                chosen[b].append(e)
                used.add(e["episode_id"])
                homes[e["home"]] += 1

    first_quota = list(quota)
    for b in sorted(range(n_bins), key=lambda b: (len(by_bin[b]), b)):
        fill(b)
    moved = []
    for b in range(n_bins):
        short = quota[b] - len(chosen[b])
        if short <= 0:
            continue
        quota[b] = len(chosen[b])
        nbrs = [n for n in (b - 1, b + 1) if 0 <= n < n_bins]
        left = {n: sum(e["episode_id"] not in used and homes[e["home"]] < per_home for e in by_bin[n]) for n in nbrs}
        if len(nbrs) == 1:
            give = {nbrs[0]: short}
        else:
            big = max(nbrs, key=lambda n: (left[n], -n))
            give = {n: short // 2 + (short % 2 if n == big else 0) for n in nbrs}
        for n, k in give.items():
            quota[n] += k
            moved.append({"from_bin": b, "to_bin": n, "episodes": k})
    for b in sorted(range(n_bins), key=lambda b: (len(by_bin[b]), b)):
        fill(b)
    out = sorted((e for c in chosen for e in c), key=lambda e: e["episode_id"])
    report = {"seed": seed, "total_asked": total, "per_home_cap": per_home, "first_quota": first_quota,
              "final_quota": quota, "moved": moved, "kept_per_bin": [len(x) for x in by_bin],
              "chosen_per_bin": [len(c) for c in chosen], "chosen": len(out), "missing": total - len(out),
              "chosen_per_home": dict(sorted(Counter(e["home"] for e in out).items()))}
    return out, report


def freeze(path: Path, episodes: List[Dict[str, Any]], extra: Dict[str, Any]) -> str:
    """Write the frozen set (episodes.load_episodes reads it) and <path>.sha256; returns the fingerprint."""
    if path.exists():
        raise FileExistsError(f"{path} exists: the set is frozen")
    dup = sorted(k for k, n in Counter(e["episode_id"] for e in episodes).items() if n > 1)
    if dup:
        raise ValueError(f"duplicate episode ids ({len(dup)}): {dup[:5]}")
    bad = [e["episode_id"] for e in episodes if "__" in e["episode_id"] or "." in e["episode_id"]]
    if bad:  # run ids are <episode_id>__s<k> and file names add suffixes (run_layout.parse_run_id)
        raise ValueError(f"episode ids with '__' or '.': {bad[:5]}")
    fp = fingerprint(episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": fp, **extra, "episodes": episodes}, indent=1) + "\n")
    (path.parent / (path.stem + ".sha256")).write_text(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\nfingerprint {fp}\n")
    return fp


def rules_dict(rules: SetRules) -> Dict[str, Any]:
    return asdict(rules)
