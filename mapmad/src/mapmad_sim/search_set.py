"""Phase 11 search episode sets: the DEV set (plan v2 §8.7 builder on the 10 Phase 1 dev homes; reused by Phase 12 as
its dev set) and the ROLLOUT set (DAgger starts in the 145 labelled train homes, Phase 8 search rules).

One episode format for both (the closed-loop runner and the simulator server read it; sidecar_server Phase 11):
    episode_id, type ("dev" | "rollout"), split, home, start_position, start_yaw, start_geodesic_m, distance_bin,
    navmesh_sha256, target = {category, category_ids (every ObjectNav goal instance of the category: detections count
    them all), instances [{object_id, centre, size, view_points}] (instances with a usable view point: the "seen" set
    and the success zone), view_points (all their usable view points: success = walking distance <= 0.2 m to the
    nearest), point (the nearest one from the start: `geodesic_m` in the server state)}.
Target = ANY instance of the category (plan D5).

Dev set (`DevSetBuilder.evaluate`, one official ObjectNav v2 TRAIN episode of a dev home; pin025 camera; rules in
this order, the first failing one is counted):
1. start_snap: the official start snapped onto LIMO's floor map <= 0.2 m away, on the real floor (exam_set rule);
2. no_reachable_instance / other_storey: some usable view point of an instance of the category on the start storey
   is reachable (exam_set.HomeSetBuilder.nearest_instance);
3. instance_visible: an instance of the category shows (any pixel) in one of 8 headings at the start;
4. distance: walking distance to the nearest usable view point outside 3-15 m;
5. not_seeable: Phase 8 start filter -- from >= 1 usable view point of >= 1 instance, one of 8 headings shows an
   instance with >= 200 px at <= 4 m (datagen_v2.HomeTasks rule; once per home and category).
Draw: exam_set.draw (seeded, <= per_home per home, distance bins 3-6 / 6-10 / 10-15 m with neighbour-bin fill).
Ids `p11d-<home5>-<category>-<ObjectNav id>`.

Rollout set (`rollout_episodes`): per home, n draws with datagen_v2.HomeGeneratorV2.sample_search (category uniform
among the home's seeable categories, start on the real floor, 3-15 m, no instance visible in 8 headings; wall starts
as Phase 8's variations), each draw seeded by datagen_v2.drive_seed(seed, "p11r-<home5>-<n>"). Ids
`p11r-<home5>-<category>-<n:03d>`.
"""

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.episodes import rounded
from mapmad_sim.exam_set import HomeSetBuilder, Instance, SetRules, distance_bin
from mapmad_sim.objectnav import Goal, OfficialEpisode
from mapmad_sim.target_layer import NoiseSpec, detect

DEV_RULES = ("start_snap", "no_reachable_instance", "other_storey", "instance_visible", "distance", "not_seeable")
HEADINGS = 8


def dev_episode_id(home: str, category: str, objectnav_id: str) -> str:
    return f"p11d-{home[:5]}-{category}-{objectnav_id}"


def set_episode_id(prefix: str, home: str, category: str, objectnav_id: str) -> str:
    """<prefix>-<home5>-<category>-<id> (p11d / p12e; OVON: p12o-<split>); a category's spaces, dots and other
    characters become "_" (ids name files: no "__" and no ".")."""
    cat = "_".join(w for w in re.split(r"[^A-Za-z0-9]+", category) if w)
    return f"{prefix}-{home[:5]}-{cat}-{objectnav_id}"


def rollout_sample_id(home: str, n: int) -> str:
    """The id that seeds a rollout draw (the category is not known before the draw)."""
    return f"p11r-{home[:5]}-{n:03d}"


def rollout_episode_id(home: str, category: str, n: int) -> str:
    return f"p11r-{home[:5]}-{category}-{n:03d}"


def target_block(category: str, category_ids: Sequence[int], instances: Sequence[Dict[str, Any]],
                 point: Sequence[float]) -> Dict[str, Any]:
    """The episode's `target` (module docstring)."""
    return {"category": category, "category_ids": sorted(int(i) for i in category_ids),
            "instances": list(instances), "view_points": [vp for i in instances for vp in i["view_points"]],
            "point": rounded(point)}


def instance_dict(inst: Instance) -> Dict[str, Any]:
    return {"object_id": inst.goal.object_id, "centre": rounded(inst.box_center), "size": rounded(inst.box_size),
            "view_points": [rounded(p) for p in inst.view_points]}


class DevSetBuilder:
    """The dev-set rules on one home (robot: LimoSim with the pin025 camera, semantic=True). Phase 12 builds the exam
    sets with the same rules: `id_prefix` (p12e / p12o-<split>), `set_type`, `prompt_of` (OVON: the recogniser's word
    per category key) and `keep_unseeable` (an episode failing ONLY the start filter is kept as "unseeable_episode",
    the report-only unseeable set, block item 5)."""

    def __init__(self, robot: Any, goals: Sequence[Goal], rules: SetRules, seen_rule: NoiseSpec,
                 id_prefix: str = "p11d", set_type: str = "dev", keep_unseeable: bool = False,
                 prompt_of: Optional[Dict[str, str]] = None) -> None:
        self.robot, self.rules, self.seen_rule = robot, rules, seen_rule
        self.base = HomeSetBuilder(robot, goals, rules, np.random.default_rng(0))
        self.goals = list(goals)
        self._seeable: Dict[str, bool] = {}
        self.id_prefix, self.set_type, self.keep_unseeable = id_prefix, set_type, keep_unseeable
        self.prompt_of = prompt_of

    def category_ids(self, category: str) -> List[int]:
        return sorted({g.object_id for g in self.goals if g.category == category})

    def seeable(self, category: str) -> bool:
        """Phase 8 start filter for one category (view points in instance order, headings k x 45 deg)."""
        if category not in self._seeable:
            ids = [i.goal.object_id for i in self.base.category_instances(category) if i.view_points]
            self._seeable[category] = False
            for inst in self.base.category_instances(category):
                for vp in inst.view_points:
                    for k in range(HEADINGS):
                        self.robot.place(np.asarray(vp, dtype=np.float64), 2.0 * math.pi * k / HEADINGS)
                        obs = self.robot.observe()
                        if detect(obs["semantic"], obs["depth"], ids, self.seen_rule):
                            self._seeable[category] = True
                            return True
        return self._seeable[category]

    def any_visible(self, start: np.ndarray, yaw: float, ids: Sequence[int]) -> bool:
        """An instance of the category has >= 1 pixel in one of the 8 start headings."""
        for k in range(HEADINGS):
            self.robot.place(start, yaw + 2.0 * math.pi * k / HEADINGS)
            if np.isin(self.robot.observe()["semantic"], list(ids)).any():
                return True
        return False

    def evaluate(self, official: OfficialEpisode, split: str) -> Dict[str, Any]:
        """Rule outcomes of one official episode, and the kept episode if it passes (module docstring)."""
        res: Dict[str, Any] = {"objectnav_episode_id": official.episode_id, "category": official.category}
        start, snap = self.base.snap_start(official.start_position)
        res["start_snap_m"] = round(snap, 4) if math.isfinite(snap) else None
        if start is None:
            return dict(res, fails=["start_snap"], first_fail="start_snap")
        insts = [i for i in self.base.category_instances(official.category) if i.view_points]
        nearest, dist, why = self.base.nearest_instance(start, insts)
        if nearest is None:
            return dict(res, fails=[why], first_fail=why)
        fails = []
        ids = self.category_ids(official.category)
        if self.any_visible(start, official.start_yaw, ids):
            fails.append("instance_visible")
        lo, hi = self.rules.distance_m
        if not lo <= dist <= hi:
            fails.append("distance")
        if not self.seeable(official.category):
            fails.append("not_seeable")
        res["walking_m"] = round(dist, 3)
        unseeable = self.keep_unseeable and fails == ["not_seeable"]
        if fails and not unseeable:
            return dict(res, fails=fails, first_fail=fails[0])
        same = [p for p in nearest.view_points if abs(p[1] - start[1]) <= self.rules.same_storey_dy_m]
        point = min(same, key=lambda p: self.robot.geodesic(start, p))
        home = self.robot.home
        target = target_block(official.category, ids, [instance_dict(i) for i in insts], point)
        if self.prompt_of is not None:
            target["prompt"] = self.prompt_of[official.category]
        episode = {
            "episode_id": set_episode_id(self.id_prefix, home, official.category, official.episode_id),
            "type": self.set_type, "split": split, "home": home, "objectnav_episode_id": official.episode_id,
            "target": target,
            "start_position": rounded(start), "start_yaw": round(official.start_yaw, 4),
            "official_start_position": rounded(official.start_position), "start_snap_m": round(snap, 4),
            "start_geodesic_m": round(dist, 3), "distance_bin": distance_bin(dist, self.rules.bins_m),
            "navmesh_sha256": self.robot.navmesh_sha256}
        if unseeable:
            return dict(res, fails=fails, first_fail=fails[0], unseeable_episode=episode)
        return dict(res, fails=[], first_fail=None, episode=episode)


def rollout_episodes(gen: Any, home: str, n: int, bins_m: Sequence[float]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """n rollout episodes of one home from a datagen_v2.HomeGeneratorV2 (cfg seed = the rollout seed); returns
    (episodes, per-draw notes). A draw without a start (no_sample) is reported and leaves its number unused."""
    from mapmad_sim.datagen_v2 import drive_seed
    from mapmad_sim.route import draw_variations

    out, notes = [], []
    for k in range(n):
        sid = rollout_sample_id(home, k)
        rng = np.random.default_rng([drive_seed(gen.cfg["seed"], sid), 0])
        gen.sampler.rng = rng
        var = draw_variations(gen.cfg, rng)
        note: Dict[str, Any] = {"sample_id": sid, "wall_start_asked": bool(var.wall_recovery)}
        sampled = gen.sample_search(var, rng, note)
        if sampled is None:
            notes.append(dict(note, outcome="no_sample"))
            continue
        start, task = sampled
        vps = [vp for inst in task.instances for vp in inst.view_points if gen.sampler.same_floor(start.position, vp)]
        point = min(vps, key=lambda p: gen.robot.geodesic(start.position, p))
        insts = [{"object_id": i.object_id, "centre": rounded(i.centre), "size": rounded(i.size),
                  "view_points": [rounded(p) for p in i.view_points]} for i in task.instances]
        eid = rollout_episode_id(home, task.category, k)
        out.append({"episode_id": eid, "type": "rollout", "split": gen.cfg["split"], "home": home, "sample_id": sid,
                    "target": target_block(task.category, task.all_ids, insts, point),
                    "start_position": rounded(start.position), "start_yaw": round(float(start.yaw), 4),
                    "wall_start": bool(start.wall_recovery), "start_geodesic_m": task.geodesic_m,
                    "distance_bin": distance_bin(min(max(task.geodesic_m, bins_m[0]), bins_m[-1]), bins_m),
                    "navmesh_sha256": gen.robot.navmesh_sha256})
        notes.append(dict(note, outcome="kept", episode_id=eid))
    return out, notes


def search_set_ok(episodes: Sequence[Dict[str, Any]]) -> Optional[str]:
    """None if the set can run: unique ids without '__' or '.', every episode with instances and view points."""
    ids = [e["episode_id"] for e in episodes]
    if len(set(ids)) != len(ids):
        return "duplicate episode ids"
    if any("__" in i or "." in i for i in ids):
        return "episode ids with '__' or '.'"
    bad = [e["episode_id"] for e in episodes if not e["target"]["view_points"]]
    return f"episodes without a usable view point: {bad[:5]}" if bad else None
