"""Build the goal trail: plan the perfect path, drive it, keep every Nth frame.

This is P2 — the piece that replaces `deployment/src/create_topomap.sh`. On the
real robot a person joysticks the route once while frames are saved; here the
scene's own traversability graph says where the perfect route goes, a small
follower drives it, and the same "save a frame every N ticks" rule turns the
drive into a topomap (plan §5).

What comes out is a directory in `navigate.py`'s exact format — `0.png`,
`1.png`, ... in node order — so `bridge.load_topomap` consumes it unchanged,
plus a `metadata.json` holding everything P3 needs to score the rollout that
follows it: the goal pose to measure success against, the geodesic length that
is SPL's numerator, the spacing the trail was built at, and the camera it was
seen through.

Two things worth knowing about the format:

  * **Frames are saved at the render resolution**, not at any checkpoint's
    `image_size`. One trail then serves every arm, each resizing it to its own
    training resolution — which the fairness protocol (plan §7) requires.
  * **The last node is always the goal**, even when arrival does not land on a
    spacing boundary. For `navigate.py` the last node *is* the goal image, so
    it cannot be whatever frame the modulo happened to pick.

No model and no metrics here: the trail is the task, and scoring a run against
it is P3.
"""

import json
from pathlib import Path

import numpy as np
import yaml

import path_follow
from path_follow import FollowerParams
from pd_control import clip_angle

SIM_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SIM_EVAL_DIR / "configs" / "topomap.yaml"

METADATA_NAME = "metadata.json"
WORLD_CONFIG_NAME = "world.yaml"


class TopomapError(RuntimeError):
    """Raised when a trail cannot be built — a message to a person, not a bug."""


class SamplerParams:
    """What makes a start/goal pair a task rather than a coincidence (plan §5).

    A pair is valid when both ends sit on the nav mesh's largest connected
    component and the geodesic path between them is neither trivially short nor
    longer than the house really affords.
    """

    def __init__(self, min_geodesic_m=3.0, max_geodesic_m=8.0, max_attempts=200):
        self.min_geodesic_m = float(min_geodesic_m)
        self.max_geodesic_m = float(max_geodesic_m)
        self.max_attempts = int(max_attempts)

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))

    def accepts(self, geodesic_m):
        return self.min_geodesic_m <= geodesic_m <= self.max_geodesic_m

    def as_dict(self):
        return {
            "min_geodesic_m": self.min_geodesic_m,
            "max_geodesic_m": self.max_geodesic_m,
            "max_attempts": self.max_attempts,
        }


class AcceptanceParams:
    """When a reference drive is good enough to be a task, and when to retry.

    A start/goal pair being *planned* is not the same as it being *drivable*.
    Gibson's traversability maps are derived from floor plans and do not always
    know about the furniture in the mesh — Rs marks a patch of living room
    traversable that the robot physically climbs onto — so the trail is
    accepted on the evidence of the drive, not of the plan.

    `stuck_ticks` / `stuck_progress_m` are what make that cheap: a wedged robot
    is recognised in a few seconds of sim time instead of burning `max_ticks`.
    """

    def __init__(self, drive_attempts=25, max_collision_ticks=0,
                 stuck_ticks=20, stuck_progress_m=0.05, stuck_progress_rad=0.2):
        self.drive_attempts = int(drive_attempts)
        self.max_collision_ticks = int(max_collision_ticks)
        self.stuck_ticks = int(stuck_ticks)
        self.stuck_progress_m = float(stuck_progress_m)
        self.stuck_progress_rad = float(stuck_progress_rad)

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))

    def as_dict(self):
        return {
            "drive_attempts": self.drive_attempts,
            "max_collision_ticks": self.max_collision_ticks,
            "stuck_ticks": self.stuck_ticks,
            "stuck_progress_m": self.stuck_progress_m,
            "stuck_progress_rad": self.stuck_progress_rad,
        }


class TopomapConfig:
    """Every knob P2 has, read from `configs/topomap.yaml`."""

    def __init__(self, scene_config, scene_id=None, floor=0, start=None, goal=None,
                 seed=0, spacing_ticks=4, max_ticks=400, sampler=None, follower=None,
                 acceptance=None):
        self.scene_config = Path(scene_config)
        if not self.scene_config.is_absolute():
            self.scene_config = SIM_EVAL_DIR / self.scene_config
        self.scene_id = scene_id
        self.floor = int(floor)
        self.start = None if start is None else [float(value) for value in start]
        self.goal = None if goal is None else [float(value) for value in goal]
        self.seed = int(seed)
        self.spacing_ticks = int(spacing_ticks)
        self.max_ticks = int(max_ticks)
        self.sampler = SamplerParams.from_dict(sampler)
        self.follower = FollowerParams.from_dict(follower)
        self.acceptance = AcceptanceParams.from_dict(acceptance)

        if self.spacing_ticks < 1:
            raise TopomapError("spacing_ticks must be at least 1")
        # A zero-tick drive captures no nodes at all, and every later step
        # reads the last one. Refusing here beats an IndexError three calls in.
        if self.max_ticks < 1:
            raise TopomapError("max_ticks must be at least 1")
        if (self.start is None) != (self.goal is None):
            raise TopomapError(
                "start and goal must be given together or not at all — half a "
                "pair is not a task. Set both to null to sample one.")

    @classmethod
    def from_yaml(cls, path=DEFAULT_CONFIG):
        with open(path, "r") as handle:
            return cls(**yaml.safe_load(handle))

    def world_config(self):
        """The iGibson world to load: `scene_config`, with `scene_id` applied.

        Returned as a dict rather than a path so the override is visible in one
        place; `write_world_config` puts it on disk next to the trail, which is
        what actually gets loaded.
        """
        with open(self.scene_config, "r") as handle:
            world = yaml.safe_load(handle)
        if self.scene_id is not None:
            world["scene_id"] = self.scene_id
        return world

    def summary(self):
        return ("scene {} floor {} · spacing {} ticks · geodesic {}-{} m · seed {}"
                .format(self.scene_id or "(from {})".format(self.scene_config.name),
                        self.floor, self.spacing_ticks,
                        self.sampler.min_geodesic_m, self.sampler.max_geodesic_m,
                        self.seed))


def write_world_config(world, output_dir):
    """Save the resolved world beside the trail, and return the path to it.

    A topomap is only reusable if the world it was driven in is pinned with it:
    the same start pose in a different scene is a different task.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / WORLD_CONFIG_NAME
    with open(path, "w") as handle:
        yaml.safe_dump(world, handle, default_flow_style=False, sort_keys=False)
    return path


def plan_path(scene, floor, start_xy, goal_xy):
    """iGibson's A* over the traversability graph: the perfect route, in metres.

    `entire_path=True` returns the whole polyline instead of the task's fixed
    number of waypoints; the geodesic length that comes back is measured on the
    unsubsampled path, so it is the true shortest-path length SPL needs.
    """
    path, geodesic = scene.get_shortest_path(
        floor, np.asarray(start_xy, dtype=float), np.asarray(goal_xy, dtype=float),
        entire_path=True)
    return np.asarray(path, dtype=float), float(geodesic)


def sample_start_goal(scene, floor, sampler):
    """Draw a start/goal pair that is connected and the right length (plan §5).

    `has_node` is checked *before* planning, not for speed: `get_shortest_path`
    grafts any off-graph endpoint onto the graph as a new node, so planning
    from rejected candidates would slowly rewrite the nav mesh underneath the
    sampler. Checking first keeps the graph the scene's, not ours.

    Draws through numpy's global RNG, because `get_random_point` does — seed it
    before calling (`build_topomap` does).
    """
    for _ in range(sampler.max_attempts):
        _floor, start = scene.get_random_point(floor=floor)
        _floor, goal = scene.get_random_point(floor=floor)
        start_xy, goal_xy = start[:2], goal[:2]

        if not (scene.has_node(floor, start_xy) and scene.has_node(floor, goal_xy)):
            continue

        path, geodesic = plan_path(scene, floor, start_xy, goal_xy)
        if sampler.accepts(geodesic):
            return start_xy, goal_xy, path, geodesic

    raise TopomapError(
        "no start/goal pair between {} and {} m after {} attempts. Either the "
        "scene is smaller than the bounds allow, or the bounds are wrong — "
        "widen them in configs/topomap.yaml."
        .format(sampler.min_geodesic_m, sampler.max_geodesic_m,
                sampler.max_attempts))


def resolve_start_goal(scene, config):
    """Use the configured pair if there is one, otherwise sample a valid one."""
    if config.start is None:
        return sample_start_goal(scene, config.floor, config.sampler)

    for name, point in (("start", config.start), ("goal", config.goal)):
        if not scene.has_node(config.floor, point):
            raise TopomapError(
                "configured {} {} is not on the nav mesh's traversable "
                "component, so no path can run through it.".format(name, point))
    path, geodesic = plan_path(scene, config.floor, config.start, config.goal)
    return np.asarray(config.start), np.asarray(config.goal), path, geodesic


def camera_intrinsics(body):
    """The camera the trail was seen through, for P3 and for anything later.

    Intrinsics are not needed to follow a trail, but they are needed to say
    what a pixel in it means — projecting a waypoint into a frame, or comparing
    this camera with the real LoCoBot's.
    """
    renderer = body.env.simulator.renderer
    return {
        "width": int(renderer.width),
        "height": int(renderer.height),
        "vertical_fov_deg": float(renderer.vertical_fov),
        "intrinsic_matrix": [[float(value) for value in row]
                             for row in renderer.get_intrinsics()],
    }


def _pose_record(pose):
    x, y, yaw = pose
    return {"x": float(x), "y": float(y), "yaw": float(yaw)}


def _path_length(poses):
    points = np.asarray([pose[:2] for pose in poses], dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


class Trail:
    """One drive down the reference path: what it captured and how it ended.

    A trail is only a task if the drive actually got there, cleanly. Keeping
    the ending next to the nodes means the caller cannot use half a drive by
    accident — `rejection` says in words why it is not usable, or None.
    """

    def __init__(self, nodes, poses, arrived, stuck, collision_ticks, goal_distance_m):
        self.nodes = nodes
        self.poses = poses
        self.arrived = arrived
        self.stuck = stuck
        self.collision_ticks = collision_ticks
        self.goal_distance_m = float(goal_distance_m)

    @property
    def ticks(self):
        return len(self.poses)

    def rejection(self, acceptance):
        """Why this drive is not a usable task, in words — or None if it is."""
        if self.stuck:
            return ("the follower stopped making progress after {} ticks, {:.2f} m "
                    "from the goal — the planned path runs through something the "
                    "traversability map does not know about"
                    .format(self.ticks, self.goal_distance_m))
        if not self.arrived:
            return ("the follower ran out of ticks after {}, still {:.2f} m from "
                    "the goal".format(self.ticks, self.goal_distance_m))
        if self.collision_ticks > acceptance.max_collision_ticks:
            return ("the drive collided on {} of {} ticks (at most {} allowed) — a "
                    "reference path that scrapes the furniture is not the perfect "
                    "path".format(self.collision_ticks, self.ticks,
                                  acceptance.max_collision_ticks))
        return None


def drive_reference_path(body, path, config, on_node=None):
    """Follow the planned path, keeping the frame every `spacing_ticks` ticks.

    The frame is captured *before* the tick it is labelled with, so node 0 is
    the view from the start pose — the node the bridge is localized to when a
    rollout begins.

    Returns a `Trail`. Writing it out is `write_topomap`'s job, so a caller can
    inspect or reject a drive without one landing on disk.
    """
    follower = config.follower
    acceptance = config.acceptance
    goal_xy = np.asarray(path[-1], dtype=float)
    nodes = []
    poses = []
    target_index = 0
    collision_ticks = 0
    arrived = False
    stuck = False

    def keep_node(tick, pose):
        node = {"node": len(nodes), "tick": int(tick), "pose": list(pose),
                "frame": body.observe()}
        nodes.append(node)
        if on_node is not None:
            on_node(node)

    def is_stuck():
        """Neither moved nor turned over the last `stuck_ticks` ticks.

        Cheaper and more specific than waiting for `max_ticks`: a robot wedged
        against furniture is recognisable in seconds, and the pair can be
        thrown back before the whole tick budget is spent on it.

        Turning counts as progress, and leaving it out is why the first version
        of this rejected most of Rs. A differential drive spins on the spot to
        line up with the path, and a start pose facing away from it needs up to
        pi / max_w = 7.9 s of that — longer than the window. Translation alone
        therefore reads a perfectly healthy turn as a wedged robot. Net yaw is
        used rather than total, so a follower oscillating around its target
        does not look like progress either.
        """
        if len(poses) <= acceptance.stuck_ticks:
            return False
        before = poses[-1 - acceptance.stuck_ticks]
        moved = float(np.linalg.norm(np.asarray(poses[-1][:2]) - np.asarray(before[:2])))
        turned = abs(clip_angle(poses[-1][2] - before[2]))
        return bool(moved < acceptance.stuck_progress_m
                    and turned < acceptance.stuck_progress_rad)

    for tick in range(config.max_ticks):
        pose = body.pose
        poses.append(pose)

        if tick % config.spacing_ticks == 0:
            keep_node(tick, pose)

        if path_follow.has_arrived(pose, goal_xy, follower.arrival_radius_m):
            arrived = True
            break
        if is_stuck():
            stuck = True
            break

        target_index = path_follow.advance_target(
            path, pose[:2], target_index, follower.lookahead_m)
        v, w = path_follow.follow_step(pose, path[target_index], body.limits, follower)
        collision_ticks += int(bool(body.command(v, w)))

    # The last node has to be the goal view whatever the modulo said, because
    # navigate.py treats the last node as the goal image and localizing onto it
    # is how a rollout decides it is done.
    if nodes[-1]["tick"] != len(poses) - 1:
        keep_node(len(poses) - 1, body.pose)

    goal_distance = float(np.linalg.norm(np.asarray(poses[-1][:2]) - goal_xy))
    return Trail(nodes, poses, arrived, stuck, collision_ticks, goal_distance)


def write_topomap(nodes, output_dir):
    """Write the frames as `0.png`, `1.png`, ... — navigate.py's exact format.

    Stale nodes are deleted rather than merged, as `create_topomap.py` does:
    leftovers from a longer trail would silently become part of this one.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()
    for node in nodes:
        node["frame"].save(output_dir / "{}.png".format(node["node"]))


def build_metadata(config, scene_id, trail, path, geodesic, intrinsics, limits):
    """Everything P3 needs to score a rollout against this trail.

    Success is measured against `goal_pose`, SPL against `geodesic_length_m`,
    and neither means anything without the `spacing` and the `world` the trail
    was built at — so all of it travels with the trail rather than living in a
    lab notebook.
    """
    nodes, poses = trail.nodes, trail.poses
    node_points = np.asarray([node["pose"][:2] for node in nodes], dtype=float)
    node_gaps = (np.linalg.norm(np.diff(node_points, axis=0), axis=1)
                 if len(node_points) > 1 else np.zeros(0))

    return {
        "format": "navigate.py topomap: 0.png, 1.png, ... in node order",
        "scene": {
            "id": str(scene_id),
            "floor": config.floor,
            "world_config": WORLD_CONFIG_NAME,
        },
        "start_pose": _pose_record(poses[0]),
        "goal_pose": _pose_record(nodes[-1]["pose"]),
        "planned_goal_xy": [float(value) for value in path[-1]],
        "geodesic_length_m": geodesic,
        "driven_length_m": _path_length(poses),
        "spacing": {
            "ticks_per_node": config.spacing_ticks,
            "seconds_per_node": config.spacing_ticks * limits.dt,
            "median_node_gap_m": float(np.median(node_gaps)) if len(node_gaps) else 0.0,
        },
        "camera": intrinsics,
        "robot": {
            "max_v": limits.max_v,
            "max_w": limits.max_w,
            "frame_rate": limits.frame_rate,
        },
        "follower": config.follower.as_dict(),
        "sampler": config.sampler.as_dict(),
        "acceptance": config.acceptance.as_dict(),
        "seed": config.seed,
        "arrived": bool(trail.arrived),
        "ticks": trail.ticks,
        "collision_ticks": trail.collision_ticks,
        "nodes": [{"node": node["node"], "tick": node["tick"],
                   "pose": _pose_record(node["pose"])} for node in nodes],
        "planned_path": [[float(x), float(y)] for x, y in path],
        "driven_path": [[float(x), float(y), float(yaw)] for x, y, yaw in poses],
    }


def open_body(config, output_dir):
    """Write the resolved world beside the trail, then open the sim on that file.

    Going through the written copy rather than `scene_config` is what makes the
    `world.yaml` in a topomap directory a fact rather than a claim: it is the
    file iGibson loaded.
    """
    # Imported here rather than at module scope so that importing this module
    # stays free of PIL and the simulator stack, the way bridge.py keeps itself
    # free of iGibson — the unit tests need neither.
    import bridge

    world_path = write_world_config(config.world_config(), output_dir)
    return bridge.SimBody(config_path=world_path)


def drive_one_attempt(body, config, scene, on_node=None):
    """Plan a route and drive it once. Returns (trail, path, geodesic).

    Placing the robot is part of the attempt, not of the setup: a sampled start
    can land on furniture too, and then this attempt is simply the one that
    finds that out.
    """
    start_xy, goal_xy, path, geodesic = resolve_start_goal(scene, config)
    yaw = path_follow.initial_yaw(path, config.follower.lookahead_m)
    floor_height = float(scene.floor_heights[config.floor])

    body.reset()
    body.place([float(start_xy[0]), float(start_xy[1]), floor_height], [0.0, 0.0, yaw])

    return drive_reference_path(body, path, config, on_node=on_node), path, geodesic


def build_topomap(body, config, output_dir, on_node=None, on_reject=None):
    """Drive a usable reference path, then write the trail and its metadata.

    A configured start/goal pair gets exactly one attempt — if that specific
    task is not drivable, saying so is more useful than quietly substituting a
    different one. A *sampled* pair is retried up to `drive_attempts` times,
    because the point of sampling is to find a task that works.

    Returns the metadata dict, which is also written to `metadata.json`.
    """
    output_dir = Path(output_dir)
    scene = body.env.scene

    # iGibson samples through numpy's global RNG, and so does the task's own
    # reset — so this one seed fixes the trail, and a rerun reproduces it
    # exactly (plan §7).
    np.random.seed(config.seed)

    attempts = 1 if config.start is not None else config.acceptance.drive_attempts
    rejections = []
    for attempt in range(1, attempts + 1):
        trail, path, geodesic = drive_one_attempt(body, config, scene, on_node=on_node)
        rejection = trail.rejection(config.acceptance)
        if rejection is None:
            break
        rejections.append("attempt {}: {}".format(attempt, rejection))
        if on_reject is not None:
            on_reject(attempt, rejection)
    else:
        raise TopomapError(
            "no drivable reference path in {} after {} attempts. The nav mesh "
            "plans routes the robot cannot physically drive, so the trail is "
            "accepted on the drive rather than the plan — widen the geodesic "
            "bounds or raise drive_attempts in the topomap config.\n  {}"
            .format(scene.scene_id, attempts, "\n  ".join(rejections)))

    write_topomap(trail.nodes, output_dir)
    metadata = build_metadata(config, scene.scene_id, trail, path, geodesic,
                              camera_intrinsics(body), body.limits)
    metadata["rejected_attempts"] = rejections
    (output_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_metadata(topomap_dir):
    """Read a trail's metadata — the counterpart to `bridge.load_topomap`."""
    path = Path(topomap_dir) / METADATA_NAME
    if not path.exists():
        raise TopomapError(
            "{} has no {}, so the trail's goal pose and geodesic length are "
            "unknown. Rebuild it with p2_1_build_test.py."
            .format(topomap_dir, METADATA_NAME))
    return json.loads(path.read_text())
