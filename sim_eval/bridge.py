"""The bridge: NoMaD's brain driving the simulator's body.

Open-loop evaluation replays frames past a model. This closes the loop — the
model's action moves the agent, iGibson renders the consequence, and the model
sees it. One tick here is one tick of `deployment/src/navigate.py` plus
`pd_controller.py` (both ported in `nomad_policy.py` and `pd_control.py`), with
only the two ends swapped (plan §4):

    ROS camera topic   ->  iGibson's rendered RGB frame
    /cmd_vel publisher ->  iGibson's differential-drive controller

The tick, in order:

    grab RGB -> resize to the checkpoint's own image_size -> rolling context
    queue (context_size+1 frames, at the checkpoint's own stride) -> goal
    mask 0 -> distance head localizes in the topomap -> subgoal -> 8
    diffusion samples -> waypoint #2 -> un-normalize into metres -> PD
    controller -> (v, w) -> step the sim one 4 Hz period

Two things the sim forces on us, both recorded here rather than buried:

  * **The queue is seeded, not filled.** navigate.py waits for the real
    camera's stream to accumulate context_size+1 frames before acting. There
    is no stream to wait for at t=0 in a sim, so the episode starts with
    frame 0 repeated — the standard cold-start for a frame-stack policy.
  * **iGibson's LoCoBot turns the wrong way.** See ANGULAR_VELOCITY_SIGN.

**The bridge makes ticks; it does not run episodes.** `tick()` is the whole of
its loop-facing interface. When an episode is over — a goal radius reached, a
tick budget spent — is `episode_runner`'s business, and there is deliberately
no second answer to it here. There used to be: a `run()` that stopped when the
distance head localized onto the last node, which is `navigate.py`'s rule on
the real robot and is a claim about where the agent *thinks* it is. Ending an
episode there lets a lost agent declare victory and stops the odometer early,
inflating its SPL. `reached_goal` survives as that claim, and the episode
records it as a diagnostic rather than obeying it.

No metrics and no episode sampling live here; they are P3.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from pd_control import RobotLimits, pd_controller

SIM_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_SIM_CONFIG = SIM_EVAL_DIR / "configs" / "locobot_rs_bridge.yaml"

# iGibson's physics rate. Only the control rate mirrors the robot; the
# integrator underneath runs at iGibson's default.
PHYSICS_TIMESTEP = 1 / 240.0


# iGibson 2.2.2's LoCoBot URDF tilts the camera down by a *fixed* joint:
# `head_tilt_joint` has rpy="0 0.3490658503988659 0" — 20 degrees, which reads
# as ~21 at rest once the base settles. It is fixed, so it cannot be driven;
# `camera_tilt_deg` in a world config re-aims the camera on the render side
# instead (see `SimBody.observe`).
URDF_CAMERA_TILT_DEG = 20.0


def tilted_camera_axes(rotation, tilt_change_deg):
    """The camera's (view, up) directions after tilting it by `tilt_change_deg`.

    iGibson aims the robot camera along the eyes link's local +x, with local +z
    as up (`MeshRenderer.render_single_robot_camera`). Tilting is a rotation
    about the camera's own local y axis, and positive is *down* — the sign the
    URDF's own 20-degree tilt has. The rotation is about the camera's optical
    centre, so the view changes and the camera's height does not: P5's pitch
    experiment changes one thing.
    """
    angle = np.radians(tilt_change_deg)
    about_local_y = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    tilted = np.asarray(rotation, dtype=float) @ about_local_y
    return tilted @ np.array([1.0, 0.0, 0.0]), tilted @ np.array([0.0, 0.0, 1.0])


def frame_to_pil(rgb):
    """iGibson's RGB frame as a PIL image — the sim's answer to `msg_to_pil`.

    The renderer hands back float32 in [0, 1] and sometimes a fourth alpha
    channel; PIL and the ImageNet transform both want 8-bit RGB. The frame is
    a view into renderer-owned memory, so this also copies it.
    """
    pixels = (np.clip(np.asarray(rgb)[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def waypoint_scale_m(model_params, limits):
    """Metres per normalized waypoint unit — navigate.py's `MAX_V / RATE`.

    `get_action` returns cumulative, normalized position deltas, and one unit is
    how far the robot travels in one control period at its top speed. A
    checkpoint trained without normalization already emits metres, so the scale
    is 1.

    It is a function rather than a line inside the tick because two places need
    the same conversion and they must not drift: the bridge converts the one
    waypoint it acts on, and P4's overlay converts all eight sampled
    trajectories to draw them beside it.
    """
    if not model_params["normalize"]:
        return 1.0
    return limits.max_v / limits.frame_rate


def contact_bearings_deg(points_xy, pose):
    """Where each contact is, as a bearing from the robot's heading, in degrees.

    0 is dead ahead, +90 the robot's left, -90 its right, +-180 behind it —
    the ROS convention everything else here uses. P5 needs it to tell a robot
    that drove into something it could see from one that clipped something at
    its shoulder, outside the camera: the difference between a model that
    ignored an obstacle and a camera that never showed it one.

    Pure arithmetic, so it is pinned without a simulator; `SimBody` supplies
    the points.
    """
    x, y, yaw = pose
    bearings = []
    for point_x, point_y in points_xy:
        world = np.arctan2(point_y - y, point_x - x)
        relative = (world - yaw + np.pi) % (2 * np.pi) - np.pi
        bearings.append(float(np.degrees(relative)))
    return bearings


def load_topomap(topomap_dir):
    """Load a topomap directory of 0.png, 1.png, ... in node order.

    Sorted numerically, not lexicographically, exactly as navigate.py does it:
    otherwise node 10 sorts before node 2 and the trail is scrambled.
    """
    topomap_dir = Path(topomap_dir)
    filenames = sorted(
        (path for path in topomap_dir.iterdir() if path.suffix == ".png"),
        key=lambda path: int(path.stem),
    )
    if not filenames:
        raise FileNotFoundError("no numbered PNG nodes in {}".format(topomap_dir))
    return [Image.open(path).convert("RGB") for path in filenames]


class ContextQueue:
    """The rolling observation window: context_size past frames plus the current.

    navigate.py's `context_queue` and `callback_obs`, with `seed` added for the
    cold start (see the module docstring), and `stride` added for checkpoints
    trained on spaced-out context: the frames fed are the current tick and the
    ones `stride`, `2 * stride`, ... `context_size * stride` ticks before it —
    training's `_context_times`. To serve that it keeps every tick of the last
    `context_size * stride`, and feeds `capacity` of them. At stride 1 the two
    are the same thing and this is navigate.py's queue exactly.
    """

    def __init__(self, context_size, stride=1):
        if stride < 1:
            raise ValueError("context stride must be >= 1, got {}".format(stride))
        self.capacity = context_size + 1
        self.stride = int(stride)
        self._history_length = context_size * self.stride + 1
        self._history = []

    def seed(self, frame):
        """Start an episode with no history by repeating the first frame.

        Every tick of the history is frame 0, so every spaced slot is too, and
        the first `context_size * stride` pushes displace it at training's spacing.
        """
        self._history = [frame] * self._history_length

    def push(self, frame):
        self._history.append(frame)
        if len(self._history) > self._history_length:
            self._history.pop(0)

    @property
    def frames(self):
        """Oldest first, which is the order the vision encoder expects."""
        return self._history[::-1][::self.stride][::-1]

    @property
    def is_ready(self):
        return len(self._history) == self._history_length


class SimBody:
    """iGibson's end of the bridge: it renders frames and executes (v, w).

    The *robot* half of the simulator adapter. World questions — the floor's
    height, what is traversable, how far two points are around the furniture —
    belong to `self.scene` (`sim_scene.SimScene`), which this opens and binds
    to a floor. `_env` is private on purpose: callers reaching through it into
    iGibson is the friction both halves exist to remove."""

    # iGibson 2.2.2's Locobot declares `base_control_idx = [1, 0]` for
    # [left, right], but its URDF lists wheel_left_joint first — so the
    # differential drive controller's left-wheel velocity is applied to the
    # physical right wheel. The magnitude is right and the sign is not:
    # commanding w=+0.4 rad/s measures as -0.39 rad/s of yaw (probe in the P1
    # notes). NoMaD inherits the ROS convention, where +w turns left, and so
    # does every turn in its training data — so a bridge that passed w through
    # would mirror every turn and could never follow a trail.
    #
    # This flip is a wiring correction on the body side, not a tuned gain:
    # NoMaD's (v, w) is untouched, and plan decision E still holds.
    ANGULAR_VELOCITY_SIGN = -1.0

    def __init__(self, config_path=DEFAULT_SIM_CONFIG, limits=None,
                 mode="headless", floor=0):
        # Imported here, not at module scope, so that importing this module
        # does not pull in iGibson (and its GPU renderer) — the unit tests in
        # sim_eval/tests exercise the queue and the controller without a GPU.
        from igibson.envs.igibson_env import iGibsonEnv

        from sim_scene import SimScene

        self.limits = limits or RobotLimits.from_config()
        self.config_path = Path(config_path)
        # This is the "step the sim at 4 Hz" of the control tick: one env.step
        # advances the world by exactly one control period of the real robot.
        self._env = iGibsonEnv(
            config_file=str(self.config_path),
            mode=mode,
            action_timestep=self.limits.dt,
            physics_timestep=PHYSICS_TIMESTEP,
        )
        # The world half of the adapter, with the floor already bound. Callers
        # ask it rather than reaching through this object into iGibson — see
        # sim_scene.py for what that cost before it existed.
        self.scene = SimScene(self._env.scene, floor)
        # None = the URDF's own tilt, rendered by iGibson's own path, exactly as
        # every phase before the pitch experiment ran. A number re-aims the
        # camera on the render side. It lives in the world config, so it is
        # saved beside every trail and covered by the task-set fingerprint.
        tilt = self._env.config.get("camera_tilt_deg")
        self.camera_tilt_deg = None if tilt is None else float(tilt)

    @property
    def robot(self):
        return self._env.robots[0]

    @property
    def intrinsics(self):
        """The camera this body sees through, for anything that must say what a
        pixel means — projecting a waypoint into a frame, or comparing this
        camera with the real LoCoBot's.

        A renderer question, so it belongs to the body (which owns the camera)
        rather than to the scene.
        """
        renderer = self._env.simulator.renderer
        return {
            "width": int(renderer.width),
            "height": int(renderer.height),
            "vertical_fov_deg": float(renderer.vertical_fov),
            "intrinsic_matrix": [[float(value) for value in row]
                                 for row in renderer.get_intrinsics()],
        }

    def render_at_vertical_fov(self, degrees):
        """Render one frame at a different vertical field of view, then restore.

        The camera's field of view is the one thing about the sim's optics that
        `configs/locobot_rs_bridge.yaml` cannot mirror from the real robot,
        because GoStanford records no intrinsics — so P5 has to *show* the
        difference rather than argue about it, by rendering the same pose
        through several lenses and putting them beside a training frame.

        It is deliberately a render, not a setting: the FOV is put back before
        this returns, so nothing a diagnostic looks at can leave the rollout
        camera changed behind it. Changing the camera an episode runs under is
        a config edit (`vertical_fov`), reviewed as such.
        """
        renderer = self._env.simulator.renderer
        original = float(renderer.vertical_fov)
        try:
            renderer.set_fov(float(degrees))
            return self.observe()
        finally:
            renderer.set_fov(original)

    def render_at_camera_tilt(self, degrees):
        """Render one frame with the camera tilted down `degrees`, changing nothing.

        P5's pitch experiment renders the same pose at several tilts and asks
        the observation encoder which one looks like training — the pitch
        counterpart of `render_at_vertical_fov`. Every tilt in a sweep goes
        through the same render path, including the URDF's own 20, so the rows
        of the sweep differ in the tilt and in nothing else.
        """
        return self._render_rgb(degrees)

    def _render_rgb(self, tilt_deg):
        """iGibson's robot-camera render, with the camera re-aimed.

        `MeshRenderer.render_single_robot_camera`, line for line, except that
        the view and up directions come from `tilted_camera_axes` rather than
        straight off the eyes link — and only RGB is rendered, which is all
        `observe` returns. The robot's physics body, collision meshes and
        camera position are untouched.
        """
        from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz

        renderer = self._env.simulator.renderer
        eyes = self.robot.eyes
        camera_pos = eyes.get_position()
        rotation = quat2rotmat(xyzw2wxyz(eyes.get_orientation()))[:3, :3]
        view, up = tilted_camera_axes(rotation, tilt_deg - URDF_CAMERA_TILT_DEG)
        renderer.set_camera(camera_pos, camera_pos + view, up)
        hidden = (self.robot.renderer_instances
                  if renderer.rendering_settings.hide_robot else [])
        return frame_to_pil(renderer.render(modes=("rgb",), hidden=hidden)[0])

    @property
    def initial_pose(self):
        """Where the simulator's own task placed the robot: (position, orientation).

        Used only by P1's hand-made trail, which drives from wherever the reset
        put the robot. Every later phase places the robot itself, from a task's
        recorded start pose.
        """
        task = self._env.task
        return ([float(value) for value in task.initial_pos],
                [float(value) for value in task.initial_orn])

    def verify_gpu(self, expected_gpu):
        """Refuse to run if EGL is not rendering where we pinned it.

        `cukurovaai` is shared, and a render that lands on a teammate's GPU
        looks perfect from here. The check used to be spelled
        `gpu.verify_renderer(body.env.simulator.renderer, selected_gpu)` at five
        call sites, each reaching three levels into the simulator; it is one
        call now, and the renderer stays private.
        """
        import gpu

        return gpu.verify_renderer(self._env.simulator.renderer, expected_gpu)

    def reset(self):
        """Reset the episode. Leaves the robot wherever the task put it."""
        self._env.reset()

    def place(self, position, orientation):
        """Put the robot at a pose and let it settle onto the floor."""
        self._env.land(self.robot, np.asarray(position), np.asarray(orientation))
        self._env.simulator.sync(force_sync=True)

    def observe(self):
        """The current camera frame, as a PIL image.

        With no `camera_tilt_deg` in the world config this is iGibson's own
        sensor path, unchanged since P1 — so every earlier result still stands
        on the code that produced it. With one, the camera is re-aimed on the
        render side (`_render_rgb`).
        """
        if self.camera_tilt_deg is None:
            return frame_to_pil(self._env.get_state()["rgb"])
        return self._render_rgb(self.camera_tilt_deg)

    def command(self, v, w):
        """Execute (v, w) for one control period. Returns True if it collided.

        It *reports* the contact and does not count it. A body has no business
        knowing how it is being scored, and it used to keep a tally that two
        later tallies superseded — one in the reference drive, one in the
        episode — while still looking authoritative on the object every caller
        holds. Collisions are counted where they are interpreted (plan §6:
        count and continue).
        """
        action = np.array([v, self.ANGULAR_VELOCITY_SIGN * w])
        _state, _reward, _done, _info = self._env.step(action)
        return len(self._env.collision_links) > 0

    def contact_points(self):
        """Where the robot is touching something right now, as world (x, y).

        iGibson keeps the last physics step's contacts as pybullet contact
        tuples queried with the robot as body A, so field 5 is the point on
        the robot's own surface. Already filtered by the world config's
        `collision_ignore_link_a_ids` — the same filter that decides whether
        `command` reports a collision at all, so the two cannot disagree.
        """
        return [(float(item[5][0]), float(item[5][1]))
                for item in self._env.collision_links]

    @property
    def pose(self):
        """(x, y, yaw) in scene coordinates."""
        # Imported here for the same reason as iGibson above: so that importing
        # this module stays free of the simulator stack, and ContextQueue can be
        # tested without it.
        import pybullet

        x, y, _z = self.robot.get_position()
        yaw = pybullet.getEulerFromQuaternion(self.robot.get_orientation())[2]
        return float(x), float(y), float(yaw)

    def close(self):
        self._env.close()


class TickRecord:
    """Everything one control tick decided, for eyeballing and for replay.

    The policy's own output is *held*, not copied out field by field. Copying
    is how this class used to lose things: `PolicyStep` carries `distances` and
    `samples` — the temporal distance to every node in the localization window,
    and all eight diffusion trajectories — which `nomad_policy` annotates as
    being "for inspection and overlays", and which a nine-field copy silently
    dropped. A consumer that wants to draw what the model was thinking
    (P4's overlays, P5's multimodality) reaches through `record.step`.

    `waypoint_m` is not on the step because it is not the policy's: the policy
    emits normalized units and the bridge converts them into metres.

    **Nothing may retain a record.** It pins a decoded camera frame — 0.92 MB
    at the render resolution — so a held episode's worth is hundreds of
    megabytes. The episode yields these one at a time and keeps none; see
    `episode_runner.Episode`.
    """

    def __init__(self, index, frame, pose, step, waypoint_m, v, w, collided,
                 contact_bearings_deg=()):
        self.index = index
        self.frame = frame
        self.pose = pose
        # The PolicyStep this tick acted on: closest_node, subgoal_node,
        # distances, samples, and the raw normalized waypoint.
        self.step = step
        self.waypoint_m = waypoint_m
        self.v = v
        self.w = w
        self.collided = collided
        # Where the contact was, relative to the heading after the tick's
        # motion — see `contact_bearings_deg`. Empty on a tick with no contact.
        self.contact_bearings_deg = list(contact_bearings_deg)

    def summary(self):
        return ("tick {:3d}  node {:>2} -> subgoal {:>2}  waypoint "
                "({:+.3f}, {:+.3f}) m  v={:.3f} w={:+.3f}{}".format(
                    self.index, self.step.closest_node, self.step.subgoal_node,
                    self.waypoint_m[0], self.waypoint_m[1], self.v, self.w,
                    "  COLLISION" if self.collided else ""))


class NomadBridge:
    """One NoMaD checkpoint driving one iGibson body along one topomap."""

    def __init__(self, policy, body):
        self.policy = policy
        self.body = body
        self.limits = body.limits
        self.context = ContextQueue(policy.spec.context_size,
                                    policy.spec.context_stride)
        self._encoded_topomap = None
        self._goal_node = None
        self._closest_node = 0
        self._tick_index = 0

    def start_episode(self, topomap, start_pose=None):
        """Load a topomap, place the robot, and seed the context queue.

        `start_pose` is (position, orientation) — normally the pose the topomap
        was recorded from, so the trail begins under the robot's nose. Without
        it the robot stays wherever the reset put it.
        """
        self.body.reset()
        if start_pose is not None:
            self.body.place(*start_pose)

        self._encoded_topomap = self.policy.encode_topomap(topomap)
        # navigate.py's `--goal-node -1`: the last node of the trail.
        self._goal_node = len(topomap) - 1
        self._closest_node = 0
        self._tick_index = 0
        self.context.seed(self.body.observe())

    @property
    def reached_goal(self):
        """navigate.py's own test: localized onto the final node."""
        return self._closest_node == self._goal_node

    def _waypoint_to_metres(self, waypoint):
        """navigate.py: `chosen_waypoint[:2] *= (MAX_V / RATE)` when normalized.

        Only the position half is scaled: a 4-element waypoint carries a
        heading in its last two, which is a direction and has no units to
        convert.
        """
        waypoint = np.array(waypoint, dtype=float)
        waypoint[:2] *= waypoint_scale_m(self.policy.model_params, self.limits)
        return waypoint

    def tick(self):
        """One control tick. Returns what it decided, after acting on it."""
        frame = self.context.frames[-1]
        pose = self.body.pose

        step = self.policy.act(
            self.context.frames, self._encoded_topomap,
            self._closest_node, self._goal_node)
        self._closest_node = step.closest_node

        waypoint_m = self._waypoint_to_metres(step.waypoint)
        v, w = pd_controller(waypoint_m, self.limits)
        collided = self.body.command(v, w)
        bearings = (contact_bearings_deg(self.body.contact_points(), self.body.pose)
                    if collided else [])

        self.context.push(self.body.observe())
        record = TickRecord(
            index=self._tick_index, frame=frame, pose=pose, step=step,
            waypoint_m=waypoint_m, v=float(v), w=float(w), collided=collided,
            contact_bearings_deg=bearings)
        self._tick_index += 1
        return record
