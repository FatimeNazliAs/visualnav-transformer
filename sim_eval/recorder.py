"""Film an episode while it runs: three panels, one MP4, nothing kept.

P3 turned a drive into a row in a table. A row says an episode failed; it does
not say *why*, and "why" is the whole of P5. This is the phase that makes an
episode watchable — plan §9's P4 and §11's "watchable per-episode videos for
every run" — and it is deliberately a **layer over** the scorer rather than a
change to it. No metric is computed here, none is read, and the runner does not
know this module exists.

The seam it attaches to is the one P3.5 opened: an episode is a *stream*, so
the consumer holds the tick loop and can do whatever it likes with each record
as it arrives (see `episode_runner.Episode`). Recording is one more line in
that loop:

    with recorder.episode(task, checkpoint, scene) as film:
        for record in episode:
            film.capture(record)

**Off costs nothing.** `Recorder.episode` hands back a `NullRecording` whose
`capture` is a `pass` when recording is disabled or the task is not in the
subset, so an unrecorded run executes exactly the code P3 executed, at P3's
speed. There is no `if recording:` inside the tick loop to get out of step with
the config.

**Three panels, in sync, from one tick's record:**

    camera          what NoMaD actually saw this tick — the frame that was fed
                    to the encoder, at render resolution.
    top-down map    the house's traversability map, the reference trail it was
                    given, the path it has driven so far, where it has hit
                    something, and which node it is currently steering at.
    overlay         the subgoal image it is steering *towards*, the distance
                    head's opinion of how far away that is, and all eight
                    diffusion samples in the robot's own frame with waypoint #2
                    — the one the PD controller acts on — picked out.

**Nothing accumulates except two floats per tick.** A `TickRecord` pins a
0.92 MB decoded frame; the film draws it, encodes it and lets it go. What it
keeps is the pose trail (and the collision points), which is what the map panel
has to redraw. A 400-tick episode therefore costs kilobytes of memory and one
file on disk, which is the same invariant `episode_runner` holds itself to.

**The figure is built once per episode and updated per tick.** Matplotlib is
fast at `set_data` and slow at `subplots`; at hundreds of ticks per episode and
tens of episodes per run that difference is minutes. So everything that cannot
change during an episode — the map, the trail, the start and goal markers — is
drawn at open time and never touched again.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: there is no display server in the container
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Subsets are spelled as a task-id list, a count, or this word for "every one".
ALL_TASKS = "all"

# The video canvas, in pixels. Both are multiples of 16 because H.264 encodes
# in 16x16 macroblocks: imageio-ffmpeg silently *resizes* a frame that is not,
# which would quietly soften every panel. Kept as a figure size in inches plus
# a dpi, which is how matplotlib says the same thing.
FRAME_WIDTH_PX = 1600
FRAME_HEIGHT_PX = 640
FRAME_DPI = 100

# How much traversable floor to show around the house, in metres. The view is
# the whole floor rather than the trail's bounding box so that an agent that
# wanders off the trail — the interesting failure — stays on screen.
MAP_MARGIN_M = 1.0

# The robot's heading arrow on the map, in metres.
HEADING_ARROW_M = 0.45


class RecordingConfig:
    """The `recording:` block of `configs/eval.yaml` — on/off, which, how fast.

    `tasks` is the subset knob and takes three spellings, because the three
    things anyone actually wants are different shapes:

        all             every episode of the run (the P6 setting)
        6               the first 6 tasks of each scene, by index
        [Rs_00, Rs_07]  exactly these, by task id (chase one failure)

    `fps` is playback speed, not simulation speed. The control loop runs at
    4 Hz (plan decision E), so 4 fps is real time and the default 10 fps is
    2.5x — fast enough that a 400-tick episode is a 40-second video, slow
    enough to see what happened. It never changes what was simulated.
    """

    def __init__(self, enabled=False, fps=10, tasks=ALL_TASKS,
                 directory="outputs/videos"):
        self.enabled = bool(enabled)
        self.fps = int(fps)
        self.tasks = tasks
        self.directory = Path(directory)

        if self.fps < 1:
            raise ValueError("recording fps must be at least 1")
        if not self._is_valid_subset():
            raise ValueError(
                "recording tasks must be {!r}, a count, or a list of task ids "
                "— not {!r}".format(ALL_TASKS, self.tasks))

    def _is_valid_subset(self):
        if isinstance(self.tasks, bool):
            return False
        return (self.tasks == ALL_TASKS or isinstance(self.tasks, int)
                or isinstance(self.tasks, (list, tuple)))

    @classmethod
    def from_dict(cls, values):
        return cls(**(values or {}))

    def records(self, task_id, task_index):
        """Is this episode filmed? Disabled beats every other answer."""
        if not self.enabled:
            return False
        if self.tasks == ALL_TASKS:
            return True
        if isinstance(self.tasks, int):
            return task_index < self.tasks
        return task_id in self.tasks

    def summary(self):
        if not self.enabled:
            return "off"
        if self.tasks == ALL_TASKS:
            which = "every episode"
        elif isinstance(self.tasks, int):
            which = "the first {} tasks per scene".format(self.tasks)
        else:
            which = "tasks {}".format(", ".join(self.tasks))
        return "{} at {} fps ({:.2g}x real time) -> {}".format(
            which, self.fps, self.fps / 4.0, self.directory)


def parse_subset(text):
    """A subset as typed on a command line, in the config's own spellings.

        all            ->  "all"
        6              ->  6
        Rs_00,Rs_07    ->  ["Rs_00", "Rs_07"]

    Kept beside `RecordingConfig` so the CLI and the YAML cannot drift into
    meaning different things by the same word.
    """
    text = str(text).strip()
    if text.lower() == ALL_TASKS:
        return ALL_TASKS
    if text.isdigit():
        return int(text)
    return [part.strip() for part in text.split(",") if part.strip()]


def subgoal_distance(step):
    """The distance head's own reading for the node being steered at.

    `PolicyStep` carries the head's output for every node in the localization
    window, but it names its two nodes by *absolute* trail index, so the window
    the array is indexed by has to be recovered: the closest node is the window
    entry the head scored lowest, which fixes where the window starts, and the
    subgoal's offset follows from that.

    Returns None rather than guessing if the arithmetic lands outside the
    array — a mislabelled number on an overlay is worse than a blank.
    """
    distances = np.asarray(step.distances, dtype=float)
    if distances.size == 0:
        return None
    window_start = int(step.closest_node) - int(np.argmin(distances))
    offset = int(step.subgoal_node) - window_start
    if not 0 <= offset < distances.size:
        return None
    return float(distances[offset])


def floor_extent(trav_map, resolution):
    """A traversability map's world bounds, as matplotlib's `extent`.

    iGibson stores the map as a square image whose centre pixel is the world
    origin (`indoor_scene.map_to_world`). Handing matplotlib the bounds in
    metres means every later point can be plotted in world coordinates
    directly, with no per-point conversion in the tick loop.

    Drawn with `origin="lower"` so that +y is up and +x is right: the panel is
    then a true top-down view, and a left turn on the map is a left turn in the
    camera. (`p2_1_build_test.py` draws its still in pixel coordinates, which
    mirrors the y axis; that is fine for a single still and confusing in
    something you watch beside the camera.)

    Takes the map and its scale rather than the scene it came from, so the
    geometry stays a pure function — the panel asks `SimScene` for those two
    values and nothing else.
    """
    size = trav_map.shape[0]
    half = size / 2.0
    # Pixel centres sit at (index - half) * resolution, so the image spans half
    # a pixel beyond the first and last centre.
    return ((-half - 0.5) * resolution, (size - half - 0.5) * resolution,
            (-half - 0.5) * resolution, (size - half - 0.5) * resolution)


def traversable_bounds(trav_map, resolution, margin_m=MAP_MARGIN_M):
    """A view box in metres around everything traversable on this floor."""
    rows, columns = np.nonzero(trav_map)
    if rows.size == 0:
        return None
    left, right, bottom, top = floor_extent(trav_map, resolution)
    return (left + columns.min() * resolution - margin_m,
            left + columns.max() * resolution + margin_m,
            bottom + rows.min() * resolution - margin_m,
            bottom + rows.max() * resolution + margin_m)


class PanelFigure:
    """The three panels, built once and redrawn per tick.

    Everything static is placed in `__init__`; `draw(record)` moves only the
    artists that a tick can change and returns the canvas as RGB pixels. That
    split is the reason this is a class rather than a function, and it is worth
    roughly 5x per frame over rebuilding the figure.
    """

    def __init__(self, task, checkpoint_name, scene, node_images,
                 waypoint_scale_m, success_radius_m):
        self.node_images = node_images
        self.waypoint_scale_m = float(waypoint_scale_m)
        self._waypoint_limit = None

        metadata = task.metadata
        self.figure = plt.figure(
            figsize=(FRAME_WIDTH_PX / FRAME_DPI, FRAME_HEIGHT_PX / FRAME_DPI),
            dpi=FRAME_DPI)
        grid = GridSpec(2, 3, figure=self.figure, width_ratios=[1.25, 1.15, 1.0],
                        height_ratios=[1.0, 1.0], left=0.02, right=0.98,
                        top=0.865, bottom=0.08, wspace=0.12, hspace=0.26)

        self.figure.suptitle(
            "{} · {} · geodesic {:.2f} m · {} nodes".format(
                task.task_id, checkpoint_name, task.geodesic_length_m,
                task.node_count),
            fontsize=11, y=0.98)
        # The one line that changes every tick, kept at the top centre rather
        # than under a panel: an axes label floats in whatever margin the
        # image's aspect ratio leaves, and a number that moves is the first
        # thing the eye should find.
        self.status = self.figure.text(0.5, 0.935, "", ha="center", va="top",
                                       fontsize=10, family="monospace")

        self._build_camera_panel(grid[:, 0])
        self._build_map_panel(grid[:, 1], metadata, scene, success_radius_m)
        self._build_subgoal_panel(grid[0, 2])
        self._build_waypoint_panel(grid[1, 2])

    # --- panel (a): what NoMaD saw ------------------------------------------

    def _build_camera_panel(self, cell):
        axes = self.figure.add_subplot(cell)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title("robot camera — what NoMaD sees", fontsize=9)
        # `set_anchor("N")` pins the frame to the top of its cell, so the title
        # sits on the image rather than above the slack an aspect-locked image
        # leaves behind.
        axes.set_anchor("N")
        # Seeded with the right shape so `set_data` never has to resize. Only
        # the artists are kept: nothing after this touches the axes.
        self.camera_image = axes.imshow(np.zeros((2, 2, 3), dtype=np.uint8))

    # --- panel (b): where it has been ---------------------------------------

    def _build_map_panel(self, cell, metadata, scene, success_radius_m):
        axes = self.figure.add_subplot(cell)
        trav_map, resolution = scene.trav_map, scene.trav_map_resolution
        axes.imshow(trav_map, cmap="gray", origin="lower",
                    extent=floor_extent(trav_map, resolution),
                    interpolation="nearest")

        nodes = np.asarray([[node["pose"]["x"], node["pose"]["y"]]
                            for node in metadata["nodes"]], dtype=float)
        axes.plot(nodes[:, 0], nodes[:, 1], color="tab:blue", linewidth=1.0,
                  alpha=0.6, label="reference trail")
        axes.scatter(nodes[:, 0], nodes[:, 1], s=9, color="tab:blue", alpha=0.7)

        start, goal = metadata["start_pose"], metadata["goal_pose"]
        axes.scatter([start["x"]], [start["y"]], s=130, marker="*",
                     color="lime", edgecolor="black", zorder=5, label="start")
        axes.scatter([goal["x"]], [goal["y"]], s=130, marker="*",
                     color="magenta", edgecolor="black", zorder=5, label="goal")
        # The success radius, so a near miss is visible as a near miss.
        axes.add_patch(plt.Circle((goal["x"], goal["y"]), success_radius_m,
                                  fill=False, color="magenta", linestyle=":",
                                  linewidth=1.0))

        self.driven_line, = axes.plot([], [], color="tab:orange", linewidth=1.8,
                                      label="path taken")
        self.collision_points = axes.scatter([], [], s=45, marker="x",
                                             color="red", linewidths=1.6,
                                             zorder=6, label="collision")
        self.subgoal_point = axes.scatter([], [], s=110, marker="o",
                                          facecolor="none", edgecolor="tab:cyan",
                                          linewidths=1.8, zorder=6,
                                          label="current subgoal")
        self.robot_point = axes.scatter([], [], s=60, color="tab:orange",
                                        edgecolor="black", zorder=7,
                                        label="robot")
        self.heading_line, = axes.plot([], [], color="tab:orange",
                                       linewidth=1.6, zorder=7)

        bounds = traversable_bounds(trav_map, resolution)
        if bounds is not None:
            axes.set_xlim(bounds[0], bounds[1])
            axes.set_ylim(bounds[2], bounds[3])
        axes.set_aspect("equal")
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title("top-down — traversable floor, {:g} m per pixel".format(
            resolution), fontsize=9)
        axes.legend(loc="upper right", fontsize=6, framealpha=0.8)

    # --- panel (c): what it is steering at ----------------------------------

    def _build_subgoal_panel(self, cell):
        axes = self.figure.add_subplot(cell)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_anchor("N")
        self.subgoal_image = axes.imshow(np.zeros((2, 2, 3), dtype=np.uint8))
        self.subgoal_title = axes.set_title("", fontsize=9)

    def _build_waypoint_panel(self, cell):
        axes = self.figure.add_subplot(cell)
        self.waypoint_axes = axes
        # Eight diffusion samples, drawn faintly; the chosen one is picked out
        # on top of them. The artists are created empty and reused, so the
        # sample count is fixed at the first tick (it is a policy constant).
        self.sample_lines = []
        self.chosen_line, = axes.plot([], [], color="tab:orange", linewidth=2.0,
                                      zorder=4, label="sample 0 (acted on)")
        self.chosen_point = axes.scatter([], [], s=90, color="red", zorder=5,
                                         label="waypoint #2 -> PD")
        axes.scatter([0], [0], s=40, marker="s", color="black", zorder=3)
        axes.set_aspect("equal")
        axes.axhline(0.0, color="0.85", linewidth=0.8, zorder=1)
        axes.axvline(0.0, color="0.85", linewidth=0.8, zorder=1)
        axes.set_xlabel("forward (m)", fontsize=8)
        axes.set_ylabel("left (m)", fontsize=8)
        axes.tick_params(labelsize=7)
        axes.set_title("predicted actions, robot frame", fontsize=9)
        axes.legend(loc="upper left", fontsize=6, framealpha=0.8)

    # --- per-tick updates ---------------------------------------------------

    def draw(self, record, poses, collisions):
        """Update every dynamic artist from one tick, and render to RGB."""
        self._draw_camera(record)
        self._draw_map(record, poses, collisions)
        self._draw_subgoal(record)
        self._draw_waypoints(record)

        self.figure.canvas.draw()
        pixels = np.asarray(self.figure.canvas.buffer_rgba())
        return pixels[:, :, :3].copy()

    def _draw_camera(self, record):
        self.camera_image.set_data(np.asarray(record.frame))
        self.camera_image.set_extent(
            (-0.5, record.frame.size[0] - 0.5, record.frame.size[1] - 0.5, -0.5))
        self.status.set_text(
            "tick {:3d}   v {:.3f} m/s   w {:+.3f} rad/s{}".format(
                record.index, record.v, record.w,
                "   COLLISION" if record.collided else ""))
        self.status.set_color("red" if record.collided else "black")

    def _draw_map(self, record, poses, collisions):
        path = np.asarray(poses, dtype=float)
        self.driven_line.set_data(path[:, 0], path[:, 1])

        x, y, yaw = record.pose
        self.robot_point.set_offsets([[x, y]])
        self.heading_line.set_data(
            [x, x + HEADING_ARROW_M * np.cos(yaw)],
            [y, y + HEADING_ARROW_M * np.sin(yaw)])
        # `set_offsets` needs a (0, 2) array, not an empty list, before the
        # first collision — otherwise matplotlib keeps the previous points.
        self.collision_points.set_offsets(
            np.asarray(collisions, dtype=float).reshape(-1, 2))
        self.subgoal_point.set_offsets([self._node_xy(record.step.subgoal_node)])

    def _draw_subgoal(self, record):
        node = int(record.step.subgoal_node)
        image = np.asarray(self.node_images.image(node))
        self.subgoal_image.set_data(image)
        self.subgoal_image.set_extent(
            (-0.5, image.shape[1] - 0.5, image.shape[0] - 0.5, -0.5))

        distance = subgoal_distance(record.step)
        self.subgoal_title.set_text(
            "subgoal: node {} of {}   ·   distance head: {}   ·   at node {}"
            .format(node, self.node_images.count - 1,
                    "—" if distance is None else "{:.2f}".format(distance),
                    record.step.closest_node))

    def _draw_waypoints(self, record):
        samples = np.asarray(record.step.samples, dtype=float) * self.waypoint_scale_m
        self._ensure_sample_lines(len(samples))
        self._ensure_waypoint_limits(samples)

        for line, sample in zip(self.sample_lines, samples):
            # Trajectories start at the robot, which `get_action`'s cumulative
            # sum leaves implicit: prepending the origin is what makes the
            # first leg visible.
            line.set_data(np.r_[0.0, sample[:, 0]], np.r_[0.0, sample[:, 1]])

        chosen = samples[0]
        self.chosen_line.set_data(np.r_[0.0, chosen[:, 0]], np.r_[0.0, chosen[:, 1]])
        self.chosen_point.set_offsets([record.waypoint_m[:2]])

    def _ensure_sample_lines(self, count):
        """Create one line per diffusion sample, on the first tick only.

        The sample count is a policy constant, so this runs once per episode;
        the legend is rebuilt with it because it is what names the new lines,
        and a legend redrawn every tick is pure cost.
        """
        if len(self.sample_lines) >= count:
            return
        while len(self.sample_lines) < count:
            line, = self.waypoint_axes.plot(
                [], [], color="tab:blue", linewidth=1.0, alpha=0.45, zorder=2,
                label="{} diffusion samples".format(count)
                if not self.sample_lines else None)
            self.sample_lines.append(line)
        self.waypoint_axes.legend(loc="upper left", fontsize=6, framealpha=0.8)

    def _ensure_waypoint_limits(self, samples):
        """Fix the axes on the first tick and never move them again.

        An autoscaled axis rescales whenever the model changes its mind, which
        in a video reads as the world lurching rather than the prediction
        changing. The limit is the furthest a trajectory could reach — every
        step at the top speed — so nothing ever falls outside it.
        """
        if self._waypoint_limit is not None:
            return
        steps = samples.shape[1]
        self._waypoint_limit = max(steps * self.waypoint_scale_m, 1e-3)
        self.waypoint_axes.set_xlim(-0.25 * self._waypoint_limit,
                                    self._waypoint_limit)
        self.waypoint_axes.set_ylim(-self._waypoint_limit, self._waypoint_limit)

    def _node_xy(self, node):
        return self.node_images.node_xy(int(node))

    def close(self):
        plt.close(self.figure)


class TrailImages:
    """The topomap's frames and node poses, read on demand and remembered.

    A trail is twenty-odd 640x480 PNGs; an episode looks at one of them per
    tick and moves through them roughly in order. Decoding on demand and
    keeping what was decoded costs a few megabytes and saves re-reading the
    same node hundreds of times.
    """

    def __init__(self, task):
        self.directory = Path(task.directory)
        self._nodes = task.metadata["nodes"]
        self._images = {}

    @property
    def count(self):
        return len(self._nodes)

    def image(self, node):
        from PIL import Image

        index = int(np.clip(node, 0, self.count - 1))
        if index not in self._images:
            with Image.open(self.directory / "{}.png".format(index)) as handle:
                self._images[index] = handle.convert("RGB")
        return self._images[index]

    def node_xy(self, node):
        pose = self._nodes[int(np.clip(node, 0, self.count - 1))]["pose"]
        return [pose["x"], pose["y"]]


class NullRecording:
    """What an unrecorded episode gets: a `capture` that does nothing.

    The point of a null object here is that the tick loop is the same loop
    whether or not anything is being filmed — so "recording off does not change
    P3's behaviour" is true by construction rather than by inspection.
    """

    path = None

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False

    def capture(self, record):
        pass

    def close(self):
        pass


class EpisodeRecording:
    """One episode's video: opened before the first tick, closed after the last.

    Used as a context manager, so a crashed episode still leaves a playable
    file of everything up to the crash rather than a zero-byte one.
    """

    def __init__(self, path, fps, panels):
        import imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.panels = panels
        self.frames = 0
        self._poses = []
        self._collisions = []
        # macro_block_size=16 with a canvas that is already a multiple of 16:
        # the writer then never resizes, so what is encoded is what was drawn.
        self._writer = imageio.get_writer(
            str(self.path), fps=fps, codec="libx264", quality=7,
            macro_block_size=16, pixelformat="yuv420p")

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
        return False

    def capture(self, record):
        """Draw and encode one tick. The record is not kept — see the module doc."""
        self._poses.append(record.pose[:2])
        if record.collided:
            self._collisions.append(record.pose[:2])
        self._writer.append_data(
            self.panels.draw(record, self._poses, self._collisions))
        self.frames += 1

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            self.panels.close()

    def summary(self):
        return "video:      {} ({} frames)".format(self.path, self.frames)


class Recorder:
    """Decides which episodes are filmed, and opens the film for the ones that are.

    Holds the config and the output root and nothing per-episode: one recorder
    serves a whole run, exactly as one `EpisodeRunner` does.
    """

    def __init__(self, config, waypoint_scale_m, success_radius_m):
        self.config = config
        self.waypoint_scale_m = waypoint_scale_m
        self.success_radius_m = success_radius_m

    def video_path(self, checkpoint_name, task_id):
        """One directory per arm, one file per task — so the same task under
        two checkpoints is two files that can be watched side by side."""
        return self.config.directory / checkpoint_name / "{}.mp4".format(task_id)

    def episode(self, task, checkpoint_name, scene, task_index=0):
        """The film for one episode, or a null one if it is not being recorded.

        `scene` is the `SimScene` the episode runs in — the floor is already
        bound into it, which is why this no longer takes one.
        """
        if not self.config.records(task.task_id, task_index):
            return NullRecording()
        panels = PanelFigure(
            task, checkpoint_name, scene, TrailImages(task),
            waypoint_scale_m=self.waypoint_scale_m,
            success_radius_m=self.success_radius_m)
        return EpisodeRecording(
            self.video_path(checkpoint_name, task.task_id),
            fps=self.config.fps, panels=panels)


def disabled():
    """A recorder that films nothing — the default for a scoring run."""
    return Recorder(RecordingConfig(enabled=False),
                    waypoint_scale_m=1.0, success_radius_m=1.0)
