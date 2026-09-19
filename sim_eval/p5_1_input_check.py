"""Is the model being fed what it was trained on? — P5's first question.

Before any failure is blamed on the domain gap, the cheap explanations have to
be ruled out. A policy fed BGR, or fed [0, 255] where it learned [0, 1], or fed
a frame cropped differently from every frame in its training set, fails in ways
that look exactly like "the sim looks different from a real corridor" — and no
amount of watching videos tells the two apart.

So this puts one sim frame and one GoStanford frame through **the same
transform the policy uses** (`nomad_policy.transform_images`, which is
`deployment/src/utils.py`'s) and reports, side by side, everything that could
silently differ:

  * **channel order** — per-channel means before normalization, plus the frames
    themselves written out, because a red/blue swap is obvious in a picture and
    invisible in a mean.
  * **value range** — the per-channel statistics of the tensor the encoder
    actually receives, under ImageNet normalization.
  * **aspect and crop** — the native size of each source, what training's 4:3
    centre crop (`vint_train.data.data_utils.IMAGE_ASPECT_RATIO`) does to it,
    and what it is resized to.
  * **lens and field of view** — the sim's, read off the renderer's own
    intrinsics. GoStanford records none, so the comparison is made twice:
    by *rendering* — one sim pose through a sweep of fields of view, printed
    beside a strip of training frames — and by *measuring*, with the model's
    own observation encoder (see `camera_measurement`), which field of view
    puts sim frames where the model's training frames sit.
  * **camera tilt** — the LoCoBot's camera is fixed 20 degrees down in its
    URDF; GoStanford's horizon sits near mid-frame. Measured the same way as
    the lens: the same poses at several tilts, through the encoder.
  * **camera height** — where the sim robot's eye sits above the floor.

Nothing here is a fix and nothing here touches the policy. It is evidence, and
what to do about it is a decision made after reading it (plan §10: the domain
gap is a documented limitation, not a bug to be tuned away).

What comes out (from `sim_eval/outputs/`):

    p5_1_input_check.png    the two sources side by side, at every stage
    p5_1_input_check.json   the same numbers, plus the lens measurement

Run (from the repo root, on the host):
    ./sim_eval/run_p5_1_input_check.sh
"""

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: there is no display server in the container
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.gridspec import GridSpec

import checkpoints
import gpu

SIM_EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SIM_EVAL_DIR / "outputs"
DEFAULT_FIGURE = OUTPUT_DIR / "p5_1_input_check.png"
DEFAULT_REPORT = OUTPUT_DIR / "p5_1_input_check.json"
DEFAULT_WORLD = SIM_EVAL_DIR / "configs" / "locobot_rs_bridge.yaml"
DEFAULT_CHECKPOINT = "best_combined"

# The training data, as the container mounts it (see lib.sh: /data is
# /mnt/shared_disk/nazli/nomad_data). This is the same folder
# train/config/nomad.yaml points `go_stanford.data_folder` at.
DEFAULT_DATASET = Path("/data/raw/go_stanford/go_stanford")

# Training's own crop, from vint_train/data/data_utils.py: every image is
# centre-cropped to 4:3 before it is resized. Both sources are already 4:3, so
# this should be a no-op for both — which is a claim worth printing rather than
# assuming.
IMAGE_ASPECT_RATIO = 4 / 3

# How many frames of each source go into the comparison strip. Enough to see
# whether one frame was a fluke, few enough to read at a glance.
SAMPLE_FRAMES = 6

# Vertical fields of view to render the same pose through. 45 is iGibson's
# LoCoBot default, which P1-P5 ran under; the rest bracket what a wide indoor
# lens covers, up to 120 deg vertical (132 deg horizontal), past which a
# rectilinear projection stretches the edges into uselessness. The renderer is
# put back to its configured value after each one.
FOV_SWEEP_DEG = (45, 60, 75, 90, 105, 120)

# Downward camera tilts to render the same poses through, at the configured
# field of view. 20 is the LoCoBot URDF's own fixed head tilt (~21 at rest);
# 0 is level, which is where GoStanford's horizon sits — near mid-frame. The
# pitch experiment's candidates, fixed here before any episode ran under them.
TILT_SWEEP_DEG = (0, 7, 14, 20)

# The lens measurement's sample sizes. Every pose is rendered at every angle
# in the sweep, so the scene content is identical across angles and only the
# lens differs; the GoStanford frames are split in half so that training can
# be measured against itself, which is what "indistinguishable" looks like.
LENS_SIM_POSES = 60
LENS_DATASET_FRAMES = 200


# --------------------------------------------------------------------------
# what the encoder is handed
# --------------------------------------------------------------------------

def aspect_crop(image, aspect_ratio=IMAGE_ASPECT_RATIO):
    """Training's centre crop to 4:3, from `data_utils.resize_and_aspect_crop`.

    Reproduced rather than imported for the same reason `nomad_policy` copies
    `transform_images`: this has to be the crop as training applied it, pinned
    here, not whatever a shared helper becomes later.
    """
    width, height = image.size
    if width > height:
        box_h, box_w = height, int(height * aspect_ratio)
    else:
        box_h, box_w = int(width / aspect_ratio), width
    left = (width - box_w) // 2
    top = (height - box_h) // 2
    return image.crop((left, top, left + box_w, top + box_h))


def channel_stats(array):
    """Per-channel min/mean/max of an (H, W, 3) or (3, H, W) array."""
    values = np.asarray(array, dtype=float)
    if values.shape[0] == 3 and values.ndim == 3:
        planes = values
    else:
        planes = values.transpose(2, 0, 1)
    return {
        name: {
            "min": round(float(plane.min()), 4),
            "mean": round(float(plane.mean()), 4),
            "max": round(float(plane.max()), 4),
        }
        for name, plane in zip(("r", "g", "b"), planes)
    }


def encoder_input(image, image_size):
    """The frame as the policy hands it to the encoder, as (3, H, W) numpy.

    `transform_images` is the deployment transform and takes a *list* of frames
    and a (width, height) size, returning them concatenated on the channel
    axis; one frame in means three channels out.
    """
    from nomad_policy import transform_images

    tensor = transform_images([image], image_size)
    return tensor.squeeze(0).cpu().numpy()


def describe_source(name, image, image_size):
    """One source, at every stage between the file and the encoder."""
    cropped = aspect_crop(image)
    resized = cropped.resize(image_size)
    return {
        "name": name,
        "native_size": list(image.size),
        "native_aspect": round(image.size[0] / image.size[1], 4),
        "after_4_3_crop": list(cropped.size),
        "crop_is_a_no_op": cropped.size == image.size,
        "resized_to": list(resized.size),
        "uint8_channels": channel_stats(np.asarray(resized)),
        "encoder_channels": channel_stats(encoder_input(image, image_size)),
        "images": {"native": image, "cropped": cropped, "resized": resized},
    }


# --------------------------------------------------------------------------
# the lens
# --------------------------------------------------------------------------

def horizontal_fov_deg(vertical_fov_deg, width, height):
    """The horizontal field of view of a rectilinear camera, from its vertical one."""
    half = math.radians(vertical_fov_deg) / 2
    return round(math.degrees(2 * math.atan(math.tan(half) * width / height)), 2)


def fov_sweep(body, degrees):
    """The pose the robot is standing at, rendered through each field of view.

    This is the only honest way to compare the two cameras. GoStanford ships
    frames and odometry and no intrinsics, so there is no number to put beside
    the sim's 45 degrees; what there is, is the same room through several
    lenses, next to the rooms the model was trained on. A pixel statistic
    cannot decide this — the obvious candidate, the dark border a fisheye
    leaves in the corners, was measured over 120 training frames and 32 sim
    frames and came out *higher for the sim* (1.6% against 0.3% at a threshold
    of 10), because GoStanford's frames are crops from inside the lens circle
    and Rs has dark furniture. The picture decides it.
    """
    return [(degrees_i, body.render_at_vertical_fov(degrees_i))
            for degrees_i in degrees]


def tilt_sweep(body, degrees):
    """The pose the robot is standing at, rendered at each downward camera tilt."""
    return [(tilt, body.render_at_camera_tilt(tilt)) for tilt in degrees]


def camera_sweep(body, poses, render_settings):
    """`poses` random poses across the house, each rendered at every setting.

    `render_settings(body)` renders the pose the robot is at under each
    candidate camera setting — `fov_sweep` or `tilt_sweep` — and returns
    [(setting, frame), ...]. Returns {setting: [frame, ...]} with the frames in
    pose order, so the same index is the same place in the house under every
    setting and only the camera differs from one row to the next.
    """
    frames = {}
    for _ in range(poses):
        _place_randomly(body)
        for setting, frame in render_settings(body):
            frames.setdefault(setting, []).append(frame)
    return frames


def standardize(embeddings, reference):
    """Express embeddings in units of the reference set's own spread, per dim."""
    mean = reference.mean(axis=0)
    spread = reference.std(axis=0) + 1e-6
    return (np.asarray(embeddings, dtype=float) - mean) / spread


def mean_nearest_distance(probe, reference):
    """How far, on average, each probe point is from its nearest reference point."""
    probe = np.asarray(probe, dtype=float)
    reference = np.asarray(reference, dtype=float)
    squared = ((probe[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
    return float(np.sqrt(squared.min(axis=1)).mean())


def centroid_gap(probe, reference):
    """Distance between the two sets' means, per dimension, in reference spreads."""
    probe = standardize(probe, reference)
    return float(np.linalg.norm(probe.mean(axis=0)) / np.sqrt(probe.shape[1]))


def camera_measurement(sim_embeddings, dataset_embeddings):
    """Which camera setting makes sim frames look like training, to the model.

    The picture in the figure asks a person; this asks the observation
    encoder, which is the only reader of a frame whose opinion decides
    anything. Both sides are embedded by the same checkpoint's psi, and each
    candidate setting — a field of view, or a camera tilt — gets two numbers:

      * `nearest_ratio` — how far a sim frame is from its nearest training
        frame, divided by how far a held-out training frame is from *its*
        nearest one. 1.0 is "as close to training as training is to itself".
      * `centroid_gap` — how far the two means sit apart, in training spreads.

    The rule for choosing is fixed here, before anything is run: **the
    setting with the lowest `nearest_ratio`**, reported with `centroid_gap`
    beside it as a check that the two agree. It is chosen on the model's
    *input*, from frames nobody drove, so no episode outcome can have
    influenced it — the property that makes a camera change a correctness fix
    rather than a tuned knob (plan decision E).

    The scene confound is real — Rs is a house and GoStanford is university
    buildings — but it is constant across a sweep: every setting is rendered
    at the same poses, so only the camera changes from one row to the next.
    What the absolute numbers mean is content plus camera; what the *trend*
    means is camera.
    """
    dataset = np.asarray(dataset_embeddings, dtype=float)
    half = len(dataset) // 2
    reference, held_out = dataset[:half], dataset[half:]
    scale = mean_nearest_distance(standardize(held_out, reference),
                                  standardize(reference, reference))

    def score(embeddings):
        probe = standardize(embeddings, reference)
        return {
            "nearest_ratio": round(mean_nearest_distance(
                probe, standardize(reference, reference)) / scale, 4),
            "centroid_gap": round(centroid_gap(embeddings, reference), 4),
        }

    rows = {int(setting): score(embeddings)
            for setting, embeddings in sorted(sim_embeddings.items())}
    best = min(rows, key=lambda setting: rows[setting]["nearest_ratio"])
    return {
        "training_against_itself": score(held_out),
        "by_setting": rows,
        "chosen": best,
        "centroid_agrees": best == min(
            rows, key=lambda setting: rows[setting]["centroid_gap"]),
    }


# --------------------------------------------------------------------------
# the two sources
# --------------------------------------------------------------------------

def sim_frames(body, count):
    """`count` frames from the open simulator, from poses across the house.

    Taken from random traversable points rather than from one spot, so the
    lens statistics are not a property of whatever corner the reset happened
    to choose.
    """
    frames = []
    for _ in range(count):
        _place_randomly(body)
        frames.append(body.observe())
    return frames


def _place_randomly(body):
    """Put the robot on a random traversable point, facing a random way."""
    point = body.scene.random_point()
    yaw = float(np.random.uniform(-math.pi, math.pi))
    body.place([point[0], point[1], body.scene.floor_height], [0.0, 0.0, yaw])


def dataset_frames(dataset_dir, count, rng):
    """`count` frames from GoStanford, one from each of `count` trajectories."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(
            "FAILED: no GoStanford at {}. Inside the container it is /data/...; "
            "on the host it is under /mnt/shared_disk/nazli/nomad_data/."
            .format(dataset_dir))

    trajectories = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not trajectories:
        raise SystemExit("FAILED: {} holds no trajectories.".format(dataset_dir))

    frames = []
    for trajectory in rng.sample(trajectories, min(count, len(trajectories))):
        images = sorted(trajectory.glob("*.jpg"),
                        key=lambda path: int(path.stem))
        if images:
            # Not frame 0: the first frames of a teleop trajectory often catch
            # the operator still setting the robot down.
            frames.append(Image.open(images[len(images) // 2]).convert("RGB"))
    if not frames:
        raise SystemExit("FAILED: {} holds no frames.".format(dataset_dir))
    return frames


def camera_height_m(body):
    """How far the sim robot's eye sits above the floor it is standing on."""
    eye_z = float(body.robot.eyes.get_position()[2])
    return round(eye_z - body.scene.floor_height, 4)


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------

def _strip(images, cell=(160, 120)):
    """Several frames laid out as one wide image, all at the same size."""
    return np.concatenate([np.asarray(image.resize(cell)) for image in images],
                          axis=1)


def _blank(axis, title=None):
    axis.set_xticks([])
    axis.set_yticks([])
    if title is not None:
        axis.set_title(title, fontsize=9)


def draw_comparison(report, sim, dataset, sweep, dataset_strip, path):
    """The pipeline, the lens sweep and the training frames, in one picture.

    Three bands, because there are three separate questions and mixing them is
    what makes a figure unreadable:

      1. the pipeline — one frame from each source at every stage between the
         file and the encoder, so reading across a row is reading what the
         transform does to it. A channel swap or a bad crop is visible here and
         nowhere in the numbers.
      2. the lens — one sim pose rendered through several fields of view.
      3. the training distribution — real GoStanford frames, at the size the
         encoder gets them, to hold band 2 against.
    """
    figure = plt.figure(figsize=(13, 11))
    grid = GridSpec(4, 3, figure=figure, height_ratios=[1.15, 1.15, 0.85, 0.85],
                    hspace=0.35, wspace=0.12)

    stages = ("native", "cropped", "resized")
    titles = ("as rendered / as recorded", "after training's 4:3 centre crop",
              "resized to the checkpoint's image_size")
    for row, source in enumerate((sim, dataset)):
        for column, (stage, title) in enumerate(zip(stages, titles)):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(source["images"][stage])
            _blank(axis, "{}  —  {}\n{}x{}".format(
                source["name"], title, *source["images"][stage].size))

    lens = figure.add_subplot(grid[2, :])
    lens.imshow(_strip([frame for _degrees, frame in sweep]))
    intrinsics = report["sim"]["intrinsics"]
    _blank(lens, "the same sim pose through {} deg vertical\n"
                 "({} deg horizontal) — the rollout runs the first".format(
                     " · ".join(str(degrees) for degrees, _frame in sweep),
                     " · ".join("{:.0f}".format(horizontal_fov_deg(
                         degrees, intrinsics["width"], intrinsics["height"]))
                         for degrees, _frame in sweep)))

    training = figure.add_subplot(grid[3, :])
    training.imshow(_strip(dataset_strip))
    _blank(training, "GoStanford — {} training frames, as the lens recorded them"
           .format(len(dataset_strip)))

    figure.text(0.5, 0.015,
                "sim camera: {}x{} · {:.0f} deg vertical · {:.0f} deg horizontal "
                "· eye {:.2f} m above the floor.   GoStanford records no "
                "intrinsics — the two bottom bands are the comparison.".format(
                    intrinsics["width"], intrinsics["height"],
                    intrinsics["vertical_fov_deg"],
                    report["sim"]["horizontal_fov_deg"],
                    report["sim"]["camera_height_m"]),
                ha="center", fontsize=9, family="monospace")
    figure.suptitle(
        "What the encoder is fed — simulator against GoStanford, "
        "through the same transform ({})".format(report["checkpoint"]),
        fontsize=13)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(path, dpi=110)
    plt.close(figure)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def print_report(report):
    sim, dataset = report["sim"], report["dataset"]

    print("\n=== channel order ===")
    print("Per-channel means of the 8-bit frame, before normalization. A red/"
          "blue swap shows up\nhere as the two ends of the triple trading "
          "places — and, unmistakably, in the figure.")
    for source in (sim, dataset):
        channels = source["uint8_channels"]
        print("  {:<14} R {:6.1f}   G {:6.1f}   B {:6.1f}".format(
            source["name"] + ":", channels["r"]["mean"],
            channels["g"]["mean"], channels["b"]["mean"]))

    print("\n=== value range (what the encoder receives) ===")
    print("ImageNet-normalized, so roughly [-2.2, +2.7] per channel. Training "
          "applied the same\nnormalization (train.py line 68) over [0, 1] "
          "tensors, so these should match in scale.")
    for source in (sim, dataset):
        channels = source["encoder_channels"]
        print("  {:<14} R [{:+.2f}, {:+.2f}]  G [{:+.2f}, {:+.2f}]  "
              "B [{:+.2f}, {:+.2f}]".format(
                  source["name"] + ":",
                  channels["r"]["min"], channels["r"]["max"],
                  channels["g"]["min"], channels["g"]["max"],
                  channels["b"]["min"], channels["b"]["max"]))

    print("\n=== aspect and crop ===")
    for source in (sim, dataset):
        print("  {:<14} {}x{} (aspect {:.3f}) -> crop {}x{} {} -> resize {}x{}"
              .format(
                  source["name"] + ":", source["native_size"][0],
                  source["native_size"][1], source["native_aspect"],
                  source["after_4_3_crop"][0], source["after_4_3_crop"][1],
                  "(no-op)" if source["crop_is_a_no_op"] else "(CROPPED)",
                  source["resized_to"][0], source["resized_to"][1]))

    print("\n=== lens and field of view ===")
    print("  sim:           {:.0f} deg vertical, {:.0f} deg horizontal, "
          "rectilinear (as configured)".format(
              sim["intrinsics"]["vertical_fov_deg"], sim["horizontal_fov_deg"]))
    print("  GoStanford:    no intrinsics in the dataset — it ships frames and "
          "odometry, nothing about the rig")
    print("  so:            the figure's bottom two bands show it; the "
          "encoder measures it:")
    width, height = sim["intrinsics"]["width"], sim["intrinsics"]["height"]
    print_camera_measurement(
        report["lens"], "vertical / horizontal",
        lambda angle: "{} / {:.0f} deg".format(
            angle, horizontal_fov_deg(angle, width, height)))

    print("\n=== camera tilt ===")
    print("  sim:           {} deg down — {}".format(
        sim["camera_tilt_deg"], "the world config's `camera_tilt_deg`"
        if sim["camera_tilt_configured"] else
        "the LoCoBot URDF's fixed head tilt (~21 at rest)"))
    print("  GoStanford:    no rig in the dataset; its horizon sits near "
          "mid-frame, i.e. roughly level")
    print("  so:            the same poses at {} deg down, at this field of "
          "view:".format(", ".join(str(tilt) for tilt in TILT_SWEEP_DEG)))
    print_camera_measurement(report["tilt"], "tilt down",
                             lambda tilt: "{} deg".format(tilt))

    print("\n=== camera height ===")
    print("  sim:           {:.2f} m above the floor ({})".format(
        sim["camera_height_m"], report["robot"]))
    print("  GoStanford:    not recoverable from the dataset — it ships frames "
          "and odometry, no rig")


def print_camera_measurement(measurement, header, label):
    """A sweep as a table — the model's own reading of each camera setting.

    `label(setting)` is the setting as the first column shows it.
    """
    print("\n    {:>22}  {:>13}  {:>12}".format(header, "nearest_ratio",
                                                 "centroid_gap"))
    for setting, row in measurement["by_setting"].items():
        print("    {:>22}  {:>13.3f}  {:>12.3f}{}".format(
            label(setting), row["nearest_ratio"], row["centroid_gap"],
            "   <- chosen" if setting == measurement["chosen"] else ""))
    baseline = measurement["training_against_itself"]
    print("    {:>22}  {:>13.3f}  {:>12.3f}   (what indistinguishable looks "
          "like)".format("GoStanford vs itself", baseline["nearest_ratio"],
                         baseline["centroid_gap"]))
    print("\n    rule, fixed before running: lowest nearest_ratio -> {}; the "
          "centroid gap {}.".format(
              label(measurement["chosen"]).strip(),
              "agrees" if measurement["centroid_agrees"]
              else "DISAGREES — read both"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="whose image_size the frames are resized to "
                             "(default: %(default)s)")
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD,
                        help="iGibson world to render from (default: %(default)s)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="GoStanford root (default: %(default)s)")
    parser.add_argument("--frames", type=int, default=SAMPLE_FRAMES,
                        help="frames per source in the strip (default: %(default)s)")
    parser.add_argument("--lens-poses", type=int, default=LENS_SIM_POSES,
                        help="sim poses rendered at every angle for the lens "
                             "measurement (default: %(default)s)")
    parser.add_argument("--lens-frames", type=int, default=LENS_DATASET_FRAMES,
                        help="GoStanford frames for the lens measurement, half "
                             "of them held out (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0,
                        help="which frames and poses are sampled (default: %(default)s)")
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE,
                        help="where the comparison goes (default: %(default)s)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help="where the numbers go (default: %(default)s)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to use; never guesses")
    return parser.parse_args()


def main():
    args = parse_args()

    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))

    import bridge

    spec = checkpoints.load(args.checkpoint)
    print("checkpoint: {}".format(spec.summary()))
    print("image_size: {}x{} (the checkpoint's own)".format(*spec.image_size))
    print("dataset:    {}".format(args.dataset))

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    body = bridge.SimBody(config_path=args.world)
    try:
        body.verify_gpu(selected_gpu)
        body.reset()
        intrinsics = body.intrinsics
        height = camera_height_m(body)
        robot_name = type(body.robot).__name__
        sim_strip = sim_frames(body, args.frames)
        # The sweep comes next, from wherever the strip left the robot, so the
        # lens band and the pipeline band above it are the same room.
        sweep = fov_sweep(body, FOV_SWEEP_DEG)
        sim_lens = camera_sweep(
            body, args.lens_poses, lambda b: fov_sweep(b, FOV_SWEEP_DEG))
        # After the lens sweep, so its poses — and every FOV number measured
        # before the tilt sweep existed — come out of the RNG unchanged.
        sim_tilt = camera_sweep(
            body, args.lens_poses, lambda b: tilt_sweep(b, TILT_SWEEP_DEG))
        camera_tilt = body.camera_tilt_deg
    finally:
        body.close()

    dataset_lens = dataset_frames(args.dataset, args.lens_frames, rng)
    dataset_strip = dataset_lens[:args.frames]

    import torch

    from nomad_policy import NomadPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = NomadPolicy(spec, device)
    dataset_embeddings = policy.embed_frames(dataset_lens)
    lens = camera_measurement(
        {angle: policy.embed_frames(frames) for angle, frames in sim_lens.items()},
        dataset_embeddings)
    tilt = camera_measurement(
        {angle: policy.embed_frames(frames) for angle, frames in sim_tilt.items()},
        dataset_embeddings)

    sim = describe_source("simulator", sim_strip[0], spec.image_size)
    sim.update({
        "intrinsics": intrinsics,
        "horizontal_fov_deg": horizontal_fov_deg(
            intrinsics["vertical_fov_deg"], intrinsics["width"],
            intrinsics["height"]),
        "camera_height_m": height,
        "camera_tilt_deg": (bridge.URDF_CAMERA_TILT_DEG if camera_tilt is None
                            else camera_tilt),
        "camera_tilt_configured": camera_tilt is not None,
        "fov_sweep_deg": list(FOV_SWEEP_DEG),
        "tilt_sweep_deg": list(TILT_SWEEP_DEG),
    })
    dataset = describe_source("GoStanford", dataset_strip[0], spec.image_size)

    report = {
        "checkpoint": args.checkpoint,
        "image_size": list(spec.image_size),
        "robot": robot_name,
        "world": str(args.world),
        "dataset": dataset,
        "sim": sim,
        "lens": dict(lens, sim_poses=args.lens_poses,
                     dataset_frames=len(dataset_lens), seed=args.seed),
        "tilt": dict(tilt, sim_poses=args.lens_poses,
                     dataset_frames=len(dataset_lens), seed=args.seed),
    }
    print_report(report)

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    draw_comparison(report, sim, dataset, sweep, dataset_strip, args.figure)
    # The PIL images are for the figure, not for the file.
    for source in (sim, dataset):
        source.pop("images")
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)

    print("\nwrote:      {}".format(args.figure))
    print("            {}".format(args.report))


if __name__ == "__main__":
    try:
        main()
    except (gpu.GpuSelectionError, checkpoints.CheckpointError) as error:
        raise SystemExit("FAILED: {}".format(error))
