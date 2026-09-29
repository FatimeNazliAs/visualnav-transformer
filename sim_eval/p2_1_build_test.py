"""P2's claim: the builder produces a real topomap, and the bridge accepts it.

Builds one reference-path trail in one scene and writes three things to
`sim_eval/outputs/`, so the trail can be judged by looking at it rather than by
trusting it:

    p2_1_topomap/        the trail itself — 0.png, 1.png, ... plus metadata.json
                         and the world.yaml it was driven in
    p2_1_path.png        top-down: the traversable map, the planned path, the
                         path actually driven, and every node on it
    p2_1_thumbnails.png  every node's camera frame in order

Then it loads the result back through `bridge.load_topomap` — P1's loader,
unchanged — and checks it comes back as the nodes that went in. That is the
phase's acceptance test: a trail the bridge cannot read is not a topomap, it is
a folder of pictures.

No model runs here. Following the trail is P1's drive test, and scoring the
follow is P3.

Run (from the repo root, on the host):
    ./sim_eval/run_p2_build_test.sh
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: there is no display server in the container
import matplotlib.pyplot as plt
import numpy as np

import gpu
import topomap_builder

SIM_EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SIM_EVAL_DIR / "outputs"
DEFAULT_TOPOMAP = OUTPUT_DIR / "p2_1_topomap"
DEFAULT_PATH_PLOT = OUTPUT_DIR / "p2_1_path.png"
DEFAULT_THUMBNAILS = OUTPUT_DIR / "p2_1_thumbnails.png"

THUMBNAILS_PER_ROW = 8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=topomap_builder.DEFAULT_CONFIG,
                        help="topomap config to build from (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_TOPOMAP,
                        help="topomap directory to write (default: %(default)s)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to render on; never guesses")
    return parser.parse_args()


def plot_path(scene, metadata, output_path):
    """Top-down view: is this trail a sane route through the house?

    The planned path and the driven path are drawn separately on purpose. They
    should sit on top of each other; where they part is where the follower cut
    a corner, and that is the thing to notice.

    Drawn in map pixels (`SimScene.world_to_map`) with `origin="upper"`, which
    mirrors the y axis against the world. That is fine for one still; the
    recorder's top-down panel, which is watched beside the camera, plots world
    metres the right way up instead.
    """
    trav_map = scene.trav_map
    planned = np.asarray(metadata["planned_path"], dtype=float)
    driven = np.asarray(metadata["driven_path"], dtype=float)
    nodes = np.asarray([[node["pose"]["x"], node["pose"]["y"]]
                        for node in metadata["nodes"]], dtype=float)

    figure, axes = plt.subplots(figsize=(7.5, 7.5))
    axes.imshow(trav_map, cmap="gray", origin="upper")

    axes.plot(*scene.world_to_map(planned), color="tab:blue", linewidth=2.0,
              label="planned (A* on the nav mesh)")
    axes.plot(*scene.world_to_map(driven), color="tab:orange", linewidth=1.5,
              linestyle="--", label="driven")
    axes.scatter(*scene.world_to_map(nodes), s=26, color="tab:red", zorder=3,
                 label="topomap nodes")

    for index, node in enumerate(nodes):
        column, row = scene.world_to_map([node])
        axes.annotate(str(index), (column[0], row[0]), color="tab:red",
                      fontsize=7, xytext=(3, 3), textcoords="offset points")

    start, goal = metadata["start_pose"], metadata["goal_pose"]
    for pose, colour, label in ((start, "lime", "start"), (goal, "magenta", "goal")):
        column, row = scene.world_to_map([[pose["x"], pose["y"]]])
        axes.scatter(column, row, s=150, marker="*", color=colour,
                     edgecolor="black", zorder=4, label=label)

    axes.set_title(
        "{} floor {} · {} nodes every {} ticks ({:.2f} s)\n"
        "geodesic {:.2f} m · driven {:.2f} m · {} collision ticks".format(
            metadata["scene"]["id"], scene.floor, len(nodes),
            metadata["spacing"]["ticks_per_node"],
            metadata["spacing"]["seconds_per_node"],
            metadata["geodesic_length_m"], metadata["driven_length_m"],
            metadata["collision_ticks"]),
        fontsize=10)
    axes.legend(loc="upper right", fontsize=8)
    axes.set_xlabel("traversability map (white = traversable), 0.1 m per pixel",
                    fontsize=8)
    axes.set_xticks([])
    axes.set_yticks([])

    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path


def plot_thumbnails(frames, output_path):
    """Every node in order — the trail as the model will see it."""
    rows = int(np.ceil(len(frames) / THUMBNAILS_PER_ROW))
    figure, axes = plt.subplots(
        rows, THUMBNAILS_PER_ROW,
        figsize=(1.6 * THUMBNAILS_PER_ROW, 1.5 * rows), squeeze=False)

    for index, axis in enumerate(axes.flat):
        axis.set_xticks([])
        axis.set_yticks([])
        if index < len(frames):
            axis.imshow(frames[index])
            axis.set_title(str(index), fontsize=8)
        else:
            axis.axis("off")

    figure.suptitle("topomap nodes, in order", fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return output_path


def check_bridge_loads(topomap_dir, metadata):
    """The acceptance test: P1's loader reads this trail, unchanged.

    `bridge.load_topomap` sorts numerically and takes every `.png` in the
    directory, so this also catches the two ways the format can quietly break —
    a gap in the numbering, and a stray image left behind by a longer run.
    """
    import bridge

    topomap = bridge.load_topomap(topomap_dir)
    expected = len(metadata["nodes"])
    if len(topomap) != expected:
        raise SystemExit(
            "FAILED: bridge.load_topomap read {} nodes but the metadata "
            "describes {}.".format(len(topomap), expected))

    camera = metadata["camera"]
    sizes = {frame.size for frame in topomap}
    if sizes != {(camera["width"], camera["height"])}:
        raise SystemExit(
            "FAILED: nodes are not all at the render resolution {}x{}: got {}."
            .format(camera["width"], camera["height"], sorted(sizes)))

    print("bridge:     load_topomap read {} nodes at {}x{} — accepted".format(
        len(topomap), camera["width"], camera["height"]))
    return topomap


def report(metadata):
    print()
    print("scene:      {} floor {}".format(
        metadata["scene"]["id"], metadata["scene"]["floor"]))
    print("start:      ({x:.2f}, {y:.2f}) yaw {yaw:+.2f}".format(**metadata["start_pose"]))
    print("goal:       ({x:.2f}, {y:.2f}) yaw {yaw:+.2f}".format(**metadata["goal_pose"]))
    print("geodesic:   {:.2f} m (shortest path on the nav mesh)".format(
        metadata["geodesic_length_m"]))
    print("driven:     {:.2f} m in {} ticks".format(
        metadata["driven_length_m"], metadata["ticks"]))
    print("spacing:    every {} ticks ({:.2f} s), median node gap {:.2f} m".format(
        metadata["spacing"]["ticks_per_node"],
        metadata["spacing"]["seconds_per_node"],
        metadata["spacing"]["median_node_gap_m"]))
    print("nodes:      {}".format(len(metadata["nodes"])))
    print("collided:   {} of {} ticks".format(
        metadata["collision_ticks"], metadata["ticks"]))
    print("rejected:   {} start/goal pairs before this one".format(
        len(metadata["rejected_attempts"])))


def main():
    args = parse_args()

    # Before importing iGibson: it reads the device variables when it
    # initialises the renderer, not when the renderer is used.
    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))

    config = topomap_builder.TopomapConfig.from_yaml(args.config)
    print("config:     {}".format(args.config))
    print("            {}".format(config.summary()))

    body = topomap_builder.open_body(config, args.output)
    try:
        body.verify_gpu(selected_gpu)
        metadata = topomap_builder.build_topomap(
            body, config, args.output,
            on_node=lambda node: print("node {:>2}  tick {:>3}  pose {}".format(
                node["node"], node["tick"], np.round(node["pose"], 3))),
            on_reject=lambda attempt, why: print(
                "\nrejected attempt {}: {}\n".format(attempt, why)))
        # Drawn before the sim closes: the traversability map belongs to the
        # scene, and the scene dies with the environment.
        plot_path(body.scene, metadata, DEFAULT_PATH_PLOT)
    finally:
        body.close()

    report(metadata)

    # Read back from disk rather than reusing the frames in memory: the point
    # is to prove what landed on disk is a topomap, not what was in RAM.
    topomap = check_bridge_loads(args.output, metadata)
    plot_thumbnails(topomap, DEFAULT_THUMBNAILS)

    print()
    print("wrote:      {}".format(args.output))
    print("            {}".format(DEFAULT_PATH_PLOT))
    print("            {}".format(DEFAULT_THUMBNAILS))
    print("PASSED: the trail is in navigate.py's format and the bridge reads it.")


if __name__ == "__main__":
    try:
        main()
    except (gpu.GpuSelectionError, topomap_builder.TopomapError) as error:
        raise SystemExit("FAILED: {}".format(error))
