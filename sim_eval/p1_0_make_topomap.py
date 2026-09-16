"""Make a hand-made topomap: drive a fixed route open-loop and keep some frames.

The bridge needs a trail to follow before it can be tested, and P1 must not
wait on P2 to build one properly. So this drives a hardcoded sequence of
(v, w) commands — no model, no planner, nobody steering — and saves the camera
frame every `TICKS_PER_NODE` ticks, which is exactly what
`deployment/src/create_topomap.py` does while a human teleoperates the real
robot at its default `--dt 1.0`.

**This is a stand-in, and P2 replaces it.** The real task definition (plan §5)
samples start/goal poses and lets the simulator's own planner drive the
reference path. A canned command list cannot do that, and it is not meant to:
its whole job is to give P1 one trail in one scene to point the bridge at.

Frames are saved at the render resolution, not at any checkpoint's
`image_size`, so one topomap serves every arm — each one resizes it to its own
training resolution, which the fairness protocol (plan §7) requires.

Run (from the repo root, on the host):
    ./sim_eval/run_p1_drive_test.sh --make-topomap
"""

import argparse
import json
from pathlib import Path

import numpy as np

import gpu

SIM_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SIM_EVAL_DIR / "outputs" / "p1_0_topomap"

# One node per second of driving, matching create_topomap.py's default --dt of
# 1.0 s against the robot's 4 Hz frame rate.
TICKS_PER_NODE = 4

# The route, as (linear m/s, angular rad/s, ticks). Speeds are filled in from
# the robot's own max_v / max_w — the reference path is driven at the same
# speed the policy will be allowed to drive it.
#
# Chosen empirically in Rs from the seed below: forward across the room, a left
# turn of about 65 degrees, then forward again — roughly 3.2 m of path in 58
# ticks, collision-free, with one turn in it so that following the trail
# demands steering rather than only going straight.
ROUTE = (
    ("forward", 1.0, 0.0, 14),
    ("left",    0.0, 1.0, 12),
    ("forward", 1.0, 0.0, 32),
)

# The task samples a random start pose, so the seed fixes where the route
# begins. 4 is P0's: the living room, looking across it rather than into a wall.
DEFAULT_SEED = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="topomap directory to write (default: %(default)s)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to render on; never guesses")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed for the start pose, so reruns match")
    return parser.parse_args()


def clear_topomap_dir(output_dir):
    """Empty the directory of nodes — create_topomap.py's own behaviour.

    A stale node left behind from a longer route would silently become part of
    the trail, so leftovers are removed rather than merged.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.png"):
        path.unlink()


def route_commands(limits):
    """Expand ROUTE into one (v, w) per tick, in SI units."""
    commands = []
    for _name, v_scale, w_scale, ticks in ROUTE:
        commands.extend([(v_scale * limits.max_v, w_scale * limits.max_w)] * ticks)
    return commands


def drive_route(body, output_dir):
    """Drive the route, saving a node every TICKS_PER_NODE ticks.

    The frame is saved *before* the tick it labels, so node 0 is the view from
    the start pose — the same node the bridge will be localized to when the
    episode begins.
    """
    nodes = []
    collisions = 0

    for tick, (v, w) in enumerate(route_commands(body.limits)):
        if tick % TICKS_PER_NODE == 0:
            index = len(nodes)
            body.observe().save(output_dir / "{}.png".format(index))
            nodes.append({"node": index, "tick": tick, "pose": list(body.pose)})
            print("node {:>2}  tick {:>2}  pose {}".format(
                index, tick, np.round(body.pose, 3)))
        collisions += int(body.command(v, w))

    return nodes, collisions


def main():
    args = parse_args()

    # Before importing iGibson: it reads the device variables when it
    # initialises the renderer, not when the renderer is used.
    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:    {} (pinned for both EGL and torch)".format(selected_gpu))

    import bridge

    body = bridge.SimBody()
    try:
        gpu.verify_renderer(body.env.simulator.renderer, selected_gpu)

        np.random.seed(args.seed)
        body.reset()
        start_pose = {
            "position": [float(value) for value in body.env.task.initial_pos],
            "orientation": [float(value) for value in body.env.task.initial_orn],
            "seed": args.seed,
        }
        print("start:  {}".format(np.round(body.pose, 3)))

        clear_topomap_dir(args.output)
        nodes, collisions = drive_route(body, args.output)
        end_pose = list(body.pose)
    finally:
        body.close()

    route = {
        "scene": "Rs",
        "start_pose": start_pose,
        "ticks_per_node": TICKS_PER_NODE,
        "route": [{"segment": name, "v_scale": v, "w_scale": w, "ticks": ticks}
                  for name, v, w, ticks in ROUTE],
        "nodes": nodes,
        "end_pose": end_pose,
        "collisions": collisions,
    }
    route_path = args.output / "route.json"
    route_path.write_text(json.dumps(route, indent=2) + "\n")

    print("end:    {}".format(np.round(end_pose, 3)))
    print("wrote:  {} nodes + {}".format(len(nodes), route_path))

    if collisions:
        raise SystemExit(
            "FAILED: the reference route collided on {} of {} ticks. A trail "
            "that drives through walls is not a trail — adjust ROUTE."
            .format(collisions, sum(segment[-1] for segment in ROUTE)))
    print("PASSED: route is collision-free.")


if __name__ == "__main__":
    try:
        main()
    except gpu.GpuSelectionError as error:
        raise SystemExit("FAILED: {}".format(error))
