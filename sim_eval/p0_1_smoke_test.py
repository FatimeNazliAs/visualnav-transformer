"""P0 smoke test: prove iGibson renders the LoCoBot's camera on the GPU.

Spawns the LoCoBot in the Rs scene, drives it straight forward for a few control
steps, and writes the camera's RGB frame to sim_eval/outputs/p0_1_camera.png.

This is the phase's only claim: the simulator is installed, it renders headless
over SSH, and the picture is of a room rather than of nothing. No model, no
policy, no metrics — those are P1 onwards.

The GPU is pinned by sim_eval/gpu.py and the pin is verified against the
renderer before any work happens: on a shared machine, rendering on the wrong
GPU is silent, so this script refuses to run rather than guess.

Run (from the repo root, on the host):
    ./sim_eval/run_p0_smoke_test.sh
"""

import argparse
from pathlib import Path

import numpy as np

import gpu

SIM_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SIM_EVAL_DIR / "configs" / "locobot_rs_static.yaml"
DEFAULT_OUTPUT = SIM_EVAL_DIR / "outputs" / "p0_1_camera.png"

# DifferentialDriveController takes [linear, angular]; the config normalises
# actions to [-1, 1], so this is "full speed ahead, no turn".
FORWARD_ACTION = np.array([1.0, 0.0])

# A render that is a single flat colour — an unbound framebuffer, a camera
# inside geometry, a failed EGL context — has near-zero spread. Real indoor
# frames are far above this; the threshold only has to separate "image" from
# "nothing".
MIN_PIXEL_STD = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="scene + robot config (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="where to write the RGB frame (default: %(default)s)")
    parser.add_argument("--steps", type=int, default=10,
                        help="forward control steps to take before rendering")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to render on; defaults to "
                             "GIBSON_DEVICE_ID, then SIM_GPU. Never guesses.")
    # The task samples a random start pose, so the seed decides what the camera
    # is pointed at. 4 puts the robot in the living room looking across it;
    # the default 0 spawns it nose-to-the-wall, which renders correctly but
    # proves nothing to a human eye.
    parser.add_argument("--seed", type=int, default=4,
                        help="seed for the robot's start pose, so reruns match")
    return parser.parse_args()


def drive_forward(env, steps):
    """Step the sim forward `steps` times and return the last observation."""
    state = env.reset()
    for _ in range(steps):
        state, _reward, done, _info = env.step(FORWARD_ACTION)
        if done:
            # Hitting a wall or the step cap is fine here: we only need a frame.
            break
    return state


def describe_frame(rgb):
    """One-line summary of what came back from the renderer."""
    return (
        "shape={} dtype={} min={:.3f} max={:.3f} mean={:.3f} std={:.3f}".format(
            rgb.shape, rgb.dtype, rgb.min(), rgb.max(), rgb.mean(), rgb.std()
        )
    )


def save_rgb(rgb, output_path):
    """Write an iGibson RGB frame (float 0-1, possibly RGBA) as a PNG."""
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixels = (np.clip(rgb[:, :, :3], 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(output_path)


def main():
    args = parse_args()

    # Before importing iGibson: it reads the device variables when it
    # initialises the renderer, not when the renderer is used.
    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:    {} (pinned for both EGL and torch)".format(selected_gpu))

    from igibson.envs.igibson_env import iGibsonEnv

    np.random.seed(args.seed)
    print("config: {}".format(args.config))

    env = iGibsonEnv(config_file=str(args.config), mode="headless")
    try:
        # Asking for a GPU and getting it are different things: iGibson falls
        # back to device 0 with only a log line. Check before doing any work.
        gpu.verify_renderer(env.simulator.renderer, selected_gpu)

        state = drive_forward(env, args.steps)

        start = env.task.initial_pos
        end = env.robots[0].get_position()
        print("robot: {} -> {}  (moved {:.3f} m)".format(
            np.round(start, 3), np.round(end, 3),
            float(np.linalg.norm(np.asarray(end)[:2] - np.asarray(start)[:2]))))

        # Read the frames here, inside the try, for two reasons: the arrays are
        # views into renderer-owned memory rather than copies, and reading them
        # after the `finally` would raise NameError over the top of whatever
        # real exception got us there.
        rgb = np.array(state["rgb"], copy=True)
        depth = np.array(state["depth"], copy=True) if "depth" in state else None
    finally:
        env.close()

    print("rgb:   {}".format(describe_frame(rgb)))
    if depth is not None:
        print("depth: {}".format(describe_frame(depth)))

    save_rgb(rgb, args.output)
    print("wrote: {}".format(args.output))

    if rgb.std() < MIN_PIXEL_STD:
        raise SystemExit(
            "FAILED: frame is effectively blank (std {:.5f} < {}) — the "
            "renderer produced no image.".format(rgb.std(), MIN_PIXEL_STD))
    print("PASSED: frame is non-blank.")


if __name__ == "__main__":
    # A GPU-selection failure is a message to a person, not a defect to debug,
    # so it exits with the reason rather than a traceback — matching how the
    # blank-frame failure above reports itself.
    try:
        main()
    except gpu.GpuSelectionError as error:
        raise SystemExit("FAILED: {}".format(error))
