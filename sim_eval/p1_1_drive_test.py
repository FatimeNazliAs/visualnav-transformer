"""P1's claim: one checkpoint drives itself toward the goal using the ported loop.

Loads one NoMaD checkpoint, places the robot at the start of the hand-made
topomap from `p1_0_make_topomap.py`, and hands control to the bridge. Every
tick's camera frame is written out, plus a GIF, so the run can be watched
rather than argued about.

There are no metrics here. Success rate, SPL, collision rate and the episode
sampler are P3; this is the "does it take sane steps at all" gate. The numbers
it does print — distance to the last node's pose, which topomap node it thinks
it is at, how many ticks collided — are there to be eyeballed alongside the
frames, not logged for analysis.

Run (from the repo root, on the host):
    ./sim_eval/run_p1_drive_test.sh
"""

import argparse
import json
from pathlib import Path

import numpy as np

import checkpoints
import gpu

SIM_EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOMAP = SIM_EVAL_DIR / "outputs" / "p1_0_topomap"
DEFAULT_OUTPUT = SIM_EVAL_DIR / "outputs" / "p1_1_drive_test"
DEFAULT_CHECKPOINT = "best_combined"

# The reference route is 58 ticks. Following it closed-loop costs more than
# driving it open-loop — the policy steers, overshoots and corrects — so the cap
# is generous. P3 ties the real timeout to the reference path length.
DEFAULT_MAX_TICKS = 200

# Milliseconds per GIF frame. 250 ms is the control period, so the replay runs
# at wall-clock speed.
GIF_FRAME_MS = 250

# Fixed so diffusion sampling repeats run to run (plan §7). P1 only needs
# reruns to match; per-episode seeding is P3's business.
DEFAULT_SEED = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="name from configs/checkpoints.yaml (default: %(default)s)")
    parser.add_argument("--topomap", type=Path, default=DEFAULT_TOPOMAP,
                        help="topomap directory to follow (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="where frames and the GIF go (default: %(default)s)")
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS,
                        help="give up after this many control ticks")
    parser.add_argument("--gpu", type=int, default=None,
                        help="physical GPU to use; never guesses")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed for diffusion sampling, so reruns match")
    return parser.parse_args()


def read_start_pose(topomap_dir):
    """The pose the reference route was driven from, so the trail starts here.

    Without it the robot would begin wherever the task's random reset put it,
    and following a trail that starts somewhere else is a different (harder,
    and unmeasured) problem than the one P1 is testing.
    """
    route_path = Path(topomap_dir) / "route.json"
    if not route_path.exists():
        raise SystemExit(
            "FAILED: {} has no route.json, so the route's start pose is "
            "unknown. Regenerate it with p1_0_make_topomap.py."
            .format(topomap_dir))
    route = json.loads(route_path.read_text())
    start = route["start_pose"]
    return (start["position"], start["orientation"]), route


def write_frames(records, output_dir):
    """One PNG per tick, plus a GIF of the whole run. None if the run had no ticks.

    An empty run is reachable (`--max-ticks 0`, or a stop before the first
    tick), and there is nothing to write then — previously this raised
    IndexError on `frames[0]` after already deleting the previous run's frames.
    """
    if not records:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.png"):
        path.unlink()

    frames = [record.frame for record in records]
    for record in records:
        record.frame.save(output_dir / "{:04d}.png".format(record.index))

    gif_path = output_dir / "drive.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=GIF_FRAME_MS, loop=0)
    return gif_path


def drive(runner, max_ticks):
    """Tick until the bridge localizes onto the last node, or the budget runs out.

    P1's own stopping rule, and it lives here rather than on the bridge because
    it is P1's alone: "the distance head thinks it is at the last node" is a
    claim about where the robot believes it is, which P3 measures rather than
    trusts. Keeping it in the one script that wants it means there is exactly
    one answer in the package to "when is an episode over", and it is
    `episode_runner`'s.

    This script does keep every frame — it writes them all out as a GIF, which
    is the point of it. That is a deliberate, local choice, not the default:
    the scorer streams and keeps nothing.
    """
    records = []
    for _ in range(max_ticks):
        record = runner.tick()
        records.append(record)
        print(record.summary())
        if runner.reached_goal:
            break
    return records


def report(records, runner, goal_pose, node_count):
    """Print what a human needs to decide whether this looks sane."""
    if not records:
        print("\nno ticks ran, so there is nothing to report.")
        return
    end_pose = runner.body.pose
    goal_distance = float(np.linalg.norm(
        np.array(end_pose[:2]) - np.array(goal_pose[:2])))
    # Each record holds the pose the tick started from, so the final pose has
    # to be appended or the last leg goes uncounted.
    poses = [record.pose for record in records] + [end_pose]
    path_length = sum(
        float(np.linalg.norm(np.array(later[:2]) - np.array(earlier[:2])))
        for earlier, later in zip(poses, poses[1:]))

    print()
    print("ticks:          {} ({:.1f} s of robot time)".format(
        len(records), len(records) * runner.limits.dt))
    print("nodes:          localized to {} of {}".format(
        records[-1].step.closest_node, node_count - 1))
    print("final pose:     {}".format(np.round(end_pose, 3)))
    print("goal pose:      {}".format(np.round(goal_pose, 3)))
    print("distance:       {:.2f} m from the last topomap node".format(goal_distance))
    print("path driven:    {:.2f} m".format(path_length))
    print("collided on:    {} of {} ticks".format(
        sum(1 for record in records if record.collided), len(records)))
    print("reached goal:   {}".format(runner.reached_goal))


def main():
    args = parse_args()

    selected_gpu = gpu.select_gpu(args.gpu)
    print("GPU:        {} (pinned for both EGL and torch)".format(selected_gpu))

    import torch

    import bridge
    from nomad_policy import NomadPolicy

    spec = checkpoints.load(args.checkpoint)
    print("checkpoint: {}".format(spec.summary()))

    (start_position, start_orientation), route = read_start_pose(args.topomap)
    topomap = bridge.load_topomap(args.topomap)
    goal_pose = route["nodes"][-1]["pose"]
    print("topomap:    {} nodes from {}".format(len(topomap), args.topomap))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:     {}".format(device))
    torch.manual_seed(args.seed)

    policy = NomadPolicy(spec, device)
    body = bridge.SimBody()
    try:
        body.verify_gpu(selected_gpu)

        # Reproduce the start pose the route was driven from. The seed matters
        # because the task's reset samples a pose before we override it, and
        # the sampling advances numpy's global RNG.
        np.random.seed(route["start_pose"]["seed"])
        runner = bridge.NomadBridge(policy, body)
        runner.start_episode(topomap, (start_position, start_orientation))
        print("start:      {}".format(np.round(body.pose, 3)))
        print()

        records = drive(runner, args.max_ticks)
        report(records, runner, goal_pose, len(topomap))
    finally:
        body.close()

    gif_path = write_frames(records, args.output)
    if gif_path is None:
        print("wrote:      nothing — the run produced no ticks")
    else:
        print("wrote:      {} frames + {}".format(len(records), gif_path))


if __name__ == "__main__":
    # A bad GPU pin or a missing checkpoint is a message to a person, not a
    # defect to debug, so it exits with the reason rather than a traceback.
    try:
        main()
    except (gpu.GpuSelectionError, checkpoints.CheckpointError) as error:
        raise SystemExit("FAILED: {}".format(error))
