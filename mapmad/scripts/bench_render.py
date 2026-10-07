"""Measure RGB-D rendering speed (gate G0 row 1): steps per second at 320x240 on one GPU.

    python mapmad/scripts/bench_render.py --gpu 0                        # MP3D example + first labelled minival home
    python mapmad/scripts/bench_render.py --gpu 1 --hm3d-home 00800-TEEsavR23oF

One step = one agent action + rendering the colour and depth cameras (no semantic camera).
The agent turns 3 times, then moves forward once, as habitat-starter's check_setup does with turns.
Results are appended as one JSON line per scene to <outputs>/p0_setup/bench_render.jsonl.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

from habitat_starter import SimSettings, make_sim
from habitat_starter.sim import renderer_name

from mapmad_sim import config
from mapmad_sim.run_info import git_state

MP3D_EXAMPLE = "data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"


def measure(settings: SimSettings, steps: int, warmup: int = 20) -> Dict[str, object]:
    """Create the simulator, run `warmup` untimed steps, then time `steps` steps."""
    t0 = time.perf_counter()
    with make_sim(settings) as sim:
        ready_s = time.perf_counter() - t0
        actions = ["turn_left", "turn_left", "turn_left", "move_forward"]
        for i in range(warmup):
            sim.step(actions[i % 4])
        t0 = time.perf_counter()
        for i in range(steps):
            sim.step(actions[i % 4])
        elapsed = time.perf_counter() - t0
        renderer = renderer_name()
    return {"renderer": renderer, "steps": steps, "steps_per_s": round(steps / elapsed, 1), "sim_ready_s": round(ready_s, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True, help="CUDA device index habitat renders on (gpu_device_id)")
    p.add_argument("--hm3d-home", help="labelled minival home (default: the first one)")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--out", type=Path, help="JSONL file (default: <outputs>/p0_setup/bench_render.jsonl)")
    args = p.parse_args()

    paths = config.paths()
    home = args.hm3d_home or config.labelled_homes("minival")[0]
    scenes: Dict[str, Dict[str, Optional[str]]] = {
        "mp3d_example/17DRP5sb8fy": {"scene": str(paths["habitat_starter"] / MP3D_EXAMPLE), "dataset": "auto"},
        f"hm3d_minival/{home}": {"scene": str(config.hm3d_scene("minival", home)),
                                 "dataset": str(config.hm3d_scene_dataset_config("minival"))},
    }
    out = args.out or paths["outputs"] / "p0_setup" / "bench_render.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, scene in scenes.items():
        settings = SimSettings(scene=scene["scene"], scene_dataset_config=scene["dataset"], width=args.width,
                               height=args.height, color_sensor=True, depth_sensor=True, semantic_sensor=False,
                               gpu_device_id=args.gpu)
        result = {"scene": name, "gpu": args.gpu, "resolution": f"{args.width}x{args.height}", "sensors": "rgb+depth",
                  **measure(settings, args.steps), "commit": git_state().get("commit")}
        print(json.dumps(result), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
