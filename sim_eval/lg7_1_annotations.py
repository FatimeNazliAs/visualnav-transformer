"""Language-goal Phase 7: check the object annotations of one house by looking.

An annotation is a claim that there is a chair here. This makes each claim
checkable:

  * every instance's goal photo, rendered from its `view_from` through the
    robot camera — the photo a photo arm will be handed;
  * CLIP's own zero-shot reading of that photo: where the instance's category
    ranks among the 32 goal words (1 = CLIP's top word). A chair photo in
    which CLIP ranks "chair" 20th is a photo of something else;
  * an overview map: footprints, photo poses, each category's 1 m success
    region, and how many cells a task of each word may start from;
  * that objects.GridMap's cell arithmetic agrees with the live scene's own
    `world_to_map`, which the distance field silently depends on.

    ./sim_eval/run_lg7_1_annotations.sh [--scene Rs]

Writes outputs/lg7_1_annotations/<scene>/: <instance>.png, overview.png,
annotations.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import checkpoints
import gpu
import object_tasks
import objects

SIM_EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SIM_EVAL_DIR / "outputs" / "lg7_1_annotations"
WORLD_CONFIG = SIM_EVAL_DIR / "configs" / "locobot_rs_bridge.yaml"
CLIP_ARM = "clip_v2b"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="Rs")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def check_grid(scene_objects, sim_scene):
    """GridMap.cell against the scene's own world_to_map, at random points."""
    rng = np.random.RandomState(0)
    for xy in rng.uniform(-4.0, 4.0, size=(50, 2)):
        rows, cols = sim_scene.world_to_map([xy])[::-1]  # SimScene returns (cols, rows)
        ours = scene_objects.grid.cell(xy)
        if ours != (int(rows[0]), int(cols[0])):
            raise SystemExit("FAILED: GridMap.cell{} = {}, scene says {}".format(
                tuple(xy), ours, (int(rows[0]), int(cols[0]))))
    return True


def clip_ranks(clip, photo, categories_to_rank):
    """Rank of each category word among GOAL_WORDS for one photo (1 = top)."""
    import torch
    from vint_train.data.clip_goal_utils import GOAL_WORDS, l2_normalize

    image = l2_normalize(clip.raw_image(photo))
    texts = l2_normalize(torch.stack([clip.raw_word(word) for word in GOAL_WORDS]))
    scores = (texts @ image).cpu().numpy()
    order = [GOAL_WORDS[i] for i in np.argsort(-scores)]
    return {word: order.index(word) + 1 for word in categories_to_rank}, order[:3]


def draw_overview(path, scene_objects, sim_scene, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from recorder import floor_extent, traversable_bounds

    colours = {"chair": "tab:orange", "sofa": "tab:purple", "table": "tab:green"}
    fig, ax = plt.subplots(figsize=(9, 11))
    trav, res = sim_scene.trav_map, sim_scene.trav_map_resolution
    ax.imshow(trav, cmap="gray", origin="lower", extent=floor_extent(trav, res),
              interpolation="nearest", alpha=0.6)
    grid = scene_objects.grid
    xs = (np.arange(grid.size) - grid.size / 2.0) * grid.resolution
    for category in objects.categories(scene_objects.instances.values()):
        colour = colours.get(category, "tab:blue")
        values = scene_objects.category_field(category).values
        ax.contour(xs, xs, np.where(np.isfinite(values), values, 99), levels=[1.0],
                   colors=[colour], linewidths=1.2, linestyles="--")
        for instance in scene_objects.of_category(category):
            for xmin, xmax, ymin, ymax in instance.boxes:
                ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       color=colour, alpha=0.45))
            vx, vy = instance.view_from
            yaw = instance.view_yaw()
            ax.annotate("", xy=(vx + 0.4 * np.cos(yaw), vy + 0.4 * np.sin(yaw)),
                        xytext=(vx, vy), arrowprops=dict(arrowstyle="->", color=colour))
            ax.text(*instance.centre, instance.id, fontsize=7, ha="center", va="center")
    bounds = traversable_bounds(trav, res)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_title("{}: footprints, photo poses (arrows), 1 m success region (dashed)\n"
                 "start cells per word: {}".format(
                     scene_objects.scene_id,
                     ", ".join("{} {}".format(k, v) for k, v in counts.items())), fontsize=9)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    selected_gpu = gpu.select_gpu(args.gpu)

    import torch

    import bridge
    import goals

    out = OUTPUT_DIR / args.scene
    out.mkdir(parents=True, exist_ok=True)
    rules = object_tasks.StartRules()
    world_path = out / "world.yaml"
    config = object_tasks.ObjectTaskSetConfig([args.scene], ["chair"], world_config=WORLD_CONFIG)
    import yaml
    world_path.write_text(yaml.safe_dump(config.world(args.scene), sort_keys=False))

    body = bridge.SimBody(config_path=world_path)
    try:
        body.verify_gpu(selected_gpu)
        scene_objects = objects.SceneObjects.for_scene(body.scene)
        check_grid(scene_objects, body.scene)
        print("grid:       GridMap.cell agrees with the scene's world_to_map (50 points)")

        spec = checkpoints.load_goal_arm(CLIP_ARM)
        clip = goals.ClipEncoder(spec.model_params, torch.device("cuda"))
        report = {"scene": args.scene, "instances": [], "start_cells": {}}
        for instance in scene_objects.instances.values():
            photo = goals.render_goal_photo(body, instance)
            photo.save(out / "{}.png".format(instance.id))
            view_distance = scene_objects.instance_field(instance.id).distance(instance.view_from)
            ranks, top = clip_ranks(clip, photo, [instance.category])
            row = {"id": instance.id, "category": instance.category,
                   "view_from_free": scene_objects.grid.is_free(instance.view_from),
                   "view_geodesic_m": None if view_distance is None else round(view_distance, 3),
                   "clip_rank": ranks[instance.category], "clip_top3": top}
            report["instances"].append(row)
            print("{:<8} {:<6} view free {!s:<5} {:>5} m from it   CLIP rank of '{}': "
                  "{:>2}/32   top-3 {}".format(
                      instance.id, instance.category, row["view_from_free"],
                      row["view_geodesic_m"], instance.category, row["clip_rank"], top))

        for category in objects.categories(scene_objects.instances.values()):
            candidates = object_tasks.start_candidates(
                scene_objects, category, rules, has_node=body.scene.has_node)
            values = scene_objects.category_field(category).values
            report["start_cells"][category] = {
                "candidates": len(candidates),
                "max_geodesic_m": round(float(np.max(values[np.isfinite(values)])), 2),
                "targets": sorted({c[2] for c in candidates}),
            }
            print("{:<6} start cells {:>4} (2-5 m, line of sight)   field max {:.2f} m   "
                  "nearest targets {}".format(category, len(candidates),
                                              report["start_cells"][category]["max_geodesic_m"],
                                              report["start_cells"][category]["targets"]))
        draw_overview(out / "overview.png", scene_objects, body.scene,
                      {k: v["candidates"] for k, v in report["start_cells"].items()})
    finally:
        body.close()
    (out / "annotations.json").write_text(json.dumps(report, indent=2))
    print("wrote:      {}".format(out))


if __name__ == "__main__":
    main()
