"""Language-goal Phase 7: look at a frozen word task set before anything runs on it.

Reads the set's manifest (never writes into the set) and writes, to a new
folder: a table of every task — word, target, start pose, geodesic to the
nearest instance, target in view at the start — and a top-down map of all the
starts with their headings, the targets' footprints and 1 m success regions.

    ./sim_eval/run_lg7.sh lg7_5_task_map.py --task-set sim_eval/outputs/p7_task_set_pilot_b \
        --output-dir sim_eval/outputs/p7_task_set_pilot_b_review
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import object_tasks
import objects

COLUMNS = ("task_id", "word", "target_instance", "view_class", "x", "y", "yaw_deg",
           "geodesic_length_m", "target_bearing_deg", "target_in_view", "seed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("FAILED: {} exists; the review goes to a new folder".format(
            args.output_dir))

    import gpu
    gpu.select_gpu(args.gpu)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    import bridge
    from recorder import floor_extent, traversable_bounds

    manifest, tasks = object_tasks.load(args.task_set)
    rows = []
    for task in tasks:
        start = task.entry["start_pose"]
        rows.append({"task_id": task.task_id, "word": task.word,
                     "target_instance": task.target_instance,
                     "view_class": task.entry.get("view_class"),
                     "x": start["x"], "y": start["y"],
                     "yaw_deg": round(math.degrees(start["yaw"]), 1),
                     "geodesic_length_m": task.geodesic_length_m,
                     "target_bearing_deg": task.entry.get("target_bearing_deg"),
                     "target_in_view": task.entry.get("target_in_view"), "seed": task.seed})

    body = bridge.SimBody(config_path=tasks[0].world_config)
    try:
        house = objects.SceneObjects.for_scene(body.scene)
        trav, res = body.scene.trav_map, body.scene.trav_map_resolution
        fig, ax = plt.subplots(figsize=(9, 11))
        ax.imshow(trav, cmap="gray", origin="lower", extent=floor_extent(trav, res),
                  interpolation="nearest", alpha=0.7)
        colours = {"chair": "tab:orange", "sofa": "tab:purple"}
        grid = house.grid
        xs = (np.arange(grid.size) + 0.5 - grid.size / 2.0) * grid.resolution
        for word in dict.fromkeys(t.word for t in tasks):
            colour = colours.get(word, "tab:blue")
            values = house.category_field(word).values
            ax.contour(xs, xs, np.where(np.isfinite(values), values, 99), levels=[1.0],
                       colors=[colour], linewidths=1.0, linestyles=":")
            for instance in house.of_category(word):
                target = any(t.target_instance == instance.id for t in tasks)
                for xmin, xmax, ymin, ymax in instance.boxes:
                    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           color=colour, alpha=0.7 if target else 0.2))
                if target:
                    ax.text(*instance.centre, "target\n" + instance.id, fontsize=8,
                            ha="center", va="center", fontweight="bold")
        for row in rows:
            colour = colours.get(row["word"], "tab:blue")
            yaw = math.radians(row["yaw_deg"])
            ax.annotate("", xy=(row["x"] + 0.35 * math.cos(yaw), row["y"] + 0.35 * math.sin(yaw)),
                        xytext=(row["x"], row["y"]),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.4))
            ax.scatter([row["x"]], [row["y"]], s=28, color=colour, edgecolor="black",
                       zorder=5, marker="o" if row["target_in_view"] else "s")
            ax.text(row["x"] + 0.06, row["y"] - 0.12, row["task_id"].split("_")[-1],
                    fontsize=7, color=colour)
        bounds = traversable_bounds(trav, res)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_title("{} — {} tasks, fingerprint {}\nstarts: arrow = heading, circle = target "
                     "in view, square = out of view; dotted = 1 m success region".format(
                         args.task_set.name, len(tasks), manifest["fingerprint"]), fontsize=9)
    finally:
        body.close()

    args.output_dir.mkdir(parents=True)
    fig.savefig(args.output_dir / "starts.png", dpi=100, bbox_inches="tight")
    with open(args.output_dir / "tasks.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print("TASK " + " ".join("{}={}".format(c, row[c]) for c in COLUMNS))
    print("fingerprint {}".format(manifest["fingerprint"]))


if __name__ == "__main__":
    main()
