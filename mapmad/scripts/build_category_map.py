"""Write configs/objectnav_categories.yaml: HM3D object names -> ObjectNav v2 categories, learnt from the train split.

    python mapmad/scripts/build_category_map.py            # inside naz_mapmad_habitat, ~1 min

Only train homes are read (never val). See mapmad_sim/categories.py for the rule.
"""

import argparse
from datetime import date

import yaml

from mapmad_sim import categories, config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-count", type=int, default=20, help="a name must be a goal at least this often")
    p.add_argument("--min-purity", type=float, default=0.95, help="... and this share of its goal uses in one category")
    p.add_argument("--out", default=str(categories.MAP_FILE))
    args = p.parse_args()

    paths = config.paths()
    counts = categories.count_goal_names(paths["objectnav"] / "train", paths["hm3d_scenes"], "train")
    mapping = categories.name_map(counts, args.min_count, args.min_purity)
    dropped = {n: dict(c) for n, c in counts.items() if n not in mapping and sum(c.values()) >= args.min_count}
    doc = {
        "about": "HM3D Semantics v0.2 object name -> ObjectNav HM3D v2 category; built by "
                 "mapmad/scripts/build_category_map.py from the 145 train homes' goals_by_category "
                 f"(min_count {args.min_count}, min_purity {args.min_purity}) on {date.today().isoformat()}",
        "names": mapping,
        "goal_counts": {n: dict(counts[n]) for n in mapping},
        "dropped_mixed_names": dropped,
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    per_cat = {c: sorted(n for n, k in mapping.items() if k == c) for c in categories.CATEGORIES}
    for c, names in per_cat.items():
        print(f"{c}: {len(names)} names, e.g. {names[:8]}")
    print(f"dropped (mixed): {dropped}\nwrote {args.out}")


if __name__ == "__main__":
    main()
