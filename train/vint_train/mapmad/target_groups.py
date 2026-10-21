"""Phase 10 report-only diagnostic (does not change G10): the frozen search list split by what the target layer shows.

Per sample (drive, frame t), from the model's own input and the simulator's truth:
- (a) `empty`: the noisy target layer is all zero in both windows (near 64 x 64 @ 10 cm, far @ 40 cm) at frame t;
- (b) `in_view`: the layer is non-empty AND the target is in the camera picture at frame t (simulator truth: a
  detection of >= 200 target pixels at depth <= 4 m, `mapmad_target.npz` det_frame, misses of the noise included);
- (c) `not_in_view`: the layer is non-empty and the target is not in the picture at frame t.
Per group: n samples / n drives, mean loss of maps, and target_zeroed, target_shifted (1 m), clean each minus maps
(drive-level paired bootstrap 95% CI, 10,000). "Learned map effect" = (maps - hidden of ema_19) - (maps - hidden of
the step-0 model), per sample, same drive-level paired CI. Reads existing `offline_eval score` outputs; CPU only.

    python -m vint_train.mapmad.target_groups --config config/mapmad_v2.yaml --list <search_list.json> \
        --scores <ema_19 scores.npz> --pretrain-scores <step-0 scores.npz> --out <dir>
"""

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np
import tqdm

from vint_train.mapmad.map_sample_v2 import LAYER_INDEX, MapOptions
from vint_train.mapmad.map_v2 import TARGET_FILE
from vint_train.mapmad.offline_stats import paired_stats

GROUPS = ("empty", "in_view", "not_in_view")
VS_MAPS = ("target_zeroed", "target_shifted", "clean")


def target_group(layer_nonempty: bool, in_view: bool) -> str:
    """The group of one sample (module docstring)."""
    if not layer_nonempty:
        return "empty"
    return "in_view" if in_view else "not_in_view"


def detection_frames(data_folder: str, drive: str) -> np.ndarray:
    """Frames with a simulator-truth target detection (>= 200 px at <= 4 m) of one drive."""
    with np.load(os.path.join(data_folder, drive, TARGET_FILE)) as z:
        return np.unique(z["det_frame"])


def label_samples(ds, samples: List[Dict[str, Any]], data_folder: str) -> List[str]:
    """The group of every listed sample, from the maps arm's own map input (noisy layer) and the detections."""
    from vint_train.mapmad.dataset import pose_at

    det: Dict[str, np.ndarray] = {}
    k = LAYER_INDEX["target"]
    out = []
    for s in tqdm.tqdm(samples, desc="groups", dynamic_ncols=True):
        d, t = s["drive"], int(s["t"])
        if d not in det:
            det[d] = detection_frames(data_folder, d)
        built = ds.builder.build(d, t, pose_at(ds._get_trajectory(d), t), True, MapOptions())
        out.append(target_group(bool((built[:, k] > 0).any()), bool(np.isin(t, det[d]))))
    return out


def group_report(scores: Dict[str, np.ndarray], pre: Dict[str, np.ndarray], groups: np.ndarray) -> Dict[str, Any]:
    """Per group (and "all"): sizes, maps mean, each VS_MAPS arm minus maps, and the learned map effect."""
    drives = np.asarray(scores["drive"])
    loss = lambda z, arm: np.asarray(z[f"{arm}.gc_action_loss"], dtype=np.float64)  # noqa: E731
    maps = loss(scores, "maps")
    learned_a = maps - loss(scores, "hidden")
    learned_b = loss(pre, "maps") - loss(pre, "hidden")
    out: Dict[str, Any] = {}
    for g in ("all",) + GROUPS:
        sel = np.ones(len(maps), bool) if g == "all" else groups == g
        if not sel.any():
            out[g] = {"n_samples": 0, "n_drives": 0}
            continue
        row: Dict[str, Any] = {"n_samples": int(sel.sum()), "n_drives": int(len(np.unique(drives[sel]))),
                               "maps": float(maps[sel].mean())}
        for arm in VS_MAPS:
            row[arm] = paired_stats(loss(scores, arm)[sel], maps[sel], drives[sel])
        row["learned_map_effect"] = paired_stats(learned_a[sel], learned_b[sel], drives[sel])
        out[g] = row
    return out


def markdown(rep: Dict[str, Any]) -> str:
    """The report as a markdown table (arm - maps and learned effect: diff [drive CI])."""
    ci = lambda r: f"{r['diff']:+.3f} [{r['diff_ci_drive'][0]:+.3f}, {r['diff_ci_drive'][1]:+.3f}]"  # noqa: E731
    lines = ["| Group | n samples / drives | maps | target zeroed - maps | target shifted 1 m - maps | clean - maps "
             "| learned map effect |", "|---|---|---|---|---|---|---|"]
    for g, r in rep.items():
        if not r["n_samples"]:
            lines.append(f"| {g} | 0 / 0 | | | | | |")
            continue
        lines.append(f"| {g} | {r['n_samples']} / {r['n_drives']} | {r['maps']:.3f} | " +
                     " | ".join(ci(r[a]) for a in VS_MAPS + ("learned_map_effect",)) + " |")
    return "\n".join(lines)


def load_scores(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    from vint_train.mapmad.config import load_config
    from vint_train.mapmad.dataset import build_mapmad_dataset
    from vint_train.mapmad.offline_eval import map_dataset_name
    from vint_train.mapmad.offline_list import load_list

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for a in ("--config", "--list", "--scores", "--pretrain-scores", "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    body = load_list(args.list)
    samples = body["samples"]
    scores, pre = load_scores(args.scores), load_scores(args.pretrain_scores)
    ids = np.array([s["id"] for s in samples])
    assert np.array_equal(scores["ids"], ids) and np.array_equal(pre["ids"], ids), "scores not in list order"
    name = map_dataset_name(cfg)
    ds = build_mapmad_dataset(cfg, name, "test", train=False)
    groups = np.array(label_samples(ds, samples, cfg["datasets"][name]["data_folder"]))
    rep = group_report(scores, pre, groups)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "groups.npy"), groups)
    with open(os.path.join(args.out, "target_groups.json"), "w") as f:
        json.dump({"report": rep, "list": args.list, "list_sha256": body["sha256"], "scores": args.scores,
                   "pretrain_scores": args.pretrain_scores}, f, indent=1)
    md = markdown(rep)
    with open(os.path.join(args.out, "target_groups.md"), "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
