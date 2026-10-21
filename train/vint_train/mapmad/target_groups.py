"""Phase 10 report-only diagnostic (does not change G10): the frozen search list split by what the target layer shows.

Per sample (drive, frame t), from the model's own input and the simulator's truth:
- (a) `empty`: the noisy target layer is all zero in both windows (near 64 x 64 @ 10 cm, far @ 40 cm) at frame t;
- (b) `in_view`: the layer is non-empty AND the target is in the camera picture at frame t (simulator truth: a
  detection of >= 200 target pixels at depth <= 4 m, `mapmad_target.npz` det_frame, misses of the noise included);
- (c) `not_in_view`: the layer is non-empty and the target is not in the picture at frame t.
Per group: n samples / n drives, mean loss of maps, and target_zeroed, target_shifted (1 m), clean each minus maps
(drive-level paired bootstrap 95% CI, 10,000). "Learned map effect" = (maps - hidden of ema_19) - (maps - hidden of
the step-0 model), per sample, same drive-level paired CI. Reads existing `offline_eval score` outputs; CPU only.

Seen-before-out-of-view check (Naz's option B, p10_rules.md v3): `make-seen-list` freezes EVERY frame t of the
held-out search drives (test split minus collision drives) with t - 15 >= 0, t + 40 <= last frame, a simulator-truth
detection at some frame < t and none at t; `seen-report` scores target_zeroed and target_shifted (1 m) against maps
(drive-level paired CI) and applies the pre-set rule: CI lower bound of (target_zeroed - maps) > 0 -> "A" else "C";
`train-share` = the share of training search-drive index entries that are seen-before-out-of-view.

    python -m vint_train.mapmad.target_groups groups --config config/mapmad_v2.yaml --list <search_list.json> \
        --scores <ema_19 scores.npz> --pretrain-scores <step-0 scores.npz> --out <dir>
    python -m vint_train.mapmad.target_groups make-seen-list --config config/mapmad_v2.yaml --out <seen_list.json>
    python -m vint_train.mapmad.target_groups seen-report --scores <scores.npz> --out <dir>
    python -m vint_train.mapmad.target_groups train-share --config config/mapmad_v2.yaml --out <json>
    python -m vint_train.mapmad.target_groups guard --scores <run C search scores.npz> --reference <run 1 ...> --out <json>
(`guard`, p10_rules_v4.md: maps loss of run C minus run 1 on the frozen 5,000 search samples, drive-level paired CI;
pass = CI upper bound <= +0.05.)
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import tqdm

from vint_train.mapmad.config import load_config
from vint_train.mapmad.map_sample_v2 import LAYER_INDEX, MapOptions
from vint_train.mapmad.map_v2 import TARGET_FILE
from vint_train.mapmad.offline_stats import paired_stats

GROUPS = ("empty", "in_view", "not_in_view")
VS_MAPS = ("target_zeroed", "target_shifted", "clean")
SEEN_ARMS = ("maps", "target_zeroed", "target_shifted")


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


def seen_out_of_view(det_frames: np.ndarray, t: int) -> bool:
    """True if the target was detected (simulator truth) at some frame < t and is not detected at frame t."""
    return bool(len(det_frames) and det_frames.min() < t and not np.isin(t, det_frames))


def seen_frames(det_frames: np.ndarray, length: int) -> List[int]:
    """Every valid frame (t - 15 >= 0, t + 40 <= length - 1) of a drive that is seen-before-out-of-view."""
    from vint_train.mapmad.offline_list import AHEAD_FRAMES, BACK_FRAMES

    return [t for t in range(BACK_FRAMES, length - AHEAD_FRAMES) if seen_out_of_view(det_frames, t)]


def decide(row: Dict[str, Any]) -> str:
    """The pre-set rule (p10_rules.md v3): drive-CI lower bound of (target_zeroed - maps) > 0 -> "A", else "C"."""
    return "A" if row["diff_ci_drive"][0] > 0 else "C"


def seen_report(scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """target_zeroed and target_shifted minus maps (drive-level paired CI) and the decision."""
    drives = np.asarray(scores["drive"])
    maps = np.asarray(scores["maps.gc_action_loss"], dtype=np.float64)
    out: Dict[str, Any] = {"n_samples": int(len(maps)), "n_drives": int(len(np.unique(drives))), "maps": float(maps.mean())}
    for arm in SEEN_ARMS[1:]:
        out[arm] = paired_stats(np.asarray(scores[f"{arm}.gc_action_loss"], dtype=np.float64), maps, drives)
    out["decision"] = decide(out["target_zeroed"])
    return out


def _search_drives(cfg: Dict[str, Any], split: str) -> Tuple[List[str], int]:
    """(search drives of a split, number of drives of the split), collision drives excluded."""
    from vint_train.mapmad.dataset import read_drive_list
    from vint_train.mapmad.map_sample_v2 import DriveStoreV2

    d = cfg["datasets"]["habitat_mapmad_v2"]
    excluded = read_drive_list(d.get("exclude_drives"))
    names = [x.strip() for x in open(os.path.join(d[split], "traj_names.txt")) if x.strip() and x.strip() not in excluded]
    kinds = DriveStoreV2(d["data_folder"], names, cfg["map"]["frontier_cache"]).kind
    return sorted(x for x in names if kinds[x] == "search"), len(names)


def cmd_make_seen_list(args) -> None:
    import pickle

    from vint_train.mapmad.offline_list import canonical_sha256
    from vint_train.mapmad.offline_v2 import v2_fields

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists (frozen)")
    cfg = load_config(args.config)
    folder = cfg["datasets"]["habitat_mapmad_v2"]["data_folder"]
    drives, n_held_out = _search_drives(cfg, "test")
    samples = []
    for d in tqdm.tqdm(drives, desc="seen list", dynamic_ncols=True):
        with open(os.path.join(folder, d, "traj_data.pkl"), "rb") as f:
            traj = pickle.load(f)
        with open(os.path.join(folder, d, "mapmad_meta.json")) as f:
            meta = json.load(f)
        for t in seen_frames(detection_frames(folder, d), len(traj["position"])):
            samples.append({"drive": d, "t": int(t), **v2_fields(meta, traj, int(t))})
    for i, s in enumerate(samples):
        s["id"] = i
    body = {"samples": samples, "kind": "seen_before_out_of_view", "n_held_out_drives": n_held_out,
            "n_search_drives": len(drives), "n_drives": len({s["drive"] for s in samples}),
            "rule": "every t with t - 15 >= 0, t + 40 <= last frame, a detection (>= 200 px, <= 4 m) at a frame < t, "
                    "none at t", "sha256": canonical_sha256(samples)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(body, f)
    print(f"{len(samples)} samples from {body['n_drives']} of {len(drives)} search drives ({n_held_out} held-out "
          f"drives); sha256 {body['sha256']}")


def cmd_seen_report(args) -> None:
    rep = seen_report(load_scores(args.scores))
    with open(os.path.join(os.path.dirname(os.path.abspath(args.scores)), "manifest.json")) as f:
        rep["manifest"] = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "seen_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    ci = lambda r: f"{r['diff']:+.4f} [{r['diff_ci_drive'][0]:+.4f}, {r['diff_ci_drive'][1]:+.4f}]"  # noqa: E731
    print(f"{rep['n_samples']} samples / {rep['n_drives']} drives; maps {rep['maps']:.4f}; target_zeroed - maps "
          f"{ci(rep['target_zeroed'])}; target_shifted - maps {ci(rep['target_shifted'])} -> decision {rep['decision']}")


def cmd_train_share(args) -> None:
    from vint_train.mapmad.dataset import build_mapmad_dataset
    from vint_train.mapmad.offline_eval import map_dataset_name

    cfg = load_config(args.config)
    folder = cfg["datasets"]["habitat_mapmad_v2"]["data_folder"]
    ds = build_mapmad_dataset(cfg, map_dataset_name(cfg), "train", train=False)
    search = set(_search_drives(cfg, "train")[0])
    det: Dict[str, np.ndarray] = {}
    n = hit = 0
    for d, t, *_ in tqdm.tqdm(ds.index_to_data, desc="train index", dynamic_ncols=True):
        if d not in search:
            continue
        if d not in det:
            det[d] = detection_frames(folder, d)
        n += 1
        hit += seen_out_of_view(det[d], int(t))
    out = {"search_index_entries": n, "seen_before_out_of_view": hit, "share": hit / n, "n_search_drives": len(det)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(out)


GUARD_MARGIN = 0.05


def guard(scores: Dict[str, np.ndarray], reference: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """maps(run C) - maps(run 1) on the same samples (drive-level paired CI); pass = CI upper bound <= GUARD_MARGIN."""
    if not np.array_equal(scores["ids"], reference["ids"]):
        raise SystemExit("guard: the two score files are not on the same samples")
    row = paired_stats(np.asarray(scores["maps.gc_action_loss"], dtype=np.float64),
                       np.asarray(reference["maps.gc_action_loss"], dtype=np.float64), np.asarray(scores["drive"]))
    row["margin"] = GUARD_MARGIN
    row["pass"] = bool(row["diff_ci_drive"][1] <= GUARD_MARGIN)
    return row


def cmd_guard(args) -> None:
    row = guard(load_scores(args.scores), load_scores(args.reference))
    row.update(scores=args.scores, reference=args.reference)
    with open(args.out, "w") as f:
        json.dump(row, f, indent=1)
    print(f"guard: maps C {row['mean_a']:.4f} vs run 1 {row['mean_b']:.4f}: {row['diff']:+.4f} "
          f"[{row['diff_ci_drive'][0]:+.4f}, {row['diff_ci_drive'][1]:+.4f}] -> {'PASS' if row['pass'] else 'FAIL'}")


def cmd_groups(args) -> None:
    from vint_train.mapmad.dataset import build_mapmad_dataset
    from vint_train.mapmad.offline_eval import map_dataset_name
    from vint_train.mapmad.offline_list import load_list

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("groups")
    for a in ("--config", "--list", "--scores", "--pretrain-scores", "--out"):
        p.add_argument(a, required=True)
    p.set_defaults(fn=cmd_groups)
    p = sub.add_parser("make-seen-list")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_make_seen_list)
    p = sub.add_parser("seen-report")
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_seen_report)
    p = sub.add_parser("train-share")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_train_share)
    p = sub.add_parser("guard")
    for a in ("--scores", "--reference", "--out"):
        p.add_argument(a, required=True)
    p.set_defaults(fn=cmd_guard)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
