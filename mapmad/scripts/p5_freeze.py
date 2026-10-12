"""Phase 5 S-freeze helpers for the unattended driver (run_p5_rest.sh). Inside naz_mapmad_habitat, from
/app/visualnav-transformer:

    python mapmad/scripts/p5_freeze.py rules                  # p5_rules.md + .sha256, BEFORE the draw (refuses overwrite)
    python mapmad/scripts/p5_freeze.py set-facts --pairs 6    # after the freeze: p5_rules_set.md + .sha256
    python mapmad/scripts/p5_freeze.py check --pairs 6 --gpus-free 1   # S-freeze conditions -> exam_set/sfreeze.md
    python mapmad/scripts/p5_freeze.py pick-videos --tables tables     # the 12 video episodes (id <TAB> label)

`rules` fills mapmad/configs/p5_rules_template.md (rules version 3, written before the exam set is drawn): waypoint
index, code snapshot hashes (<out_dir>/code_snapshot/SHA256SUMS, written by the driver), config sha256.
`set-facts` (after the freeze) writes the set-dependent values by the procedures fixed in p5_rules.md: fingerprint and
counts, the 8 seeded video episodes (default_rng(1414), 3/3/2 over the bins), the determinism episode
(default_rng(6006)) and the GPU-time estimate -> p5_rules_set.md (+ sha256) and p5_rules_choices.json.
`check` tests Naz's S-freeze conditions (2026-10-11; bins >= 50 since 2026-10-12): >= 250 episodes from >= 30 homes,
every bin >= 50; p5_rules.md and p5_rules_set.md match their sha256, p5_rules.md is older than the exam set, no exam
log yet; checkpoint sha256s match; goal-photo target share median in [0.05, 0.40]; GPUs free of other users' jobs
(measured on the host by the driver: --gpus-free); estimate <= 16 h.
Exit code 3 if any fails (the reasons are printed and in sfreeze.json).
"""

import argparse
import hashlib
import json
import string
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from mapmad_sim import config
from mapmad_sim.episodes import load_episodes
from mapmad_sim.exam_arms import arms_of
from mapmad_sim.run_layout import RunLayout

CONFIG = config.CONFIG_DIR / "p5_exam.yaml"
TEMPLATE = config.CONFIG_DIR / "p5_rules_template.md"
VIDEO_SEED, VIDEO_PER_BIN = 1414, (3, 3, 2)
PICK_SEED = 1515
DET_SEED = 6006
MIN_EPISODES, MIN_HOMES, MIN_PER_BIN = 250, 30, 50  # bins >= 50: Naz 2026-10-12 (was 60)
SHARE_RANGE = (0.05, 0.40)
MAX_HOURS = 16.0
SHARD_MARGIN = 1.10  # the slowest shard finishes after the mean
DEV_TIMING = {"A": "runs/dev", "B": "runs/dev_1a", "C": "runs/dev", "D": "runs/dev", "E": "runs/dev", "F": "runs/dev_1a"}
DEV_ARM = {"C": "C_wp1"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def video_episodes(episodes: List[Dict[str, Any]]) -> List[str]:
    """8 ids: default_rng(1414), 3 / 3 / 2 from the bins (uniform within a bin over the sorted ids)."""
    rng = np.random.default_rng(VIDEO_SEED)
    out = []
    for b, k in enumerate(VIDEO_PER_BIN):
        ids = sorted(e["episode_id"] for e in episodes if e["distance_bin"] == b)
        out += [ids[i] for i in sorted(rng.choice(len(ids), min(k, len(ids)), replace=False))]
    return out


def determinism_episode(episodes: List[Dict[str, Any]]) -> str:
    ids = sorted(e["episode_id"] for e in episodes)
    return ids[int(np.random.default_rng(DET_SEED).integers(len(ids)))]


def gpu_estimate(layout: RunLayout, n_episodes: int, pairs: int) -> Dict[str, Any]:
    """Wall hours from the dev runs' mean seconds per run (dev ran 6-7 pairs at once, like the exam)."""
    seeds = int(layout.cfg["run"]["seeds"])
    per_arm, total = {}, 0.0
    for a in arms_of(layout.cfg):
        arm = a.name
        files = sorted((layout.base / DEV_TIMING[arm] / "timing").glob(f"{DEV_ARM.get(arm, arm)}.shard*.jsonl"))
        walls = [json.loads(line)["wall_s"] for f in files for line in f.read_text().splitlines() if line.strip()]
        runs = n_episodes * len(a.seed_indices(range(seeds)))
        mean = float(np.mean(walls)) if walls else float("nan")
        per_arm[arm] = {"dev_runs": len(walls), "mean_wall_s": round(mean, 1), "exam_runs": runs}
        total += mean * runs
    hours = total / pairs / 3600 * SHARD_MARGIN
    return {"pairs": pairs, "shard_margin": SHARD_MARGIN, "per_arm": per_arm, "hours": round(hours, 2)}


def snapshot(layout: RunLayout) -> Dict[str, str]:
    """commit / patch / untracked hashes from <out_dir>/code_snapshot/SHA256SUMS (written by the driver)."""
    folder = layout.base / "code_snapshot"
    sums = dict(reversed(line.split()) for line in (folder / "SHA256SUMS").read_text().splitlines() if line.strip())
    return {"commit": (folder / "COMMIT").read_text().strip(), "patch": sums["tracked.patch"],
            "untracked": sums["untracked.tar"]}


def cmd_rules(args, layout: RunLayout) -> None:
    out = layout.base / "p5_rules.md"
    if out.exists():
        raise SystemExit(f"{out} exists: the rules are frozen")
    if layout.episodes_file.exists():
        raise SystemExit(f"{layout.episodes_file} exists: the rules must be written before the draw")
    snap = snapshot(layout)
    text = string.Template(TEMPLATE.read_text()).substitute(
        WRITTEN=datetime.now(timezone.utc).isoformat(timespec="seconds"), COMMIT=snap["commit"],
        PATCH_SHA=snap["patch"], UNTRACKED_SHA=snap["untracked"], CONFIG_SHA=sha256(CONFIG),
        WP=layout.cfg["mapmad"]["waypoint_index"])
    out.write_text(text)
    (layout.base / "p5_rules.sha256").write_text(f"{sha256(out)}  {out.name}\n")
    print(f"{out} sha256 {sha256(out)}")


def cmd_set_facts(args, layout: RunLayout) -> None:
    out = layout.base / "p5_rules_set.md"
    if out.exists():
        raise SystemExit(f"{out} exists")
    eps, fp = load_episodes(layout.episodes_file)
    est = gpu_estimate(layout, len(eps), args.pairs)
    bins = layout.cfg["set_rules"]["bins_m"]
    per_bin = ", ".join(f"{bins[b]:g}-{bins[b + 1]:g} m {sum(e['distance_bin'] == b for e in eps)}"
                        for b in range(len(bins) - 1))
    videos, det = video_episodes(eps), determinism_episode(eps)
    rules = layout.base / "p5_rules.md"
    lines = ["# Phase 5 exam: values of the drawn set (procedures in p5_rules.md)", "",
             f"Written {datetime.now(timezone.utc).isoformat(timespec='seconds')}, after the freeze and before any exam "
             f"run. p5_rules.md sha256 `{sha256(rules)}`.", "",
             f"- Exam set: `exam_set/episodes.json`, fingerprint `{fp}`, file sha256 `{sha256(layout.episodes_file)}`.",
             f"- {len(eps)} episodes from {len(set(e['home'] for e in eps))} homes; per bin: {per_bin}.",
             f"- Goal-photo target share: median {np.median([e['goal_photo_share'] for e in eps]):.3f}.",
             f"- Determinism episode (default_rng(6006)): **{det}**.",
             f"- Video episodes (default_rng(1414), 3 / 3 / 2 over the bins): {', '.join(videos)}.",
             f"- GPU estimate: {est['hours']:.1f} h on {est['pairs']} pairs (dev mean seconds per run, margin "
             f"{est['shard_margin']})."]
    out.write_text("\n".join(lines) + "\n")
    (layout.base / "p5_rules_set.sha256").write_text(f"{sha256(out)}  {out.name}\n")
    (layout.base / "p5_rules_choices.json").write_text(json.dumps(
        {"video_episodes": videos, "determinism_episode": det, "gpu_estimate": est, "fingerprint": fp}, indent=1) + "\n")
    print("\n".join(lines))


def conditions(layout: RunLayout, pairs: int, gpus_free: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eps, fp = load_episodes(layout.episodes_file)
    homes = {e["home"] for e in eps}
    per_bin = [sum(e["distance_bin"] == b for e in eps) for b in range(len(layout.cfg["set_rules"]["bins_m"]) - 1)]
    def sha_ok(name: str) -> bool:
        f, h = layout.base / f"{name}.md", layout.base / f"{name}.sha256"
        return f.exists() and h.exists() and h.read_text().split()[0] == sha256(f)

    rules = layout.base / "p5_rules.md"
    before = rules.exists() and rules.stat().st_mtime < layout.episodes_file.stat().st_mtime
    rules_ok = sha_ok("p5_rules") and sha_ok("p5_rules_set") and before
    logs = list((layout.base / "runs" / "logs").glob("*/*.jsonl*")) if (layout.base / "runs" / "logs").exists() else []
    ck = {"mapmad": (layout.resolve(layout.cfg["mapmad"]["checkpoint"]), layout.cfg["mapmad"]["checkpoint_sha256"]),
          "nomad": (layout.resolve(layout.cfg["nomad"]["weights"]), layout.cfg["nomad"]["weights_sha256"])}
    ck_now = {k: sha256(p) for k, (p, _) in ck.items()}
    share = float(np.median([e["goal_photo_share"] for e in eps]))
    est = gpu_estimate(layout, len(eps), pairs)
    rows = [
        {"check": f"exam set >= {MIN_EPISODES} episodes from >= {MIN_HOMES} homes, every bin >= {MIN_PER_BIN}",
         "value": f"{len(eps)} episodes, {len(homes)} homes, bins {per_bin}",
         "pass": len(eps) >= MIN_EPISODES and len(homes) >= MIN_HOMES and min(per_bin) >= MIN_PER_BIN},
        {"check": "rules written before the draw, rules + set values match their sha256, no exam run yet",
         "value": f"p5_rules.md + p5_rules_set.md {'match' if sha_ok('p5_rules') and sha_ok('p5_rules_set') else 'missing or NOT matching'} "
                  f"their sha256; p5_rules.md {'older' if before else 'NOT older'} than the exam set; exam logs present: {len(logs)}",
         "pass": rules_ok and not logs},
        {"check": "checkpoint sha256 matches", "value": ", ".join(f"{k} {v[:12]}..." for k, v in ck_now.items()),
         "pass": all(ck_now[k] == ck[k][1] for k in ck)},
        {"check": f"goal-photo target share median in [{SHARE_RANGE[0]:.0%}, {SHARE_RANGE[1]:.0%}]",
         "value": f"{share:.3f}", "pass": SHARE_RANGE[0] <= share <= SHARE_RANGE[1]},
        {"check": "GPUs free of other users' jobs (host nvidia-smi)", "value": "free" if gpus_free else "NOT free",
         "pass": gpus_free},
        {"check": f"GPU estimate <= {MAX_HOURS:g} h", "value": f"{est['hours']:.1f} h on {pairs} pairs",
         "pass": est["hours"] <= MAX_HOURS},
    ]
    return rows, {"fingerprint": fp, "episodes": len(eps), "homes": len(homes), "per_bin": per_bin,
                  "photo_share_median": share, "gpu_estimate": est}


def cmd_check(args, layout: RunLayout) -> None:
    rows, facts = conditions(layout, args.pairs, bool(args.gpus_free))
    ok = all(r["pass"] for r in rows)
    folder = layout.episodes_dir
    (folder / "sfreeze.json").write_text(json.dumps({"pass": ok, "rows": rows, **facts}, indent=1, default=float) + "\n")
    est = facts["gpu_estimate"]
    lines = ["# S-freeze (automatic, pre-approved by Naz 2026-10-11 if every row passes)", "",
             f"Checked {datetime.now(timezone.utc).isoformat(timespec='seconds')}. Exam set `{facts['fingerprint']}`.", "",
             "| condition | value | |", "|---|---|---|"]
    lines += [f"| {r['check']} | {r['value']} | {'pass' if r['pass'] else 'FAIL'} |" for r in rows]
    lines += ["", f"**{'ALL PASS: the exam runs start.' if ok else 'STOP: the exam runs do not start.'}**", "",
              "## GPU estimate (dev mean seconds per run x exam runs / pairs x shard margin)", "",
              "| arm | dev runs | mean s/run | exam runs |", "|---|---|---|---|"]
    lines += [f"| {a} | {v['dev_runs']} | {v['mean_wall_s']} | {v['exam_runs']} |" for a, v in est["per_arm"].items()]
    lines += ["", f"{est['hours']} h on {est['pairs']} pairs (margin {est['shard_margin']}).", "",
              "Set statistics: `build_report.md`; 20 random goal photos: `goal_photos_sheet.png`; rules: "
              "`../p5_rules.md`, `../p5_rules_set.md` (+ `.sha256`)."]
    (folder / "sfreeze.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not ok:
        sys.exit(3)


def cmd_pick_videos(args, layout: RunLayout) -> None:
    """Print the 12 video episodes: the 8 pre-registered ones, then 4 picked by the rule in p5_rules.md 9."""
    import csv

    choices = json.loads((layout.base / "p5_rules_choices.json").read_text())
    picked = [(e, "random pick (seeded)") for e in choices["video_episodes"]]
    runs: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(layout.base / args.tables / "runs.csv") as f:
        for r in csv.DictReader(f):
            if r["seed_index"] == "0":
                runs[(r["arm"], r["episode_id"])] = r
    ids = sorted({e for _, e in runs})
    order = [ids[i] for i in np.random.default_rng(PICK_SEED).permutation(len(ids))]

    def ok(arm: str, eid: str, cond) -> bool:
        return (arm, eid) in runs and cond(runs[(arm, eid)])

    rules = [
        ("picked by rule: C stuck", lambda e: ok("C", e, lambda r: r["failure_type"] == "stuck")),
        ("picked by rule: C circling", lambda e: ok("C", e, lambda r: r["failure_type"] == "circling")),
        ("picked by rule: C timeout while moving", lambda e: ok("C", e, lambda r: r["failure_type"] == "timeout_moving")),
        ("picked by rule: C success, B fails", lambda e: ok("C", e, lambda r: float(r["success"]) == 1)
         and ok("B", e, lambda r: float(r["success"]) == 0)),
        ("picked by rule: C fails, E succeeds", lambda e: ok("C", e, lambda r: float(r["success"]) == 0)
         and ok("E", e, lambda r: float(r["success"]) == 1)),
    ]
    used = {e for e, _ in picked}
    have_circling = False
    for label, cond in rules:
        if "timeout" in label and have_circling:
            continue
        for e in order:
            if e not in used and cond(e):
                picked.append((e, label))
                used.add(e)
                have_circling = have_circling or "circling" in label
                break
    for e, label in picked:
        print(f"{e}\t{label}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["rules", "set-facts", "check", "pick-videos"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--gpus-free", type=int, default=0)
    p.add_argument("--tables", default="tables")
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    {"rules": cmd_rules, "set-facts": cmd_set_facts, "check": cmd_check, "pick-videos": cmd_pick_videos}[args.cmd](args, layout)


if __name__ == "__main__":
    main()
