"""Phase 11 closed-loop tables from the run summaries (logs alone; both containers can run it, numpy only).

    python -m vint_train.mapmad.closed_loop.search_tables pick --config <p11_dagger.yaml> --subdir step0
    python -m vint_train.mapmad.closed_loop.search_tables compare --config <p11_dagger.yaml> --subdir dev_v21 \
        --arms v2=step0/run1 v21 --out dev_v21/table

`pick` (G11 row 0, block item 1; p11_rules.md): success per episode = mean over its 3 seeds; Delta = mean over the dev
episodes of (run C - run 1); **run C becomes v2 only if Delta >= +0.03, otherwise run 1 (a tie -> run 1)**. Episodes
with a crash record in either judged arm are dropped from both (Phase 5 rule) and counted; a crash rate > 1% of the
judged runs or a missing run -> automatic stop (exit code 3, nothing written but the reason). SPL, search / approach
steps and stuck streaks are reported, never decide; run 1 with maps hidden (1 seed) is reported only. Writes
<subdir>/decision.json + decision.md and <out_dir>/V2_FINAL.json + V2_FINAL.txt (path, sha256, train config, Delta,
the rule).
`compare` (report only): the same table for any arms (e.g. v2.1 vs v2 on the same dev set, seeds and settings); an arm
`<label>=<subdir>/<arm>` is read from another run folder (v2's runs are its step-0 runs, same code and seeds).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_sim.episodes import load_episodes
from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic

MARGIN = 0.03  # p11_rules.md: run C must beat run 1 by at least this much (per-episode mean success)
MAX_CRASH_RATE = 0.01


def pick_rule(success_run1: np.ndarray, success_runC: np.ndarray, margin: float = MARGIN) -> Tuple[float, str]:
    """(Delta, winner): Delta = mean over episodes of (run C - run 1) per-episode success; run C only if Delta >=
    margin, otherwise run 1 (a tie -> run 1)."""
    delta = float((np.asarray(success_runC) - np.asarray(success_run1)).mean())
    return delta, ("runC" if delta >= margin - 1e-12 else "run1")


def read_runs(layout: RunLayout, arm: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """episode id -> seed index -> summary (crash records as {"crash": True})."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for p in sorted((layout.root / "summaries" / arm).glob("*.json")):
        eid, k = parse_run_id(p.name)
        out[eid][k] = {"crash": True} if p.name.endswith(".crash.json") else json.loads(p.read_text())
    return out


def per_episode(runs: Dict[str, Dict[int, Dict[str, Any]]], episodes: Sequence[str], key: str) -> np.ndarray:
    """Mean of a summary field over each episode's seeds (None counts as 0 for success, else NaN)."""
    vals = []
    for e in episodes:
        xs = [r.get(key) for r in runs[e].values()]
        xs = [float(x) if x is not None else (0.0 if key == "success" else np.nan) for x in xs]
        vals.append(np.nanmean(xs) if xs and not all(np.isnan(xs)) else np.nan)
    return np.asarray(vals, dtype=np.float64)


def arm_table(runs: Dict[str, Dict[int, Dict[str, Any]]], episodes: Sequence[str],
              cats: Dict[str, str]) -> Dict[str, Any]:
    """Success, SPL, steps to first sight, approach steps, collisions, stuck share; per category success."""
    flat = [r for e in episodes for r in runs[e].values()]
    seen = [r for r in flat if r.get("first_in_view_step") is not None]
    succ = per_episode(runs, episodes, "success")
    row = {"episodes": len(episodes), "runs": len(flat), "success": float(succ.mean()),
           "spl": float(np.nanmean(per_episode(runs, episodes, "spl"))),
           "seen_share": len(seen) / len(flat) if flat else float("nan"),
           "search_steps_median": float(np.median([r["search_steps"] for r in flat])) if flat else float("nan"),
           "approach_steps_median": (float(np.median([r["approach_steps"] for r in seen])) if seen else None),
           "success_given_seen": (float(np.mean([r["success"] for r in seen])) if seen else None),
           "collisions_mean": float(np.mean([r["collisions"] for r in flat])) if flat else float("nan"),
           "stuck_run_share": float(np.mean([r["stuck_streaks"] > 0 for r in flat])) if flat else float("nan"),
           "per_category": {}}
    for c in sorted(set(cats.values())):
        idx = [i for i, e in enumerate(episodes) if cats[e] == c]
        row["per_category"][c] = {"episodes": len(idx), "success": float(succ[idx].mean())}
    return row


def judged(layout: RunLayout, arms: Sequence[str], episodes: Sequence[str], seeds: Dict[str, List[int]]):
    """(runs per arm, kept episodes, crash count, total runs, missing runs) for the judged arms."""
    runs = {a: read_runs(layout, a) for a in arms}
    missing = [f"{a}/{e}__s{k}" for a in arms for e in episodes for k in seeds[a] if k not in runs[a].get(e, {})]
    crashed = {e for a in arms for e in episodes for r in runs[a].get(e, {}).values() if r.get("crash")}
    n_crash = sum(1 for a in arms for e in episodes for r in runs[a].get(e, {}).values() if r.get("crash"))
    total = sum(len(seeds[a]) for a in arms) * len(episodes)
    kept = [e for e in episodes if e not in crashed]
    return runs, kept, n_crash, total, missing


def table_md(title: str, rows: Dict[str, Dict[str, Any]]) -> List[str]:
    cats = sorted({c for r in rows.values() for c in r["per_category"]})
    lines = [f"## {title}", "", "| arm | episodes | runs | success | SPL | seen share | success if seen | search steps "
             "(median) | approach steps (median) | collisions (mean) | runs with a stuck streak |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    f = (lambda x: "-" if x is None else f"{x:.3f}")
    for a, r in rows.items():
        lines.append(f"| {a} | {r['episodes']} | {r['runs']} | {r['success']:.3f} | {r['spl']:.3f} | {f(r['seen_share'])} | "
                     f"{f(r['success_given_seen'])} | {r['search_steps_median']:.0f} | {f(r['approach_steps_median'])} | "
                     f"{r['collisions_mean']:.1f} | {r['stuck_run_share']:.3f} |")
    lines += ["", "| arm | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for a, r in rows.items():
        lines.append(f"| {a} | " + " | ".join(f"{r['per_category'][c]['success']:.3f} ({r['per_category'][c]['episodes']})"
                                               for c in cats) + " |")
    return lines


def seeds_of(layout: RunLayout, arm: str) -> List[int]:
    own = layout.cfg["arms"][arm].get("seeds")
    return [k for k in range(int(layout.cfg["run"]["seeds"])) if own is None or k in own]


def cmd_pick(args) -> None:
    layout = RunLayout.load(args.config, subdir=args.subdir)
    eps, fp = load_episodes(layout.base / layout.cfg["sets"]["dev"]["folder"] / "episodes.json")
    ids = [e["episode_id"] for e in eps]
    cats = {e["episode_id"]: e["target"]["category"] for e in eps}
    seeds = {a: seeds_of(layout, a) for a in ("run1", "runC", "run1_hidden")}
    runs, kept, n_crash, total, missing = judged(layout, ["run1", "runC"], ids, seeds)
    stop = None
    if missing:
        stop = f"step 0: {len(missing)} judged runs missing (e.g. {missing[:3]})"
    elif n_crash > MAX_CRASH_RATE * total:
        stop = f"step 0: crash rate {n_crash}/{total} > 1%"
    if stop:
        write_text_atomic(layout.root / "STOP_pick.txt", stop + "\n")
        print(stop, flush=True)
        raise SystemExit(3)
    s1, sc = per_episode(runs["run1"], kept, "success"), per_episode(runs["runC"], kept, "success")
    delta, winner = pick_rule(s1, sc)
    hidden = read_runs(layout, "run1_hidden")
    rows = {"run1": arm_table(runs["run1"], kept, cats), "runC": arm_table(runs["runC"], kept, cats)}
    hid_eps = [e for e in kept if all(k in hidden.get(e, {}) and not hidden[e][k].get("crash") for k in seeds["run1_hidden"])]
    if hid_eps:
        rows["run1_hidden (report only, seed 0)"] = arm_table({e: hidden[e] for e in hid_eps}, hid_eps, cats)
    ck = layout.cfg["checkpoints"][winner]
    final = {"arm": winner, "path": ck["path"], "sha256": ck["sha256"], "config": ck["config"], "delta": delta,
             "margin": MARGIN, "rule": "run C becomes v2 only if mean over dev episodes of (C - run 1) per-episode "
             "success (mean of 3 seeds) >= +0.03; otherwise run 1 (tie -> run 1)",
             "episodes": len(kept), "dropped_crash_episodes": len(ids) - len(kept), "dev_set_fingerprint": fp}
    decision = {**final, "success_run1": float(s1.mean()), "success_runC": float(sc.mean()),
                "per_episode_delta_sd": float((sc - s1).std(ddof=1)), "crashes": n_crash, "judged_runs": total,
                "tables": rows}
    write_text_atomic(layout.root / "decision.json", json.dumps(decision, indent=1) + "\n")
    lines = ["# Phase 11 step 0: v2 pick (G11 row 0)", "",
             f"Dev set `{fp[:12]}…`: {len(kept)} episodes x 3 seeds per arm ({len(ids) - len(kept)} dropped for a crash; "
             f"{n_crash} crashed runs of {total}). Rule (p11_rules.md): {final['rule']}.", "",
             f"**Delta = mean(C - run 1) = {delta:+.4f}** (run 1 {s1.mean():.4f}, run C {sc.mean():.4f}) -> "
             f"**v2 = {winner}** (`{ck['path']}`, sha256 `{ck['sha256']}`).", ""]
    lines += table_md("Arms (oracle stop; search / approach split by simulator truth)", rows)
    write_text_atomic(layout.root / "decision.md", "\n".join(lines) + "\n")
    write_text_atomic(layout.base / "V2_FINAL.json", json.dumps(final, indent=1) + "\n")
    write_text_atomic(layout.base / "V2_FINAL.txt", f"v2 = {winner} ema_19 (path {ck['path']}, sha256 {ck['sha256']}, "
                      f"config {ck['config']}); Delta = mean(C - run 1) = {delta:+.4f}; rule: {final['rule']} "
                      f"(p11_rules.md; Phase 11 step 0)\n")
    print("\n".join(lines), flush=True)


def cmd_compare(args) -> None:
    layout = RunLayout.load(args.config, subdir=args.subdir)
    eps, fp = load_episodes(layout.base / layout.cfg["sets"]["dev"]["folder"] / "episodes.json")
    ids = [e["episode_id"] for e in eps]
    cats = {e["episode_id"]: e["target"]["category"] for e in eps}
    specs = {}  # label -> (layout, arm): "<label>=<subdir>/<arm>" reads another run folder (e.g. v2=step0/run1)
    for spec in args.arms:
        label, _, where = spec.partition("=")
        sub, _, arm = where.rpartition("/") if where else (args.subdir, "", label)
        specs[label] = (RunLayout.load(args.config, subdir=sub), arm)
    seeds = {lab: seeds_of(lay, arm) for lab, (lay, arm) in specs.items()}
    runs, crashed_eps, n_crash, total, missing = {}, set(), 0, 0, []
    for lab, (lay, arm) in specs.items():
        r, kept_l, c, t, m = judged(lay, [arm], ids, {arm: seeds[lab]})
        runs[lab] = r[arm]
        crashed_eps |= set(ids) - set(kept_l)
        n_crash, total, missing = n_crash + c, total + t, missing + m
    kept = [e for e in ids if e not in crashed_eps]
    args.arms = list(specs)
    rows = {a: arm_table(runs[a], kept, cats) for a in args.arms}
    out = {"dev_set_fingerprint": fp, "episodes": len(kept), "crashes": n_crash, "runs": total,
           "missing": len(missing), "tables": rows}
    if len(args.arms) == 2:
        a, b = (per_episode(runs[x], kept, "success") for x in args.arms)
        d = b - a
        rng = np.random.default_rng(11501)
        boot = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(10000)]
        out["difference"] = {"arms": f"{args.arms[1]} - {args.arms[0]}", "mean": float(d.mean()),
                             "ci95_episode_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    base = layout.base / args.out
    write_text_atomic(base.with_suffix(".json"), json.dumps(out, indent=1) + "\n")
    lines = [f"# Phase 11 dev check: {' vs '.join(args.arms)} (report only)", "",
             f"Dev set `{fp[:12]}…`: {len(kept)} episodes; {n_crash} crashed runs of {total}; {len(missing)} missing.", ""]
    if "difference" in out:
        dd = out["difference"]
        lines += [f"Success {dd['arms']}: {dd['mean']:+.4f} [{dd['ci95_episode_bootstrap'][0]:+.4f}, "
                  f"{dd['ci95_episode_bootstrap'][1]:+.4f}] (episode bootstrap, 10,000, seed 11501).", ""]
    lines += table_md("Arms (oracle stop)", rows)
    write_text_atomic(base.with_suffix(".md"), "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pick")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--subdir", default="step0")
    q = sub.add_parser("compare")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--subdir", required=True)
    q.add_argument("--arms", nargs="+", required=True)
    q.add_argument("--out", required=True, help="below <out_dir>, without suffix")
    args = p.parse_args(argv)
    {"pick": cmd_pick, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
