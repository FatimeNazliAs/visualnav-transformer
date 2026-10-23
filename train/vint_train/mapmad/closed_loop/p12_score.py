"""Phase 12 scoring from the step logs (block items 1, 4, 7, 8; p12_rules.md). numpy only, both containers.

    python -m vint_train.mapmad.closed_loop.p12_score dev-choices --config <p12_exam.yaml>       # P12_DEV_CHOICES.json
    python -m vint_train.mapmad.closed_loop.p12_score score --config <p12_exam.yaml> --subdir runs/exam --set exam \
        --arms M2 M2h ...                                                     # -> <subdir>/scores/<arm>.jsonl
    python -m vint_train.mapmad.closed_loop.p12_score same-as-p11 --config <p12_exam.yaml> --p12 dev/hidden/logs/v2h_dev \
        --p11 ../p11_dagger/step0/logs/run1_hidden --seed 0                   # the refactor check of block item 1

One run = one log of max_steps steps (no stop); two scores (block item 4):
- oracle stop: success = the first step whose walking distance to the nearest usable view point of any instance is
  <= success_m (0.2 m) (`success_step`; step 0 counts); SPL with the path driven up to it;
- rule stop at tau: the first step t >= 1 with rule_value >= tau (the largest target value among cells <= 1.0 m ahead,
  +-30 deg, seen from >= 2 views; logged by the server); success if that step is inside the zone; SPL with the path up
  to t; no step -> no stop, a failure. Wrong-object stop: the triggering cell is > 1.0 m (horizontally) from the box of
  every usable instance of the category; near-miss: a stop outside the zone with walking distance <= success_m + 1.0 m.
  STOP precision = rule-stop successes / rule stops.
- search phase: first sight = the first step (0 = start) where simulator truth sees an instance of the category
  (>= 200 px at <= 4 m; `in_view`); found by 1,000 = a first sight exists. Approach phase: success given seen;
  steps from first sight to the stop (oracle or rule).
- success@500 (both stops); collisions; stuck streaks (>= 100 consecutive collision steps); house seen at 250 / 500 /
  1,000 steps (share of house cells, house_seen.py); redundancy = 1 - distinct 0.5 m cells / 0.5 m cell entries.
tau grid (block item 4): 0.30, 0.35, ..., 0.95; the headline arm + YOLOE on the dev set (3 seeds) is scored at every
tau from the same logs; tau = the highest rule-stop success (mean over episodes of the seed mean), tie -> higher tau.
Headline (block item 1): Delta(c) = success(maps) - success(maps hidden) on the dev set (maps = Phase 11's runs, hidden
= Phase 12's); headline = v2.1 unless Delta(v2.1) < Delta(v2) - 0.02, then v2. Crash records drop the episode from
both arms of a comparison (Phase 5 rule); a crash rate > 1% or a missing run -> exit 3 (automatic stop).
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mapmad_bridge import steplog
from mapmad_sim.run_layout import RunLayout, parse_run_id, write_text_atomic

TAU_GRID = tuple(round(0.30 + 0.05 * k, 2) for k in range(14))  # 0.30 ... 0.95
HEADLINE_MARGIN = 0.02
MAX_CRASH_RATE = 0.01
CHECKPOINTS = ("v2", "v21")


# --- one run ------------------------------------------------------------------------------------------------------
def box_distance_2d(cell: Sequence[float], inst: Dict[str, Any]) -> float:
    """Horizontal distance from a 2D-frame point (X = -z, Y = -x of Habitat) to an instance's Habitat box."""
    cx, _, cz = inst["centre"]
    sx, _, sz = inst["size"]
    x, z = -cell[1], -cell[0]
    return math.hypot(max(abs(x - cx) - sx / 2.0, 0.0), max(abs(z - cz) - sz / 2.0, 0.0))


def first_index(flags: Iterable[bool]) -> Optional[int]:
    for i, f in enumerate(flags):
        if f:
            return i
    return None


def redundancy(positions: Sequence[Sequence[float]], cell_m: float) -> float:
    """1 - distinct cells / cell entries, over the 0.5 m cells (Habitat x, z) the robot passes."""
    cells = [(math.floor(p[0] / cell_m), math.floor(p[2] / cell_m)) for p in positions]
    entries = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
    return 1.0 - len(set(entries)) / len(entries) if entries else 0.0


def spl(success: bool, shortest: Optional[float], path: float) -> float:
    if not success or shortest is None:
        return 0.0
    return shortest / max(shortest, path) if shortest > 0 else 1.0


def score_log(log: List[Dict[str, Any]], taus: Sequence[float] = TAU_GRID, success_m: float = 0.2,
              near_m: float = 1.0, wrong_m: float = 1.0, revisit_m: float = 0.5,
              marks: Sequence[int] = (250, 500, 1000), at: int = 500) -> Dict[str, Any]:
    """Every Phase 12 number of one run (module docstring)."""
    head, steps = log[0], steps_of(log)
    vp = [head["start_vp_geodesic_m"]] + [s["vp_geodesic_m"] for s in steps]
    path = np.concatenate([[0.0], np.cumsum([s["travelled_m"] for s in steps])])
    inzone = [v is not None and v <= success_m for v in vp]
    shortest = head["start_vp_geodesic_m"]
    o = first_index(inzone)
    seen = first_index([bool(head["start"].get("in_view"))] + [bool(s["in_view"]) for s in steps])
    out: Dict[str, Any] = {
        "episode_id": head["episode_id"], "arm": head["arm"], "seed_index": head["seed_index"], "home": head["home"],
        "category": head["category"], "distance_bin": head["distance_bin"], "steps": len(steps),
        "oracle_success": o is not None, "oracle_step": o, "oracle_spl": spl(o is not None, shortest, path[o] if o is not None else 0.0),
        f"oracle_success_at_{at}": o is not None and o <= at,
        "first_sight": seen, "seen": seen is not None,
        "oracle_approach_steps": (o - seen) if (o is not None and seen is not None and o >= seen) else None,
        "collisions": int(sum(bool(s["collided"]) for s in steps)),
        "stuck_streaks": len(steplog.stuck_streaks(log, int(head.get("stuck_steps", 100)))),
        "redundancy": redundancy([head["start"]["position"]] + [s["position"] for s in steps], revisit_m)}
    if steps and "rule_value" in steps[0]:
        rv = [s["rule_value"] or 0.0 for s in steps]
        rule: Dict[str, Any] = {}
        for tau in taus:
            k = first_index(v >= tau - 1e-12 for v in rv)
            t = None if k is None else k + 1  # step numbers start at 1
            r: Dict[str, Any] = {"step": t, "stopped": t is not None}
            if t is not None:
                ok = inzone[t]
                cell = steps[k]["rule_cell"]
                insts = head.get("instances") or []
                r.update(success=bool(ok), spl=spl(ok, shortest, path[t]), success_at=bool(ok and t <= at),
                         wrong_object=bool(insts) and cell is not None and min(box_distance_2d(cell, i) for i in insts) > wrong_m,
                         near_miss=(not ok) and vp[t] is not None and vp[t] <= success_m + near_m,
                         approach_steps=(t - seen) if (seen is not None and t >= seen) else None)
            else:
                r.update(success=False, spl=0.0, success_at=False, wrong_object=False, near_miss=False,
                         approach_steps=None)
            rule[f"{tau:.2f}"] = r
        out["rule"] = rule
    if steps and "house_seen" in steps[0]:
        cells = max(int(head.get("house_cells") or 0), 1)
        hs = [s["house_seen"] for s in steps]
        out["house_seen"] = {str(m): hs[min(m, len(hs)) - 1] / cells for m in marks}
        out["house_cells"] = cells
    return out


def steps_of(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [x for x in log if x["kind"] == "step"]


# --- sets of runs -------------------------------------------------------------------------------------------------
def read_summaries(root: Path, arm: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """episode id -> seed index -> summary (crash records as {"crash": True})."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for p in sorted((root / "summaries" / arm).glob("*.json")):
        eid, k = parse_run_id(p.name)
        out.setdefault(eid, {})[k] = {"crash": True} if p.name.endswith(".crash.json") else json.loads(p.read_text())
    return out


def episode_success(runs: Dict[str, Dict[int, Dict[str, Any]]], ids: Sequence[str], seeds: Sequence[int],
                    key: str = "success") -> Tuple[Dict[str, float], List[str], int, int]:
    """(episode -> mean over seeds of summary[key], missing runs, crashes, judged runs)."""
    vals, missing, crashes = {}, [], 0
    for e in ids:
        xs = []
        for k in seeds:
            r = runs.get(e, {}).get(k)
            if r is None:
                missing.append(f"{e}__s{k}")
            elif r.get("crash"):
                crashes += 1
            else:
                xs.append(float(bool(r[key])))
        if len(xs) == len(seeds):
            vals[e] = float(np.mean(xs))
    return vals, missing, crashes, len(ids) * len(seeds)


def headline_rule(delta: Dict[str, float], margin: float = HEADLINE_MARGIN) -> str:
    """Block item 1: v2.1 unless Delta(v2.1) < Delta(v2) - margin, then v2."""
    return "v2" if delta["v21"] < delta["v2"] - margin - 1e-12 else "v21"


def tau_rule(success_by_tau: Dict[float, float]) -> float:
    """Block item 4: the tau with the highest rule-stop success; a tie -> the higher tau."""
    best = max(success_by_tau.values())
    return max(t for t, v in success_by_tau.items() if v >= best - 1e-12)


def cmd_dev_choices(args) -> None:
    layout = RunLayout.load(args.config)
    base, dc = layout.base, layout.cfg["dev_choice"]
    from mapmad_sim.episodes import load_episodes

    eps, fp = load_episodes(base / layout.cfg["sets"]["dev"]["folder"] / "episodes.json")
    ids = [e["episode_id"] for e in eps]
    seeds = list(range(int(layout.cfg["run"]["seeds"])))
    stop, rows, delta = [], {}, {}
    for c in CHECKPOINTS:
        maps_root, maps_arm = base / dc["maps"][c]["subdir"], dc["maps"][c]["arm"]
        hid_root, hid_arm = base / dc["hidden"][c]["subdir"], dc["hidden"][c]["arm"]
        m, mm, mc, mt = episode_success(read_summaries(maps_root, maps_arm), ids, seeds)
        h, hm, hc, ht = episode_success(read_summaries(hid_root, hid_arm), ids, seeds)
        if mm or hm:
            stop.append(f"{c}: {len(mm) + len(hm)} runs missing (e.g. {(mm + hm)[:3]})")
        if mc > MAX_CRASH_RATE * mt or hc > MAX_CRASH_RATE * ht:
            stop.append(f"{c}: crash rate > 1% (maps {mc}/{mt}, hidden {hc}/{ht})")
        kept = [e for e in ids if e in m and e in h]
        delta[c] = float(np.mean([m[e] - h[e] for e in kept])) if kept else float("nan")
        rows[c] = {"maps": float(np.mean([m[e] for e in kept])), "hidden": float(np.mean([h[e] for e in kept])),
                   "delta": delta[c], "episodes": len(kept), "crashes": mc + hc,
                   "maps_runs": f"{dc['maps'][c]['subdir']}/{maps_arm}", "hidden_runs": f"{dc['hidden'][c]['subdir']}/{hid_arm}"}
    if stop:
        write_text_atomic(base / "STOP_dev.txt", "; ".join(stop) + "\n")
        print("; ".join(stop), flush=True)
        raise SystemExit(3)
    headline = headline_rule(delta)
    both_le0 = delta["v2"] <= 0 and delta["v21"] <= 0
    out = {"headline": headline, "deltas": delta, "rows": rows, "both_delta_le_0": both_le0,
           "headline_rule": "headline = v21 unless Delta(v21) < Delta(v2) - 0.02, then v2 (Delta = success(maps) - "
                            "success(maps hidden), dev set, oracle stop, 3 seeds)", "dev_set_fingerprint": fp}
    if args.headline_only:
        write_text_atomic(base / "P12_HEADLINE.json", json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
        return
    # tau from the headline arm + YOLOE dev runs (scores written by `score`)
    rows_t = read_scores(base / dc["tau"]["subdir"], dc["tau"]["arm"])
    by_ep: Dict[str, Dict[float, List[float]]] = {}
    for r in rows_t:
        for t in TAU_GRID:
            by_ep.setdefault(r["episode_id"], {}).setdefault(t, []).append(float(r["rule"][f"{t:.2f}"]["success"]))
    crashes = sum(1 for _ in (base / dc["tau"]["subdir"] / "summaries" / dc["tau"]["arm"]).glob("*.crash.json"))
    full = [e for e in ids if e in by_ep and all(len(by_ep[e][t]) == len(seeds) for t in TAU_GRID)]
    if len(full) < len(ids) - crashes or crashes > MAX_CRASH_RATE * len(ids) * len(seeds):
        msg = f"tau runs: {len(full)} complete episodes of {len(ids)} ({crashes} crashes)"
        write_text_atomic(base / "STOP_dev.txt", msg + "\n")
        raise SystemExit(3)
    succ = {t: float(np.mean([np.mean(by_ep[e][t]) for e in full])) for t in TAU_GRID}
    tau = tau_rule(succ)
    out.update(tau=tau, rule_stop_success_by_tau={f"{t:.2f}": v for t, v in succ.items()},
               tau_rule="the tau of 0.30-0.95 (step 0.05) with the highest rule-stop success of the headline arm + "
                        "YOLOE on the dev set (3 seeds, one set of runs scored at every tau), tie -> higher tau",
               tau_runs=f"{dc['tau']['subdir']}/{dc['tau']['arm']}", tau_episodes=len(full))
    write_text_atomic(base / "P12_DEV_CHOICES.json", json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def read_scores(root: Path, arm: str) -> List[Dict[str, Any]]:
    f = root / "scores" / f"{arm}.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def instances_of(layout: RunLayout, set_name: str) -> Dict[str, List[Dict[str, Any]]]:
    from mapmad_sim.episodes import load_episodes

    eps, _ = load_episodes(layout.base / layout.cfg["sets"][set_name]["folder"] / "episodes.json")
    return {e["episode_id"]: e["target"]["instances"] for e in eps}


def cmd_score(args) -> None:
    layout = RunLayout.load(args.config, subdir=args.subdir)
    run = layout.cfg["run"]
    insts = instances_of(layout, args.set)
    for arm in args.arms:
        rows = []
        for p in layout.logs(arm):
            log = steplog.read(p)
            log[0] = dict(log[0], instances=insts.get(log[0]["episode_id"], []), stuck_steps=run["stuck_steps"])
            rows.append(score_log(log, success_m=run["success_m"], revisit_m=run["revisit_cell_m"],
                                  at=int(run["success_at"][0])))
        out = layout.root / "scores" / f"{arm}.jsonl"
        write_text_atomic(out, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        print(f"{arm}: {len(rows)} runs -> {out}", flush=True)


def restrict(a: Any, b: Any) -> Any:
    """a limited to b's keys, recursively in dicts (fields Phase 12 added are left out)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: restrict(a.get(k), b[k]) for k in b}
    return a


def cmd_same_as_p11(args) -> None:
    """Block item 1: seed `seed` of the Phase 12 maps-hidden v2 dev runs = Phase 11's run1_hidden logs on every step
    and field Phase 11 logged (the Phase 11 run ended at success: its log is a prefix; header field `arm` excluded)."""
    layout = RunLayout.load(args.config)
    a_dir, b_dir = layout.base / args.p12, layout.base / args.p11
    bad, n = [], 0
    for fa in sorted(a_dir.glob(f"*__s{args.seed}.jsonl")):  # every Phase 12 log of that seed needs its Phase 11 twin
        fb = b_dir / fa.name
        n += 1
        if not fb.exists():
            bad.append(f"{fa.name}: no Phase 11 log")
            continue
        la, lb = steplog.read(fa), steplog.read(fb)
        skip = {"arm", *args.ignore}
        hb = {k: v for k, v in lb[0].items() if k not in skip}
        ha = restrict(la[0], hb)
        if ha != hb:
            bad.append(f"{fb.name}: header differs in {[k for k in hb if ha.get(k) != hb[k]]}")
            continue
        sa, sb = steps_of(la), steps_of(lb)  # compared over the common prefix (the exam's logs have all 1,000 steps)
        diff = next((i for i, s in enumerate(sb[:len(sa)]) if restrict(sa[i], s) != s), None)
        if diff is not None:
            bad.append(f"{fb.name}: step {diff + 1} differs")
    res = {"compared": n, "pass": n > 0 and not bad, "differ": bad, "p12": str(a_dir), "p11": str(b_dir)}
    write_text_atomic(Path(args.out), json.dumps(res, indent=1) + "\n")
    print(json.dumps({"compared": n, "pass": res["pass"], "differ": bad[:5]}), flush=True)
    raise SystemExit(0 if res["pass"] else 3)


def cmd_fairness(args) -> None:
    """G12 row 3: every (arm, episode, seed) of the given run groups has a log; every log's header carries the frozen
    set fingerprint and its arm's checkpoint sha256 (arm table). groups: "<subdir>:<set>:<arm>,<arm>..."."""
    from mapmad_sim.arm_table import ArmTable
    from mapmad_sim.episodes import load_episodes

    layout = RunLayout.load(args.config)
    table = ArmTable.from_layout(layout)
    run_seeds = list(range(int(layout.cfg["run"]["seeds"])))
    problems, counts = [], {}
    for g in args.groups:
        subdir, set_name, arms = g.split(":")
        eps, fp = load_episodes(layout.base / layout.cfg["sets"][set_name]["folder"] / "episodes.json")
        for arm in arms.split(","):
            a = table.arm(arm)
            sha = table.checkpoint(a.checkpoint).sha256 if a.checkpoint else None
            n = 0
            for e in eps:
                for k in table.seeds(arm, run_seeds):
                    f = layout.base / subdir / "logs" / arm / f"{e['episode_id']}__s{k}.jsonl"
                    crash = layout.base / subdir / "summaries" / arm / f"{e['episode_id']}__s{k}.crash.json"
                    if not f.exists():
                        problems.append(f"{subdir}/{arm}/{e['episode_id']}__s{k}: " + ("crash record" if crash.exists() else "missing"))
                        continue
                    with open(f) as fh:
                        h = json.loads(fh.readline())
                    if h["episodes_fingerprint"] != fp or (sha is not None and h.get("weights_sha256") != sha):
                        problems.append(f"{subdir}/{arm}/{f.name}: fingerprint or sha256 differs")
                    n += 1
            counts[f"{subdir}/{arm}"] = n
    res = {"pass": not problems, "runs": counts, "problems": problems[:200], "n_problems": len(problems),
           "summary": f"{sum(counts.values())} runs in {len(counts)} arm groups; {len(problems)} problems"}
    write_text_atomic(layout.base / "checks" / "fairness.json", json.dumps(res, indent=1) + "\n")
    print(res["summary"], flush=True)


def cmd_live_check(args) -> None:
    """Rule stop scored from the full log = the rule stop run live (same episodes, seed, GPU; block item 4 check)."""
    layout = RunLayout.load(args.config)
    tau = f"{args.tau:.2f}"
    rows, ok = [], True
    for f in sorted((layout.base / args.full / "logs" / args.arm).glob("*.jsonl")):
        log = steplog.read(f)
        full = score_log(log, taus=(args.tau,), success_m=layout.cfg["run"]["success_m"])["rule"][tau]
        live = json.loads((layout.base / args.live / "summaries" / args.arm / f"{f.stem}.json").read_text())
        same = live["rule_stop_step"] == full["step"] and bool(live["rule_stop_success"]) == bool(full["success"])
        ok &= same
        rows.append({"run": f.stem, "log_step": full["step"], "live_step": live["rule_stop_step"],
                     "log_success": full["success"], "live_success": live["rule_stop_success"], "same": same})
    res = {"pass": ok and len(rows) > 0, "tau": args.tau, "runs": rows}
    write_text_atomic(Path(args.out), json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)
    raise SystemExit(0 if res["pass"] else 3)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("dev-choices")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--headline-only", action="store_true", help="only the headline rule (before the tau runs)")
    q = sub.add_parser("score")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--subdir", required=True)
    q.add_argument("--set", required=True)
    q.add_argument("--arms", nargs="+", required=True)
    q = sub.add_parser("same-as-p11")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--p12", required=True, help="Phase 12 log folder below out_dir")
    q.add_argument("--p11", required=True, help="Phase 11 log folder below out_dir")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--ignore", nargs="*", default=[], help="DRY runs only: header fields that differ by design")
    q.add_argument("--out", required=True)
    q = sub.add_parser("fairness")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--groups", nargs="+", required=True, help="<subdir>:<set>:<arm>,<arm>...")
    q = sub.add_parser("live-check")
    q.add_argument("--config", type=Path, required=True)
    q.add_argument("--full", required=True, help="subdir of the full (no-stop) runs")
    q.add_argument("--live", required=True, help="subdir of the live rule-stop runs")
    q.add_argument("--arm", required=True)
    q.add_argument("--tau", type=float, required=True)
    q.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"dev-choices": cmd_dev_choices, "score": cmd_score, "same-as-p11": cmd_same_as_p11,
     "fairness": cmd_fairness, "live-check": cmd_live_check}[args.cmd](args)


if __name__ == "__main__":
    main()
