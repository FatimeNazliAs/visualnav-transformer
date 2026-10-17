"""Phase 7 sweep tables and the mechanical camera choice (p7_rules.md §4-6, §8), from the run logs.

    python -m vint_train.mapmad.closed_loop.p7_tables --config /app/visualnav-transformer/mapmad/configs/p7_sweep.yaml
    python -m vint_train.mapmad.closed_loop.p7_tables --config ... --write-camera   # + mapmad/configs/camera_fair.yaml

Per camera c: main arm B_c (in view, 3 seeds) and secondary arm A_c (out of view, report only). Per episode the metric
is the mean over its seeds; a camera's value = mean over episodes; 95% CIs: episode bootstrap (seed 7700) and home
bootstrap (seed 7701), 10,000 resamples; paired differences vs pin018. Writes <out_dir>/sweep/tables.md, choice.md,
choice.json (+ extra.md when the report-only context-spacing arm has run).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from mapmad_bridge import steplog
from mapmad_sim import config as mapmad_config
from mapmad_sim.camera_setting import cameras_of
from mapmad_sim.episodes import load_episodes
from mapmad_sim.exam_arms import arms_of
from mapmad_sim.run_layout import RunLayout, parse_run_id
from vint_train.mapmad.closed_loop.exam_stats import cluster_bootstrap, episode_bootstrap, paired
from vint_train.mapmad.closed_loop.p7_choice import choose

EPISODE_SEED, HOME_SEED = 7700, 7701
BASELINE = "pin018"
REPO = Path(__file__).resolve().parents[4]


def first_collision_m(log_path: Path) -> Optional[float]:
    """Distance driven before the first collision step (None: the run never collided)."""
    driven = 0.0
    for s in steplog.steps(steplog.read(log_path)):
        if s["collided"]:
            return driven
        driven += s["travelled_m"]
    return None


def run_rows(layout: RunLayout, arm: str) -> List[Dict[str, Any]]:
    """One row per finished run of an arm: the summary plus the distance before the first collision."""
    rows = []
    for f in sorted((layout.root / "summaries" / arm).glob("*.json")):
        if f.name.endswith(".crash.json"):
            continue
        r = json.loads(f.read_text())
        eid, k = parse_run_id(f.name)
        fc = first_collision_m(layout.log(arm, f.stem))
        rows.append({**r, "episode_id": eid, "seed_index": k, "first_collision_m": fc,
                     "target_point": r["target_point_step"] is not None})
    return rows


def per_episode(rows: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    """Mean of a metric over each episode's seeds (None values left out; episodes with none dropped)."""
    acc: Dict[str, List[float]] = {}
    for r in rows:
        if r[key] is not None:
            acc.setdefault(r["episode_id"], []).append(float(r[key]))
    return {e: float(np.mean(v)) for e, v in acc.items()}


def ci(values: Dict[str, float], home_of: Dict[str, str]) -> Dict[str, Any]:
    ids = sorted(values)
    v = np.array([values[i] for i in ids])
    e = episode_bootstrap(v, seed=EPISODE_SEED)
    h = cluster_bootstrap(v, [home_of[i] for i in ids], seed=HOME_SEED)
    return {"mean": e[0], "episode_ci": [e[1], e[2]], "home_ci": [h[1], h[2]], "n": len(ids)}


def fmt_m(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.2f}"


def fmt(c: Dict[str, Any], digits: int = 3) -> str:
    return f"{c['mean']:.{digits}f} [{c['episode_ci'][0]:.{digits}f}, {c['episode_ci'][1]:.{digits}f}]"


def summarise_arm(rows: List[Dict[str, Any]], home_of: Dict[str, str]) -> Dict[str, Any]:
    out = {"runs": len(rows)}
    for key in ("success", "target_point", "spl", "collisions", "stuck_streaks"):
        out[key] = ci(per_episode(rows, key), home_of)
    fc = [r["first_collision_m"] for r in rows]
    out["first_collision_m_median"] = float(np.median([v for v in fc if v is not None])) if any(v is not None for v in fc) else None
    out["runs_without_collision"] = int(sum(v is None for v in fc))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--subdir", default="sweep/runs")
    p.add_argument("--write-camera", action="store_true", help="write mapmad/configs/camera_fair.yaml")
    p.add_argument("--camera-out", type=Path, default=REPO / "mapmad" / "configs" / "camera_fair.yaml",
                   help="where --write-camera writes (dry runs: their own folder)")
    args = p.parse_args()
    layout = RunLayout.load(args.config, subdir=args.subdir)
    cfg = layout.cfg
    episodes, fp = load_episodes(layout.episodes_file)
    home_of = {e["episode_id"]: e["home"] for e in episodes}
    n_iv = sum(e["type"] == "in_view" for e in episodes)
    cams = cameras_of(cfg)
    arms = {a.name: a for a in arms_of(cfg)}
    stats: Dict[str, Dict[str, Any]] = {}
    per_ep_success: Dict[str, Dict[str, float]] = {}
    for c in cams:
        for kind in ("B", "A"):
            rows = run_rows(layout, f"{kind}_{c}")
            if rows:
                stats[f"{kind}_{c}"] = summarise_arm(rows, home_of)
                per_ep_success[f"{kind}_{c}"] = per_episode(rows, "success")
    expected = {f"B_{c}": n_iv * int(cfg["run"]["seeds"]) for c in cams}
    complete = all(stats.get(a, {}).get("runs") == n for a, n in expected.items())
    out = layout.base / "sweep"
    out.mkdir(parents=True, exist_ok=True)

    lines = ["# Phase 7 sweep — official NoMaD per camera", "", f"episodes `{fp}`; main = B (NoMaD + goal photo, "
             f"{n_iv} in-view episodes x 3 seeds); secondary = A (explore, out of view, report only). 95% CI: episode "
             f"bootstrap [home bootstrap in the paired table].", "",
             "| camera | lens | height | B success (primary) | B Phase 1 metric | B SPL | B collisions/run | "
             "B dist. to 1st collision (median m) | B runs w/o collision | A success | A collisions/run |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, cam in cams.items():
        b, a = stats.get(f"B_{c}"), stats.get(f"A_{c}")
        if b is None:
            continue
        lines.append(f"| {c} | {cam.lens} | {cam.height_m:.2f} | {fmt(b['success'])} | {fmt(b['target_point'])} | "
                     f"{fmt(b['spl'])} | {fmt(b['collisions'], 1)} | {fmt_m(b['first_collision_m_median'])} | "
                     f"{b['runs_without_collision']}/{b['runs']} | {fmt(a['success']) if a else '-'} | "
                     f"{fmt(a['collisions'], 1) if a else '-'} |")
    lines += ["", "Phase 1 metric = target point <= 1.0 m reached at any step; runs end at the primary success (oracle stop "
              "at a view point), which can be > 1 m from the Phase 1 target point (as in Phase 5).", "",
              f"## Paired differences vs {BASELINE} (primary success, B)", "",
              "| camera | diff | episode CI | home CI |", "|---|---|---|---|"]
    for c in cams:
        if c == BASELINE or f"B_{c}" not in per_ep_success or f"B_{BASELINE}" not in per_ep_success:
            continue
        d = paired(per_ep_success[f"B_{c}"], per_ep_success[f"B_{BASELINE}"], home_of, seed=EPISODE_SEED)
        lines.append(f"| {c} | {d['diff']:+.3f} | [{d['episode_ci'][0]:+.3f}, {d['episode_ci'][1]:+.3f}] | "
                     f"[{d['home_ci'][0]:+.3f}, {d['home_ci'][1]:+.3f}] |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")

    result: Dict[str, Any] = {"episodes_fingerprint": fp, "complete": complete, "stats": stats}
    if complete:
        success = {c: stats[f"B_{c}"]["success"]["mean"] for c in cams}
        coll = {c: stats[f"B_{c}"]["collisions"]["mean"] for c in cams}
        win, steps = choose(success, coll, {c: v.lens for c, v in cams.items()}, {c: v.height_m for c, v in cams.items()})
        result.update({"winner": win, "steps": steps})
        (out / "choice.md").write_text("# Camera choice (p7_rules.md §6, applied mechanically)\n\n" + "\n".join(steps) +
                                       f"\n\n**Winner: {win}**\n")
        if args.write_camera:
            write_camera(cams[win], win, fp, args.camera_out, layout.root.parent)
        extra = extra_report(layout, win, home_of)
        if extra:
            (out / "extra.md").write_text(extra)
    (out / "choice.json").write_text(json.dumps(result, indent=1, default=float) + "\n")
    print("\n".join(lines))
    print(f"\ncomplete: {complete}" + (f", winner: {result['winner']}" if complete else ""))


def extra_report(layout: RunLayout, win: str, home_of: Dict[str, str]) -> str:
    """Report-only extra (§8): the winner with a context frame every 3rd step vs its own seed-index-0 runs."""
    rows = run_rows(layout, f"X_{win}_cs3")
    if not rows:
        return ""
    base = [r for r in run_rows(layout, f"B_{win}") if r["seed_index"] == 0]
    x, b = per_episode(rows, "success"), per_episode(base, "success")
    d = paired(x, b, home_of, seed=EPISODE_SEED)
    flag = "→ open issue for the next phase (> 5 points better)" if d["diff"] > 0.05 else "(not > 5 points better)"
    return (f"# Report-only extra: {win} with a context frame every 3rd step\n\n{d['n']} in-view episodes, seed index 0.\n\n"
            f"- context every step: {np.mean(list(b.values())):.3f}; every 3rd step: {np.mean(list(x.values())):.3f}\n"
            f"- paired difference {d['diff']:+.3f}, episode CI [{d['episode_ci'][0]:+.3f}, {d['episode_ci'][1]:+.3f}], "
            f"home CI [{d['home_ci'][0]:+.3f}, {d['home_ci'][1]:+.3f}] {flag}\n- never changes the choice (p7_rules.md §8)\n")


def write_camera(cam, name: str, fp: str, path: Path, tables_dir: Path) -> None:
    """mapmad/configs/camera_fair.yaml: the camera for all later phases."""
    robot = mapmad_config.robot()
    pinhole_px = [int(robot["sim"]["render_width"]), int(robot["sim"]["render_height"])]
    doc = {"name": name, "lens": cam.lens, "height_m": cam.height_m, "pitch_deg": cam.pitch_deg,
           "forward_of_base_m": float(robot["camera"]["forward_of_base_m"]),
           "hfov_deg": cam.hfov_deg if cam.lens == "pinhole" else None,
           "resolution": ([cam.double_sphere.width, cam.double_sphere.height] if cam.double_sphere else pinhole_px),
           "double_sphere": cam.double_sphere.to_config() if cam.double_sphere else None}
    head = ("# Phase 7 result: the fair camera = where the original NoMaD drives best (p7_rules.md §6, applied\n"
            f"# mechanically to the sweep on episodes {fp[:8]}…; tables in {tables_dir}/).\n"
            "# Camera for all later phases (LIMO's camera only when the model is final). Load with\n"
            "# mapmad_sim.camera_setting.CameraSetting.from_config(name, this dict).\n")
    path.write_text(head + yaml.safe_dump(doc, sort_keys=False))


if __name__ == "__main__":
    main()
