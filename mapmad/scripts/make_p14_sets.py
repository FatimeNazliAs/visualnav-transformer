"""Build and freeze the Phase 14a practice set (configs/p14a_finish.yaml `sets.p14d` + `sets.practice`; kickoff 14.0):
new MP3D train-home episodes `p14d-<home>-<category>-<id>` with Phase 13's dev-set builder rules, then the practice
set = p13d (Phase 13 dev, frozen) ∪ p14d with home-level halves A / B. Inside naz_mapmad_habitat, from
/app/visualnav-transformer:

    python mapmad/scripts/make_p14_sets.py candidates --gpu 0 --worker 0 --workers 6
    python mapmad/scripts/make_p14_sets.py freeze                                     # once, after all workers

Homes = the MP3D train homes of <p13>/mp3d_homes.json with no p13d episode. Per home, `candidates_per_home` official
train episodes of our 6 categories (seeded by candidate_seed + the home's index in the train list), each judged by
make_p13_sets.evaluate_home (Phase 13 exactly: start snap <= 0.2 m, a reachable instance on the start storey, nothing
visible in 8 headings, 3-15 m, Phase 8 start filter). `freeze`: exam_set.draw (seeded, <= per_home per home, bins
3-6 / 6-10 / 10-15 m, round-robin over the categories) -> <p14a>/sets/p14d/episodes.json (+ fingerprint,
build_report.md). Then the halves (practice_split): homes of p13d ∪ p14d in a seeded order, each to the half with
fewer episodes so far; the first order (seed, seed + 1, ...) in which both halves hold every category is kept ->
<p14a>/sets/practice.json (both parts' fingerprints, the practice fingerprint, halves, episode -> half).
Kickoff §3 fallback: fewer than 180 kept -> all kept are used; fewer than 100 -> still p13d + what exists (reported).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Dict, List, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_p13_sets as p13  # noqa: E402

from mapmad_sim import config  # noqa: E402
from mapmad_sim.categories import CATEGORIES  # noqa: E402
from mapmad_sim.episodes import load_episodes  # noqa: E402
from mapmad_sim.exam_set import draw, freeze  # noqa: E402
from mapmad_sim.run_info import save_run_info  # noqa: E402
from mapmad_sim.run_layout import RunLayout, read_config, write_text_atomic  # noqa: E402
from mapmad_sim.search_set import search_set_ok  # noqa: E402

CONFIG = config.CONFIG_DIR / "p14a_finish.yaml"
P13_DIR = "p13_claim"
FALLBACK_MIN = 100  # kickoff §3: below this the A / B split is reported as unbalanced


def p13_set_layout(layout: RunLayout) -> SimpleNamespace:
    """A layout-like object whose `sets.dev` is the p14d block, so make_p13_sets.evaluate_home builds p14d episodes."""
    return SimpleNamespace(cfg=dict(layout.cfg, sets={"dev": layout.cfg["sets"]["p14d"]}), base=layout.base)


def unused_homes(layout: RunLayout) -> List[Tuple[int, str]]:
    """(index in the train list, home) of the MP3D train homes without a p13d episode."""
    train = json.loads((layout.outputs / P13_DIR / "mp3d_homes.json").read_text())["lists"]["train"]
    p13d, _ = load_episodes(set_file(layout, "p13d"))
    used = {e["home"] for e in p13d}
    return [(i, h) for i, h in enumerate(train) if h not in used]


def set_file(layout: RunLayout, name: str) -> Path:
    return layout.base / layout.cfg["sets"][name]["folder"] / "episodes.json"


def cmd_candidates(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    folder = layout.base / layout.cfg["sets"]["p14d"]["folder"]
    mine = unused_homes(layout)[args.worker::args.workers]
    save_run_info(folder / "_runs", {**vars(args), "set": layout.cfg["sets"]["p14d"], "homes": [h for _, h in mine]},
                  int(layout.cfg["sets"]["p14d"]["draw_seed"]), name=f"run_info_candidates_w{args.worker}of{args.workers}.json")
    shim = p13_set_layout(layout)
    for idx, home in mine:
        path = folder / "candidates" / f"{home}.json"
        if path.exists():
            continue
        t0 = time.time()
        out = p13.evaluate_home(shim, dg, home, "dev", args.gpu, idx)
        out["seconds"] = round(time.time() - t0, 1)
        write_text_atomic(path, json.dumps(out, indent=1, default=float) + "\n")
        print(json.dumps({"home": home, "examined": out["examined"], "seconds": out["seconds"],
                          "kept": sum(r.get("episode") is not None for r in out["results"])}), flush=True)


# --- the halves ------------------------------------------------------------------------------------------------------
def practice_split(episodes: Sequence[Dict[str, Any]], seed: int, max_tries: int = 1000) -> Dict[str, Any]:
    """Home-level halves A / B (module docstring): {"A": homes, "B": homes, "order_seed": the seed used}."""
    per_home = Counter(e["home"] for e in episodes)
    cats = {h: {e["target"]["category"] for e in episodes if e["home"] == h} for h in per_home}
    homes = sorted(per_home)
    for k in range(max_tries):
        order = [homes[i] for i in np.random.default_rng(seed + k).permutation(len(homes))]
        halves: Dict[str, List[str]] = {"A": [], "B": []}
        n = {"A": 0, "B": 0}
        for h in order:
            side = "A" if n["A"] <= n["B"] else "B"
            halves[side].append(h)
            n[side] += per_home[h]
        covered = [set().union(*(cats[h] for h in halves[s])) for s in ("A", "B")]
        every = {e["target"]["category"] for e in episodes}
        if all(c == every for c in covered):
            return {"A": sorted(halves["A"]), "B": sorted(halves["B"]), "order_seed": seed + k}
    raise SystemExit(f"no home split with every category in both halves after {max_tries} orders")


def practice_fingerprint(parts: Dict[str, str]) -> str:
    """sha256 over the parts' fingerprints in name order (the practice set is the union of frozen sets)."""
    return hashlib.sha256("\n".join(f"{k} {parts[k]}" for k in sorted(parts)).encode()).hexdigest()


def freeze_practice(layout: RunLayout) -> Dict[str, Any]:
    out = layout.base / layout.cfg["sets"]["practice"]["file"]
    if out.exists():
        raise SystemExit(f"{out} exists: the practice set is frozen")
    pr = layout.cfg["sets"]["practice"]
    eps: List[Dict[str, Any]] = []
    fps: Dict[str, str] = {}
    for name in pr["parts"]:
        e, fp = load_episodes(set_file(layout, name))
        eps += e
        fps[name] = fp
    ids = [e["episode_id"] for e in eps]
    if len(set(ids)) != len(ids):
        write_text_atomic(layout.base / "STOP_reason.txt", "practice set: duplicate episode ids\n")
        raise SystemExit(p13.STOP_EXIT)
    halves = practice_split(eps, int(pr["split_seed"]))
    half_of = {h: s for s in ("A", "B") for h in halves[s]}
    doc = {"fingerprint": practice_fingerprint(fps), "parts": fps, "episodes_total": len(eps),
           "halves": {s: halves[s] for s in ("A", "B")}, "order_seed": halves["order_seed"],
           "half_episodes": dict(Counter(half_of[e["home"]] for e in eps)),
           "episode_half": {e["episode_id"]: half_of[e["home"]] for e in sorted(eps, key=lambda e: e["episode_id"])},
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_text_atomic(out, json.dumps(doc, indent=1) + "\n")
    (out.with_suffix(".sha256")).write_text(f"{hashlib.sha256(out.read_bytes()).hexdigest()}  {out.name}\n"
                                                f"fingerprint {doc['fingerprint']}\n")
    return doc


def cmd_freeze(args, layout: RunLayout, dg: Dict[str, Any]) -> None:
    s = layout.cfg["sets"]["p14d"]
    homes = [h for _, h in unused_homes(layout)]
    folder = layout.base / s["folder"]
    files = [folder / "candidates" / f"{h}.json" for h in homes]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"p14d candidates missing for {len(missing)} homes: {missing[:5]}")
    docs = [json.loads(f.read_text()) for f in files]
    results = [dict(r, home=d["home"], kind=d["kind"]) for d in docs for r in d["results"]]
    kept = [r["episode"] for r in results if r.get("episode") is not None]
    bins = layout.cfg["set_rules"]["bins_m"]
    chosen, rep = draw(kept, int(s["total"]), int(s["per_home"]), len(bins) - 1, int(s["draw_seed"]),
                       "round_robin_category")
    bad = search_set_ok(chosen)
    if bad:
        write_text_atomic(layout.base / "STOP_reason.txt", f"p14d set refused: {bad}\n")
        raise SystemExit(p13.STOP_EXIT)
    fp = freeze(folder / "episodes.json", chosen, {"set": "p14d", "rules": layout.cfg["set_rules"], "set_config": s,
                                                   "navmesh_sha256": {d["home"]: d["navmesh_sha256"] for d in docs},
                                                   "draw": rep})
    fallback = ("none" if len(chosen) >= int(s["total"]) else
                f"fewer than {s['total']} kept: all {len(chosen)} used" if len(chosen) >= FALLBACK_MIN else
                f"fewer than {FALLBACK_MIN} ({len(chosen)}): p13d + what exists, A / B split reported as small")
    lines = [f"# Phase 14a p14d set: build report", "",
             f"Fingerprint `{fp}`; {len(chosen)} episodes (asked {s['total']}, <= {s['per_home']} per home, draw seed "
             f"{s['draw_seed']}); candidates examined: {len(results)} in {len(docs)} unused MP3D train homes.", "",
             f"Kickoff §3 fallback: {fallback}.", "", "## Drops (first failing rule)", ""] + p13.drop_table(results)
    lines += ["", "## Distance bins", "", "| bin | kept | chosen |", "|---|---|---|"]
    lines += [f"| {bins[b]:g}-{bins[b + 1]:g} m | {rep['kept_per_bin'][b]} | {rep['chosen_per_bin'][b]} |"
              for b in range(len(bins) - 1)]
    kh = Counter(e["home"] for e in kept)
    lines += ["", "| home | examined | kept | chosen |", "|---|---|---|---|"]
    lines += [f"| {d['home']} | {d['examined']} | {kh[d['home']]} | {rep['chosen_per_home'].get(d['home'], 0)} |"
              for d in docs]
    kc, cc = Counter(e["target"]["category"] for e in kept), Counter(e["target"]["category"] for e in chosen)
    lines += ["", "| category | kept | chosen |", "|---|---|---|"] + [f"| {c} | {kc[c]} | {cc[c]} |" for c in CATEGORIES]
    write_text_atomic(folder / "build_report.md", "\n".join(lines) + "\n")
    doc = freeze_practice(layout)
    print(json.dumps({"p14d": fp, "p14d_episodes": len(chosen), "fallback": fallback, "practice": doc["fingerprint"],
                      "total": doc["episodes_total"], "half_episodes": doc["half_episodes"]}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["candidates", "freeze"])
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    layout = RunLayout.load(args.config)
    dg = read_config(config.CONFIG_DIR / layout.cfg["datagen"])
    (cmd_freeze if args.cmd == "freeze" else cmd_candidates)(args, layout, dg)


if __name__ == "__main__":
    main()
