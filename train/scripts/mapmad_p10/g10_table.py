#!/usr/bin/env python3
"""Draft gate G10 table (chain step 5; p10_rules.md §5). Host side, standard library only.

    python3 g10_table.py <p10 dir> <label: main|twin>

Reads checks/g10_rows_1_2_gpu.json (rows 1-2, Part 1), offline/<label>_ema19/search/compare.json (row 3 = maps -
zeroed, rules v2), offline/<label>_ema19/gostanford/gostanford.json (row 4) and offline/<label>_ema19/ema_19.sha256
(row 5); writes offline/<label>_ema19/g10_table_draft.{md,json}. A missing input leaves its row "pending". Report-only lines: the
exploration list, maps - hidden, the pre-training (step-0) numbers, attention per map token.
"""

import json
import os
import sys


def load(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt_ci(r) -> str:
    return f"{r['diff']:+.3f} [{r['diff_ci_drive'][0]:+.3f}, {r['diff_ci_drive'][1]:+.3f}]"


def main() -> None:
    p10, label = sys.argv[1], sys.argv[2]
    off = os.path.join(p10, "offline", f"{label}_ema19")
    os.makedirs(off, exist_ok=True)
    g12 = load(os.path.join(p10, "checks", "g10_rows_1_2_gpu.json"))
    search = load(os.path.join(off, "search", "compare.json"))
    explore = load(os.path.join(off, "explore", "compare.json"))
    gs = load(os.path.join(off, "gostanford", "gostanford.json"))
    pre = load(os.path.join(p10, "offline", "pretrain", "search", "compare.json"))
    sha_path = os.path.join(off, "ema_19.sha256")
    words = open(sha_path).read().split() if os.path.exists(sha_path) else []
    sha = words[0] if words else None
    rows = []
    if g12:
        r1, r2 = g12["row1"], g12["row2"]
        rows.append((1, "Switch off", "NoMaD forward byte-identical", "all outputs torch.equal" if r1["pass"] else "differs",
                     r1["pass"]))
        rows.append((2, "Maps hidden", "allclose 1e-6; 7 tokens; all keys load",
                     f"max diff {r2['max_abs_diff']:.1e}; {r2['masks']['new_tokens']} tokens; "
                     f"{len(r2['load_report']['new_keys'])} new keys", r2["pass"]))
    rule3 = "drive CI of (maps - zeroed) < 0"  # p10_rules.md v2 (maps - hidden is report only)
    if search and search["row3"].get("control") == "zeroed":
        r = search["row3"]
        rows.append((3, "Offline map effect", rule3,
                     f"maps {r['mean_a']:.3f} vs zeroed {r['mean_b']:.3f}: {fmt_ci(r)}; R {r['R']:.3f}", r["pass"]))
    else:
        rows.append((3, "Offline map effect", rule3, "zeroed arm missing" if search else "pending",
                     False if search else None))
    if gs:
        rows.append((4, "GoStanford no-harm", "<= official + 0.10",
                     f"official {gs['official']:.3f} -> v2 {gs['mapmad']:.3f}, {gs['diff']:+.3f} "
                     f"[{gs['diff_ci_sample'][0]:+.3f}, {gs['diff_ci_sample'][1]:+.3f}]", gs["pass"]))
    else:
        rows.append((4, "GoStanford no-harm", "<= official + 0.10", "pending", None))
    rows.append((5, "Runs", "ema_19 seed 1 fixed in advance; twin report only",
                 f"ema_19 sha256 {sha[:16]}..." if sha else "pending", bool(sha) if label == "main" else None))
    md = [f"# G10 draft ({label} ema_19)", "", "| # | Check | Pass rule | Result | Pass |", "|---|---|---|---|---|"]
    for n, name, rule, res, ok in rows:
        md.append(f"| {n} | {name} | {rule} | {res} | {'✅' if ok else ('❌' if ok is False else '…')} |")
    md.append("")
    if search:
        md.append("Report only (search list): means " + json.dumps({k: round(v, 3) for k, v in search["means"].items()}))
        md.append("Attention share per map token: " + json.dumps(search["attention_share"]))
        md.append("Per category (maps - hidden): " + json.dumps(
            {c: round(v["diff"], 3) for c, v in search["per_category"].items() if "diff" in v}))
        md.append("Forward/turn (maps - hidden): " + json.dumps(
            {c: {"diff": round(v["diff"], 3), "share": round(v["share"], 3)} for c, v in search["forward_turn"].items()
             if "diff" in v}))
    if explore:
        md.append(f"Exploration list: explore_maps - hidden {fmt_ci(explore['row3'])}")
    if search and "maps_vs_hidden" in search:
        md.append(f"Report only (rules v1 row 3): maps - hidden {fmt_ci(search['maps_vs_hidden'])}; "
                  f"R {search['maps_vs_hidden']['R']:.3f}")
    if pre:
        mvh = pre.get("maps_vs_hidden", pre["row3"])
        md.append(f"Step-0 baseline (untrained v2, zero-init map encoders): maps - hidden {fmt_ci(mvh)} (two extra "
                  f"tokens + mean pooling, no map content used); attention {json.dumps(pre['attention_share'])}")
        if pre["row3"].get("control") == "zeroed":
            md.append(f"Step-0 baseline for row 3: maps - zeroed {fmt_ci(pre['row3'])}")
        if search and "maps_vs_hidden" in search:
            md.append(f"maps - hidden beyond step 0: trained {search['maps_vs_hidden']['diff']:+.3f} vs step-0 "
                      f"{mvh['diff']:+.3f}")
    with open(os.path.join(off, "g10_table_draft.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(off, "g10_table_draft.json"), "w") as f:
        json.dump({"rows": rows, "ema_19_sha256": sha}, f, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
