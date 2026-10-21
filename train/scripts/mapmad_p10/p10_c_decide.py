#!/usr/bin/env python3
"""Run C decision (p10_rules_v4.md §3-4). Host side, standard library only.

    python3 p10_c_decide.py <p10 dir>

Reads offline/c_ema19/{g10_rows_1_2_cpu.json, search/compare.json, gostanford/gostanford.json, seen/seen_report.json,
guard.json}; writes offline/c_ema19/decision.{md,json}; the last printed line is the verdict:
C_IS_V2 | RUN1_IS_V2 | STOP. A missing input counts as a failed check.
"""

import json
import os
import sys
from typing import Any, Dict, Optional


def load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def checks(off: str) -> Dict[str, Dict[str, Any]]:
    """Each check -> {"pass": bool, "result": text}."""
    g12 = load(os.path.join(off, "g10_rows_1_2_cpu.json"))
    search = load(os.path.join(off, "search", "compare.json"))
    gs = load(os.path.join(off, "gostanford", "gostanford.json"))
    seen = load(os.path.join(off, "seen", "seen_report.json"))
    guard = load(os.path.join(off, "guard.json"))
    ci = lambda r, k="diff_ci_drive": f"{r['diff']:+.4f} [{r[k][0]:+.4f}, {r[k][1]:+.4f}]"  # noqa: E731
    out = {}
    out["row1"] = {"pass": bool(g12 and g12["row1"]["pass"]), "result": "torch.equal" if g12 and g12["row1"]["pass"] else "fail/missing"}
    out["row2"] = {"pass": bool(g12 and g12["row2"]["pass"]),
                   "result": f"max diff {g12['row2']['max_abs_diff']:.1e}" if g12 else "missing"}
    r3 = search["row3"] if search else None
    out["row3"] = {"pass": bool(r3 and r3.get("control") == "zeroed" and r3["pass"]),
                   "result": f"maps {r3['mean_a']:.3f} vs zeroed {r3['mean_b']:.3f}: {ci(r3)}" if r3 else "missing"}
    out["row4"] = {"pass": bool(gs and gs["pass"]),
                   "result": f"official {gs['official']:.3f} -> C {gs['mapmad']:.3f}: {ci(gs, 'diff_ci_sample')}" if gs else "missing"}
    tz = seen["target_zeroed"] if seen else None
    out["seen"] = {"pass": bool(seen and seen["decision"] == "A"),
                   "result": f"target_zeroed - maps {ci(tz)} ({seen['n_samples']} frames); target_shifted - maps "
                             f"{ci(seen['target_shifted'])}" if seen else "missing"}
    out["guard"] = {"pass": bool(guard and guard["pass"]),
                    "result": f"maps C {guard['mean_a']:.4f} - run 1 {guard['mean_b']:.4f}: {ci(guard)} (pass: upper <= +{guard['margin']})"
                    if guard else "missing"}
    return out


def verdict(c: Dict[str, Dict[str, Any]]) -> str:
    """p10_rules_v4.md §4."""
    if not (c["row4"]["pass"] and c["guard"]["pass"]):
        return "STOP"
    g10 = all(c[k]["pass"] for k in ("row1", "row2", "row3"))
    if c["seen"]["pass"]:
        return "C_IS_V2" if g10 else "STOP"
    return "RUN1_IS_V2"


def main() -> None:
    off = os.path.join(sys.argv[1], "offline", "c_ema19")
    c = checks(off)
    v = verdict(c)
    names = {"row1": "G10 row 1 switch off", "row2": "G10 row 2 maps hidden", "row3": "G10 row 3 maps - zeroed",
             "row4": "G10 row 4 GoStanford no-harm", "seen": "seen-list rule (lower bound > 0)",
             "guard": "guard vs run 1 (upper <= +0.05)"}
    md = ["# Run C decision (p10_rules_v4.md)", "", "| Check | Result | Pass |", "|---|---|---|"]
    md += [f"| {names[k]} | {c[k]['result']} | {'✅' if c[k]['pass'] else '❌'} |" for k in names]
    md += ["", f"**Verdict: {v}** (C_IS_V2 = run C ema_19 is v2; RUN1_IS_V2 = run 1 ema_19 stays v2, limitation recorded; "
           "STOP = stop for Naz)"]
    with open(os.path.join(off, "decision.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(off, "decision.json"), "w") as f:
        json.dump({"checks": c, "verdict": v}, f, indent=1)
    print("\n".join(md))
    print(v)


if __name__ == "__main__":
    main()
