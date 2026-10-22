#!/usr/bin/env python3
"""Draft gate G11 table (chain step 9; p11_rules.md, v2 plan §5). Host side, standard library only.

    python3 g11_table.py <p11 dir>

Reads step0/decision.json (row 0), rollouts/ summaries + labels/label_check.json + labels/label_stats.json +
sheets/label_sheet.json (row 1), offline/v21_ema9/{g10_rows_1_2_cpu.json, search/compare.json,
gostanford/gostanford.json, seen/seen_report.json, v21_minus_v2.json} (row 2), dev_v21/table.json (row 3) and the twin's
offline/twin_ema19/ (addendum); writes G11_table_draft.md / .json. A missing input leaves its row "pending".
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


def ci(r) -> str:
    return f"{r['diff']:+.3f} [{r['diff_ci_drive'][0]:+.3f}, {r['diff_ci_drive'][1]:+.3f}]"


def count_runs(p11: str, sub: str, arm: str):
    d = os.path.join(p11, sub, "summaries", arm)
    names = os.listdir(d) if os.path.isdir(d) else []
    return sum(n.endswith(".json") and not n.endswith(".crash.json") for n in names), sum(n.endswith(".crash.json") for n in names)


def main() -> None:
    p11 = sys.argv[1]
    rows, notes = [], []
    dec = load(os.path.join(p11, "step0", "decision.json"))
    if dec:
        rows.append((0, "v2 pick", "dev check of run 1 vs run C applied mechanically from p11_rules.md (tie -> run 1); "
                     "winner named", f"Delta = mean(C - run 1) = {dec['delta']:+.4f} (run 1 {dec['success_run1']:.3f}, "
                     f"run C {dec['success_runC']:.3f}; {dec['episodes']} episodes x 3 seeds) -> **v2 = {dec['arm']}**", True))
    else:
        rows.append((0, "v2 pick", "rule applied mechanically; winner named", "pending", None))
    done, crashed = count_runs(p11, "rollouts", "v2")
    chk, stats = load(os.path.join(p11, "labels", "label_check.json")), load(os.path.join(p11, "labels", "label_stats.json"))
    sheet = load(os.path.join(p11, "sheets", "label_sheet.json"))
    if stats and chk and sheet:
        res = (f"{done} rollouts ({crashed} crashed) in 145 labelled homes; label check median {chk['median_m']:.3f} m "
               f"(p95 {chk['p95_m']:.3f}; limit {chk['limit_m']}); {len(sheet['states'])}-state sheet; labelled "
               f"{stats['overall']['ok']:.3f}, teacher-fail {stats['teacher_fail_share']:.3f} (limit "
               f"{stats['limit_teacher_fail_share']}), collision {stats['overall']['collision']:.3f}")
        ok = chk["pass"] and stats["pass"] and done > 0
        rows.append((1, "Rollouts + labels", "done in the 145 labelled train homes; 10-state sheet; teacher-fail share "
                     "reported", res, ok))
        notes.append("Teacher-fail share per category: " + json.dumps(
            {c: v["teacher_fail"] for c, v in stats["per_category"].items()}))
    else:
        rows.append((1, "Rollouts + labels", "rollouts, sheet, skipped share", f"{done} rollouts; labels pending", None))
    o = os.path.join(p11, "offline", "v21_ema9")
    g12, comp = load(os.path.join(o, "g10_rows_1_2_cpu.json")), load(os.path.join(o, "search", "compare.json"))
    gs, seen = load(os.path.join(o, "gostanford", "gostanford.json")), load(os.path.join(o, "seen", "seen_report.json"))
    if g12 and comp and gs:
        r3 = comp["row3"]
        parts = [f"row 1 switch off {'✅' if g12['row1']['pass'] else '❌'}",
                 f"row 2 maps hidden {'✅' if g12['row2']['pass'] else '❌'} (max diff {g12['row2']['max_abs_diff']:.1e})",
                 f"row 3 maps - zeroed {ci(r3)} {'✅' if r3['pass'] else '❌'}",
                 f"row 4 GoStanford {gs['official']:.3f} -> {gs['mapmad']:.3f} ({gs['diff']:+.3f}) {'✅' if gs['pass'] else '❌'}"]
        ok = g12["row1"]["pass"] and g12["row2"]["pass"] and r3["pass"] and gs["pass"]
        rows.append((2, "Model checks", "G10 rows 1-4 again on ema_9 (+ seen-list rule, report only)", "; ".join(parts), ok))
        if seen:
            tz = seen["target_zeroed"]
            notes.append(f"Seen-list rule (report only): target_zeroed - maps {ci(tz)} -> "
                         f"{'uses the mark' if tz['diff_ci_drive'][0] > 0 else 'CI lower bound <= 0'}")
        diff = load(os.path.join(o, "v21_minus_v2.json"))
        if diff:
            notes.append(f"v2.1 - v2 on the frozen search list (report only): {ci(diff)}")
        if "maps_vs_hidden" in comp:
            notes.append(f"maps - hidden (report only): {ci(comp['maps_vs_hidden'])}")
    else:
        rows.append((2, "Model checks", "G10 rows 1-4 on ema_9", "pending", None))
    dev = load(os.path.join(p11, "dev_v21", "table.json"))
    if dev and "difference" in dev:
        d = dev["difference"]
        t = dev["tables"]
        rows.append((3, "Dev check (report only)", "v2.1 vs v2 on the dev set, oracle-stop success",
                     " / ".join(f"{a} {r['success']:.3f}" for a, r in t.items()) + f"; {d['arms']} {d['mean']:+.4f} "
                     f"[{d['ci95_episode_bootstrap'][0]:+.4f}, {d['ci95_episode_bootstrap'][1]:+.4f}]", True))
    else:
        rows.append((3, "Dev check (report only)", "v2.1 vs v2, oracle-stop success", "pending", None))
    twin = os.path.join(p11, "offline", "twin_ema19")
    tc, tg, ts = (load(os.path.join(twin, *x)) for x in (("search", "compare.json"), ("gostanford", "gostanford.json"),
                                                         ("seen", "seen_report.json")))
    if tc and tg:
        notes.append(f"Twin (seed 2, report only): maps - zeroed {ci(tc['row3'])}; GoStanford {tg['diff']:+.3f}"
                     + (f"; seen list target_zeroed - maps {ci(ts['target_zeroed'])}" if ts else ""))
    else:
        notes.append("Twin (seed 2, report only): pending / not run")
    md = ["# G11 draft (Phase 11)", "", "| # | Check | Pass | Result | ✓ |", "|---|---|---|---|---|"]
    for n, name, rule, res, ok in rows:
        md.append(f"| {n} | {name} | {rule} | {res} | {'✅' if ok else ('❌' if ok is False else '…')} |")
    md += [""] + [f"- {x}" for x in notes]
    with open(os.path.join(p11, "G11_table_draft.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(p11, "G11_table_draft.json"), "w") as f:
        json.dump({"rows": rows, "notes": notes}, f, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
