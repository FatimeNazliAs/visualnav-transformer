# p14b3_rules.md — amendment v1.1 (training-data mix only)

- **Written:** 2026-11-06, before the relaunch and before any v3 training step. Approved by Naz in VS Code (11-06).
- **Base rules:** `p14b3_rules.md` v1 (sha256 5cab73e5d3f34314f86de7506edc3b045b2f5d1b9dbabb468550489474e6691c), unchanged.
  The exam-B rules (arms, seeds, rows 1–3, Holm, cuts) are **not** touched.

## What happened
- The chain stopped at 2026-11-05 23:23 at "2 data index": the equal-count rule gave
  q = p(search_map) · s / p(explore) = 0.45 · 0.44469 / 0.20 = **1.00055** (needs 0 < q ≤ 1).
- Cause: round 2 grew map-DAgger (Phase 11 + round 2 maps shown) to 44.47% of the search_map pool. Equal draws would
  need the hidden-DAgger to take 100.06% of the explore draws: impossible with modes 45/20/20/15. (Phase 13: q = 0.513.)

## Amended rule (§3 "Data", equal-count rule)
- q is **capped at 1** (`map.hidden_q_cap: 1.0` in `train/config/mapmad_v3.yaml`): every explore draw takes a
  hidden-DAgger sample. Hidden-DAgger gets **99.94%** of the map-DAgger draws per epoch (gap 0.06%).
- Consequence: the Phase 8 drives get no plain-explore draws; they are still drawn in explore_map, photo and search_map.
- Everything else unchanged: mode shares 45/20/20/15, uniform within each union, `ema_9` fixed, no picking.
- Reported in `labels/union.json` (`hidden_q`, `hidden_q_raw`, `hidden_q_capped`) and in the summary under
  "Decisions to confirm" (rules amendment v1.1).
