# p10_rules_v4.md — Phase 10 run C (seen-before-out-of-view oversampling), rules frozen before launch

Version 4, 2026-10-21 19:1x. Naz's instruction after the option-B result (rules v3 §9 -> C). Earlier versions stay in
force for what they judged (G10 of run 1: `p10_rules.md` v3, sha256 `9e0e90dc…d0d6`). sha256 of this file in
`p10_rules_v4.sha256`; a copy in `mapmad/configs/p10_rules_v4.md`. Written and hashed before run C is launched.

## 1. Run C (one change)
- From the official `nomad.pth`; the same recipe as run 1 (`mapmad_v2.yaml`: 50/50 Habitat / GoStanford, spacing 5,
  20 epochs x 200k samples, lr 1e-5 pretrained / 1e-4 map encoders, warmup + cosine, the same mode shares 45 / 20 / 20 / 15,
  noisy target layer, the same false-mark rate, data, splits and frontier cache), **seed 1**, one auto-resume, the same
  launch setting as run 1 (batch 256 x grad_accum_steps 1, 12 workers; no new timing step: sampling only changes).
- **Only change:** `map.seen_oversample: 0.25` (`train/config/mapmad_v2_seen.yaml`, run `v2c_s1`): when the
  search_map mode is drawn, with probability 0.25 the sample is a uniform draw from the training search samples whose
  frame is "seen before, now out of view" (a simulator-truth detection, >= 200 px at <= 4 m, at an earlier frame and
  none at t; 43,549 of 1,773,475 training search samples); otherwise the run-1 rule. Training split only (test-loss
  curves and all checks unchanged). CPU check before launch: pool 43,549, seen share of search_map draws 0.265.
- **Checkpoint fixed in advance: `ema_19` of run C.**

## 2. Checks on run C's ema_19 (offline, deterministic, per-sample noise, the run-1 harness)
- G10 rows 1-2 (step-0 model, unchanged model code; re-run on the CPU), row 3 (maps - zeroed on the frozen 5,000-sample
  search list, drive-level paired 95% CI, 10,000 resamples), row 4 (GoStanford full test set vs official nomad.pth,
  <= official + 0.10); report-only arms as in run 1; exploration list.
- Seen list: `offline/lists/seen_list.json` (sha256 `bd44465c…8c93`, 2,151 frames / 107 drives), arms maps,
  target_zeroed, target_shifted (1 m).
- Guard: maps loss of run C minus maps loss of run 1 (`offline/main_ema19/search/scores.npz`) on the same 5,000
  search samples, drive-level paired 95% CI.

## 3. Pass rules
1. **Seen-list rule:** 95% CI lower bound of (target_zeroed - maps) > 0.
2. **G10 rows 1-4** pass as defined in `p10_rules.md` v3 §5 (row 3 = maps - zeroed).
3. **Guard:** 95% CI upper bound of (maps run C - maps run 1) <= +0.05.

## 4. Outcomes (decided by `p10_c_decide.py`, written to `offline/c_ema19/decision.md`)
- 1, 2 and 3 pass -> **run C ema_19 becomes MapMaD v2 for Phases 11-12.**
- 1 fails, while row 4 and the guard pass -> **v2 = run 1 ema_19** (sha256 `6d2492f1…`), recorded as a limitation
  ("the model does not use the remembered target out of view"); go to Phase 11; **no third run**.
- Row 4 or the guard fails -> **stop for Naz** (no twin).
- Not covered by Naz's wording, decided here before launch: rule 1 passes but G10 row 1, 2 or 3 fails -> stop for Naz.
  If rule 1 fails and row 1, 2 or 3 fails -> v2 = run 1 as above (run 1 passed G10), with the failed row recorded.

## 5. Twin (report only)
The old-recipe twin is cancelled. After the decision (not after a stop): seed 2 of whichever recipe becomes v2
(`mapmad_v2_seen_twin.yaml` = run C recipe, or `mapmad_v2_twin.yaml` = run 1 recipe), only after Phase 9 no longer needs
a GPU (its STATUS.md shows AT GATE / STOPPED / done), on a GPU free for us; the same offline checks, report only. It
never changes the v2 choice.

## 6. GPU, stops
- Never interrupt Phase 9; never create `P10_WANTS_GPU`. Take the first GPU that passes the free check (4 x 5 min with
  no other process, `gpu_watch.py`; Phase 9 and other users count as busy); pin it; log every GPU event.
- Automatic stops as `p10_rules.md` §7 (NaN loss, a second crash, out-of-memory, checkpoint sha256 not written,
  DataLoader timeout 900 s, a training log silent > 60 min = a crash, offline checks > 5 h, unexpected driver exit).
