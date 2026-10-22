# p10_rules_v5.md — Phase 10 run C: one more resume after the second crash (pre-set before resuming)

Version 5, 2026-10-22 06:3x. Naz's instruction after run C's automatic stop (second crash, 03:18). Adds to
`p10_rules_v4.md` (sha256 `a526a467…bd6f`), which stays in force for everything not changed here. sha256 of this file
in `p10_rules_v5.sha256`; a copy in `mapmad/configs/p10_rules_v5.md`. Written and hashed before the resume.

## 1. What happened (record)
Run C (`runs/v2c_s1_2026_10_21_21_44_11`, GPU1 from 00:44) trained epochs 0-10 cleanly; in epoch 11 a DataLoader worker
died with a segmentation fault twice (03:09 step 273; after resume 1, 03:17 step 457) -> automatic stop (v4 §6).
**Deviation, cause: suspected memory pressure while Phase 9's qwen3vl ran** (Phase 9 took GPU0 at 03:06 for the qwen3vl
plug-in until 03:23; both crashes inside that window; swap 1 / 1 GB used afterwards; kernel log not readable, so not
proven). Run 1 (20 epochs) and run C epochs 0-10 ran without a crash while nothing else heavy ran.

## 2. Resume 2 (Naz's exception to "a second crash stops")
- Resume run C ONCE more from `latest.pth` (end of epoch 10) on the first GPU that passes the free check; recipe,
  config (`launch/c.yaml`) and run folder unchanged; ema_19 of run C stays the checkpoint fixed in advance.
- Checked before resuming (CPU): latest.pth holds epoch 10 (-> next 11), model, EMA weights equal to ema_10.pth, EMA
  step 8,602 (= 11 x 782) and decay, Adam state (2 groups, step 8,602), torch / CUDA / numpy RNG states; the lr
  schedule is a function of (epoch, step) (WarmupCosine) and the sampler is re-seeded per epoch -- resume 1 started
  epoch 11 at lr 4.587e-05, the same as run 1's epoch 11.
- Free RAM and swap are logged at the start, after every epoch and at the end (driver/c.log, phase10_log.md;
  every 60 s in driver/mem_chain.csv).

## 3. Pre-set outcome if it crashes again (no more stops)
Any run-C training that does not finish (crash, hang, NaN) -> **v2 = run 1 ema_19** (sha256 `6d2492f1…`), recorded as
a limitation (no result for run C), then the run-1 recipe's twin (seed 2, report only). No further resume.

## 4. If it finishes
Rules v4 §2-4 unchanged (checks, pass rules, C_IS_V2 / RUN1_IS_V2 / STOP). After the verdict (not after STOP) the twin
of the winning recipe runs on any GPU that passes the free check, one training at a time (Phase 9 is at its gate).
