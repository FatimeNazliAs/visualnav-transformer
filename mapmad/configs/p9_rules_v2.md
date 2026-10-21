# p9_rules_v2.md — Phase 9 amendment 1 (GPU rule, run order, threshold-edge report)

Written 2026-10-21 18:35, decided by Naz in the Claude Code chat after the 16:14 GPU incident. It is frozen with its
sha256 (`/outputs/mapmad/p9_recognisers/p9_rules_v2.md.sha256`) **before** the driver applies it. It amends
`p9_rules.md` (v1, sha256 `e8b0b9e99ba902b1bccd6aeb7d528a1c0df06a58c021fd0ab25f6b8f11a5c407`, frozen 10:06); every
rule not named here is unchanged.

State when written:
- Done: frame set, v1 rules, truth parity (pass), YOLOE (threshold 0.01).
- Not yet run: yolo_world, rfdetr, llmdet, grounding_dino, owlv2, sam3, qwen3vl, molmo2.
- Nothing has been scored on half B.

## A1. GPU rule (replaces v1 §11 "Taking a GPU")
- **Free:** as before, no Phase 10 or other user's process for 4 consecutive 5-min checks (20 min). In addition, the
  GPU is **checked live with nvidia-smi right before it is taken**; any other process there → not taken.
- **Taking a GPU:** Phase 9 may take a free GPU when **Phase 10 holds no GPU**, and, as in v1, when Phase 10 holds a
  GPU and the GPU taken is another one. Together: any free GPU may be taken, unless the flag below exists.
- **Flag:** the file `/mnt/shared_disk/nazli/nomad_outputs/mapmad/P10_WANTS_GPU` (created by Naz or Phase 10) means
  Phase 10 needs a GPU. While it exists:
  - Phase 9 starts no new model on any GPU;
  - a model already running finishes; then the GPU is released (servers stopped).
  - The flag is checked before every model and every 5 min while one runs.
- **Unchanged:** one model per GPU, one GPU at a time; withdrawal if a Phase 10 or other process appears on our GPU
  (finish the current model, no new work there); budget ≤ 8 h (stop at 12 h).
- **Cut short ≠ failure:** a model cut short or timed while another process shared our GPU is moved to
  `photo_test/_partial/` and re-run from zero. It does not count as one of its 2 attempts.
- **yolo_world:**
  - attempt 1 (15:43–16:14) failed with an unexplained server exit after 1,000 items (no traceback, no kernel kill;
    the same items did not crash in a CPU replay). It counts as a failed attempt;
  - attempt 2 (16:14–16:20) started 1 s after koray's job arrived on GPU 0 (a stale watcher file) and ran out of
    memory there. It does not count.
  - So yolo_world has **one** attempt left.

## A2. Run order (replaces the queue order in v1 §3)
yolo_world → rfdetr → llmdet → grounding_dino → owlv2 (the 5 detectors, ~15 min each, so a release costs ≤ ~15 min) →
sam3 (its download retried once just before; skipped with the reason if still unavailable) → qwen3vl → molmo2 (VLMs
last). Then the threshold-edge passes (A3), scoring, the choice, and the R1/R2 plug-in runs.

## A3. Threshold at a grid edge (report only)
- The pre-registered grid (0.01 … 0.99, ties → lowest) is **kept**. Thresholds, scores and the choice rule do not
  change.
- **Flag:** any model whose tuned threshold is 0.01 or 0.99 is flagged in the tables.
- **Extra report:** for a flagged model, its map F1 (and H, R) on half B at thresholds **0.005 and 0.001**, marked
  report-only. The v1 raw pass kept detections ≥ 0.01 only, so this needs one extra raw pass of that model with
  score ≥ 0.001 (top 10 per item, otherwise identical). It is stored in `photo_test/<model>/raw_edge/` and scored by
  the v1 rules into `photo_test/<model>/edge.json`.
  - As a consistency check, F1 at 0.01 from the extra pass is also reported.
- A3 never changes R1 / R2, the threshold, or any half-B number used by the choice rule.
