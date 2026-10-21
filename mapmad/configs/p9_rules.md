# p9_rules.md — Phase 9 recogniser comparison (photo test), pre-registered

Written 2026-10-21 before any frame was rendered or any model was scored. This text is committed to git before the
Phase 9 driver runs. The driver then appends **Appendix A** (frame-set fingerprints, label tables, OVON mapping,
halves, weight sha256s), all of it generated mechanically from the frozen frame set and the downloads. It writes the
whole file to `/outputs/mapmad/p9_recognisers/p9_rules.md` plus its sha256 **before the first model is scored**. The
copy in `mapmad/configs/` is updated to that full file at the gate.

Source: Phase 9 confirmation block (app chat, final), v2 plan v2.2 §8.4 / §8 / §9. Code: `mapmad/src/mapmad_sim/
photo_labels.py`, `photo_frames.py`, `photo_scoring.py`, `recogniser.py`, `mapmad/src/mapmad_recog/`,
`mapmad/scripts/p9_*.py`; configs `p9_frames.yaml`, `p9_models.yaml`.

## 1. Frame set (6 categories)
- **Homes:** only the 10 labelled Phase 1 homes of the frozen val-drive split (`p2_home_splits.json`): 00746, 00440,
  00272, 00330, 00241, 00732, 00081, 00534, 00397, 00401. These are HM3D train-split scenes held out of MapMaD
  training since Phase 2. Never exam homes (36 HM3D val), never MapMaD training homes.
- **Camera:** pin025 (`camera_fair.yaml`): pinhole 66.5°, 0.25 m, 320 × 240, pitch −2°, 0.084 m forward, depth
  co-located. Stored as raw PNG: RGB, depth (uint16 mm), semantic (uint16 ids).
- **Poses:**
  - (a) `rnd`: 300 seeded random real-floor poses per home. The floor rule is Phase 1's: a big island, and the surface
    is floor / carpet / rug / mat. Heading is uniform, and the camera must be in free space.
  - (b) `drv`: every frame of the home's Phase 8 test-split drives, re-rendered at the saved pose, thinned per drive:
    a frame is kept if it is ≥ 1 m or ≥ 30° from the last kept one.
- **Ids:** frame `p9f-<home>-<rnd|drv>-<n>`; item `p9q-<frame>-<category>` (6 categories), `p9o-<frame>-<OVON
  name>` (OVON). Duplicates are refused.
- **Query item** = one frame + one category, asked alone (one category per call, never all 6 at once).
- **Item status for the asked category:**
  - *positive*: ≥ 1 counted instance (§2);
  - *target-free*: no pixel of any instance of the category;
  - *ignore-only*: the category is in view only below the counted rule.
- **Selection** (seed 20261021, per category):
  - up to 200 positive items;
  - target-free items so that they are 55% of positives + target-free;
  - ignore-only items at their natural rate among non-positive frames (≤ 0.10 × target-free). The resulting
    target-free share is 52–55%.
  - Each class is drawn round-robin over the homes in a seeded order.
  - No home may hold more than 20% of all items: the largest home loses target-free items, logged.
  - Fewer frames possible → the maximum, logged, no stop.
- **Halves:** 5 + 5 homes, all C(10, 5) splits in a seeded order. The first split in which every category has ≥ 30% of
  its positives in each half is used; if there is none, the best worst-category share (logged).
- **Frozen** with sha256 (Appendix A) before any scoring. The re-render must reproduce the candidate pass's instances
  exactly, otherwise the build stops.

## 2. Truth
- **Category mapping:** HM3D raw label → category by `objectnav_categories.yaml` (Phase 8's mapping; sha256 in
  Appendix A).
- **Counted instance:** raw label counted for the asked category, **≥ 200 pixels and median depth (pixels with depth
  > 0) ≤ 4 m**.
- **Ignore:**
  - (a) unlabelled pixels (id 0);
  - (b) instances of the asked category that fail the counted rule;
  - (c) near-synonym instances: raw labels of the 10 homes that share a word with the category's list in
    `photo_labels.NEAR_WORDS`, minus `NOT_NEAR` and minus counted labels. The resulting lists are frozen in
    Appendix A.
- Detections on ignore are neither hits nor false alarms. Every other label is background, and a detection there is
  a false alarm.
- Prompts: chair → `chair`, sofa → `couch`, bed → `bed`, toilet → `toilet`, tv_monitor → `tv`, plant → `potted plant`.

## 3. Models (exact checkpoints: `p9_models.yaml` revisions; weight sha256 in Appendix A)
| key | model | family | kind | output |
|---|---|---|---|---|
| yoloe | YOLOE-11-L-seg (ultralytics 8.4.183) | fast YOLO | open | mask |
| yolo_world | YOLO-World v2-L (ultralytics 8.4.183) | fast YOLO | open | box |
| llmdet | LLMDet-T | DINO | open | box |
| grounding_dino | Grounding DINO-T | DINO | open | box |
| owlv2 | OWLv2-B/16 `google/owlv2-base-patch16` (self-trained) | CLIP/ViT | open | box |
| sam3 | SAM 3 (gated) | segmentation with text | open | mask |
| rfdetr | RF-DETR-M (rfdetr 1.11.2), fixed COCO list, 6 categories only | fixed | fixed | box |
| qwen3vl | Qwen3-VL-8B-Instruct | VLM | vlm | box (text) |
| molmo2 | Molmo2-8B (Apache-2.0) | VLM | vlm | point (text) |
| truth | simulator truth, Phase 8 rule (upper limit, not a candidate) | — | truth | mask |

- **Mask rule:** box models → MobileSAM mask (`dhkim2810/MobileSAM` @ 4e2e524, through ultralytics), one call per
  item for all its boxes. Point models → point + depth.
- **Raw pass:** top 10 detections per item with score ≥ 0.01.
- **Inputs:** every model gets the 320 × 240 RGB. Model-specific inputs:
  - ultralytics: default `imgsz`, the BGR array, `retina_masks`;
  - Grounding DINO / LLMDet: text `"<prompt>."`, box threshold = the floor, text threshold 0;
  - OWLv2: text `[[prompt]]`, boxes on the padded square, clipped.
- **VLM prompts** (greedy, ≤ 256 new tokens, score = 1, one operating point, AP50 n/a):
  - Qwen3-VL: `Locate every {name} in the image. Answer only with a JSON list in which each element is
    {"bbox_2d": [x1, y1, x2, y2], "label": "{name}"}. If there is no {name} in the image, answer [].`
    (coordinates 0–1000 relative; boxes → MobileSAM).
  - Molmo2: `Point to every {name} in the image. If there is no {name} in the image, say that there is none.`
    (`<points coords=…>`, 0–1000 relative; v1 percent format also parsed).
- **Truth "model":**
  - 6 categories: the home's ObjectNav goal ids, ≥ 200 pixels at depth (0, 4] m (exactly Phase 8's generator rule);
  - OVON: the instances counted by name, with the same pixel rule.
- **Skips:** a model that cannot be downloaded (licence, gate) or fails twice is skipped with the reason. SAM 3 is
  downloaded last, and the download is retried once right before its turn.

## 4. Matching and metrics (half B; per category and overall)
**Pixel level** (detections in score order, best first, greedy):
- **TP:**
  - box / mask models: box IoU ≥ 0.5 with a not-yet-matched counted instance's 2-D box (best IoU);
  - point models: the point lies on a not-yet-matched counted instance's pixels.
- **IGN:** IoU ≥ 0.5 with an ignore instance's box, or ≥ 50% of the mask (box region without a mask; the point's
  pixel) on id 0 or ignore instances.
- **FP:** anything else.
- **Metrics:**
  - **AP50** = VOC all-point AP over all raw detections;
  - **precision / recall** at the threshold;
  - **false alarms per target-free item** = kept FP on target-free items ÷ target-free items (ignore-only items are
    excluded from it).

**Map level (the 1 m rule):**
- **Point:** median (per axis) of the world points of the mask pixels (a point model: its pixel) with depth (0, 4] m
  and height ≥ floor − 0.10 m. No such pixel → *off-map* (not drawn; neither hit nor miss; counted and reported).
- **Hit:** within 1 m, horizontal (map plane), of a counted instance's world box (HM3D boxes are axis-aligned,
  `objects.object_box`). Otherwise IGN if within 1 m of an ignore instance's box. Otherwise *miss*.
- **H** = hits ÷ (hits + misses) over kept detections.
- **R** = share of (item, counted instance) pairs with ≥ 1 kept hit.
- **Map F1** = 2·H·R/(H+R).

**Reported** (per category and overall):
- the metrics above;
- latency median + p95;
- peak VRAM (`torch.cuda.max_memory_allocated`);
- recall by distance 0–2 / 2–4 m (instance median depth);
- by "cut by the image edge" vs "whole in view" (the visible-fraction measure);
- H by detection distance;
- by source (rnd / drv);
- top confusion pairs (asked → mapped category of the raw label under the false alarm).

## 5. Threshold and scoring
- One threshold per model: the value on the grid 0.01, 0.02, …, 0.99 with the highest map F1 on **half A** (6
  categories). Ties → the lowest threshold.
- Per-category thresholds (same rule per category) are reported only.
- All metrics are on **half B** at that threshold.
- VLMs and truth have one operating point (no tuning).

## 6. Speed
- **Latency** = round trip from the Habitat-side client to the recogniser sidecar and back, over mapmad-net, including
  the model, MobileSAM and the socket. Measured on an RTX 4500 with 320 × 240 input, one model alone on its GPU.
- **Timing pass:** 350 seeded half-B 6-category items at the model's threshold; the first 50 are warm-up and not
  counted. VLMs: their raw pass is the driving setting, so they use half-B 6-category items of the raw pass, excluding
  the pass's first 50 calls.
- Qualifies for driving if the **median ≤ 0.25 s per query item**.
- Phase 12 cost = 5,000,000 calls × median latency.

## 7. Choice rule (applied mechanically, `photo_scoring.choose`)
- **Score** = map F1, 6 categories, half B, at the model's threshold.
- **Pool** = open-vocabulary models that ran (kind open or vlm). RF-DETR (fixed list) and truth are reported only.
- **R1** = the best qualifying model by score. Tie (F1 within 0.01 of the best) → the fastest of the tied.
- **R2** = the same over qualifying models of a **different family** from R1. Families: fast YOLO · DINO · CLIP/ViT ·
  segmentation with text · VLM.
- **VLMs** may take a slot only if they have the best F1 of that slot's candidates (and qualify).
- **Fallback:** fewer than 2 qualify → fill with the fastest remaining open-vocabulary models (another family if
  possible), flagged. No stop.

## 8. Score mapping (plug-in)
- Kept score s (≥ t) → **0.4 + 0.6 · clip((s − t)/(p99 − t), 0, 1)**, where p99 = the 99th percentile of the model's
  kept scores on half A.
- A VLM's score 1 → 1.0. The raw score is saved too.
- Truth keeps Phase 8's values (clean = 1.0).

## 9. OVON photo test (report only)
- **Names:** OVON val category names per split (val_seen, val_seen_synonyms, val_unseen), read from the episode
  files only. No OVON home is rendered: all of them are exam homes.
- **Mapping by name** (lower case, "-"/"_" → " "):
  - a raw label equal to the name is counted;
  - a raw label whose words contain all of the name's words, or are all contained in them, is ignore.
  - Mapped and unmapped names are logged (Appendix A).
- **Items:** the same frames and truth rules, up to 5 positive items per mapped name (= the coverage minimum; ~1,400 items for 133 mapped names), 55% target-free (§1 numbers).
- **Covered:** a name counts as covered only with ≥ 5 positive items.
- **Models:** open-vocabulary models only (incl. VLMs; not RF-DETR), at each model's 6-category threshold, on **all**
  OVON items (no tuning on OVON).
- Metrics as §4, overall and per split (over covered names).

## 10. Plug-in
- **Interface:** `Recogniser.detect(rgb, depth, category) → list[Detection(mask | point, score)]` (`recogniser.py`).
  Implementations: simulator truth, R1 and R2 (sidecar `naz_mapmad_recog` over mapmad-net).
- **Layer:**
  - a detection's cells = the storey cells under its mask pixels (point: its pixel) through Phase 8's
    `frame_cells`;
  - per frame, the maximum per cell;
  - fused over views by Phase 8's `TargetFusion`.
- **Written** in the Phase 8 format (`mapmad_target.npz`):
  - `noisy_*` = the recogniser's layer;
  - `clean_*` = truth (Phase 8 rule);
  - `det_*` raw detections (+ `det_raw_score`).
- **Parity (G9 row 4):** truth through the plug-in = the drive's saved Phase 8 clean layer, exactly (cell, frame,
  value arrays), on 2 seeded Phase 8 test-split search drives in the dev homes. R1 and R2 run on the same 2 drives:
  layers drawn, latency per step, one video each.

## 11. GPUs (v2 plan §8 + block item 9)
- **Free:** a GPU holds no Phase 10 or other user's process for 4 consecutive 5-min checks.
- **Taking a GPU:** Phase 9 starts work on a GPU only if, after taking it, another GPU is still free or Phase 10
  already holds a GPU.
- **Scope:** one model per GPU, one GPU at a time.
- **Withdrawal:** re-checked every 5 min. If a Phase 10 or other process appears on our GPU, the current model
  finishes and no new work starts there. A model cut short is re-run from zero.
- **Budget:** ≤ 8 h GPU (stop at 12 h); waiting time is not counted.

## 12. Gate G9
| # | Check | Pass |
|---|---|---|
| 1 | Frames | ≥ 2,000 items (6 categories) + ≥ 1,000 (OVON), or the maximum possible in the 10 homes (logged); frozen with sha256; target-free share 50–60% per category; never exam homes |
| 2 | Models | every model ran or was skipped with a logged reason |
| 3 | Choice | rule applied mechanically from this file (incl. the fallback); R1, R2 named |
| 4 | Plug-in | R1 and R2 run inside the Habitat harness and write the target layer in the Phase 8 format; truth through the plug-in = Phase 8 clean layer on 2 dev drives |

## Appendix A (generated before scoring)
