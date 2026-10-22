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

Generated by p9_rules_appendix.py at commit fc632b5671d80846b391afd97119992c135b6789.

### A.1 Frame-set fingerprints
```json
{
 "six": {
  "files": 7715,
  "sha256": "5a87326769b7c82d78edec467581a90ed2df81be256f6a59506822d7e2232421",
  "items": 2730,
  "frames": 2571,
  "items_sha256": "0c2d285c20de5c81815e54133e6159c936ef71d8a22b13f1bb2d1dd7d2b11d8e"
 },
 "ovon": {
  "files": 3983,
  "sha256": "5e2c1b8c73fe8a5410fba2e073a807efc9ba918b5ebc454e12604f45d3602bb1",
  "items": 1395,
  "frames": 1327,
  "items_sha256": "6aacfa30492b8d3c26a772f14b5b4828e6f1bf5552c364adc6588b9682bb6789"
 },
 "frames_json_sha256": "fcc65f3ed22caa1238d99a5facfb28b2fee341351f05b29d9a90fdb9366c0f4d"
}
```

Halves: A = ['00081-5biL7VEkByM', '00440-wPLokgvCnuk', '00534-DBBESbk4Y3k', '00732-Z2DQddYp1fn', '00746-RTV2n6fXB2w'], B = ['00241-h6nwVLpAKQz', '00272-kA2nG18hCAr', '00330-WhNyDTnd9g5', '00397-ZNanfzgCdm3', '00401-H8rQCnvBgo6'] (first seeded split with every category >= 0.3, worst share 0.425).

| category | items | positive | target-free | ignore-only | target-free share |
|---|---|---|---|---|---|
| bed | 453 | 200 | 244 | 9 | 0.539 |
| chair | 461 | 200 | 244 | 17 | 0.529 |
| plant | 460 | 200 | 244 | 16 | 0.530 |
| sofa | 456 | 200 | 244 | 12 | 0.535 |
| toilet | 447 | 200 | 244 | 3 | 0.546 |
| tv_monitor | 453 | 200 | 244 | 9 | 0.539 |

OVON: 1395 items over 129 mapped names.

### A.2 Label tables (6 categories)
Mapping file `objectnav_categories.yaml` sha256 `ddd233df7c8a580b46c03349f3d806ba6e5c314909d8671006a4b78a4e4271d7`.

- **chair** (prompt `chair`): counted ['armchair', 'bar chair', 'chair', 'desk chair', 'dining chair', 'office chair', 'sofa chair']; ignore ['bench', 'kitchen chair', 'piano stool', 'pouffe', 'seat', 'shower seat', 'stacked chair', 'stone bench', 'stool']
- **sofa** (prompt `couch`): counted ['couch', 'sofa']; ignore ['armchair', 'pouffe']
- **bed** (prompt `bed`): counted ['bed']; ignore ['bed comforter', 'bedding', 'crib']
- **toilet** (prompt `toilet`): counted ['toilet']; ignore ['chamber pot', 'urinal']
- **tv_monitor** (prompt `tv`): counted ['display', 'monitor', 'tv']; ignore ['laptop', 'led tv', 'wall tv']
- **plant** (prompt `potted plant`): counted ['decorative plant', 'flower', 'flower vase', 'flowerpot', 'plant']; ignore ['chamber pot', 'christmas tree', 'pot', 'vase']

### A.3 OVON mapping (by name)

- val_seen: 79 names, 59 mapped: ['bag', 'balustrade', 'basket', 'bathroom cabinet', 'bathroom towel', 'bathtub', 'bed', 'bench', 'blanket', 'board', 'box', 'cabinet', 'cardboard box', 'case', 'chair', 'closet shelf', 'cloth', 'clothes', 'clothes dryer', 'container', 'cooker', 'couch', 'countertop', 'crib', 'desk', 'display cabinet', 'drawer', 'fire extinguisher', 'flag', 'high shelf', 'kitchen appliance', 'kitchen countertop items', 'kitchen lower cabinet', 'kitchen shelf', 'monitor', 'oven', 'oven and stove', 'pillar', 'plush toy', 'rack', 'shower cabin', 'shower soap shelf', 'sideboard', 'sink', 'sink cabinet', 'stool', 'stove', 'support beam', 'table', 'toilet', 'towel', 'toy', 'tv', 'wardrobe', 'washbasin', 'washer-dryer', 'washing machine', 'window', 'window shade']
- val_seen_synonyms: 50 names, 34 mapped: ['appliance', 'armchair', 'backpack', 'bar chair', 'bath cabinet', 'bathroom counter', 'bed table', 'bedside lamp', 'bicycle', 'bookshelf', 'chest of drawers', 'coffee table', 'computer desk', 'curtain', 'curtain rod', 'cushion', 'desk chair', 'dining chair', 'dining table', 'electric box', 'fireplace', 'furnace', 'kitchen cabinet', 'kitchen counter', 'lamp', 'railing', 'shelf', 'shower curtain', 'stairs railing', 'table lamp', 'throw blanket', 'tv stand', 'window curtain', 'window frame']
- val_unseen: 49 names, 40 mapped: ['blinds', 'boiler', 'book', 'calendar', 'carpet', 'christmas tree', 'clothes hanger rod', 'coffee machine', 'decorative plant', 'dishwasher', 'dresser', 'exercise bike', 'flower vase', 'flowerpot', 'food', 'footrest', 'glass', 'handrail', 'hanger', 'hanging clothes', 'microwave', 'mirror', 'nightstand', 'parapet', 'photo', 'photo mount', 'piano', 'picture', 'pillow', 'plant', 'printer', 'radiator', 'refrigerator', 'rug', 'shower dial', 'speaker', 'staircase handrail', 'statue', 'vase', 'window glass']
- unmapped: ['bath sink', 'bath towel', 'bathrobe', 'bathtub platform', 'bed sheet', 'bedframe', 'bowl of fruit', 'cabinet table', 'chest drawer', 'closet', 'clothes rack', 'computer', 'computer chair', 'exercise equipment', 'file cabinet', 'folding chair', 'footstool', 'freezer', 'guitar', 'handbag', 'heater', 'iron board', 'ironing board', 'island', 'kitchen island', 'lamp table', 'mantle', 'note', 'office chair', 'ottoman', 'refrigerator cabinet', 'shower', 'shower glass', 'shower rail', 'shower tub', 'sofa chair', 'sofa seat', 'sofa set', 'spa bench', 'stair', 'storage shelving', 'tablecloth', 'water heater', 'window shutter', 'worktop']

### A.4 Weights (sha256)

- mobile_sam: huggingface.co/dhkim2810/MobileSAM @ 4e2e5245fe9dc63096d575d36e70eb39b03b3aba
  - `mobile_sam.pt` 6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f
- yoloe: ultralytics assets (yoloe-11l-seg.pt) @ ultralytics==8.4.183
  - `mobileclip_blt.ts` a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54
  - `yoloe-11l-seg.pt` a993fb0fc7c8830939ae14e6434a925dd1179428158c2761482eb8a8d8a3699f
- yolo_world: ultralytics assets (yolov8l-worldv2.pt) @ ultralytics==8.4.183
  - `mobileclip_blt.ts` a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54
  - `weights/clip/ViT-B-32.pt` 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
  - `yoloe-11l-seg.pt` a993fb0fc7c8830939ae14e6434a925dd1179428158c2761482eb8a8d8a3699f
  - `yolov8l-worldv2.pt` 276b618c2f04e6e08e32f085b3c0e8baf2dbffa46af5119b888423e859a6ee01
- rfdetr: rfdetr medium @ rfdetr==1.11.2
- llmdet: huggingface.co/iSEE-Laboratory/llmdet_tiny @ d05199165a19320a9236396e20fca0a5e065189c
  - `model.safetensors` f0e476be3d65c7c50e13029b0adc4bb5d4e05f3eb6d7bb04856578095e6ad92f
- grounding_dino: huggingface.co/IDEA-Research/grounding-dino-tiny @ a2bb814dd30d776dcf7e30523b00659f4f141c71
  - `model.safetensors` 1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3
- owlv2: huggingface.co/google/owlv2-base-patch16 @ 2a1560802f8cf3c408fec9b809d705f56a2f7146
  - `model.safetensors` c4a566429e6b2c015abacb53087bf673260486c369a84e626e9b09fe57ca316d
- qwen3vl: huggingface.co/Qwen/Qwen3-VL-8B-Instruct @ 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
  - `model-00001-of-00004.safetensors` d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa
  - `model-00002-of-00004.safetensors` 8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5
  - `model-00003-of-00004.safetensors` 83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192
  - `model-00004-of-00004.safetensors` 0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5
- molmo2: huggingface.co/allenai/Molmo2-8B @ e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b
  - `model-00001-of-00008.safetensors` 0e36bb1dd408569baa883ebdbff142d6d71ad9b2bcdb335c2d7c103ae8184aad
  - `model-00002-of-00008.safetensors` a148abecde4269be5b71362d1e2b8d8b7f3b59a8761565d3ca8f1472bf1d463b
  - `model-00003-of-00008.safetensors` 7d4570c4a93212a62649fc205a4c1b894b58d69b32419686547be0f172a056c8
  - `model-00004-of-00008.safetensors` 3c572605ad05b34b25d9ecd0f0af3bc7af9f59f2201c2425e1417841a0b82a0c
  - `model-00005-of-00008.safetensors` 844a285fc8c07677c48cd2066d2e6ed753eaa74bf2934dc905fa5e2f0ac1c0cf
  - `model-00006-of-00008.safetensors` 7537a5af433d4ce6373b065b96d8504b530875ef1b9becb3c558066f45c35499
  - `model-00007-of-00008.safetensors` fe4373ca73e9951d3f11a38a0344cc7772d7ead0152b43584779f4f728306f11
  - `model-00008-of-00008.safetensors` 0c2b1c96b4e41af8c8729e4d97e0f9d5039bec6f2fa39808bc7cfcb4b0ca55e0
- sam3: NOT AVAILABLE at freeze time (GatedRepoError: 401 Client Error. (Request ID: Root=1-6ad8644b-54824b2a73464bd84b2ffa4a;b3c145ea-e9a4-43df-bfbd-0e8f5bc78aad)); SAM 3 is retried once before its turn
