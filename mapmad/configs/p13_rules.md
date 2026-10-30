# Phase 13 rules — Goal 1: make the map claim airtight (v1)

Written 2026-10-30 before any judged run, from the Phase 13 kickoff (2026-10-28; §3 decisions, §4 steps, §6 G13) and
the v2 plan v2.6. Frozen with its sha256 (`/outputs/mapmad/p13_claim/p13_rules.sha256`); copy in
`mapmad/configs/p13_rules.md`. A rule may be replaced only before its result is judged, with a new version + sha256.
Run configs: `mapmad/configs/p13_claim.yaml` (arm table, sets, run settings), `mapmad/configs/p13_hidden_dagger.yaml`
(13.3), `train/config/mapmad_v21b.yaml` (13.4). Code: the commit named in `driver/chain.log`.

## 1. What is judged (G13)
| # | Check | Pass |
|---|---|---|
| 1 | DAgger control (HM3D exam) | M21b − M21bh > 0, oracle stop, home-clustered one-sided p < 0.05 |
| 2 | Fresh replication (exam B) | M21 − M21h > 0, same test |
| 3 | Fresh control (exam B) | M21b − M21bh > 0, same test |
| 4 | Reported (no pass/fail) | M21bh − M21h, M21b − M21 (both sets); VF vs F, M21, N; port check; MP3D integrity; bins, categories; search/approach; rule stop at τ 0.45 |
| 5 | Fairness | all arms × episodes × seeds on frozen fingerprints; sha256 matched; none missing |
| 6 | Determinism | one episode twice on the same GPU identical; one Phase 12 M21 episode reproduced byte-identical with Phase 13 code |
| 7 | Videos | automatic: 20 random frames, drawn boxes = logged boxes; pose check; Naz answers target / YOLOE saw it / when scored / right object |

- Rows 1–3: the raw one-sided p and the **Holm-adjusted p over rows 1–3** are both reported; pass = raw p < 0.05
  and difference > 0 (kickoff §6). A failed row is recorded; no extra runs.
- **Pre-set reading of row 1 (kickoff §3 item 4):** passes and ≥ +0.10 → "map effect holds with equal practice";
  passes but < +0.10 → the claim narrows to that number; fails → the headline becomes "map + practice" and the clean
  claim is M2 − M2h (+0.073 [+0.005, +0.147], Phase 12).
- Row 6, second part fails → the Phase 12 arms on the HM3D exam are re-run with Phase 13 code only if the budget
  allows (≤ 1.5×); otherwise the comparison is reported as cross-code. No stop.

## 2. MP3D (13.0, 13.2)
- **Copy:** 90 homes from the PONI folder (read only) to `/mnt/shared_disk/nazli/mapmad_data/mp3d/` (`rsync -a`,
  dates kept); file dates = the releases (glb 2019-11-10, house 2017-10-21, semantic.ply 2019-04-01; 90 each);
  sha256 list `checks/mp3d_copy_sha256.txt`. Mounted read only at `/mp3d` in `naz_mapmad_habitat`.
- **Habitat:** every MP3D home loads with our copy of habitat-starter's `mp3d.scene_dataset_config.json`
  (`mapmad/configs/`; without it the semantic picture is misaligned: 10 of 162 official view points see their goal,
  with it 161 of 162); the LIMO navmesh is rebuilt (r 0.195 m, h 0.55 m; the shipped `.navmesh` is never used);
  labels = mpcat40 categories (chair, sofa, bed, toilet, tv_monitor, plant = our 6; YOLOE word for sofa = "couch").
- **Integrity** (homes with official ObjectNav MP3D episodes, train + val): every listed goal object exists in the
  home's labels with the same mpcat40 category and a box centre within 0.1 m; every listed view point lies on the
  floor of the original mesh (LIMO's downward floor probe finds a surface −0.05 to 0.35 m below it). **Drop rule:** a
  home with any label mismatch, or with more than 10% of its view points off the floor (a systematic offset = another
  mesh version), is dropped and reported. Isolated off-floor view points (mesh holes, ramps: 0–8.5% per home) never
  enter a success zone (the builder's usable rule: snapped ≤ 0.2 m onto LIMO's real floor). (The first wording, 0.5%,
  would have kept 1 of 11 val homes; replaced before the lists were used: phase13_log.md, Decisions to confirm.)
  Test homes (no official episodes): the date check only. Result: 0 label mismatches in 8,825 goals; 0 homes dropped;
  0 homes shared with HM3D (ids, mesh sha256). Lists: `mp3d_homes.json` (train 56, val 11, test 18).
- **Exam B** (`examb_set`, ids `p13b-<scene>-<category>-<id>`): MP3D val + test homes (29), Phase 12 builder rules:
  start snap ≤ 0.2 m on the real floor; an instance reachable on the start storey; nothing visible (any pixel) in 8
  headings at pin025; 3–15 m to the nearest usable view point; start filter (Phase 8). Val homes: official val
  episodes of our 6 categories. **Test homes** (no official episodes): goals = every label object of our 6 categories;
  view points generated with ObjectNav's recipe (`mapmad_sim/mp3d_goals.py`: 0.1 m grid points within 1.0 m of the box,
  on LIMO's floor map, the object visible (≥ 1 px) from pin025 facing it; radius chosen on 6 seeded MP3D **train**
  homes against their official view points: 1.0 m, recall 0.959 / precision 0.912, `checks/vp_calibration.json`);
  60 seeded candidate starts per home (a random point of LIMO's floor map, a uniform heading, the category
  round-robin over the home's categories with a usable view point), then the same rules. Draw: seeded (13201), ≤ 20
  per home (the smallest cap with ≥ 300 reachable: 303 of 345 kept; a first freeze with cap 12 gave 232 and was
  replaced before any run), aim 100 per bin (3–6 / 6–10 / 10–15 m) with the Phase 12 neighbour-bin fill, round-robin over the
  categories within a bin; a bin < 50 after the fill is reported, still run. Noise salt `mapmad-p13b`.
- **Dev set** (`dev_set`, ids `p13d-...`, report only, no choice is made on it): MP3D train homes, official train
  episodes of our 6 categories, 30 seeded candidates per home (13203), draw 120 (≤ 4 per home, seed 13204). Salt
  `mapmad-p13d`. The HM3D dev set is retired.
- **Frozen:** exam B `98a8d920…` (291 episodes, 26 homes: 101 val + 190 test; bins 100 / 105 / 86; 3 val homes keep no
  episode); dev `5745b340…` (120). `sets_mp3d.json`, preflight block below.

## 3. Hidden DAgger (13.3)
- v2 `ema_19` drives with **both maps hidden** on Phase 11's rollout set (fingerprint `8e3428c7…`, 1,448 episodes, 145
  labelled train homes), exactly as Phase 11's rollouts otherwise (sim-truth target layer with the Phase 8 noise, salt
  `mapmad-p11`, seed index 0, stop at the success zone, ≤ 1,000 steps), recorded as Phase 8 drives in
  `habitat_mapmad_v2_dagger_hidden`.
- Labels: the Phase 11 labeller unchanged (re-ask look-ahead 40 steps / every 5th, frozen blacklist; skip order
  collision streak ≥ 20 first, then teacher fail). Phase 11's label check is re-run and reported (limits median ≤ 0.25
  m, waypoint 1 ≤ 0.10 m: report only); a teacher-fail share > 40% is recorded, no stop.
- **Kept: search-phase states only** = labelled frames before the drive's first true sight (`first_in_view_step`:
  simulator truth sees an instance ≥ 200 px at ≤ 4 m; never seen = every frame). Counts per category and the kept
  share are reported (`hidden_dagger/labels`, `hidden_dagger/union.json`). Rollouts in the 10 dev homes = test split.

## 4. v2.1b (13.4)
- From v2 `ema_19`, Phase 11 recipe exactly (`mapmad_v21b.yaml` ← `mapmad_v21.yaml`): 10 × 200,000 samples, seed 1, lr
  groups (pretrained 1e-5, map encoders 1e-4), warm-up epoch 1 + cosine, batch 256, modes 45/20/20/15, GoStanford 50/50.
- Data: map-DAgger (Phase 11, 524,409 samples) in search_map as in v2.1; hidden-DAgger in the explore mode (photo and
  maps hidden). **Equal-count rule:** an explore draw takes a hidden-DAgger sample with probability
  q = p(search_map) · s / p(explore), s = map-DAgger share of the search_map pool, so the hidden-DAgger draws per epoch
  equal the map-DAgger draws in expectation (≈ 10.3% of Habitat draws ≈ half of the explore draws). Too few hidden
  samples → drawn with replacement (they always are, uniformly); the repeat factor is reported.
- **`ema_9` fixed in advance; no dev pick.** Training on GPU 1 preferred, `naz_mapmad` restarted first.
- G10 rows 1–4 offline on `ema_9` (Phase 10/11 checks). Rows 3–4 failing → recorded, exams still run; rows 1–2
  failing = a code bug → two fixes, then a safety stop.
- Speed settings: as Phase 11 (no bf16 / TF32 changes, same batch, same workers).

## 5. Arms, VF and the port check (13.5)
| Arm | Policy | Checkpoint | Maps | Seeds |
|---|---|---|---|---|
| M21 / M21h | MapMaD v2 settings | v2.1 `ema_9` (`ebaabe28…`) | shown / hidden | 3 |
| M21b / M21bh | same | v2.1b `ema_9` (V21B.json, sha256 checked) | shown / hidden | 3 |
| N | official NoMaD explore | `nomad.pth` (`70f79b82…`) | – | 3 |
| F | map + frontier (Phase 12) | – | – | 1 (deterministic) |
| VF | VLFM-style | – | – | 1 (deterministic) |
- Every arm: YOLOE-11-L-seg target layer (threshold 0.01, p99 0.788), pin025, 1,000 steps, no stop; YOLOE boxes +
  scores of every frame logged (`dets/<arm>/<run>.jsonl`; the step log is unchanged); **τ = 0.45 for all arms**.
- **VF:** VLFM's frontier choice (BLIP-2 ITM value map, prompt "Seems like there is a <word> ahead.", confidence-weighted
  fusion, best-value frontier) on our online map (our camera, LIMO footprint, YOLOE detections); **our planner + motion
  replace VLFM's PointNav policy**; once a target mark exists VF behaves exactly as F (layer mark ≥ τ from ≥ 2 views,
  goal within 1 m, arrive + face → search on). One BLIP-2 ITM server per GPU (`Salesforce/blip2-itm-vit-g`, ITC
  cosine, fp16). Report only. Details and every ported constant: `mapmad/src/mapmad_sim/search_vf.py`.
- **Port check (report only):** the original VLFM code and settings on 100 HM3D v2 val episodes (seeded, numpy 13501,
  list + sha256 written before any run) vs its paper (63.6% SR). Cannot run after two honest fixes → reported as "not
  done" with the reason; VF still runs.
- HSSD: report only if cheap (≤ 2 h work + ≤ 3 h GPU); judged not cheap in Part 1 → skipped, logged.

## 6. HM3D exam reuse
- The Phase 12 main exam (`6fb61eb8…`, 300 episodes, 34 homes, salt `mapmad-p12`) is used a **second time**, only for:
  row 1 (M21b − M21bh), the reported M21bh − M21h and M21b − M21 against the Phase 12 logs of M21 / M21h, and VF vs F,
  M21, N. Nothing is chosen or tuned on it. The Phase 12 arms are not re-run: their logs are reused (valid if row 6b
  shows Phase 13 code reproduces them).

## 7. Statistics
- As Phase 12: per-episode mean over seeds (a deterministic arm's one run counts for all); paired differences over the
  episodes both arms ran (a crash record in either arm drops the episode); home-clustered bootstrap (10,000 resamples,
  seed 12601): one-sided p for rows 1–3, 95% CIs everywhere; episode-level CIs too; Holm over rows 1–3.
- Oracle stop = first step within 0.2 m walking distance of a usable view point; rule stop = first step ≥ 1 with the
  rule value ≥ 0.45 (cells ≤ 1.0 m ahead, ±30°, ≥ 2 views). Wrong-object / near-miss as Phase 12.
- Exam B is also reported split by val / test homes.

## 8. Order, budgets, automatic stops
- Part 2 chain order: 13.1 videos → 13.3 rollouts + labels → index → restart `naz_mapmad` → 13.4 fine-tune → G10 rows
  → determinism (6a) → 13.6 dev sanity (M21b, M21bh, F, VF; seed 0) → 24-run timing → 13.7 exams (HM3D M21b/M21bh →
  exam B M21/M21h → exam B M21b/M21bh → exam B N, F, VF → HM3D VF) → port check → tables → CHAIN DONE.
- Budgets: 13.1 ≤ 1 h; 13.3 ≤ 6 h GPU; 13.4 ≤ 3 h; 13.6 ≤ 3 h; 13.7 ≤ 40 h; stop only at > 1.5×.
- Automatic stops (`STOP_reason.txt`, STATUS.md STOPPED + a question): checkpoint sha256 mismatch; duplicate ids;
  crash rate > 1%; determinism mismatch on the same GPU; runners exiting without progress; a recogniser / ITM server
  restarted > 3 times; budget > 1.5×.
- GPUs (v2 plan §8 rules 1–11): free = no other user's process for 4 × 5 min AND none in the last 60 min; live check
  right before taking a GPU; withdraw on another user's process; never share.

## 9. Fallbacks (kickoff §3, taken without a stop, logged under Decisions to confirm)
MP3D not loadable → skip exam B + dev; VLFM port check impossible → "not done"; VLFM licence / BLIP-2 weights
unavailable → VF skipped; 13.1 render-only impossible (Phase 12 logs hold no YOLOE boxes: the case) → the 12 video
episodes × 8 arms re-run on their Phase 12 GPU, a run whose pose sequence differs from the exam log is marked "not
reproducible"; hidden-DAgger too small → with replacement; exam-B bins < 50 → run what exists; too few MP3D train
homes for 120 dev episodes → use what exists; only one GPU usable → 3 pairs on it; v2.1b G10 rows 3–4 fail → recorded.

## 10. Pre-flight (checked by the chain before the first run)
```json preflight
{
 "checkpoints": {
  "p10_v2_train/runs/v2_s1_2026_10_21_08_42_50/ema_19.pth": "6d2492f19f7f1a5ccadea3f120b66a78b04bb5e88d10ca701fc523e0f9aee984",
  "p11_dagger/runs/v21_s1_2026_10_23_06_11_59/ema_9.pth": "ebaabe28c490567cbd453ae426332864ca1ef22961bf1fabae59bb62bdb7c74f",
  "p4_train/C_2026_10_11_05_21_43/ema_19.pth": "8093c6dba5f91bce48903e74ad9128fde50cccbf9a8f587f9aec8b288bffd9e2",
  "weights/official/nomad.pth": "70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333"
 },
 "sets": {
  "p12_search_exam/exam_set/episodes.json": "6fb61eb81abdcdaf6e948806186ca8659b85267dc5975c29941492be51814722",
  "p11_dagger/rollouts/set/episodes.json": "8e3428c74454f7a6f0a6021694c0e0eb8565c3c646d3ec5780b83930e187aa40",
  "p13_claim/examb_set/episodes.json": "98a8d920e198903e41e1b6b82c16074049ab1f5bb098dd02d9c441024b74cf1a",
  "p13_claim/dev_set/episodes.json": "5745b34074faa0bb654fc8cb71938af533ac1d8382361531afdfc721df4ab085"
 }
}
```
