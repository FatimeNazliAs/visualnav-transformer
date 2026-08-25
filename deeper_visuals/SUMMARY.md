# Deeper Visualization — project summary

All six phases (**P0–P5**) are complete and published: each ships an advisor webpage at a fixed artifact URL and a technical Notion page group. Known-good state is tagged `deeper-viz-v1`.

| Phase | What it shows | Code | Advisor page | Notion |
| --- | --- | --- | --- | --- |
| **P0** — Overview | What NoMaD is: one checkpoint, two behaviours (navigate / explore). Static, no model run. | `deeper_visuals/p0_overview/` | [artifact](https://claude.ai/code/artifact/1681b5a2-1e17-499a-af10-29bc876ba130) | [page](https://app.notion.com/p/3c1d0eaafe5581cbaf68eb0be625a13f) |
| **P1** — Inputs | The five pictures the model gets: four past camera frames plus the goal frame. | `deeper_visuals/p1_inputs/` | [artifact](https://claude.ai/code/artifact/2fa526f2-d3ac-40bc-b7b5-3dce08190389) | [page](https://app.notion.com/p/3c1d0eaafe5581fc87e3e553e088cf2c) |
| **P2** — Encoders | Pixels to numbers: ψ and φ (separate EfficientNet-B0s) turn each frame into a 256-dim token. | `deeper_visuals/p2_encoders/` | [artifact](https://claude.ai/code/artifact/aed15c20-7968-4d4c-aff3-3cec670fb980) | [page](https://app.notion.com/p/3c1d0eaafe5581699496cbd93aa7a3eb) |
| **P3** — Transformer & goal masking | Attention across the 5 tokens, and the goal-mask switch that flips navigation to exploration. | `deeper_visuals/p3_transformer_masking/` | [artifact](https://claude.ai/code/artifact/52beac61-bada-4600-89a1-b52892f1992a) | [page](https://app.notion.com/p/3c1d0eaafe5581b2a9fac9f2834524a4) |
| **P4** — Diffusion policy | cₜ + noise denoised over K=10 steps into an 8-step action sequence; plus the distance head. | `deeper_visuals/p4_diffusion/` | [artifact](https://claude.ai/code/artifact/6799a471-a066-4aab-b81c-5dee6add5535) | [page](https://app.notion.com/p/3c1d0eaafe5581a98637ec11d389ad83) |
| **P5** — Output & multimodality | Seed spread: samples agree on the first step and part company later — sampled, not computed. | `deeper_visuals/p5_output_multimodal/` | [artifact](https://claude.ai/code/artifact/69585ff5-375c-43d0-bb17-a34e1cd61ecc) | [page](https://app.notion.com/p/3c1d0eaafe5581f5998ee7533b8524c4) |

## Fixed facts

- **Branch:** `feature/deeper-visualization` (the old `debug_visuals/` folder is untouched).
- **Checkpoint:** `nomad_2026_06_13_18_04_23`, epoch 99/100, default weight `ema_99.pth` (`latest.pth` = raw fallback).
- **Hero scene:** `right_turn_doors` (`no11vc_9_1` @ frame 54), threaded through P1–P5.
- **Dataset:** `go_stanford`; scenes curated in `deeper_visuals/common/samples.yaml`.
- **Artifact URL registry:** `deeper_visuals/common/artifacts.yaml` — fixed per phase; republishing the same path updates the same link.

## Running a phase

From the repo root on the host (`update.sh` shells into the container itself):

```bash
# 1. edit deeper_visuals/pN_name/config.yaml  (sample: key from samples.yaml, checkpoint: ema|latest)
./deeper_visuals/pN_name/update.sh          # run_model -> build_page -> serve :8001
./deeper_visuals/pN_name/update.sh --page-only   # skip the forward pass
# 2. preview at localhost:8001/pN/latest.html
```

P0 is static (no model, no config). P1–P5 each run a forward pass on the checkpoint; every figure is a matplotlib PNG produced by the real model, never redrawn in JS.

## Layout

```
deeper_visuals/
  common/      shared code: samples.yaml, artifacts.yaml, model.py, data.py,
               actions.py, denoise.py, figures.py, facts.py, measure.py,
               build_page.py, template/shell.html
  pN_name/     config.yaml, page.yaml, run_model.py, update.sh
  tests/       run.py + test_actions, test_copy_contract, test_facts_seam,
               test_scene_vocabulary, test_spread  (pure numpy, no GPU needed)
  out/pN/      generated latest.html + PNGs
```

Plan and full phase detail: `.claude/plans/nomad-deeper-visualization.md`. Project rules and decision log: `CLAUDE.md`.
