# Capstone audit — what the vanilla NoMaD run can and cannot be compared against

The capstone asks whether an improved recipe (`context_size 3` + `context_stride 3` +
`image_size [160, 120]`) beats the fully-trained vanilla NoMaD. The vanilla run is
**reused, never retrained**, which means it was not produced under the ablations'
controls. This document records exactly what is matched between the two and what is not,
so the headline comparison can be read for what it is.

Run under audit: `/mnt/shared_disk/nazli/nomad_outputs/nomad/nomad_2026_06_13_18_04_23`.

---

## 1. How the vanilla config was recovered

**The run saved no config.** `train.py` never writes one to the output directory: it
builds the config in memory, creates `logs/<project>/<run_name>/`, and relies on
`wandb.config.update(config)` for the record. The run directory holds only weights:

```
0.pth .. 99.pth          100 raw epoch checkpoints
ema_0.pth .. ema_99.pth  100 EMA checkpoints, no gaps
latest.pth               byte-different from 99.pth, but tensor-for-tensor identical
                         (two separate torch.save calls of the same state_dict)
optimizer_latest.pth, scheduler_latest.pth, visualize/
```

**The wandb record is empty.** Every wandb run directory from 2026-06-13 has a
zero-length `files/config.yaml`, so the intended fallback is not available either.

The config was therefore recovered from two independent directions, which agree.

### 1a. Git history pins the file that was on disk

The run started **2026-06-13 18:04:23** and finished **2026-06-14 11:31** (local time,
from checkpoint mtimes). `train/config/nomad.yaml` has exactly three commits ever:

| commit | date | what it changed |
|---|---|---|
| `76d3a15` | 2023-10-06 | upstream ViNT v1 release |
| `7531ac9` | **2026-06-13 09:48** | GoStanford2 data paths — **no hyperparameter lines** |
| `bd3d0db` | **2026-06-16 09:50** | added `wandb_entity: fnazlias` — nothing else |

The last edit before the run landed **8 hours before it started**; the next edit came
**two days after it finished** and added one logging key. Every hyperparameter in today's
`train/config/nomad.yaml` is therefore byte-identical to the one that trained this run.

### 1b. The checkpoint's tensor shapes corroborate the architecture

Independently of git, `ema_99.pth` pins the architectural hyperparameters:

| evidence in `ema_99.pth` | implies |
|---|---|
| `positional_encoding.pos_enc` → `(1, 5, 256)` | 5 tokens = 1 goal + (context_size + 1) obs → **`context_size: 3`** |
| `compress_obs_enc.weight` → `(256, 1280)` | **`encoding_size: 256`** |
| transformer layer indices `0,1,2,3` | **`mha_num_attention_layers: 4`** |
| `goal_encoder._conv_stem.weight` → `(32, **6**, 3, 3)` | separate 6-channel goal encoder (current frame ⊕ goal), stock NoMaD |
| `down_modules.0.0.cond_encoder.1.weight` → `(64, 512)` | first `down_dims` = 64; output width 64 not 128 → **`cond_predict_scale: False`** |
| `dist_pred_net` → 256 → 64 → 16 → 1 | stock distance head |

**Honest limit:** the weights cannot pin `image_size`, `epochs`, `lr`, `batch_size`,
`seed`, `len_traj_pred`, `goal_mask_prob` or `num_diffusion_iters`. NoMaD's encoder
adaptively pools to 1280 features regardless of input resolution, so all four resolutions
in Ablation B produce identically-shaped weights. Those values rest on §1a alone — that
the file on disk provably did not change around the run. The epoch count is separately
confirmed by the checkpoints themselves: `ema_0` through `ema_99`, no gaps, matching
`epochs: 100`.

---

## 2. Recipe parity: vanilla vs the `ctx03` ablation baseline

Both configs were merged over `defaults.yaml` and compared key by key. They are
**identical on every axis that decides what the model is and how it learns**:

`context_size` (3) · `image_size` ([96, 96]) · `lr` (1e-4) · `optimizer` (adamw) ·
`scheduler` (cosine) · `warmup` / `warmup_epochs` (4) · `seed` (0) · `len_traj_pred` (8) ·
`goal_mask_prob` (0.5) · `num_diffusion_iters` (10) · `alpha` (1e-4) · `encoding_size` ·
`obs_encoder` · `mha_*` · `down_dims` · `cond_predict_scale` · `attn_unet` ·
`normalize` · `learn_angle` · `distance` / `action` bounds · `goal_type` ·
`datasets` (same `go_stanford`, same train/test split folders)

The full set of differences is eight keys. Three of them are cosmetic or logging-only and
cannot move the weights (`project_name`, `run_name`, `image_log_freq`) or affect only the
in-training eval that this study does not use (`eval_batch_size`). The other **five keys
form three real confounds**, below.

---

## 3. The three confounds — and which way each one leans

| # | axis | vanilla | `ctx03` ablation | leans toward |
|---|---|---|---|---|
| 1 | `epochs` | **100** | 30 | the axis under test |
| 2 | `index_context_size` | absent → `context_size` = **3** | **20** | **vanilla** |
| 3 | batch mechanics | `batch_size: 256`, no accumulation | `batch_size: 32` × `gradient_accumulation_steps: 8` | **vanilla** (weakly) |

### Confound 2 — vanilla trained on 26% more data

`index_context_size` sets how many leading timesteps of each trajectory are skipped when
the sample index is built. It did not exist yet when the vanilla run trained, so indexing
skipped `context_size` = 3; the ablations pinned it to 20 so that arms with different
context lengths would see the *same* samples. Measured on the `go_stanford` train split:

| indexing | train samples |
|---|---|
| `index_context_size = 3` (vanilla) | **134,178** |
| `index_context_size = 20` (ablations, capstone) | **106,502** |

**Vanilla saw +27,676 samples, +26.0%.** Nothing was withheld from it; it simply had a
larger natural training set. This confound is not removable without retraining vanilla.

### Confound 3 — accumulation is not exactly a true 256 batch

The plan attributed this to differing BatchNorm statistics. **That mechanism does not
apply here** and has been corrected: `train.py` calls `replace_bn_with_gn` on the vision
encoder, and `ema_99.pth` contains **zero** `running_mean` / `running_var` /
`num_batches_tracked` buffers. There is no BatchNorm anywhere in the trained model, and
GroupNorm is batch-size independent.

The real mechanism is `action_reduce`, which NoMaD applies to both the diffusion and the
distance loss:

```python
(loss * action_mask).mean() / (action_mask.mean() + 1e-2)
```

This is **nonlinear in the batch**, so averaging it over eight microbatches of 32 is not
the same quantity as computing it once over 256. `train/ablation/check_grad_accum.py`
proves the two agree exactly when `action_mask` is all ones, and therefore that they can
legitimately differ when it is not. A true 256-sample batch computes the exact,
lower-variance gradient, so the lean is toward vanilla — but weakly, and the size of the
effect on real data is not quantified here.

### Why this makes the headline result conservative

Both extra confounds favour the vanilla run. So:

- **Best-combined@100 beating vanilla@100 is a strong result** — it wins despite the
  baseline having more data and the exact-gradient batch.
- **Tying or losing is ambiguous**, not a refutation: the confounds could account for it.
  That is the plan's Path B trigger, and it is why Phase 6 checks the margin against the
  across-seed SD rather than against zero.

---

## 4. What everything is scored on

Every model in this study — vanilla included — is scored on one aligned test set:
**`index_context_size = 20` → 26,648 samples**, independent of what any of them trained
on. Each model is fed at its own native input recipe, because scoring a model on inputs it
was never trained to read would measure the mismatch rather than the model.

`train/ablation/check_sample_alignment.py` verifies that this actually holds rather than
assuming it. Across the vanilla recipe (ctx 3 / stride 1 / 96×96), the `ctx03` baseline,
the `stride3` arm and the best-combined recipe (ctx 3 / stride 3 / 160×120), all four see
the same 26,648 samples with **bit-identical** `actions`, `distance`, `goal_pos` and
`action_mask` — same trajectories, same timesteps, same sampled goals, same negatives.
Only the pixels differ, as each recipe intends. That is what makes the paired per-sample
deltas in the results tables interpretable.

`train/ablation/check_eval_determinism.py` separately confirms that scoring a checkpoint
twice, in two separate processes, reproduces every per-sample number bit for bit.

---

## 5. Limitations carried into the write-up

- `gc_action_loss` is a **proxy** for navigation quality. Closed-loop rollout metrics
  (success rate, distance-to-goal, collisions) are out of scope for this pass.
- `image_size`, `lr`, `seed` and the other non-architectural vanilla hyperparameters rest
  on git provenance (§1a), not on evidence inside the checkpoint.
- Confound 3's magnitude is argued, not measured.
- The vanilla run is n = 1 seed and cannot be re-seeded without retraining; its
  uncertainty is bounded only by the `ctx03` across-seed ruler built in Phase 4.
